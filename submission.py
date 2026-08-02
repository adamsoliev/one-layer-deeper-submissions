"""Bidirectional digit-scan recurrence for repeated modular squaring."""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
    assert_model_state,
)
from torch import Tensor, nn

WIDTH = 64
HIDDEN_WIDTH = 160
MAX_DECIMAL_DIGITS = 8
MAX_VALUE_BITS = 24
PRODUCT_DIGITS = 2 * MAX_DECIMAL_DIGITS
MAX_TRAIN_TIME_STEPS = 16
MAX_EVAL_TIME_STEPS = 64
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 2_048
MAX_TRAINING_STEPS = 1_000_000
BASE_LEARNING_RATE = 3e-3
MIN_LEARNING_RATE_RATIO = 0.05
WARMUP_FRACTION = 0.05
SEMANTIC_DECIMAL_WEIGHT = 0.5
SEMANTIC_BINARY_WEIGHT = 0.25
STATE_ENTROPY_WEIGHT = 0.01
PAD_TOKEN_ID = 0
X_TOKEN_ID = 3
T_TOKEN_ID = 4
DIGIT_OFFSET = 7
NUM_DIGITS = 10
MAX_PRODUCT_COEFFICIENT = MAX_DECIMAL_DIGITS * 9 * 9


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, hidden: Tensor) -> Tensor:
        return F.rms_norm(
            hidden,
            (hidden.shape[-1],),
            self.weight,
            eps=1e-5,
        )


class BidirectionalDigitScan(nn.Module):
    """Sweep multiplication carry upward and reduction state downward."""

    def __init__(self) -> None:
        super().__init__()
        self.cell_projection = nn.Linear(8, WIDTH)
        self.context_projection = nn.Linear(5, WIDTH)
        self.carry_scan = nn.GRU(WIDTH, WIDTH, batch_first=True)
        self.reduction_input = nn.Linear(3 * WIDTH, WIDTH)
        self.reduction_scan = nn.GRU(WIDTH, WIDTH, batch_first=True)
        self.fusion_norm = RMSNorm(3 * WIDTH)
        self.fusion_left = nn.Linear(3 * WIDTH, HIDDEN_WIDTH)
        self.fusion_right = nn.Linear(3 * WIDTH, HIDDEN_WIDTH)
        self.fusion_out = nn.Linear(HIDDEN_WIDTH, WIDTH)
        self.output_norm = RMSNorm(WIDTH)
        self.residue_head = nn.Linear(WIDTH, NUM_DIGITS)
        self.bit_query_projection = nn.Linear(5, WIDTH)
        self.bit_key = nn.Linear(WIDTH, WIDTH, bias=False)
        self.bit_value = nn.Linear(WIDTH, WIDTH, bias=False)
        self.bit_head = nn.Linear(WIDTH, 1)

    def forward(
        self,
        residue_probabilities: Tensor,
        modulus_digits: Tensor,
        coefficient_map: Tensor,
        product_positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        digit_values = torch.arange(
            NUM_DIGITS,
            device=residue_probabilities.device,
            dtype=residue_probabilities.dtype,
        )
        residue_digits = torch.einsum(
            "bdk,k->bd",
            residue_probabilities,
            digit_values,
        )
        product_coefficients = torch.einsum(
            "bi,bj,ijk->bk",
            residue_digits,
            residue_digits,
            coefficient_map.to(residue_digits.dtype),
        )
        extended_residue = F.pad(
            residue_digits,
            (0, PRODUCT_DIGITS - MAX_DECIMAL_DIGITS),
        )
        extended_modulus = F.pad(
            modulus_digits,
            (0, PRODUCT_DIGITS - MAX_DECIMAL_DIGITS),
        )
        cell_features = self._cell_features(
            product_coefficients,
            extended_residue,
            extended_modulus,
            product_positions,
        )
        context_features = self._context_features(
            extended_residue,
            extended_modulus,
            product_positions,
        )
        cells = self.cell_projection(cell_features)
        context = self.context_projection(context_features)

        carry_initial = context.mean(dim=1, keepdim=False)[None, :, :]
        carry_states, _ = self.carry_scan(
            cells + context,
            carry_initial,
        )
        reduction_inputs = self.reduction_input(
            torch.cat((cells, context, carry_states), dim=-1),
        )
        reduction_initial = (carry_states[:, -1] + context[:, -1])[None, :, :]
        reversed_reduction, _ = self.reduction_scan(
            torch.flip(reduction_inputs, dims=(1,)),
            reduction_initial,
        )
        reduction_states = torch.flip(reversed_reduction, dims=(1,))

        fused_features = self.fusion_norm(
            torch.cat((cells, carry_states, reduction_states), dim=-1),
        )
        hidden = F.silu(self.fusion_left(fused_features)) * torch.tanh(
            self.fusion_right(fused_features),
        )
        fused = self.output_norm(cells + self.fusion_out(hidden))
        residue_logits = self.residue_head(
            fused[:, :MAX_DECIMAL_DIGITS],
        )
        binary_logits = self._decode_bits(fused)
        return residue_logits, binary_logits

    @staticmethod
    def _cell_features(
        coefficients: Tensor,
        residue: Tensor,
        modulus: Tensor,
        positions: Tensor,
    ) -> Tensor:
        normalized = coefficients / MAX_PRODUCT_COEFFICIENT
        logarithm = torch.log1p(coefficients) / math.log1p(
            MAX_PRODUCT_COEFFICIENT,
        )
        residue = residue / 9.0
        modulus = modulus / 9.0
        return torch.stack(
            (
                normalized,
                logarithm,
                residue,
                modulus,
                residue * modulus,
                positions,
                torch.sin(2.0 * math.pi * positions),
                torch.cos(2.0 * math.pi * positions),
            ),
            dim=-1,
        )

    @staticmethod
    def _context_features(
        residue: Tensor,
        modulus: Tensor,
        positions: Tensor,
    ) -> Tensor:
        residue = residue / 9.0
        modulus = modulus / 9.0
        return torch.stack(
            (
                residue,
                modulus,
                positions,
                torch.sin(2.0 * math.pi * positions),
                torch.cos(2.0 * math.pi * positions),
            ),
            dim=-1,
        )

    def _decode_bits(self, state: Tensor) -> Tensor:
        bit_positions = torch.linspace(
            0.0,
            1.0,
            MAX_VALUE_BITS,
            device=state.device,
            dtype=state.dtype,
        )
        query_features = torch.stack(
            (
                bit_positions,
                torch.sin(2.0 * math.pi * bit_positions),
                torch.cos(2.0 * math.pi * bit_positions),
                torch.sin(4.0 * math.pi * bit_positions),
                torch.cos(4.0 * math.pi * bit_positions),
            ),
            dim=-1,
        )
        queries = self.bit_query_projection(query_features)
        keys = self.bit_key(state)
        scores = torch.einsum("qc,blc->bql", queries, keys)
        scores = scores / math.sqrt(WIDTH)
        weights = F.softmax(scores, dim=-1)
        values = self.bit_value(state)
        pooled = torch.einsum("bql,blc->bqc", weights, values)
        return self.bit_head(pooled).squeeze(-1)


class DigitScanModel(nn.Module):
    """Compose soft semantic residues through one tied directional scan."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.transition = BidirectionalDigitScan()
        self.special_logits = nn.Parameter(torch.empty(DIGIT_OFFSET))
        coefficient_map = torch.zeros(
            MAX_DECIMAL_DIGITS,
            MAX_DECIMAL_DIGITS,
            PRODUCT_DIGITS,
        )
        for left in range(MAX_DECIMAL_DIGITS):
            for right in range(MAX_DECIMAL_DIGITS):
                coefficient_map[left, right, left + right] = 1.0
        self.register_buffer("coefficient_map", coefficient_map)
        self.register_buffer(
            "decimal_powers",
            10 ** torch.arange(MAX_DECIMAL_DIGITS, dtype=torch.long),
        )
        self.apply(self._initialize)
        nn.init.normal_(self.special_logits, mean=-0.1, std=0.02)
        nn.init.normal_(self.transition.residue_head.weight, std=0.005)
        nn.init.zeros_(self.transition.residue_head.bias)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.GRU)):
            for name, parameter in module.named_parameters(recurse=False):
                if "weight" in name:
                    nn.init.normal_(parameter, mean=0.0, std=0.02)
                elif "bias" in name:
                    nn.init.zeros_(parameter)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        if attention_mask is None:
            attention_mask = input_ids != PAD_TOKEN_ID
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)

        modulus, residue, time_steps = self._parse_fields(
            input_ids,
            attention_mask,
        )
        modulus_digits = self._decimal_digits(modulus).to(
            self.transition.cell_projection.weight.dtype,
        )
        residue_digits = self._decimal_digits(residue)
        residue_probabilities = F.one_hot(
            residue_digits,
            NUM_DIGITS,
        ).to(modulus_digits.dtype)
        residue_logits = torch.log(
            residue_probabilities * 0.99 + 0.001,
        )
        binary_logits = torch.zeros(
            input_ids.shape[0],
            MAX_VALUE_BITS,
            device=input_ids.device,
            dtype=modulus_digits.dtype,
        )
        product_positions = torch.linspace(
            0.0,
            1.0,
            PRODUCT_DIGITS,
            device=input_ids.device,
            dtype=modulus_digits.dtype,
        )[None, :].expand(input_ids.shape[0], -1)
        state_entropies: list[Tensor] = []

        maximum_steps = MAX_TRAIN_TIME_STEPS if self.training else MAX_EVAL_TIME_STEPS
        for step in range(maximum_steps):
            active_indices = torch.nonzero(
                time_steps > step,
                as_tuple=False,
            ).squeeze(-1)
            if active_indices.numel() == 0:
                continue
            proposal_logits, proposed_bits = self.transition(
                residue_probabilities[active_indices],
                modulus_digits[active_indices],
                self.coefficient_map,
                product_positions[active_indices],
            )
            residue_logits = residue_logits.index_copy(
                0,
                active_indices,
                proposal_logits,
            )
            binary_logits = binary_logits.index_copy(
                0,
                active_indices,
                proposed_bits,
            )
            residue_probabilities = F.softmax(residue_logits, dim=-1)
            active_probabilities = residue_probabilities[active_indices]
            state_entropies.append(
                -(active_probabilities * active_probabilities.clamp_min(1e-8).log())
                .sum(dim=-1)
                .mean(),
            )

        slot_logits = torch.cat(
            (
                self.special_logits[None, None, :].expand(
                    input_ids.shape[0],
                    MAX_DECIMAL_DIGITS,
                    -1,
                ),
                residue_logits,
            ),
            dim=-1,
        )
        logits = self._place_slot_logits(
            slot_logits,
            input_ids,
            attention_mask,
        )
        state_entropy = torch.stack(state_entropies).mean()
        return logits, (residue_logits, binary_logits, state_entropy)

    def _decimal_digits(self, value: Tensor) -> Tensor:
        return (value[:, None] // self.decimal_powers) % NUM_DIGITS

    @staticmethod
    def _place_slot_logits(
        slot_logits: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        positions = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
        )[None, :]
        valid_lengths = attention_mask.long().sum(dim=1, keepdim=True)
        answer_slots = (valid_lengths - positions - 1).clamp(
            min=0,
            max=MAX_DECIMAL_DIGITS - 1,
        )
        batch_indices = torch.arange(
            input_ids.shape[0],
            device=input_ids.device,
        )[:, None]
        return slot_logits[batch_indices, answer_slots]

    @staticmethod
    def _parse_fields(
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        modulus = torch.zeros(
            input_ids.shape[0],
            device=input_ids.device,
            dtype=torch.long,
        )
        residue = torch.zeros_like(modulus)
        time_steps = torch.zeros_like(modulus)
        field = torch.zeros_like(modulus)
        for position in range(input_ids.shape[1]):
            token = input_ids[:, position]
            field = torch.where(token == X_TOKEN_ID, 1, field)
            field = torch.where(token == T_TOKEN_ID, 2, field)
            is_digit = (token >= DIGIT_OFFSET) & attention_mask[:, position]
            digit = (token - DIGIT_OFFSET).clamp(
                min=0,
                max=NUM_DIGITS - 1,
            )
            modulus = torch.where(
                is_digit & (field == 0),
                modulus * 10 + digit,
                modulus,
            )
            residue = torch.where(
                is_digit & (field == 1),
                residue * 10 + digit,
                residue,
            )
            time_steps = torch.where(
                is_digit & (field == 2),
                time_steps * 10 + digit,
                time_steps,
            )
        return modulus, residue, time_steps.clamp_min(1)


class DeviceAdamW(torch.optim.Optimizer):
    """Keep all tensor state beside parameters on every accelerator."""

    def __init__(
        self,
        parameters: object,
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
    ) -> None:
        super().__init__(
            parameters,
            {"lr": lr, "betas": betas, "eps": eps, "weight_decay": 0.0},
        )

    @torch.no_grad()
    def step(self, closure: object = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            learning_rate = float(group["lr"])
            beta1, beta2 = group["betas"]
            epsilon = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("DeviceAdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["first_moment"] = torch.zeros_like(parameter)
                    state["second_moment"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                first_moment = state["first_moment"]
                second_moment = state["second_moment"]
                if weight_decay:
                    parameter.mul_(1.0 - learning_rate * weight_decay)
                first_moment.lerp_(gradient, 1.0 - beta1)
                second_moment.mul_(beta2).addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - beta2,
                )
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denominator = (
                    second_moment.sqrt()
                    .div_(
                        math.sqrt(bias_correction2),
                    )
                    .add_(epsilon)
                )
                parameter.addcdiv_(
                    first_moment,
                    denominator,
                    value=-learning_rate / bias_correction1,
                )
        return loss


class WallClockSchedule:
    """Spend the learning-rate schedule according to the evaluator clock."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        training_time_seconds: float,
    ) -> None:
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.training_time_seconds = max(float(training_time_seconds), 1e-3)
        self.started_at = time.monotonic()
        self._set_multiplier(MIN_LEARNING_RATE_RATIO)

    def _set_multiplier(self, multiplier: float) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * multiplier

    def step(self) -> None:
        progress = min(
            (time.monotonic() - self.started_at) / self.training_time_seconds,
            1.0,
        )
        if progress < WARMUP_FRACTION:
            multiplier = (
                MIN_LEARNING_RATE_RATIO
                + (1.0 - MIN_LEARNING_RATE_RATIO) * progress / WARMUP_FRACTION
            )
        else:
            decay_progress = (progress - WARMUP_FRACTION) / (1.0 - WARMUP_FRACTION)
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            multiplier = (
                MIN_LEARNING_RATE_RATIO + (1.0 - MIN_LEARNING_RATE_RATIO) * cosine
            )
        self._set_multiplier(multiplier)


def build_model(spec: ModelSpec) -> DigitScanModel:
    model = DigitScanModel(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(
    model: nn.Module,
    spec: OptimizerSpec,
) -> OptimizerBundle:
    decay_parameters: list[Tensor] = []
    stable_parameters: list[Tensor] = []
    for parameter in model.parameters():
        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            stable_parameters.append(parameter)
    optimizer = DeviceAdamW(
        (
            {"params": decay_parameters, "weight_decay": 0.01},
            {"params": stable_parameters, "weight_decay": 0.0},
        ),
        lr=BASE_LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    scheduler = WallClockSchedule(optimizer, spec.training_time_seconds)
    return OptimizerBundle(optimizer, scheduler)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    token_losses = F.cross_entropy(
        batch.logits.transpose(1, 2).float(),
        batch.labels,
        ignore_index=-100,
        reduction="none",
    )
    target_counts = batch.valid_mask.sum(dim=1)
    valid_rows = target_counts > 0
    sequence_loss = (
        (token_losses * batch.valid_mask).sum(dim=1) / target_counts.clamp_min(1)
    )[valid_rows].mean()

    target_values = torch.zeros(
        batch.labels.shape[0],
        device=batch.labels.device,
        dtype=torch.long,
    )
    for position in range(batch.labels.shape[1]):
        valid = batch.valid_mask[:, position]
        digit = (batch.labels[:, position] - DIGIT_OFFSET).clamp(
            min=0,
            max=NUM_DIGITS - 1,
        )
        target_values = torch.where(
            valid,
            target_values * 10 + digit,
            target_values,
        )
    decimal_powers = 10 ** torch.arange(
        MAX_DECIMAL_DIGITS,
        device=batch.labels.device,
        dtype=torch.long,
    )
    target_decimal_digits = (target_values[:, None] // decimal_powers) % NUM_DIGITS
    bit_shifts = torch.arange(
        MAX_VALUE_BITS,
        device=batch.labels.device,
        dtype=torch.long,
    )
    target_bits = ((target_values[:, None] >> bit_shifts) & 1).float()

    if not isinstance(batch.auxiliary, tuple) or len(batch.auxiliary) != 3:
        raise TypeError(
            "auxiliary output must contain decimal logits, binary logits, and entropy",
        )
    decimal_logits, binary_logits, state_entropy = batch.auxiliary
    if not isinstance(decimal_logits, Tensor) or not isinstance(binary_logits, Tensor):
        raise TypeError("semantic logits must be tensors")
    if not isinstance(state_entropy, Tensor) or state_entropy.ndim != 0:
        raise TypeError("state entropy must be a scalar tensor")
    semantic_decimal_loss = F.cross_entropy(
        decimal_logits.transpose(1, 2).float(),
        target_decimal_digits,
    )
    semantic_binary_loss = F.binary_cross_entropy_with_logits(
        binary_logits.float(),
        target_bits,
    )
    return (
        sequence_loss
        + SEMANTIC_DECIMAL_WEIGHT * semantic_decimal_loss
        + SEMANTIC_BINARY_WEIGHT * semantic_binary_loss
        + STATE_ENTROPY_WEIGHT * state_entropy
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    max_steps=MAX_TRAINING_STEPS,
)
