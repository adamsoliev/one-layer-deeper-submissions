"""Learned long-division recurrence for repeated modular squaring."""

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
    assert_model_state,
)
from torch import Tensor, nn

WIDTH = 48
HIDDEN_WIDTH = 128
NUM_HEADS = 4
MAX_DECIMAL_DIGITS = 8
PRODUCT_DIGITS = 2 * MAX_DECIMAL_DIGITS
REFINEMENT_STEPS = 2
MAX_TRAIN_TIME_STEPS = 16
MAX_EVAL_TIME_STEPS = 64
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 2_048
MAX_TRAINING_STEPS = 1_000_000
BASE_LEARNING_RATE = 3e-3
MIN_LEARNING_RATE_RATIO = 0.05
WARMUP_FRACTION = 0.05
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


class ArithmeticRefinement(nn.Module):
    """Share local carry propagation and global quotient interaction."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_norm = RMSNorm(4 * WIDTH)
        self.query = nn.Linear(4 * WIDTH, WIDTH, bias=False)
        self.key = nn.Linear(4 * WIDTH, WIDTH, bias=False)
        self.value = nn.Linear(4 * WIDTH, WIDTH, bias=False)
        self.attention_out = nn.Linear(WIDTH, WIDTH, bias=False)
        self.local = nn.Conv1d(WIDTH, WIDTH, kernel_size=3, padding=1)
        self.hidden_left = nn.Linear(4 * WIDTH, HIDDEN_WIDTH)
        self.hidden_right = nn.Linear(4 * WIDTH, HIDDEN_WIDTH)
        self.hidden_out = nn.Linear(HIDDEN_WIDTH, WIDTH)
        self.gate = nn.Linear(4 * WIDTH, WIDTH)
        self.output_norm = RMSNorm(WIDTH)

    def forward(self, state: Tensor, context: Tensor) -> Tensor:
        features = self.feature_norm(
            torch.cat(
                (
                    state,
                    context,
                    state * context,
                    state - context,
                ),
                dim=-1,
            ),
        )
        batch_size, cells, _ = state.shape
        head_width = WIDTH // NUM_HEADS
        query = (
            self.query(features)
            .view(
                batch_size,
                cells,
                NUM_HEADS,
                head_width,
            )
            .transpose(1, 2)
        )
        key = (
            self.key(features)
            .view(
                batch_size,
                cells,
                NUM_HEADS,
                head_width,
            )
            .transpose(1, 2)
        )
        value = (
            self.value(features)
            .view(
                batch_size,
                cells,
                NUM_HEADS,
                head_width,
            )
            .transpose(1, 2)
        )
        attention = F.scaled_dot_product_attention(query, key, value)
        attention = attention.transpose(1, 2).reshape(
            batch_size,
            cells,
            WIDTH,
        )
        local = self.local(state.transpose(1, 2)).transpose(1, 2)
        hidden = F.silu(self.hidden_left(features)) * torch.tanh(
            self.hidden_right(features),
        )
        update = self.attention_out(attention) + local + self.hidden_out(hidden)
        gate = torch.sigmoid(self.gate(features))
        return self.output_norm(state + 0.25 * gate * update)


class LearnedLongDivision(nn.Module):
    """Estimate a quotient and decode its learned coefficient residual."""

    def __init__(self) -> None:
        super().__init__()
        self.product_projection = nn.Linear(7, WIDTH)
        self.modulus_projection = nn.Linear(5, WIDTH)
        self.residual_projection = nn.Linear(8, WIDTH)
        self.quotient_context = nn.Linear(WIDTH, WIDTH, bias=False)
        self.refinement = ArithmeticRefinement()
        self.quotient_head = nn.Linear(WIDTH, NUM_DIGITS)
        self.residue_head = nn.Linear(WIDTH, NUM_DIGITS)
        self.quotient_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        residue_probabilities: Tensor,
        modulus_digits: Tensor,
        coefficient_map: Tensor,
        positions: Tensor,
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
        square_coefficients = self._multiply_digits(
            residue_digits,
            residue_digits,
            coefficient_map,
        )
        extended_modulus = F.pad(
            modulus_digits,
            (0, PRODUCT_DIGITS - MAX_DECIMAL_DIGITS),
        )
        product_features = self._product_features(
            square_coefficients,
            extended_modulus,
            positions,
        )
        modulus_features = self._modulus_features(
            extended_modulus,
            positions,
        )
        modulus_context = self.modulus_projection(modulus_features)
        quotient_state = self.product_projection(product_features)
        for _ in range(REFINEMENT_STEPS):
            quotient_state = self.refinement(
                quotient_state,
                modulus_context,
            )

        temperature = self.quotient_temperature.abs().clamp_min(0.25)
        quotient_logits = self.quotient_head(
            quotient_state[:, :MAX_DECIMAL_DIGITS],
        )
        quotient_probabilities = F.softmax(
            quotient_logits / temperature,
            dim=-1,
        )
        quotient_digits = torch.einsum(
            "bdk,k->bd",
            quotient_probabilities,
            digit_values,
        )
        quotient_product = self._multiply_digits(
            quotient_digits,
            modulus_digits,
            coefficient_map,
        )
        residual_coefficients = square_coefficients - quotient_product
        extended_quotient = F.pad(
            quotient_digits,
            (0, PRODUCT_DIGITS - MAX_DECIMAL_DIGITS),
        )
        residual_features = self._residual_features(
            residual_coefficients,
            square_coefficients,
            extended_modulus,
            extended_quotient,
            positions,
        )
        residue_state = self.residual_projection(residual_features)
        residue_context = modulus_context + self.quotient_context(quotient_state)
        for _ in range(REFINEMENT_STEPS):
            residue_state = self.refinement(
                residue_state,
                residue_context,
            )
        residue_logits = self.residue_head(
            residue_state[:, :MAX_DECIMAL_DIGITS],
        )
        return residue_logits, quotient_logits

    @staticmethod
    def _multiply_digits(
        left: Tensor,
        right: Tensor,
        coefficient_map: Tensor,
    ) -> Tensor:
        return torch.einsum(
            "bi,bj,ijk->bk",
            left,
            right,
            coefficient_map.to(left.dtype),
        )

    @staticmethod
    def _product_features(
        coefficients: Tensor,
        modulus: Tensor,
        positions: Tensor,
    ) -> Tensor:
        normalized = coefficients / MAX_PRODUCT_COEFFICIENT
        logarithm = torch.log1p(coefficients) / math.log1p(
            MAX_PRODUCT_COEFFICIENT,
        )
        modulus = modulus / 9.0
        return torch.stack(
            (
                normalized,
                logarithm,
                modulus,
                normalized * modulus,
                positions,
                torch.sin(2.0 * math.pi * positions),
                torch.cos(2.0 * math.pi * positions),
            ),
            dim=-1,
        )

    @staticmethod
    def _modulus_features(modulus: Tensor, positions: Tensor) -> Tensor:
        modulus = modulus / 9.0
        return torch.stack(
            (
                modulus,
                positions,
                torch.sin(2.0 * math.pi * positions),
                torch.cos(2.0 * math.pi * positions),
                (modulus != 0).to(modulus.dtype),
            ),
            dim=-1,
        )

    @staticmethod
    def _residual_features(
        residual: Tensor,
        square: Tensor,
        modulus: Tensor,
        quotient: Tensor,
        positions: Tensor,
    ) -> Tensor:
        residual_scale = float(MAX_PRODUCT_COEFFICIENT)
        signed_logarithm = residual.sign() * torch.log1p(residual.abs())
        signed_logarithm = signed_logarithm / math.log1p(
            MAX_PRODUCT_COEFFICIENT,
        )
        return torch.stack(
            (
                residual / residual_scale,
                signed_logarithm,
                square / residual_scale,
                modulus / 9.0,
                quotient / 9.0,
                positions,
                torch.sin(2.0 * math.pi * positions),
                torch.cos(2.0 * math.pi * positions),
            ),
            dim=-1,
        )


class DigitwiseSquaringModel(nn.Module):
    """Compose a learned decimal long-division transition in latent space."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.transition = LearnedLongDivision()
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
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if attention_mask is None:
            attention_mask = input_ids != PAD_TOKEN_ID
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)

        modulus, residue, time_steps = self._parse_fields(
            input_ids,
            attention_mask,
        )
        modulus_digits = self._decimal_digits(modulus).to(
            self.transition.product_projection.weight.dtype,
        )
        residue_digits = self._decimal_digits(residue)
        residue_probabilities = F.one_hot(
            residue_digits,
            NUM_DIGITS,
        ).to(modulus_digits.dtype)
        residue_logits = torch.log(
            residue_probabilities * 0.99 + 0.001,
        )
        quotient_logits = torch.zeros_like(residue_logits)
        positions = torch.linspace(
            0.0,
            1.0,
            PRODUCT_DIGITS,
            device=input_ids.device,
            dtype=modulus_digits.dtype,
        )[None, :].expand(input_ids.shape[0], -1)

        maximum_steps = MAX_TRAIN_TIME_STEPS if self.training else MAX_EVAL_TIME_STEPS
        for step in range(maximum_steps):
            active_indices = torch.nonzero(
                time_steps > step,
                as_tuple=False,
            ).squeeze(-1)
            if active_indices.numel() == 0:
                continue
            proposal_logits, proposed_quotient = self.transition(
                residue_probabilities[active_indices],
                modulus_digits[active_indices],
                self.coefficient_map,
                positions[active_indices],
            )
            residue_logits = residue_logits.index_copy(
                0,
                active_indices,
                proposal_logits,
            )
            quotient_logits = quotient_logits.index_copy(
                0,
                active_indices,
                proposed_quotient,
            )
            residue_probabilities = F.softmax(residue_logits, dim=-1)

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
        return logits, (residue_logits, quotient_logits)

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


def build_model(spec: ModelSpec) -> DigitwiseSquaringModel:
    model = DigitwiseSquaringModel(spec)
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


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    max_steps=MAX_TRAINING_STEPS,
)
