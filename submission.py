"""Worst-digit neural cellular automaton for repeated modular squaring."""

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
HIDDEN_WIDTH = 128
NUM_HEADS = 4
MAX_VALUE_BITS = 24
MAX_DECIMAL_DIGITS = 8
MICRO_STEPS = 2
MAX_TRAIN_TIME_STEPS = 16
MAX_EVAL_TIME_STEPS = 64
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 2_048
MAX_TRAINING_STEPS = 1_000_000
BASE_LEARNING_RATE = 3e-3
MIN_LEARNING_RATE_RATIO = 0.05
WARMUP_FRACTION = 0.05
RECONSTRUCTION_WEIGHT = 0.05
WORST_DIGIT_WEIGHT = 0.35
WORST_DIGIT_TEMPERATURE = 0.5
PAD_TOKEN_ID = 0
X_TOKEN_ID = 3
T_TOKEN_ID = 4
DIGIT_OFFSET = 7
NUM_DIGITS = 10


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


class CellularRefinement(nn.Module):
    """Mix local carries and global bit interactions with shared weights."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_norm = RMSNorm(4 * WIDTH)
        self.query = nn.Linear(4 * WIDTH, WIDTH, bias=False)
        self.key = nn.Linear(4 * WIDTH, WIDTH, bias=False)
        self.value = nn.Linear(4 * WIDTH, WIDTH, bias=False)
        self.attention_out = nn.Linear(WIDTH, WIDTH, bias=False)
        self.local = nn.Conv1d(
            WIDTH,
            WIDTH,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.hidden_left = nn.Linear(4 * WIDTH, HIDDEN_WIDTH)
        self.hidden_right = nn.Linear(4 * WIDTH, HIDDEN_WIDTH)
        self.hidden_out = nn.Linear(HIDDEN_WIDTH, WIDTH)
        self.gate = nn.Linear(4 * WIDTH, WIDTH)
        self.output_norm = RMSNorm(WIDTH)

    def forward(self, state: Tensor, modulus_context: Tensor) -> Tensor:
        features = self.feature_norm(
            torch.cat(
                (
                    state,
                    modulus_context,
                    state * modulus_context,
                    state - modulus_context,
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


class LearnedSquaringTransition(nn.Module):
    """Apply the same learned state transition for every modular square."""

    def __init__(self) -> None:
        super().__init__()
        self.refinement = CellularRefinement()
        self.transition_norm = RMSNorm(2 * WIDTH)
        self.transition_left = nn.Linear(2 * WIDTH, HIDDEN_WIDTH)
        self.transition_right = nn.Linear(2 * WIDTH, HIDDEN_WIDTH)
        self.transition_out = nn.Linear(HIDDEN_WIDTH, WIDTH)
        self.output_norm = RMSNorm(WIDTH)

    def forward(self, state: Tensor, modulus_context: Tensor) -> Tensor:
        working = state
        for _ in range(MICRO_STEPS):
            working = self.refinement(working, modulus_context)
        features = self.transition_norm(
            torch.cat((working, state), dim=-1),
        )
        hidden = F.silu(self.transition_left(features)) * torch.tanh(
            self.transition_right(features),
        )
        return self.output_norm(
            state + 0.25 * self.transition_out(hidden),
        )


class NeuralCellularAutomaton(nn.Module):
    """Represent integers on a bit grid and compose one tied transition."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.residue_projection = nn.Linear(6, WIDTH)
        self.modulus_projection = nn.Linear(6, WIDTH)
        self.residue_norm = RMSNorm(WIDTH)
        self.modulus_norm = RMSNorm(WIDTH)
        self.transition = LearnedSquaringTransition()
        self.answer_queries = nn.Embedding(MAX_DECIMAL_DIGITS, WIDTH)
        self.answer_key = nn.Linear(WIDTH, WIDTH, bias=False)
        self.answer_value = nn.Linear(WIDTH, WIDTH, bias=False)
        self.answer_head = nn.Linear(WIDTH, spec.vocab_size)
        self.residue_reconstruction = nn.Linear(WIDTH, 1)
        self.modulus_reconstruction = nn.Linear(WIDTH, 1)
        self.apply(self._initialize)
        nn.init.normal_(self.transition.transition_out.weight, std=0.005)
        nn.init.zeros_(self.transition.transition_out.bias)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if attention_mask is None:
            attention_mask = input_ids != PAD_TOKEN_ID
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)

        modulus, residue, time_steps = self._parse_fields(
            input_ids,
            attention_mask,
        )
        residue_bits = self._bits(residue)
        modulus_bits = self._bits(modulus)
        positions = torch.linspace(
            0.0,
            1.0,
            MAX_VALUE_BITS,
            device=input_ids.device,
            dtype=self.residue_projection.weight.dtype,
        )[None, :, None]
        positions = positions.expand(input_ids.shape[0], -1, -1)
        residue_features = self._bit_features(
            residue_bits,
            modulus_bits,
            positions,
        )
        modulus_features = self._bit_features(
            modulus_bits,
            residue_bits,
            positions,
        )
        state = self.residue_norm(
            self.residue_projection(residue_features),
        )
        modulus_context = self.modulus_norm(
            self.modulus_projection(modulus_features),
        )

        maximum_steps = MAX_TRAIN_TIME_STEPS if self.training else MAX_EVAL_TIME_STEPS
        for step in range(maximum_steps):
            proposal = self.transition(state, modulus_context)
            active = (time_steps > step)[:, None, None]
            state = torch.where(active, proposal, state)

        slot_logits = self._decode_slots(state)
        logits = self._place_slot_logits(
            slot_logits,
            input_ids,
            attention_mask,
        )
        reconstruction_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(
                self.residue_reconstruction(
                    self.residue_norm(
                        self.residue_projection(residue_features),
                    ),
                )
                .squeeze(-1)
                .float(),
                residue_bits.float(),
            )
            + F.binary_cross_entropy_with_logits(
                self.modulus_reconstruction(modulus_context).squeeze(-1).float(),
                modulus_bits.float(),
            )
        )
        return logits, reconstruction_loss

    @staticmethod
    def _bit_features(
        primary_bits: Tensor,
        secondary_bits: Tensor,
        positions: Tensor,
    ) -> Tensor:
        primary = primary_bits[..., None].to(positions.dtype)
        secondary = secondary_bits[..., None].to(positions.dtype)
        return torch.cat(
            (
                primary,
                secondary,
                primary * secondary,
                positions,
                torch.sin(2.0 * math.pi * positions),
                torch.cos(2.0 * math.pi * positions),
            ),
            dim=-1,
        )

    def _decode_slots(self, state: Tensor) -> Tensor:
        queries = self.answer_queries.weight
        keys = self.answer_key(state)
        scores = torch.einsum("dc,blc->bdl", queries, keys)
        scores = scores / math.sqrt(WIDTH)
        weights = F.softmax(scores, dim=-1)
        values = self.answer_value(state)
        pooled = torch.einsum("bdl,blc->bdc", weights, values)
        return self.answer_head(pooled)

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
    def _bits(value: Tensor) -> Tensor:
        shifts = torch.arange(
            MAX_VALUE_BITS,
            device=value.device,
            dtype=torch.long,
        )
        return (value[:, None] >> shifts) & 1

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
        return (
            modulus,
            residue,
            time_steps.clamp_min(1),
        )


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


def build_model(spec: ModelSpec) -> NeuralCellularAutomaton:
    model = NeuralCellularAutomaton(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(
    model: nn.Module,
    spec: OptimizerSpec,
) -> OptimizerBundle:
    decay_parameters: list[Tensor] = []
    stable_parameters: list[Tensor] = []
    for name, parameter in model.named_parameters():
        if parameter.ndim >= 2 and "answer_queries" not in name:
            decay_parameters.append(parameter)
        else:
            stable_parameters.append(parameter)
    optimizer = DeviceAdamW(
        (
            {"params": decay_parameters, "weight_decay": 0.02},
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
    valid = batch.valid_mask
    target_counts = valid.sum(dim=1)
    valid_rows = target_counts > 0
    sequence_nll = (token_losses * valid).sum(dim=1)
    masked_losses = token_losses.masked_fill(~valid, -torch.inf)
    smooth_worst = WORST_DIGIT_TEMPERATURE * torch.logsumexp(
        masked_losses / WORST_DIGIT_TEMPERATURE,
        dim=1,
    )
    smooth_worst = smooth_worst - WORST_DIGIT_TEMPERATURE * torch.log(
        target_counts.clamp_min(1).to(token_losses.dtype),
    )
    prediction_loss = (
        (1.0 - WORST_DIGIT_WEIGHT) * sequence_nll + WORST_DIGIT_WEIGHT * smooth_worst
    )[valid_rows].mean()
    if not isinstance(batch.auxiliary, Tensor) or batch.auxiliary.ndim != 0:
        raise TypeError("auxiliary output must be a scalar reconstruction loss")
    return prediction_loss + RECONSTRUCTION_WEIGHT * batch.auxiliary


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    max_steps=MAX_TRAINING_STEPS,
)
