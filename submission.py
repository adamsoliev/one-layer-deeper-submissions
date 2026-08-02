"""Learned redundant-digit recurrence for repeated modular squaring."""

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
CARRY_WIDTH = 48
MAX_DECIMAL_DIGITS = 8
MAX_TRAIN_TIME_STEPS = 16
MAX_EVAL_TIME_STEPS = 64
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 2_048
MAX_TRAINING_STEPS = 1_000_000
BASE_LEARNING_RATE = 3e-3
MIN_LEARNING_RATE_RATIO = 0.05
WARMUP_FRACTION = 0.05
SEMANTIC_DECIMAL_WEIGHT = 0.75
INVARIANT_WEIGHT = 0.5
ENTROPY_WEIGHT = 0.002
PAD_TOKEN_ID = 0
X_TOKEN_ID = 3
T_TOKEN_ID = 4
DIGIT_OFFSET = 7
NUM_DIGITS = 10
QUOTIENT_MIN = 0
QUOTIENT_MAX = 18
CARRY_MIN = -32
CARRY_MAX = 32
MAX_RAW_DIGIT_MAGNITUDE = 180.0


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


def straight_through_choice(logits: Tensor, choices: Tensor) -> tuple[Tensor, Tensor]:
    """Choose a discrete value while differentiating through its expectation."""
    probabilities = F.softmax(logits.float(), dim=-1).to(logits.dtype)
    soft_value = torch.einsum("...k,k->...", probabilities, choices)
    hard_value = choices[probabilities.argmax(dim=-1)]
    value = hard_value + soft_value - soft_value.detach()
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    return value, entropy


class LearnedRedundantSquare(nn.Module):
    """Learn bounded modular reductions in a signed decimal workspace."""

    def __init__(self) -> None:
        super().__init__()
        self.quotient_network = nn.Sequential(
            nn.Linear(7, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, QUOTIENT_MAX - QUOTIENT_MIN + 1),
        )
        self.carry_scan = nn.GRU(8, CARRY_WIDTH, batch_first=True)
        self.carry_head = nn.Linear(
            CARRY_WIDTH,
            CARRY_MAX - CARRY_MIN + 1,
        )
        self.digit_head = nn.Linear(CARRY_WIDTH, NUM_DIGITS)
        self.register_buffer(
            "quotient_choices",
            torch.arange(QUOTIENT_MIN, QUOTIENT_MAX + 1, dtype=torch.float32),
        )
        self.register_buffer(
            "carry_choices",
            torch.arange(CARRY_MIN, CARRY_MAX + 1, dtype=torch.float32),
        )
        self.register_buffer(
            "digit_choices",
            torch.arange(NUM_DIGITS, dtype=torch.float32),
        )
        self.register_buffer(
            "decimal_powers_float",
            10.0 ** torch.arange(MAX_DECIMAL_DIGITS, dtype=torch.float32),
        )
        positions = torch.linspace(0.0, 1.0, MAX_DECIMAL_DIGITS)
        self.register_buffer("digit_positions", positions)

    def forward(
        self,
        multiplicand: Tensor,
        modulus_digits: Tensor,
        microsteps: int = MAX_DECIMAL_DIGITS,
    ) -> tuple[Tensor, Tensor, Tensor]:
        accumulator = torch.zeros_like(multiplicand)
        modulus_value = torch.einsum(
            "bd,d->b",
            modulus_digits.float(),
            self.decimal_powers_float,
        )
        multiplicand_value = torch.einsum(
            "bd,d->b",
            multiplicand.float(),
            self.decimal_powers_float,
        )
        invariant_losses: list[Tensor] = []
        entropies: list[Tensor] = []

        digit_order = tuple(reversed(range(MAX_DECIMAL_DIGITS)))
        for digit_index in digit_order[:microsteps]:
            multiplier_digit = multiplicand[:, digit_index]
            shifted_accumulator = F.pad(
                accumulator[:, :-1],
                (1, 0),
            )
            candidate_digits = (
                shifted_accumulator + multiplier_digit[:, None] * multiplicand
            )
            accumulator_value = torch.einsum(
                "bd,d->b",
                accumulator.float(),
                self.decimal_powers_float,
            )
            candidate_value = torch.einsum(
                "bd,d->b",
                candidate_digits.float(),
                self.decimal_powers_float,
            )
            denominator = modulus_value.clamp_min(1.0)
            candidate_ratio = candidate_value / denominator
            quotient_features = torch.stack(
                (
                    candidate_ratio / (QUOTIENT_MAX + 1.0),
                    torch.tanh(candidate_ratio),
                    accumulator_value / denominator,
                    multiplicand_value / denominator,
                    multiplier_digit.float() / 9.0,
                    torch.log1p(denominator) / math.log(100_000_001.0),
                    torch.ones_like(candidate_ratio),
                ),
                dim=-1,
            ).to(multiplicand.dtype)
            quotient_logits = self.quotient_network(quotient_features)
            quotient, quotient_entropy = straight_through_choice(
                quotient_logits,
                self.quotient_choices.to(quotient_logits.dtype),
            )

            raw_digits = candidate_digits - quotient[:, None] * modulus_digits
            raw_features = self._carry_features(
                raw_digits,
                accumulator,
                multiplicand,
                modulus_digits,
                quotient,
            )
            carry_state, _ = self.carry_scan(raw_features)
            carry_logits = self.carry_head(carry_state)
            carries, carry_entropy = straight_through_choice(
                carry_logits,
                self.carry_choices.to(carry_logits.dtype),
            )
            digit_logits = self.digit_head(carry_state)
            next_digits, digit_entropy = straight_through_choice(
                digit_logits,
                self.digit_choices.to(digit_logits.dtype),
            )
            incoming_carries = F.pad(carries[:, :-1], (1, 0))
            normalized_coefficients = raw_digits + incoming_carries - 10.0 * carries
            accumulator = next_digits

            reduced_value = candidate_value - quotient.float() * modulus_value
            represented_value = torch.einsum(
                "bd,d->b",
                accumulator.float(),
                self.decimal_powers_float,
            )
            below_zero = F.relu(-reduced_value / denominator)
            above_modulus = F.relu(
                (reduced_value - (modulus_value - 1.0)) / denominator,
            )
            below_digit = F.relu(-normalized_coefficients.float() / 10.0)
            above_digit = F.relu(
                (normalized_coefficients.float() - 9.0) / 10.0,
            )
            representation_error = (represented_value - reduced_value) / denominator
            overflow = carries[:, -1].float() / max(abs(CARRY_MIN), CARRY_MAX)
            invariant_losses.append(
                below_zero.square().mean()
                + above_modulus.square().mean()
                + below_digit.square().mean()
                + above_digit.square().mean()
                + representation_error.square().mean()
                + overflow.square().mean(),
            )
            entropies.append(
                quotient_entropy.float().mean()
                + carry_entropy.float().mean()
                + digit_entropy.float().mean(),
            )

        return (
            accumulator,
            torch.stack(invariant_losses).mean(),
            torch.stack(entropies).mean(),
        )

    def _carry_features(
        self,
        raw_digits: Tensor,
        accumulator: Tensor,
        multiplicand: Tensor,
        modulus_digits: Tensor,
        quotient: Tensor,
    ) -> Tensor:
        positions = self.digit_positions.to(raw_digits.dtype)[None, :].expand(
            raw_digits.shape[0],
            -1,
        )
        normalized_quotient = quotient[:, None] / QUOTIENT_MAX
        return torch.stack(
            (
                raw_digits / MAX_RAW_DIGIT_MAGNITUDE,
                torch.tanh(raw_digits / 20.0),
                accumulator / 9.0,
                multiplicand / 9.0,
                modulus_digits / 9.0,
                normalized_quotient.expand_as(raw_digits),
                torch.sin(2.0 * math.pi * positions),
                torch.cos(2.0 * math.pi * positions),
            ),
            dim=-1,
        )


class RedundantDigitModel(nn.Module):
    """Compose one learned redundant-digit square for every requested step."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.transition = LearnedRedundantSquare()
        self.digit_decoder = nn.Sequential(
            nn.Linear(7, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, NUM_DIGITS),
        )
        self.special_logits = nn.Parameter(torch.empty(DIGIT_OFFSET))
        self.curriculum_progress = 0.0
        self.register_buffer(
            "decimal_powers",
            10 ** torch.arange(MAX_DECIMAL_DIGITS, dtype=torch.long),
        )
        positions = torch.linspace(0.0, 1.0, MAX_DECIMAL_DIGITS)
        self.register_buffer("output_positions", positions)
        self.apply(self._initialize)
        nn.init.normal_(self.special_logits, mean=-0.1, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.GRU)):
            for name, parameter in module.named_parameters(recurse=False):
                if "weight" in name:
                    nn.init.normal_(parameter, mean=0.0, std=0.02)
                elif "bias" in name:
                    nn.init.normal_(parameter, mean=0.0, std=0.002)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        if attention_mask is None:
            attention_mask = input_ids != PAD_TOKEN_ID
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)

        modulus, residue, time_steps = self._parse_fields(
            input_ids,
            attention_mask,
        )
        parameter_dtype = self.special_logits.dtype
        modulus_digits = self._decimal_digits(modulus).to(parameter_dtype)
        state = self._decimal_digits(residue).to(parameter_dtype)
        invariant_losses: list[Tensor] = []
        entropies: list[Tensor] = []

        if self.training:
            microsteps = min(
                MAX_DECIMAL_DIGITS,
                1 + int(self.curriculum_progress * 14.0),
            )
            maximum_steps = (
                1 if self.curriculum_progress < 0.6 else MAX_TRAIN_TIME_STEPS
            )
            supervised_weight = min(
                max((self.curriculum_progress - 0.6) / 0.1, 0.0),
                1.0,
            )
        else:
            microsteps = MAX_DECIMAL_DIGITS
            maximum_steps = MAX_EVAL_TIME_STEPS
            supervised_weight = 1.0
        for step in range(maximum_steps):
            active_indices = torch.nonzero(
                time_steps > step,
                as_tuple=False,
            ).squeeze(-1)
            if active_indices.numel() == 0:
                continue
            proposal, invariant_loss, entropy = self.transition(
                state[active_indices],
                modulus_digits[active_indices],
                microsteps,
            )
            state = state.index_copy(0, active_indices, proposal)
            invariant_losses.append(invariant_loss)
            entropies.append(entropy)

        decimal_logits = self._decode_digits(state, modulus_digits)
        slot_logits = torch.cat(
            (
                self.special_logits[None, None, :].expand(
                    input_ids.shape[0],
                    MAX_DECIMAL_DIGITS,
                    -1,
                ),
                decimal_logits,
            ),
            dim=-1,
        )
        logits = self._place_slot_logits(
            slot_logits,
            input_ids,
            attention_mask,
        )
        return (
            logits,
            (
                decimal_logits,
                torch.stack(invariant_losses).mean(),
                torch.stack(entropies).mean(),
                state.new_tensor(supervised_weight),
            ),
        )

    def _decode_digits(self, state: Tensor, modulus_digits: Tensor) -> Tensor:
        positions = self.output_positions.to(state.dtype)[None, :].expand(
            state.shape[0],
            -1,
        )
        features = torch.stack(
            (
                state / 9.0,
                torch.tanh(state),
                modulus_digits / 9.0,
                state * modulus_digits / 81.0,
                positions,
                torch.sin(2.0 * math.pi * positions),
                torch.cos(2.0 * math.pi * positions),
            ),
            dim=-1,
        )
        return self.digit_decoder(features)

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
        return modulus, residue, time_steps


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
        model: RedundantDigitModel,
    ) -> None:
        self.optimizer = optimizer
        self.model = model
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
        self.model.curriculum_progress = progress
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


def build_model(spec: ModelSpec) -> RedundantDigitModel:
    model = RedundantDigitModel(spec)
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
    if not isinstance(model, RedundantDigitModel):
        raise TypeError("unexpected model type")
    scheduler = WallClockSchedule(optimizer, spec.training_time_seconds, model)
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

    if not isinstance(batch.auxiliary, tuple) or len(batch.auxiliary) != 4:
        raise TypeError(
            "auxiliary output must contain decimal logits and invariant terms",
        )
    decimal_logits, invariant_loss, entropy, supervised_weight = batch.auxiliary
    if not isinstance(decimal_logits, Tensor):
        raise TypeError("decimal logits must be a tensor")
    if not isinstance(invariant_loss, Tensor) or invariant_loss.ndim != 0:
        raise TypeError("invariant loss must be a scalar tensor")
    if not isinstance(entropy, Tensor) or entropy.ndim != 0:
        raise TypeError("entropy must be a scalar tensor")
    if not isinstance(supervised_weight, Tensor) or supervised_weight.ndim != 0:
        raise TypeError("supervised weight must be a scalar tensor")
    semantic_decimal_loss = F.cross_entropy(
        decimal_logits.transpose(1, 2).float(),
        target_decimal_digits,
    )
    return (
        supervised_weight.float()
        * (sequence_loss + SEMANTIC_DECIMAL_WEIGHT * semantic_decimal_loss)
        + INVARIANT_WEIGHT * invariant_loss.float()
        + ENTROPY_WEIGHT * entropy.float()
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    max_steps=MAX_TRAINING_STEPS,
)
