"""Bounded radix-gate recurrence for repeated modular squaring."""

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

RADIX = 4
STATE_DIGITS = 12
OUTPUT_DIGITS = 8
WIDTH = 64
CARRY_WIDTH = 48
FOURIER_HARMONICS = 8
MAX_TRAIN_TIME_STEPS = 16
MAX_EVAL_TIME_STEPS = 64
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 2_048
MAX_TRAINING_STEPS = 1_000_000
BASE_LEARNING_RATE = 3e-3
MIN_LEARNING_RATE_RATIO = 0.05
WARMUP_FRACTION = 0.05
SEMANTIC_RADIX_WEIGHT = 0.75
INVARIANT_WEIGHT = 0.5
ENTROPY_WEIGHT = 0.002
PAD_TOKEN_ID = 0
X_TOKEN_ID = 3
T_TOKEN_ID = 4
DIGIT_OFFSET = 7
NUM_DECIMAL_DIGITS = 10
QUOTIENT_MIN = 0
QUOTIENT_MAX = 2 * RADIX - 2
CARRY_MIN = -12
CARRY_MAX = 12
MAX_RAW_DIGIT_MAGNITUDE = 24.0


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


class LearnedRadixSquare(nn.Module):
    """Learn bounded Horner reduction and local radix carry gates."""

    def __init__(self) -> None:
        super().__init__()
        self.quotient_network = nn.Sequential(
            nn.Linear(7, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, QUOTIENT_MAX - QUOTIENT_MIN + 1),
        )
        self.carry_scan = nn.GRU(9, CARRY_WIDTH, batch_first=True)
        self.carry_head = nn.Linear(
            CARRY_WIDTH,
            CARRY_MAX - CARRY_MIN + 1,
        )
        self.digit_head = nn.Linear(CARRY_WIDTH, RADIX)
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
            torch.arange(RADIX, dtype=torch.float32),
        )
        self.register_buffer(
            "radix_powers_float",
            RADIX ** torch.arange(STATE_DIGITS, dtype=torch.float32),
        )
        positions = torch.linspace(0.0, 1.0, STATE_DIGITS)
        self.register_buffer("digit_positions", positions)
        self.register_buffer(
            "digit_indices",
            torch.arange(STATE_DIGITS, dtype=torch.long),
        )

    def forward(
        self,
        multiplicand: Tensor,
        modulus_digits: Tensor,
        microsteps: int = STATE_DIGITS,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        accumulator = torch.zeros_like(multiplicand)
        final_digit_logits = multiplicand.new_zeros(
            multiplicand.shape[0],
            STATE_DIGITS,
            RADIX,
        )
        modulus_value = torch.einsum(
            "bd,d->b",
            modulus_digits.float(),
            self.radix_powers_float,
        )
        multiplicand_value = torch.einsum(
            "bd,d->b",
            multiplicand.float(),
            self.radix_powers_float,
        )
        highest_digit = (
            (modulus_digits > 0).long() * self.digit_indices[None, :]
        ).amax(dim=1)
        invariant_losses: list[Tensor] = []
        entropies: list[Tensor] = []

        for rank in range(microsteps):
            active_indices = torch.nonzero(
                highest_digit >= rank,
                as_tuple=False,
            ).squeeze(-1)
            if active_indices.numel() == 0:
                continue
            active_accumulator = accumulator[active_indices]
            active_multiplicand = multiplicand[active_indices]
            active_modulus = modulus_digits[active_indices]
            active_modulus_value = modulus_value[active_indices]
            active_multiplicand_value = multiplicand_value[active_indices]
            digit_indices = highest_digit[active_indices] - rank
            multiplier_digit = active_multiplicand.gather(
                1,
                digit_indices[:, None],
            ).squeeze(1)

            shifted_accumulator = F.pad(
                active_accumulator[:, :-1],
                (1, 0),
            )
            candidate_digits = (
                shifted_accumulator + multiplier_digit[:, None] * active_multiplicand
            )
            accumulator_value = torch.einsum(
                "bd,d->b",
                active_accumulator.float(),
                self.radix_powers_float,
            )
            candidate_value = torch.einsum(
                "bd,d->b",
                candidate_digits.float(),
                self.radix_powers_float,
            )
            denominator = active_modulus_value.clamp_min(1.0)
            candidate_ratio = candidate_value / denominator
            quotient_features = torch.stack(
                (
                    candidate_ratio / (QUOTIENT_MAX + 1.0),
                    torch.tanh(candidate_ratio),
                    accumulator_value / denominator,
                    active_multiplicand_value / denominator,
                    multiplier_digit.float() / (RADIX - 1),
                    torch.log1p(denominator) / math.log(16_777_217.0),
                    torch.ones_like(candidate_ratio),
                ),
                dim=-1,
            ).to(multiplicand.dtype)
            quotient_logits = self.quotient_network(quotient_features)
            quotient, quotient_entropy = straight_through_choice(
                quotient_logits,
                self.quotient_choices.to(quotient_logits.dtype),
            )

            raw_digits = candidate_digits - quotient[:, None] * active_modulus
            raw_features = self._carry_features(
                raw_digits,
                active_accumulator,
                active_multiplicand,
                active_modulus,
                quotient,
                multiplier_digit,
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
            normalized_coefficients = raw_digits + incoming_carries - RADIX * carries
            accumulator = accumulator.index_copy(
                0,
                active_indices,
                next_digits,
            )
            final_digit_logits = final_digit_logits.index_copy(
                0,
                active_indices,
                digit_logits,
            )

            reduced_value = candidate_value - quotient.float() * active_modulus_value
            represented_value = torch.einsum(
                "bd,d->b",
                next_digits.float(),
                self.radix_powers_float,
            )
            below_zero = F.relu(-reduced_value / denominator)
            above_modulus = F.relu(
                (reduced_value - (active_modulus_value - 1.0)) / denominator,
            )
            below_digit = F.relu(
                -normalized_coefficients.float() / RADIX,
            )
            above_digit = F.relu(
                (normalized_coefficients.float() - (RADIX - 1)) / RADIX,
            )
            representation_error = (
                represented_value - reduced_value
            ).abs() / denominator
            overflow = carries[:, -1].float() / max(
                abs(CARRY_MIN),
                CARRY_MAX,
            )
            invariant_losses.append(
                torch.log1p(below_zero).square().mean()
                + torch.log1p(above_modulus).square().mean()
                + torch.log1p(below_digit).square().mean()
                + torch.log1p(above_digit).square().mean()
                + torch.log1p(representation_error).square().mean()
                + overflow.square().mean(),
            )
            entropies.append(
                quotient_entropy.float().mean()
                + carry_entropy.float().mean()
                + digit_entropy.float().mean(),
            )

        return (
            accumulator,
            final_digit_logits,
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
        multiplier_digit: Tensor,
    ) -> Tensor:
        positions = self.digit_positions.to(raw_digits.dtype)[None, :].expand(
            raw_digits.shape[0],
            -1,
        )
        normalized_quotient = quotient[:, None] / QUOTIENT_MAX
        normalized_multiplier = multiplier_digit[:, None] / (RADIX - 1)
        return torch.stack(
            (
                raw_digits / MAX_RAW_DIGIT_MAGNITUDE,
                torch.tanh(raw_digits / RADIX),
                accumulator / (RADIX - 1),
                multiplicand / (RADIX - 1),
                modulus_digits / (RADIX - 1),
                normalized_quotient.expand_as(raw_digits),
                normalized_multiplier.expand_as(raw_digits),
                torch.sin(2.0 * math.pi * positions),
                torch.cos(2.0 * math.pi * positions),
            ),
            dim=-1,
        )


class RadixGateModel(nn.Module):
    """Compose one bounded learned radix circuit per requested square."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.transition = LearnedRadixSquare()
        self.decimal_decoder = nn.Sequential(
            nn.Linear(2 * FOURIER_HARMONICS, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, NUM_DECIMAL_DIGITS),
        )
        self.special_logits = nn.Parameter(torch.empty(DIGIT_OFFSET))
        self.register_buffer(
            "radix_powers",
            RADIX ** torch.arange(STATE_DIGITS, dtype=torch.long),
        )
        self.register_buffer(
            "radix_powers_float",
            RADIX ** torch.arange(STATE_DIGITS, dtype=torch.float32),
        )
        self.register_buffer(
            "decimal_place_scales",
            10.0 ** torch.arange(1, OUTPUT_DIGITS + 1, dtype=torch.float32),
        )
        self.register_buffer(
            "fourier_indices",
            torch.arange(1, FOURIER_HARMONICS + 1, dtype=torch.float32),
        )
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
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        if attention_mask is None:
            attention_mask = input_ids != PAD_TOKEN_ID
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)

        modulus, residue, time_steps = self._parse_fields(
            input_ids,
            attention_mask,
        )
        parameter_dtype = self.special_logits.dtype
        modulus_digits = self._radix_digits(modulus).to(parameter_dtype)
        state = self._radix_digits(residue).to(parameter_dtype)
        state_logits = state.new_zeros(
            state.shape[0],
            STATE_DIGITS,
            RADIX,
        )
        invariant_losses: list[Tensor] = []
        entropies: list[Tensor] = []

        microsteps = STATE_DIGITS
        maximum_steps = MAX_TRAIN_TIME_STEPS if self.training else MAX_EVAL_TIME_STEPS

        for step in range(maximum_steps):
            active_indices = torch.nonzero(
                time_steps > step,
                as_tuple=False,
            ).squeeze(-1)
            if active_indices.numel() == 0:
                continue
            (
                proposal,
                proposal_logits,
                invariant_loss,
                entropy,
            ) = self.transition(
                state[active_indices],
                modulus_digits[active_indices],
                microsteps,
            )
            state = state.index_copy(0, active_indices, proposal)
            state_logits = state_logits.index_copy(
                0,
                active_indices,
                proposal_logits,
            )
            invariant_losses.append(invariant_loss)
            entropies.append(entropy)

        decimal_logits = self._decode_decimal(state)
        slot_logits = torch.cat(
            (
                self.special_logits[None, None, :].expand(
                    input_ids.shape[0],
                    OUTPUT_DIGITS,
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
                state_logits,
                torch.stack(invariant_losses).mean(),
                torch.stack(entropies).mean(),
            ),
        )

    def _decode_decimal(self, state: Tensor) -> Tensor:
        state_value = torch.einsum(
            "bd,d->b",
            state.float(),
            self.radix_powers_float,
        )
        phase = (
            2.0
            * math.pi
            * state_value[:, None, None]
            * self.fourier_indices[None, None, :]
            / self.decimal_place_scales[None, :, None]
        )
        features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        return self.decimal_decoder(features.to(state.dtype))

    def _radix_digits(self, value: Tensor) -> Tensor:
        return (value[:, None] // self.radix_powers) % RADIX

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
            max=OUTPUT_DIGITS - 1,
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
                max=NUM_DECIMAL_DIGITS - 1,
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


def build_model(spec: ModelSpec) -> RadixGateModel:
    model = RadixGateModel(spec)
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
            max=NUM_DECIMAL_DIGITS - 1,
        )
        target_values = torch.where(
            valid,
            target_values * 10 + digit,
            target_values,
        )
    radix_powers = RADIX ** torch.arange(
        STATE_DIGITS,
        device=batch.labels.device,
        dtype=torch.long,
    )
    target_radix_digits = (target_values[:, None] // radix_powers) % RADIX

    if not isinstance(batch.auxiliary, tuple) or len(batch.auxiliary) != 3:
        raise TypeError(
            "auxiliary output must contain radix logits and invariant terms",
        )
    radix_logits, invariant_loss, entropy = batch.auxiliary
    if not isinstance(radix_logits, Tensor):
        raise TypeError("radix logits must be a tensor")
    if not isinstance(invariant_loss, Tensor) or invariant_loss.ndim != 0:
        raise TypeError("invariant loss must be a scalar tensor")
    if not isinstance(entropy, Tensor) or entropy.ndim != 0:
        raise TypeError("entropy must be a scalar tensor")
    semantic_radix_loss = F.cross_entropy(
        radix_logits.transpose(1, 2).float(),
        target_radix_digits,
    )
    return (
        sequence_loss
        + SEMANTIC_RADIX_WEIGHT * semantic_radix_loss
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
