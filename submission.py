"""Scalar-quotient coarse-radix recurrence for modular squaring."""

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

RADIX = 16
STATE_DIGITS = 6
EXTENDED_DIGITS = STATE_DIGITS + 1
OUTPUT_DIGITS = 8
WIDTH = 96
CARRY_WIDTH = 64
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
WEAK_DIGIT_TEMPERATURE = 0.25
CONSTRAINT_TEMPERATURE = 0.1
PAD_TOKEN_ID = 0
X_TOKEN_ID = 3
T_TOKEN_ID = 4
DIGIT_OFFSET = 7
NUM_DECIMAL_DIGITS = 10
QUOTIENT_MAX = 2 * RADIX - 2
MAX_RAW_COEFFICIENT = QUOTIENT_MAX * (RADIX - 1)


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


def straight_through_choice(
    logits: Tensor,
    choices: Tensor,
    temperature: float,
    *,
    hard: bool,
) -> tuple[Tensor, Tensor]:
    """Compose categorical values with a hard training forward pass."""
    probabilities = F.softmax(
        logits.float() / max(temperature, 0.05),
        dim=-1,
    ).to(logits.dtype)
    soft_value = torch.einsum("...k,k->...", probabilities, choices)
    hard_value = choices[probabilities.argmax(dim=-1)]
    value = hard_value if hard else hard_value + soft_value - soft_value.detach()
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    return value, entropy


def straight_through_round(value: Tensor, *, hard: bool) -> Tensor:
    """Compose integer carries while retaining an identity gradient."""
    rounded = value.round()
    return rounded if hard else rounded + value - value.detach()


class CoarseRadixSquare(nn.Module):
    """Reduce one square by tied most-significant-first Horner steps."""

    def __init__(self) -> None:
        super().__init__()
        self.quotient_trunk = nn.Sequential(
            nn.Linear(9, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, WIDTH),
            nn.SiLU(),
        )
        self.quotient_head = nn.Linear(WIDTH, 1)
        self.carry_scan = nn.GRU(10, CARRY_WIDTH, batch_first=True)
        self.carry_head = nn.Linear(CARRY_WIDTH, 1)
        self.digit_head = nn.Linear(CARRY_WIDTH, RADIX)
        self.register_buffer(
            "digit_choices",
            torch.arange(RADIX, dtype=torch.float32),
        )
        self.register_buffer(
            "state_powers_float",
            RADIX ** torch.arange(STATE_DIGITS, dtype=torch.float32),
        )
        self.register_buffer(
            "extended_powers_float",
            RADIX ** torch.arange(EXTENDED_DIGITS, dtype=torch.float32),
        )
        self.register_buffer(
            "digit_indices",
            torch.arange(STATE_DIGITS, dtype=torch.long),
        )
        self.register_buffer(
            "extended_positions",
            torch.linspace(0.0, 1.0, EXTENDED_DIGITS),
        )

    def forward(
        self,
        multiplicand: Tensor,
        modulus_digits: Tensor,
        temperature: float,
        hard: bool,
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
            self.state_powers_float,
        )
        multiplicand_value = torch.einsum(
            "bd,d->b",
            multiplicand.float(),
            self.state_powers_float,
        )
        highest_digit = (
            (modulus_digits > 0).long() * self.digit_indices[None, :]
        ).amax(dim=1)
        invariant_losses: list[Tensor] = []
        entropies: list[Tensor] = []

        for rank in range(STATE_DIGITS):
            active_indices = torch.nonzero(
                highest_digit >= rank,
                as_tuple=False,
            ).squeeze(-1)
            if active_indices.numel() == 0:
                break
            active_accumulator = accumulator[active_indices]
            active_multiplicand = multiplicand[active_indices]
            active_modulus = modulus_digits[active_indices]
            active_modulus_value = modulus_value[active_indices]
            digit_indices = highest_digit[active_indices] - rank
            multiplier_digit = active_multiplicand.gather(
                1,
                digit_indices[:, None],
            ).squeeze(1)

            extended_accumulator = F.pad(active_accumulator, (0, 1))
            extended_multiplicand = F.pad(active_multiplicand, (0, 1))
            extended_modulus = F.pad(active_modulus, (0, 1))
            shifted_accumulator = F.pad(
                extended_accumulator[:, :-1],
                (1, 0),
            )
            candidate_coefficients = (
                shifted_accumulator + multiplier_digit[:, None] * extended_multiplicand
            )
            accumulator_value = torch.einsum(
                "bd,d->b",
                active_accumulator.float(),
                self.state_powers_float,
            )
            candidate_value = torch.einsum(
                "bd,d->b",
                candidate_coefficients.float(),
                self.extended_powers_float,
            )
            denominator = active_modulus_value.clamp_min(1.0)
            candidate_ratio = candidate_value / denominator
            rank_feature = torch.full_like(
                candidate_ratio,
                rank / max(STATE_DIGITS - 1, 1),
            )
            quotient_features = torch.stack(
                (
                    candidate_ratio / (QUOTIENT_MAX + 1.0),
                    torch.tanh(candidate_ratio / RADIX),
                    accumulator_value / denominator,
                    multiplicand_value[active_indices] / denominator,
                    multiplier_digit.float() / (RADIX - 1),
                    torch.log1p(denominator) / math.log(16_777_217.0),
                    rank_feature,
                    torch.sin(math.pi * candidate_ratio / RADIX),
                    torch.cos(math.pi * candidate_ratio / RADIX),
                ),
                dim=-1,
            ).to(multiplicand.dtype)
            quotient_hidden = self.quotient_trunk(quotient_features)
            soft_quotient = QUOTIENT_MAX * torch.sigmoid(
                self.quotient_head(quotient_hidden).squeeze(-1),
            )
            quotient = straight_through_round(soft_quotient, hard=hard)

            raw_coefficients = (
                candidate_coefficients - quotient[:, None] * extended_modulus
            )
            carry_features = self._carry_features(
                raw_coefficients,
                extended_accumulator,
                extended_multiplicand,
                extended_modulus,
                quotient,
                multiplier_digit,
            )
            carry_state, _ = self.carry_scan(carry_features)
            soft_carries = self.carry_head(carry_state).squeeze(-1)
            carries = straight_through_round(soft_carries, hard=hard)
            digit_logits = self.digit_head(carry_state)
            next_digits, digit_entropy = straight_through_choice(
                digit_logits,
                self.digit_choices.to(digit_logits.dtype),
                temperature,
                hard=hard,
            )
            incoming_carries = F.pad(carries[:, :-1], (1, 0))
            normalized_coefficients = (
                raw_coefficients + incoming_carries - RADIX * carries
            )
            accumulator = accumulator.index_copy(
                0,
                active_indices,
                next_digits[:, :STATE_DIGITS],
            )
            final_digit_logits = final_digit_logits.index_copy(
                0,
                active_indices,
                digit_logits[:, :STATE_DIGITS],
            )

            reduced_value = candidate_value - quotient.float() * active_modulus_value
            represented_value = torch.einsum(
                "bd,d->b",
                next_digits.float(),
                self.extended_powers_float,
            )
            coefficient_constraints = torch.cat(
                (
                    (normalized_coefficients - next_digits).abs(),
                    carries[:, -1:].abs(),
                ),
                dim=1,
            )
            constraint_magnitudes = torch.log1p(coefficient_constraints)
            smooth_worst_constraint = CONSTRAINT_TEMPERATURE * (
                torch.logsumexp(
                    constraint_magnitudes / CONSTRAINT_TEMPERATURE,
                    dim=1,
                )
                - math.log(coefficient_constraints.shape[1])
            )
            below_zero = F.relu(-reduced_value / denominator)
            above_modulus = F.relu(
                (reduced_value - (active_modulus_value - 1.0)) / denominator,
            )
            representation_error = (
                represented_value - reduced_value
            ).abs() / denominator
            invariant_losses.append(
                smooth_worst_constraint.square().mean()
                + torch.log1p(below_zero).square().mean()
                + torch.log1p(above_modulus).square().mean()
                + torch.log1p(representation_error).square().mean(),
            )
            entropies.append(digit_entropy.float().mean())

        return (
            accumulator,
            final_digit_logits,
            torch.stack(invariant_losses).mean(),
            torch.stack(entropies).mean(),
        )

    def _carry_features(
        self,
        raw: Tensor,
        accumulator: Tensor,
        multiplicand: Tensor,
        modulus: Tensor,
        quotient: Tensor,
        multiplier: Tensor,
    ) -> Tensor:
        positions = self.extended_positions.to(raw.dtype)[None, :].expand(
            raw.shape[0],
            -1,
        )
        normalized_quotient = quotient[:, None] / (QUOTIENT_MAX + 1.0)
        normalized_multiplier = multiplier[:, None] / (RADIX - 1)
        return torch.stack(
            (
                raw / (MAX_RAW_COEFFICIENT + 1.0),
                torch.tanh(raw / RADIX),
                accumulator / (RADIX - 1),
                multiplicand / (RADIX - 1),
                modulus / (RADIX - 1),
                normalized_quotient.expand_as(raw),
                normalized_multiplier.expand_as(raw),
                positions,
                torch.sin(2.0 * math.pi * positions),
                torch.cos(2.0 * math.pi * positions),
            ),
            dim=-1,
        )


class CoarseRadixModel(nn.Module):
    """Compose one tied coarse-radix reducer per requested square."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.transition = CoarseRadixSquare()
        self.decimal_decoder = nn.Sequential(
            nn.Linear(2 * FOURIER_HARMONICS, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, NUM_DECIMAL_DIGITS),
        )
        self.special_logits = nn.Parameter(torch.empty(DIGIT_OFFSET))
        self.gate_temperature = 2.0
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
        modulus_digits = self._radix_digits(modulus).to(parameter_dtype)
        state = self._radix_digits(residue).to(parameter_dtype)
        state_logits = state.new_zeros(
            state.shape[0],
            STATE_DIGITS,
            RADIX,
        )
        invariant_losses: list[Tensor] = []
        entropies: list[Tensor] = []
        maximum_steps = MAX_TRAIN_TIME_STEPS if self.training else MAX_EVAL_TIME_STEPS

        for step in range(maximum_steps):
            active_indices = torch.nonzero(
                time_steps > step,
                as_tuple=False,
            ).squeeze(-1)
            if active_indices.numel() == 0:
                break
            (
                proposal,
                proposal_logits,
                invariant_loss,
                entropy,
            ) = self.transition(
                state[active_indices],
                modulus_digits[active_indices],
                self.gate_temperature,
                not self.training,
            )
            state = state.index_copy(0, active_indices, proposal)
            state_logits = state_logits.index_copy(
                0,
                active_indices,
                proposal_logits,
            )
            invariant_losses.append(invariant_loss)
            entropies.append(entropy)

        decoder_state = state.detach() if self.training else state
        decimal_logits = self._decode_decimal(decoder_state)
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
                modulus_digits,
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
    """Coordinate the learning rate and categorical temperature."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        training_time_seconds: float,
        model: CoarseRadixModel,
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
        self.model.gate_temperature = 2.0 * (0.05 / 2.0) ** progress
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


def build_model(spec: ModelSpec) -> CoarseRadixModel:
    model = CoarseRadixModel(spec)
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
    if not isinstance(model, CoarseRadixModel):
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
    masked_token_losses = token_losses.masked_fill(
        ~batch.valid_mask,
        -torch.inf,
    )
    sequence_losses = WEAK_DIGIT_TEMPERATURE * (
        torch.logsumexp(
            masked_token_losses / WEAK_DIGIT_TEMPERATURE,
            dim=1,
        )
        - target_counts.clamp_min(1).float().log()
    )
    sequence_loss = sequence_losses[valid_rows].mean()

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

    if not isinstance(batch.auxiliary, tuple) or len(batch.auxiliary) != 4:
        raise TypeError(
            "auxiliary output must contain radix logits and modulus digits",
        )
    radix_logits, invariant_loss, entropy, modulus_digits = batch.auxiliary
    if not isinstance(radix_logits, Tensor):
        raise TypeError("radix logits must be a tensor")
    if not isinstance(invariant_loss, Tensor) or invariant_loss.ndim != 0:
        raise TypeError("invariant loss must be a scalar tensor")
    if not isinstance(entropy, Tensor) or entropy.ndim != 0:
        raise TypeError("entropy must be a scalar tensor")
    if not isinstance(modulus_digits, Tensor):
        raise TypeError("modulus digits must be a tensor")
    radix_losses = F.cross_entropy(
        radix_logits.transpose(1, 2).float(),
        target_radix_digits,
        reduction="none",
    )
    nonzero_modulus = modulus_digits != 0
    significant_mask = torch.flip(
        torch.cumsum(
            torch.flip(nonzero_modulus.long(), dims=(1,)),
            dim=1,
        ),
        dims=(1,),
    ).bool()
    semantic_radix_loss = (
        (radix_losses * significant_mask).sum(dim=1)
        / significant_mask.sum(dim=1).clamp_min(1)
    ).mean()
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
