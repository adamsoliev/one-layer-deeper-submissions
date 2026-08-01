"""Paired-step factor-coordinate automaton for One Layer Deeper."""

from __future__ import annotations

import math

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

WIDTH = 256
TRANSITION_WIDTH = 1_024
MAX_VALUE = 8_192
RESIDUE_CODES = 2_048
PAIR_MEMORY_SIZE = 131_071
PAIR_MEMORY_WIDTH = 64
MIN_CANDIDATE_FACTOR = 2
MAX_CANDIDATE_FACTOR = 64
COORDINATE_CLASSES = MAX_CANDIDATE_FACTOR
FACTOR_SLOTS = MAX_CANDIDATE_FACTOR + 1
NUM_CANDIDATE_FACTORS = (
    MAX_CANDIDATE_FACTOR - MIN_CANDIDATE_FACTOR + 1
)
MEMORY_TOP_K = 8
MAX_OUTPUT_DIGITS = 4
TRAIN_BATCH_SIZE = 512
MAX_TRAINING_STEPS = 2_000
WARMUP_STEPS = 50
MAX_TRAIN_T = 3
MAX_EVAL_T = 8
MAX_TRAIN_MACRO_STEPS = 2
MAX_EVAL_MACRO_STEPS = 4
PAD_TOKEN_ID = 0
X_TOKEN_ID = 3
T_TOKEN_ID = 4
DIGIT_OFFSET = 7
NUM_DIGITS = 10
RECONSTRUCTION_WEIGHT = 0.25
ENTROPY_WEIGHT = 0.05
RESIDUE_SUPERVISION_WEIGHT = 1.0
FACTOR_CONSISTENCY_WEIGHT = 1.0
FACTOR_ENTROPY_WEIGHT = 0.05
ROW_MARKER_BASE = -100.0


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, hidden: Tensor) -> Tensor:
        return F.rms_norm(hidden, (hidden.shape[-1],), self.weight)


class SquaringTransition(nn.Module):
    """Produce a codebook query from residue, modulus, and learned memory."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_norm = RMSNorm(5 * WIDTH)
        self.left = nn.Linear(5 * WIDTH, TRANSITION_WIDTH)
        self.right = nn.Linear(5 * WIDTH, TRANSITION_WIDTH)
        self.query = nn.Linear(TRANSITION_WIDTH, WIDTH)

    def forward(
        self,
        state: Tensor,
        modulus: Tensor,
        memory: Tensor,
    ) -> Tensor:
        features = torch.cat(
            (
                state,
                modulus,
                state * modulus,
                state - modulus,
                memory,
            ),
            dim=-1,
        )
        features = self.feature_norm(features)
        hidden = F.silu(self.left(features)) * self.right(features)
        return self.query(hidden)


class CanonicalResidueModel(nn.Module):
    """Compose learned one- and two-squaring coordinate operators."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.modulus_embedding = nn.Embedding(MAX_VALUE, WIDTH)
        self.residue_embedding = nn.Embedding(MAX_VALUE, WIDTH)
        self.pair_memory = nn.Embedding(
            PAIR_MEMORY_SIZE,
            PAIR_MEMORY_WIDTH,
        )
        self.factor_selector = nn.Embedding(
            MAX_VALUE,
            NUM_CANDIDATE_FACTORS,
        )
        self.memory_projection = nn.Linear(
            PAIR_MEMORY_WIDTH,
            WIDTH,
            bias=False,
        )
        self.modulus_norm = RMSNorm(WIDTH)
        self.residue_norm = RMSNorm(WIDTH)
        self.query_norm = RMSNorm(WIDTH)
        self.transition = SquaringTransition()
        self.coordinate_transition = nn.Embedding(
            FACTOR_SLOTS * COORDINATE_CLASSES,
            COORDINATE_CLASSES,
        )
        self.paired_coordinate_transition = nn.Embedding(
            FACTOR_SLOTS * COORDINATE_CLASSES,
            COORDINATE_CLASSES,
        )
        self.code_log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.answer_decoder = nn.Linear(
            WIDTH,
            MAX_OUTPUT_DIGITS * spec.vocab_size,
        )
        self.apply(self._initialize)
        nn.init.normal_(self.transition.query.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.factor_selector.weight)
        nn.init.zeros_(self.coordinate_transition.weight)
        nn.init.zeros_(self.paired_coordinate_transition.weight)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        if attention_mask is None:
            attention_mask = input_ids != PAD_TOKEN_ID
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)

        modulus, residue, time_steps = self._parse_fields(
            input_ids,
            attention_mask,
        )
        modulus_index = modulus.clamp(max=MAX_VALUE - 1)
        residue_index = residue.clamp(max=MAX_VALUE - 1)
        modulus_vector = self.modulus_norm(
            self.modulus_embedding(modulus_index),
        )
        factor_probabilities = F.softmax(
            self.factor_selector(modulus_index),
            dim=-1,
        )
        initial_state = self.residue_norm(
            self.residue_embedding(residue_index),
        )
        reconstruction_loss = self._reconstruction_loss(
            initial_state,
            input_ids,
            attention_mask,
        )

        state = initial_state
        state_probabilities = F.one_hot(
            residue.clamp(max=RESIDUE_CODES - 1),
            num_classes=RESIDUE_CODES,
        ).to(state.dtype)
        state_indices = residue.clamp(max=RESIDUE_CODES - 1)[:, None].expand(
            -1,
            MEMORY_TOP_K,
        )
        state_weights = state.new_zeros(
            state.shape[0],
            MEMORY_TOP_K,
        )
        state_weights[:, 0] = 1.0
        entropy_sum = state.new_zeros(state.shape[0])
        active_steps = state.new_zeros(state.shape[0])
        single_first = time_steps.remainder(2).to(dtype=torch.bool)
        macro_steps = time_steps // 2 + single_first.long()
        maximum_steps = (
            MAX_TRAIN_MACRO_STEPS
            if self.training
            else MAX_EVAL_MACRO_STEPS
        )
        for step in range(maximum_steps):
            use_single = single_first & (step == 0)
            transition_state = (
                self._factor_fiber_state(
                    modulus_index,
                    state_indices,
                    factor_probabilities,
                )
                if step == 0
                else self._symmetric_state(
                    state,
                    modulus_index,
                    state_indices,
                    state_weights,
                )
            )
            memory = self._memory_context(
                modulus_index,
                state_indices,
                state_weights,
            )
            if step == 0:
                memory = torch.zeros_like(memory)
            query = self.transition(
                transition_state,
                modulus_vector,
                memory,
            )
            (
                candidate,
                entropy,
                probabilities,
                coordinate_logits,
                coordinate_factors,
            ) = self._canonicalize(
                query,
                modulus_index,
                state_indices,
                state_weights,
                factor_probabilities,
                use_single,
            )
            if step == 0:
                first_coordinate_logits = coordinate_logits
                first_coordinate_factors = coordinate_factors
                candidate_factor_logits = self._candidate_factor_logits(
                    residue,
                    use_single,
                )
            candidate_weights, candidate_indices = probabilities.topk(
                MEMORY_TOP_K,
                dim=-1,
            )
            candidate_weights = candidate_weights / candidate_weights.sum(
                dim=-1,
                keepdim=True,
            )
            active = macro_steps > step
            state = torch.where(active[:, None], candidate, state)
            state_probabilities = torch.where(
                active[:, None],
                probabilities,
                state_probabilities,
            )
            state_indices = torch.where(
                active[:, None],
                candidate_indices,
                state_indices,
            )
            state_weights = torch.where(
                active[:, None],
                candidate_weights,
                state_weights,
            )
            entropy_sum = entropy_sum + entropy * active
            active_steps = active_steps + active

        code_entropy = (
            entropy_sum / active_steps.clamp_min(1.0)
        ).mean() / math.log(RESIDUE_CODES)
        logits = self._decode_sequence(state, input_ids, attention_mask)
        return logits, (
            reconstruction_loss,
            code_entropy,
            state_probabilities,
            factor_probabilities,
            modulus,
            residue,
            time_steps,
            first_coordinate_logits,
            first_coordinate_factors,
            candidate_factor_logits,
        )

    def _candidate_factor_logits(
        self,
        residue: Tensor,
        use_single: Tensor,
    ) -> Tensor:
        factors = torch.arange(
            MIN_CANDIDATE_FACTOR,
            MAX_CANDIDATE_FACTOR + 1,
            device=residue.device,
        )[None, :]
        factor_residues = residue[:, None] % factors
        input_coordinates = torch.minimum(
            factor_residues,
            (factors - factor_residues) % factors,
        )
        keys = factors * COORDINATE_CLASSES + input_coordinates
        single_logits = self.coordinate_transition(keys)
        paired_logits = self.paired_coordinate_transition(keys)
        return torch.where(
            use_single[:, None, None],
            single_logits,
            paired_logits,
        )

    def _memory_context(
        self,
        modulus: Tensor,
        residue_indices: Tensor,
        residue_weights: Tensor,
    ) -> Tensor:
        modulus = modulus.clamp_min(1)[:, None]
        residues = residue_indices % modulus
        reflected = (modulus - residues) % modulus
        representatives = torch.minimum(residues, reflected)
        keys = (
            modulus * RESIDUE_CODES + representatives
        ) % PAIR_MEMORY_SIZE
        memories = self.pair_memory(keys)
        memory = (
            memories * residue_weights.to(memories.dtype)[..., None]
        ).sum(dim=1)
        return self.memory_projection(memory)

    def _factor_fiber_state(
        self,
        modulus: Tensor,
        residue_indices: Tensor,
        factor_probabilities: Tensor,
    ) -> Tensor:
        integer_modulus = modulus.clamp_min(1)[:, None]
        factors = (
            factor_probabilities.argmax(dim=-1)
            + MIN_CANDIDATE_FACTOR
        )[:, None]
        partners = torch.div(
            integer_modulus + factors // 2,
            factors,
            rounding_mode="floor",
        ).clamp(min=2, max=MAX_VALUE - 1)
        residues = residue_indices[:, :1] % integer_modulus
        target_factor_residues = residues % factors
        target_factor_coordinates = torch.minimum(
            target_factor_residues,
            (factors - target_factor_residues) % factors,
        )
        target_partner_residues = residues % partners
        target_partner_coordinates = torch.minimum(
            target_partner_residues,
            (partners - target_partner_residues) % partners,
        )

        candidates = torch.arange(
            RESIDUE_CODES,
            device=modulus.device,
        )[None, :]
        candidate_factor_residues = candidates % factors
        candidate_factor_coordinates = torch.minimum(
            candidate_factor_residues,
            (factors - candidate_factor_residues) % factors,
        )
        candidate_partner_residues = candidates % partners
        candidate_partner_coordinates = torch.minimum(
            candidate_partner_residues,
            (partners - candidate_partner_residues) % partners,
        )
        same_fiber = (
            (candidates < integer_modulus)
            & (
                candidate_factor_coordinates
                == target_factor_coordinates
            )
            & (
                candidate_partner_coordinates
                == target_partner_coordinates
            )
        )
        weights = same_fiber.to(self.residue_embedding.weight.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        codes = self.residue_norm(
            self.residue_embedding.weight[:RESIDUE_CODES],
        )
        return self.residue_norm(weights @ codes)

    def _symmetric_state(
        self,
        state: Tensor,
        modulus: Tensor,
        residue_indices: Tensor,
        residue_weights: Tensor,
    ) -> Tensor:
        modulus = modulus.clamp_min(1)[:, None]
        residues = residue_indices % modulus
        reflected = ((modulus - residues) % modulus).clamp(
            max=MAX_VALUE - 1,
        )
        reflected_codes = self.residue_norm(
            self.residue_embedding(reflected),
        )
        reflected_state = (
            reflected_codes
            * residue_weights.to(reflected_codes.dtype)[..., None]
        ).sum(dim=1)
        return self.residue_norm(state + reflected_state)

    def _canonicalize(
        self,
        query: Tensor,
        modulus: Tensor,
        residue_indices: Tensor,
        residue_weights: Tensor,
        factor_probabilities: Tensor,
        use_single: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        query = F.normalize(self.query_norm(query), dim=-1)
        code_values = torch.arange(
            RESIDUE_CODES,
            device=query.device,
        )
        codes = F.normalize(
            self.residue_norm(
                self.residue_embedding(code_values),
            ),
            dim=-1,
        )
        scale = self.code_log_scale.exp().clamp(max=30.0)
        scores = scale * (query @ codes.T)

        integer_modulus = modulus.clamp_min(1)[:, None]
        factors = (
            factor_probabilities.argmax(dim=-1)
            + MIN_CANDIDATE_FACTOR
        )[:, None]
        partners = torch.div(
            integer_modulus + factors // 2,
            factors,
            rounding_mode="floor",
        ).clamp(min=2, max=MAX_VALUE - 1)
        low_factors = torch.minimum(factors, partners)
        high_factors = torch.maximum(factors, partners)
        residues = residue_indices % integer_modulus
        low_residues = residues % low_factors
        low_input_coordinates = torch.minimum(
            low_residues,
            (low_factors - low_residues) % low_factors,
        )
        high_residues = residues % high_factors
        high_input_coordinates = torch.minimum(
            high_residues,
            (high_factors - high_residues) % high_factors,
        )
        low_keys = (
            low_factors.clamp(max=MAX_CANDIDATE_FACTOR)
            * COORDINATE_CLASSES
            + low_input_coordinates % COORDINATE_CLASSES
        )
        high_keys = (
            high_factors.clamp(max=MAX_CANDIDATE_FACTOR)
            * COORDINATE_CLASSES
            + high_input_coordinates % COORDINATE_CLASSES
        )
        low_transition_logits = torch.where(
            use_single[:, None, None],
            self.coordinate_transition(low_keys),
            self.paired_coordinate_transition(low_keys),
        )
        high_transition_logits = torch.where(
            use_single[:, None, None],
            self.coordinate_transition(high_keys),
            self.paired_coordinate_transition(high_keys),
        )
        low_logits = (
            low_transition_logits
            * residue_weights[..., None].to(scores.dtype)
        ).sum(dim=1)
        high_logits = (
            high_transition_logits
            * residue_weights[..., None].to(scores.dtype)
        ).sum(dim=1)
        candidates = code_values[None, :]
        low_output_coordinates = (
            (candidates % low_factors) % COORDINATE_CLASSES
        )
        high_output_coordinates = (
            (candidates % high_factors) % COORDINATE_CLASSES
        )
        coordinate_scores = (
            low_logits.gather(1, low_output_coordinates)
            + high_logits.gather(1, high_output_coordinates)
        )
        coordinate_logits = torch.stack((low_logits, high_logits), dim=1)
        coordinate_factors = torch.cat((low_factors, high_factors), dim=1)
        if self.training:
            scores = scores + coordinate_scores
        else:
            predicted_coordinates = coordinate_logits.argmax(dim=-1)
            valid_coordinates = (
                (candidates < integer_modulus)
                & (
                    low_output_coordinates
                    == predicted_coordinates[:, :1]
                )
                & (
                    high_output_coordinates
                    == predicted_coordinates[:, 1:]
                )
            )
            scores = scores.masked_fill(
                ~valid_coordinates,
                torch.finfo(scores.dtype).min,
            )
        probabilities = F.softmax(scores, dim=-1)
        state = probabilities @ codes
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
            dim=-1,
        )
        return (
            self.residue_norm(state),
            entropy,
            probabilities,
            coordinate_logits,
            coordinate_factors,
        )

    def _decode_sequence(
        self,
        state: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        digit_logits = self.answer_decoder(state).view(
            input_ids.shape[0],
            MAX_OUTPUT_DIGITS,
            self.config.vocab_size,
        )
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)[None, :]
        valid_lengths = attention_mask.long().sum(dim=1, keepdim=True)
        answer_slots = (valid_lengths - positions - 1).clamp(
            min=0,
            max=MAX_OUTPUT_DIGITS - 1,
        )
        batch_indices = torch.arange(
            input_ids.shape[0],
            device=input_ids.device,
        )[:, None]
        logits = digit_logits[batch_indices, answer_slots].clone()
        row_markers = (
            logits.new_tensor(ROW_MARKER_BASE)
            - torch.arange(
                input_ids.shape[0],
                device=input_ids.device,
                dtype=logits.dtype,
            )
        )
        logits[:, :, PAD_TOKEN_ID] = row_markers[:, None]
        return logits

    def _reconstruction_loss(
        self,
        initial_state: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)[None, :]
        x_positions = torch.argmax((input_ids == X_TOKEN_ID).long(), dim=1)
        t_positions = torch.argmax((input_ids == T_TOKEN_ID).long(), dim=1)
        x_mask = (
            (input_ids >= DIGIT_OFFSET)
            & attention_mask
            & (positions > x_positions[:, None])
            & (positions < t_positions[:, None])
        )
        significance = (
            torch.flip(
                torch.cumsum(torch.flip(x_mask.long(), dims=(1,)), dim=1),
                dims=(1,),
            )
            - 1
        ).clamp(min=0, max=MAX_OUTPUT_DIGITS - 1)
        digit_logits = self.answer_decoder(initial_state).view(
            input_ids.shape[0],
            MAX_OUTPUT_DIGITS,
            self.config.vocab_size,
        )
        batch_indices = torch.arange(
            input_ids.shape[0],
            device=input_ids.device,
        )[:, None]
        reconstruction_logits = digit_logits[batch_indices, significance]
        return F.cross_entropy(
            reconstruction_logits[x_mask].float(),
            input_ids[x_mask],
        )

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
        return modulus, residue, time_steps.clamp(min=1, max=MAX_EVAL_T)


def build_model(spec: ModelSpec) -> CanonicalResidueModel:
    model = CanonicalResidueModel(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(
    model: nn.Module,
    spec: OptimizerSpec,
) -> OptimizerBundle:
    if not isinstance(model, CanonicalResidueModel):
        raise TypeError("optimizer requires CanonicalResidueModel")
    table_parameters = [
        model.factor_selector.weight,
        model.coordinate_transition.weight,
        model.paired_coordinate_transition.weight,
    ]
    table_parameter_ids = {id(parameter) for parameter in table_parameters}
    base_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in table_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": 1e-3},
            {
                "params": table_parameters,
                "lr": 1e-2,
                "weight_decay": 0.0,
            },
        ],
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.02,
        capturable=spec.device_type == "cuda",
    )

    def learning_rate_multiplier(step: int) -> float:
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        progress = min(
            (step - WARMUP_STEPS) / (MAX_TRAINING_STEPS - WARMUP_STEPS),
            1.0,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_multiplier,
    )
    return OptimizerBundle(optimizer, scheduler)


def _factor_collision_loss(
    factor_probabilities: Tensor,
    modulus: Tensor,
    residue: Tensor,
    time_steps: Tensor,
    target_residues: Tensor,
    valid_rows: Tensor,
) -> Tensor:
    selected = valid_rows.nonzero(as_tuple=False).flatten()
    zero = factor_probabilities.sum() * 0.0
    if selected.numel() < 2:
        return zero

    groups = (
        (
            modulus[selected] * (MAX_TRAIN_T + 1)
            + time_steps[selected]
        )
        * RESIDUE_CODES
        + target_residues[selected]
    )
    pairs = torch.triu(
        groups[:, None] == groups[None, :],
        diagonal=1,
    ).nonzero(as_tuple=False)
    if pairs.numel() == 0:
        return zero

    integer_modulus = modulus[selected].clamp_min(1)[:, None]
    factors = torch.arange(
        MIN_CANDIDATE_FACTOR,
        MAX_CANDIDATE_FACTOR + 1,
        device=modulus.device,
    )[None, :]
    partners = torch.div(
        integer_modulus + factors // 2,
        factors,
        rounding_mode="floor",
    ).clamp(min=2, max=MAX_VALUE - 1)
    residues = residue[selected, None] % integer_modulus
    factor_residues = residues % factors
    factor_coordinates = torch.minimum(
        factor_residues,
        (factors - factor_residues) % factors,
    )
    partner_residues = residues % partners
    partner_coordinates = torch.minimum(
        partner_residues,
        (partners - partner_residues) % partners,
    )

    left, right = pairs[:, 0], pairs[:, 1]
    matches = (
        (factor_coordinates[left] == factor_coordinates[right])
        & (partner_coordinates[left] == partner_coordinates[right])
    )
    pair_probabilities = 0.5 * (
        factor_probabilities[selected[left]]
        + factor_probabilities[selected[right]]
    )
    matching_mass = (
        pair_probabilities * matches.to(pair_probabilities.dtype)
    ).sum(dim=-1)
    usable = matches.any(dim=-1)
    return -(
        matching_mass.clamp_min(1e-8).log()
        * usable.to(matching_mass.dtype)
    ).sum() / usable.sum().clamp_min(1)


def _factor_consistency_loss(
    candidate_factor_logits: Tensor,
    factor_probabilities: Tensor,
    target_residues: Tensor,
    direct_rows: Tensor,
) -> Tensor:
    selected = direct_rows.nonzero(as_tuple=False).flatten()
    if selected.numel() == 0:
        return candidate_factor_logits.sum() * 0.0

    factors = torch.arange(
        MIN_CANDIDATE_FACTOR,
        MAX_CANDIDATE_FACTOR + 1,
        device=target_residues.device,
    )[None, :]
    targets = target_residues[selected, None] % factors
    log_probabilities = F.log_softmax(
        candidate_factor_logits[selected].float(),
        dim=-1,
    )
    target_log_probabilities = log_probabilities.gather(
        -1,
        targets[..., None],
    ).squeeze(-1)
    routing_log_probabilities = factor_probabilities[selected].float().clamp_min(
        1e-8,
    ).log()
    return -torch.logsumexp(
        routing_log_probabilities + target_log_probabilities,
        dim=-1,
    ).mean()


def training_loss(logits: Tensor, labels: Tensor, auxiliary: object) -> Tensor:
    if not isinstance(auxiliary, tuple) or len(auxiliary) != 10:
        raise TypeError(
            "auxiliary output must contain residue and factor supervision state",
        )
    (
        reconstruction_loss,
        code_entropy,
        residue_probabilities,
        factor_probabilities,
        modulus,
        residue,
        time_steps,
        coordinate_logits,
        coordinate_factors,
        candidate_factor_logits,
    ) = auxiliary
    if not isinstance(reconstruction_loss, Tensor) or reconstruction_loss.ndim != 0:
        raise TypeError("reconstruction loss must be a scalar tensor")
    if not isinstance(code_entropy, Tensor) or code_entropy.ndim != 0:
        raise TypeError("code entropy must be a scalar tensor")
    if not isinstance(residue_probabilities, Tensor) or residue_probabilities.ndim != 2:
        raise TypeError("residue probabilities must be a rank-2 tensor")
    if not isinstance(factor_probabilities, Tensor) or factor_probabilities.ndim != 2:
        raise TypeError("factor probabilities must be a rank-2 tensor")
    for name, value in (
        ("modulus", modulus),
        ("residue", residue),
        ("time steps", time_steps),
    ):
        if not isinstance(value, Tensor) or value.ndim != 1:
            raise TypeError(f"{name} must be a rank-1 tensor")
    if not isinstance(coordinate_logits, Tensor) or coordinate_logits.ndim != 3:
        raise TypeError("coordinate logits must be a rank-3 tensor")
    if not isinstance(coordinate_factors, Tensor) or coordinate_factors.ndim != 2:
        raise TypeError("coordinate factors must be a rank-2 tensor")
    if (
        not isinstance(candidate_factor_logits, Tensor)
        or candidate_factor_logits.ndim != 3
    ):
        raise TypeError("candidate factor logits must be a rank-3 tensor")

    row_indices = (
        ROW_MARKER_BASE - logits[:, PAD_TOKEN_ID]
    ).round().to(torch.long)
    batch_size = residue_probabilities.shape[0]
    row_indices = row_indices.clamp(min=0, max=batch_size - 1)
    digit_counts = torch.bincount(row_indices, minlength=batch_size)
    row_starts = torch.cumsum(digit_counts, dim=0) - digit_counts
    positions = (
        torch.arange(labels.numel(), device=labels.device)
        - row_starts[row_indices]
    )
    powers = digit_counts[row_indices] - positions - 1
    place_values = torch.pow(
        torch.full_like(powers, 10),
        powers,
    )
    digit_values = (labels - DIGIT_OFFSET).clamp(
        min=0,
        max=NUM_DIGITS - 1,
    )
    target_residues = torch.zeros(
        batch_size,
        device=labels.device,
        dtype=torch.long,
    )
    target_residues.scatter_add_(
        0,
        row_indices,
        digit_values * place_values,
    )
    valid_rows = (digit_counts > 0) & (target_residues < RESIDUE_CODES)
    residue_log_probabilities = residue_probabilities.float().clamp_min(1e-8).log()
    supervised_log_probabilities = residue_log_probabilities[
        torch.arange(batch_size, device=labels.device),
        target_residues.clamp(max=RESIDUE_CODES - 1),
    ]
    residue_loss = -(
        supervised_log_probabilities * valid_rows.to(supervised_log_probabilities.dtype)
    ).sum() / valid_rows.sum().clamp_min(1)
    factor_collision_loss = _factor_collision_loss(
        factor_probabilities,
        modulus,
        residue,
        time_steps,
        target_residues,
        valid_rows,
    )
    coordinate_rows = valid_rows & (time_steps <= 2)
    selected_coordinate_rows = coordinate_rows.nonzero(
        as_tuple=False,
    ).flatten()
    if selected_coordinate_rows.numel() == 0:
        coordinate_loss = coordinate_logits.sum() * 0.0
    else:
        coordinate_targets = (
            target_residues[:, None]
            % coordinate_factors.clamp_min(1)
        ) % COORDINATE_CLASSES
        coordinate_loss = F.cross_entropy(
            coordinate_logits[selected_coordinate_rows].float().reshape(
                -1,
                COORDINATE_CLASSES,
            ),
            coordinate_targets[selected_coordinate_rows].reshape(-1),
        )
    factor_consistency_loss = _factor_consistency_loss(
        candidate_factor_logits,
        factor_probabilities,
        target_residues,
        coordinate_rows,
    )
    factor_entropy = -(
        factor_probabilities.float()
        * factor_probabilities.float().clamp_min(1e-8).log()
    ).sum(dim=-1).mean() / math.log(NUM_CANDIDATE_FACTORS)
    return (
        F.cross_entropy(logits, labels)
        + RECONSTRUCTION_WEIGHT * reconstruction_loss
        + ENTROPY_WEIGHT * code_entropy
        + RESIDUE_SUPERVISION_WEIGHT * residue_loss
        + factor_collision_loss
        + coordinate_loss
        + FACTOR_CONSISTENCY_WEIGHT * factor_consistency_loss
        + FACTOR_ENTROPY_WEIGHT * factor_entropy
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
    batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=512,
    max_steps=MAX_TRAINING_STEPS,
)
