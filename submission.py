"""Modulus-specialized residue recurrence for One Layer Deeper."""

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
MEMORY_TOP_K = 8
MAX_OUTPUT_DIGITS = 4
TRAIN_BATCH_SIZE = 512
MAX_TRAINING_STEPS = 2_000
WARMUP_STEPS = 50
MAX_TRAIN_T = 3
MAX_EVAL_T = 8
PAD_TOKEN_ID = 0
X_TOKEN_ID = 3
T_TOKEN_ID = 4
DIGIT_OFFSET = 7
NUM_DIGITS = 10
RECONSTRUCTION_WEIGHT = 0.25
ENTROPY_WEIGHT = 0.05


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
    """Produce a query with a learned hidden-channel program for each N."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_norm = RMSNorm(5 * WIDTH)
        self.left = nn.Linear(5 * WIDTH, TRANSITION_WIDTH)
        self.right = nn.Linear(5 * WIDTH, TRANSITION_WIDTH)
        self.modulus_film = nn.Embedding(
            MAX_VALUE,
            2 * TRANSITION_WIDTH,
        )
        self.query = nn.Linear(TRANSITION_WIDTH, WIDTH)

    def forward(
        self,
        state: Tensor,
        modulus: Tensor,
        memory: Tensor,
        modulus_index: Tensor,
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
        film_scale, film_shift = self.modulus_film(modulus_index).chunk(
            2,
            dim=-1,
        )
        hidden = hidden * (1.0 + film_scale) + film_shift
        return self.query(hidden)


class CanonicalResidueModel(nn.Module):
    """Apply one modulus-specialized transition exactly T times."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.modulus_embedding = nn.Embedding(MAX_VALUE, WIDTH)
        self.residue_embedding = nn.Embedding(MAX_VALUE, WIDTH)
        self.pair_memory = nn.Embedding(
            PAIR_MEMORY_SIZE,
            PAIR_MEMORY_WIDTH,
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
        self.code_log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.answer_decoder = nn.Linear(
            WIDTH,
            MAX_OUTPUT_DIGITS * spec.vocab_size,
        )
        self.apply(self._initialize)
        nn.init.normal_(self.transition.query.weight, mean=0.0, std=0.01)

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
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
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
        initial_state = self.residue_norm(
            self.residue_embedding(residue_index),
        )
        reconstruction_loss = self._reconstruction_loss(
            initial_state,
            input_ids,
            attention_mask,
        )

        state = initial_state
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
        maximum_steps = MAX_TRAIN_T if self.training else MAX_EVAL_T
        for step in range(maximum_steps):
            memory = self._memory_context(
                modulus_index,
                state_indices,
                state_weights,
            )
            query = self.transition(
                state,
                modulus_vector,
                memory,
                modulus_index,
            )
            candidate, entropy, probabilities = self._canonicalize(query)
            candidate_weights, candidate_indices = probabilities.topk(
                MEMORY_TOP_K,
                dim=-1,
            )
            candidate_weights = candidate_weights / candidate_weights.sum(
                dim=-1,
                keepdim=True,
            )
            active = time_steps > step
            state = torch.where(active[:, None], candidate, state)
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
        return logits, (reconstruction_loss, code_entropy)

    def _memory_context(
        self,
        modulus: Tensor,
        residue_indices: Tensor,
        residue_weights: Tensor,
    ) -> Tensor:
        keys = (
            modulus[:, None] * RESIDUE_CODES + residue_indices
        ) % PAIR_MEMORY_SIZE
        memories = self.pair_memory(keys)
        memory = (
            memories * residue_weights.to(memories.dtype)[..., None]
        ).sum(dim=1)
        return self.memory_projection(memory)

    def _canonicalize(
        self,
        query: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        query = F.normalize(self.query_norm(query), dim=-1)
        codes = F.normalize(
            self.residue_norm(
                self.residue_embedding.weight[:RESIDUE_CODES],
            ),
            dim=-1,
        )
        scale = self.code_log_scale.exp().clamp(max=30.0)
        probabilities = F.softmax(scale * (query @ codes.T), dim=-1)
        state = probabilities @ codes
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
            dim=-1,
        )
        return self.residue_norm(state), entropy, probabilities

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
        return digit_logits[batch_indices, answer_slots]

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
    optimizer = torch.optim.AdamW(
        model.parameters(),
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


def training_loss(logits: Tensor, labels: Tensor, auxiliary: object) -> Tensor:
    if not isinstance(auxiliary, tuple) or len(auxiliary) != 2:
        raise TypeError("auxiliary output must contain reconstruction and entropy")
    reconstruction_loss, code_entropy = auxiliary
    if not isinstance(reconstruction_loss, Tensor) or reconstruction_loss.ndim != 0:
        raise TypeError("reconstruction loss must be a scalar tensor")
    if not isinstance(code_entropy, Tensor) or code_entropy.ndim != 0:
        raise TypeError("code entropy must be a scalar tensor")
    return (
        F.cross_entropy(logits, labels)
        + RECONSTRUCTION_WEIGHT * reconstruction_loss
        + ENTROPY_WEIGHT * code_entropy
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
    batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=512,
    max_steps=MAX_TRAINING_STEPS,
)
