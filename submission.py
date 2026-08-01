"""Range-independent digit-compositional refinement model."""

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
HIDDEN_WIDTH = 768
MEMORY_WIDTH = 64
MEMORY_SIZE = 65_537
MAX_DECIMAL_DIGITS = 8
VALUE_BITS = 24
TIME_BITS = 7
REFINEMENT_STEPS = 4
TRAIN_BATCH_SIZE = 512
EVAL_BATCH_SIZE = 4_096
MAX_TRAINING_STEPS = 20_000
WARMUP_STEPS = 100
PAD_TOKEN_ID = 0
X_TOKEN_ID = 3
T_TOKEN_ID = 4
DIGIT_OFFSET = 7
NUM_DIGITS = 10
RECONSTRUCTION_WEIGHT = 0.1


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


class RefinementBlock(nn.Module):
    """Apply one tied, bounded update to the learned numeric state."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_norm = RMSNorm(4 * WIDTH)
        self.update_left = nn.Linear(4 * WIDTH, HIDDEN_WIDTH)
        self.update_right = nn.Linear(4 * WIDTH, HIDDEN_WIDTH)
        self.update_out = nn.Linear(HIDDEN_WIDTH, WIDTH)
        self.gate = nn.Linear(4 * WIDTH, WIDTH)

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
        hidden = F.silu(self.update_left(features)) * torch.tanh(
            self.update_right(features),
        )
        update = self.update_out(hidden)
        gate = torch.sigmoid(self.gate(features))
        return state + 0.25 * gate * update


class DigitCompositionalModel(nn.Module):
    """Encode bounded-length integers without a vocabulary entry per value."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.digit_embedding = nn.Embedding(NUM_DIGITS, WIDTH)
        self.place_embedding = nn.Embedding(MAX_DECIMAL_DIGITS, WIDTH)
        self.field_embedding = nn.Embedding(3, WIDTH)
        self.modulus_memory = nn.Embedding(MEMORY_SIZE, MEMORY_WIDTH)
        self.prompt_memory = nn.Embedding(MEMORY_SIZE, MEMORY_WIDTH)
        self.memory_projection = nn.Linear(2 * MEMORY_WIDTH, WIDTH, bias=False)
        self.numeric_projection = nn.Linear(
            2 * VALUE_BITS + TIME_BITS + 3,
            WIDTH,
            bias=False,
        )
        self.value_norm = RMSNorm(WIDTH)
        self.context_norm = RMSNorm(WIDTH)
        self.state_norm = RMSNorm(WIDTH)
        self.context_up = nn.Linear(WIDTH, 2 * WIDTH)
        self.context_down = nn.Linear(2 * WIDTH, WIDTH)
        self.initial_projection = nn.Linear(4 * WIDTH, WIDTH)
        self.step_embedding = nn.Embedding(REFINEMENT_STEPS, WIDTH)
        self.refinement = RefinementBlock()
        self.answer_head = nn.Linear(
            WIDTH,
            MAX_DECIMAL_DIGITS * spec.vocab_size,
        )
        self.reconstruction_head = nn.Linear(
            WIDTH,
            MAX_DECIMAL_DIGITS * NUM_DIGITS,
        )
        self.register_buffer(
            "decimal_powers",
            10 ** torch.arange(MAX_DECIMAL_DIGITS, dtype=torch.long),
        )
        self.register_buffer(
            "value_bit_shifts",
            torch.arange(VALUE_BITS, dtype=torch.long),
        )
        self.register_buffer(
            "time_bit_shifts",
            torch.arange(TIME_BITS, dtype=torch.long),
        )
        self.apply(self._initialize)
        nn.init.normal_(self.refinement.update_out.weight, std=0.005)
        nn.init.zeros_(self.refinement.update_out.bias)

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
    ) -> tuple[Tensor, tuple[Tensor]]:
        if attention_mask is None:
            attention_mask = input_ids != PAD_TOKEN_ID
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)

        modulus, residue, time_steps = self._parse_fields(
            input_ids,
            attention_mask,
        )
        modulus_vector, modulus_digits, modulus_mask = self._encode_value(
            modulus,
            0,
        )
        residue_vector, residue_digits, residue_mask = self._encode_value(
            residue,
            1,
        )
        time_vector, time_digits, time_mask = self._encode_value(
            time_steps,
            2,
        )

        numeric_features = self._numeric_features(
            modulus,
            residue,
            time_steps,
            self.numeric_projection.weight.dtype,
        )
        memory = self._memory_context(modulus, residue, time_steps)
        context = (
            modulus_vector
            + residue_vector
            + time_vector
            + self.numeric_projection(numeric_features)
            + memory
        )
        context = self.context_norm(
            context + self.context_down(F.silu(self.context_up(context))),
        )
        state = self.state_norm(
            residue_vector
            + self.initial_projection(
                torch.cat(
                    (
                        residue_vector,
                        modulus_vector,
                        time_vector,
                        memory,
                    ),
                    dim=-1,
                ),
            ),
        )
        for step in range(REFINEMENT_STEPS):
            step_context = self.context_norm(
                context + self.step_embedding.weight[step],
            )
            state = self.state_norm(
                self.refinement(state, step_context),
            )

        logits = self._decode_sequence(state, input_ids, attention_mask)
        reconstruction_loss = (
            self._reconstruction_loss(
                modulus_vector,
                modulus_digits,
                modulus_mask,
            )
            + self._reconstruction_loss(
                residue_vector,
                residue_digits,
                residue_mask,
            )
            + self._reconstruction_loss(
                time_vector,
                time_digits,
                time_mask,
            )
        ) / 3.0
        return logits, (reconstruction_loss,)

    def _encode_value(
        self,
        value: Tensor,
        field: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        digits = (value[:, None] // self.decimal_powers) % NUM_DIGITS
        mask = self.decimal_powers[None, :] <= value[:, None]
        mask[:, 0] = True
        places = self.place_embedding.weight[None, :, :]
        field_vector = self.field_embedding.weight[field][None, None, :]
        embedded = self.digit_embedding(digits) + places + field_vector
        weights = mask.to(embedded.dtype)[..., None]
        pooled = (embedded * weights).sum(dim=1) / weights.sum(
            dim=1,
        ).clamp_min(1.0)
        return self.value_norm(pooled), digits, mask

    def _numeric_features(
        self,
        modulus: Tensor,
        residue: Tensor,
        time_steps: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        modulus_bits = ((modulus[:, None] >> self.value_bit_shifts) & 1).to(dtype)
        residue_bits = ((residue[:, None] >> self.value_bit_shifts) & 1).to(dtype)
        time_bits = ((time_steps[:, None] >> self.time_bit_shifts) & 1).to(dtype)
        magnitudes = torch.stack(
            (
                torch.log2(modulus.float().clamp_min(1.0)) / VALUE_BITS,
                torch.log2(residue.float() + 1.0) / VALUE_BITS,
                torch.log2(time_steps.float().clamp_min(1.0)) / TIME_BITS,
            ),
            dim=-1,
        ).to(dtype)
        return torch.cat(
            (modulus_bits, residue_bits, time_bits, magnitudes),
            dim=-1,
        )

    def _memory_context(
        self,
        modulus: Tensor,
        residue: Tensor,
        time_steps: Tensor,
    ) -> Tensor:
        modulus_keys = modulus.remainder(MEMORY_SIZE)
        prompt_keys = (
            modulus * 1_000_003 + residue * 97_409 + time_steps * 65_537
        ).remainder(MEMORY_SIZE)
        return self.memory_projection(
            torch.cat(
                (
                    self.modulus_memory(modulus_keys),
                    self.prompt_memory(prompt_keys),
                ),
                dim=-1,
            ),
        )

    def _decode_sequence(
        self,
        state: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        digit_logits = self.answer_head(state).view(
            input_ids.shape[0],
            MAX_DECIMAL_DIGITS,
            self.config.vocab_size,
        )
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        valid_lengths = attention_mask.long().sum(dim=1, keepdim=True)
        answer_slots = (valid_lengths - positions - 1).clamp(
            min=0,
            max=MAX_DECIMAL_DIGITS - 1,
        )
        batch_indices = torch.arange(
            input_ids.shape[0],
            device=input_ids.device,
        )[:, None]
        return digit_logits[batch_indices, answer_slots]

    def _reconstruction_loss(
        self,
        vector: Tensor,
        digits: Tensor,
        mask: Tensor,
    ) -> Tensor:
        logits = self.reconstruction_head(vector).view(
            vector.shape[0],
            MAX_DECIMAL_DIGITS,
            NUM_DIGITS,
        )
        return F.cross_entropy(
            logits[mask].float(),
            digits[mask],
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
        return modulus, residue, time_steps.clamp_min(1)


def build_model(spec: ModelSpec) -> DigitCompositionalModel:
    model = DigitCompositionalModel(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(
    model: nn.Module,
    spec: OptimizerSpec,
) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-3,
        betas=(0.9, 0.95),
        weight_decay=0.01,
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


def training_loss(
    logits: Tensor,
    labels: Tensor,
    auxiliary: object,
) -> Tensor:
    if not isinstance(auxiliary, tuple) or len(auxiliary) != 1:
        raise TypeError("auxiliary output must contain reconstruction loss")
    reconstruction_loss = auxiliary[0]
    if not isinstance(reconstruction_loss, Tensor) or reconstruction_loss.ndim != 0:
        raise TypeError("reconstruction loss must be a scalar tensor")
    return (
        F.cross_entropy(logits.float(), labels)
        + RECONSTRUCTION_WEIGHT * reconstruction_loss
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
    batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    max_steps=MAX_TRAINING_STEPS,
)
