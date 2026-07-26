"""Explicit residue-state Transformer for One Layer Deeper."""

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

WIDTH = 192
NUM_HEADS = 6
NUM_LAYERS = 3
MLP_WIDTH = 768
TRAIN_RECURRENT_STEPS = 3
MAX_RECURRENT_STEPS = 64
PAD_TOKEN_ID = 0
X_TOKEN_ID = 3
T_TOKEN_ID = 4
DIGIT_OFFSET = 7
NUM_DIGITS = 10
LEADING_STATE_WEIGHT = 0.1


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class TransformerBlock(nn.Module):
    """Pre-LN bidirectional Transformer block."""

    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.attention_out = nn.Linear(WIDTH, WIDTH)
        self.mlp_norm = nn.LayerNorm(WIDTH)
        self.mlp_up = nn.Linear(WIDTH, MLP_WIDTH)
        self.mlp_down = nn.Linear(MLP_WIDTH, WIDTH)

    def forward(self, hidden: Tensor) -> Tensor:
        residual = hidden
        hidden = self.attention_norm(hidden)
        batch, length, _ = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        query = query.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        key = key.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        value = value.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        hidden = F.scaled_dot_product_attention(query, key, value)
        hidden = hidden.transpose(1, 2).contiguous().view(batch, length, WIDTH)
        hidden = residual + self.attention_out(hidden)
        hidden = hidden + self.mlp_down(
            F.gelu(self.mlp_up(self.mlp_norm(hidden)), approximate="tanh")
        )
        return hidden


class ExplicitResidueTransformer(nn.Module):
    """Apply one learned digit-state transition exactly T times."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.num_slots = max(1, (spec.max_seq_len - 3) // 2)
        self.digit_embedding = nn.Embedding(NUM_DIGITS, WIDTH)
        self.slot_embedding = nn.Embedding(self.num_slots, WIDTH)
        self.type_embedding = nn.Embedding(2, WIDTH)
        self.blocks = nn.ModuleList(TransformerBlock() for _ in range(NUM_LAYERS))
        self.final_norm = nn.LayerNorm(WIDTH)
        self.digit_output = nn.Linear(WIDTH, NUM_DIGITS)
        self.apply(self._initialize)

        residual_std = 0.02 / math.sqrt(2 * NUM_LAYERS)
        for block in self.blocks:
            nn.init.normal_(block.attention_out.weight, std=residual_std)
            nn.init.normal_(block.mlp_down.weight, std=residual_std)

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
    ) -> tuple[Tensor, Tensor]:
        if attention_mask is None:
            attention_mask = input_ids != PAD_TOKEN_ID
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)

        modulus_digits, residue_digits, modulus_widths = self._extract_state_digits(
            input_ids,
            attention_mask,
        )
        modulus_state = self.digit_embedding(modulus_digits)
        residue_state = self.digit_embedding(residue_digits)
        time_steps = self._decode_time_steps(input_ids, attention_mask)

        batch = input_ids.shape[0]
        final_digit_logits = residue_state.new_zeros(
            batch,
            self.num_slots,
            NUM_DIGITS,
        )
        recurrent_steps = (
            TRAIN_RECURRENT_STEPS if self.training else MAX_RECURRENT_STEPS
        )
        for step in range(recurrent_steps):
            candidate_state, candidate_logits = self._transition(
                modulus_state,
                residue_state,
            )
            active = (time_steps > step).view(batch, 1, 1)
            residue_state = torch.where(active, candidate_state, residue_state)
            final_digit_logits = torch.where(
                active,
                candidate_logits,
                final_digit_logits,
            )

        output_digit_logits = self._align_output_logits(
            final_digit_logits,
            attention_mask,
        )
        prefix = output_digit_logits.new_full(
            (
                output_digit_logits.shape[0],
                output_digit_logits.shape[1],
                DIGIT_OFFSET,
            ),
            -30.0,
        )
        suffix_width = self.config.vocab_size - DIGIT_OFFSET - NUM_DIGITS
        suffix = output_digit_logits.new_full(
            (
                output_digit_logits.shape[0],
                output_digit_logits.shape[1],
                suffix_width,
            ),
            -30.0,
        )
        logits = torch.cat((prefix, output_digit_logits, suffix), dim=-1)
        slot_positions = torch.arange(
            self.num_slots,
            device=input_ids.device,
        )[None, :]
        guaranteed_zero = slot_positions < (
            self.num_slots - modulus_widths
        )[:, None]
        zero_targets = torch.zeros(
            batch,
            self.num_slots,
            device=input_ids.device,
            dtype=torch.long,
        )
        zero_loss = F.cross_entropy(
            final_digit_logits.reshape(batch * self.num_slots, NUM_DIGITS),
            zero_targets.reshape(batch * self.num_slots),
            reduction="none",
        ).reshape(batch, self.num_slots)
        zero_weights = guaranteed_zero.to(dtype=zero_loss.dtype)
        leading_zero_loss = (zero_loss * zero_weights).sum() / zero_weights.sum().clamp(
            min=1
        )
        return logits, leading_zero_loss

    def _transition(
        self,
        modulus_state: Tensor,
        residue_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        slots = torch.arange(self.num_slots, device=residue_state.device)
        positions = self.slot_embedding(slots)[None, :, :]
        modulus_type = self.type_embedding.weight[0][None, None, :]
        residue_type = self.type_embedding.weight[1][None, None, :]
        hidden = torch.cat(
            (
                modulus_state + positions + modulus_type,
                residue_state + positions + residue_type,
            ),
            dim=1,
        )
        for block in self.blocks:
            hidden = block(hidden)

        residue_hidden = self.final_norm(hidden[:, self.num_slots :, :])
        digit_logits = self.digit_output(residue_hidden)
        probabilities = F.softmax(digit_logits.float(), dim=-1).to(
            dtype=digit_logits.dtype
        )
        hard_digits = F.one_hot(
            probabilities.argmax(dim=-1),
            num_classes=NUM_DIGITS,
        ).to(dtype=probabilities.dtype)
        if self.training:
            recurrent_digits = hard_digits + probabilities - probabilities.detach()
        else:
            recurrent_digits = hard_digits
        next_state = recurrent_digits @ self.digit_embedding.weight
        return next_state, digit_logits

    def _extract_state_digits(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)[None, :]
        x_position = torch.argmax((input_ids == X_TOKEN_ID).long(), dim=1)
        t_position = torch.argmax((input_ids == T_TOKEN_ID).long(), dim=1)
        is_digit = (input_ids >= DIGIT_OFFSET) & attention_mask
        modulus_mask = is_digit & (positions < x_position[:, None])
        residue_mask = (
            is_digit
            & (positions > x_position[:, None])
            & (positions < t_position[:, None])
        )
        digit_ids = (input_ids - DIGIT_OFFSET).clamp(
            min=0,
            max=NUM_DIGITS - 1,
        )
        modulus_digits = self._right_align_digits(digit_ids, modulus_mask)
        residue_digits = self._right_align_digits(digit_ids, residue_mask)
        modulus_widths = modulus_mask.long().sum(dim=1)
        return modulus_digits, residue_digits, modulus_widths

    def _right_align_digits(self, digit_ids: Tensor, mask: Tensor) -> Tensor:
        batch, length = digit_ids.shape
        counts = mask.long().sum(dim=1)
        ranks = mask.long().cumsum(dim=1) - 1
        destinations = self.num_slots - counts[:, None] + ranks
        slots = torch.zeros(
            batch,
            self.num_slots,
            device=digit_ids.device,
            dtype=torch.long,
        )
        slot_positions = torch.arange(
            self.num_slots,
            device=digit_ids.device,
        )[None, :]
        for position in range(length):
            write_mask = mask[:, position, None] & (
                slot_positions == destinations[:, position, None]
            )
            slots = torch.where(
                write_mask,
                digit_ids[:, position, None],
                slots,
            )
        return slots

    def _align_output_logits(
        self,
        digit_logits: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        batch, length = attention_mask.shape
        positions = torch.arange(length, device=attention_mask.device)[None, :]
        valid_lengths = attention_mask.long().sum(dim=1, keepdim=True)
        slots_from_right = (valid_lengths - positions - 1).clamp(
            min=0,
            max=self.num_slots - 1,
        )
        residue_indices = self.num_slots - slots_from_right - 1
        return digit_logits.gather(
            1,
            residue_indices.unsqueeze(-1).expand(batch, length, NUM_DIGITS),
        )

    @staticmethod
    def _decode_time_steps(
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        time_steps = torch.zeros(
            input_ids.shape[0],
            device=input_ids.device,
            dtype=torch.long,
        )
        after_marker = torch.zeros_like(time_steps, dtype=torch.bool)
        for position in range(input_ids.shape[1]):
            token = input_ids[:, position]
            after_marker = after_marker | (token == T_TOKEN_ID)
            is_digit = after_marker & (token >= DIGIT_OFFSET)
            is_digit = is_digit & attention_mask[:, position]
            digit = (token - DIGIT_OFFSET).clamp(min=0, max=NUM_DIGITS - 1)
            time_steps = torch.where(
                is_digit,
                time_steps * 10 + digit,
                time_steps,
            )
        return time_steps.clamp(min=1, max=MAX_RECURRENT_STEPS)


def build_model(spec: ModelSpec) -> ExplicitResidueTransformer:
    model = ExplicitResidueTransformer(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(
    model: nn.Module,
    spec: OptimizerSpec,
) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(optimizer)


def training_loss(logits: Tensor, labels: Tensor, auxiliary: object) -> Tensor:
    main_loss = F.cross_entropy(logits, labels)
    if not isinstance(auxiliary, Tensor) or auxiliary.ndim != 0:
        raise TypeError("leading-zero state loss must be a scalar tensor")
    return main_loss + LEADING_STATE_WEIGHT * auxiliary


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
)
