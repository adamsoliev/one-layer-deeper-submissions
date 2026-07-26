"""Hierarchical T-step recurrent model for One Layer Deeper."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
)


WIDTH = 128
MLP_WIDTH = 512
NUM_FIELDS = 3
MAX_RECURRENT_STEPS = 64
PAD_TOKEN_ID = 0
X_TOKEN_ID = 3
T_TOKEN_ID = 4
DIGIT_OFFSET = 7
NUM_DIGITS = 10


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class HierarchicalRecurrentModel(nn.Module):
    """Encode N, x, and T separately, then reuse one transition T times."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.digit_embedding = nn.Embedding(NUM_DIGITS, WIDTH)
        self.field_initial = nn.Parameter(torch.empty(NUM_FIELDS, WIDTH))
        self.field_encoder = nn.GRUCell(WIDTH, WIDTH)
        self.field_norm = nn.LayerNorm(WIDTH)

        self.state_initial = nn.Linear(2 * WIDTH, WIDTH)
        self.transition = nn.GRUCell(WIDTH, WIDTH)
        self.state_norm = nn.LayerNorm(WIDTH)

        self.context_projection = nn.Linear(2 * WIDTH, WIDTH)
        self.answer_slot_embedding = nn.Embedding(spec.max_seq_len, WIDTH)
        self.decoder_norm = nn.LayerNorm(WIDTH)
        self.decoder_up = nn.Linear(WIDTH, MLP_WIDTH)
        self.decoder_down = nn.Linear(MLP_WIDTH, spec.vocab_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.field_initial, mean=0.0, std=0.02)
        nn.init.normal_(self.digit_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.answer_slot_embedding.weight, mean=0.0, std=0.02)
        for module in (self.state_initial, self.context_projection, self.decoder_up):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        nn.init.normal_(self.decoder_down.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.decoder_down.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        if attention_mask is None:
            attention_mask = input_ids != PAD_TOKEN_ID
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)

        n_vector, x_vector, t_vector = self._encode_fields(
            input_ids,
            attention_mask,
        )
        state = torch.tanh(
            self.state_initial(torch.cat((x_vector, n_vector), dim=-1))
        )
        time_steps = self._decode_time_steps(input_ids, attention_mask)

        recurrent_steps = 3 if self.training else MAX_RECURRENT_STEPS
        for step in range(recurrent_steps):
            candidate = self.transition(n_vector, state)
            active = (time_steps > step).unsqueeze(-1)
            state = torch.where(active, candidate, state)
        state = self.state_norm(state)

        context = self.context_projection(torch.cat((n_vector, t_vector), dim=-1))
        answer_slots = self._answer_slots(attention_mask)
        hidden = (
            state[:, None, :]
            + context[:, None, :]
            + self.answer_slot_embedding(answer_slots)
        )
        hidden = self.decoder_norm(hidden)
        logits = self.decoder_down(F.gelu(self.decoder_up(hidden), approximate="tanh"))
        return logits, None

    def _encode_fields(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)[None, :]
        x_position = torch.argmax((input_ids == X_TOKEN_ID).long(), dim=1)
        t_position = torch.argmax((input_ids == T_TOKEN_ID).long(), dim=1)

        is_digit = (input_ids >= DIGIT_OFFSET) & attention_mask
        field_masks = torch.stack(
            (
                is_digit & (positions < x_position[:, None]),
                is_digit
                & (positions > x_position[:, None])
                & (positions < t_position[:, None]),
                is_digit & (positions > t_position[:, None]),
            ),
            dim=1,
        )

        states = self.field_initial[None, :, :].expand(batch, -1, -1)
        for position in range(length):
            digit_ids = (input_ids[:, position] - DIGIT_OFFSET).clamp(
                min=0,
                max=NUM_DIGITS - 1,
            )
            embedded = self.digit_embedding(digit_ids)
            expanded = embedded[:, None, :].expand(-1, NUM_FIELDS, -1)
            candidate = self.field_encoder(
                expanded.reshape(batch * NUM_FIELDS, WIDTH),
                states.reshape(batch * NUM_FIELDS, WIDTH),
            ).reshape(batch, NUM_FIELDS, WIDTH)
            active = field_masks[:, :, position].unsqueeze(-1)
            states = torch.where(active, candidate, states)

        states = self.field_norm(states)
        return states[:, 0, :], states[:, 1, :], states[:, 2, :]

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

    def _answer_slots(self, attention_mask: Tensor) -> Tensor:
        length = attention_mask.shape[1]
        positions = torch.arange(length, device=attention_mask.device)[None, :]
        valid_lengths = attention_mask.long().sum(dim=1, keepdim=True)
        return (valid_lengths - positions - 1).clamp(
            min=0,
            max=self.config.max_seq_len - 1,
        )


def build_model(spec: ModelSpec) -> HierarchicalRecurrentModel:
    model = HierarchicalRecurrentModel(spec)
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


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
)
