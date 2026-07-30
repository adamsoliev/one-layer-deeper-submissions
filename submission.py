"""Multiplicative numeric recurrence for One Layer Deeper."""

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
FOURIER_FEATURES = 64
TRANSITION_WIDTH = 768
DECODER_WIDTH = 768
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
VALUE_SCALE = 2_048.0
RECONSTRUCTION_WEIGHT = 0.25


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


class ScalarEncoder(nn.Module):
    """Embed one scalar with learned Fourier coordinates."""

    def __init__(self) -> None:
        super().__init__()
        self.frequency = nn.Parameter(torch.empty(FOURIER_FEATURES))
        self.phase = nn.Parameter(torch.empty(FOURIER_FEATURES))
        self.input = nn.Linear(2 * FOURIER_FEATURES + 2, WIDTH)
        self.output = nn.Linear(WIDTH, WIDTH)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.frequency, mean=0.0, std=6.0)
        nn.init.uniform_(self.phase, -math.pi, math.pi)
        nn.init.normal_(self.input.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.input.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: Tensor, raw_value: Tensor) -> Tensor:
        angles = value[:, None] * self.frequency[None, :] + self.phase[None, :]
        features = torch.cat(
            (
                value[:, None],
                torch.log1p(raw_value)[:, None] / 8.0,
                torch.sin(angles),
                torch.cos(angles),
            ),
            dim=-1,
        )
        return self.output(F.silu(self.input(features)))


class MultiplicativeTransition(nn.Module):
    """A tied gated update whose learned branches interact bilinearly."""

    def __init__(self) -> None:
        super().__init__()
        self.state_norm = RMSNorm(WIDTH)
        self.context_norm = RMSNorm(WIDTH)
        self.left = nn.Linear(2 * WIDTH, TRANSITION_WIDTH)
        self.right = nn.Linear(2 * WIDTH, TRANSITION_WIDTH)
        self.down = nn.Linear(TRANSITION_WIDTH, WIDTH)
        self.gate = nn.Linear(2 * WIDTH, WIDTH)

    def forward(self, state: Tensor, context: Tensor) -> Tensor:
        joined = torch.cat(
            (self.state_norm(state), self.context_norm(context)),
            dim=-1,
        )
        product = F.silu(self.left(joined)) * self.right(joined)
        update = self.down(product)
        gate = torch.sigmoid(self.gate(joined))
        return state + gate * update


class NumericRecurrentModel(nn.Module):
    """Parse decimal fields, then apply one learned transition exactly T times."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.modulus_encoder = ScalarEncoder()
        self.residue_encoder = ScalarEncoder()
        self.context = nn.Linear(WIDTH, WIDTH)
        self.transition = MultiplicativeTransition()
        self.answer_embedding = nn.Embedding(spec.max_seq_len, WIDTH)
        self.decoder_norm = RMSNorm(WIDTH)
        self.decoder_up = nn.Linear(WIDTH, DECODER_WIDTH)
        self.decoder_down = nn.Linear(DECODER_WIDTH, spec.vocab_size)
        self.apply(self._initialize)
        self.modulus_encoder.reset_parameters()
        self.residue_encoder.reset_parameters()

        nn.init.normal_(self.transition.down.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.transition.down.bias)
        nn.init.zeros_(self.transition.gate.weight)
        nn.init.constant_(self.transition.gate.bias, -1.0)

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

        modulus, residue, time_steps = self._parse_fields(
            input_ids,
            attention_mask,
        )
        modulus_vector = self.modulus_encoder(
            modulus / VALUE_SCALE,
            modulus,
        )
        residue_vector = self.residue_encoder(
            residue / VALUE_SCALE,
            residue,
        )
        context = self.context(modulus_vector)
        reconstruction_loss = self._reconstruction_loss(
            residue_vector,
            input_ids,
            attention_mask,
        )
        state = residue_vector

        maximum_steps = MAX_TRAIN_T if self.training else MAX_EVAL_T
        for step in range(maximum_steps):
            candidate = self.transition(state, context)
            active = (time_steps > step).unsqueeze(-1)
            state = torch.where(active, candidate, state)

        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)[None, :]
        valid_lengths = attention_mask.long().sum(dim=1, keepdim=True)
        answer_slots = (valid_lengths - positions - 1).clamp(
            min=0,
            max=self.answer_embedding.num_embeddings - 1,
        )
        logits = self._decode(state, answer_slots)
        return logits, reconstruction_loss

    def _decode(self, state: Tensor, answer_slots: Tensor) -> Tensor:
        decoded = state[:, None, :] + self.answer_embedding(answer_slots)
        return self.decoder_down(F.silu(self.decoder_up(self.decoder_norm(decoded))))

    def _reconstruction_loss(
        self,
        residue_vector: Tensor,
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
        )
        reconstruction_logits = self._decode(
            residue_vector,
            significance.clamp_min(0),
        )
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
            dtype=torch.float32,
        )
        residue = torch.zeros_like(modulus)
        time_steps = torch.zeros(
            input_ids.shape[0],
            device=input_ids.device,
            dtype=torch.long,
        )
        field = torch.zeros_like(time_steps)
        for position in range(input_ids.shape[1]):
            token = input_ids[:, position]
            field = torch.where(token == X_TOKEN_ID, 1, field)
            field = torch.where(token == T_TOKEN_ID, 2, field)
            is_digit = (token >= DIGIT_OFFSET) & attention_mask[:, position]
            digit_long = (token - DIGIT_OFFSET).clamp(
                min=0,
                max=NUM_DIGITS - 1,
            )
            digit = digit_long.to(dtype=modulus.dtype)
            modulus = torch.where(
                is_digit & (field == 0),
                modulus * 10.0 + digit,
                modulus,
            )
            residue = torch.where(
                is_digit & (field == 1),
                residue * 10.0 + digit,
                residue,
            )
            time_steps = torch.where(
                is_digit & (field == 2),
                time_steps * 10 + digit_long,
                time_steps,
            )
        return (
            modulus,
            residue,
            time_steps.clamp(min=1, max=MAX_EVAL_T),
        )


def build_model(spec: ModelSpec) -> NumericRecurrentModel:
    model = NumericRecurrentModel(spec)
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
        weight_decay=0.05,
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
    if not isinstance(auxiliary, Tensor) or auxiliary.ndim != 0:
        raise TypeError("reconstruction loss must be a scalar tensor")
    return F.cross_entropy(logits, labels) + RECONSTRUCTION_WEIGHT * auxiliary


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
    batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=512,
    max_steps=MAX_TRAINING_STEPS,
)
