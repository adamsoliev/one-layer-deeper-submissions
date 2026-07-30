"""Shallow joint-answer bottleneck for One Layer Deeper."""

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

INPUT_WIDTH = 256
JOINT_WIDTH = 1_024
ANSWER_WIDTH = 96
MAX_VALUE = 8_192
MAX_TIME = 64
MAX_OUTPUT_DIGITS = 4
TRAIN_BATCH_SIZE = 512
MAX_TRAINING_STEPS = 2_000
WARMUP_STEPS = 50
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
        return F.rms_norm(hidden, (hidden.shape[-1],), self.weight)


class JointAnswerModel(nn.Module):
    """Map an entire prompt into one narrow code that emits every answer digit."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.modulus_embedding = nn.Embedding(MAX_VALUE, INPUT_WIDTH)
        self.residue_embedding = nn.Embedding(MAX_VALUE, INPUT_WIDTH)
        self.time_embedding = nn.Embedding(MAX_TIME + 1, INPUT_WIDTH)
        self.scalar_projection = nn.Linear(6, INPUT_WIDTH)
        self.modulus_norm = RMSNorm(INPUT_WIDTH)
        self.residue_norm = RMSNorm(INPUT_WIDTH)
        self.time_norm = RMSNorm(INPUT_WIDTH)
        self.scalar_norm = RMSNorm(INPUT_WIDTH)

        feature_width = 8 * INPUT_WIDTH
        self.feature_norm = RMSNorm(feature_width)
        self.joint_left = nn.Linear(feature_width, JOINT_WIDTH)
        self.joint_right = nn.Linear(feature_width, JOINT_WIDTH)
        self.answer_projection = nn.Linear(JOINT_WIDTH, ANSWER_WIDTH)
        self.answer_norm = RMSNorm(ANSWER_WIDTH)
        self.answer_decoder = nn.Linear(
            ANSWER_WIDTH,
            MAX_OUTPUT_DIGITS * spec.vocab_size,
        )
        self.apply(self._initialize)
        nn.init.normal_(self.answer_projection.weight, mean=0.0, std=0.01)

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
    ) -> tuple[Tensor, None]:
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
        time_index = time_steps.clamp(max=MAX_TIME)

        modulus_vector = self.modulus_norm(
            self.modulus_embedding(modulus_index),
        )
        residue_vector = self.residue_norm(
            self.residue_embedding(residue_index),
        )
        time_vector = self.time_norm(self.time_embedding(time_index))

        modulus_float = modulus.to(dtype=torch.float32)
        residue_float = residue.to(dtype=torch.float32)
        time_float = time_steps.to(dtype=torch.float32)
        scalar_features = torch.stack(
            (
                modulus_float / MAX_VALUE,
                residue_float / MAX_VALUE,
                time_float / MAX_TIME,
                torch.log1p(modulus_float) / 10.0,
                torch.log1p(residue_float) / 10.0,
                residue_float / modulus_float.clamp_min(1.0),
            ),
            dim=-1,
        )
        scalar_vector = self.scalar_norm(
            self.scalar_projection(scalar_features),
        )

        features = torch.cat(
            (
                modulus_vector,
                residue_vector,
                time_vector,
                scalar_vector,
                modulus_vector * residue_vector,
                modulus_vector * time_vector,
                residue_vector * time_vector,
                modulus_vector * residue_vector * time_vector,
            ),
            dim=-1,
        )
        features = self.feature_norm(features)
        joint = F.silu(self.joint_left(features)) * self.joint_right(features)
        answer = self.answer_norm(self.answer_projection(joint))
        digit_logits = self.answer_decoder(answer).view(
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
        logits = digit_logits[batch_indices, answer_slots]
        return logits, None

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


def build_model(spec: ModelSpec) -> JointAnswerModel:
    model = JointAnswerModel(spec)
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


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=512,
    max_steps=MAX_TRAINING_STEPS,
)
