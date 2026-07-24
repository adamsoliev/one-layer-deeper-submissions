"""Minimal valid submission for testing the One Layer Deeper pipeline."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
)


WIDTH = 64
NUM_HEADS = 4


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class TinyTransformer(nn.Module):
    """One bidirectional Transformer block followed by token predictions."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, WIDTH)
        self.position_embedding = nn.Embedding(spec.max_seq_len, WIDTH)
        self.block = nn.TransformerEncoderLayer(
            d_model=WIDTH,
            nhead=NUM_HEADS,
            dim_feedforward=2 * WIDTH,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.final_norm = nn.LayerNorm(WIDTH)
        self.output = nn.Linear(WIDTH, spec.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)

        padding_mask = None
        full_mask = None
        if attention_mask is not None:
            allowed = attention_mask.to(device=input_ids.device, dtype=torch.bool)
            if allowed.ndim == 2:
                padding_mask = ~allowed
            elif allowed.ndim == 3:
                full_mask = (~allowed).repeat_interleave(NUM_HEADS, dim=0)
            else:
                raise ValueError("attention_mask must have two or three dimensions")

        hidden = self.block(
            hidden,
            src_mask=full_mask,
            src_key_padding_mask=padding_mask,
        )
        logits = self.output(self.final_norm(hidden))
        return logits, None


def build_model(spec: ModelSpec) -> TinyTransformer:
    model = TinyTransformer(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(
    model: nn.Module,
    spec: OptimizerSpec,
) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(optimizer)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
)
