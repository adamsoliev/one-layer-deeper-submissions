"""GPT-2 small-sized submission for One Layer Deeper."""

from __future__ import annotations

import math

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


WIDTH = 768
NUM_HEADS = 12
NUM_LAYERS = 12
MLP_WIDTH = 3072


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class GPT2Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.attention_out = nn.Linear(WIDTH, WIDTH)
        self.mlp_norm = nn.LayerNorm(WIDTH)
        self.mlp_up = nn.Linear(WIDTH, MLP_WIDTH)
        self.mlp_down = nn.Linear(MLP_WIDTH, WIDTH)

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        residual = hidden
        hidden = self.attention_norm(hidden)
        batch, length, _ = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)

        query = query.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        key = key.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        value = value.view(batch, length, NUM_HEADS, -1).transpose(1, 2)

        allowed = None
        if attention_mask is not None:
            allowed = attention_mask.to(device=hidden.device, dtype=torch.bool)
            if allowed.shape == (batch, length):
                allowed = allowed[:, None, None, :]
            elif allowed.shape == (batch, length, length):
                allowed = allowed[:, None, :, :]
            else:
                raise ValueError("invalid attention_mask shape")

        hidden = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
        )
        hidden = hidden.transpose(1, 2).contiguous().view(batch, length, WIDTH)
        hidden = residual + self.attention_out(hidden)
        hidden = hidden + self.mlp_down(
            F.gelu(self.mlp_up(self.mlp_norm(hidden)), approximate="tanh")
        )
        return hidden


class GPT2Small(nn.Module):
    """GPT-2 small dimensions with evaluator-provided bidirectional attention."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, WIDTH)
        self.position_embedding = nn.Embedding(spec.max_seq_len, WIDTH)
        self.blocks = nn.ModuleList(GPT2Block() for _ in range(NUM_LAYERS))
        self.final_norm = nn.LayerNorm(WIDTH)
        self.output = nn.Linear(WIDTH, spec.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
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
    ) -> tuple[Tensor, None]:
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        logits = self.output(self.final_norm(hidden))
        return logits, None


def build_model(spec: ModelSpec) -> GPT2Small:
    model = GPT2Small(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(
    model: nn.Module,
    spec: OptimizerSpec,
) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=6e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(optimizer)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
)
