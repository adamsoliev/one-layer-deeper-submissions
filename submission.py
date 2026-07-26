"""Soft Thoughtbubbles with length-conditioned recurrent depth."""

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


WIDTH = 128
NUM_HEADS = 4
NUM_LAYERS = 3
MLP_WIDTH = 512
DROPOUT = 0.1
WARMUP_STEPS = 10
SCHEDULE_STEPS = 80
T_TOKEN_ID = 4
DIGIT_OFFSET = 7
MAX_RECURRENT_STEPS = 64
NUM_BUBBLES = 2
DEPTH_FEATURES = 6


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class TransformerBlock(nn.Module):
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
        hidden = residual + F.dropout(
            self.attention_out(hidden),
            p=DROPOUT,
            training=self.training,
        )
        hidden = hidden + F.dropout(
            self.mlp_down(
                F.gelu(self.mlp_up(self.mlp_norm(hidden)), approximate="tanh")
            ),
            p=DROPOUT,
            training=self.training,
        )
        return hidden


class T2MLRFusion(nn.Module):
    """Paper-faithful gated fusion of the current stream and recurrent cache."""

    def __init__(self) -> None:
        super().__init__()
        self.current_gate = nn.Linear(2 * WIDTH, WIDTH)
        self.recurrent_gate = nn.Linear(2 * WIDTH, WIDTH)
        self.recurrent_projection = nn.Linear(WIDTH, WIDTH)
        self.gamma_current = nn.Parameter(torch.zeros(()))
        self.gamma_recurrent = nn.Parameter(torch.zeros(()))

    def forward(self, current: Tensor, recurrent_cache: Tensor) -> Tensor:
        joined = torch.cat((current, recurrent_cache), dim=-1)
        current_update = (
            torch.tanh(self.gamma_current)
            * torch.sigmoid(self.current_gate(joined))
            * current
        )
        recurrent_update = (
            torch.tanh(self.gamma_recurrent)
            * torch.sigmoid(self.recurrent_gate(joined))
            * self.recurrent_projection(recurrent_cache)
        )
        return current + current_update + recurrent_update


class DepthEncoding(nn.Module):
    """Project extrapolatable recurrence-progress features into model space."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(DEPTH_FEATURES, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, WIDTH),
        )

    def forward(self, time_steps: Tensor, step: int) -> Tensor:
        total = time_steps.to(dtype=torch.float32).clamp_min(1.0)
        current = torch.full_like(total, step + 1, dtype=torch.float32)
        progress = (current / total).clamp(max=1.0)
        remaining = ((total - current).clamp_min(0.0) / total).clamp(max=1.0)
        absolute = current / MAX_RECURRENT_STEPS
        total_scale = torch.log2(total + 1.0) / math.log2(MAX_RECURRENT_STEPS + 1.0)
        features = torch.stack(
            (
                progress,
                remaining,
                absolute,
                total_scale,
                torch.sin(math.pi * progress),
                torch.cos(math.pi * progress),
            ),
            dim=-1,
        )
        return self.projection(features).to(dtype=self.projection[0].weight.dtype)


class ThoughtbubbleTransformer(nn.Module):
    """Run a scored pair of latent streams through exactly T tied steps."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, WIDTH)
        self.position_embedding = nn.Embedding(spec.max_seq_len, WIDTH)
        self.blocks = nn.ModuleList(TransformerBlock() for _ in range(NUM_LAYERS))
        self.fusion = T2MLRFusion()
        self.depth_encoding = DepthEncoding()
        self.bubble_embedding = nn.Parameter(torch.empty(NUM_BUBBLES, WIDTH))
        self.bubble_score = nn.Linear(WIDTH, 1)
        self.cache_norm = nn.RMSNorm(WIDTH)
        self.final_norm = nn.LayerNorm(WIDTH)
        self.output = nn.Linear(WIDTH, spec.vocab_size, bias=False)
        self.apply(self._initialize)
        nn.init.normal_(self.bubble_embedding, mean=0.0, std=0.02)

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
        hidden = F.dropout(hidden, p=DROPOUT, training=self.training)
        current = self.blocks[0](hidden, attention_mask)
        bubbles = current[:, None, :, :] + self.bubble_embedding[None, :, None, :]
        recurrent_cache = torch.zeros_like(bubbles)
        final_middle = bubbles
        bubble_logits = self.bubble_score(bubbles).squeeze(-1)

        time_steps = self._decode_time_steps(input_ids)
        recurrent_steps = 3 if self.training else MAX_RECURRENT_STEPS
        for step in range(recurrent_steps):
            depth = self.depth_encoding(time_steps, step)[:, None, None, :]
            expanded_current = current[:, None, :, :].expand_as(recurrent_cache)
            fused = self.fusion(expanded_current, recurrent_cache)
            bubble_weights = torch.softmax(bubble_logits, dim=1)
            scored = fused + self.bubble_embedding[None, :, None, :] + depth
            scored = scored * (0.5 + bubble_weights[..., None])

            batch, _, length, _ = scored.shape
            flat_scored = scored.reshape(
                batch,
                NUM_BUBBLES * length,
                WIDTH,
            )
            flat_mask = None
            if attention_mask is not None:
                flat_mask = (
                    attention_mask[:, None, :]
                    .expand(
                        batch,
                        NUM_BUBBLES,
                        length,
                    )
                    .reshape(batch, NUM_BUBBLES * length)
                )
            middle = self.blocks[1](flat_scored, flat_mask).reshape(
                batch,
                NUM_BUBBLES,
                length,
                WIDTH,
            )
            candidate_logits = bubble_logits + self.bubble_score(middle).squeeze(-1)
            candidate_weights = torch.softmax(candidate_logits, dim=1)
            updated_cache = self.cache_norm(
                recurrent_cache + candidate_weights[..., None] * middle
            )
            active = (time_steps > step).view(-1, 1, 1, 1)
            active_scores = active.squeeze(-1)
            recurrent_cache = torch.where(active, updated_cache, recurrent_cache)
            final_middle = torch.where(active, middle, final_middle)
            bubble_logits = torch.where(
                active_scores,
                candidate_logits,
                bubble_logits,
            )

        bubble_weights = torch.softmax(bubble_logits, dim=1)
        merged = (bubble_weights[..., None] * final_middle).sum(dim=1)
        hidden = self.blocks[2](merged, attention_mask)
        logits = self.output(self.final_norm(hidden))
        return logits, None

    @staticmethod
    def _decode_time_steps(input_ids: Tensor) -> Tensor:
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
            digit = (token - DIGIT_OFFSET).clamp(min=0, max=9)
            time_steps = torch.where(
                is_digit,
                time_steps * 10 + digit,
                time_steps,
            )
        return time_steps.clamp(min=1, max=MAX_RECURRENT_STEPS)


def build_model(spec: ModelSpec) -> ThoughtbubbleTransformer:
    model = ThoughtbubbleTransformer(spec)
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
        weight_decay=0.1,
        capturable=spec.device_type == "cuda",
    )

    def learning_rate_multiplier(step: int) -> float:
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        progress = min(
            (step - WARMUP_STEPS) / (SCHEDULE_STEPS - WARMUP_STEPS),
            1.0,
        )
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_multiplier,
    )
    return OptimizerBundle(optimizer, scheduler)


def training_loss(logits: Tensor, labels: Tensor, auxiliary: object) -> Tensor:
    del auxiliary
    return F.cross_entropy(logits, labels, label_smoothing=0.05)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
    max_steps=SCHEDULE_STEPS,
)
