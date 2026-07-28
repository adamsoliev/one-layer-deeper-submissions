"""Sparse 64-expert MoE for One Layer Deeper."""

from __future__ import annotations

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

NUM_EXPERTS = 64
WIDTH = 98
EXPERT_HIDDEN_WIDTH = 395
MODEL_STATE_LIMIT = 5_000_000
ROUTER_AUX_LOSS_WEIGHT = 0.01


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Expert(nn.Module):
    """Two-layer GELU feed-forward expert."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(WIDTH, EXPERT_HIDDEN_WIDTH)
        self.down = nn.Linear(EXPERT_HIDDEN_WIDTH, WIDTH)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down(F.gelu(self.up(hidden), approximate="tanh"))


class SparseMoE(nn.Module):
    """Route each embedded token through one of 64 feed-forward experts."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.embedding = nn.Embedding(spec.vocab_size, WIDTH)
        self.router = nn.Linear(WIDTH, NUM_EXPERTS)
        self.experts = nn.ModuleList(Expert() for _ in range(NUM_EXPERTS))
        self.output = nn.Linear(WIDTH, spec.vocab_size)
        self.apply(self._initialize)
        self.output.weight = self.embedding.weight

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
        hidden = self.embedding(input_ids)
        router_logits = self.router(hidden)
        router_probabilities = F.softmax(router_logits.float(), dim=-1)
        expert_ids = router_probabilities.argmax(dim=-1)
        routing_weights = router_probabilities.gather(
            dim=-1,
            index=expert_ids.unsqueeze(-1),
        ).squeeze(-1)

        flat_hidden = hidden.reshape(-1, WIDTH)
        flat_expert_ids = expert_ids.reshape(-1)
        flat_routing_weights = routing_weights.reshape(-1).to(hidden.dtype)
        flat_output = torch.zeros_like(flat_hidden)

        for expert_id, expert in enumerate(self.experts):
            token_indices = torch.nonzero(
                flat_expert_ids == expert_id,
                as_tuple=False,
            ).squeeze(-1)
            expert_input = flat_hidden.index_select(0, token_indices)
            expert_output = expert(expert_input)
            weighted_output = expert_output * flat_routing_weights.index_select(
                0,
                token_indices,
            ).unsqueeze(-1)
            flat_output.index_copy_(0, token_indices, weighted_output)

        routed_hidden = flat_output.view_as(hidden)
        logits = self.output(routed_hidden)
        router_loss = self._router_load_balancing_loss(
            router_probabilities,
            expert_ids,
            attention_mask,
        )
        return logits, router_loss

    @staticmethod
    def _router_load_balancing_loss(
        probabilities: Tensor,
        expert_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        assignments = F.one_hot(
            expert_ids,
            num_classes=NUM_EXPERTS,
        ).to(probabilities.dtype)

        if attention_mask is None:
            token_weights = probabilities.new_ones(expert_ids.shape)
        else:
            if attention_mask.shape != expert_ids.shape:
                raise ValueError("attention_mask must match input_ids")
            token_weights = attention_mask.to(
                device=probabilities.device,
                dtype=probabilities.dtype,
            )

        normalizer = token_weights.sum().clamp(min=1)
        token_weights = token_weights.unsqueeze(-1)
        expert_fraction = (assignments * token_weights).sum(dim=(0, 1))
        expert_fraction = expert_fraction / normalizer
        probability_fraction = (probabilities * token_weights).sum(dim=(0, 1))
        probability_fraction = probability_fraction / normalizer
        return NUM_EXPERTS * torch.sum(expert_fraction * probability_fraction)


def build_model(spec: ModelSpec) -> SparseMoE:
    model = SparseMoE(spec)
    model_state = sum(parameter.numel() for parameter in model.parameters())
    model_state += sum(buffer.numel() for buffer in model.buffers())
    if model_state > MODEL_STATE_LIMIT:
        raise ValueError(
            f"model state {model_state:,} exceeds {MODEL_STATE_LIMIT:,}"
        )
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
    if not isinstance(auxiliary, Tensor) or auxiliary.ndim != 0:
        raise TypeError("router loss must be a scalar tensor")
    return F.cross_entropy(logits, labels) + ROUTER_AUX_LOSS_WEIGHT * auxiliary


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
)
