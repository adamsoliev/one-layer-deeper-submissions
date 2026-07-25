"""Simple recurrent neural network for One Layer Deeper."""

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


WIDTH = 128


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class SimpleRNN(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.embedding = nn.Embedding(spec.vocab_size, WIDTH)
        self.rnn = nn.RNN(
            input_size=WIDTH,
            hidden_size=WIDTH,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Linear(2 * WIDTH, spec.vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        del attention_mask
        hidden, _ = self.rnn(self.embedding(input_ids))
        return self.output(hidden), None


def build_model(spec: ModelSpec) -> SimpleRNN:
    model = SimpleRNN(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(
    model: nn.Module,
    spec: OptimizerSpec,
) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(optimizer)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
)
