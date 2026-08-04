"""Universal Transformer with a convolutional transition function.

Architecture references:
- https://arxiv.org/abs/1807.03819v3
- https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py
"""

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


class Config:
    """Architecture hyperparameters for a basic Universal Transformer."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        *,
        hidden_size: int = 128,
        filter_size: int = 512,
        num_heads: int = 4,
        num_recurrent_steps: int = 6,
        first_kernel_size: int = 3,
        second_kernel_size: int = 5,
        layer_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        transition_dropout: float = 0.1,
    ) -> None:
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if hidden_size % 2:
            raise ValueError("hidden_size must be even for sinusoidal positions")
        if num_recurrent_steps < 1:
            raise ValueError("num_recurrent_steps must be positive")
        if first_kernel_size % 2 != 1 or second_kernel_size % 2 != 1:
            raise ValueError("convolution kernels must be odd")
        for dropout in (
            layer_dropout,
            attention_dropout,
            transition_dropout,
        ):
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout probabilities must be in [0, 1)")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.num_heads = num_heads
        self.num_recurrent_steps = num_recurrent_steps
        self.first_kernel_size = first_kernel_size
        self.second_kernel_size = second_kernel_size
        self.layer_dropout = layer_dropout
        self.attention_dropout = attention_dropout
        self.transition_dropout = transition_dropout


class SinusoidalPositionEncoding(nn.Module):
    """Return the fixed horizontal timing signal from the UT paper."""

    def __init__(self, width: int) -> None:
        super().__init__()
        frequencies = torch.exp(-math.log(10_000.0) * torch.arange(0, width, 2) / width)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(
        self,
        length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32)
        angles = positions.unsqueeze(-1) * self.frequencies
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2).to(dtype)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.dropout = config.attention_dropout
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, state: Tensor, attention_mask: Tensor | None) -> Tensor:
        batch_size, sequence_length, _ = state.shape
        query, key, value = self.qkv(state).chunk(3, dim=-1)
        query = query.view(batch_size, sequence_length, self.num_heads, -1).transpose(
            1, 2
        )
        key = key.view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_heads, -1).transpose(
            1, 2
        )

        mask = None
        if attention_mask is not None:
            if attention_mask.shape == (batch_size, sequence_length):
                mask = attention_mask[:, None, None, :]
            elif attention_mask.shape == (
                batch_size,
                sequence_length,
                sequence_length,
            ):
                mask = attention_mask[:, None, :, :]
            else:
                raise ValueError("invalid attention_mask shape")
            mask = mask.to(device=state.device, dtype=torch.bool)

        state = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        state = (
            state.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.hidden_size,
            )
        )
        return self.output(state)


class SeparableConvolution(nn.Module):
    """Depthwise convolution followed by a pointwise channel projection."""

    def __init__(self, input_size: int, output_size: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            input_size,
            input_size,
            kernel_size,
            padding=kernel_size // 2,
            groups=input_size,
            bias=False,
        )
        self.pointwise = nn.Conv1d(input_size, output_size, 1)

    def forward(self, state: Tensor) -> Tensor:
        state = state.transpose(1, 2)
        state = self.pointwise(self.depthwise(state))
        return state.transpose(1, 2)


class ConvolutionalTransition(nn.Module):
    """The paper's separable-convolution transition function."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.first = SeparableConvolution(
            config.hidden_size,
            config.filter_size,
            config.first_kernel_size,
        )
        self.second = SeparableConvolution(
            config.filter_size,
            config.hidden_size,
            config.second_kernel_size,
        )
        self.dropout = nn.Dropout(config.transition_dropout)

    def forward(self, state: Tensor, token_mask: Tensor) -> Tensor:
        state = state.masked_fill(~token_mask[:, :, None], 0.0)
        state = self.dropout(F.relu(self.first(state)))
        state = state.masked_fill(~token_mask[:, :, None], 0.0)
        return self.second(state)


class RecurrentTransformerBlock(nn.Module):
    """One attention-plus-convolution block shared across recurrent depth."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadSelfAttention(config)
        self.transition_norm = nn.LayerNorm(config.hidden_size)
        self.transition = ConvolutionalTransition(config)
        self.residual_dropout = nn.Dropout(config.layer_dropout)

    def forward(
        self,
        state: Tensor,
        attention_mask: Tensor | None,
        token_mask: Tensor,
    ) -> Tensor:
        attention = self.attention(self.attention_norm(state), attention_mask)
        state = state + self.residual_dropout(attention)
        transition = self.transition(self.transition_norm(state), token_mask)
        state = state + self.residual_dropout(transition)
        return state.masked_fill(~token_mask[:, :, None], 0.0)


class UniversalTransformer(nn.Module):
    """Basic fixed-depth Universal Transformer encoder with tied parameters."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.num_loops = config.num_recurrent_steps
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_encoding = SinusoidalPositionEncoding(config.hidden_size)
        self.step_embedding = nn.Embedding(
            config.num_recurrent_steps,
            config.hidden_size,
        )
        nn.init.normal_(self.step_embedding.weight)
        self.input_dropout = nn.Dropout(config.layer_dropout)
        self.recurrent_block = RecurrentTransformerBlock(config)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        token_mask = self._token_mask(input_ids, attention_mask)
        model_attention_mask = token_mask if attention_mask is None else attention_mask

        state = self.input_dropout(self.token_embedding(input_ids))
        position_signal = self.position_encoding(
            input_ids.shape[1],
            device=input_ids.device,
            dtype=state.dtype,
        )

        for step in range(self.config.num_recurrent_steps):
            step_signal = self.step_embedding.weight[step].to(state.dtype)
            state = self.recurrent_block(
                state + position_signal[None, :, :] + step_signal,
                model_attention_mask,
                token_mask,
            )

        return self.output(self.output_norm(state)), None

    @staticmethod
    def _token_mask(input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        if attention_mask is None:
            return torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.ndim == 2:
            return attention_mask.to(device=input_ids.device, dtype=torch.bool)
        if attention_mask.ndim == 3:
            return attention_mask.any(dim=-1).to(device=input_ids.device)
        raise ValueError("invalid attention_mask rank")


def build_model(spec: ModelSpec) -> UniversalTransformer:
    model = UniversalTransformer(Config(spec.vocab_size, spec.max_seq_len))
    assert_model_state(model, spec)
    return model


def build_optimizer(
    model: nn.Module,
    spec: OptimizerSpec,
) -> OptimizerBundle:
    return OptimizerBundle(
        torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.98),
            eps=1e-9,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
)
