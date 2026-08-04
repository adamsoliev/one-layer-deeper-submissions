"""T-controlled Universal Transformer for repeated modular squaring.

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

D_MODEL = 128
NUM_HEADS = 4
FFN_DIM = 4 * D_MODEL
MAX_RECURRENT_STEPS = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
T_TOKEN_ID = 4
DIGIT_OFFSET = 7
NUM_DIGITS = 10


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class SinusoidalEncoding(nn.Module):
    """Encode positions or recurrent steps without a learned depth limit."""

    def __init__(self, width: int) -> None:
        super().__init__()
        frequencies = torch.exp(-math.log(10_000.0) * torch.arange(0, width, 2) / width)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, coordinates: Tensor, dtype: torch.dtype) -> Tensor:
        angles = coordinates.float().unsqueeze(-1) * self.frequencies
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2).to(dtype)


class RecurrentTransformerBlock(nn.Module):
    """One self-attention and position-wise transition shared across depth."""

    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.attention_output = nn.Linear(D_MODEL, D_MODEL)
        self.transition_norm = nn.LayerNorm(D_MODEL)
        self.transition_up = nn.Linear(D_MODEL, FFN_DIM)
        self.transition_down = nn.Linear(FFN_DIM, D_MODEL)

    def forward(self, state: Tensor, attention_mask: Tensor | None) -> Tensor:
        residual = state
        state = self.attention_norm(state)
        batch_size, sequence_length, _ = state.shape
        query, key, value = self.qkv(state).chunk(3, dim=-1)
        query = query.view(batch_size, sequence_length, NUM_HEADS, -1).transpose(1, 2)
        key = key.view(batch_size, sequence_length, NUM_HEADS, -1).transpose(1, 2)
        value = value.view(batch_size, sequence_length, NUM_HEADS, -1).transpose(1, 2)

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
        )
        state = (
            state.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                D_MODEL,
            )
        )
        state = residual + self.attention_output(state)
        transition = self.transition_down(
            F.relu(self.transition_up(self.transition_norm(state)))
        )
        return state + transition


class UniversalTransformer(nn.Module):
    """Refine every token in parallel for the number of steps encoded by T."""

    num_loops = MAX_RECURRENT_STEPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.coordinate_encoding = SinusoidalEncoding(D_MODEL)
        self.recurrent_block = RecurrentTransformerBlock()
        self.output_norm = nn.LayerNorm(D_MODEL)
        self.output = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        token_mask = self._token_mask(input_ids, attention_mask)
        time_steps = self._parse_time_steps(input_ids, token_mask)
        recurrent_steps = int(time_steps.amax().item())
        model_attention_mask = token_mask if attention_mask is None else attention_mask

        state = self.token_embedding(input_ids)
        positions = torch.arange(
            1,
            input_ids.shape[1] + 1,
            device=input_ids.device,
        )
        position_signal = self.coordinate_encoding(positions, state.dtype)[None, :, :]

        for step in range(recurrent_steps):
            step_coordinate = torch.tensor(step + 1, device=input_ids.device)
            step_signal = self.coordinate_encoding(step_coordinate, state.dtype)
            proposal = self.recurrent_block(
                state + position_signal + step_signal,
                model_attention_mask,
            )
            active = (time_steps > step)[:, None, None]
            state = torch.where(active, proposal, state)
            state = state.masked_fill(~token_mask[:, :, None], 0.0)

        return self.output(self.output_norm(state)), None

    @staticmethod
    def _token_mask(input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        if attention_mask is None:
            return input_ids != 0
        if attention_mask.ndim == 2:
            return attention_mask.to(device=input_ids.device, dtype=torch.bool)
        if attention_mask.ndim == 3:
            return attention_mask.any(dim=1).to(device=input_ids.device)
        raise ValueError("invalid attention_mask rank")

    @staticmethod
    def _parse_time_steps(input_ids: Tensor, token_mask: Tensor) -> Tensor:
        time_steps = torch.zeros(
            input_ids.shape[0],
            device=input_ids.device,
            dtype=torch.long,
        )
        reading_time = torch.zeros_like(time_steps, dtype=torch.bool)
        for position in range(input_ids.shape[1]):
            token = input_ids[:, position]
            reading_time = reading_time | (token == T_TOKEN_ID)
            digit = token - DIGIT_OFFSET
            is_digit = (
                reading_time
                & token_mask[:, position]
                & (digit >= 0)
                & (digit < NUM_DIGITS)
            )
            time_steps = torch.where(is_digit, 10 * time_steps + digit, time_steps)
        return time_steps.clamp(min=1, max=MAX_RECURRENT_STEPS)


class DeviceAdamW(torch.optim.Optimizer):
    """AdamW with every tensor state stored beside its parameter."""

    def __init__(
        self,
        parameters: object,
        *,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            parameters,
            {
                "lr": lr,
                "betas": betas,
                "weight_decay": weight_decay,
                "eps": eps,
            },
        )

    @torch.no_grad()
    def step(self, closure: object = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            learning_rate = float(group["lr"])
            beta1, beta2 = group["betas"]
            weight_decay = float(group["weight_decay"])
            epsilon = float(group["eps"])
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("DeviceAdamW does not support sparse gradients")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["first_moment"] = torch.zeros_like(parameter)
                    state["second_moment"] = torch.zeros_like(parameter)
                state["step"] += 1

                first_moment = state["first_moment"]
                second_moment = state["second_moment"]
                first_moment.lerp_(gradient, 1.0 - beta1)
                second_moment.mul_(beta2).addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - beta2,
                )

                step = state["step"]
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denominator = (
                    second_moment.sqrt().div_(math.sqrt(bias_correction2)).add_(epsilon)
                )
                parameter.mul_(1.0 - learning_rate * weight_decay)
                parameter.addcdiv_(
                    first_moment,
                    denominator,
                    value=-learning_rate / bias_correction1,
                )

        return loss


def build_model(spec: ModelSpec) -> UniversalTransformer:
    model = UniversalTransformer(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(
    model: nn.Module,
    spec: OptimizerSpec,
) -> OptimizerBundle:
    del spec
    return OptimizerBundle(
        DeviceAdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.95),
            weight_decay=WEIGHT_DECAY,
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
)
