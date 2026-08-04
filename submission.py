"""Encoder-decoder Adaptive Universal Transformer.

Architecture references:
- https://arxiv.org/abs/1807.03819v3
- https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class UniversalTransformerConfig:
    """Hyperparameters shared by the encoder and decoder."""

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        act_max_steps: int,
        *,
        hidden_size: int = 512,
        filter_size: int = 2048,
        num_heads: int = 8,
        first_kernel_size: int = 3,
        second_kernel_size: int = 5,
        layer_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        transition_dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        act_epsilon: float = 0.01,
        act_halting_bias_init: float = 1.0,
        act_loss_weight: float = 0.01,
        share_source_target_embeddings: bool = False,
        tie_target_embedding: bool = True,
    ) -> None:
        if source_vocab_size < 1 or target_vocab_size < 1:
            raise ValueError("vocabulary sizes must be positive")
        if act_max_steps < 1:
            raise ValueError("act_max_steps must be positive")
        if hidden_size < 2 or hidden_size % 2:
            raise ValueError("hidden_size must be a positive even number")
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if filter_size < 1:
            raise ValueError("filter_size must be positive")
        if first_kernel_size < 1 or first_kernel_size % 2 != 1:
            raise ValueError("first_kernel_size must be a positive odd number")
        if second_kernel_size < 1 or second_kernel_size % 2 != 1:
            raise ValueError("second_kernel_size must be a positive odd number")
        if layer_norm_epsilon <= 0.0:
            raise ValueError("layer_norm_epsilon must be positive")
        if not 0.0 < act_epsilon < 1.0:
            raise ValueError("act_epsilon must be in (0, 1)")
        if act_loss_weight < 0.0:
            raise ValueError("act_loss_weight must be non-negative")
        for dropout in (
            layer_dropout,
            attention_dropout,
            transition_dropout,
        ):
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout probabilities must be in [0, 1)")
        if share_source_target_embeddings and source_vocab_size != target_vocab_size:
            raise ValueError(
                "shared source and target embeddings require equal vocabulary sizes"
            )

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.act_max_steps = act_max_steps
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.num_heads = num_heads
        self.first_kernel_size = first_kernel_size
        self.second_kernel_size = second_kernel_size
        self.layer_dropout = layer_dropout
        self.attention_dropout = attention_dropout
        self.transition_dropout = transition_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.act_epsilon = act_epsilon
        self.act_halting_bias_init = act_halting_bias_init
        self.act_loss_weight = act_loss_weight
        self.share_source_target_embeddings = share_source_target_embeddings
        self.tie_target_embedding = tie_target_embedding


class SinusoidalPositionEncoding(nn.Module):
    """Fixed horizontal timing signal used at every recurrent step."""

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


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention with learned projections."""

    def __init__(self, config: UniversalTransformerConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.attention_dropout = config.attention_dropout
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        query_state: Tensor,
        memory_state: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        batch_size, query_length, _ = query_state.shape
        memory_length = memory_state.shape[1]
        if attention_mask.shape != (batch_size, query_length, memory_length):
            raise ValueError(
                "attention_mask must have shape (batch, query_length, memory_length)"
            )

        query = self._split_heads(self.query(query_state))
        key = self._split_heads(self.key(memory_state))
        value = self._split_heads(self.value(memory_state))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask[:, None, :, :].to(
                device=query_state.device,
                dtype=torch.bool,
            ),
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, query_length, self.hidden_size)
        )
        return self.output(attended)

    def _split_heads(self, state: Tensor) -> Tensor:
        batch_size, length, _ = state.shape
        return state.view(batch_size, length, self.num_heads, -1).transpose(1, 2)


class SeparableConvolution(nn.Module):
    """Depthwise convolution followed by a pointwise channel projection."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int,
        *,
        causal: bool,
    ) -> None:
        super().__init__()
        self.left_padding = kernel_size - 1 if causal else 0
        convolution_padding = 0 if causal else kernel_size // 2
        self.depthwise = nn.Conv1d(
            input_size,
            input_size,
            kernel_size,
            padding=convolution_padding,
            groups=input_size,
            bias=False,
        )
        self.pointwise = nn.Conv1d(input_size, output_size, 1)

    def forward(self, state: Tensor) -> Tensor:
        state = state.transpose(1, 2)
        if self.left_padding:
            state = F.pad(state, (self.left_padding, 0))
        state = self.pointwise(self.depthwise(state))
        return state.transpose(1, 2)


class ConvolutionalTransition(nn.Module):
    """Separable-convolution transition from the UT reference implementation."""

    def __init__(
        self,
        config: UniversalTransformerConfig,
        *,
        causal: bool,
    ) -> None:
        super().__init__()
        self.first = SeparableConvolution(
            config.hidden_size,
            config.filter_size,
            config.first_kernel_size,
            causal=causal,
        )
        self.second = SeparableConvolution(
            config.filter_size,
            config.hidden_size,
            config.second_kernel_size,
            causal=causal,
        )
        self.dropout = nn.Dropout(config.transition_dropout)

    def forward(self, state: Tensor, token_mask: Tensor) -> Tensor:
        state = state.masked_fill(~token_mask[:, :, None], 0.0)
        state = self.dropout(F.relu(self.first(state)))
        state = state.masked_fill(~token_mask[:, :, None], 0.0)
        return self.second(state)


class UniversalTransformerEncoderBlock(nn.Module):
    """One encoder transition shared across all encoder recurrent steps."""

    def __init__(self, config: UniversalTransformerConfig) -> None:
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.self_attention = MultiHeadAttention(config)
        self.transition_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.transition = ConvolutionalTransition(config, causal=False)
        self.residual_dropout = nn.Dropout(config.layer_dropout)

    def forward(
        self,
        state: Tensor,
        token_mask: Tensor,
        self_attention_mask: Tensor,
    ) -> Tensor:
        normalized = self.self_attention_norm(state)
        attended = self.self_attention(normalized, normalized, self_attention_mask)
        state = state + self.residual_dropout(attended)
        transitioned = self.transition(self.transition_norm(state), token_mask)
        state = state + self.residual_dropout(transitioned)
        return state.masked_fill(~token_mask[:, :, None], 0.0)


class UniversalTransformerDecoderBlock(nn.Module):
    """One decoder transition shared across all decoder recurrent steps."""

    def __init__(self, config: UniversalTransformerConfig) -> None:
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.cross_attention = MultiHeadAttention(config)
        self.transition_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.transition = ConvolutionalTransition(config, causal=True)
        self.residual_dropout = nn.Dropout(config.layer_dropout)

    def forward(
        self,
        state: Tensor,
        token_mask: Tensor,
        self_attention_mask: Tensor,
        encoder_state: Tensor,
        cross_attention_mask: Tensor,
    ) -> Tensor:
        normalized = self.self_attention_norm(state)
        attended = self.self_attention(normalized, normalized, self_attention_mask)
        state = state + self.residual_dropout(attended)

        normalized = self.cross_attention_norm(state)
        attended = self.cross_attention(
            normalized,
            encoder_state,
            cross_attention_mask,
        )
        state = state + self.residual_dropout(attended)

        transitioned = self.transition(self.transition_norm(state), token_mask)
        state = state + self.residual_dropout(transitioned)
        return state.masked_fill(~token_mask[:, :, None], 0.0)


class AdaptiveRecurrentStack(nn.Module):
    """Apply one tied block with learned step signals and position-wise ACT."""

    def __init__(
        self,
        config: UniversalTransformerConfig,
        block: nn.Module,
    ) -> None:
        super().__init__()
        self.config = config
        self.block = block
        self.position_encoding = SinusoidalPositionEncoding(config.hidden_size)
        self.step_embedding = nn.Embedding(
            config.act_max_steps,
            config.hidden_size,
        )
        nn.init.normal_(self.step_embedding.weight)
        self.halting_projection = nn.Linear(config.hidden_size, 1)
        nn.init.constant_(self.halting_projection.bias, config.act_halting_bias_init)

    def forward(
        self,
        state: Tensor,
        token_mask: Tensor,
        *block_arguments: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        state = state.masked_fill(~token_mask[:, :, None], 0.0)
        position_signal = self.position_encoding(
            state.shape[1],
            device=state.device,
            dtype=state.dtype,
        )
        halting_probability = (~token_mask).to(state.dtype)
        remainders = state.new_zeros(token_mask.shape)
        ponder_times = state.new_zeros(token_mask.shape)
        weighted_state = torch.zeros_like(state)
        threshold = 1.0 - self.config.act_epsilon
        steps_executed = 0

        for step in range(self.config.act_max_steps):
            running = (halting_probability < threshold) & token_mask
            if not running.any().item():
                break
            steps_executed += 1

            step_signal = self.step_embedding.weight[step].to(state.dtype)
            timed_state = state + position_signal[None, :, :] + step_signal
            timed_state = timed_state.masked_fill(~token_mask[:, :, None], 0.0)
            halting_scores = torch.sigmoid(
                self.halting_projection(timed_state).squeeze(-1)
            )
            newly_halted = (
                halting_probability + halting_scores * running > threshold
            ) & running
            continuing = (
                halting_probability + halting_scores * running <= threshold
            ) & running

            halting_probability = halting_probability + halting_scores * continuing
            step_remainders = newly_halted * (1.0 - halting_probability)
            remainders = remainders + step_remainders
            halting_probability = halting_probability + step_remainders
            ponder_times = ponder_times + continuing + newly_halted
            update_weights = (halting_scores * continuing + step_remainders).unsqueeze(
                -1
            )

            proposal = self.block(timed_state, token_mask, *block_arguments)
            state = torch.where(running[:, :, None], proposal, state)
            weighted_state = state * update_weights + weighted_state * (
                1.0 - update_weights
            )

        valid_positions = token_mask.sum().clamp_min(1)
        ponder_cost = ((ponder_times + remainders) * token_mask).sum() / valid_positions
        statistics = {
            "halting_probability": halting_probability,
            "remainders": remainders,
            "ponder_times": ponder_times,
            "ponder_cost": ponder_cost,
            "steps_executed": torch.tensor(
                steps_executed,
                device=state.device,
                dtype=torch.int64,
            ),
        }
        return weighted_state, statistics


class UniversalTransformerEncoder(nn.Module):
    """Adaptive Universal Transformer encoder."""

    def __init__(
        self,
        config: UniversalTransformerConfig,
        embedding: nn.Embedding,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(config.layer_dropout)
        self.recurrent_stack = AdaptiveRecurrentStack(
            config,
            UniversalTransformerEncoderBlock(config),
        )
        self.output_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

    def forward(
        self,
        source_ids: Tensor,
        source_mask: Tensor,
        self_attention_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        state = self.embedding(source_ids) * math.sqrt(self.hidden_size)
        state = self.embedding_dropout(state)
        state, statistics = self.recurrent_stack(
            state,
            source_mask,
            self_attention_mask,
        )
        state = self.output_norm(state)
        state = state.masked_fill(~source_mask[:, :, None], 0.0)
        return state, statistics


class UniversalTransformerDecoder(nn.Module):
    """Autoregressive Adaptive Universal Transformer decoder."""

    def __init__(
        self,
        config: UniversalTransformerConfig,
        embedding: nn.Embedding,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(config.layer_dropout)
        self.recurrent_stack = AdaptiveRecurrentStack(
            config,
            UniversalTransformerDecoderBlock(config),
        )
        self.output_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

    def forward(
        self,
        decoder_input_ids: Tensor,
        decoder_mask: Tensor,
        self_attention_mask: Tensor,
        encoder_state: Tensor,
        cross_attention_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        state = self.embedding(decoder_input_ids) * math.sqrt(self.hidden_size)
        state = self.embedding_dropout(state)
        state, statistics = self.recurrent_stack(
            state,
            decoder_mask,
            self_attention_mask,
            encoder_state,
            cross_attention_mask,
        )
        state = self.output_norm(state)
        state = state.masked_fill(~decoder_mask[:, :, None], 0.0)
        return state, statistics


class UniversalTransformer(nn.Module):
    """Encoder-decoder Adaptive Universal Transformer for sequence transduction."""

    def __init__(self, config: UniversalTransformerConfig) -> None:
        super().__init__()
        self.config = config
        source_embedding = nn.Embedding(
            config.source_vocab_size,
            config.hidden_size,
        )
        nn.init.normal_(
            source_embedding.weight,
            mean=0.0,
            std=config.hidden_size**-0.5,
        )
        if config.share_source_target_embeddings:
            target_embedding = source_embedding
        else:
            target_embedding = nn.Embedding(
                config.target_vocab_size,
                config.hidden_size,
            )
            nn.init.normal_(
                target_embedding.weight,
                mean=0.0,
                std=config.hidden_size**-0.5,
            )

        self.encoder = UniversalTransformerEncoder(config, source_embedding)
        self.decoder = UniversalTransformerDecoder(config, target_embedding)
        self.output_projection = nn.Linear(
            config.hidden_size,
            config.target_vocab_size,
            bias=False,
        )
        if config.tie_target_embedding:
            self.output_projection.weight = target_embedding.weight

    def encode(
        self,
        source_ids: Tensor,
        *,
        source_padding_mask: Tensor | None = None,
        source_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        source_mask = self._token_mask(source_ids, source_padding_mask)
        self_attention_mask = self._self_attention_mask(
            source_mask,
            causal=False,
            supplied_mask=source_attention_mask,
        )
        return self.encoder(source_ids, source_mask, self_attention_mask)

    def decode(
        self,
        decoder_input_ids: Tensor,
        encoder_state: Tensor,
        *,
        source_padding_mask: Tensor | None = None,
        decoder_padding_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        cross_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch_size, source_length, _ = encoder_state.shape
        if decoder_input_ids.shape[0] != batch_size:
            raise ValueError("source and decoder batch sizes must match")
        source_mask = self._state_mask(
            batch_size,
            source_length,
            encoder_state.device,
            source_padding_mask,
        )
        decoder_mask = self._token_mask(decoder_input_ids, decoder_padding_mask)
        self_attention_mask = self._self_attention_mask(
            decoder_mask,
            causal=True,
            supplied_mask=decoder_attention_mask,
        )
        encoder_decoder_mask = decoder_mask[:, :, None] & source_mask[:, None, :]
        encoder_decoder_mask = self._merge_attention_mask(
            encoder_decoder_mask,
            cross_attention_mask,
        )
        return self.decoder(
            decoder_input_ids,
            decoder_mask,
            self_attention_mask,
            encoder_state,
            encoder_decoder_mask,
        )

    def forward(
        self,
        source_ids: Tensor,
        decoder_input_ids: Tensor,
        *,
        source_padding_mask: Tensor | None = None,
        decoder_padding_mask: Tensor | None = None,
        source_attention_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        cross_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        """Return target logits for already-right-shifted decoder inputs."""
        encoder_state, encoder_statistics = self.encode(
            source_ids,
            source_padding_mask=source_padding_mask,
            source_attention_mask=source_attention_mask,
        )
        decoder_state, decoder_statistics = self.decode(
            decoder_input_ids,
            encoder_state,
            source_padding_mask=source_padding_mask,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=decoder_attention_mask,
            cross_attention_mask=cross_attention_mask,
        )
        logits = self.output_projection(decoder_state)
        act_loss = self.config.act_loss_weight * (
            encoder_statistics["ponder_cost"] + decoder_statistics["ponder_cost"]
        )
        auxiliary: dict[str, object] = {
            "encoder": encoder_statistics,
            "decoder": decoder_statistics,
            "act_loss": act_loss,
        }
        return logits, auxiliary

    def compute_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        auxiliary: dict[str, object],
        *,
        ignore_index: int = -100,
    ) -> Tensor:
        """Combine autoregressive cross-entropy with the ACT ponder penalty."""
        prediction_loss = F.cross_entropy(
            logits.flatten(0, 1),
            labels.flatten(),
            ignore_index=ignore_index,
        )
        act_loss = auxiliary["act_loss"]
        if not isinstance(act_loss, Tensor):
            raise TypeError("auxiliary['act_loss'] must be a tensor")
        return prediction_loss + act_loss

    @torch.no_grad()
    def generate(
        self,
        source_ids: Tensor,
        *,
        bos_token_id: int,
        eos_token_id: int,
        max_new_tokens: int,
        source_padding_mask: Tensor | None = None,
        source_attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Greedily decode one target token at a time without a decoder cache."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        encoder_state, _ = self.encode(
            source_ids,
            source_padding_mask=source_padding_mask,
            source_attention_mask=source_attention_mask,
        )
        generated = torch.full(
            (source_ids.shape[0], 1),
            bos_token_id,
            device=source_ids.device,
            dtype=torch.long,
        )
        finished = torch.zeros(
            source_ids.shape[0], device=source_ids.device, dtype=torch.bool
        )
        for _ in range(max_new_tokens):
            decoder_state, _ = self.decode(
                generated,
                encoder_state,
                source_padding_mask=source_padding_mask,
            )
            next_token = self.output_projection(decoder_state[:, -1]).argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, eos_token_id),
                next_token,
            )
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            finished = finished | (next_token == eos_token_id)
            if finished.all().item():
                break
        return generated

    @staticmethod
    def _token_mask(token_ids: Tensor, padding_mask: Tensor | None) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token IDs must have shape (batch, length)")
        return UniversalTransformer._state_mask(
            token_ids.shape[0],
            token_ids.shape[1],
            token_ids.device,
            padding_mask,
        )

    @staticmethod
    def _state_mask(
        batch_size: int,
        length: int,
        device: torch.device,
        padding_mask: Tensor | None,
    ) -> Tensor:
        if padding_mask is None:
            return torch.ones((batch_size, length), device=device, dtype=torch.bool)
        if padding_mask.shape != (batch_size, length):
            raise ValueError("padding masks must have shape (batch, length)")
        return padding_mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _self_attention_mask(
        token_mask: Tensor,
        *,
        causal: bool,
        supplied_mask: Tensor | None,
    ) -> Tensor:
        length = token_mask.shape[1]
        attention_mask = token_mask[:, :, None] & token_mask[:, None, :]
        if causal:
            causal_mask = torch.ones(
                (length, length),
                device=token_mask.device,
                dtype=torch.bool,
            ).tril()
            attention_mask = attention_mask & causal_mask[None, :, :]
        return UniversalTransformer._merge_attention_mask(
            attention_mask,
            supplied_mask,
        )

    @staticmethod
    def _merge_attention_mask(
        base_mask: Tensor,
        supplied_mask: Tensor | None,
    ) -> Tensor:
        if supplied_mask is None:
            return base_mask
        if supplied_mask.shape != base_mask.shape:
            raise ValueError(f"attention mask must have shape {tuple(base_mask.shape)}")
        return base_mask & supplied_mask.to(device=base_mask.device, dtype=torch.bool)
