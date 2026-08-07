"""Encoder-decoder Adaptive Universal Transformer.

Architecture references:
- https://arxiv.org/abs/1807.03819v3
- https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py

Tensor shape suffixes use the following dimension key:
- B: batch size
- S: source sequence length
- T: target/decoder sequence length
- L: sequence or attention-query length inside reusable components
- M: attention-memory length
- D: model hidden size
- V: vocabulary size
- F: convolutional transition hidden size
- H: number of attention heads
- K: channels per attention head (D / H)
- C: generic convolution input channels
- O: generic convolution output channels
- R: number of sinusoidal frequencies (D / 2)

Suffix letters appear in physical axis order. Scalar tensors and heterogeneous
tensor containers do not carry a shape suffix.
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
        frequencies_R = torch.exp(
            -math.log(10_000.0) * torch.arange(0, width, 2) / width
        )
        self.register_buffer("frequencies_R", frequencies_R, persistent=False)

    def forward(
        self,
        length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        positions_L = torch.arange(length, device=device, dtype=torch.float32)
        angles_LR = positions_L.unsqueeze(-1) * self.frequencies_R
        signal_LD = torch.stack(
            (angles_LR.sin(), angles_LR.cos()),
            dim=-1,
        ).flatten(-2)
        return signal_LD.to(dtype)


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
        query_state_BLD: Tensor,
        memory_state_BMD: Tensor,
        attention_mask_BLM: Tensor,
    ) -> Tensor:
        batch_size, query_length, _ = query_state_BLD.shape
        memory_length = memory_state_BMD.shape[1]
        if attention_mask_BLM.shape != (batch_size, query_length, memory_length):
            raise ValueError(
                "attention_mask must have shape (batch, query_length, memory_length)"
            )

        query_BHLK = self._split_heads(self.query(query_state_BLD))
        key_BHMK = self._split_heads(self.key(memory_state_BMD))
        value_BHMK = self._split_heads(self.value(memory_state_BMD))
        attended_BHLK = F.scaled_dot_product_attention(
            query_BHLK,
            key_BHMK,
            value_BHMK,
            attn_mask=attention_mask_BLM[:, None, :, :].to(
                device=query_state_BLD.device,
                dtype=torch.bool,
            ),
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended_BLD = (
            attended_BHLK.transpose(1, 2)
            .contiguous()
            .view(batch_size, query_length, self.hidden_size)
        )
        output_BLD = self.output(attended_BLD)
        return output_BLD

    def _split_heads(self, state_BLD: Tensor) -> Tensor:
        batch_size, length, _ = state_BLD.shape
        state_BLHK = state_BLD.view(batch_size, length, self.num_heads, -1)
        state_BHLK = state_BLHK.transpose(1, 2)
        return state_BHLK


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

    def forward(self, state_BLC: Tensor) -> Tensor:
        state_BCL = state_BLC.transpose(1, 2)
        if self.left_padding:
            state_BCL = F.pad(state_BCL, (self.left_padding, 0))
        state_BOL = self.pointwise(self.depthwise(state_BCL))
        state_BLO = state_BOL.transpose(1, 2)
        return state_BLO


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

    def forward(self, state_BLD: Tensor, nonpadding_mask_BL: Tensor) -> Tensor:
        state_BLD = state_BLD.masked_fill(~nonpadding_mask_BL[:, :, None], 0.0)
        hidden_BLF = self.dropout(F.relu(self.first(state_BLD)))
        hidden_BLF = hidden_BLF.masked_fill(~nonpadding_mask_BL[:, :, None], 0.0)
        output_BLD = self.second(hidden_BLF)
        return output_BLD


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
        state_BLD: Tensor,
        nonpadding_mask_BL: Tensor,
        self_attention_mask_BLL: Tensor,
    ) -> Tensor:
        normalized_BLD = self.self_attention_norm(state_BLD)
        attended_BLD = self.self_attention(
            normalized_BLD,
            normalized_BLD,
            self_attention_mask_BLL,
        )
        state_BLD = state_BLD + self.residual_dropout(attended_BLD)
        transitioned_BLD = self.transition(
            self.transition_norm(state_BLD),
            nonpadding_mask_BL,
        )
        state_BLD = state_BLD + self.residual_dropout(transitioned_BLD)
        return state_BLD.masked_fill(~nonpadding_mask_BL[:, :, None], 0.0)


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
        state_BLD: Tensor,
        nonpadding_mask_BL: Tensor,
        self_attention_mask_BLL: Tensor,
        encoder_state_BMD: Tensor,
        cross_attention_mask_BLM: Tensor,
    ) -> Tensor:
        normalized_BLD = self.self_attention_norm(state_BLD)
        attended_BLD = self.self_attention(
            normalized_BLD,
            normalized_BLD,
            self_attention_mask_BLL,
        )
        state_BLD = state_BLD + self.residual_dropout(attended_BLD)

        normalized_BLD = self.cross_attention_norm(state_BLD)
        attended_BLD = self.cross_attention(
            normalized_BLD,
            encoder_state_BMD,
            cross_attention_mask_BLM,
        )
        state_BLD = state_BLD + self.residual_dropout(attended_BLD)

        transitioned_BLD = self.transition(
            self.transition_norm(state_BLD),
            nonpadding_mask_BL,
        )
        state_BLD = state_BLD + self.residual_dropout(transitioned_BLD)
        return state_BLD.masked_fill(~nonpadding_mask_BL[:, :, None], 0.0)


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
        state_BLD: Tensor,
        nonpadding_mask_BL: Tensor,
        *block_argument_tensors: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Refine token representations by recurrently applying a tied block with per-token adaptive halting..

        IN: state,
            non-padding mask and *,
            self-attention mask for encoder
            self-attention mask, encoder state, and cross-attention mask for decoder

        OUT: contextual source states [B, S, D] and ACT statistics.
        """
        state_BLD = state_BLD.masked_fill(~nonpadding_mask_BL[:, :, None], 0.0)
        position_signal_LD = self.position_encoding(
            state_BLD.shape[1],
            device=state_BLD.device,
            dtype=state_BLD.dtype,
        )
        # initialize accumulators
        # padding tokens halted (1), others continuing (0)
        halting_probability_BL = (~nonpadding_mask_BL).to(state_BLD.dtype)
        remainders_BL = state_BLD.new_zeros(nonpadding_mask_BL.shape)
        ponder_times_BL = state_BLD.new_zeros(nonpadding_mask_BL.shape)
        weighted_state_BLD = torch.zeros_like(state_BLD)
        threshold = 1.0 - self.config.act_epsilon
        steps_executed = 0

        for step in range(self.config.act_max_steps):
            active_token_mask_BL = (
                halting_probability_BL < threshold
            ) & nonpadding_mask_BL
            if not active_token_mask_BL.any().item():
                break
            steps_executed += 1

            step_signal_D = self.step_embedding.weight[step].to(state_BLD.dtype)
            timed_state_BLD = state_BLD + position_signal_LD[None, :, :] + step_signal_D
            timed_state_BLD = timed_state_BLD.masked_fill(
                ~nonpadding_mask_BL[:, :, None],
                0.0,
            )
            # timed state now has token’s previous-step representation, its position signal, and its recurrent step signal
            # learned projection uses this current representation to decide halting
            halting_scores_BL = torch.sigmoid(
                self.halting_projection(timed_state_BLD).squeeze(-1)
            )
            active_score_BL = halting_scores_BL * active_token_mask_BL
            candidate_probability_BL = halting_probability_BL + active_score_BL
            newly_halted_mask_BL = (
                candidate_probability_BL > threshold
            ) & active_token_mask_BL
            continuing_mask_BL = (
                candidate_probability_BL <= threshold
            ) & active_token_mask_BL

            halting_probability_BL = (
                halting_probability_BL + halting_scores_BL * continuing_mask_BL
            )
            step_remainders_BL = newly_halted_mask_BL * (1.0 - halting_probability_BL)
            remainders_BL = remainders_BL + step_remainders_BL
            # ensures 1 for each halted token
            halting_probability_BL = halting_probability_BL + step_remainders_BL
            ponder_times_BL = (
                ponder_times_BL + continuing_mask_BL + newly_halted_mask_BL
            )
            update_weights_BL = (
                halting_scores_BL * continuing_mask_BL + step_remainders_BL
            )

            proposal_BLD = self.block(
                timed_state_BLD,
                nonpadding_mask_BL,
                *block_argument_tensors,
            )
            state_BLD = torch.where(
                active_token_mask_BL[:, :, None],
                proposal_BLD,
                state_BLD,
            )
            weighted_state_BLD = state_BLD * update_weights_BL[
                :, :, None
            ] + weighted_state_BLD * (1.0 - update_weights_BL[:, :, None])

        nonpadding_token_count = nonpadding_mask_BL.sum().clamp_min(1)
        ponder_cost = (
            (ponder_times_BL + remainders_BL) * nonpadding_mask_BL
        ).sum() / nonpadding_token_count
        statistics = {
            "halting_probability_BL": halting_probability_BL,
            "remainders_BL": remainders_BL,
            "ponder_times_BL": ponder_times_BL,
            "ponder_cost": ponder_cost,
            "steps_executed": torch.tensor(
                steps_executed,
                device=state_BLD.device,
                dtype=torch.int64,
            ),
        }
        return weighted_state_BLD, statistics


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
        source_token_id_BS: Tensor,
        source_nonpadding_mask_BS: Tensor,
        self_attention_mask_BSS: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Encode source tokens into contextual representations.

        IN: source token IDs [B, S],
            non-padding mask [B, S],
            self-attention mask [B, S, S].

        OUT: contextual source states [B, S, D] and ACT statistics.
        """
        # Embeddings are initialized with std D^(-1/2); scaling by sqrt(D)
        # gives the looked-up vectors unit standard deviation.
        state_BSD = self.embedding(source_token_id_BS) * math.sqrt(self.hidden_size)
        state_BSD = self.embedding_dropout(state_BSD)

        # Recurrently apply the shared encoder block with per-position ACT halting.
        state_BSD, statistics = self.recurrent_stack(
            state_BSD,
            source_nonpadding_mask_BS,
            self_attention_mask_BSS,
        )

        # Normalize the final state after the recurrent residual updates.
        state_BSD = self.output_norm(state_BSD)

        # LayerNorm's learned bias can make padded states nonzero; zero them again.
        state_BSD = state_BSD.masked_fill(
            ~source_nonpadding_mask_BS[:, :, None],
            0.0,
        )
        return state_BSD, statistics


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
        decoder_input_token_id_BT: Tensor,
        decoder_nonpadding_mask_BT: Tensor,
        self_attention_mask_BTT: Tensor,
        encoder_state_BSD: Tensor,
        cross_attention_mask_BTS: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        state_BTD = self.embedding(decoder_input_token_id_BT) * math.sqrt(
            self.hidden_size
        )
        state_BTD = self.embedding_dropout(state_BTD)
        state_BTD, statistics = self.recurrent_stack(
            state_BTD,
            decoder_nonpadding_mask_BT,
            self_attention_mask_BTT,
            encoder_state_BSD,
            cross_attention_mask_BTS,
        )
        state_BTD = self.output_norm(state_BTD)
        state_BTD = state_BTD.masked_fill(
            ~decoder_nonpadding_mask_BT[:, :, None],
            0.0,
        )
        return state_BTD, statistics


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
        source_token_id_BS: Tensor,
        *,
        source_nonpadding_mask_BS: Tensor | None = None,
        source_attention_mask_BSS: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        source_nonpadding_mask_BS = self._nonpadding_mask(
            source_token_id_BS,
            source_nonpadding_mask_BS,
        )
        self_attention_mask_BSS = self._self_attention_mask(
            source_nonpadding_mask_BS,
            causal=False,
            supplied_mask_BLL=source_attention_mask_BSS,
        )
        return self.encoder(
            source_token_id_BS,
            source_nonpadding_mask_BS,
            self_attention_mask_BSS,
        )

    def decode(
        self,
        decoder_input_token_id_BT: Tensor,
        encoder_state_BSD: Tensor,
        *,
        source_nonpadding_mask_BS: Tensor | None = None,
        decoder_nonpadding_mask_BT: Tensor | None = None,
        decoder_attention_mask_BTT: Tensor | None = None,
        cross_attention_mask_BTS: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch_size, source_length, _ = encoder_state_BSD.shape
        if decoder_input_token_id_BT.shape[0] != batch_size:
            raise ValueError("source and decoder batch sizes must match")
        source_nonpadding_mask_BS = self._state_nonpadding_mask(
            batch_size,
            source_length,
            encoder_state_BSD.device,
            source_nonpadding_mask_BS,
        )
        decoder_nonpadding_mask_BT = self._nonpadding_mask(
            decoder_input_token_id_BT,
            decoder_nonpadding_mask_BT,
        )
        self_attention_mask_BTT = self._self_attention_mask(
            decoder_nonpadding_mask_BT,
            causal=True,
            supplied_mask_BLL=decoder_attention_mask_BTT,
        )
        base_cross_attention_mask_BTS = (
            decoder_nonpadding_mask_BT[:, :, None]
            & source_nonpadding_mask_BS[:, None, :]
        )
        cross_attention_mask_BTS = self._merge_attention_mask(
            base_cross_attention_mask_BTS,
            supplied_mask_BLM=cross_attention_mask_BTS,
        )
        return self.decoder(
            decoder_input_token_id_BT,
            decoder_nonpadding_mask_BT,
            self_attention_mask_BTT,
            encoder_state_BSD,
            cross_attention_mask_BTS,
        )

    def forward(
        self,
        source_token_id_BS: Tensor,
        decoder_input_token_id_BT: Tensor,
        *,
        source_nonpadding_mask_BS: Tensor | None = None,
        decoder_nonpadding_mask_BT: Tensor | None = None,
        source_attention_mask_BSS: Tensor | None = None,
        decoder_attention_mask_BTT: Tensor | None = None,
        cross_attention_mask_BTS: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        """Return target logits for already-right-shifted decoder inputs."""
        encoder_state_BSD, encoder_statistics = self.encode(
            source_token_id_BS,
            source_nonpadding_mask_BS=source_nonpadding_mask_BS,
            source_attention_mask_BSS=source_attention_mask_BSS,
        )
        decoder_state_BTD, decoder_statistics = self.decode(
            decoder_input_token_id_BT,
            encoder_state_BSD,
            source_nonpadding_mask_BS=source_nonpadding_mask_BS,
            decoder_nonpadding_mask_BT=decoder_nonpadding_mask_BT,
            decoder_attention_mask_BTT=decoder_attention_mask_BTT,
            cross_attention_mask_BTS=cross_attention_mask_BTS,
        )
        logits_BTV = self.output_projection(decoder_state_BTD)
        act_loss = self.config.act_loss_weight * (
            encoder_statistics["ponder_cost"] + decoder_statistics["ponder_cost"]
        )
        auxiliary: dict[str, object] = {
            "encoder": encoder_statistics,
            "decoder": decoder_statistics,
            "act_loss": act_loss,
        }
        return logits_BTV, auxiliary

    def compute_loss(
        self,
        logits_BTV: Tensor,
        labels_BT: Tensor,
        auxiliary: dict[str, object],
        *,
        ignore_index: int = -100,
    ) -> Tensor:
        """Combine autoregressive cross-entropy with the ACT ponder penalty."""
        prediction_loss = F.cross_entropy(
            logits_BTV.flatten(0, 1),
            labels_BT.flatten(),
            ignore_index=ignore_index,
        )
        act_loss = auxiliary["act_loss"]
        if not isinstance(act_loss, Tensor):
            raise TypeError("auxiliary['act_loss'] must be a tensor")
        return prediction_loss + act_loss

    @torch.no_grad()
    def generate(
        self,
        source_token_id_BS: Tensor,
        *,
        bos_token_id: int,
        eos_token_id: int,
        max_new_tokens: int,
        source_nonpadding_mask_BS: Tensor | None = None,
        source_attention_mask_BSS: Tensor | None = None,
    ) -> Tensor:
        """Greedily decode one target token at a time without a decoder cache."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        encoder_state_BSD, _ = self.encode(
            source_token_id_BS,
            source_nonpadding_mask_BS=source_nonpadding_mask_BS,
            source_attention_mask_BSS=source_attention_mask_BSS,
        )
        generated_token_id_BT = torch.full(
            (source_token_id_BS.shape[0], 1),
            bos_token_id,
            device=source_token_id_BS.device,
            dtype=torch.long,
        )
        finished_B = torch.zeros(
            source_token_id_BS.shape[0],
            device=source_token_id_BS.device,
            dtype=torch.bool,
        )
        for _ in range(max_new_tokens):
            decoder_state_BTD, _ = self.decode(
                generated_token_id_BT,
                encoder_state_BSD,
                source_nonpadding_mask_BS=source_nonpadding_mask_BS,
            )
            next_token_id_B = self.output_projection(decoder_state_BTD[:, -1]).argmax(
                dim=-1
            )
            next_token_id_B = torch.where(
                finished_B,
                torch.full_like(next_token_id_B, eos_token_id),
                next_token_id_B,
            )
            generated_token_id_BT = torch.cat(
                (generated_token_id_BT, next_token_id_B[:, None]),
                dim=1,
            )
            finished_B = finished_B | (next_token_id_B == eos_token_id)
            if finished_B.all().item():
                break
        return generated_token_id_BT

    @staticmethod
    def _nonpadding_mask(
        token_id_BL: Tensor,
        nonpadding_mask_BL: Tensor | None,
    ) -> Tensor:
        if token_id_BL.ndim != 2:
            raise ValueError("token IDs must have shape (batch, length)")
        return UniversalTransformer._state_nonpadding_mask(
            token_id_BL.shape[0],
            token_id_BL.shape[1],
            token_id_BL.device,
            nonpadding_mask_BL,
        )

    @staticmethod
    def _state_nonpadding_mask(
        batch_size: int,
        length: int,
        device: torch.device,
        nonpadding_mask_BL: Tensor | None,
    ) -> Tensor:
        if nonpadding_mask_BL is None:
            return torch.ones((batch_size, length), device=device, dtype=torch.bool)
        if nonpadding_mask_BL.shape != (batch_size, length):
            raise ValueError("nonpadding masks must have shape (batch, length)")
        return nonpadding_mask_BL.to(device=device, dtype=torch.bool)

    @staticmethod
    def _self_attention_mask(
        token_nonpadding_mask_BL: Tensor,
        *,
        causal: bool,
        supplied_mask_BLL: Tensor | None,
    ) -> Tensor:
        length = token_nonpadding_mask_BL.shape[1]
        attention_mask_BLL = (
            token_nonpadding_mask_BL[:, :, None] & token_nonpadding_mask_BL[:, None, :]
        )
        if causal:
            causal_mask_LL = torch.ones(
                (length, length),
                device=token_nonpadding_mask_BL.device,
                dtype=torch.bool,
            ).tril()
            attention_mask_BLL = attention_mask_BLL & causal_mask_LL[None, :, :]
        return UniversalTransformer._merge_attention_mask(
            attention_mask_BLL,
            supplied_mask_BLM=supplied_mask_BLL,
        )

    @staticmethod
    def _merge_attention_mask(
        base_mask_BLM: Tensor,
        supplied_mask_BLM: Tensor | None,
    ) -> Tensor:
        if supplied_mask_BLM is None:
            return base_mask_BLM
        if supplied_mask_BLM.shape != base_mask_BLM.shape:
            raise ValueError(
                f"attention mask must have shape {tuple(base_mask_BLM.shape)}"
            )
        return base_mask_BLM & supplied_mask_BLM.to(
            device=base_mask_BLM.device,
            dtype=torch.bool,
        )
