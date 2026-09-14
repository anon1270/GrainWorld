"""Shared PyTorch memory-encoding and decoder-construction components.

GrainWorld uses the detailed past-scene encoder below. The internal parent
decoder components are retained to preserve construction order and checkpoints.
The public model retains joint horizon and spatial interaction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .checkpoint import checkpoint as cp


def _check_sequence(name, value, channels):
    if value.ndim != 3 or value.shape[-1] != channels:
        raise ValueError(
            f"{name} must be [B,S,{channels}], got {tuple(value.shape)}"
        )


class _TokenFFN(nn.Module):
    """Token-wise FFN; the token/query axis is never reduced or mixed."""

    def __init__(self, channels, ratio, dropout):
        super().__init__()
        hidden = int(round(channels * ratio))
        self.layers = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, value):
        return self.layers(value)


class Q4OccContinuousTimeEmbedding(nn.Module):
    """Embed measured relative image time in seconds, not sweep ordinals."""

    def __init__(self, channels, num_bands=16):
        super().__init__()
        if channels < 1 or num_bands < 1:
            raise ValueError("channels and num_bands must be positive")
        frequencies = 2.0 ** torch.arange(num_bands, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.projection = nn.Sequential(
            nn.Linear(2 * num_bands + 1, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, relative_seconds):
        if relative_seconds.ndim != 2:
            raise ValueError(
                "relative_seconds must be [B,T], got "
                f"{tuple(relative_seconds.shape)}"
            )
        seconds = relative_seconds.to(dtype=torch.float32)
        phase = (
            seconds[..., None]
            * self.frequencies.to(device=seconds.device)[None, None, :]
            * (2.0 * math.pi)
        )
        encoded = torch.cat(
            [seconds[..., None], torch.sin(phase), torch.cos(phase)], dim=-1
        )
        # Keep the Fourier phase in FP32, then match the learned projection.
        # MMCV's legacy FP16 optimizer hook converts Linear weights to half
        # without wrapping this helper in autocast.
        encoded = encoded.to(dtype=self.projection[0].weight.dtype)
        return self.projection(encoded)


class Q4OccConditionedIndependentDecoderLayer(nn.Module):
    """Cross-attend fixed latents with condition separated from the residual.

    ``condition`` contains position/time/trajectory information.  It changes
    the attention query but is never added to the returned residual stream;
    therefore position cannot accumulate once per refinement stage.
    """

    def __init__(
        self,
        channels,
        num_heads,
        ffn_ratio,
        dropout,
        dual_memory=False,
    ):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.content_norm = nn.LayerNorm(channels)
        self.condition_norm = nn.LayerNorm(channels)
        self.latent_norm = nn.LayerNorm(channels)
        self.cross_attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.residual_dropout = nn.Dropout(dropout)
        self.dual_memory = bool(dual_memory)
        if self.dual_memory:
            # The local bank is indexed by scene and anchor.  Future queries
            # for the same anchor share projected keys/values, but each query
            # has its own cross-attention row and never reads another query.
            self.local_content_norm = nn.LayerNorm(channels)
            self.local_condition_norm = nn.LayerNorm(channels)
            self.local_memory_norm = nn.LayerNorm(channels)
            self.local_cross_attention = nn.MultiheadAttention(
                channels, num_heads, dropout=dropout, batch_first=True
            )
            self.local_residual_dropout = nn.Dropout(dropout)
            # Start as a small, active residual: it preserves the trained
            # global path's scale while allowing local MHA parameters to
            # receive gradients from the first optimizer step.
            self.local_memory_gate_logit = nn.Parameter(
                torch.tensor(-2.1972245773362196)
            )
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = _TokenFFN(channels, ffn_ratio, dropout)

    def _attend_anchor_local(
        self,
        content,
        condition,
        anchor_local_memory,
        anchor_local_padding_mask,
    ):
        if anchor_local_memory.ndim != 4:
            raise ValueError(
                "anchor_local_memory must be [B,N,L,C], got "
                f"{tuple(anchor_local_memory.shape)}"
            )
        batch, anchors, local_tokens, channels = anchor_local_memory.shape
        if channels != self.channels or local_tokens < 1:
            raise ValueError("anchor-local memory channel/token contract changed")
        if anchor_local_padding_mask.shape != (
            batch,
            anchors,
            local_tokens,
        ):
            raise ValueError(
                "anchor_local_padding_mask must be [B,N,L]"
            )
        if content.shape[1] != anchors or content.shape[0] % batch:
            raise ValueError(
                "content must be horizon-major [future*B,N,C] aligned "
                "with anchor-local memory"
            )

        future_count = content.shape[0] // batch
        # [F*B,N,C] -> [B*N,F,C].  Cross-attention does not mix the F query
        # rows; this layout only avoids projecting the same 192 K/V tokens F
        # separate times.
        local_query = self.local_content_norm(content)
        local_query = local_query + self.local_condition_norm(condition)
        local_query = local_query.reshape(
            future_count, batch, anchors, channels
        ).permute(1, 2, 0, 3).reshape(
            batch * anchors, future_count, channels
        )
        local_value = self.local_memory_norm(anchor_local_memory).reshape(
            batch * anchors, local_tokens, channels
        )
        padding = anchor_local_padding_mask.to(dtype=torch.bool).reshape(
            batch * anchors, local_tokens
        )
        all_invalid = padding.all(dim=1)
        # PyTorch attention returns NaN if every key is masked.  Expose a zero
        # dummy value without a tensor-to-Python synchronization, then zero
        # that row's update so an invisible anchor cannot acquire a learned
        # attention-projection prior.
        padding = padding.clone()
        padding[:, 0] = padding[:, 0] & ~all_invalid
        update, _ = self.local_cross_attention(
            local_query,
            local_value,
            local_value,
            key_padding_mask=padding,
            need_weights=False,
        )
        update = update.masked_fill(all_invalid[:, None, None], 0.0)
        update = update.reshape(
            batch, anchors, future_count, channels
        ).permute(2, 0, 1, 3).reshape(
            future_count * batch, anchors, channels
        )
        gate = torch.sigmoid(self.local_memory_gate_logit).to(content.dtype)
        return content + gate * self.local_residual_dropout(update)

    def forward(
        self,
        content,
        scene_latent,
        condition,
        anchor_local_memory=None,
        anchor_local_padding_mask=None,
    ):
        _check_sequence("content", content, self.channels)
        _check_sequence("scene_latent", scene_latent, self.channels)
        _check_sequence("condition", condition, self.channels)
        if content.shape != condition.shape:
            raise ValueError("content and condition must have identical shapes")
        if content.shape[0] != scene_latent.shape[0]:
            raise ValueError("content and scene_latent batches differ")
        attention_query = self.content_norm(content) + self.condition_norm(
            condition
        )
        latent = self.latent_norm(scene_latent)
        update, _ = self.cross_attention(
            attention_query,
            latent,
            latent,
            need_weights=False,
        )
        content = content + self.residual_dropout(update)
        local_arguments = (
            anchor_local_memory is not None,
            anchor_local_padding_mask is not None,
        )
        if self.dual_memory:
            if not all(local_arguments):
                raise ValueError(
                    "dual-memory decoder requires local memory and its mask"
                )
            content = self._attend_anchor_local(
                content,
                condition,
                anchor_local_memory,
                anchor_local_padding_mask,
            )
        elif any(local_arguments):
            raise ValueError(
                "local memory was supplied to a global-only decoder"
            )
        return content + self.ffn(self.ffn_norm(content))


class _MaskedTemporalBlock(nn.Module):
    """Past-token self-attention inside one spatial anchor only."""

    def __init__(self, channels, num_heads, ffn_ratio, dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = _TokenFFN(channels, ffn_ratio, dropout)

    def forward(self, tokens, key_padding_mask):
        normalized = self.attn_norm(tokens)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.dropout(update)
        return tokens + self.ffn(self.ffn_norm(tokens))


class _MaskedSceneLatentBlock(nn.Module):
    """Masked context cross-attention followed by latent self-attention."""

    def __init__(self, channels, num_heads, ffn_ratio, dropout):
        super().__init__()
        self.latent_cross_norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(channels)
        self.cross_attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.latent_self_norm = nn.LayerNorm(channels)
        self.self_attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = _TokenFFN(channels, ffn_ratio, dropout)

    def forward(self, latents, context, context_padding_mask):
        latent_query = self.latent_cross_norm(latents)
        context_value = self.context_norm(context)
        update, _ = self.cross_attention(
            latent_query,
            context_value,
            context_value,
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        latents = latents + self.dropout(update)
        normalized = self.latent_self_norm(latents)
        update, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        latents = latents + self.dropout(update)
        return latents + self.ffn(self.ffn_norm(latents))


class Q4OccDetailedPastSceneEncoder(nn.Module):
    """Preserve every deformable point before causal latent compression.

    Input samples have shape ``[B,N,Tpast,P,C]``.  No point mean is taken.
    Attention is first restricted to the ``Tpast*P`` tokens of each anchor,
    then all valid detail and anchor-summary tokens are exposed to the scene
    latents.  Invalid image samples are masked instead of becoming positional
    prior tokens.
    """

    def __init__(
        self,
        channels,
        num_points,
        num_latents=256,
        num_heads=8,
        temporal_layers=2,
        latent_layers=6,
        ffn_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if min(num_points, num_latents, temporal_layers, latent_layers) < 1:
            raise ValueError("point/latent/layer counts must be positive")
        self.channels = channels
        self.num_points = num_points
        self.num_latents = num_latents
        self.sensor_norm = nn.LayerNorm(channels)
        self.point_embedding = nn.Parameter(
            torch.empty(1, 1, 1, num_points, channels)
        )
        self.summary_seed = nn.Parameter(torch.zeros(1, 1, channels))
        self.global_context_token = nn.Parameter(
            torch.empty(1, 1, channels)
        )
        self.latent_tokens = nn.Parameter(
            torch.empty(1, num_latents, channels)
        )
        self.temporal_blocks = nn.ModuleList(
            [
                _MaskedTemporalBlock(
                    channels, num_heads, ffn_ratio, dropout
                )
                for _ in range(temporal_layers)
            ]
        )
        self.latent_blocks = nn.ModuleList(
            [
                _MaskedSceneLatentBlock(
                    channels, num_heads, ffn_ratio, dropout
                )
                for _ in range(latent_layers)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.point_embedding, std=0.02)
        nn.init.normal_(self.global_context_token, std=0.02)
        nn.init.normal_(self.latent_tokens, std=0.02)

    def encode_with_detail(
        self,
        sensor_samples,
        valid_mask,
        anchor_position,
        past_time_embedding,
        detail_position=None,
        compute_global=True,
    ):
        if sensor_samples.ndim != 5:
            raise ValueError("sensor_samples must be [B,N,T,P,C]")
        batch, anchors, frames, points, channels = sensor_samples.shape
        if channels != self.channels or points != self.num_points:
            raise ValueError("sensor sample channel/point contract changed")
        if valid_mask.shape != sensor_samples.shape[:-1]:
            raise ValueError("valid_mask must be [B,N,T,P]")
        if anchor_position.shape != (batch, anchors, channels):
            raise ValueError("anchor_position must be [B,N,C]")
        if past_time_embedding.shape != (batch, frames, channels):
            raise ValueError("past_time_embedding must be [B,T,C]")
        if detail_position is None:
            detail_position = anchor_position[:, :, None, :].expand(
                batch, anchors, points, channels
            )
        if detail_position.shape != (batch, anchors, points, channels):
            raise ValueError("detail_position must be [B,N,P,C]")
        if type(compute_global) is not bool:
            raise TypeError("compute_global must be bool")

        valid_mask = valid_mask.to(dtype=torch.bool)
        tokens = self.sensor_norm(sensor_samples)
        tokens = (
            tokens
            + detail_position[:, :, None, :, :]
            + past_time_embedding[:, None, :, None, :]
            + self.point_embedding
        )
        tokens = tokens * valid_mask[..., None].to(tokens.dtype)

        detail_count = frames * points
        detail = tokens.reshape(batch * anchors, detail_count, channels)
        detail_valid = valid_mask.reshape(batch * anchors, detail_count)
        anchor_valid = detail_valid.any(dim=1)
        summary = self.summary_seed.expand(batch * anchors, -1, -1)
        summary = summary + anchor_position.reshape(
            batch * anchors, 1, channels
        )
        sequence = torch.cat([summary, detail], dim=1)
        padding = torch.cat(
            [
                torch.zeros(
                    batch * anchors,
                    1,
                    dtype=torch.bool,
                    device=tokens.device,
                ),
                ~detail_valid,
            ],
            dim=1,
        )
        for block in self.temporal_blocks:
            if self.training and sequence.requires_grad:
                sequence = cp(
                    block, sequence, padding, use_reentrant=False
                )
            else:
                sequence = block(sequence, padding)

        summary = sequence[:, 0].reshape(batch, anchors, channels)
        summary = summary * anchor_valid.reshape(
            batch, anchors, 1
        ).to(summary.dtype)
        detail = sequence[:, 1:].reshape(
            batch, anchors, detail_count, channels
        )
        detail = detail * detail_valid.reshape(
            batch, anchors, detail_count, 1
        ).to(detail.dtype)
        local_padding = ~detail_valid.reshape(
            batch, anchors, detail_count
        )
        if not compute_global:
            return None, detail, local_padding
        context = torch.cat([summary[:, :, None, :], detail], dim=2)
        context = context.reshape(batch, anchors * (detail_count + 1), channels)
        context_valid = torch.cat(
            [
                anchor_valid.reshape(batch, anchors, 1),
                detail_valid.reshape(batch, anchors, detail_count),
            ],
            dim=2,
        ).reshape(batch, anchors * (detail_count + 1))

        global_token = self.global_context_token.expand(batch, -1, -1)
        context = torch.cat([global_token, context], dim=1)
        context_padding = torch.cat(
            [
                torch.zeros(
                    batch, 1, dtype=torch.bool, device=context.device
                ),
                ~context_valid,
            ],
            dim=1,
        )
        latents = self.latent_tokens.expand(batch, -1, -1)
        for block in self.latent_blocks:
            if self.training and latents.requires_grad:
                latents = cp(
                    block,
                    latents,
                    context,
                    context_padding,
                    use_reentrant=False,
                )
            else:
                latents = block(latents, context, context_padding)
        expected = (batch, self.num_latents, channels)
        if tuple(latents.shape) != expected:
            raise AssertionError(
                f"scene latent shape {tuple(latents.shape)} != {expected}"
            )
        return latents, detail, local_padding

    def forward(
        self,
        sensor_samples,
        valid_mask,
        anchor_position,
        past_time_embedding,
        detail_position=None,
    ):
        latents, _, _ = self.encode_with_detail(
            sensor_samples,
            valid_mask,
            anchor_position,
            past_time_embedding,
            detail_position=detail_position,
        )
        return latents
