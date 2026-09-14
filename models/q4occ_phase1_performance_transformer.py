"""Parent helpers used during construction of the GrainWorld joint decoder.

GrainWorld replaces the parent decoder layers with DualContextJointDecoderLayer.
No independent-query training configuration is included in this release."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import ModuleList
from mmdet.models.utils.builder import TRANSFORMER

from .bbox.utils import decode_points, encode_points
from .csrc.wrapper import MSMV_CUDA
from .q4occ_phase1_performance_modules import (
    Q4OccConditionedIndependentDecoderLayer,
    Q4OccContinuousTimeEmbedding,
    Q4OccDetailedPastSceneEncoder,
)
from .sparse_world_transformer import (
    SparseWorldTransformer,
    SparseWorldTransformerDecoder,
    SparseWorldTransformerDecoderLayer,
)
from .utils import DUMP


class Q4OccPerformanceSceneState(NamedTuple):
    """Trajectory-independent cached state for one observed scene."""

    scene_latent: torch.Tensor
    anchor_points: torch.Tensor
    anchor_features: torch.Tensor
    prepared_features: Tuple[torch.Tensor, ...]
    occ2img: torch.Tensor
    img_metas: Sequence[Mapping[str, Any]]
    relative_past_seconds: torch.Tensor
    past_time_embedding: torch.Tensor
    anchor_local_detail: Optional[torch.Tensor] = None
    anchor_local_padding_mask: Optional[torch.Tensor] = None


def _performance_options(raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    options = dict(raw or {})
    enabled = bool(options.get("enabled", False))
    phase = str(options.get("phase", "phase1_performance"))
    if enabled and phase != "phase1_performance":
        raise ValueError(
            "Q4OccPhase1PerformanceTransformer requires "
            f"phase='phase1_performance', got {phase!r}"
        )
    if enabled and bool(options.get("query_self_attention", False)):
        raise ValueError("output-query self-attention is forbidden")
    if enabled and float(options.get("decode_fraction", 1.0)) != 1.0:
        raise ValueError("all output queries must be decoded in Phase 1")
    extent = options.get(
        "sensor_anchor_extent", [0.00625, 0.00625, 0.015625]
    )
    if isinstance(extent, (int, float)):
        extent = [float(extent)] * 3
    extent = [float(value) for value in extent]
    if len(extent) != 3 or any(value <= 0.0 or value >= 0.5 for value in extent):
        raise ValueError("sensor_anchor_extent must be three values in (0, .5)")
    normalized = {
        "enabled": enabled,
        "phase": phase,
        "num_latents": int(options.get("num_latents", 256)),
        "temporal_layers": int(options.get("temporal_layers", 2)),
        "latent_layers": int(options.get("latent_layers", 6)),
        "decoder_layers": int(options.get("decoder_layers", 6)),
        "num_heads": int(options.get("num_heads", 8)),
        "ffn_ratio": float(options.get("ffn_ratio", 4.0)),
        "dropout": float(options.get("dropout", 0.1)),
        "dual_memory": bool(options.get("dual_memory", False)),
        "local_memory_tokens": int(options.get("local_memory_tokens", 0)),
        "query_self_attention": False,
        "decode_fraction": 1.0,
        "sensor_anchor_extent": extent,
        "valid_feature_epsilon": float(
            options.get("valid_feature_epsilon", 1.0e-8)
        ),
        # Scale only the backward signal crossing a cached-scene geometry
        # refinement boundary.  The forward point values are bit-identical to
        # the detached path.  Zero preserves the earlier K512/dual behaviour;
        # the new experiment configs select a conservative nonzero value.
        "memory_geometry_grad_scale": float(
            options.get("memory_geometry_grad_scale", 0.0)
        ),
    }
    for name in (
        "num_latents",
        "temporal_layers",
        "latent_layers",
        "decoder_layers",
        "num_heads",
    ):
        if enabled and normalized[name] < 1:
            raise ValueError(f"{name} must be positive")
    if enabled and normalized["dual_memory"]:
        if normalized["local_memory_tokens"] < 1:
            raise ValueError(
                "dual_memory requires a positive local_memory_tokens contract"
            )
    elif normalized["local_memory_tokens"] != 0:
        raise ValueError(
            "local_memory_tokens must be zero when dual_memory is disabled"
        )
    if not 0.0 <= normalized["memory_geometry_grad_scale"] <= 1.0:
        raise ValueError("memory_geometry_grad_scale must be inside [0, 1]")
    return normalized


@TRANSFORMER.register_module()
class Q4OccPhase1PerformanceTransformer(SparseWorldTransformer):
    """SparseWorld-compatible corrected Phase-1 wrapper."""

    def __init__(
        self,
        embed_dims,
        num_frames=8,
        future_frames=None,
        num_views=6,
        num_points=4,
        num_layers=6,
        num_levels=4,
        num_classes=10,
        num_groups=4,
        num_refines=None,
        scales=None,
        pc_range=None,
        q4occ=None,
        init_cfg=None,
    ):
        future_frames = [] if future_frames is None else future_frames
        num_refines = [1, 4, 16, 32, 64, 128] if num_refines is None else num_refines
        scales = [1.0] if scales is None else scales
        pc_range = [] if pc_range is None else pc_range
        options = _performance_options(q4occ)
        if options["enabled"] and options["decoder_layers"] != num_layers:
            raise ValueError(
                "one independent decoder is required for every released "
                f"refinement stage ({num_layers})"
            )
        super().__init__(
            embed_dims=embed_dims,
            num_frames=num_frames,
            future_frames=future_frames,
            num_views=num_views,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=num_classes,
            num_groups=num_groups,
            num_refines=num_refines,
            scales=scales,
            pc_range=pc_range,
            init_cfg=init_cfg,
        )
        self.q4occ = options
        if options["enabled"]:
            released_decoder = self.decoder
            # Q4-only construction must not change common SparseWorld/head RNG.
            with torch.random.fork_rng(devices=[]):
                corrected_decoder = Q4OccPhase1PerformanceDecoder(
                    embed_dims=embed_dims,
                    num_frames=num_frames,
                    future_frames=future_frames,
                    num_views=num_views,
                    num_points=num_points,
                    num_layers=num_layers,
                    num_levels=num_levels,
                    num_classes=num_classes,
                    num_refines=num_refines,
                    num_groups=num_groups,
                    scales=scales,
                    pc_range=pc_range,
                    q4occ=options,
                )
            incompatible = corrected_decoder.load_state_dict(
                released_decoder.state_dict(), strict=False
            )
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    "corrected decoder rejected released state keys: "
                    f"{incompatible.unexpected_keys}"
                )
            changed_common = [
                key for key in incompatible.missing_keys if "q4occ_" not in key
            ]
            if changed_common:
                raise RuntimeError(
                    "corrected decoder changed released state keys: "
                    f"{changed_common}"
                )
            self.decoder = corrected_decoder

    def encode_scene(self, query_points, query_feat, mlvl_feats, img_metas):
        if not self.q4occ["enabled"]:
            raise RuntimeError("encode_scene requires q4occ.enabled=True")
        return self.decoder.encode_scene(
            query_points, query_feat, mlvl_feats, img_metas
        )

    def decode_queries(self, scene_state, fut2cur, fut_list):
        if not self.q4occ["enabled"]:
            raise RuntimeError("decode_queries requires q4occ.enabled=True")
        scores, points = self.decoder.decode_queries(
            scene_state, fut2cur, fut_list
        )
        return (
            [torch.nan_to_num(value) for value in scores],
            [torch.nan_to_num(value) for value in points],
        )

    def forward(
        self,
        query_points,
        query_feat,
        mlvl_feats,
        img_metas,
        fut2cur,
        fut_list,
    ):
        if not self.q4occ["enabled"]:
            return super().forward(
                query_points,
                query_feat,
                mlvl_feats,
                img_metas,
                fut2cur,
                fut_list,
            )
        state = self.encode_scene(
            query_points, query_feat, mlvl_feats, img_metas
        )
        return self.decode_queries(state, fut2cur, fut_list)


class Q4OccPhase1PerformanceDecoder(SparseWorldTransformerDecoder):
    """Detailed past encoder and six-stage independent local refinement."""

    def __init__(
        self,
        embed_dims,
        num_frames=8,
        future_frames=None,
        num_views=6,
        num_points=4,
        num_layers=6,
        num_levels=4,
        num_classes=10,
        num_refines=16,
        num_groups=4,
        scales=None,
        pc_range=None,
        q4occ=None,
        init_cfg=None,
    ):
        future_frames = [] if future_frames is None else future_frames
        scales = [1.0] if scales is None else scales
        pc_range = [] if pc_range is None else pc_range
        options = _performance_options(q4occ)
        super().__init__(
            embed_dims=embed_dims,
            num_frames=num_frames,
            future_frames=future_frames,
            num_views=num_views,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=num_classes,
            num_refines=num_refines,
            num_groups=num_groups,
            scales=scales,
            pc_range=pc_range,
            init_cfg=init_cfg,
        )
        if len(scales) == 1:
            scales = scales * num_layers
        if not isinstance(num_refines, list):
            num_refines = [num_refines]
        if len(num_refines) == 1:
            num_refines = num_refines * num_layers
        last_refines = [1] + num_refines

        self.q4occ = options
        self.num_points = num_points
        self.num_levels = num_levels
        self.q4occ_local_memory_tokens = (
            num_frames * num_points * num_layers
        )
        if (
            options["dual_memory"]
            and options["local_memory_tokens"]
            != self.q4occ_local_memory_tokens
        ):
            raise ValueError(
                "configured local_memory_tokens does not match the complete "
                "past sensor detail bank: expected "
                f"{self.q4occ_local_memory_tokens}, got "
                f"{options['local_memory_tokens']}"
            )
        self.decoder_layers = ModuleList(
            [
                Q4OccPhase1PerformanceDecoderLayer(
                    embed_dims=embed_dims,
                    num_frames=num_frames,
                    future_frames=future_frames,
                    num_views=num_views,
                    num_points=num_points,
                    num_levels=num_levels,
                    num_classes=num_classes,
                    num_groups=num_groups,
                    num_refines=num_refines[index],
                    last_refines=last_refines[index],
                    layer_idx=index,
                    scale=scales[index],
                    pc_range=pc_range,
                    q4occ=options,
                )
                for index in range(num_layers)
            ]
        )
        self.q4occ_time_encoder = Q4OccContinuousTimeEmbedding(embed_dims)
        self.q4occ_scene_encoder = Q4OccDetailedPastSceneEncoder(
            channels=embed_dims,
            # Every released refinement stage contributes its complete P-axis
            # patch.  Nothing is averaged before the global latent encoder.
            num_points=num_points * num_layers,
            num_latents=options["num_latents"],
            num_heads=options["num_heads"],
            temporal_layers=options["temporal_layers"],
            latent_layers=options["latent_layers"],
            ffn_ratio=options["ffn_ratio"],
            dropout=options["dropout"],
        )
        self.q4occ_trajectory_encoder = nn.Sequential(
            nn.Linear(17, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, embed_dims),
        )
        self.register_buffer(
            "q4occ_sensor_anchor_extent",
            torch.tensor(options["sensor_anchor_extent"], dtype=torch.float32),
        )

    @staticmethod
    def _sinusoidal_embedding(frames, channels, dtype):
        frames = frames.to(dtype=torch.float32)
        half = channels // 2
        frequencies = torch.exp(
            torch.arange(half, device=frames.device, dtype=torch.float32)
            * -(math.log(10000.0) / max(half - 1, 1))
        )
        phase = frames[:, None] * frequencies[None, :]
        result = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if channels % 2:
            result = torch.cat([result, result.new_zeros(len(frames), 1)], dim=-1)
        return result.to(dtype=dtype)

    @staticmethod
    def _normalize_future_frames(fut_list, batch_size, device):
        if not isinstance(fut_list, (list, tuple)) or not fut_list:
            raise ValueError("fut_list must be a non-empty sequence")
        values = []
        for index, raw in enumerate(fut_list):
            value = torch.as_tensor(raw, device=device).reshape(-1)
            if value.numel() not in (1, batch_size):
                raise ValueError(
                    f"fut_list[{index}] has {value.numel()} values, expected "
                    f"1 or {batch_size}"
                )
            if value.numel() == batch_size and not torch.equal(
                value, value[:1].expand_as(value)
            ):
                raise ValueError("future horizons must match across a batch")
            values.append(value[0].float())
        return torch.stack(values)

    def _prepare_sensor_features(self, mlvl_feats, batch_size):
        prepared = []
        for level, feature in enumerate(mlvl_feats):
            if feature.ndim != 5:
                raise ValueError(
                    f"mlvl_feats[{level}] must be [B,TN,C,H,W]"
                )
            batch, time_views, grouped_channels, height, width = feature.shape
            if batch != batch_size:
                raise ValueError("feature and anchor batch sizes differ")
            if time_views != self.num_frames * self.num_views:
                raise ValueError("scene encoder received non-past image features")
            if grouped_channels % self.num_groups:
                raise ValueError("channels must divide num_groups")
            channels = grouped_channels // self.num_groups
            grouped = feature.reshape(
                batch,
                self.num_frames,
                self.num_views,
                self.num_groups,
                channels,
                height,
                width,
            )
            if MSMV_CUDA:
                grouped = grouped.permute(0, 1, 3, 2, 5, 6, 4)
                grouped = grouped.reshape(
                    batch * self.num_frames * self.num_groups,
                    self.num_views,
                    height,
                    width,
                    channels,
                )
            else:
                grouped = grouped.permute(0, 1, 3, 4, 2, 5, 6)
                grouped = grouped.reshape(
                    batch * self.num_frames * self.num_groups,
                    channels,
                    self.num_views,
                    height,
                    width,
                )
            prepared.append(grouped.contiguous())
        return tuple(prepared)

    def _build_occ2img(self, query_feat, img_metas):
        lidar2img = query_feat.new_tensor(
            np.asarray([meta["lidar2img"] for meta in img_metas], dtype=np.float32)
        )
        ego2lidar = query_feat.new_tensor(
            np.asarray([meta["ego2lidar"] for meta in img_metas], dtype=np.float32)
        ).unsqueeze(1)
        if lidar2img.shape[1] != self.num_frames * self.num_views:
            raise ValueError("projection metadata contains a future/non-past image")
        return torch.matmul(lidar2img, ego2lidar.expand_as(lidar2img))

    def _relative_past_seconds(self, query_feat, img_metas):
        relative = []
        for batch_index, meta in enumerate(img_metas):
            if "img_timestamp" not in meta:
                raise ValueError(
                    "actual img_timestamp is required; ordinal fallback is forbidden"
                )
            timestamps = np.asarray(meta["img_timestamp"], dtype=np.float64)
            expected = self.num_frames * self.num_views
            if timestamps.size != expected:
                raise ValueError(
                    f"img_metas[{batch_index}] has {timestamps.size} timestamps; "
                    f"expected {expected}"
                )
            grouped = timestamps.reshape(self.num_frames, self.num_views)
            if np.any(grouped[1:] > grouped[:1] + 1.0e-3):
                raise ValueError("Future image sweep detected in scene encoder")
            frame_time = np.median(grouped, axis=1)
            frame_time = frame_time - frame_time[0]
            frame_time[0] = 0.0
            if np.any(frame_time > 1.0e-6):
                raise ValueError("past relative timestamps must be non-positive")
            relative.append(frame_time.astype(np.float32))
        # Keep measured seconds in FP32 even when MMCV converts model weights
        # and query features to FP16. Fourier phase precision is handled by the
        # time encoder before matching its learned projection dtype.
        return torch.as_tensor(
            np.stack(relative, axis=0),
            device=query_feat.device,
            dtype=torch.float32,
        )

    def _sampling_support(self, query_points):
        if query_points.shape[2] != 1:
            return query_points
        extent = self.q4occ_sensor_anchor_extent.to(
            device=query_points.device, dtype=query_points.dtype
        ).view(1, 1, 1, 3)
        # Deliberately do not clamp: clamping shifts boundary-anchor centers.
        return torch.cat(
            [query_points - extent, query_points + extent], dim=2
        )

    def _detailed_sensor_samples(self, sampled):
        batch, anchors, groups, frame_points, group_channels = sampled.shape
        if groups != self.num_groups:
            raise AssertionError("sample group contract changed")
        if frame_points != self.num_frames * self.num_points:
            raise AssertionError("sample point/time contract changed")
        if groups * group_channels != self.embed_dims:
            raise AssertionError("sample channels do not reconstruct embed_dims")
        detailed = sampled.reshape(
            batch,
            anchors,
            groups,
            self.num_frames,
            self.num_points,
            group_channels,
        ).permute(0, 1, 3, 4, 2, 5)
        return detailed.contiguous().reshape(
            batch,
            anchors,
            self.num_frames,
            self.num_points,
            self.embed_dims,
        )

    @staticmethod
    def _scale_geometry_gradient(points, scale):
        """Preserve point values while scaling the inter-stage gradient."""

        scale = float(scale)
        if scale == 0.0:
            # Preserve the earlier graph cut and its activation-memory cost.
            return points.detach()
        if scale == 1.0:
            return points
        detached = points.detach()
        return detached + scale * (points - detached)

    def encode_scene(self, query_points, query_feat, mlvl_feats, img_metas):
        if query_points.ndim != 4 or query_points.shape[2:] != (1, 3):
            raise ValueError("initial query_points must be [B,Q,1,3]")
        if query_feat.ndim != 3 or query_feat.shape[:2] != query_points.shape[:2]:
            raise ValueError("query_feat must align with initial anchors")
        batch, anchors, _ = query_feat.shape
        if len(img_metas) != batch:
            raise ValueError("img_metas batch differs from anchors")

        relative_seconds = self._relative_past_seconds(query_feat, img_metas)
        past_time = self.q4occ_time_encoder(relative_seconds).to(query_feat.dtype)
        prepared_features = self._prepare_sensor_features(mlvl_feats, batch)
        occ2img = self._build_occ2img(query_feat, img_metas)

        identity = torch.eye(
            4, device=query_feat.device, dtype=query_feat.dtype
        ).unsqueeze(0).expand(batch, -1, -1).contiguous()
        # A strong D4RT-style encoder is essential: run the released six-stage
        # sensor tower with an identity trajectory, preserving every sampled
        # patch before latent compression.  This is still strictly past-only.
        scene_points = query_points.detach()
        scene_content = query_feat
        detail_stages, valid_stages, position_stages = [], [], []
        anchor_position = None
        layer_count = len(self.decoder_layers)
        for stage_index, decoder_layer in enumerate(self.decoder_layers):
            DUMP.stage_count = stage_index
            if stage_index:
                scene_points = self._scale_geometry_gradient(
                    scene_points,
                    self.q4occ["memory_geometry_grad_scale"],
                )
            anchor_position = decoder_layer.position_encoder(
                scene_points.flatten(2, 3)
            )
            sensor_query = scene_content + anchor_position
            sensor_points = self._sampling_support(scene_points)
            # The released sampler's public positional names are swapped.
            # (occ2img, img_metas, fut2cur) is the released caller contract.
            sampled = decoder_layer.sampling(
                sensor_points,
                sensor_query,
                prepared_features,
                occ2img,
                img_metas,
                [identity],
            )
            # Visibility is derived from raw sensor evidence.  A learned time
            # modulation must never be able to turn a visible token into an
            # invalid token by saturating to zero in FP16.
            raw_detailed_stage = self._detailed_sensor_samples(sampled)
            valid_stage = (
                raw_detailed_stage.detach().float().abs().sum(dim=-1)
                > self.q4occ["valid_feature_epsilon"]
            )
            sampled = decoder_layer.modulate_past_time(sampled, past_time)
            detailed_stage = self._detailed_sensor_samples(sampled)
            detail_stages.append(detailed_stage)
            valid_stages.append(valid_stage)
            position_stages.append(
                anchor_position[:, :, None, :].expand(
                    batch, anchors, self.num_points, self.embed_dims
                )
            )
            local_sensor = decoder_layer.sensor_only_local(
                sampled, sensor_query
            )
            scene_content = decoder_layer.norm1(
                scene_content
                + decoder_layer.q4occ_sensor_gate * local_sensor
            )
            # Only stages 0--4 produce a point set consumed by a later memory
            # sampling stage.  Computing stage 5's offset here cannot affect
            # the cached memory and would create a misleading dead branch.
            if stage_index + 1 < layer_count:
                scene_offset = (
                    decoder_layer.scale
                    * decoder_layer.reg_branch(scene_content)
                )
                scene_points = decoder_layer.refine_points(
                    scene_points, scene_offset
                )

        detailed = torch.cat(detail_stages, dim=3)
        valid = torch.cat(valid_stages, dim=3)
        detail_position = torch.cat(position_stages, dim=2)
        if self.q4occ["dual_memory"]:
            (
                scene_latent,
                anchor_local_detail,
                anchor_local_padding_mask,
            ) = self.q4occ_scene_encoder.encode_with_detail(
                detailed,
                valid,
                anchor_position,
                past_time,
                detail_position=detail_position,
            )
            expected_detail = (
                batch,
                anchors,
                self.q4occ_local_memory_tokens,
                self.embed_dims,
            )
            expected_padding = expected_detail[:-1]
            if tuple(anchor_local_detail.shape) != expected_detail:
                raise AssertionError(
                    "anchor-local detail shape changed: "
                    f"{tuple(anchor_local_detail.shape)} != {expected_detail}"
                )
            if tuple(anchor_local_padding_mask.shape) != expected_padding:
                raise AssertionError(
                    "anchor-local padding shape changed: "
                    f"{tuple(anchor_local_padding_mask.shape)} != "
                    f"{expected_padding}"
                )
        else:
            scene_latent = self.q4occ_scene_encoder(
                detailed,
                valid,
                anchor_position,
                past_time,
                detail_position=detail_position,
            )
            anchor_local_detail = None
            anchor_local_padding_mask = None
        return Q4OccPerformanceSceneState(
            scene_latent=scene_latent,
            anchor_points=query_points,
            anchor_features=query_feat,
            prepared_features=prepared_features,
            occ2img=occ2img,
            img_metas=img_metas,
            relative_past_seconds=relative_seconds,
            past_time_embedding=past_time,
            anchor_local_detail=anchor_local_detail,
            anchor_local_padding_mask=anchor_local_padding_mask,
        )

    def decode_queries(self, scene_state, fut2cur, fut_list):
        if not isinstance(scene_state, Q4OccPerformanceSceneState):
            raise TypeError("scene_state must be Q4OccPerformanceSceneState")
        scene_latent = scene_state.scene_latent
        query_points = scene_state.anchor_points
        query_feat = scene_state.anchor_features
        batch, anchors, channels = query_feat.shape
        future_count = len(fut2cur)
        if future_count < 1 or len(fut_list) != future_count:
            raise ValueError("fut2cur/fut_list horizon counts differ or are empty")
        if scene_latent.shape[0] != batch:
            raise ValueError("scene latent batch differs from anchors")
        if self.q4occ["dual_memory"]:
            local_detail = scene_state.anchor_local_detail
            local_padding = scene_state.anchor_local_padding_mask
            if local_detail is None or local_padding is None:
                raise ValueError("dual-memory scene state is missing local detail")
            expected_detail = (
                batch,
                anchors,
                self.q4occ_local_memory_tokens,
                channels,
            )
            if tuple(local_detail.shape) != expected_detail:
                raise ValueError(
                    "anchor-local detail does not align with output anchors"
                )
            if tuple(local_padding.shape) != expected_detail[:-1]:
                raise ValueError(
                    "anchor-local padding does not align with output anchors"
                )
        else:
            local_detail = None
            local_padding = None

        frames = self._normalize_future_frames(
            fut_list, batch, query_feat.device
        )
        future_time = self._sinusoidal_embedding(
            frames, channels, query_feat.dtype
        )
        future_time = future_time[:, None, None, :].expand(
            future_count, batch, anchors, channels
        ).reshape(future_count * batch, anchors, channels)
        query_points = query_points.unsqueeze(0).expand(
            future_count, -1, -1, -1, -1
        ).reshape(future_count * batch, anchors, 1, 3)
        content = query_feat.unsqueeze(0).expand(
            future_count, -1, -1, -1
        ).reshape(future_count * batch, anchors, channels)
        content = content + future_time

        matrices = []
        for index, raw_matrix in enumerate(fut2cur):
            matrix = torch.as_tensor(
                raw_matrix, device=query_feat.device, dtype=query_feat.dtype
            )
            if tuple(matrix.shape) != (batch, 4, 4):
                raise ValueError(
                    f"fut2cur[{index}] must be [B,4,4], got {tuple(matrix.shape)}"
                )
            matrices.append(matrix)
        trajectory_matrix = torch.stack(matrices, dim=0)
        flat_matrix = trajectory_matrix.reshape(future_count * batch, 16)
        expanded_matrix = flat_matrix[:, None, :].expand(-1, anchors, -1)
        content = self.pe_mln(content, expanded_matrix)
        frame_scalar = frames[:, None].expand(future_count, batch).reshape(
            future_count * batch, 1
        )
        trajectory_input = torch.cat([flat_matrix.float(), frame_scalar], dim=-1)
        trajectory_condition = self.q4occ_trajectory_encoder(
            trajectory_input.to(query_feat.dtype)
        )[:, None, :].expand(-1, anchors, -1)
        base_condition = future_time + trajectory_condition

        expanded_latent = scene_latent.unsqueeze(0).expand(
            future_count, -1, -1, -1
        ).reshape(future_count * batch, scene_latent.shape[1], channels)
        expanded_occ2img = scene_state.occ2img.repeat(
            future_count, 1, 1, 1
        )
        expanded_past_time = scene_state.past_time_embedding.unsqueeze(0).expand(
            future_count, -1, -1, -1
        ).reshape(future_count * batch, self.num_frames, channels)

        cls_scores, refine_points = [], []
        for index, decoder_layer in enumerate(self.decoder_layers):
            DUMP.stage_count = index
            query_points = query_points.detach()
            content, cls_score, query_points = decoder_layer(
                query_points=query_points,
                query_content=content,
                scene_latent=expanded_latent,
                base_condition=base_condition,
                mlvl_feats=scene_state.prepared_features,
                occ2img=expanded_occ2img,
                img_metas=scene_state.img_metas,
                fut2cur=matrices,
                past_time_embedding=expanded_past_time,
                sensor_anchor_extent=self.q4occ_sensor_anchor_extent,
                anchor_local_memory=local_detail,
                anchor_local_padding_mask=local_padding,
            )
            cls_scores.append(cls_score)
            refine_points.append(query_points)
        return cls_scores, refine_points

    def forward(
        self,
        query_points,
        query_feat,
        mlvl_feats,
        img_metas,
        fut2cur,
        fut_list,
    ):
        state = self.encode_scene(
            query_points, query_feat, mlvl_feats, img_metas
        )
        return self.decode_queries(state, fut2cur, fut_list)


class Q4OccPhase1PerformanceDecoderLayer(SparseWorldTransformerDecoderLayer):
    """One baseline-like sensor refinement plus independent latent decode."""

    def __init__(self, *args, q4occ=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.q4occ = _performance_options(q4occ)
        self.q4occ_query_decoder = Q4OccConditionedIndependentDecoderLayer(
            channels=self.embed_dims,
            num_heads=self.q4occ["num_heads"],
            ffn_ratio=self.q4occ["ffn_ratio"],
            dropout=self.q4occ["dropout"],
            dual_memory=self.q4occ["dual_memory"],
        )
        self.q4occ_past_time_scale = nn.Linear(
            self.embed_dims, self.embed_dims
        )
        # Start as the released sensor path (multiplicative scale exactly 1)
        # and learn physical-time modulation from there.
        nn.init.zeros_(self.q4occ_past_time_scale.weight)
        nn.init.zeros_(self.q4occ_past_time_scale.bias)
        self.q4occ_sensor_gate = nn.Parameter(torch.ones(1))
        # The released AdaptiveMixing output bias is a query-independent
        # prior.  The corrected local branch deliberately omits it so a zero
        # sensor tensor produces an exactly zero local update.  Keep the state
        # key for baseline/full-checkpoint compatibility, but do not train an
        # unused parameter under find_unused_parameters=False.
        if self.mixing.out_proj.bias is not None:
            self.mixing.out_proj.bias.requires_grad = False
        # Sampling, AdaptiveMixing, norm1, position encoder and released heads
        # remain active in all six stages. Only output-query interactions and
        # their now-unused post norms/FFN are frozen.
        for module in (
            self.self_attn,
            self.ffn,
            self.norm2,
            self.norm3,
            self.tempo_attn,
            self.ln,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _sampling_support(query_points, extent):
        if query_points.shape[2] != 1:
            return query_points
        extent = extent.to(
            device=query_points.device, dtype=query_points.dtype
        ).view(1, 1, 1, 3)
        return torch.cat([query_points - extent, query_points + extent], dim=2)

    def modulate_past_time(self, sampled, past_time_embedding):
        """Multiplicative time conditioning keeps invalid zero samples zero."""
        batch, anchors, groups, frame_points, group_channels = sampled.shape
        if frame_points != self.sampling.num_frames * self.sampling.num_points:
            raise ValueError("sampled past time/point axis changed")
        if past_time_embedding.shape != (
            batch,
            self.sampling.num_frames,
            self.embed_dims,
        ):
            raise ValueError("past_time_embedding must be [B,T,C]")
        scale = 1.0 + torch.tanh(
            self.q4occ_past_time_scale(past_time_embedding)
        )
        scale = scale.reshape(
            batch,
            self.sampling.num_frames,
            groups,
            group_channels,
        ).permute(0, 2, 1, 3)
        scale = scale[:, None, :, :, None, :].expand(
            batch,
            anchors,
            groups,
            self.sampling.num_frames,
            self.sampling.num_points,
            group_channels,
        ).reshape(batch, anchors, groups, frame_points, group_channels)
        return sampled * scale

    def sensor_only_local(self, sampled, sampling_query):
        """Released adaptive mixing without its query residual or bias.

        Reimplementing the short value path is intentional.  Subtracting
        ``query`` and ``out_proj.bias`` from ``mixing(sampled, query)`` after
        the fact is not exact in FP16 and can leave a static local prior.  The
        parameter generator, channel/point mixing, normalizations, activation,
        and output weight below are the released SparseWorld operations.
        """
        batch, anchors, groups, points, group_channels = sampled.shape
        mixing = self.mixing
        if groups != mixing.n_groups or points != mixing.in_points:
            raise ValueError("AdaptiveMixing sensor shape contract changed")
        if group_channels != mixing.eff_in_dim:
            raise ValueError("AdaptiveMixing sensor channel contract changed")

        parameters = mixing.parameter_generator(sampling_query).reshape(
            batch * anchors, groups, -1
        )
        channel_parameters, point_parameters = parameters.split(
            [mixing.m_parameters, mixing.s_parameters], dim=2
        )
        channel_parameters = channel_parameters.reshape(
            batch * anchors,
            groups,
            mixing.eff_in_dim,
            mixing.eff_out_dim,
        )
        point_parameters = point_parameters.reshape(
            batch * anchors,
            groups,
            mixing.out_points,
            mixing.in_points,
        )
        value = sampled.reshape(
            batch * anchors, groups, points, group_channels
        )
        value = torch.matmul(value, channel_parameters)
        value = F.layer_norm(value, [value.size(-2), value.size(-1)])
        value = mixing.act(value)
        value = torch.matmul(point_parameters, value)
        value = F.layer_norm(value, [value.size(-2), value.size(-1)])
        value = mixing.act(value).reshape(batch, anchors, -1)
        # No bias: if sampled is exactly zero, every operation above and this
        # projection remain exactly zero in both FP32 and FP16.
        return F.linear(value, mixing.out_proj.weight, bias=None)

    def _aligned_points(self, query_points, fut2cur):
        metric = decode_points(query_points, self.pc_range)
        ones = torch.ones_like(metric[..., :1])
        homogeneous = torch.cat([metric, ones], dim=-1)
        matrices = torch.cat(fut2cur, dim=0).to(
            device=query_points.device, dtype=query_points.dtype
        )
        if matrices.shape[0] != query_points.shape[0]:
            raise ValueError("time-major fut2cur batch does not match queries")
        aligned = torch.einsum("bij,bqrj->bqri", matrices, homogeneous)
        return encode_points(aligned[..., :3], self.pc_range)

    def forward(
        self,
        query_points,
        query_content,
        scene_latent,
        base_condition,
        mlvl_feats,
        occ2img,
        img_metas,
        fut2cur,
        past_time_embedding,
        sensor_anchor_extent,
        anchor_local_memory=None,
        anchor_local_padding_mask=None,
    ):
        aligned_points = self._aligned_points(query_points, fut2cur)
        aligned_position = self.position_encoder(
            aligned_points.flatten(2, 3)
        )
        condition = base_condition + aligned_position
        sampling_query = query_content + condition
        sampling_points = self._sampling_support(
            query_points, sensor_anchor_extent
        )
        # Preserve the released positional caller contract; see encode_scene.
        sampled = self.sampling(
            sampling_points,
            sampling_query,
            mlvl_feats,
            occ2img,
            img_metas,
            fut2cur,
        )
        sampled = self.modulate_past_time(sampled, past_time_embedding)
        local_sensor = self.sensor_only_local(sampled, sampling_query)
        query_content = self.norm1(
            query_content + self.q4occ_sensor_gate * local_sensor
        )
        query_content = self.q4occ_query_decoder(
            query_content,
            scene_latent,
            condition,
            anchor_local_memory=anchor_local_memory,
            anchor_local_padding_mask=anchor_local_padding_mask,
        )

        batch_time, anchors = query_points.shape[:2]
        cls_score = self.cls_branch(query_content).reshape(
            batch_time, anchors, self.num_refines, self.num_classes
        )
        reg_offset = self.scale * self.reg_branch(query_content)
        refine_point = self.refine_points(query_points, reg_offset)
        expected = (batch_time, anchors, self.num_refines, 3)
        if tuple(refine_point.shape) != expected:
            raise AssertionError(
                f"refined point shape {tuple(refine_point.shape)} != {expected}"
            )
        return query_content, cls_score, refine_point
