"""GrainWorld: observation-only evidence writing and six-stage global/local retrieval.

Internal class and parameter names are retained for existing checkpoints."""

from __future__ import annotations

from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import ModuleList
from mmdet.models.utils.builder import TRANSFORMER

from .q4occ_phase1_performance_transformer import (
    Q4OccPhase1PerformanceDecoder,
    _performance_options,
)
from .bbox.utils import decode_points
from .sparse_world_sampling import sampling_4d
from .sparse_world_transformer import (
    SparseWorldSampling,
    SparseWorldTransformer,
    SparseWorldTransformerDecoderLayer,
)
from .utils import DUMP


class DualContextJointSceneState(NamedTuple):
    """DCM-only scene state; shared K512/dual-memory state stays untouched."""

    scene_latent: Optional[torch.Tensor]
    anchor_points: torch.Tensor
    anchor_features: torch.Tensor
    prepared_features: Tuple[torch.Tensor, ...]
    occ2img: torch.Tensor
    img_metas: Sequence[Mapping[str, Any]]
    relative_past_seconds: torch.Tensor
    past_time_embedding: torch.Tensor
    anchor_local_detail: Optional[torch.Tensor]
    anchor_local_padding_mask: Optional[torch.Tensor]
    anchor_appearance_bank: Optional[torch.Tensor] = None
    anchor_appearance_valid: Optional[torch.Tensor] = None


def _joint_options(raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Validate the structural DCM-JR contract used by this branch.

    K (global latent count) and L (anchor-local token count) are scale axes.
    Their exact values are therefore validated against the constructed model
    later, rather than being frozen to the Small preset here.
    """

    raw = dict(raw or {})
    options = _performance_options(raw)
    if options["enabled"]:
        if not options["dual_memory"]:
            raise ValueError(
                "DualContextJointTransformer requires global and local memory"
            )
        if options["latent_layers"] != 1:
            raise ValueError("joint-memory scene latent depth must remain 1")
        if options["decoder_layers"] != 6:
            raise ValueError("joint-memory refinement requires six stages")
    gate_init = float(raw.get("joint_memory_layer_scale_init", 1.0e-3))
    if not 0.0 <= gate_init < 1.0:
        raise ValueError(
            "joint_memory_layer_scale_init must be inside [0, 1)"
        )
    options["joint_memory_layer_scale_init"] = gate_init
    # The released decoder cuts the geometry graph between refinement stages.
    # A cached-memory loss must not add a second objective to the released
    # occupancy reg_branch.  Learnable memory geometry requires a separate
    # q4occ-prefixed head and is deliberately not part of this corrected arm.
    geometry_grad_scale = float(raw.get("memory_geometry_grad_scale", 0.0))
    if geometry_grad_scale != 0.0:
        raise ValueError(
            "corrected DCM-JR requires memory_geometry_grad_scale=0; "
            "do not backpropagate cached-memory geometry through the released "
            "occupancy reg_branch"
        )
    options["memory_geometry_grad_scale"] = 0.0
    detach_prepass = bool(raw.get("detach_memory_prepass", True))
    if not detach_prepass:
        raise ValueError(
            "corrected DCM-JR requires detach_memory_prepass=True so the "
            "released sensor path is not optimized by a second memory loss"
        )
    options["detach_memory_prepass"] = True
    local_scope = str(raw.get("joint_local_memory_scope", "all"))
    if local_scope not in {"stage", "all"}:
        raise ValueError("joint_local_memory_scope must be 'stage' or 'all'")
    options["joint_local_memory_scope"] = local_scope
    parallel_fusion = bool(raw.get("parallel_memory_fusion", True))
    if not parallel_fusion:
        raise ValueError(
            "corrected DCM-JR requires parallel global/local fusion from the "
            "same released sensor state"
        )
    options["parallel_memory_fusion"] = True
    options["align_frame_group_sampling"] = bool(
        raw.get("align_frame_group_sampling", False)
    )
    joint_interaction = bool(raw.get("joint_query_interaction", True))
    if options["enabled"] and not joint_interaction:
        raise ValueError("joint_query_interaction must remain enabled")
    options["joint_query_interaction"] = joint_interaction
    global_memory_enabled = bool(
        raw.get("joint_global_memory_enabled", True)
    )
    local_memory_enabled = bool(
        raw.get("joint_local_memory_enabled", True)
    )
    if options["enabled"] and not (
        global_memory_enabled or local_memory_enabled
    ):
        raise ValueError(
            "at least one of joint_global_memory_enabled or "
            "joint_local_memory_enabled must be true"
        )
    options["joint_global_memory_enabled"] = global_memory_enabled
    options["joint_local_memory_enabled"] = local_memory_enabled
    appearance_enabled = bool(raw.get("appearance_memory_enabled", False))
    options["appearance_memory_enabled"] = appearance_enabled
    options["appearance_feature_level"] = int(
        raw.get("appearance_feature_level", 0)
    )
    options["appearance_patch_size"] = int(
        raw.get("appearance_patch_size", 3)
    )
    options["appearance_current_frame_only"] = bool(
        raw.get("appearance_current_frame_only", True)
    )
    options["appearance_detach_evidence"] = bool(
        raw.get("appearance_detach_evidence", True)
    )
    if appearance_enabled:
        if options["appearance_feature_level"] != 0:
            raise ValueError("A1 appearance memory must use current-frame P2")
        if options["appearance_patch_size"] != 3:
            raise ValueError("A1 appearance patch must remain ordered 3x3")
        if not options["appearance_current_frame_only"]:
            raise ValueError("A1 must not read future or stale appearance frames")
        if not options["appearance_detach_evidence"]:
            raise ValueError(
                "A1 appearance evidence must be detached from the released FPN"
            )
    return options


class _GlobalMemoryResidual(nn.Module):
    """Query the shared scene memory without mixing output-query rows."""

    def __init__(self, channels, num_heads, gate_init):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = int(channels)
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads, dropout=0.0, batch_first=True
        )
        # Direct LayerScale keeps d(output)/d(scale)=update even at zero.  The
        # production 1e-3 setting is near-identity but gives attention weights
        # a nonzero gradient from the first backward; tests may set exact zero.
        self.layer_scale = nn.Parameter(torch.full((channels,), gate_init))

    def forward(self, content, memory):
        """Return only the scaled global-memory delta."""
        if content.ndim != 3 or content.shape[-1] != self.channels:
            raise ValueError("global content must be [F*B,N,C]")
        if memory.ndim != 3 or memory.shape[0] != content.shape[0]:
            raise ValueError("expanded global memory must be [F*B,K,C]")
        if memory.shape[-1] != self.channels:
            raise ValueError("global memory channel contract changed")
        update, _ = self.attention(
            self.query_norm(content),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        scale = self.layer_scale.to(content.dtype)
        return scale.view(1, 1, -1) * update

class _AnchorLocalMemoryResidual(nn.Module):
    """Fuse the selected detail bank for the same persistent anchor index."""

    def __init__(self, channels, num_heads, gate_init):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = int(channels)
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads, dropout=0.0, batch_first=True
        )
        self.layer_scale = nn.Parameter(torch.full((channels,), gate_init))

    def forward(self, content, memory, padding_mask, future_count):
        """Return only the scaled anchor-local-memory delta."""
        if memory.ndim != 4:
            raise ValueError("local memory must be [B,N,L,C]")
        batch, anchors, local_tokens, channels = memory.shape
        if channels != self.channels or local_tokens < 1:
            raise ValueError("local memory channel/token contract changed")
        if tuple(padding_mask.shape) != (batch, anchors, local_tokens):
            raise ValueError("local padding mask must be [B,N,L]")
        if tuple(content.shape) != (
            future_count * batch,
            anchors,
            channels,
        ):
            raise ValueError(
                "content must be horizon-major [F*B,N,C] and anchor-aligned"
            )

        # [F*B,N,C] -> [B*N,F,C].  MHA evaluates every query row against K/V;
        # it does not mix the F rows with each other.
        query = self.query_norm(content).reshape(
            future_count, batch, anchors, channels
        ).permute(1, 2, 0, 3).reshape(batch * anchors, future_count, channels)
        value = self.memory_norm(memory).reshape(
            batch * anchors, local_tokens, channels
        )
        padding = padding_mask.to(dtype=torch.bool).reshape(
            batch * anchors, local_tokens
        )
        all_invalid = padding.all(dim=1)
        # MultiheadAttention emits NaN when all keys are masked.  Temporarily
        # expose a zero dummy key, then force that row's complete update to 0.
        padding = padding.clone()
        padding[:, 0] = padding[:, 0] & ~all_invalid
        update, _ = self.attention(
            query, value, value, key_padding_mask=padding, need_weights=False
        )
        update = update.masked_fill(all_invalid[:, None, None], 0.0)
        update = update.reshape(
            batch, anchors, future_count, channels
        ).permute(2, 0, 1, 3).reshape(
            future_count * batch, anchors, channels
        )
        scale = self.layer_scale.to(content.dtype)
        return scale.view(1, 1, -1) * update


class _AnchorAppearanceResidual(nn.Module):
    """Fuse one stage-aligned current-P2 patch token per anchor.

    This is intentionally not attention over a singleton token (which would
    collapse to a query-independent value projection).  A query-conditioned
    MLP combines the released sensor state and ordered 3x3 appearance token.
    LayerScale is the only zero mechanism, preserving a nonzero gate gradient
    at exact-zero initialization and upstream gradients after the first update.
    """

    def __init__(self, channels, gate_init):
        super().__init__()
        self.channels = int(channels)
        self.query_norm = nn.LayerNorm(channels)
        self.appearance_norm = nn.LayerNorm(channels)
        self.fusion = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.layer_scale = nn.Parameter(torch.full((channels,), gate_init))

    def forward(self, content, appearance, valid, future_count):
        if appearance.ndim != 3:
            raise ValueError("appearance memory must be [B,N,C]")
        batch, anchors, channels = appearance.shape
        if channels != self.channels:
            raise ValueError("appearance memory channel contract changed")
        if tuple(valid.shape) != (batch, anchors):
            raise ValueError("appearance validity must be [B,N]")
        if tuple(content.shape) != (
            future_count * batch,
            anchors,
            channels,
        ):
            raise ValueError(
                "appearance content must be horizon-major [F*B,N,C]"
            )
        expanded = appearance.unsqueeze(0).expand(
            future_count, -1, -1, -1
        ).reshape(future_count * batch, anchors, channels)
        update = self.fusion(torch.cat([
            self.query_norm(content),
            self.appearance_norm(expanded),
        ], dim=-1))
        expanded_valid = valid.unsqueeze(0).expand(
            future_count, -1, -1
        ).reshape(future_count * batch, anchors)
        update = torch.nan_to_num(update).masked_fill(
            ~expanded_valid[..., None].to(dtype=torch.bool), 0.0
        )
        scale = self.layer_scale.to(content.dtype)
        return scale.view(1, 1, -1) * update

class _FrameGroupAlignedSampling(SparseWorldSampling):
    """Correct the released frame/group scale-weight flattening mismatch.

    Released features and locations flatten in ``[B,T,G]`` order, while the
    scale tensor is flattened in ``[B,G,T]`` order inside ``sampling_4d``.
    This branch pre-swizzles only the weights so the unmodified CUDA/PyTorch
    operator receives aligned rows.  A corrected baseline must use the same
    alignment when reporting the gain of the memory method.
    """

    q4occ_frame_group_aligned = True

    def inner_forward(
        self,
        query_points,
        query_feat,
        mlvl_feats,
        occ2img,
        img_metas,
        fut2cur,
    ):
        batch, anchors = query_points.shape[:2]
        image_h, image_w, _ = img_metas[0]["img_shape"][0]
        metric_points = decode_points(query_points, self.pc_range)
        if metric_points.shape[2] == 1:
            query_center = metric_points
            query_scale = torch.zeros_like(query_center)
        else:
            query_center = metric_points.mean(dim=2, keepdim=True)
            query_scale = metric_points.std(dim=2, keepdim=True)

        sampling_offset = self.sampling_offset(query_feat).view(
            batch, anchors, -1, 3
        )
        sampling_points = query_center + sampling_offset * query_scale
        sampling_points = sampling_points.view(
            batch, anchors, self.num_groups, self.num_points, 3
        )[:, :, None, ...].expand(
            batch,
            anchors,
            self.num_frames,
            self.num_groups,
            self.num_points,
            3,
        )

        scale_weights = self.scale_weights(query_feat).view(
            batch,
            anchors,
            self.num_groups,
            1,
            self.num_points,
            self.num_levels,
        )
        scale_weights = torch.softmax(scale_weights, dim=-1).expand(
            batch,
            anchors,
            self.num_groups,
            self.num_frames,
            self.num_points,
            self.num_levels,
        )
        # sampling_4d will reshape as [B,Q,G,T,...] and then permute to
        # [B,G,T,Q,...].  Populate that view with a linear [B,T,G,Q,...]
        # sequence so its final flatten matches features and locations.
        aligned = scale_weights.permute(0, 3, 2, 1, 4, 5).contiguous()
        scale_weights = aligned.reshape(
            batch,
            self.num_groups,
            self.num_frames,
            anchors,
            self.num_points,
            self.num_levels,
        ).permute(0, 3, 1, 2, 4, 5).contiguous()
        return sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            occ2img,
            fut2cur,
            image_h,
            image_w,
            self.num_views,
        )


class DualContextJointDecoderLayer(SparseWorldTransformerDecoderLayer):
    """Released joint layer with two pre-interaction memory residuals."""

    def __init__(self, *args, q4occ=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.q4occ = _joint_options(q4occ)
        if self.q4occ["align_frame_group_sampling"]:
            released_sampling = self.sampling
            self.sampling = _FrameGroupAlignedSampling(
                embed_dims=self.embed_dims,
                num_frames=released_sampling.num_frames,
                num_views=released_sampling.num_views,
                num_groups=released_sampling.num_groups,
                num_points=released_sampling.num_points,
                num_levels=released_sampling.num_levels,
                pc_range=released_sampling.pc_range,
            )
            self.sampling.load_state_dict(released_sampling.state_dict())
        # Construct both candidates in a fixed order so an enabled branch has
        # identical initialization in FULL and its single-context ablation.
        # Only enabled modules are registered; disabled branches therefore add
        # no parameters, optimizer state, or accidental zero-gate path.
        global_memory_fusion = _GlobalMemoryResidual(
            self.embed_dims,
            self.q4occ["num_heads"],
            self.q4occ["joint_memory_layer_scale_init"],
        )
        local_memory_fusion = _AnchorLocalMemoryResidual(
            self.embed_dims,
            self.q4occ["num_heads"],
            self.q4occ["joint_memory_layer_scale_init"],
        )
        if self.q4occ["joint_global_memory_enabled"]:
            self.q4occ_global_memory_fusion = global_memory_fusion
        if self.q4occ["joint_local_memory_enabled"]:
            self.q4occ_local_memory_fusion = local_memory_fusion
        if self.q4occ["appearance_memory_enabled"]:
            # A1-only parameter draws must not shift initialization of common
            # global/local modules in later layers.  The enclosing decoder
            # fork protects modules outside this decoder; this per-layer fork
            # also preserves cross-arm equality inside its layer sequence.
            with torch.random.fork_rng(devices=[]):
                self.q4occ_appearance_memory_fusion = (
                    _AnchorAppearanceResidual(
                        self.embed_dims,
                        self.q4occ["joint_memory_layer_scale_init"],
                    )
                )

    def _select_local_memory(self, memory, padding_mask):
        """Return the stage-matched slice or the complete local bank.

        ``encode_with_detail`` flattens detail in ``[T, stage, point]`` order.
        Selecting the matching stage prevents every joint layer from replaying
        all cached refinement levels while preserving the complete bank for
        the global scene encoder.
        """

        if self.q4occ["joint_local_memory_scope"] == "all":
            return memory, padding_mask
        batch, anchors, tokens, channels = memory.shape
        stages = int(self.q4occ["decoder_layers"])
        frames = int(self.sampling.num_frames)
        points = int(self.sampling.num_points)
        expected = frames * stages * points
        if tokens != expected:
            raise ValueError(
                f"local memory has {tokens} tokens; expected {expected} in "
                "[T,stage,point] order"
            )
        memory = memory.reshape(
            batch, anchors, frames, stages, points, channels
        )[:, :, :, self.layer_idx, :, :]
        padding_mask = padding_mask.reshape(
            batch, anchors, frames, stages, points
        )[:, :, :, self.layer_idx, :]
        return (
            memory.reshape(batch, anchors, frames * points, channels),
            padding_mask.reshape(batch, anchors, frames * points),
        )

    def released_sensor_prefix(
        self,
        query_points,
        query_feat,
        mlvl_feats,
        occ2img,
        img_metas,
        fut2cur,
    ):
        """Run the released position/sampling/AdaptiveMixing prefix exactly.

        Both the cached past-only prepass and the normal future-conditioned
        joint path call this single implementation.  This prevents a manual
        memory-side approximation from silently dropping the query residual,
        output bias, or released query-point support again.
        """

        query_position = self.position_encoder(query_points.flatten(2, 3))
        sensor_query = query_feat + query_position
        sampled = self.sampling(
            query_points,
            sensor_query,
            mlvl_feats,
            occ2img,
            img_metas,
            fut2cur,
        )
        sensor_feat = self.norm1(self.mixing(sampled, sensor_query))
        return sensor_feat, sampled, query_position

    @staticmethod
    def _scene_temporal_forward(module, content, future_count):
        """Apply released temporal attention with correct B>1 row grouping."""

        batch_time, anchors, channels = content.shape
        if batch_time % future_count:
            raise ValueError("horizon-major rows are not divisible by F")
        batch = batch_time // future_count
        scene_sequence = content.reshape(
            future_count, batch, anchors, channels
        ).permute(1, 0, 2, 3).reshape(
            batch, future_count * anchors, channels
        )
        # Preserve the released operation order and the intentionally shared
        # FFN instance used both before and after spatial self-attention.
        scene_sequence = module.ln(
            module.ffn(module.tempo_attn(scene_sequence))
        )
        return scene_sequence.reshape(
            batch, future_count, anchors, channels
        ).permute(1, 0, 2, 3).reshape(
            batch_time, anchors, channels
        )

    def forward(
        self,
        query_points,
        query_feat,
        scene_latent,
        anchor_local_memory,
        anchor_local_padding_mask,
        anchor_appearance_memory,
        anchor_appearance_valid,
        mlvl_feats,
        occ2img,
        img_metas,
        fut2cur,
    ):
        future_count = len(fut2cur)
        if future_count < 1:
            raise ValueError("fut2cur must contain at least one horizon")
        # The same helper is used by the past-only cache and this released
        # future-conditioned joint path.
        query_feat, _, _ = self.released_sensor_prefix(
            query_points,
            query_feat,
            mlvl_feats,
            occ2img,
            img_metas,
            fut2cur,
        )

        # Compute complementary global and local updates from the same released
        # sensor state.  The former implementation was sequential, so local
        # attention queried an already globally modified feature despite the
        # method defining two parallel residuals.
        sensor_feat = query_feat
        if self.q4occ["joint_global_memory_enabled"]:
            if scene_latent is None:
                raise ValueError("global memory is enabled but memory is missing")
            global_delta = self.q4occ_global_memory_fusion(
                sensor_feat, scene_latent
            )
        else:
            if scene_latent is not None:
                raise ValueError("disabled global branch received global memory")
            global_delta = 0.0
        if self.q4occ["joint_local_memory_enabled"]:
            if (
                anchor_local_memory is None
                or anchor_local_padding_mask is None
            ):
                raise ValueError("local memory is enabled but memory is missing")
            local_memory, local_padding = self._select_local_memory(
                anchor_local_memory, anchor_local_padding_mask
            )
            local_delta = self.q4occ_local_memory_fusion(
                sensor_feat,
                local_memory,
                local_padding,
                future_count,
            )
        else:
            if (
                anchor_local_memory is not None
                or anchor_local_padding_mask is not None
            ):
                raise ValueError("disabled local branch received local memory")
            local_delta = 0.0
        if self.q4occ["appearance_memory_enabled"]:
            if anchor_appearance_memory is None or anchor_appearance_valid is None:
                raise ValueError("A1 requires stage-aligned appearance memory")
            appearance_delta = self.q4occ_appearance_memory_fusion(
                sensor_feat,
                anchor_appearance_memory,
                anchor_appearance_valid,
                future_count,
            )
        else:
            if anchor_appearance_memory is not None or anchor_appearance_valid is not None:
                raise ValueError("appearance memory supplied to a non-A1 decoder")
            appearance_delta = 0.0
        # Add each scaled update exactly once.  Avoid forming
        # ``(sensor + delta) - sensor``: at FP16 and a 1e-3 LayerScale that
        # intermediate addition can round the memory signal down to zero.
        query_feat = sensor_feat + global_delta + local_delta + appearance_delta
        query_feat = self._scene_temporal_forward(
            self, query_feat, future_count
        )
        query_feat = self.norm2(self.self_attn(query_points, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))

        batch_time, anchors = query_points.shape[:2]
        cls_score = self.cls_branch(query_feat).reshape(
            batch_time, anchors, self.num_refines, self.num_classes
        )
        reg_offset = self.scale * self.reg_branch(query_feat)
        refine_point = self.refine_points(query_points, reg_offset)
        expected = (batch_time, anchors, self.num_refines, 3)
        if tuple(refine_point.shape) != expected:
            raise AssertionError(
                f"refined point shape {tuple(refine_point.shape)} != {expected}"
            )
        return query_feat, cls_score, refine_point


class DualContextJointDecoder(Q4OccPhase1PerformanceDecoder):
    """Past-only global/local memory feeding six released joint stages."""

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
        options = _joint_options(q4occ)
        # The production CUDA sampler uses fixed-size local arrays bounded by
        # MAX_POINT=32.  Fail before model construction instead of risking a
        # kernel-side overwrite when a custom scale preset exceeds the bound.
        if not 1 <= int(num_points) <= 32:
            raise ValueError(
                "DCM-JR num_points must be inside [1, 32] for the released "
                "CUDA sampler"
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
            num_refines=num_refines,
            num_groups=num_groups,
            scales=scales,
            pc_range=pc_range,
            q4occ=options,
            init_cfg=init_cfg,
        )
        self.q4occ = options
        if not self.q4occ["joint_global_memory_enabled"]:
            # LOCAL_ONLY stops before global latent compression.  Freeze the
            # global-only tensors retained for matched initialization/state
            # layout so DDP never sees trainable parameters outside the graph.
            self.q4occ_scene_encoder.global_context_token.requires_grad_(False)
            self.q4occ_scene_encoder.latent_tokens.requires_grad_(False)
            for block in self.q4occ_scene_encoder.latent_blocks:
                block.requires_grad_(False)
        if len(scales) == 1:
            scales = scales * num_layers
        if not isinstance(num_refines, list):
            num_refines = [num_refines]
        if len(num_refines) == 1:
            num_refines = num_refines * num_layers
        last_refines = [1] + num_refines

        # Replace only the independent decoder layers created by the reusable
        # parent helper.  Copy every released common key, then drop the old
        # objects entirely so DDP has no trainable-but-unused parameters.
        source_layers = self.decoder_layers
        joint_layers = ModuleList(
            [
                DualContextJointDecoderLayer(
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
        for target, source in zip(joint_layers, source_layers):
            source_state = source.state_dict()
            target_state = target.state_dict()
            common = {
                key: value
                for key, value in source_state.items()
                if key in target_state and target_state[key].shape == value.shape
            }
            incompatible = target.load_state_dict(common, strict=False)
            common_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith("q4occ_")
            ]
            if incompatible.unexpected_keys or common_missing:
                raise RuntimeError(
                    "joint layer failed to preserve released state keys: "
                    f"missing={common_missing}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
        self.decoder_layers = joint_layers
        # The parent creates this only for its independent query branch.  Drop
        # it entirely so checkpoints and DDP contain no dead parameters.
        del self.q4occ_trajectory_encoder
        if self.q4occ["appearance_memory_enabled"]:
            patch_elements = self.q4occ["appearance_patch_size"] ** 2
            self.q4occ_appearance_patch_encoder = nn.Sequential(
                nn.Linear(patch_elements * embed_dims, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.GELU(),
                nn.Linear(embed_dims, embed_dims),
            )

    @torch.no_grad()
    def _sample_current_p2_patch(
        self,
        query_points,
        current_p2,
        occ2img,
        img_metas,
    ):
        """Project stage anchors and sample one ordered current-P2 3x3 patch.

        A 3D grid is used only as a compact way to select an exact camera
        slice and a spatial 3x3 neighborhood.  The view coordinate lands on a
        discrete camera plane with ``align_corners=True``, so adjacent cameras
        are never interpolated.  Camera selection is deterministic: among
        valid views, choose the projection closest to the image centre.
        """

        if query_points.ndim != 4 or query_points.shape[-1] != 3:
            raise ValueError("appearance query points must be [B,N,R,3]")
        if current_p2.ndim != 5:
            raise ValueError("current P2 must be [B,V,C,H,W]")
        batch, views, channels, height, width = current_p2.shape
        anchors = query_points.shape[1]
        if views != self.num_views or channels != self.embed_dims:
            raise ValueError("current P2 view/channel contract changed")
        if tuple(occ2img.shape[:2]) != (
            batch,
            self.num_frames * self.num_views,
        ):
            raise ValueError("appearance projection metadata contract changed")
        if len(img_metas) != batch:
            raise ValueError("appearance metadata batch changed")
        if height < 3 or width < 3:
            raise ValueError("P2 is too small for an ordered 3x3 patch")

        metric = decode_points(query_points.float(), self.pc_range).mean(dim=2)
        homogeneous = torch.cat(
            [metric, torch.ones_like(metric[..., :1])], dim=-1
        )
        current_projection = occ2img[:, :self.num_views].float()
        projected = torch.einsum(
            "bvij,bqj->bvqi", current_projection, homogeneous
        )
        depth = projected[..., 2]
        safe_depth = depth.clamp_min(1.0e-5)
        pixel = projected[..., :2] / safe_depth[..., None]

        image_shapes = []
        for index, meta in enumerate(img_metas):
            shapes = meta.get("img_shape")
            if not isinstance(shapes, (list, tuple)) or not shapes:
                raise ValueError(
                    f"img_metas[{index}] lacks augmented img_shape"
                )
            image_shapes.append((float(shapes[0][0]), float(shapes[0][1])))
        image_hw = pixel.new_tensor(image_shapes)
        normalized = pixel / image_hw[:, None, None, [1, 0]]

        # Require the complete 3x3 P2 patch to remain inside the image.  This
        # makes boundary padding incapable of masquerading as appearance.
        margin_x = 1.0 / float(width - 1)
        margin_y = 1.0 / float(height - 1)
        valid = (
            (depth > 1.0e-5)
            & torch.isfinite(normalized).all(dim=-1)
            & (normalized[..., 0] > margin_x)
            & (normalized[..., 0] < 1.0 - margin_x)
            & (normalized[..., 1] > margin_y)
            & (normalized[..., 1] < 1.0 - margin_y)
        )
        centre_distance = ((normalized - 0.5) ** 2).sum(dim=-1)
        centre_distance = centre_distance.masked_fill(~valid, float("inf"))
        chosen_view = centre_distance.argmin(dim=1)
        any_valid = valid.any(dim=1)

        normalized_by_anchor = normalized.permute(0, 2, 1, 3)
        gather_index = chosen_view[..., None, None].expand(-1, -1, 1, 2)
        centre_xy = normalized_by_anchor.gather(2, gather_index).squeeze(2)
        centre_xy = torch.nan_to_num(centre_xy).mul(2.0).sub(1.0)
        # An all-invalid anchor has no meaningful selected projection.  Keep
        # its grid finite before grid_sample, then mask its sampled patch back
        # to exact zero below.  This avoids feeding extreme nan_to_num values
        # into the CUDA sampler even though the final residual is invalid.
        centre_xy = torch.where(
            any_valid[..., None], centre_xy, torch.zeros_like(centre_xy)
        )
        view_z = chosen_view.to(dtype=centre_xy.dtype)
        view_z = view_z.mul(2.0 / float(max(views - 1, 1))).sub(1.0)
        centre_grid = torch.cat([centre_xy, view_z[..., None]], dim=-1)

        x_offsets = centre_grid.new_tensor(
            [-2.0 / (width - 1), 0.0, 2.0 / (width - 1)]
        )
        y_offsets = centre_grid.new_tensor(
            [-2.0 / (height - 1), 0.0, 2.0 / (height - 1)]
        )
        yy, xx = torch.meshgrid(y_offsets, x_offsets, indexing="ij")
        offsets = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1)
        grid = centre_grid[:, :, None, None, :] + offsets[None, None]
        volume = current_p2.detach().permute(0, 2, 1, 3, 4).contiguous()
        sampled = F.grid_sample(
            volume,
            grid.to(dtype=volume.dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        # [B,C,N,3,3] -> [B,N,9,C], preserving raster patch order.
        sampled = sampled.permute(0, 2, 3, 4, 1).reshape(
            batch, anchors, 9, channels
        )
        sampled = torch.nan_to_num(sampled).masked_fill(
            ~any_valid[..., None, None], 0.0
        )
        return sampled.detach(), any_valid

    def encode_scene(self, query_points, query_feat, mlvl_feats, img_metas):
        """Build a past-only cache through the exact released sensor prefix.

        The released path is a stop-gradient evidence provider.  This preserves
        both its forward algebra and its optimization contract: memory loss
        trains only Q4Occ time/scene/fusion modules and cannot repurpose the
        released occupancy sampling, mixing, normalization, or regression
        weights.  Timestamp conditioning is applied once inside the scene
        encoder, never before released AdaptiveMixing.
        """

        if query_points.ndim != 4 or query_points.shape[2:] != (1, 3):
            raise ValueError("initial query_points must be [B,Q,1,3]")
        if query_feat.ndim != 3 or query_feat.shape[:2] != query_points.shape[:2]:
            raise ValueError("query_feat must align with initial anchors")
        batch, anchors, _ = query_feat.shape
        if len(img_metas) != batch:
            raise ValueError("img_metas batch differs from anchors")

        relative_seconds = self._relative_past_seconds(query_feat, img_metas)
        past_time = self.q4occ_time_encoder(relative_seconds).to(query_feat.dtype)
        current_p2 = None
        if self.q4occ["appearance_memory_enabled"]:
            if not torch.equal(
                relative_seconds[:, 0],
                torch.zeros_like(relative_seconds[:, 0]),
            ):
                raise ValueError("A1 requires the first image group to be current")
            raw_p2 = mlvl_feats[self.q4occ["appearance_feature_level"]]
            if raw_p2.ndim != 5:
                raise ValueError("A1 raw P2 must be [B,T*V,C,H,W]")
            current_p2 = raw_p2[:, :self.num_views]
        prepared_features = self._prepare_sensor_features(mlvl_feats, batch)
        occ2img = self._build_occ2img(query_feat, img_metas)
        identity = torch.eye(
            4, device=query_feat.device, dtype=query_feat.dtype
        ).unsqueeze(0).expand(batch, -1, -1).contiguous()

        # The exact sensor prefix is intentionally evaluated without a graph.
        # The same released modules remain trainable through the untouched main
        # joint path later in this forward pass.
        with torch.no_grad():
            scene_points = query_points.detach()
            scene_content = query_feat.detach()
            detail_stages, valid_stages, position_stages = [], [], []
            appearance_patches, appearance_valid_stages = [], []
            anchor_position = None
            layer_count = len(self.decoder_layers)
            for stage_index, decoder_layer in enumerate(self.decoder_layers):
                DUMP.stage_count = stage_index
                if self.q4occ["appearance_memory_enabled"]:
                    patch, patch_valid = self._sample_current_p2_patch(
                        scene_points,
                        current_p2,
                        occ2img,
                        img_metas,
                    )
                    appearance_patches.append(patch)
                    appearance_valid_stages.append(patch_valid)
                # Bit-for-bit released prefix: a single-point anchor stays
                # single-point, and AdaptiveMixing retains query residual/bias.
                scene_content, sampled, anchor_position = (
                    decoder_layer.released_sensor_prefix(
                        scene_points,
                        scene_content,
                        prepared_features,
                        occ2img,
                        img_metas,
                        [identity],
                    )
                )
                raw_detail = self._detailed_sensor_samples(sampled)
                valid_stage = (
                    raw_detail.float().abs().sum(dim=-1)
                    > self.q4occ["valid_feature_epsilon"]
                )
                detail_stages.append(raw_detail)
                valid_stages.append(valid_stage)
                position_stages.append(
                    anchor_position[:, :, None, :].expand(
                        batch, anchors, self.num_points, self.embed_dims
                    )
                )
                if stage_index + 1 < layer_count:
                    scene_offset = (
                        decoder_layer.scale
                        * decoder_layer.reg_branch(scene_content)
                    )
                    scene_points = decoder_layer.refine_points(
                        scene_points, scene_offset
                    ).detach()

            detailed = torch.cat(detail_stages, dim=3).detach()
            valid = torch.cat(valid_stages, dim=3)
            detail_position = torch.cat(position_stages, dim=2).detach()
            scene_summary = scene_content.detach()
            if self.q4occ["appearance_memory_enabled"]:
                appearance_patch_bank = torch.stack(
                    appearance_patches, dim=2
                ).detach()
                appearance_valid = torch.stack(
                    appearance_valid_stages, dim=2
                )
            else:
                appearance_patch_bank = None
                appearance_valid = None
        if detailed.requires_grad or detail_position.requires_grad:
            raise AssertionError("cached sensor evidence must be detached")
        if scene_summary.requires_grad:
            raise AssertionError("cached sensor summary must be detached")
        if appearance_patch_bank is not None:
            if appearance_patch_bank.requires_grad:
                raise AssertionError("appearance evidence must be detached")
            expected_patch = (
                batch,
                anchors,
                len(self.decoder_layers),
                9,
                self.embed_dims,
            )
            if tuple(appearance_patch_bank.shape) != expected_patch:
                raise AssertionError("stage-aligned P2 patch shape changed")
            patch_encoder_dtype = next(
                self.q4occ_appearance_patch_encoder.parameters()
            ).dtype
            appearance_bank = self.q4occ_appearance_patch_encoder(
                appearance_patch_bank.reshape(
                    batch,
                    anchors,
                    len(self.decoder_layers),
                    9 * self.embed_dims,
                ).to(dtype=patch_encoder_dtype)
            ).to(dtype=query_feat.dtype)
            appearance_bank = torch.nan_to_num(appearance_bank).masked_fill(
                ~appearance_valid[..., None], 0.0
            )
        else:
            appearance_bank = None
        (
            scene_latent,
            anchor_local_detail,
            anchor_local_padding_mask,
        ) = self.q4occ_scene_encoder.encode_with_detail(
            detailed,
            valid,
            # scene_summary already contains the released positional residual;
            # adding anchor_position again would double-count geometry.
            scene_summary,
            past_time,
            detail_position=detail_position,
            compute_global=self.q4occ["joint_global_memory_enabled"],
        )
        expected_detail = (
            batch,
            anchors,
            self.q4occ_local_memory_tokens,
            self.embed_dims,
        )
        if self.q4occ["joint_global_memory_enabled"]:
            expected_global = (
                batch,
                self.q4occ["num_latents"],
                self.embed_dims,
            )
            if scene_latent is None or tuple(scene_latent.shape) != expected_global:
                raise AssertionError("global memory shape changed")
        elif scene_latent is not None:
            raise AssertionError("LOCAL_ONLY must skip global construction")
        if self.q4occ["joint_local_memory_enabled"]:
            if (
                anchor_local_detail is None
                or tuple(anchor_local_detail.shape) != expected_detail
            ):
                raise AssertionError("anchor-local detail shape changed")
            if (
                anchor_local_padding_mask is None
                or tuple(anchor_local_padding_mask.shape) != expected_detail[:-1]
            ):
                raise AssertionError("anchor-local padding shape changed")
        else:
            anchor_local_detail = None
            anchor_local_padding_mask = None
        return DualContextJointSceneState(
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
            anchor_appearance_bank=appearance_bank,
            anchor_appearance_valid=appearance_valid,
        )

    def decode_queries(self, scene_state, fut2cur, fut_list):
        if not isinstance(scene_state, DualContextJointSceneState):
            raise TypeError("scene_state must be DualContextJointSceneState")
        scene_latent = scene_state.scene_latent
        local_detail = scene_state.anchor_local_detail
        local_padding = scene_state.anchor_local_padding_mask
        appearance_bank = scene_state.anchor_appearance_bank
        appearance_valid = scene_state.anchor_appearance_valid
        query_points = scene_state.anchor_points
        query_feat = scene_state.anchor_features
        batch, anchors, channels = query_feat.shape
        future_count = len(fut2cur)
        if future_count < 1 or len(fut_list) != future_count:
            raise ValueError("fut2cur/fut_list horizon counts differ or are empty")
        expected_local = (
            batch,
            anchors,
            self.q4occ_local_memory_tokens,
            channels,
        )
        if self.q4occ["joint_local_memory_enabled"]:
            if local_detail is None or tuple(local_detail.shape) != expected_local:
                raise ValueError("joint decoder requires aligned local memory")
            if local_padding is None or tuple(local_padding.shape) != expected_local[:-1]:
                raise ValueError("joint decoder local padding contract changed")
        elif local_detail is not None or local_padding is not None:
            raise ValueError("GLOBAL_ONLY state unexpectedly contains local memory")
        if self.q4occ["joint_global_memory_enabled"]:
            if scene_latent is None or tuple(scene_latent.shape) != (
                batch,
                self.q4occ["num_latents"],
                channels,
            ):
                raise ValueError("joint decoder global memory contract changed")
        elif scene_latent is not None:
            raise ValueError("LOCAL_ONLY state unexpectedly contains global memory")
        if self.q4occ["appearance_memory_enabled"]:
            expected_appearance = (
                batch,
                anchors,
                len(self.decoder_layers),
                channels,
            )
            if appearance_bank is None or tuple(appearance_bank.shape) != expected_appearance:
                raise ValueError("A1 appearance bank contract changed")
            if appearance_valid is None or tuple(appearance_valid.shape) != expected_appearance[:-1]:
                raise ValueError("A1 appearance validity contract changed")
        elif appearance_bank is not None or appearance_valid is not None:
            raise ValueError("non-A1 state unexpectedly contains appearance")

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
                    f"fut2cur[{index}] must be [B,4,4], got "
                    f"{tuple(matrix.shape)}"
                )
            matrices.append(matrix)
        flat_matrix = torch.stack(matrices, dim=0).reshape(
            future_count * batch, 16
        )
        content = self.pe_mln(
            content, flat_matrix[:, None, :].expand(-1, anchors, -1)
        )
        if scene_latent is None:
            expanded_latent = None
        else:
            expanded_latent = scene_latent.unsqueeze(0).expand(
                future_count, -1, -1, -1
            ).reshape(
                future_count * batch, self.q4occ["num_latents"], channels
            )
        expanded_occ2img = scene_state.occ2img.repeat(
            future_count, 1, 1, 1
        )

        cls_scores, refine_points = [], []
        for index, decoder_layer in enumerate(self.decoder_layers):
            DUMP.stage_count = index
            query_points = query_points.detach()
            content, cls_score, query_points = decoder_layer(
                query_points=query_points,
                query_feat=content,
                scene_latent=expanded_latent,
                anchor_local_memory=local_detail,
                anchor_local_padding_mask=local_padding,
                anchor_appearance_memory=(
                    appearance_bank[:, :, index]
                    if appearance_bank is not None else None
                ),
                anchor_appearance_valid=(
                    appearance_valid[:, :, index]
                    if appearance_valid is not None else None
                ),
                mlvl_feats=scene_state.prepared_features,
                occ2img=expanded_occ2img,
                img_metas=scene_state.img_metas,
                fut2cur=matrices,
            )
            cls_scores.append(cls_score)
            refine_points.append(query_points)
        return cls_scores, refine_points


@TRANSFORMER.register_module()
class DualContextJointTransformer(SparseWorldTransformer):
    """SparseWorld-compatible wrapper for dual-context joint refinement."""

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
        num_refines = (
            [1, 4, 16, 32, 64, 128]
            if num_refines is None
            else num_refines
        )
        scales = [1.0] if scales is None else scales
        pc_range = [] if pc_range is None else pc_range
        options = _joint_options(q4occ)
        if options["enabled"] and options["decoder_layers"] != num_layers:
            raise ValueError("one memory-augmented joint layer is required per stage")
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
            with torch.random.fork_rng(devices=[]):
                joint_decoder = DualContextJointDecoder(
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
            incompatible = joint_decoder.load_state_dict(
                released_decoder.state_dict(), strict=False
            )
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    "joint decoder rejected released state keys: "
                    f"{incompatible.unexpected_keys}"
                )
            changed_common = [
                key
                for key in incompatible.missing_keys
                if "q4occ_" not in key
            ]
            if changed_common:
                raise RuntimeError(
                    "joint decoder changed released state keys: "
                    f"{changed_common}"
                )
            self.decoder = joint_decoder

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


__all__ = [
    "DualContextJointSceneState",
    "DualContextJointDecoder",
    "DualContextJointDecoderLayer",
    "DualContextJointTransformer",
]
