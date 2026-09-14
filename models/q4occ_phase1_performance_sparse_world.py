"""Inherited detector/head API used by the batch-safe GrainWorld classes."""

from __future__ import annotations

from mmdet.models import DETECTORS, HEADS

from .sparse_world import SparseWorld
from .sparse_world_head import SparseWorldHead


@HEADS.register_module()
class Q4OccPhase1PerformanceHead(SparseWorldHead):
    """Keep the released heads/loss while exposing split scene/query APIs."""

    def encode_scene(self, mlvl_feats, img_metas):
        batch = mlvl_feats[0].shape[0]
        anchors = self.init_points.weight[None, :, None, :].repeat(
            batch, 1, 1, 1
        )
        features = anchors.new_zeros(
            batch, self.num_query, self.embed_dims
        )
        return self.transformer.encode_scene(
            anchors, features, mlvl_feats, img_metas
        )

    def decode_queries(self, scene_state, fut2cur, fut_list):
        scores, points = self.transformer.decode_queries(
            scene_state, fut2cur, fut_list
        )
        return {
            "init_points": scene_state.anchor_points.repeat(
                len(fut2cur), 1, 1, 1
            ),
            "all_cls_scores": scores,
            "all_refine_pts": points,
        }

    def forward(self, mlvl_feats, img_metas, fut2cur, fut_list):
        state = self.encode_scene(mlvl_feats, img_metas)
        return self.decode_queries(state, fut2cur, fut_list)


@DETECTORS.register_module()
class Q4OccPhase1PerformanceSparseWorld(SparseWorld):
    """Cache one past-only scene state and decode many trajectories."""

    def encode_scene(self, img, img_metas):
        image_features = self.extract_feat(img, img_metas)
        return self.pts_bbox_head.encode_scene(image_features, img_metas)

    def decode_queries(self, scene_state, fut2cur, fut_list):
        return self.pts_bbox_head.decode_queries(
            scene_state, fut2cur, fut_list
        )
