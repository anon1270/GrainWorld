"""Horizon-major batching and per-scene loss reduction for GrainWorld."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from mmdet.models import DETECTORS, HEADS

from .q4occ_phase1_performance_sparse_world import (
    Q4OccPhase1PerformanceHead,
    Q4OccPhase1PerformanceSparseWorld,
)


def _canonical_horizon_major_targets(
    value,
    *,
    batch_size: int,
    future_count: int,
    name: str,
):
    """Return ``[future_count * batch_size, ...]`` in horizon-major order.

    The production MMCV collate path yields a sequence of ``future_count``
    tensors shaped ``[batch_size, ...]``.  The scene-major form is accepted
    too when its leading dimensions are unambiguous, which keeps failures
    explicit if a future data pipeline changes the representation.
    """
    if batch_size < 1 or future_count < 1:
        raise ValueError(
            f"{name}: batch_size and future_count must be positive"
        )

    if torch.is_tensor(value):
        if value.ndim < 2:
            raise ValueError(
                f"{name}: tensor must expose future and batch axes, got "
                f"{tuple(value.shape)}"
            )
        leading = tuple(value.shape[:2])
        horizon_shape = (future_count, batch_size)
        scene_shape = (batch_size, future_count)
        if leading == horizon_shape and leading == scene_shape:
            # B == FT is intrinsically ambiguous for a bare tensor.  The
            # standard loader supplies a sequence, so fail closed here.
            raise ValueError(
                f"{name}: bare tensor layout is ambiguous because "
                f"batch_size == future_count == {batch_size}"
            )
        if leading == horizon_shape:
            return value.reshape(
                future_count * batch_size, *value.shape[2:]
            )
        if leading == scene_shape:
            return value.transpose(0, 1).reshape(
                future_count * batch_size, *value.shape[2:]
            )
        raise ValueError(
            f"{name}: expected leading [FT,B]={horizon_shape} or "
            f"[B,FT]={scene_shape}, got {leading}"
        )

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(
            f"{name}: expected a tensor or tensor sequence, got "
            f"{type(value).__name__}"
        )
    items = list(value)
    if not items or not all(torch.is_tensor(item) for item in items):
        raise TypeError(f"{name}: every sequence item must be a tensor")

    horizon_major = (
        len(items) == future_count
        and all(item.ndim >= 1 and item.shape[0] == batch_size for item in items)
    )
    scene_major = (
        len(items) == batch_size
        and all(item.ndim >= 1 and item.shape[0] == future_count for item in items)
    )
    if not horizon_major and not scene_major:
        prefixes = [tuple(item.shape[:1]) for item in items]
        raise ValueError(
            f"{name}: cannot identify target layout from outer length "
            f"{len(items)} and item prefixes {prefixes}"
        )

    # When B == FT both predicates are true.  Recursive MMCV collate defines
    # the outer sequence as the original horizon axis, so horizon-major is the
    # production contract and is selected deliberately.
    if horizon_major:
        tail_shape = tuple(items[0].shape[1:])
        if any(tuple(item.shape[1:]) != tail_shape for item in items[1:]):
            raise ValueError(f"{name}: horizon tensors have different shapes")
        result = torch.cat(items, dim=0)
    else:
        full_shape = tuple(items[0].shape)
        if any(tuple(item.shape) != full_shape for item in items[1:]):
            raise ValueError(f"{name}: scene tensors have different shapes")
        stacked = torch.stack(items, dim=0)
        result = stacked.transpose(0, 1).reshape(
            future_count * batch_size, *stacked.shape[2:]
        )

    expected_rows = future_count * batch_size
    if result.shape[0] != expected_rows:
        raise RuntimeError(
            f"{name}: canonicalization produced {result.shape[0]} rows, "
            f"expected {expected_rows}"
        )
    return result


def _canonical_horizon_validity(
    value,
    *,
    batch_size: int,
    future_count: int,
):
    """Return a strict bool ``[future_count * batch_size]`` validity vector."""
    if torch.is_tensor(value) and value.ndim == 1:
        if batch_size != 1 or value.shape[0] != future_count:
            raise ValueError(
                'future_valid: a flat tensor is only unambiguous for B=1; '
                f'got shape {tuple(value.shape)}, B={batch_size}, '
                f'FT={future_count}'
            )
        result = value
    else:
        result = _canonical_horizon_major_targets(
            value,
            batch_size=batch_size,
            future_count=future_count,
            name='future_valid',
        )
    if result.ndim != 1:
        raise ValueError(
            'future_valid must contain one scalar per horizon and scene, got '
            f'{tuple(result.shape)}'
        )
    if result.dtype != torch.bool:
        raise TypeError('future_valid must have bool dtype')
    return result


def _row_index(reference, *, scene: int, batch_size: int, future_count: int):
    rows = range(scene, batch_size * future_count, batch_size)
    return reference.new_tensor(tuple(rows), dtype=torch.long)


def _slice_prediction_rows(
    predictions,
    *,
    scene: int,
    batch_size: int,
    future_count: int,
):
    expected_rows = batch_size * future_count
    result = dict(predictions)
    for key in ("all_cls_scores", "all_refine_pts"):
        stages = predictions.get(key)
        if not isinstance(stages, (list, tuple)) or not stages:
            raise TypeError(f"predictions[{key!r}] must be a non-empty sequence")
        selected = []
        for stage, tensor in enumerate(stages):
            if not torch.is_tensor(tensor) or tensor.shape[0] != expected_rows:
                shape = tuple(tensor.shape) if torch.is_tensor(tensor) else None
                raise ValueError(
                    f"predictions[{key!r}][{stage}] must have "
                    f"{expected_rows} leading rows, got {shape}"
                )
            index = _row_index(
                tensor,
                scene=scene,
                batch_size=batch_size,
                future_count=future_count,
            )
            selected.append(tensor.index_select(0, index))
        result[key] = selected

    initial = predictions.get("init_points")
    if initial is not None:
        if not torch.is_tensor(initial) or initial.shape[0] != expected_rows:
            shape = tuple(initial.shape) if torch.is_tensor(initial) else None
            raise ValueError(
                f"predictions['init_points'] must have {expected_rows} "
                f"leading rows, got {shape}"
            )
        index = _row_index(
            initial,
            scene=scene,
            batch_size=batch_size,
            future_count=future_count,
        )
        result["init_points"] = initial.index_select(0, index)
    return result


@HEADS.register_module()
class Q4OccPhase1PerformanceBatchSafeHead(Q4OccPhase1PerformanceHead):
    """Average unchanged inherited losses per scene for local B > 1."""

    def loss(
        self,
        voxel_semantics,
        mask_camera,
        preds_dicts,
        future_valid=None,
    ):
        future_count = len(self.future_frames)
        stages = preds_dicts.get("all_cls_scores")
        if not isinstance(stages, (list, tuple)) or not stages:
            raise TypeError("all_cls_scores must be a non-empty sequence")
        total_rows = int(stages[0].shape[0])
        if future_count < 1 or total_rows % future_count:
            raise ValueError(
                f"prediction rows {total_rows} are not divisible by "
                f"future_count {future_count}"
            )
        batch_size = total_rows // future_count
        if batch_size == 1:
            return super().loss(
                voxel_semantics,
                mask_camera,
                preds_dicts,
                future_valid=future_valid,
            )

        expected_rows = batch_size * future_count
        for name, target in (
            ("voxel_semantics", voxel_semantics),
            ("mask_camera", mask_camera),
        ):
            if not torch.is_tensor(target) or target.shape[0] != expected_rows:
                shape = tuple(target.shape) if torch.is_tensor(target) else None
                raise ValueError(
                    f"{name} must have {expected_rows} horizon-major rows, "
                    f"got {shape}"
                )
        if future_valid is not None:
            if (not torch.is_tensor(future_valid)
                    or future_valid.ndim != 1
                    or future_valid.shape[0] != expected_rows
                    or future_valid.dtype != torch.bool):
                shape = (
                    tuple(future_valid.shape)
                    if torch.is_tensor(future_valid) else None
                )
                raise ValueError(
                    'future_valid must be a bool horizon-major vector with '
                    f'{expected_rows} rows, got {shape}'
                )

        scene_losses = []
        for scene in range(batch_size):
            target_index = _row_index(
                voxel_semantics,
                scene=scene,
                batch_size=batch_size,
                future_count=future_count,
            )
            mask_index = _row_index(
                mask_camera,
                scene=scene,
                batch_size=batch_size,
                future_count=future_count,
            )
            scene_predictions = _slice_prediction_rows(
                preds_dicts,
                scene=scene,
                batch_size=batch_size,
                future_count=future_count,
            )
            scene_losses.append(
                super().loss(
                    voxel_semantics.index_select(0, target_index),
                    mask_camera.index_select(0, mask_index),
                    scene_predictions,
                    future_valid=(
                        None if future_valid is None else
                        future_valid.index_select(0, target_index)
                    ),
                )
            )

        keys = tuple(scene_losses[0])
        if any(tuple(losses) != keys for losses in scene_losses[1:]):
            raise RuntimeError("per-scene loss dictionaries have different keys")
        averaged = {}
        for key in keys:
            values = [losses[key] for losses in scene_losses]
            if not all(torch.is_tensor(value) for value in values):
                raise TypeError(f"loss {key!r} is not tensor-valued")
            combined = values[0]
            for value in values[1:]:
                combined = combined + value
            averaged[key] = combined / float(batch_size)
        return averaged


@DETECTORS.register_module()
class Q4OccPhase1PerformanceBatchSafeSparseWorld(
    Q4OccPhase1PerformanceSparseWorld
):
    """Canonicalize training targets without changing model computation."""

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
        voxel_semantics=None,
        mask_camera=None,
        future_valid=None,
        fut2cur=None,
        fut_list=None,
    ):
        batch_size = len(img_metas) if isinstance(img_metas, (list, tuple)) else 0
        if batch_size < 1:
            raise ValueError("img_metas must contain at least one scene")
        screening_enabled = bool(getattr(
            self.pts_bbox_head, 'q4occ_screening_loss_enabled', False
        ))
        if batch_size == 1 and not screening_enabled:
            # Preserve the already-audited production B=1 path exactly.
            return super().forward_train(
                points=points,
                img_metas=img_metas,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                gt_labels=gt_labels,
                gt_bboxes=gt_bboxes,
                img=img,
                proposals=proposals,
                gt_bboxes_ignore=gt_bboxes_ignore,
                img_depth=img_depth,
                img_mask=img_mask,
                voxel_semantics=voxel_semantics,
                mask_camera=mask_camera,
                fut2cur=fut2cur,
                fut_list=fut_list,
            )

        if not isinstance(fut2cur, (list, tuple)) or not fut2cur:
            raise TypeError("fut2cur must be a non-empty horizon sequence")
        future_count = len(fut2cur)
        if not isinstance(fut_list, (list, tuple)) or len(fut_list) != future_count:
            raise ValueError("fut2cur and fut_list horizon counts must match")
        for horizon, matrix in enumerate(fut2cur):
            if not torch.is_tensor(matrix) or tuple(matrix.shape) != (
                batch_size,
                4,
                4,
            ):
                shape = tuple(matrix.shape) if torch.is_tensor(matrix) else None
                raise ValueError(
                    f"fut2cur[{horizon}] must be [B,4,4] with B="
                    f"{batch_size}, got {shape}"
                )

        horizon_voxels = _canonical_horizon_major_targets(
            voxel_semantics,
            batch_size=batch_size,
            future_count=future_count,
            name="voxel_semantics",
        )
        horizon_masks = _canonical_horizon_major_targets(
            mask_camera,
            batch_size=batch_size,
            future_count=future_count,
            name="mask_camera",
        )
        horizon_validity = None
        if future_valid is not None:
            horizon_validity = _canonical_horizon_validity(
                future_valid,
                batch_size=batch_size,
                future_count=future_count,
            )
        image_features = self.extract_feat(img, img_metas)
        predictions = self.pts_bbox_head(
            image_features, img_metas, fut2cur, fut_list
        )
        return self.pts_bbox_head.loss(
            horizon_voxels,
            horizon_masks,
            predictions,
            future_valid=horizon_validity,
        )


__all__ = [
    "Q4OccPhase1PerformanceBatchSafeHead",
    "Q4OccPhase1PerformanceBatchSafeSparseWorld",
]
