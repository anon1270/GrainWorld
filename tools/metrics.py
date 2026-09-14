"""Streaming final-stage evaluation. No raw prediction files are collected."""
import math
import numpy as np
import torch
import torch.distributed as dist
import mmcv
from mmcv.runner import get_dist_info
from loaders.old_metrics import Metric_mIoU
from models.utils import sparse2dense
from tools.runtime import dataset_occ_root


def bundle_time_major_result(result, future_count):
    """Convert ``[t0b0,t0b1,t1b0,...]`` to one horizon list per sample."""
    if not isinstance(result, list) or not result:
        raise TypeError("SparseWorld validation must return a non-empty list")
    if future_count < 1:
        raise ValueError("future_count must be positive")
    if len(result) % future_count != 0:
        raise AssertionError(
            f"result count {len(result)} is not divisible by FT={future_count}"
        )
    local_batch = len(result) // future_count
    return [
        [
            result[time_index * local_batch + batch_index]
            for time_index in range(future_count)
        ]
        for batch_index in range(local_batch)
    ]



def unpadded_distributed_sample_index(
    local_ordinal, sampler_index, dataset_size, rank, world_size
):
    """Map a rank-local sampler item to the unpadded global dataset index.

    MMDetection's non-shuffled distributed sampler uses strided rank shards.
    Its final items may repeat the beginning of the dataset so every rank has
    the same length. Those padded repeats must not enter metric histograms.
    """
    if min(local_ordinal, sampler_index, dataset_size, rank) < 0:
        raise ValueError("sample indices and rank must be non-negative")
    if world_size < 1 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    global_position = local_ordinal * world_size + rank
    if global_position >= dataset_size:
        return None
    if sampler_index != global_position:
        raise AssertionError(
            "validation sampler is not the expected non-shuffled strided "
            f"sampler: rank={rank}, local_ordinal={local_ordinal}, "
            f"sampler_index={sampler_index}, expected={global_position}"
        )
    return sampler_index



def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None



def _accumulate_prediction(
    dataset, sample_index, future_index, result_dict, semantic, binary
):
    """Add one prediction to the released occupancy confusion matrices."""
    target_index = sample_index
    current_info = dataset.data_infos[sample_index]
    for offset in range(future_index):
        candidate = sample_index + 1 + offset
        if candidate < len(dataset.data_infos):
            next_info = dataset.data_infos[candidate]
            if next_info["scene_name"] == current_info["scene_name"]:
                target_index += 1
    if target_index - sample_index != future_index:
        return False

    target_info = dataset.data_infos[target_index]
    label_path = (
        dataset_occ_root(dataset)
        / target_info["scene_name"]
        / target_info["token"]
        / "labels.npz"
    )
    with np.load(str(label_path)) as labels:
        occupancy = labels["semantics"]
        mask_lidar = labels["mask_lidar"].astype(np.bool_)
        mask_camera = labels["mask_camera"].astype(np.bool_)
        dense_prediction, _ = sparse2dense(
            result_dict["occ_loc"],
            result_dict["sem_pred"],
            dense_shape=occupancy.shape,
            empty_value=17,
        )
        semantic.add_batch(
            dense_prediction, occupancy, mask_lidar, mask_camera
        )
        binary.add_batch(
            dense_prediction, occupancy, mask_lidar, mask_camera
        )
    return True



def _summarize_metric_state(state):
    """Format an all-reduced state like the released evaluator."""
    semantic = state["semantic"]
    binary = state["binary"]
    if semantic.cnt != binary.cnt:
        raise AssertionError(
            f"semantic/binary counts differ: {semantic.cnt} vs {binary.cnt}"
        )
    semantic_values = semantic.per_class_iu(semantic.hist) * 100.0
    binary_values = binary.per_class_iu(binary.hist) * 100.0
    per_class = {
        name: _finite_or_none(value)
        for name, value in zip(semantic.class_names, semantic_values)
    }
    selected_names = (
        "bicycle",
        "bus",
        "car",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "truck",
    )
    metrics = {
        "Semantic mIoU": semantic.count_miou(),
        "Binary IoU": binary.count_miou(),
    }
    class_metrics = {
        "evaluated_samples": int(semantic.cnt),
        "semantic_iou_per_class": per_class,
        "dynamic_small_iou": {
            name: per_class[name] for name in selected_names
        },
        "semantic_miou_non_free": round(
            float(np.nanmean(semantic_values[:-1])), 2
        ),
        "binary_occupied_iou": round(float(binary_values[0]), 2),
    }
    if (
        abs(
            float(metrics["Semantic mIoU"])
            - class_metrics["semantic_miou_non_free"]
        )
        > 1e-6
        or abs(
            float(metrics["Binary IoU"])
            - class_metrics["binary_occupied_iou"]
        )
        > 1e-6
    ):
        raise AssertionError(
            "Structured class audit does not reproduce the released "
            f"evaluator: released={metrics}, structured={class_metrics}"
        )
    return metrics, class_metrics



def _raise_if_validation_stage_failed(local_exception, stage, distributed):
    """Make every rank leave the same validation stage or fail together."""
    if distributed:
        success = torch.tensor(
            [0 if local_exception is not None else 1],
            dtype=torch.int32,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        dist.all_reduce(success, op=dist.ReduceOp.MIN)
        if not bool(success.item()):
            detail = (
                f"{type(local_exception).__name__}: {local_exception}"
                if local_exception is not None
                else "validation stage failed on another rank"
            )
            raise RuntimeError(
                f"distributed validation failed during {stage}: {detail}"
            )
    elif local_exception is not None:
        raise local_exception



def _all_reduce_metric_states(states, distributed):
    """Sum integer histograms/counts without gathering raw predictions."""
    tensor = None
    local_exception = None
    try:
        packed = []
        for state in states:
            packed.extend(
                np.rint(state["semantic"].hist).astype(np.int64).ravel()
            )
            packed.extend(
                np.rint(state["binary"].hist).astype(np.int64).ravel()
            )
            packed.extend(
                (
                    int(state["semantic"].cnt),
                    int(state["binary"].cnt),
                    int(state["predictions"]),
                )
            )
        tensor = torch.tensor(packed, dtype=torch.int64, device="cuda")
    except Exception as exception:
        local_exception = exception
    _raise_if_validation_stage_failed(
        local_exception, "metric-state packing", distributed
    )
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    local_exception = None
    try:
        reduced = tensor.cpu().numpy()
        cursor = 0
        for state in states:
            semantic_size = state["semantic"].hist.size
            binary_size = state["binary"].hist.size
            state["semantic"].hist = reduced[
                cursor : cursor + semantic_size
            ].reshape(state["semantic"].hist.shape).astype(np.float64)
            cursor += semantic_size
            state["binary"].hist = reduced[
                cursor : cursor + binary_size
            ].reshape(state["binary"].hist.shape).astype(np.float64)
            cursor += binary_size
            state["semantic"].cnt = int(reduced[cursor])
            state["binary"].cnt = int(reduced[cursor + 1])
            state["predictions"] = int(reduced[cursor + 2])
            cursor += 3
        if cursor != len(reduced):
            raise AssertionError(
                f"metric state unpack mismatch: {cursor} != {len(reduced)}"
            )
    except Exception as exception:
        local_exception = exception
    _raise_if_validation_stage_failed(
        local_exception, "metric-state unpacking", distributed
    )
    return states



def _multi_horizon_test(
    model,
    data_loader,
    future_frames,
    distributed,
    forward_fn=None,
    show_progress=True,
):
    """Stream predictions into all-reduced occupancy confusion matrices.

    MMDetection's stock CPU collector pickles every rank's complete output in
    ``./.dist_test``. SparseWorld's four occupancy predictions can exceed ten
    GiB per rank, so only the released 18x18 semantic and 2x2 binary matrices
    are retained here.
    """
    rank, world_size = get_dist_info()
    local_exception = None
    dataset = None
    states = None
    sampler_indices = None
    iterator = None
    progress = None
    try:
        if model is None and forward_fn is None:
            raise ValueError("model or forward_fn is required")
        if model is not None:
            model.eval()
        dataset = data_loader.dataset
        future_frames = tuple(int(index) for index in future_frames)
        states = [
            {
                "future_index": future_index,
                "semantic": Metric_mIoU(use_image_mask=True),
                "binary": Metric_mIoU(num_classes=2, use_image_mask=True),
                "predictions": 0,
            }
            for future_index in future_frames
        ]
        sampler = getattr(data_loader, "sampler", None)
        sampler_indices = (
            list(iter(sampler))
            if sampler is not None
            else list(range(len(dataset)))
        )
        iterator = iter(data_loader)
        progress = (
            mmcv.ProgressBar(len(dataset))
            if rank == 0 and show_progress
            else None
        )
        batch_count = len(data_loader)
    except Exception as exception:
        local_exception = exception
    _raise_if_validation_stage_failed(
        local_exception, "loader/sampler setup", distributed
    )
    if distributed:
        count_min = torch.tensor(
            [batch_count],
            dtype=torch.int64,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        count_max = count_min.clone()
        dist.all_reduce(count_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(count_max, op=dist.ReduceOp.MAX)
        if int(count_min.item()) != int(count_max.item()):
            # Every rank observes the same bounds and exits before entering a
            # different number of per-batch status collectives.
            raise RuntimeError(
                "distributed validation loader lengths differ across ranks: "
                f"min={int(count_min.item())}, max={int(count_max.item())}"
            )
    local_ordinal = 0
    progress_count = 0
    # All distributed ranks have the same number of sampler-padded batches.
    # Synchronize success after each batch so a loader/forward/evaluator error
    # on one rank cannot send that rank to the hook's status collective while
    # its peers wait forever in the final metric collective.
    for step in range(batch_count):
        local_exception = None
        local_batch = 0
        try:
            data = next(iterator)
            with torch.no_grad():
                if forward_fn is None:
                    result = model(return_loss=False, rescale=True, **data)
                else:
                    result = forward_fn(data)
            local_bundles = bundle_time_major_result(
                result, len(future_frames)
            )
            local_batch = len(local_bundles)
            batch_sampler_indices = sampler_indices[
                local_ordinal : local_ordinal + local_batch
            ]
            if len(batch_sampler_indices) != local_batch:
                raise AssertionError(
                    "validation loader emitted more samples than its sampler: "
                    f"ordinal={local_ordinal}, batch={local_batch}, "
                    f"sampler={len(sampler_indices)}"
                )
            for batch_offset, bundle in enumerate(local_bundles):
                ordinal = local_ordinal + batch_offset
                sample_index = unpadded_distributed_sample_index(
                    ordinal,
                    int(batch_sampler_indices[batch_offset]),
                    len(dataset),
                    rank,
                    world_size,
                )
                if sample_index is None:
                    continue
                for state, result_dict in zip(states, bundle):
                    state["predictions"] += 1
                    _accumulate_prediction(
                        dataset,
                        sample_index,
                        state["future_index"],
                        result_dict,
                        state["semantic"],
                        state["binary"],
                    )
            local_ordinal += local_batch
            if progress is not None:
                for _ in range(local_batch * world_size):
                    if progress_count < len(dataset):
                        progress.update()
                        progress_count += 1
        except Exception as exception:
            local_exception = exception
        _raise_if_validation_stage_failed(
            local_exception, f"batch {step}", distributed
        )

    local_exception = None
    try:
        if local_ordinal != len(sampler_indices):
            raise AssertionError(
                f"validation loader consumed {local_ordinal} sampler items, "
                f"expected {len(sampler_indices)}"
            )
    except Exception as exception:
        local_exception = exception
    _raise_if_validation_stage_failed(
        local_exception, "post-loop sample accounting", distributed
    )
    states = _all_reduce_metric_states(states, distributed)
    local_exception = None
    try:
        for state in states:
            if state["predictions"] != len(dataset):
                raise AssertionError(
                    f"horizon {state['future_index']} processed "
                    f"{state['predictions']} predictions for dataset {len(dataset)}"
                )
    except Exception as exception:
        local_exception = exception
    _raise_if_validation_stage_failed(
        local_exception, "global prediction accounting", distributed
    )
    if rank != 0:
        return None
    return states

