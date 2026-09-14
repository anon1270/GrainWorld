"""Optional live validation without changing the training RNG or precision."""
from __future__ import annotations

import random
from pathlib import Path
from queue import Queue

import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import Hook, get_dist_info

from tools.runtime import write_json


class ValidationHook(Hook):
    def __init__(self, loader, future_frames, interval=5):
        self.loader = loader
        self.future_frames = tuple(future_frames)
        self.interval = int(interval)

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1
        if epoch % self.interval and epoch != runner.max_epochs:
            return

        from tools.metrics import _multi_horizon_test, _raise_if_validation_stage_failed
        from tools.val import make_report

        rank, world_size = get_dist_info()
        raw = runner.model.module if hasattr(runner.model, "module") else runner.model
        device = torch.cuda.current_device()
        rng = (
            random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state(device),
        )
        modes = [(module, module.training) for module in raw.modules()]
        flags = [
            (module, module.fp16_enabled)
            for module in raw.modules() if hasattr(module, "fp16_enabled")
        ]
        buffers = [(buffer, buffer.detach().clone()) for buffer in raw.buffers()]
        memory = getattr(raw, "memory", None)
        queue = getattr(raw, "queue", None)
        try:
            if memory is not None:
                raw.memory = {}
            if queue is not None:
                raw.queue = Queue()
            wrapped = MMDataParallel(raw, device_ids=[device])
            states = _multi_horizon_test(
                wrapped, self.loader, self.future_frames,
                distributed=world_size > 1, show_progress=False,
            )
            report_error = None
            if rank == 0:
                try:
                    report = make_report(
                        states, dict(epoch=epoch, model_spec=runner.meta["model_spec"])
                    )
                    report["precision"] = "live_mixed_precision"
                    report["seed"] = runner.meta["seed"]
                    report["dataset_samples"] = len(self.loader.dataset)
                    path = Path(runner.work_dir) / "validation" / f"epoch_{epoch:03d}_live.json"
                    write_json(path, report)
                    runner.logger.info(
                        "Epoch %d live forecast mean: %s", epoch, report["forecast_mean_1s_3s"]
                    )
                except Exception as exc:
                    report_error = exc
            _raise_if_validation_stage_failed(
                report_error, "live validation report", world_size > 1
            )
        finally:
            with torch.no_grad():
                for buffer, saved in buffers:
                    buffer.copy_(saved)
            for module, training in modes:
                module.training = training
            for module, flag in flags:
                module.fp16_enabled = flag
            if memory is not None:
                raw.memory = memory
            if queue is not None:
                raw.queue = queue
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])
            torch.cuda.set_rng_state(rng[3], device)
