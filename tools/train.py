#!/usr/bin/env python3
"""Fresh G448+L192 training from the NuImages ResNet-50 pretrain."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.cli import CONFIG, make_parser, batch_settings


def main():
    parser = make_parser('train')
    args = parser.parse_args()
    if args.dry_run:
        from tools.launch import launch
        return launch('train', sys.argv[1:])
    import numpy as np
    import torch
    import torch.distributed as dist
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
    from mmcv.runner import EpochBasedRunner, build_optimizer
    from mmdet.apis import set_random_seed
    from mmdet3d.datasets import build_dataset, build_dataloader as build_val_loader
    from mmdet3d.models import build_model
    from tools.runtime import (apply_paths, check_data_files, initialize_device, require_cuda_sampler,
                               init_logging, load_backbone, model_spec, write_json)
    local_rank, rank, world_size = initialize_device(args.local_rank)
    try:
        micro, accumulation, warmup = batch_settings(args,world_size)
        import models  # register the existing detector, head and memory classes
        import loaders
        from loaders.builder import build_dataloader
        require_cuda_sampler()
        cfg = apply_paths(Config.fromfile(str(CONFIG)),args)
        check_data_files(cfg,training=True)
        cfg.total_epochs = args.epochs
        cfg.batch_size = micro
        cfg.optimizer.lr = args.lr
        cfg.optimizer_config.cumulative_iters = accumulation
        cfg.lr_config.warmup_iters = warmup
        if not warmup:
            cfg.lr_config.warmup = None
        cfg.checkpoint_config.interval = args.save_interval
        cfg.validation_interval = args.val_interval
        cfg.seed = args.seed
        cfg.load_from = str(Path(args.pretrained).expanduser().resolve())
        cfg.resume_from = None
        work_dir = Path(args.work_dir).expanduser().resolve()
        # Do not silently turn a fresh experiment into a checkpoint overwrite.
        if work_dir.exists() and any(work_dir.glob('*.pth')):
            raise FileExistsError(f'Fresh training requires a directory without checkpoints: {work_dir}')
        work_dir.mkdir(parents=True,exist_ok=True)
        logger = init_logging(work_dir/'train.log',rank)
        set_random_seed(args.seed,deterministic=True)
        torch.backends.cudnn.benchmark = False
        train_dataset = build_dataset(cfg.data.train)
        train_loader = build_dataloader(train_dataset,samples_per_gpu=args.batch_size,
            workers_per_gpu=args.workers,num_gpus=world_size,dist=world_size>1,
            shuffle=True,seed=args.seed)
        # Match the existing four-GPU run's non-invasive periodic monitoring.
        val_loader = None
        if args.val_interval and world_size > 1:
            rng = (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state(local_rank))
            try:
                val_dataset = build_dataset(copy.deepcopy(cfg.data.val))
                val_loader = build_val_loader(val_dataset,samples_per_gpu=1,
                    workers_per_gpu=args.workers,num_gpus=world_size,dist=True,shuffle=False,seed=args.seed)
            finally:
                random.setstate(rng[0]); np.random.set_state(rng[1])
                torch.set_rng_state(rng[2]); torch.cuda.set_rng_state(rng[3],local_rank)
        elif args.val_interval:
            logger.info('Single GPU: periodic validation is disabled. Use scripts/val.sh on saved checkpoints.')
        raw_model = build_model(cfg.model)
        raw_model.init_weights()
        raw_model.cuda(local_rank)
        raw_model.train()
        spec = model_spec(raw_model)
        if world_size > 1:
            model = MMDistributedDataParallel(raw_model,[local_rank],broadcast_buffers=False,
                                            find_unused_parameters=False)
        else:
            model = MMDataParallel(raw_model,[local_rank])
        optimizer = build_optimizer(model,cfg.optimizer)
        metadata = dict(model='GrainWorld',variant='grainworld_g448_l192',model_spec=spec,
            seed=args.seed,precision='fp16',total_batch_size=micro,local_batch_size=args.batch_size,
            effective_batch_size=args.effective_batch,gradient_cumulative_iters=accumulation,
            warmup_micro_iterations=warmup,initial_checkpoint_kind='backbone-pretrain',
            initial_checkpoint=cfg.load_from)
        runner = EpochBasedRunner(model,optimizer=optimizer,work_dir=str(work_dir),
                                  logger=logger,max_epochs=args.epochs,meta=metadata)
        runner.register_lr_hook(cfg.lr_config)
        runner.register_optimizer_hook(cfg.optimizer_config)
        runner.register_checkpoint_hook(cfg.checkpoint_config)
        runner.register_logger_hooks(cfg.log_config)
        runner.register_timer_hook(dict(type='IterTimerHook'))
        runner.register_custom_hooks(dict(type='DistSamplerSeedHook'))
        if val_loader is not None:
            from tools.validation_hook import ValidationHook
            runner.register_hook(ValidationHook(val_loader,cfg.future_frames,args.val_interval),priority='VERY_LOW')
        # This occurs after construction exactly as in the previous fresh runner.
        loaded = load_backbone(raw_model,args.pretrained)
        if rank == 0:
            cfg.dump(str(work_dir/'resolved_config.py'))
            write_json(work_dir/'run_config.json',dict(arguments=vars(args),model_spec=spec,
                global_micro_batch=micro,gradient_accumulation=accumulation,
                warmup_micro_iterations=warmup,pretrained_backbone_tensors=loaded,
                periodic_validation=val_loader is not None,torch_version=str(torch.__version__)))
        logger.info('GrainWorld G448+L192. Loaded %d backbone tensors. Fresh LayerScale = 1e-3.',loaded)
        runner.run([train_loader],[('train',1)])
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
