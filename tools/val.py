#!/usr/bin/env python3
"""FP32 final-stage validation with camera-visible 1-3 s forecasting metrics."""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.cli import CONFIG, make_parser


def make_report(states, metadata=None):
    from tools.metrics import _summarize_metric_state
    horizons=[]
    for state in states:
        metrics,classes=_summarize_metric_state(state)
        horizons.append(dict(future_index=int(state['future_index']),
            seconds=0.5*state['future_index'],metrics=metrics,class_metrics=classes))
    future=[x for x in horizons if x['future_index'] in (2,4,6)]
    if len(future)!=3:
        raise ValueError('Expected all three future horizons 1, 2, 3 s')
    mean={key:sum(float(x['metrics'][key]) for x in future)/3
          for key in ('Semantic mIoU','Binary IoU')}
    return dict(model='GrainWorld',model_spec=(metadata or {}).get('model_spec'),
        horizons=horizons,forecast_mean_1s_3s=mean,future_frames=[0,2,4,6],
        evaluated_samples_per_horizon={str(x['future_index']):x['class_metrics']['evaluated_samples'] for x in horizons},
        evaluation_mask='camera',reconstruction_in_forecast_mean=False,
        scene_end_policy='exclude unavailable future targets',
        metric_aggregation='arithmetic mean of the three reported horizon scores',
        checkpoint_epoch=(metadata or {}).get('epoch'))


def main():
    args=make_parser('val').parse_args()
    if args.dry_run:
        from tools.launch import launch
        return launch('val',sys.argv[1:])
    import torch
    import torch.distributed as dist
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmdet.apis import set_random_seed
    from mmdet3d.datasets import build_dataset,build_dataloader
    from mmdet3d.models import build_model
    from tools.runtime import (apply_paths,check_data_files,initialize_device,require_cuda_sampler,
                               init_logging,load_full_model,model_spec,write_json)
    local_rank,rank,world_size=initialize_device(args.local_rank)
    try:
        import models
        import loaders
        require_cuda_sampler()
        logger=init_logging(rank=rank)
        cfg=apply_paths(Config.fromfile(str(CONFIG)),args)
        check_data_files(cfg,training=False)
        set_random_seed(args.seed,deterministic=True)
        torch.backends.cudnn.benchmark=False
        dataset=build_dataset(cfg.data.val)
        data_loader=build_dataloader(dataset,samples_per_gpu=1,workers_per_gpu=args.workers,
            num_gpus=world_size,dist=world_size>1,shuffle=False,seed=args.seed)
        model=build_model(cfg.model)
        model.cuda(local_rank)
        metadata=load_full_model(model,args.weights)
        spec=model_spec(model)
        model.float()
        for module in model.modules():
            if hasattr(module,'fp16_enabled'):
                module.fp16_enabled=False
        # No gradient synchronization is needed during inference.
        wrapped=MMDataParallel(model,device_ids=[local_rank])
        from tools.metrics import _multi_horizon_test
        logger.info('FP32 validation, camera mask, outputs 0/1/2/3 s, forecast mean 1/2/3 s.')
        states=_multi_horizon_test(wrapped,data_loader,cfg.future_frames,distributed=world_size>1)
        if rank==0:
            report=make_report(states,metadata)
            report.update(model_spec=spec,weights=str(Path(args.weights).expanduser().resolve()),
                data_root=cfg.dataset_root,occ_root=cfg.occ_root,ann_file=cfg.data.val.ann_file,
                seed=args.seed,precision='fp32',world_size=world_size,local_batch_size=1,
                dataset_samples=len(dataset),collected_predictions=len(dataset)*len(cfg.future_frames))
            write_json(args.output_json,report)
            logger.info('Forecast mean: %s',report['forecast_mean_1s_3s'])
            logger.info('Saved: %s',Path(args.output_json).resolve())
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__=='__main__':
    main()
