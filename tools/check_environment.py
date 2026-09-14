#!/usr/bin/env python3
"""Check the active environment without changing installed packages."""
import argparse
import importlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--require-gpu',action='store_true')
    p.add_argument('--cuda-smoke-test',action='store_true')
    args=p.parse_args()
    errors=[]
    for name in ('torch','mmcv','mmdet','mmdet3d','nuscenes','cv2','numpy'):
        try:
            mod=importlib.import_module(name)
            print(f'{name}: {getattr(mod,"__version__","import OK")}')
        except Exception as exc:
            errors.append(f'{name}: {exc}')
    try:
        import torch
        from mmcv.ops import Voxelization,knn
        import models
        import loaders
        from models.csrc.wrapper import MSMV_CUDA
        if not MSMV_CUDA:
            errors.append('Repository MSMV extension is unavailable. Run bash scripts/build_ops.sh.')
        if args.require_gpu and not torch.cuda.is_available():
            errors.append('No CUDA GPU is visible.')
        if args.cuda_smoke_test:
            from models.csrc.wrapper import msmv_sampling
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA smoke test requires a GPU.')
            torch.cuda.set_device(0)
            features=[torch.randn(1,6,8,8,8,device='cuda',requires_grad=True) for _ in range(4)]
            positions=torch.rand(1,2,4,3,device='cuda',requires_grad=True)
            weights=torch.randn(1,2,4,4,device='cuda').softmax(-1).requires_grad_()
            msmv_sampling(features,positions,weights).sum().backward()
            torch.cuda.synchronize()
            print('MSMV CUDA forward/backward OK')
    except Exception as exc:
        errors.append(str(exc))
    for message in errors:
        print('ERROR:',message,file=sys.stderr)
    return 1 if errors else 0


if __name__=='__main__':
    raise SystemExit(main())
