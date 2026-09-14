#!/usr/bin/env python3
"""Launch one fresh-training or FP32-validation worker per selected GPU."""
from __future__ import annotations
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.cli import make_parser, resolve_gpus, batch_settings


def launch(mode, arguments):
    parser = make_parser(mode)
    args = parser.parse_args(arguments)
    try:
        devices, count = resolve_gpus(args.gpus)
        if mode == 'train':
            micro, accumulation, warmup = batch_settings(args, count)
            print(f'[GrainWorld] G448+L192 | batch {micro} x accumulation {accumulation}'
                  f' = {args.effective_batch} | epochs {args.epochs} | lr {args.lr:g}', flush=True)
    except ValueError as exc:
        parser.error(str(exc))
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={count}', str(ROOT/'tools'/f'{mode}.py')]
    # Serialize from the shared parser so no shell-only options leak downstream.
    for name, value in vars(args).items():
        if name in ('gpus', 'dry_run', 'local_rank') or value is None:
            continue
        command += ['--' + name.replace('_', '-'), str(value)]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = devices
    env.setdefault('OMP_NUM_THREADS', '1')
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    print('[RUN] CUDA_VISIBLE_DEVICES=' + shlex.quote(devices) + ' ' + shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('train', 'val'):
        raise SystemExit('Usage: tools/launch.py {train|val} [options]')
    raise SystemExit(launch(sys.argv[1], sys.argv[2:]))
