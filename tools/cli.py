"""Shared command-line definitions for the shell launchers and workers."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs' / 'grainworld_g448_l192.py'


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('must be non-negative')
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return number


def make_parser(mode):
    if mode not in ('train', 'val'):
        raise ValueError('mode must be train or val')
    p = argparse.ArgumentParser(description=f'GrainWorld G448+L192 {mode}')
    p.add_argument('--data-root', required=True, help='nuScenes root containing samples/, sweeps/ and v1.0-trainval/')
    p.add_argument('--occ-root', help='Occ3D gts root, defaults to DATA_ROOT/gts')
    p.add_argument('--ann-root', help='directory containing the prepared sweep/occupancy info PKLs')
    p.add_argument('--val-ann-file', help='explicit validation PKL, overrides --ann-root')
    p.add_argument('--gpus', default=None, help='comma-separated visible GPU IDs or GPU UUIDs')
    p.add_argument('--workers', type=nonnegative_int, default=4, help='data-loader workers per GPU')
    p.add_argument('--seed', type=nonnegative_int, default=0)
    p.add_argument('--dry-run', action='store_true', help='print the launch command without loading data or weights')
    p.add_argument('--local-rank', '--local_rank', type=int, default=0, help=argparse.SUPPRESS)
    if mode == 'train':
        p.add_argument('--pretrained', required=True, help='NuImages R50 backbone pretrain, not an occupancy checkpoint')
        p.add_argument('--train-ann-file', help='explicit training PKL, overrides --ann-root')
        p.add_argument('--work-dir', default='work_dirs/grainworld_g448_l192')
        p.add_argument('--epochs', type=positive_int, default=80)
        p.add_argument('--batch-size', '--local-batch', dest='batch_size', type=positive_int, default=2,
                       help='training samples per GPU')
        p.add_argument('--effective-batch', type=positive_int, default=8)
        p.add_argument('--lr', type=positive_float, default=3e-4)
        p.add_argument('--warmup-updates', type=nonnegative_int, default=500)
        p.add_argument('--val-interval', type=nonnegative_int, default=5,
                       help='live mixed-precision validation interval, 0 disables it')
        p.add_argument('--save-interval', type=positive_int, default=5)
    else:
        p.add_argument('--weights', required=True, help='full G448+L192 checkpoint')
        p.add_argument('--output-json', default='work_dirs/validation.json')
    return p


def resolve_gpus(value=None):
    selected = value or os.environ.get('CUDA_VISIBLE_DEVICES') or '0'
    devices = [part.strip() for part in selected.split(',')]
    if any(not x or x == '-1' for x in devices) or len(set(devices)) != len(devices):
        raise ValueError('--gpus must contain distinct nonempty device IDs')
    return ','.join(devices), len(devices)


def batch_settings(args, world_size):
    micro = args.batch_size * world_size
    if args.effective_batch % micro:
        raise ValueError(
            f'effective batch {args.effective_batch} must be divisible by '
            f'{world_size} GPUs x {args.batch_size} samples/GPU = {micro}')
    accumulation = args.effective_batch // micro
    return micro, accumulation, args.warmup_updates * accumulation
