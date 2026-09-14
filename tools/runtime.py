"""Portable paths, checkpoint loading and the fixed public model definition."""
from __future__ import annotations
from collections import OrderedDict
import json
import logging
import os
from pathlib import Path


def absolute(path):
    return str(Path(path).expanduser().resolve())


def apply_paths(cfg, args):
    data_root = absolute(args.data_root)
    occ_root = absolute(args.occ_root or Path(data_root)/'gts')
    ann_root = Path(absolute(args.ann_root or data_root))
    cfg.dataset_root = data_root + os.sep
    cfg.occ_root = occ_root + os.sep
    # Existing data-loader helper names are kept for checkpoint-code compatibility.
    os.environ['DUPLEXWORLD_DATA_ROOT'] = data_root
    os.environ['DUPLEXWORLD_OCC_ROOT'] = occ_root
    os.environ['GRAINWORLD_DATA_ROOT'] = data_root
    os.environ['GRAINWORLD_OCC_ROOT'] = occ_root
    for split in ('train', 'val'):
        split_cfg = cfg['data'][split]
        split_cfg['data_root'] = cfg.dataset_root
        provided = getattr(args, f'{split}_ann_file', None)
        split_cfg['ann_file'] = absolute(provided or ann_root/f'nuscenes_infos_{split}_sweep_occ.pkl')
        for step in split_cfg['pipeline']:
            if step.get('type') == 'LoadOccFromFile':
                step['occ_root'] = cfg.occ_root
            if split == 'val' and step.get('type') == 'LoadMultiViewImageFromMultiSweeps':
                step['test_mode'] = True
                step['force_offline'] = True
    cfg['data']['workers_per_gpu'] = args.workers
    return cfg


def dataset_occ_root(dataset):
    return Path(os.environ.get('GRAINWORLD_OCC_ROOT') or
                os.environ.get('DUPLEXWORLD_OCC_ROOT') or Path(dataset.data_root)/'gts')


def check_data_files(cfg, training):
    root = Path(cfg.dataset_root)
    for directory in (root/'samples', root/'sweeps', root/'v1.0-trainval', Path(cfg.occ_root)):
        if not directory.is_dir():
            raise FileNotFoundError(f'Required data directory not found: {directory}')
    splits = ('train', 'val') if training else ('val',)
    for split in splits:
        path = Path(cfg['data'][split]['ann_file'])
        if not path.is_file():
            raise FileNotFoundError(f'{split} annotation PKL not found: {path}. See README.md.')


def initialize_device(local_rank=0):
    import torch
    import torch.distributed as dist
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for GrainWorld training and validation.')
    local_rank = int(os.environ.get('LOCAL_RANK', local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group('nccl', init_method='env://')
    rank = dist.get_rank() if dist.is_initialized() else 0
    return local_rank, rank, world_size


def require_cuda_sampler():
    from models.csrc.wrapper import MSMV_CUDA
    if not MSMV_CUDA:
        raise RuntimeError('MSMV CUDA extension is not loaded. Run bash scripts/build_ops.sh in this repository. '
                           'The inherited Python fallback is not used for published training/evaluation.')


def init_logging(path=None, rank=0):
    logger = logging.getLogger('grainworld')
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if path is not None and rank == 0:
        file_handler = logging.FileHandler(path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def read_checkpoint(path):
    import torch
    # OpenMMLab training files contain optimizer/metadata objects, not tensors only.
    # Only load checkpoint files obtained from a trusted source.
    checkpoint = torch.load(absolute(path), map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError('Checkpoint must be a state dict or an OpenMMLab checkpoint dictionary.')
    raw = checkpoint.get('state_dict', checkpoint)
    if not isinstance(raw, dict):
        raise TypeError('checkpoint.state_dict must be a dictionary')
    state = OrderedDict()
    for key, value in raw.items():
        if not isinstance(key, str):
            raise TypeError('State-dict keys must be strings')
        while key.startswith('module.'):
            key = key[len('module.'):]
        if key in state:
            raise ValueError(f'Duplicate normalized checkpoint key: {key}')
        state[key] = value
    return checkpoint, state


def load_backbone(model, path):
    checkpoint, state = read_checkpoint(path)
    if any(key.startswith(('pts_bbox_head.', 'img_neck.')) for key in state):
        raise ValueError('Fresh training needs a backbone pretrain, not a trained occupancy checkpoint.')
    subset = OrderedDict()
    for key, value in state.items():
        for prefix in ('backbone.', 'img_backbone.'):
            if key.startswith(prefix):
                subset[key[len(prefix):]] = value
                break
    if not subset:
        raise ValueError('No backbone.* or img_backbone.* parameters found in pretrain checkpoint.')
    # Detector heads/neck from the image pretrain are intentionally not loaded.
    model.img_backbone.load_state_dict(subset, strict=True)
    return len(subset)


def load_full_model(model, path):
    checkpoint, state = read_checkpoint(path)
    # Optional historical metadata never controls loading. Model keys/shapes do.
    model.load_state_dict(state, strict=True)
    return checkpoint.get('meta', {}) or {}


def model_spec(model):
    decoder = model.pts_bbox_head.transformer.decoder
    layers = list(decoder.decoder_layers)
    spec = dict(global_rows=int(decoder.q4occ_scene_encoder.latent_tokens.shape[1]),
                local_rows_per_anchor=int(decoder.q4occ_local_memory_tokens),
                anchors=int(model.pts_bbox_head.num_query),
                channels=int(decoder.q4occ_scene_encoder.latent_tokens.shape[-1]),
                refinement_stages=len(layers),
                global_readers=sum(hasattr(x,'q4occ_global_memory_fusion') for x in layers),
                local_readers=sum(hasattr(x,'q4occ_local_memory_fusion') for x in layers),
                global_enabled=bool(decoder.q4occ.get('joint_global_memory_enabled')),
                local_enabled=bool(decoder.q4occ.get('joint_local_memory_enabled')),
                local_scope=decoder.q4occ.get('joint_local_memory_scope'),
                horizons=list(model.pts_bbox_head.future_frames))
    expected = dict(global_rows=448, local_rows_per_anchor=192, anchors=600, channels=256,
                    refinement_stages=6, global_readers=6, local_readers=6,
                    global_enabled=True, local_enabled=True, local_scope='all', horizons=[0,2,4,6])
    if spec != expected:
        raise ValueError(f'This release supports G448+L192 only. Constructed model: {spec}')
    return spec


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+'\n',encoding='utf-8')
        os.replace(temporary,path)
    finally:
        if temporary.exists():
            temporary.unlink()
