#!/usr/bin/env python3
"""Apply the small compatibility edits needed by GrainWorld.

The installer downloads the official MMCV 1.7.2 and MMDetection3D 1.0.0rc6
source archives.  Those releases predate PyTorch 2.7 and CUDA 12.8, so their
sources need a few narrow, idempotent updates before they are installed.  This
script intentionally uses no repository commit, checksum, or machine contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence, Tuple


Replacement = Tuple[str, str]


def _rewrite(path: Path, replacements: Sequence[Replacement]) -> int:
    """Replace matching text in *path* and return the number of edits."""

    if not path.is_file():
        print(f"[patch] not present in this source release: {path}")
        return 0

    original = path.read_text(encoding="utf-8")
    updated = original
    changes = 0
    for old, new in replacements:
        occurrences = updated.count(old)
        if occurrences:
            updated = updated.replace(old, new)
            changes += occurrences

    if updated != original:
        path.write_text(updated, encoding="utf-8")
        print(f"[patch] updated {path} ({changes} replacement(s))")
    else:
        print(f"[patch] already compatible: {path}")
    return changes


def patch_mmcv(source: Path) -> int:
    """Patch removed PyTorch headers and a removed private DDP attribute."""

    source = source.resolve()
    if not (source / "setup.py").is_file():
        raise FileNotFoundError(f"MMCV source tree is incomplete: {source}")

    atomic_include = (("THC/THCAtomics.cuh", "ATen/cuda/Atomic.cuh"),)
    candidates: Iterable[Path] = (
        source / "mmcv/ops/csrc/common/pytorch_cuda_helper.hpp",
        source / "mmcv/ops/csrc/pytorch/cuda/ms_deform_attn_cuda.cu",
    )
    changes = sum(_rewrite(path, atomic_include) for path in candidates)

    # MMCV 1.7.2 already selects C++17 for PyTorch newer than 1.12.1.  This
    # conditional fallback also makes the patch safe for a repacked archive
    # where that version-aware block was flattened to C++14.
    setup_path = source / "setup.py"
    setup_text = setup_path.read_text(encoding="utf-8")
    has_version_aware_cpp17 = (
        "parse_version(torch.__version__)" in setup_text
        and "'-std=c++17'" in setup_text
    )
    if has_version_aware_cpp17:
        print(f"[patch] C++17 selection already present: {setup_path}")
    else:
        changes += _rewrite(
            setup_path,
            (
                ("extra_compile_args['cxx'] = ['-std=c++14']",
                 "extra_compile_args['cxx'] = ['-std=c++17']"),
                ("extra_compile_args['nvcc'] += ['-std=c++14']",
                 "extra_compile_args['nvcc'] += ['-std=c++17']"),
            ),
        )
    changes += _rewrite(
        source / "mmcv/parallel/distributed.py",
        (
            ("if self._use_replicated_tensor_module:",
             "if getattr(self, '_use_replicated_tensor_module', False):"),
            ("self._use_replicated_tensor_module else self.module",
             "getattr(self, '_use_replicated_tensor_module', False) "
             "else self.module"),
        ),
    )
    return changes


def patch_mmdet3d(source: Path) -> int:
    """Allow the tested MMCV version and record the tested runtime ranges."""

    source = source.resolve()
    if not (source / "setup.py").is_file():
        raise FileNotFoundError(
            f"MMDetection3D source tree is incomplete: {source}"
        )

    changes = _rewrite(
        source / "mmdet3d/__init__.py",
        (("mmcv_maximum_version = '1.7.0'",
          "mmcv_maximum_version = '1.7.2'"),),
    )
    changes += _rewrite(
        source / "requirements/runtime.txt",
        (
            ("networkx>=2.2,<2.3", "networkx>=3.1,<4"),
            ("numba==0.53.0", "numba>=0.57.1,<0.58"),
            ("trimesh>=2.35.39,<2.35.40", "trimesh>=3.23.5,<4"),
        ),
    )
    return changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmcv-source",
        type=Path,
        help="path to an extracted official MMCV 1.7.2 source archive",
    )
    parser.add_argument(
        "--mmdet3d-source",
        type=Path,
        help="path to an extracted official MMDetection3D 1.0.0rc6 archive",
    )
    args = parser.parse_args()
    if args.mmcv_source is None and args.mmdet3d_source is None:
        parser.error("provide --mmcv-source and/or --mmdet3d-source")
    return args


def main() -> int:
    args = parse_args()
    changes = 0
    if args.mmcv_source is not None:
        changes += patch_mmcv(args.mmcv_source)
    if args.mmdet3d_source is not None:
        changes += patch_mmdet3d(args.mmdet3d_source)
    print(f"[patch] complete; {changes} replacement(s) applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
