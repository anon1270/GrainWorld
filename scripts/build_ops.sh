#!/usr/bin/env bash
# Rebuild only the repository-local sampler using the active Python/CUDA stack.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
if [[ -z "${CUDA_HOME:-}" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/nvcc" ]]; then
    export CUDA_HOME="$CONDA_PREFIX"
fi
export MAX_JOBS="${MAX_JOBS:-4}"
cd "$ROOT/models/csrc"
"$PYTHON" setup.py build_ext --inplace --force
cd "$ROOT"
"$PYTHON" - <<'PY'
from models.csrc.wrapper import MSMV_CUDA
if not MSMV_CUDA:
    raise SystemExit('MSMV CUDA extension import failed after build.')
print('MSMV CUDA extension is available.')
PY
