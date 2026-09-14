#!/usr/bin/env bash
# Install the supported GrainWorld stack into the currently active environment.
# No Git checkout is used: legacy OpenMMLab releases come from official source
# archives and are patched locally before compilation.
set -Eeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
MMCV_VERSION=1.7.2
MMDET3D_VERSION=1.0.0rc6
PYTORCH_INDEX_URL=${GRAINWORLD_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
MMCV_ARCHIVE_URL=${GRAINWORLD_MMCV_ARCHIVE_URL:-https://github.com/open-mmlab/mmcv/archive/refs/tags/v${MMCV_VERSION}.tar.gz}
MMDET3D_ARCHIVE_URL=${GRAINWORLD_MMDET3D_ARCHIVE_URL:-https://github.com/open-mmlab/mmdetection3d/archive/refs/tags/v${MMDET3D_VERSION}.tar.gz}

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    DEFAULT_BUILD_ROOT="${CONDA_PREFIX}/var/cache/grainworld/openmmlab"
else
    DEFAULT_BUILD_ROOT="${TMPDIR:-/tmp}/grainworld-openmmlab-${UID:-0}"
fi
BUILD_ROOT=${GRAINWORLD_BUILD_DIR:-$DEFAULT_BUILD_ROOT}

log() {
    printf '[GrainWorld setup] %s\n' "$*" >&2
}

warn() {
    printf '[GrainWorld setup] WARNING: %s\n' "$*" >&2
}

die() {
    printf '[GrainWorld setup] ERROR: %s\n' "$*" >&2
    exit 1
}

is_true() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

command -v python >/dev/null 2>&1 || die "python is not on PATH"
command -v tar >/dev/null 2>&1 || die "tar is required to unpack source archives"

if [[ "$(uname -s)" != Linux ]]; then
    warn "the supported extension build target is Linux x86-64"
fi
if [[ "$(uname -m)" != x86_64 ]]; then
    warn "the source-build recipe targets x86-64, not $(uname -m)"
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    warn "no active Conda environment detected; installation will use $(command -v python)"
fi

PYTHON_MINOR=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_MINOR" != 3.10 ]]; then
    warn "the packaged recipe targets Python 3.10; current interpreter is $PYTHON_MINOR"
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
    export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
elif command -v nvcc >/dev/null 2>&1; then
    NVCC_PATH=$(command -v nvcc)
    export CUDA_HOME=${CUDA_HOME:-$(cd "$(dirname "$NVCC_PATH")/.." && pwd -P)}
else
    die "nvcc was not found; create the environment from environment.yml first"
fi

export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if [[ -z "${CC:-}" && -n "${CONDA_PREFIX:-}" ]]; then
    for candidate in \
        "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc" \
        "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"; do
        if [[ -x "$candidate" ]]; then
            export CC=$candidate
            break
        fi
    done
fi
if [[ -z "${CXX:-}" && -n "${CONDA_PREFIX:-}" ]]; then
    for candidate in \
        "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++" \
        "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"; do
        if [[ -x "$candidate" ]]; then
            export CXX=$candidate
            break
        fi
    done
fi

NVCC_RELEASE=$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | tail -1)
log "Python $PYTHON_MINOR; CUDA_HOME=$CUDA_HOME; nvcc=${NVCC_RELEASE:-unknown}"

if [[ -n "${GRAINWORLD_MAX_JOBS:-}" ]]; then
    BUILD_JOBS=$GRAINWORLD_MAX_JOBS
elif [[ -n "${MAX_JOBS:-}" ]]; then
    BUILD_JOBS=$MAX_JOBS
else
    BUILD_JOBS=$(python - <<'PY'
import os
print(max(1, min(os.cpu_count() or 1, 4)))
PY
)
fi
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || die "build job count must be a positive integer"
export MAX_JOBS=$BUILD_JOBS

log "installing PyTorch build tools"
python -m pip install \
    pip==25.1.1 setuptools==78.1.0 wheel==0.43.0 \
    packaging==24.1 ninja==1.11.1.1 cmake==3.30.5 \
    psutil==6.0.0 Cython==3.0.10

if ! is_true "${GRAINWORLD_SKIP_TORCH_INSTALL:-0}"; then
    log "installing PyTorch 2.7.1 / CUDA 12.8 wheels"
    python -m pip install \
        torch==2.7.1 torchvision==0.22.1 \
        --index-url "$PYTORCH_INDEX_URL"
else
    log "keeping the caller-provided PyTorch installation"
fi

log "installing pinned Python runtime dependencies"
python -m pip install -r "$REPO_ROOT/requirements.txt"

# Both SDKs declare the GUI opencv-python distribution even though GrainWorld
# only needs cv2 APIs provided by opencv-python-headless.  Installing the SDKs
# without dependency resolution avoids shipping two conflicting cv2 wheels.
log "installing dataset SDKs without GUI OpenCV"
python -m pip install --no-deps \
    nuscenes-devkit==1.1.10 lyft-dataset-sdk==0.0.8

detect_arch_list() {
    python - <<'PY'
import torch

caps = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(index)
        if cap not in caps:
            caps.append(cap)
caps.sort()
if caps:
    values = [f"{major}.{minor}" for major, minor in caps]
    values[-1] += "+PTX"
    print(";".join(values))
else:
    # Build-node fallback: Ampere, Ada, and Blackwell workstation GPUs.
    print("8.0;8.6;8.9;12.0+PTX")
PY
}

CUDA_ARCH_LIST=${GRAINWORLD_CUDA_ARCH_LIST:-$(detect_arch_list)}
export TORCH_CUDA_ARCH_LIST=$CUDA_ARCH_LIST
log "CUDA architectures: $TORCH_CUDA_ARCH_LIST; MAX_JOBS=$MAX_JOBS"

mkdir -p "$BUILD_ROOT/archives" "$BUILD_ROOT/sources"
CLEANUP_PATHS=()
cleanup() {
    local path
    for path in "${CLEANUP_PATHS[@]:-}"; do
        if [[ -n "$path" && -d "$path" ]]; then
            rm -rf -- "$path"
        fi
    done
}
trap cleanup EXIT

download_archive() {
    local url=$1
    local destination=$2
    if [[ -s "$destination" ]]; then
        log "reusing archive $destination"
        return
    fi
    local partial
    partial=$(mktemp "${destination}.part.XXXXXX")
    log "downloading $url"
    python - "$url" "$partial" <<'PY'
import shutil
import sys
import urllib.request

url, destination = sys.argv[1:]
request = urllib.request.Request(url, headers={"User-Agent": "GrainWorld-installer"})
with urllib.request.urlopen(request, timeout=120) as response:
    with open(destination, "wb") as output:
        shutil.copyfileobj(response, output)
PY
    mv "$partial" "$destination"
}

prepare_official_source() {
    local name=$1
    local url=$2
    local archive_root=$3
    local source_dir="$BUILD_ROOT/sources/$archive_root"
    local archive="$BUILD_ROOT/archives/${archive_root}.tar.gz"

    if [[ -f "$source_dir/setup.py" ]]; then
        log "reusing extracted $name source at $source_dir"
        printf '%s\n' "$source_dir"
        return
    fi
    if [[ -e "$source_dir" ]]; then
        die "incomplete cached source at $source_dir; move it aside and rerun"
    fi

    download_archive "$url" "$archive"
    local extract_dir
    extract_dir=$(mktemp -d "$BUILD_ROOT/sources/.extract-${name}.XXXXXX")
    CLEANUP_PATHS+=("$extract_dir")
    tar -xzf "$archive" -C "$extract_dir"
    [[ -f "$extract_dir/$archive_root/setup.py" ]] || \
        die "archive did not contain expected directory $archive_root"
    mv "$extract_dir/$archive_root" "$source_dir"
    printf '%s\n' "$source_dir"
}

if [[ -n "${GRAINWORLD_MMCV_SOURCE:-}" ]]; then
    MMCV_SOURCE=$(cd "$GRAINWORLD_MMCV_SOURCE" && pwd -P)
    [[ -f "$MMCV_SOURCE/setup.py" ]] || die "invalid GRAINWORLD_MMCV_SOURCE"
else
    MMCV_SOURCE=$(prepare_official_source \
        mmcv "$MMCV_ARCHIVE_URL" "mmcv-${MMCV_VERSION}")
fi

if [[ -n "${GRAINWORLD_MMDET3D_SOURCE:-}" ]]; then
    MMDET3D_SOURCE=$(cd "$GRAINWORLD_MMDET3D_SOURCE" && pwd -P)
    [[ -f "$MMDET3D_SOURCE/setup.py" ]] || die "invalid GRAINWORLD_MMDET3D_SOURCE"
else
    MMDET3D_SOURCE=$(prepare_official_source \
        mmdetection3d "$MMDET3D_ARCHIVE_URL" \
        "mmdetection3d-${MMDET3D_VERSION}")
fi

log "applying idempotent PyTorch 2.7 compatibility patches"
python "$REPO_ROOT/tools/patch_legacy_openmmlab.py" \
    --mmcv-source "$MMCV_SOURCE" \
    --mmdet3d-source "$MMDET3D_SOURCE"

mmcv_is_ready() {
    python - <<PY >/dev/null 2>&1
import mmcv
assert mmcv.__version__ == "${MMCV_VERSION}"
import mmcv._ext  # noqa: F401
from mmcv.ops import Voxelization, knn  # noqa: F401
PY
}

if is_true "${GRAINWORLD_FORCE_REBUILD:-0}" || ! mmcv_is_ready; then
    log "building MMCV-full $MMCV_VERSION from official patched source"
    python -m pip uninstall -y mmcv mmcv-full >/dev/null 2>&1 || true
    (
        cd "$MMCV_SOURCE"
        MMCV_WITH_OPS=1 FORCE_CUDA=1 \
        TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
        MAX_JOBS="$MAX_JOBS" \
        python -m pip install -v --no-cache-dir --no-deps \
            --no-build-isolation .
    )
else
    log "compatible MMCV-full $MMCV_VERSION with CUDA ops is already installed"
fi

log "installing legacy OpenMMLab Python packages without resolver substitutions"
python -m pip install --no-deps \
    mmcls==0.25.0 mmsegmentation==0.30.0 mmdet==2.28.2

mmdet3d_is_ready() {
    python - <<PY >/dev/null 2>&1
import mmdet3d
assert mmdet3d.__version__ == "${MMDET3D_VERSION}"
PY
}

if is_true "${GRAINWORLD_FORCE_REBUILD:-0}" || ! mmdet3d_is_ready; then
    log "installing MMDetection3D $MMDET3D_VERSION from official patched source"
    python -m pip install -v --no-cache-dir --no-deps \
        --no-build-isolation --force-reinstall "$MMDET3D_SOURCE"
else
    log "compatible MMDetection3D $MMDET3D_VERSION is already installed"
fi

log "building the GrainWorld MSMV CUDA extension from source"
(
    cd "$REPO_ROOT/models/csrc"
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    MAX_JOBS="$MAX_JOBS" \
    python setup.py build_ext --inplace --force
)

log "running environment import checks"
python "$REPO_ROOT/tools/check_environment.py"

log "installation complete"
