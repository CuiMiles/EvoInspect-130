#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_prefix="${EVOINSPECT_EFFICIENTAD_ENV:-/home/CuiMinghao/envs/evoinspect-efficientad}"
conda_exe="${EVOINSPECT_CONDA:-/home/CuiMinghao/miniforge3/bin/conda}"
upstream="${repo_root}/third_party/anomalib-2.3.0"
expected_commit="091ca6aca92c8d0e416394f79e52f5a3cea3db73"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ -x "${conda_exe}" ]] || die "conda executable missing: ${conda_exe}"
[[ -d "${upstream}/.git" ]] || die "pinned anomalib checkout missing: ${upstream}"
[[ "$(git -C "${upstream}" rev-parse HEAD)" == "${expected_commit}" ]] \
  || die "anomalib commit mismatch"
[[ -z "$(git -C "${upstream}" status --short)" ]] || die "anomalib checkout is modified"

if [[ ! -x "${env_prefix}/bin/python" ]]; then
  "${conda_exe}" create -y -p "${env_prefix}" python=3.11 pip
fi

python_exe="${env_prefix}/bin/python"
"${python_exe}" -m pip install --upgrade "pip<26" wheel
# Install the pinned local source. CUDA Torch is explicitly pinned so this environment never
# mutates the already-working PatchCore or project environments.
"${python_exe}" -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
"${python_exe}" -m pip install -e "${upstream}"
"${python_exe}" -m pip install -e "${repo_root}[metrics,yaml,dev]"

cd "${repo_root}"
env PYTHONPATH="${repo_root}/src:${upstream}/src" "${python_exe}" - <<'PY'
import anomalib
import torch
from pathlib import Path
from anomalib.data.utils import download_and_extract
from anomalib.models import EfficientAd
from anomalib.models.image.efficient_ad.lightning_model import (
    IMAGENETTE_DOWNLOAD_INFO,
    WEIGHTS_DOWNLOAD_INFO,
)

print({"anomalib": anomalib.__version__, "torch": torch.__version__, "cuda": torch.cuda.is_available()})
EfficientAd(model_size="small")
teacher_root = Path("pre_trained")
if not (teacher_root / "efficientad_pretrained_weights").is_dir():
    download_and_extract(teacher_root, WEIGHTS_DOWNLOAD_INFO)
imagenette_root = Path("/home/CuiMinghao/models/evoinspect/imagenette")
if not imagenette_root.is_dir() or not any(imagenette_root.iterdir()):
    download_and_extract(imagenette_root, IMAGENETTE_DOWNLOAD_INFO)
PY
printf 'EfficientAD environment ready: %s\n' "${env_prefix}"
