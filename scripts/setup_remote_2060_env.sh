#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bootstrap="${EVOINSPECT_BOOTSTRAP_PYTHON:-python3}"
env_dir="${EVOINSPECT_REMOTE_ENV:-${repo_root}/.venv-2060}"
upstream="${repo_root}/third_party/anomalib-2.3.0"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
[[ -f "${upstream}/pyproject.toml" ]] || die "bundled anomalib source is missing"
"${python_bootstrap}" -m venv "${env_dir}"
"${env_dir}/bin/python" -m pip install --upgrade "pip<26" wheel
"${env_dir}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
"${env_dir}/bin/python" -m pip install --only-binary=:all: imagecodecs==2026.1.14
"${env_dir}/bin/python" -m pip install -e "${upstream}"
"${env_dir}/bin/python" -m pip install -e "${repo_root}[yaml,metrics,images]"
"${env_dir}/bin/python" -m pip check
PYTHONPATH="${repo_root}/src:${repo_root}:${upstream}/src" "${env_dir}/bin/python" - <<'PY'
import anomalib
import torch
from anomalib.models import EfficientAd

assert torch.cuda.is_available(), "CUDA is not available"
EfficientAd(model_size="medium")
print({"anomalib": anomalib.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
PY
printf 'Remote environment ready: %s\n' "${env_dir}"
