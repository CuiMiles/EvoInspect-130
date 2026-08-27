#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
payload="${EVOINSPECT_DEPLOYMENT_PAYLOAD:-${repo_root}/deployment_payload}"
python_exe="${EVOINSPECT_REMOTE_PYTHON:-${repo_root}/.venv-2060/bin/python}"
gpu="${EVOINSPECT_GPU_ID:-0}"
output_dir="${EVOINSPECT_2060_OUTPUT:-${repo_root}/remote_2060_result}"
allow_other_gpu="${EVOINSPECT_ALLOW_NON_2060:-0}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ -x "${python_exe}" ]] || die "remote environment is missing; run setup_remote_2060_env.sh"
for file in model.ckpt metrics.json quality-gate.json benchmark_input.png config.yaml bundle_manifest.json; do
  [[ -f "${payload}/${file}" ]] || die "deployment payload missing ${file}"
done
gpu_name="$(nvidia-smi -i "${gpu}" --query-gpu=name --format=csv,noheader | tr -d '\r')"
if [[ "${allow_other_gpu}" != "1" && ! "${gpu_name}" =~ (GTX|RTX)[[:space:]]*2060 ]]; then
  die "actual GPU is '${gpu_name}', not a 2060; set EVOINSPECT_ALLOW_NON_2060=1 only for non-claim diagnostics"
fi
processes="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
[[ -z "${processes}" ]] || die "GPU ${gpu} already has compute processes; no process was touched"
[[ ! -e "${output_dir}" ]] || die "refusing to overwrite result directory: ${output_dir}"
mkdir -p "${output_dir}"
exec 9>"/tmp/evoinspect-130-remote-gpu-${gpu}.lock"
flock -n 9 || die "GPU ${gpu} cooperative lock is held"
nvidia-smi -i "${gpu}" -q >"${output_dir}/nvidia-smi-before.txt"
"${python_exe}" -m pip freeze >"${output_dir}/pip-freeze.txt"
env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${repo_root}/src:${repo_root}:${repo_root}/third_party/anomalib-2.3.0/src" \
  "${python_exe}" "${repo_root}/scripts/benchmark_efficientad_latency.py" \
  --checkpoint "${payload}/model.ckpt" \
  --metrics "${payload}/metrics.json" \
  --image "${payload}/benchmark_input.png" \
  --config "${payload}/config.yaml" \
  --output "${output_dir}/latency-2500.json" \
  --physical-gpu "${gpu}" --resolution 2500 --warmup 100 --repeats 1000
cp "${payload}/bundle_manifest.json" "${output_dir}/bundle_manifest.json"
printf 'Completed actual-device benchmark: %s\n' "${output_dir}/latency-2500.json"
