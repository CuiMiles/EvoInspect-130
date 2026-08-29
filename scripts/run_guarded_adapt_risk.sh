#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python_bin="${EVOINSPECT_PATCHCORE_PYTHON:-/home/CuiMinghao/envs/evoinspect-patchcore/bin/python}"
upstream="${repo_root}/third_party/patchcore-inspection-fcaa92f/src"
config="configs/innovations/guarded_adapt_risk.yaml"
partitions="evidence/guarded_adapt_risk_partitions.json"
stamp="${EVOINSPECT_GUARDED_RISK_STAMP:-20260829-preregistered-e17419c}"
output_root="reports/experiments/guarded-adapt-risk-${stamp}"
device="${EVOINSPECT_GUARDED_RISK_DEVICE:-cuda:0}"
physical_gpu="${EVOINSPECT_PHYSICAL_GPU:-UNRECORDED}"

mkdir -p "${output_root}/image_tasks"
PYTHONPATH="src:.:${upstream}" "${python_bin}" \
  scripts/evaluate_guarded_adapt_risk_legacy.py \
  --source reports/experiments/guarded-adapt-replay-20260827T194500-cpu/report.json \
  --config "${config}" \
  --output "${output_root}/legacy_75.json"

while read -r category seed; do
  output="${output_root}/image_tasks/${category}-s${seed}.json"
  if [[ -f "${output}" ]]; then
    continue
  fi
  EVOINSPECT_PHYSICAL_GPU="${physical_gpu}" \
    PYTHONPATH="src:.:${upstream}" \
    "${python_bin}" scripts/evaluate_guarded_adapt_risk_images.py \
      --config "${config}" \
      --partitions "${partitions}" \
      --category "${category}" \
      --seed "${seed}" \
      --output "${output}" \
      --device "${device}" \
      --batch-size "${EVOINSPECT_GUARDED_RISK_BATCH_SIZE:-4}"
done < <(
  PYTHONPATH=src:. "${python_bin}" - "${partitions}" <<'PY'
import json
import sys
for run in json.load(open(sys.argv[1], encoding="utf-8"))["runs"]:
    print(run["category"], run["seed"])
PY
)

PYTHONPATH=src:. "${python_bin}" scripts/aggregate_guarded_adapt_risk.py \
  --config "${config}" \
  --legacy "${output_root}/legacy_75.json" \
  --image-dir "${output_root}/image_tasks" \
  --output "${output_root}/report.json"
echo "${output_root}/report.json"
