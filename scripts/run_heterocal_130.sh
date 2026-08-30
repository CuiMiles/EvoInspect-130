#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
batch_root="${repo_root}/reports/experiments/efficientad-m-frozen-20260828T095200Z-shared23"
python_bin="/home/CuiMinghao/envs/evoinspect-efficientad/bin/python"
gpu="${EVOINSPECT_HETEROCAL_GPU:-5}"
minimum_free_mib="${EVOINSPECT_HETEROCAL_MIN_FREE_MIB:-10000}"

[[ -x "${python_bin}" ]] || { printf 'missing EfficientAD Python: %s\n' "${python_bin}" >&2; exit 2; }
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
[[ "${free_mib}" =~ ^[0-9]+$ ]] || { printf 'cannot read GPU %s free memory\n' "${gpu}" >&2; exit 2; }
(( free_mib >= minimum_free_mib )) || {
  printf 'GPU %s has only %s MiB free; require %s MiB\n' "${gpu}" "${free_mib}" "${minimum_free_mib}" >&2
  exit 2
}

export CUDA_VISIBLE_DEVICES="${gpu}"
export EVOINSPECT_PHYSICAL_GPU="${gpu}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

completed=0
for run_dir in "${batch_root}"/runs/*; do
  [[ -f "${run_dir}/result/model.ckpt" ]] || continue
  [[ -f "${run_dir}/strict_result_v2/metrics.json" ]] || continue
  if [[ -f "${run_dir}/heterocal_result/metrics.json" ]]; then
    completed=$((completed + 1))
    continue
  fi
  "${python_bin}" "${repo_root}/scripts/evaluate_heterocal_130.py" \
    --run-dir "${run_dir}" \
    --config "${repo_root}/configs/innovations/heterocal_130.yaml" \
    --training-config "${repo_root}/configs/baselines/efficientad_m_100_30.yaml" \
    --evaluator-config "${repo_root}/configs/evaluation/efficientad_strict_100_30_v2.yaml" \
    --device cuda:0
  completed=$((completed + 1))
  printf 'HeteroCal completed %d/45: %s\n' "${completed}" "$(basename "${run_dir}")"
done

"${python_bin}" "${repo_root}/scripts/aggregate_heterocal_130.py" \
  --batch-root "${batch_root}" \
  --config "${repo_root}/configs/innovations/heterocal_130.yaml" \
  --output "${batch_root}/heterocal-quality-gate.json"
