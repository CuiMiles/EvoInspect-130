#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

batch_root="${1:-reports/experiments/efficientad-m-frozen-20260828T095200Z-shared23}"
size="${2:-m}"
python_bin="${EVOINSPECT_EFFICIENTAD_PYTHON:-/home/CuiMinghao/envs/evoinspect-efficientad/bin/python}"
case "${size}" in
  m)
    training_config="configs/baselines/efficientad_m_100_30.yaml"
    evaluator_config="configs/evaluation/efficientad_strict_100_30_v2.yaml"
    expected_runs=45
    ;;
  s)
    training_config="configs/baselines/efficientad_s_100_30.yaml"
    evaluator_config="configs/evaluation/efficientad_strict_100_30_s_v2.yaml"
    expected_runs=15
    ;;
  *)
    echo "usage: $0 BATCH_ROOT {m|s}" >&2
    exit 2
    ;;
esac
read -r -a gpus <<< "${EVOINSPECT_EFFICIENTAD_V2_GPUS:-0 1 2 3}"

mapfile -t run_dirs < <(find "${batch_root}/runs" -mindepth 1 -maxdepth 1 -type d | sort)
checkpoint_count=0
for run_dir in "${run_dirs[@]}"; do
  if [[ -f "${run_dir}/result/model.ckpt" && -f "${run_dir}/result/metrics.json" ]]; then
    checkpoint_count=$((checkpoint_count + 1))
  fi
done
if [[ ${checkpoint_count} -ne ${expected_runs} ]]; then
  echo "strict evaluator requires ${expected_runs} completed checkpoints; found ${checkpoint_count}" >&2
  exit 2
fi
if [[ ${#gpus[@]} -eq 0 ]]; then
  echo "at least one GPU is required" >&2
  exit 2
fi

pids=()
cleanup() {
  for pid in "${pids[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

worker() {
  local slot="$1"
  local gpu="$2"
  local index run_dir
  for ((index=slot; index<${#run_dirs[@]}; index+=${#gpus[@]})); do
    run_dir="${run_dirs[index]}"
    if [[ -f "${run_dir}/strict_result_v2/metrics.json" ]]; then
      continue
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" \
      EVOINSPECT_PHYSICAL_GPU="${gpu}" \
      OMP_NUM_THREADS="${EVOINSPECT_CPU_THREADS_PER_TASK:-2}" \
      MKL_NUM_THREADS="${EVOINSPECT_CPU_THREADS_PER_TASK:-2}" \
      PYTHONPATH=src:. \
      "${python_bin}" scripts/evaluate_efficientad_strict_100_30.py \
        --run-dir "${run_dir}" \
        --training-config "${training_config}" \
        --evaluator-config "${evaluator_config}" \
        --device cuda:0
  done
}

for slot in "${!gpus[@]}"; do
  worker "${slot}" "${gpus[slot]}" >"${batch_root}/strict-v2-gpu${gpus[slot]}.log" 2>&1 &
  pids+=("$!")
done
failure=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failure=1
  fi
done
trap - INT TERM
if [[ ${failure} -ne 0 ]]; then
  echo "one or more strict evaluator workers failed" >&2
  exit 1
fi

PYTHONPATH=src:. "${python_bin}" scripts/aggregate_efficientad_v2.py \
  --batch-root "${batch_root}" \
  --config "${evaluator_config}" \
  --output "${batch_root}/strict-quality-gate-v2.json"
echo "${batch_root}/strict-quality-gate-v2.json"
