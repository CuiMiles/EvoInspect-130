#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
batch_root="${EVOINSPECT_S384_SCREEN_BATCH:-${repo_root}/reports/experiments/efficientad-s384-screen-20260831}"
manifest="${EVOINSPECT_MANIFEST:-/home/CuiMinghao/data/mvtec_ad_official/manifests/mvtec_ad_manifest.csv}"
main_python="${EVOINSPECT_PYTHON:-/home/CuiMinghao/envs/evoinspect-130/bin/python}"
efficient_python="${EVOINSPECT_EFFICIENTAD_PYTHON:-/home/CuiMinghao/envs/evoinspect-efficientad/bin/python}"
training_config="${repo_root}/configs/baselines/efficientad_s_384_screen_20260831.yaml"
evaluator_config="${repo_root}/configs/evaluation/efficientad_s_384_screen_20260831.yaml"
gpu_pool_raw="${EVOINSPECT_S384_SCREEN_GPUS:-4,5,6,7}"
categories=(cable capsule screw carpet transistor wood)
seed=143

IFS=',' read -r -a gpus <<<"${gpu_pool_raw}"
[[ "${#gpus[@]}" -gt 0 ]] || { echo "empty GPU pool" >&2; exit 2; }
[[ -x "${main_python}" && -x "${efficient_python}" ]] || {
  echo "required Python environment missing" >&2
  exit 2
}
[[ -f "${manifest}" && -f "${training_config}" && -f "${evaluator_config}" ]] || {
  echo "manifest or config missing" >&2
  exit 2
}

mkdir -p "${batch_root}/runs"
monitor_log="${batch_root}/monitor.log"
tasks_file="${batch_root}/running_tasks.txt"
touch "${monitor_log}"
: >"${tasks_file}"

declare -A pid_category=()
declare -A pid_gpu=()
declare -A pid_run=()

gpu_free_mb() {
  local gpu="$1" row free
  row="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits)"
  free="${row//[[:space:]]/}"
  [[ "${free}" =~ ^[0-9]+$ ]] && printf '%s\n' "${free}"
}

gpu_has_external_process() {
  local gpu="$1" process pid known
  process="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits || true)"
  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    pid="${pid//[[:space:]]/}"
    known=0
    [[ -n "${pid_gpu[${pid}]+x}" ]] && known=1
    ((known == 0)) && return 0
  done <<<"${process}"
  return 1
}

gpu_can_start() {
  local gpu="$1" free util row
  free="$(gpu_free_mb "${gpu}")"
  [[ "${free}" =~ ^[0-9]+$ ]] && ((free >= 4096)) || return 1
  gpu_has_external_process "${gpu}" && return 1
  for pid in "${!pid_gpu[@]}"; do
    [[ "${pid_gpu[${pid}]}" == "${gpu}" ]] && return 1
  done
  row="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
  util="${row//[[:space:]]/}"
  [[ "${util}" =~ ^[0-9]+$ ]] && ((util <= 5))
}

log_gpu_snapshot() {
  local timestamp gpu
  timestamp="$(date -Is)"
  for gpu in "${gpus[@]}"; do
    printf '%s gpu=%s ' "${timestamp}" "${gpu}" >>"${monitor_log}"
    nvidia-smi -i "${gpu}" \
      --query-gpu=index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits >>"${monitor_log}" 2>&1 || true
  done
}

start_one() {
  local category="$1" gpu="$2" run_dir run_id log
  run_id="efficientad-s384-${category}-s${seed}"
  run_dir="${batch_root}/runs/${run_id}"
  if [[ -f "${run_dir}/strict_result_v2/metrics.json" ]]; then
    echo "$(date -Is) REUSE category=${category} gpu=${gpu}" >>"${monitor_log}"
    return 0
  fi
  if [[ -e "${run_dir}" ]]; then
    mv "${run_dir}" "${run_dir}.retry-$(date +%s)"
  fi
  mkdir -p "${run_dir}"
  log="${run_dir}/screen.log"
  (
    set -Eeuo pipefail
    env CUDA_VISIBLE_DEVICES="${gpu}" EVOINSPECT_PHYSICAL_GPU="${gpu}" \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      PYTHONPATH="${repo_root}/src:${repo_root}" \
      "${main_python}" "${repo_root}/scripts/patchcore_lite_bottle.py" prepare \
      --manifest "${manifest}" --output-dir "${run_dir}" --seed "${seed}" \
      --category "${category}"
    env CUDA_VISIBLE_DEVICES="${gpu}" EVOINSPECT_PHYSICAL_GPU="${gpu}" \
      EVOINSPECT_GPU_MEMORY_FRACTION=0.30 EVOINSPECT_NUM_WORKERS=1 \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      PYTHONPATH="${repo_root}/src:${repo_root}:${repo_root}/third_party/anomalib-2.3.0/src" \
      "${efficient_python}" "${repo_root}/scripts/efficientad_baseline_100_30.py" \
      --adaptation "${run_dir}/adaptation.csv" --test-inputs "${run_dir}/test_inputs.csv" \
      --test-truth "${run_dir}/test_truth.csv" --split "${run_dir}/split.json" \
      --config "${training_config}" --output-dir "${run_dir}/result" --run-id "${run_id}" \
      --seed "${seed}" --registry "${batch_root}/experiment_registry.csv"
    env CUDA_VISIBLE_DEVICES="${gpu}" EVOINSPECT_PHYSICAL_GPU="${gpu}" \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      PYTHONPATH="${repo_root}/src:${repo_root}:${repo_root}/third_party/anomalib-2.3.0/src" \
      "${efficient_python}" "${repo_root}/scripts/evaluate_efficientad_strict_100_30.py" \
      --run-dir "${run_dir}" --training-config "${training_config}" \
      --evaluator-config "${evaluator_config}" --device cuda:0
  ) >"${log}" 2>&1 &
  local pid=$!
  pid_category["${pid}"]="${category}"
  pid_gpu["${pid}"]="${gpu}"
  pid_run["${pid}"]="${run_dir}"
  printf '%s %s %s %s\n' "${pid}" "${category}" "${gpu}" "${run_dir}" >>"${tasks_file}"
  echo "$(date -Is) START pid=${pid} category=${category} gpu=${gpu}" >>"${monitor_log}"
}

remove_task_line() {
  local pid="$1" tmp="${tasks_file}.tmp"
  awk -v target="${pid}" '$1 != target' "${tasks_file}" >"${tmp}"
  mv "${tmp}" "${tasks_file}"
}

reap_children() {
  local pid category gpu run_dir status
  for pid in "${!pid_category[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      category="${pid_category[${pid}]}"
      gpu="${pid_gpu[${pid}]}"
      run_dir="${pid_run[${pid}]}"
      status=0
      wait "${pid}" || status=$?
      echo "$(date -Is) DONE pid=${pid} category=${category} gpu=${gpu} exit=${status} run=${run_dir}" >>"${monitor_log}"
      remove_task_line "${pid}"
      unset "pid_category[${pid}]" "pid_gpu[${pid}]" "pid_run[${pid}]"
    fi
  done
}

next_index=0
while ((next_index < ${#categories[@]} || ${#pid_category[@]} > 0)); do
  reap_children
  while ((next_index < ${#categories[@]})); do
    scheduled=0
    for gpu in "${gpus[@]}"; do
      if gpu_can_start "${gpu}"; then
        category="${categories[${next_index}]}"
        start_one "${category}" "${gpu}"
        ((next_index += 1))
        scheduled=1
        ((next_index >= ${#categories[@]})) && break
      fi
    done
    ((scheduled == 1)) || break
  done
  log_gpu_snapshot
  ((next_index < ${#categories[@]} || ${#pid_category[@]} > 0)) && sleep 30
done

echo "$(date -Is) EfficientAD-S384 six-category screen finished" >>"${monitor_log}"
