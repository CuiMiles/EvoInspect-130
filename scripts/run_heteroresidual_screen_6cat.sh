#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
batch_root="${EVOINSPECT_HETERORESIDUAL_BATCH:-${repo_root}/reports/experiments/heteroresidual-screen-20260831}"
source_batch="${repo_root}/reports/experiments/efficientad-s-frozen-20260830T004009Z-seed143-gpu0-3"
python_bin="${EVOINSPECT_EFFICIENTAD_PYTHON:-/home/CuiMinghao/envs/evoinspect-efficientad/bin/python}"
gpu_pool_raw="${EVOINSPECT_HETERORESIDUAL_GPUS:-5,6,7}"
max_per_gpu="${EVOINSPECT_HETERORESIDUAL_MAX_PER_GPU:-2}"
categories=(cable capsule screw carpet transistor wood)

IFS=',' read -r -a gpus <<<"${gpu_pool_raw}"
[[ "${#gpus[@]}" -gt 0 ]] || { echo "empty GPU pool" >&2; exit 2; }
[[ "${max_per_gpu}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid max per GPU: ${max_per_gpu}" >&2; exit 2; }

mkdir -p "${batch_root}/runs"
monitor_log="${batch_root}/monitor.log"
tasks_file="${batch_root}/running_tasks.txt"
touch "${monitor_log}"
: >"${tasks_file}"

declare -A pid_category=()
declare -A pid_gpu=()
declare -A pid_output=()

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

gpu_free_mb() {
  local gpu="$1" row free
  row="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits)"
  free="${row//[[:space:]]/}"
  [[ "${free}" =~ ^[0-9]+$ ]] && printf '%s\n' "${free}"
}

active_count_for_gpu() {
  local gpu="$1" pid count=0
  for pid in "${!pid_gpu[@]}"; do
    [[ "${pid_gpu[${pid}]}" == "${gpu}" ]] && ((count += 1))
  done
  printf '%s\n' "${count}"
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
  local gpu="$1" free active util row
  free="$(gpu_free_mb "${gpu}")"
  [[ "${free}" =~ ^[0-9]+$ ]] && ((free >= 4096)) || return 1
  gpu_has_external_process "${gpu}" && return 1
  active="$(active_count_for_gpu "${gpu}")"
  ((active < max_per_gpu)) || return 1
  # With no child of ours, require an idle GPU. Once ours are running,
  # permit another bounded child as long as memory remains available.
  if ((active == 0)); then
    row="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
    util="${row//[[:space:]]/}"
    [[ "${util}" =~ ^[0-9]+$ ]] && ((util <= 5)) || return 1
  fi
  return 0
}

source_for_category() {
  local category="$1"
  find "${source_batch}/runs" -mindepth 1 -maxdepth 1 -type d \
    -name "efficientad-s-${category}-s143-*" -print -quit
}

start_one() {
  local category="$1" gpu="$2" source output output_base log
  source="$(source_for_category "${category}")"
  [[ -n "${source}" ]] || { echo "missing source run for ${category}" >&2; return 2; }
  output_base="${batch_root}/runs/heteroresidual_s-${category}-s143"
  if [[ -f "${output_base}/metrics.json" ]]; then
    echo "$(date -Is) REUSE category=${category}" >>"${monitor_log}"
    return 0
  fi
  output="${output_base}"
  if [[ -e "${output}" ]]; then
    output="${output_base}.retry-$(date +%s)"
    mv "${output_base}" "${output}"
  fi
  log="${batch_root}/runs/heteroresidual_s-${category}-s143.gpu${gpu}.launcher.log"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  EVOINSPECT_PHYSICAL_GPU="${gpu}" \
  EVOINSPECT_GPU_MEMORY_FRACTION="0.30" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH="${repo_root}/src:${repo_root}" \
  timeout --signal=TERM --kill-after=30s 2h \
  "${python_bin}" "${repo_root}/scripts/evaluate_heteroresidual_screen.py" \
    --source-run "${source}" --output-dir "${output}" --device cuda:0 \
    --run-id "heteroresidual-screen-${category}-s143" >"${log}" 2>&1 &
  local pid=$!
  pid_category["${pid}"]="${category}"
  pid_gpu["${pid}"]="${gpu}"
  pid_output["${pid}"]="${output}"
  printf '%s %s %s %s\n' "${pid}" "${category}" "${gpu}" "${output}" >>"${tasks_file}"
  echo "$(date -Is) START pid=${pid} category=${category} gpu=${gpu}" >>"${monitor_log}"
}

remove_task_line() {
  local pid="$1" tmp
  tmp="${tasks_file}.tmp"
  awk -v target="${pid}" '$1 != target' "${tasks_file}" >"${tmp}"
  mv "${tmp}" "${tasks_file}"
}

reap_children() {
  local pid status category gpu output
  for pid in "${!pid_category[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      category="${pid_category[${pid}]}"
      gpu="${pid_gpu[${pid}]}"
      output="${pid_output[${pid}]}"
      status=0
      wait "${pid}" || status=$?
      echo "$(date -Is) DONE pid=${pid} category=${category} gpu=${gpu} exit=${status} output=${output}" >>"${monitor_log}"
      remove_task_line "${pid}"
      unset "pid_category[${pid}]" "pid_gpu[${pid}]" "pid_output[${pid}]"
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

echo "$(date -Is) HeteroResidual-S six-category screen finished" >>"${monitor_log}"
