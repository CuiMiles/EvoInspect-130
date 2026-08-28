#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
batch_root="${EVOINSPECT_BATCH_ROOT:-${repo_root}/reports/experiments/efficientad-m-frozen-20260828T095200Z-shared23}"
log_path="${EVOINSPECT_MONITOR_LOG:-${batch_root}/hourly_monitor.log}"
resume_stamp="20260828T095200Z-shared23"
interval_seconds="${EVOINSPECT_MONITOR_INTERVAL_SECONDS:-3600}"

mkdir -p "${batch_root}"
log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${log_path}"; }

our_count_on_gpu() {
  local gpu="$1" uuid
  if ! uuid="$(nvidia-smi --query-gpu=uuid -i "${gpu}" --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')" || [[ -z "${uuid}" ]]; then
    echo -1
    return 0
  fi
  nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader,nounits \
    | awk -F',' -v u="${uuid}" '$1==u && $2 ~ /CuiMinghao\/envs\/evoinspect-efficientad/ {n++} END {print n+0}'
}

snapshot() {
  local metrics failures active23 gpu4 gpu5 gpu6 gpu7
  metrics="$(find "${batch_root}" -name metrics.json | wc -l)"
  failures="$(find "${batch_root}" -name failure.json | wc -l)"
  active23=$(( $(our_count_on_gpu 2) + $(our_count_on_gpu 3) ))
  gpu4="$(our_count_on_gpu 4)"; gpu5="$(our_count_on_gpu 5)"
  gpu6="$(our_count_on_gpu 6)"; gpu7="$(our_count_on_gpu 7)"
  log "metrics=${metrics}/45 failures=${failures} our_cuda_gpu2_3=${active23} our_cuda_gpu4_7=${gpu4},${gpu5},${gpu6},${gpu7}"
  if ! nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits >>"${log_path}" 2>&1; then
    log "GPU_QUERY_UNAVAILABLE; resume is disabled until nvidia-smi recovers"
  fi
}

log "monitor started batch=${batch_root} interval=${interval_seconds}s"
launch_pid=""
resumed=0
while true; do
  snapshot
  active23=$(( $(our_count_on_gpu 2) + $(our_count_on_gpu 3) ))
  launcher_count="$(pgrep -af 'scripts/run_efficientad_frozen_8gpu.sh m' | wc -l)"

  if (( resumed == 0 && active23 == 0 && launcher_count == 0 )); then
    log "GPU2/3 training ended; preparing recoverable resume of missing tasks"
    moved=0
    while IFS= read -r failure; do
      run_dir="$(dirname "$(dirname "${failure}")")"
      result_dir="${run_dir}/result"
      if [[ -d "${result_dir}" ]]; then
        mv "${result_dir}" "${run_dir}/result.interrupted-${resume_stamp}"
        moved=$((moved + 1))
      fi
    done < <(find "${batch_root}/runs" -type f -path '*/result/failure.json' -print)
    log "moved_interrupted_result_dirs=${moved}; original artifacts retained"
    EVOINSPECT_ALLOW_SHARED_GPU=1 \
    EVOINSPECT_GPU_SLOTS='0 0 0 1 1 1 2 2 2 3 3 3' \
    EVOINSPECT_MIN_FREE_MEMORY_MB=2048 \
    EVOINSPECT_GPU_MEMORY_FRACTION=0.12 \
    EVOINSPECT_NUM_WORKERS=1 \
    EVOINSPECT_CPU_THREADS_PER_TASK=1 \
    EVOINSPECT_BATCH_STAMP="${resume_stamp}" \
    EVOINSPECT_BATCH_ROOT="${batch_root}" \
      "${repo_root}/scripts/run_efficientad_frozen_8gpu.sh" m \
      >"${batch_root}/resume-launch.log" 2>&1 &
    launch_pid="$!"
    resumed=1
    log "resume_launcher_pid=${launch_pid}"
  fi

  if (( resumed == 1 )) && ! kill -0 "${launch_pid}" 2>/dev/null; then
    snapshot
    if [[ "$(find "${batch_root}" -name metrics.json | wc -l)" -eq 45 ]]; then
      log "resume launcher ended with all 45 metrics present"
    else
      log "resume launcher ended before 45 metrics; inspect resume-launch.log"
    fi
    break
  fi
  sleep "${interval_seconds}"
done
