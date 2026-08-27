#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
size="${1:-m}"
case "${size}" in
  m) config="${repo_root}/configs/baselines/efficientad_m_100_30.yaml" ;;
  s) config="${repo_root}/configs/baselines/efficientad_s_100_30.yaml" ;;
  *) printf 'usage: %s {m|s}\n' "$0" >&2; exit 2 ;;
esac

efficient_python="${EVOINSPECT_EFFICIENTAD_PYTHON:-/home/CuiMinghao/envs/evoinspect-efficientad/bin/python}"
main_python="${EVOINSPECT_PYTHON:-/home/CuiMinghao/envs/evoinspect-130/bin/python}"
manifest="${EVOINSPECT_MANIFEST:-/home/CuiMinghao/data/mvtec_ad_official/manifests/mvtec_ad_manifest.csv}"
gpu_text="${EVOINSPECT_GPU_IDS:-0 1 2 3 4 5 6 7}"
seeds_text="${EVOINSPECT_SEEDS:-143 144 145}"
max_memory_mb="${EVOINSPECT_MAX_IDLE_MEMORY_MB:-256}"
max_utilization="${EVOINSPECT_MAX_IDLE_UTILIZATION:-5}"
task_timeout="${EVOINSPECT_TASK_TIMEOUT:-18h}"
dry_run="${EVOINSPECT_DRY_RUN:-0}"
stamp="${EVOINSPECT_BATCH_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
batch_root="${EVOINSPECT_BATCH_ROOT:-${repo_root}/reports/experiments/efficientad-${size}-frozen-${stamp}}"
categories_text="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }
gpu_is_idle() {
  local gpu="$1" snapshot memory utilization processes
  snapshot="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)" || return 1
  IFS=',' read -r memory utilization <<<"${snapshot}"
  memory="${memory//[[:space:]]/}"; utilization="${utilization//[[:space:]]/}"
  is_uint "${memory}" && is_uint "${utilization}" || return 1
  processes="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -z "${processes}" ]] && (( memory <= max_memory_mb && utilization <= max_utilization ))
}

[[ -x "${main_python}" && -x "${efficient_python}" ]] || die "required Python environment missing"
[[ -f "${manifest}" && -f "${config}" ]] || die "manifest or config missing"
read -r -a requested_gpus <<<"${gpu_text}"
read -r -a seeds <<<"${seeds_text}"
read -r -a categories <<<"${categories_text}"
declare -a gpu_ids=()
for gpu in "${requested_gpus[@]}"; do
  is_uint "${gpu}" || die "invalid GPU id: ${gpu}"
  if [[ "${dry_run}" == "1" ]] || gpu_is_idle "${gpu}"; then
    gpu_ids+=("${gpu}")
  else
    printf 'SKIP BUSY GPU %s; no process is touched\n' "${gpu}" >&2
  fi
done
(( ${#gpu_ids[@]} > 0 )) || die "no requested GPU is safely idle"
printf 'EfficientAD-%s: %s category-seed tasks on GPUs: %s\n' \
  "${size^^}" "$(( ${#categories[@]} * ${#seeds[@]} ))" "${gpu_ids[*]}"
if [[ "${dry_run}" == "1" ]]; then
  printf 'DRY RUN output=%s config=%s seeds=%s\n' "${batch_root}" "${config}" "${seeds[*]}"
  exit 0
fi

mkdir -p "${batch_root}/runs"
run_task() {
  local gpu="$1" category="$2" seed="$3"
  local run_id="efficientad-${size}-${category}-s${seed}-${stamp}"
  local run_dir="${batch_root}/runs/${run_id}"
  local log="${run_dir}/run.log"
  if [[ -f "${run_dir}/result/metrics.json" ]]; then
    printf 'REUSE %s\n' "${run_id}"
    return 0
  fi
  mkdir -p "${run_dir}"
  gpu_is_idle "${gpu}" || { printf 'GPU %s became busy before %s\n' "${gpu}" "${run_id}" >>"${log}"; return 90; }
  timeout --signal=TERM --kill-after=60s "${task_timeout}" \
    env CUDA_VISIBLE_DEVICES="${gpu}" EVOINSPECT_PHYSICAL_GPU="${gpu}" \
    PYTHONPATH="${repo_root}/src:${repo_root}" \
    "${main_python}" "${repo_root}/scripts/patchcore_lite_bottle.py" prepare \
    --manifest "${manifest}" --output-dir "${run_dir}" --seed "${seed}" --category "${category}" \
    >>"${log}" 2>&1
  timeout --signal=TERM --kill-after=60s "${task_timeout}" \
    env CUDA_VISIBLE_DEVICES="${gpu}" EVOINSPECT_PHYSICAL_GPU="${gpu}" \
    EVOINSPECT_GPU_MEMORY_FRACTION="${EVOINSPECT_GPU_MEMORY_FRACTION:-0.45}" \
    PYTHONPATH="${repo_root}/src:${repo_root}:${repo_root}/third_party/anomalib-2.3.0/src" \
    "${efficient_python}" "${repo_root}/scripts/efficientad_baseline_100_30.py" \
    --adaptation "${run_dir}/adaptation.csv" --test-inputs "${run_dir}/test_inputs.csv" \
    --test-truth "${run_dir}/test_truth.csv" --split "${run_dir}/split.json" \
    --config "${config}" --output-dir "${run_dir}/result" --run-id "${run_id}" \
    --seed "${seed}" --registry "${batch_root}/experiment_registry.csv" >>"${log}" 2>&1
  printf 'PASS gpu=%s %s\n' "${gpu}" "${run_id}"
}

worker_count="${#gpu_ids[@]}"
declare -a pids=()
for slot in "${!gpu_ids[@]}"; do
  (
    exec 9>"/tmp/evoinspect-130-gpu-${gpu_ids[slot]}.lock"
    flock -n 9 || exit 91
    index=0; failures=0
    for category in "${categories[@]}"; do
      for seed in "${seeds[@]}"; do
        if (( index % worker_count == slot )); then
          run_task "${gpu_ids[slot]}" "${category}" "${seed}" || failures=$((failures + 1))
        fi
        index=$((index + 1))
      done
    done
    exit "${failures}"
  ) >"${batch_root}/worker-${slot}.log" 2>&1 &
  pids+=("$!")
done
failures=0
for pid in "${pids[@]}"; do wait "${pid}" || failures=$((failures + 1)); done
(( failures == 0 )) || die "one or more workers failed; inspect ${batch_root}/worker-*.log"

env PYTHONPATH="${repo_root}/src:${repo_root}" "${main_python}" \
  "${repo_root}/scripts/aggregate_efficientad.py" --batch-root "${batch_root}" \
  --config "${config}" --expected-runs "$(( ${#categories[@]} * ${#seeds[@]} ))" \
  --output "${batch_root}/quality-gate.json" --enforce
printf 'Completed frozen EfficientAD-%s gate: %s/quality-gate.json\n' "${size^^}" "${batch_root}"
