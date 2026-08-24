#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-development}"
efficient_python="${EVOINSPECT_EFFICIENTAD_PYTHON:-/home/CuiMinghao/envs/evoinspect-efficientad/bin/python}"
main_python="${EVOINSPECT_PYTHON:-/home/CuiMinghao/envs/evoinspect-130/bin/python}"
manifest="${EVOINSPECT_MANIFEST:-/home/CuiMinghao/data/mvtec_ad_official/manifests/mvtec_ad_manifest.csv}"
baseline_config="${EVOINSPECT_BASELINE_CONFIG:-${repo_root}/configs/baselines/efficientad_s_100_30.yaml}"
rcbr_config="${EVOINSPECT_RCBR_CONFIG:-${repo_root}/configs/innovations/rcbr_v1_dev.yaml}"
prepare_runner="${repo_root}/scripts/patchcore_lite_bottle.py"
task_runner="${repo_root}/scripts/efficientad_rcbr_100_30.py"
aggregate_runner="${repo_root}/scripts/aggregate_rcbr.py"
gpu_text="${EVOINSPECT_GPU_IDS:-0 1 2 3 4 5 6 7}"
max_memory_mb="${EVOINSPECT_MAX_IDLE_MEMORY_MB:-256}"
max_utilization="${EVOINSPECT_MAX_IDLE_UTILIZATION:-5}"
task_timeout="${EVOINSPECT_TASK_TIMEOUT:-8h}"
dry_run="${EVOINSPECT_DRY_RUN:-0}"
batch_stamp="${EVOINSPECT_BATCH_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
batch_root="${EVOINSPECT_BATCH_ROOT:-${repo_root}/reports/experiments/rcbr-${mode}-${batch_stamp}}"
patchcore_source="${EVOINSPECT_PATCHCORE_DEV_SOURCE:-${repo_root}/reports/experiments/masked-prototype-v4-upstream-dev15-3seed-8gpu-20260824T003300Z-26470/aggregate.json}"
patchcore_reference_root="${EVOINSPECT_PATCHCORE_DEV_REFERENCE_ROOT:-${repo_root}/reports/references/patchcore-dev-130-132-localization-v3}"
patchcore_reference="${patchcore_reference_root}/aggregate.json"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }
gpu_processes() {
  nvidia-smi -i "$1" --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null || true
}
gpu_is_idle() {
  local gpu="$1" snapshot memory utilization processes
  snapshot="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)" || return 1
  IFS=',' read -r memory utilization <<<"${snapshot}"
  memory="${memory//[[:space:]]/}"; utilization="${utilization//[[:space:]]/}"
  is_uint "${memory}" && is_uint "${utilization}" || return 1
  processes="$(gpu_processes "${gpu}")"
  [[ -z "${processes}" ]] && (( memory <= max_memory_mb && utilization <= max_utilization ))
}
describe_gpu() {
  nvidia-smi -i "$1" --query-gpu=index,name,uuid,memory.used,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null || true
}

case "${mode}" in
  smoke|development|confirmation) ;;
  *) die "usage: $0 {smoke|development|confirmation}" ;;
esac
[[ -x "${main_python}" ]] || die "main Python missing: ${main_python}"
[[ -f "${manifest}" && -f "${baseline_config}" && -f "${rcbr_config}" ]] || die "input/config missing"
if [[ "${dry_run}" != "1" ]]; then
  [[ -x "${efficient_python}" ]] || die "run scripts/setup_efficientad_env.sh first"
fi

read -r -a requested_gpus <<<"${gpu_text}"
declare -a gpu_ids=()
declare -A seen_gpu=()
for gpu in "${requested_gpus[@]}"; do
  is_uint "${gpu}" || die "invalid GPU ID: ${gpu}"
  [[ -z "${seen_gpu[${gpu}]:-}" ]] || die "duplicate GPU ID: ${gpu}"
  seen_gpu["${gpu}"]=1
  if gpu_is_idle "${gpu}"; then
    gpu_ids+=("${gpu}")
    printf 'AVAILABLE %s\n' "$(describe_gpu "${gpu}")"
  else
    printf 'SKIP BUSY GPU %s (no process will be touched)\n' "${gpu}" >&2
  fi
done
(( ${#gpu_ids[@]} > 0 )) || die "no requested GPU is safely idle"
printf 'Using %s safe GPU(s): %s\n' "${#gpu_ids[@]}" "${gpu_ids[*]}"

if [[ "${mode}" == "confirmation" ]]; then
  [[ "${EVOINSPECT_ALLOW_CONFIRMATION:-0}" == "1" ]] \
    || die "confirmation is sealed; set EVOINSPECT_ALLOW_CONFIRMATION=1 after development review"
  frozen_manifest="${EVOINSPECT_FROZEN_MANIFEST:-}"
  [[ -n "${frozen_manifest}" && -f "${frozen_manifest}" ]] || die "frozen manifest is required"
  env PYTHONPATH="${repo_root}/src" "${main_python}" "${repo_root}/scripts/freeze_rcbr.py" verify \
    --root "${repo_root}" --manifest "${frozen_manifest}"
fi

if [[ "${dry_run}" != "1" && ! -f "${patchcore_reference}" ]]; then
  [[ -f "${patchcore_source}" ]] || die "PatchCore dev source missing: ${patchcore_source}"
  env PYTHONPATH="${repo_root}/src" "${main_python}" \
    "${repo_root}/scripts/evaluate_saved_localization.py" \
    --source-aggregate "${patchcore_source}" --output-dir "${patchcore_reference_root}" \
    --registry "${patchcore_reference_root}/experiment_registry.csv" \
    --run-id "patchcore-dev-130-132-localization-v3" --workers 8 --expected-runs 45
fi

run_task() {
  local gpu="$1" category="$2" seed="$3"
  local run_id="rcbr-${category}-s${seed}-${batch_stamp}"
  local run_dir="${batch_root}/runs/${run_id}"
  local log_file="${run_dir}/run.log"
  if [[ -f "${run_dir}/result/metrics.json" ]]; then
    printf 'REUSE %s\n' "${run_id}"
    return 0
  fi
  mkdir -p "${run_dir}"
  printf 'run_id=%s physical_gpu=%s category=%s seed=%s started=%s\n' \
    "${run_id}" "${gpu}" "${category}" "${seed}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${log_file}"
  if ! gpu_is_idle "${gpu}"; then
    printf 'GPU became busy before task; refusing to start\n' >>"${log_file}"
    return 90
  fi
  if timeout --signal=TERM --kill-after=60s "${task_timeout}" \
      env CUDA_VISIBLE_DEVICES="${gpu}" EVOINSPECT_PHYSICAL_GPU="${gpu}" \
      PYTHONPATH="${repo_root}/src:${repo_root}/third_party/anomalib-2.3.0/src" \
      "${main_python}" "${prepare_runner}" prepare \
      --manifest "${manifest}" --output-dir "${run_dir}" --seed "${seed}" --category "${category}" \
      >>"${log_file}" 2>&1 \
    && cd "${repo_root}" \
    && timeout --signal=TERM --kill-after=60s "${task_timeout}" \
      env CUDA_VISIBLE_DEVICES="${gpu}" EVOINSPECT_PHYSICAL_GPU="${gpu}" \
      PYTHONPATH="${repo_root}/src:${repo_root}/third_party/anomalib-2.3.0/src" \
      "${efficient_python}" "${task_runner}" \
      --adaptation "${run_dir}/adaptation.csv" --test-inputs "${run_dir}/test_inputs.csv" \
      --test-truth "${run_dir}/test_truth.csv" --split "${run_dir}/split.json" \
      --baseline-config "${baseline_config}" --rcbr-config "${rcbr_config}" \
      --output-dir "${run_dir}/result" --run-id "${run_id}" --seed "${seed}" \
      --registry "${batch_root}/experiment_registry.csv" \
      >>"${log_file}" 2>&1; then
    printf 'PASS gpu=%s %s\n' "${gpu}" "${run_id}"
    return 0
  fi
  printf 'FAIL gpu=%s %s log=%s\n' "${gpu}" "${run_id}" "${log_file}" >&2
  return 1
}

run_stage() {
  local category_text="$1" seed_text="$2" label="$3"
  read -r -a categories <<<"${category_text}"
  read -r -a seeds <<<"${seed_text}"
  local task_count=$(( ${#categories[@]} * ${#seeds[@]} ))
  printf 'Stage %s: %s tasks\n' "${label}" "${task_count}"
  if [[ "${dry_run}" == "1" ]]; then return 0; fi
  mkdir -p "${batch_root}/runs"
  local worker_count="${#gpu_ids[@]}"
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
    ) >"${batch_root}/worker-${label}-${slot}.log" 2>&1 &
    pids+=("$!")
  done
  failures=0
  for pid in "${pids[@]}"; do wait "${pid}" || failures=$((failures + 1)); done
  (( failures == 0 )) || die "stage ${label} failed; inspect ${batch_root}/worker-${label}-*.log"
}

smoke_categories="wood capsule transistor hazelnut"
all_categories="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
remaining_categories="bottle cable carpet grid leather metal_nut pill screw tile toothbrush zipper"
if [[ "${mode}" == "confirmation" ]]; then
  run_stage "${all_categories}" "138 139 140 141 142" "confirmation"
  env PYTHONPATH="${repo_root}/src" "${main_python}" "${aggregate_runner}" \
    --batch-root "${batch_root}" --patchcore-reference "${patchcore_reference}" \
    --config "${rcbr_config}" --gate full_development --expected-runs 75 \
    --output "${batch_root}/confirmation-summary.json"
  printf 'Confirmation complete. Do not rerun or tune from these results: %s\n' "${batch_root}"
  exit 0
fi

# Functional seed-130 and the remaining two seeds are separate stages but share one batch,
# so the four smoke categories are trained exactly three times rather than duplicated.
run_stage "${smoke_categories}" "130" "smoke-s130"
run_stage "${smoke_categories}" "131 132" "smoke-s131-132"
if [[ "${dry_run}" == "1" ]]; then
  if [[ "${mode}" == "development" ]]; then run_stage "${remaining_categories}" "130 131 132" "full-remaining"; fi
  printf 'Dry run complete; no files, models, or GPU processes were created.\n'
  exit 0
fi
env PYTHONPATH="${repo_root}/src" "${main_python}" "${aggregate_runner}" \
  --batch-root "${batch_root}" --patchcore-reference "${patchcore_reference}" \
  --config "${rcbr_config}" --gate smoke --expected-runs 12 \
  --output "${batch_root}/smoke-gate.json" --enforce
if [[ "${mode}" == "smoke" ]]; then
  printf 'Smoke gate passed: %s/smoke-gate.json\n' "${batch_root}"
  exit 0
fi
run_stage "${remaining_categories}" "130 131 132" "full-remaining"
env PYTHONPATH="${repo_root}/src" "${main_python}" "${aggregate_runner}" \
  --batch-root "${batch_root}" --patchcore-reference "${patchcore_reference}" \
  --config "${rcbr_config}" --gate full_development --expected-runs 45 \
  --output "${batch_root}/full-development-gate.json" --enforce
benchmark_run="$(find "${batch_root}/runs" -maxdepth 1 -type d -name 'rcbr-wood-s130-*' -print -quit)"
[[ -n "${benchmark_run}" ]] || die "wood seed-130 run missing for latency benchmark"
benchmark_gpu="${gpu_ids[0]}"
gpu_is_idle "${benchmark_gpu}" || die "benchmark GPU became busy; refusing to use it"
(
  exec 9>"/tmp/evoinspect-130-gpu-${benchmark_gpu}.lock"
  flock -n 9 || die "benchmark GPU lock is held"
  cd "${repo_root}"
  env CUDA_VISIBLE_DEVICES="${benchmark_gpu}" PYTHONPATH="${repo_root}/src:${repo_root}" \
    "${efficient_python}" "${repo_root}/scripts/benchmark_rcbr_latency.py" \
    --checkpoint "${benchmark_run}/result/model.ckpt" \
    --router-state "${benchmark_run}/result/router_state.npz" \
    --test-inputs "${benchmark_run}/test_inputs.csv" --config "${rcbr_config}" \
    --baseline-config "${baseline_config}" \
    --output "${batch_root}/latency-2500-rtx3090.json" --physical-gpu "${benchmark_gpu}"
)
printf 'Development gate passed. Return the complete batch before confirmation: %s\n' "${batch_root}"
