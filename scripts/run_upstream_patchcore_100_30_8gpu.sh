#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
main_python="${EVOINSPECT_PYTHON:-/home/CuiMinghao/envs/evoinspect-130/bin/python}"
patch_python="${EVOINSPECT_PATCHCORE_PYTHON:-/home/CuiMinghao/envs/evoinspect-patchcore/bin/python}"
manifest="${EVOINSPECT_MANIFEST:-/home/CuiMinghao/data/mvtec_ad_official/manifests/mvtec_ad_manifest.csv}"
upstream="${EVOINSPECT_PATCHCORE_UPSTREAM:-${repo_root}/third_party/patchcore-inspection-fcaa92f}"
config="${EVOINSPECT_CONFIG:-${repo_root}/configs/baselines/patchcore_upstream_100_30.yaml}"
gpu_text="${EVOINSPECT_GPU_IDS:-0 1 2 3 4 5 6 7}"
seed_text="${EVOINSPECT_SEEDS:-133 134 135 136 137}"
category_text="${EVOINSPECT_CATEGORIES:-screw capsule cable pill tile transistor toothbrush grid bottle zipper carpet leather hazelnut metal_nut wood}"
max_memory_used_mb="${EVOINSPECT_MAX_IDLE_MEMORY_MB:-256}"
max_idle_utilization="${EVOINSPECT_MAX_IDLE_UTILIZATION:-5}"
stage_timeout="${EVOINSPECT_STAGE_TIMEOUT:-30m}"
dry_run="${EVOINSPECT_DRY_RUN:-0}"
batch_stamp="${EVOINSPECT_BATCH_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
batch_label="${EVOINSPECT_BATCH_LABEL:-upstream-patchcore-100-30-mvtec15-5seed-8gpu}"
batch_root="${EVOINSPECT_BATCH_ROOT:-${repo_root}/reports/experiments/${batch_label}-${batch_stamp}}"
main_registry="${EVOINSPECT_REGISTRY:-${repo_root}/evidence/experiment_registry.csv}"
prepare_runner="${repo_root}/scripts/patchcore_lite_bottle.py"
runner="${EVOINSPECT_RUNNER:-${repo_root}/scripts/patchcore_upstream_100_30.py}"
run_id_prefix="${EVOINSPECT_RUN_ID_PREFIX:-upstream-pc-100-30}"
upstream_commit="fcaa92f124fb1ad74a7acf56726decd4b27cbcad"
weights_checkpoint="${EVOINSPECT_WEIGHTS_CHECKPOINT:-/home/CuiMinghao/.cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth}"

read -r -a categories <<<"${category_text}"
read -r -a gpu_ids <<<"${gpu_text}"
read -r -a seeds <<<"${seed_text}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require_uint() {
  local name="$1" value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer: ${value}"
}
gpu_processes() {
  nvidia-smi -i "$1" --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null || true
}
assert_gpu_idle() {
  local gpu="$1" snapshot uuid memory_used utilization processes
  snapshot="$(nvidia-smi -i "${gpu}" --query-gpu=uuid,memory.used,utilization.gpu \
    --format=csv,noheader,nounits)" || die "cannot query physical GPU ${gpu}"
  IFS=',' read -r uuid memory_used utilization <<<"${snapshot}"
  uuid="${uuid//[[:space:]]/}"
  memory_used="${memory_used//[[:space:]]/}"
  utilization="${utilization//[[:space:]]/}"
  require_uint "GPU ${gpu} memory.used" "${memory_used}"
  require_uint "GPU ${gpu} utilization" "${utilization}"
  processes="$(gpu_processes "${gpu}")"
  if [[ -n "${processes}" ]]; then
    printf 'GPU %s has compute processes:\n%s\n' "${gpu}" "${processes}" >&2
    return 1
  fi
  if (( memory_used > max_memory_used_mb || utilization > max_idle_utilization )); then
    printf 'GPU %s not idle: memory=%sMiB utilization=%s%%\n' \
      "${gpu}" "${memory_used}" "${utilization}" >&2
    return 1
  fi
  printf 'GPU %s idle: uuid=%s memory=%sMiB utilization=%s%%\n' \
    "${gpu}" "${uuid}" "${memory_used}" "${utilization}"
}
run_stage() {
  local log_file="$1"
  shift
  timeout --signal=TERM --kill-after=30s "${stage_timeout}" "$@" >>"${log_file}" 2>&1
}
run_task() {
  local gpu="$1" category="$2" seed="$3"
  local run_id="${run_id_prefix}-${category}-s${seed}-${batch_stamp}"
  local run_dir="${batch_root}/runs/${run_id}"
  local task_registry="${run_dir}/registry.csv"
  local log_file="${run_dir}/run.log"
  mkdir -p "${run_dir}"
  printf 'run_id=%s gpu=%s category=%s seed=%s started=%s\n' \
    "${run_id}" "${gpu}" "${category}" "${seed}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >"${log_file}"
  if run_stage "${log_file}" env PYTHONPATH="${repo_root}/src" \
      EVOINSPECT_MANIFEST_SHA256="${EVOINSPECT_MANIFEST_SHA256}" \
      "${main_python}" "${prepare_runner}" prepare \
      --manifest "${manifest}" --output-dir "${run_dir}" --seed "${seed}" \
      --category "${category}" \
    && assert_gpu_idle "${gpu}" >>"${log_file}" \
    && run_stage "${log_file}" env CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONPATH="${repo_root}/src:${upstream}/src" \
      "${patch_python}" "${runner}" train \
      --adaptation "${run_dir}/adaptation.csv" --split "${run_dir}/split.json" \
      --output-model "${run_dir}/model" --seed "${seed}" --config "${config}" \
    && run_stage "${log_file}" env CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONPATH="${repo_root}/src:${upstream}/src" EVOINSPECT_VERIFY_MODEL_HASH=0 \
      "${patch_python}" "${runner}" infer \
      --test-inputs "${run_dir}/test_inputs.csv" --model "${run_dir}/model" \
      --output "${run_dir}/predictions.jsonl" \
    && run_stage "${log_file}" env CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONPATH="${repo_root}/src:${upstream}/src" \
      "${patch_python}" "${runner}" evaluate \
      --truth "${run_dir}/test_truth.csv" --predictions "${run_dir}/predictions.jsonl" \
      --model "${run_dir}/model" --split "${run_dir}/split.json" \
      --output "${run_dir}/metrics.json" --registry "${task_registry}"; then
    printf 'PASS gpu=%s category=%s seed=%s\n' "${gpu}" "${category}" "${seed}"
    return 0
  fi
  env PYTHONPATH="${repo_root}/src:${upstream}/src" "${patch_python}" "${runner}" \
    record-failure --registry "${task_registry}" --run-id "${run_id}" --seed "${seed}" \
    --category "${category}" --physical-gpu "${gpu}" --config "${config}" \
    --manifest "${manifest}" --run-dir "${run_dir}" >>"${log_file}" 2>&1 || true
  printf 'FAIL gpu=%s category=%s seed=%s log=%s\n' \
    "${gpu}" "${category}" "${seed}" "${log_file}" >&2
  return 1
}
worker() {
  local slot="$1" gpu="$2" task_index=0 failures=0
  exec 9>"/tmp/evoinspect-130-gpu-${gpu}.lock"
  flock -n 9 || die "GPU ${gpu} is reserved by another EvoInspect launcher"
  for category in "${categories[@]}"; do
    for seed in "${seeds[@]}"; do
      if (( task_index % 8 == slot )); then
        if ! assert_gpu_idle "${gpu}"; then return 90; fi
        if ! run_task "${gpu}" "${category}" "${seed}"; then failures=$((failures + 1)); fi
      fi
      task_index=$((task_index + 1))
    done
  done
  return "${failures}"
}

(( ${#gpu_ids[@]} == 8 )) || die "exactly 8 GPU IDs are required"
(( ${#seeds[@]} >= 1 )) || die "at least one seed is required"
declare -A seen_gpus=() seen_seeds=()
for gpu in "${gpu_ids[@]}"; do
  require_uint "GPU ID" "${gpu}"
  [[ -z "${seen_gpus[${gpu}]:-}" ]] || die "duplicate GPU ID: ${gpu}"
  seen_gpus["${gpu}"]=1
done
for seed in "${seeds[@]}"; do
  require_uint "seed" "${seed}"
  [[ -z "${seen_seeds[${seed}]:-}" ]] || die "duplicate seed: ${seed}"
  seen_seeds["${seed}"]=1
done
[[ -x "${main_python}" && -x "${patch_python}" ]] || die "Python environment missing"
[[ -f "${manifest}" && -f "${config}" && -f "${runner}" ]] || die "input missing"
[[ -f "${weights_checkpoint}" ]] || die "backbone weights missing"
[[ "$(git -C "${upstream}" rev-parse HEAD)" == "${upstream_commit}" ]] \
  || die "upstream commit mismatch"
[[ -z "$(git -C "${upstream}" status --short)" ]] || die "upstream checkout modified"
printf 'Preflight: checking all eight physical GPUs.\n'
for gpu in "${gpu_ids[@]}"; do assert_gpu_idle "${gpu}" || die "preflight failed"; done
task_count=$(( ${#categories[@]} * ${#seeds[@]} ))
printf 'Tasks: %s categories x %s seeds = %s\nOutput: %s\n' \
  "${#categories[@]}" "${#seeds[@]}" "${task_count}" "${batch_root}"
if [[ "${dry_run}" == "1" ]]; then printf 'Dry run complete.\n'; exit 0; fi
[[ ! -e "${batch_root}" ]] || die "batch output exists"
mkdir -p "${batch_root}/runs"
export EVOINSPECT_MANIFEST_SHA256
EVOINSPECT_MANIFEST_SHA256="$(sha256sum "${manifest}" | awk '{print $1}')"
{
  printf 'upstream_commit=%s\n' "${upstream_commit}"
  printf 'manifest_sha256=%s\n' "${EVOINSPECT_MANIFEST_SHA256}"
  printf 'weights_sha256=%s\n' "$(sha256sum "${weights_checkpoint}" | awk '{print $1}')"
  printf 'config_sha256=%s\n' "$(sha256sum "${config}" | awk '{print $1}')"
} >"${batch_root}/static-provenance.txt"
declare -a pids=()
for slot in "${!gpu_ids[@]}"; do
  worker "${slot}" "${gpu_ids[slot]}" >"${batch_root}/worker-${slot}.log" 2>&1 &
  pids+=("$!")
done
worker_failures=0
for slot in "${!pids[@]}"; do
  if wait "${pids[slot]}"; then
    printf 'WORKER PASS slot=%s gpu=%s\n' "${slot}" "${gpu_ids[slot]}"
  else
    printf 'WORKER FAIL slot=%s gpu=%s\n' "${slot}" "${gpu_ids[slot]}" >&2
    worker_failures=$((worker_failures + 1))
  fi
done
mapfile -t run_dirs < <(find "${batch_root}/runs" -mindepth 1 -maxdepth 1 \
  -type d -exec test -f '{}/metrics.json' \; -print | sort)
mapfile -t task_registries < <(find "${batch_root}/runs" -mindepth 2 -maxdepth 2 \
  -type f -name registry.csv | sort)
if (( ${#run_dirs[@]} > 0 )); then
  env PYTHONPATH="${repo_root}/src:${upstream}/src" "${patch_python}" "${runner}" aggregate \
    --run-dirs "${run_dirs[@]}" --output "${batch_root}/aggregate.json"
fi
if (( ${#task_registries[@]} > 0 )); then
  exec 8>>"${main_registry}.lock"
  flock 8
  for task_registry in "${task_registries[@]}"; do tail -n +2 "${task_registry}" >>"${main_registry}"; done
  flock -u 8
fi
printf 'Completed %s/%s tasks.\n' "${#run_dirs[@]}" "${task_count}"
if (( ${#run_dirs[@]} != task_count || worker_failures > 0 )); then
  die "batch incomplete; inspect logs"
fi
printf 'Aggregate: %s/aggregate.json\n' "${batch_root}"
