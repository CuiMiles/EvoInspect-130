#!/usr/bin/env bash
set -Eeuo pipefail

# MVTec AD 15 categories x 3 seeds on eight independent GPUs. The method is
# explicitly PatchCore-lite; this batch is not the formal upstream PatchCore baseline.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${EVOINSPECT_PYTHON:-/home/CuiMinghao/envs/evoinspect-130/bin/python}"
manifest="${EVOINSPECT_MANIFEST:-/home/CuiMinghao/data/mvtec_ad_official/manifests/mvtec_ad_manifest.csv}"
config="${EVOINSPECT_CONFIG:-${repo_root}/configs/baselines/patchcore_lite_mvtec.yaml}"
gpu_text="${EVOINSPECT_GPU_IDS:-0 1 2 3 4 5 6 7}"
seed_text="${EVOINSPECT_SEEDS:-130 131 132}"
category_text="${EVOINSPECT_CATEGORIES:-cable capsule pill zipper screw carpet leather tile hazelnut metal_nut bottle grid transistor wood toothbrush}"
max_memory_used_mb="${EVOINSPECT_MAX_IDLE_MEMORY_MB:-256}"
max_idle_utilization="${EVOINSPECT_MAX_IDLE_UTILIZATION:-5}"
stage_timeout="${EVOINSPECT_STAGE_TIMEOUT:-30m}"
dry_run="${EVOINSPECT_DRY_RUN:-0}"
batch_stamp="${EVOINSPECT_BATCH_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
batch_label="${EVOINSPECT_BATCH_LABEL:-pc-lite-mvtec15-3seed-8gpu}"
batch_id="${batch_label}-${batch_stamp}"
batch_root="${EVOINSPECT_BATCH_ROOT:-${repo_root}/reports/experiments/${batch_id}}"
main_registry="${EVOINSPECT_REGISTRY:-${repo_root}/evidence/experiment_registry.csv}"
runner="${repo_root}/scripts/patchcore_lite_bottle.py"
weights_checkpoint="${EVOINSPECT_WEIGHTS_CHECKPOINT:-/home/CuiMinghao/.cache/torch/hub/checkpoints/wide_resnet50_2-9ba9bcbe.pth}"

read -r -a categories <<<"${category_text}"
read -r -a gpu_ids <<<"${gpu_text}"
read -r -a seeds <<<"${seed_text}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_uint() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer: ${value}"
}

gpu_processes() {
  local gpu="$1"
  nvidia-smi -i "${gpu}" \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null || true
}

assert_gpu_idle() {
  local gpu="$1"
  local snapshot uuid memory_used utilization processes
  snapshot="$(nvidia-smi -i "${gpu}" \
    --query-gpu=uuid,memory.used,utilization.gpu \
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
  if (( memory_used > max_memory_used_mb )); then
    printf 'GPU %s memory=%sMiB exceeds idle limit=%sMiB\n' \
      "${gpu}" "${memory_used}" "${max_memory_used_mb}" >&2
    return 1
  fi
  if (( utilization > max_idle_utilization )); then
    printf 'GPU %s utilization=%s%% exceeds idle limit=%s%%\n' \
      "${gpu}" "${utilization}" "${max_idle_utilization}" >&2
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
  local gpu="$1"
  local category="$2"
  local seed="$3"
  local run_id="pc-lite-mvtec-${category}-s${seed}-${batch_stamp}"
  local run_dir="${batch_root}/runs/${run_id}"
  local task_registry="${run_dir}/registry.csv"
  local log_file="${run_dir}/run.log"

  mkdir -p "${run_dir}"
  printf 'run_id=%s gpu=%s category=%s seed=%s started=%s\n' \
    "${run_id}" "${gpu}" "${category}" "${seed}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${log_file}"

  if run_stage "${log_file}" env CUDA_VISIBLE_DEVICES="${gpu}" \
      "${python_bin}" "${runner}" prepare \
      --manifest "${manifest}" \
      --output-dir "${run_dir}" \
      --seed "${seed}" \
      --category "${category}" \
    && assert_gpu_idle "${gpu}" >>"${log_file}" \
    && run_stage "${log_file}" env CUDA_VISIBLE_DEVICES="${gpu}" \
      "${python_bin}" "${runner}" train \
      --adaptation "${run_dir}/adaptation.csv" \
      --split "${run_dir}/split.json" \
      --output-model "${run_dir}/model.pt" \
      --seed "${seed}" \
      --config "${config}" \
    && run_stage "${log_file}" env CUDA_VISIBLE_DEVICES="${gpu}" \
      "${python_bin}" "${runner}" infer \
      --test-inputs "${run_dir}/test_inputs.csv" \
      --model "${run_dir}/model.pt" \
      --output "${run_dir}/predictions.jsonl" \
    && run_stage "${log_file}" env CUDA_VISIBLE_DEVICES="${gpu}" \
      "${python_bin}" "${runner}" evaluate \
      --truth "${run_dir}/test_truth.csv" \
      --predictions "${run_dir}/predictions.jsonl" \
      --model "${run_dir}/model.pt" \
      --split "${run_dir}/split.json" \
      --output "${run_dir}/metrics.json" \
      --registry "${task_registry}"; then
    printf 'completed=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${log_file}"
    printf 'PASS gpu=%s category=%s seed=%s\n' "${gpu}" "${category}" "${seed}"
    return 0
  fi

  "${python_bin}" "${runner}" record-failure \
    --registry "${task_registry}" \
    --run-id "${run_id}" \
    --seed "${seed}" \
    --category "${category}" \
    --physical-gpu "${gpu}" \
    --config "${config}" \
    --manifest "${manifest}" \
    --run-dir "${run_dir}" >>"${log_file}" 2>&1 || true
  printf 'FAIL gpu=%s category=%s seed=%s log=%s\n' \
    "${gpu}" "${category}" "${seed}" "${log_file}" >&2
  return 1
}

worker() {
  local slot="$1"
  local gpu="$2"
  local task_index=0
  local failures=0
  local lock_file="/tmp/evoinspect-130-gpu-${gpu}.lock"
  exec 9>"${lock_file}"
  flock -n 9 || die "GPU ${gpu} is reserved by another EvoInspect launcher"

  for category in "${categories[@]}"; do
    for seed in "${seeds[@]}"; do
      if (( task_index % 8 == slot )); then
        if ! assert_gpu_idle "${gpu}"; then
          printf 'GPU %s became busy; worker stops without preemption.\n' "${gpu}" >&2
          return 90
        fi
        if ! run_task "${gpu}" "${category}" "${seed}"; then
          failures=$((failures + 1))
        fi
      fi
      task_index=$((task_index + 1))
    done
  done
  return "${failures}"
}

(( ${#gpu_ids[@]} == 8 )) || die "exactly 8 GPU IDs are required; got ${#gpu_ids[@]}"
(( ${#seeds[@]} >= 1 )) || die "at least one seed is required"
require_uint "idle memory limit" "${max_memory_used_mb}"
require_uint "idle utilization limit" "${max_idle_utilization}"

declare -A seen_gpus=()
declare -A seen_seeds=()
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

command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
command -v flock >/dev/null || die "flock not found"
command -v timeout >/dev/null || die "timeout not found"
command -v sha256sum >/dev/null || die "sha256sum not found"
[[ -x "${python_bin}" ]] || die "Python executable unavailable: ${python_bin}"
[[ -f "${manifest}" ]] || die "manifest unavailable: ${manifest}"
[[ -f "${config}" ]] || die "config unavailable: ${config}"
[[ -f "${runner}" ]] || die "runner unavailable: ${runner}"
[[ -f "${weights_checkpoint}" ]] || die "backbone weights unavailable: ${weights_checkpoint}"

printf 'Preflight: checking all eight physical GPUs before starting any task.\n'
for gpu in "${gpu_ids[@]}"; do
  assert_gpu_idle "${gpu}" || die "preflight failed; no training was started"
done

task_count=$(( ${#categories[@]} * ${#seeds[@]} ))
printf 'Batch: %s\nTasks: %s categories x %s seeds = %s\nOutput: %s\n' \
  "${batch_id}" "${#categories[@]}" "${#seeds[@]}" "${task_count}" "${batch_root}"
for slot in "${!gpu_ids[@]}"; do
  assigned=$((task_count / 8))
  (( slot < task_count % 8 )) && assigned=$((assigned + 1))
  printf '  worker %s physical GPU %s: %s tasks\n' "${slot}" "${gpu_ids[slot]}" "${assigned}"
done

if [[ "${dry_run}" == "1" ]]; then
  printf 'Dry run complete; no experiment directory or training process was created.\n'
  exit 0
fi

[[ ! -e "${batch_root}" ]] || die "batch output already exists: ${batch_root}"
mkdir -p "${batch_root}/runs"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Static inputs are hashed once per batch, then reused by all 45 subprocesses.
# Per-split, model and prediction hashes remain per-run for traceability.
export EVOINSPECT_MANIFEST_SHA256
export EVOINSPECT_WEIGHTS_SHA256
EVOINSPECT_MANIFEST_SHA256="$(sha256sum "${manifest}" | awk '{print $1}')"
EVOINSPECT_WEIGHTS_SHA256="$(sha256sum "${weights_checkpoint}" | awk '{print $1}')"
printf 'manifest_sha256=%s\nweights_sha256=%s\n' \
  "${EVOINSPECT_MANIFEST_SHA256}" "${EVOINSPECT_WEIGHTS_SHA256}" \
  >"${batch_root}/static-input-hashes.txt"

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
    printf 'WORKER FAIL slot=%s gpu=%s log=%s/worker-%s.log\n' \
      "${slot}" "${gpu_ids[slot]}" "${batch_root}" "${slot}" >&2
    worker_failures=$((worker_failures + 1))
  fi
done

mapfile -t run_dirs < <(find "${batch_root}/runs" -mindepth 1 -maxdepth 1 \
  -type d -exec test -f '{}/metrics.json' \; -print | sort)
mapfile -t task_registries < <(find "${batch_root}/runs" -mindepth 2 -maxdepth 2 \
  -type f -name registry.csv | sort)

if (( ${#run_dirs[@]} > 0 )); then
  "${python_bin}" "${runner}" aggregate \
    --run-dirs "${run_dirs[@]}" \
    --output "${batch_root}/aggregate.json"
fi

if (( ${#task_registries[@]} > 0 )); then
  exec 8>>"${main_registry}.lock"
  flock 8
  if [[ ! -s "${main_registry}" ]]; then
    head -n 1 "${task_registries[0]}" >"${main_registry}"
  fi
  for task_registry in "${task_registries[@]}"; do
    tail -n +2 "${task_registry}" >>"${main_registry}"
  done
  flock -u 8
fi

completed="${#run_dirs[@]}"
printf 'Completed %s/%s tasks.\n' "${completed}" "${task_count}"
if (( completed != task_count || worker_failures > 0 )); then
  die "batch is incomplete; inspect worker and task logs before retrying"
fi
printf 'Aggregate: %s/aggregate.json\n' "${batch_root}"
