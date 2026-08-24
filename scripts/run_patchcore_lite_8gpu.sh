#!/usr/bin/env bash
set -Eeuo pipefail

# Eight independent PatchCore-lite bottle replications. This is a preliminary
# single-category batch, not the formal upstream PatchCore/full-MVTec baseline.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${EVOINSPECT_PYTHON:-/home/CuiMinghao/envs/evoinspect-130/bin/python}"
manifest="${EVOINSPECT_MANIFEST:-/home/CuiMinghao/data/mvtec_ad_official/manifests/mvtec_ad_bottle_manifest.csv}"
config="${EVOINSPECT_CONFIG:-${repo_root}/configs/baselines/patchcore_lite_bottle.yaml}"
gpu_text="${EVOINSPECT_GPU_IDS:-0 1 2 3 4 5 6 7}"
seed_text="${EVOINSPECT_SEEDS:-133 134 135 136 137 138 139 140}"
max_memory_used_mb="${EVOINSPECT_MAX_IDLE_MEMORY_MB:-256}"
max_idle_utilization="${EVOINSPECT_MAX_IDLE_UTILIZATION:-5}"
stage_timeout="${EVOINSPECT_STAGE_TIMEOUT:-30m}"
dry_run="${EVOINSPECT_DRY_RUN:-0}"
batch_stamp="${EVOINSPECT_BATCH_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
batch_id="pc-lite-bottle-8gpu-${batch_stamp}"
batch_root="${EVOINSPECT_BATCH_ROOT:-${repo_root}/reports/experiments/${batch_id}}"
main_registry="${EVOINSPECT_REGISTRY:-${repo_root}/evidence/experiment_registry.csv}"
script_path="${repo_root}/scripts/patchcore_lite_bottle.py"

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

gpu_snapshot() {
  local gpu="$1"
  nvidia-smi -i "${gpu}" \
    --query-gpu=uuid,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
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
  snapshot="$(gpu_snapshot "${gpu}")" || die "cannot query physical GPU ${gpu}"
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
    printf 'GPU %s memory use is %s MiB, above idle limit %s MiB\n' \
      "${gpu}" "${memory_used}" "${max_memory_used_mb}" >&2
    return 1
  fi
  if (( utilization > max_idle_utilization )); then
    printf 'GPU %s utilization is %s%%, above idle limit %s%%\n' \
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

run_one() {
  local gpu="$1"
  local seed="$2"
  local run_id="pc-lite-bottle-s${seed}-${batch_stamp}"
  local run_dir="${batch_root}/${run_id}"
  local worker_registry="${run_dir}/registry.csv"
  local log_file="${run_dir}/run.log"
  local lock_file="/tmp/evoinspect-130-gpu-${gpu}.lock"

  mkdir -p "${run_dir}"
  (
    exec 9>"${lock_file}"
    flock -n 9 || die "GPU ${gpu} is reserved by another EvoInspect launcher"
    assert_gpu_idle "${gpu}"
    printf 'run_id=%s gpu=%s seed=%s started=%s\n' \
      "${run_id}" "${gpu}" "${seed}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${log_file}"

    run_stage "${log_file}" env CUDA_VISIBLE_DEVICES="${gpu}" \
      "${python_bin}" "${script_path}" prepare \
      --manifest "${manifest}" \
      --output-dir "${run_dir}" \
      --seed "${seed}" \
      --category bottle

    # Recheck immediately before the first CUDA allocation. The lock protects
    # cooperating EvoInspect launches; the live query protects shared-server users.
    assert_gpu_idle "${gpu}" >>"${log_file}"
    run_stage "${log_file}" env CUDA_VISIBLE_DEVICES="${gpu}" \
      "${python_bin}" "${script_path}" train \
      --adaptation "${run_dir}/adaptation.csv" \
      --split "${run_dir}/split.json" \
      --output-model "${run_dir}/model.pt" \
      --seed "${seed}" \
      --config "${config}"

    run_stage "${log_file}" env CUDA_VISIBLE_DEVICES="${gpu}" \
      "${python_bin}" "${script_path}" infer \
      --test-inputs "${run_dir}/test_inputs.csv" \
      --model "${run_dir}/model.pt" \
      --output "${run_dir}/predictions.jsonl"

    run_stage "${log_file}" env CUDA_VISIBLE_DEVICES="${gpu}" \
      "${python_bin}" "${script_path}" evaluate \
      --truth "${run_dir}/test_truth.csv" \
      --predictions "${run_dir}/predictions.jsonl" \
      --model "${run_dir}/model.pt" \
      --split "${run_dir}/split.json" \
      --output "${run_dir}/metrics.json" \
      --registry "${worker_registry}"

    printf 'completed=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${log_file}"
  )
}

(( ${#gpu_ids[@]} == 8 )) || die "exactly 8 GPU IDs are required; got ${#gpu_ids[@]}"
(( ${#seeds[@]} == 8 )) || die "exactly 8 seeds are required; got ${#seeds[@]}"
require_uint "idle memory limit" "${max_memory_used_mb}"
require_uint "idle utilization limit" "${max_idle_utilization}"

declare -A seen_gpus=()
declare -A seen_seeds=()
for index in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[index]}"
  seed="${seeds[index]}"
  require_uint "GPU ID" "${gpu}"
  require_uint "seed" "${seed}"
  [[ -z "${seen_gpus[${gpu}]:-}" ]] || die "duplicate GPU ID: ${gpu}"
  [[ -z "${seen_seeds[${seed}]:-}" ]] || die "duplicate seed: ${seed}"
  seen_gpus["${gpu}"]=1
  seen_seeds["${seed}"]=1
done

command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
command -v flock >/dev/null || die "flock not found"
command -v timeout >/dev/null || die "timeout not found"
[[ -x "${python_bin}" ]] || die "Python executable is unavailable: ${python_bin}"
[[ -f "${manifest}" ]] || die "manifest is unavailable: ${manifest}"
[[ -f "${config}" ]] || die "config is unavailable: ${config}"
[[ -f "${script_path}" ]] || die "runner is unavailable: ${script_path}"

printf 'Preflight: checking all eight GPUs before starting any experiment.\n'
for gpu in "${gpu_ids[@]}"; do
  assert_gpu_idle "${gpu}" || die "preflight failed; no training was started"
done

printf 'Batch: %s\nOutput: %s\n' "${batch_id}" "${batch_root}"
for index in "${!gpu_ids[@]}"; do
  printf '  physical GPU %s -> seed %s\n' "${gpu_ids[index]}" "${seeds[index]}"
done

if [[ "${dry_run}" == "1" ]]; then
  printf 'Dry run complete; no directories or training processes were created.\n'
  exit 0
fi

[[ ! -e "${batch_root}" ]] || die "batch output already exists: ${batch_root}"
mkdir -p "${batch_root}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

declare -a pids=()
declare -a run_dirs=()
for index in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[index]}"
  seed="${seeds[index]}"
  run_id="pc-lite-bottle-s${seed}-${batch_stamp}"
  run_dirs+=("${batch_root}/${run_id}")
  run_one "${gpu}" "${seed}" &
  pids+=("$!")
done

failures=0
for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then
    printf 'PASS gpu=%s seed=%s\n' "${gpu_ids[index]}" "${seeds[index]}"
  else
    printf 'FAIL gpu=%s seed=%s log=%s/run.log\n' \
      "${gpu_ids[index]}" "${seeds[index]}" "${run_dirs[index]}" >&2
    failures=$((failures + 1))
  fi
done

if (( failures > 0 )); then
  die "${failures} worker(s) failed; successful runs were retained, aggregation was skipped"
fi

"${python_bin}" "${script_path}" aggregate \
  --run-dirs "${run_dirs[@]}" \
  --output "${batch_root}/aggregate.json"

# Workers write isolated registries to avoid concurrent CSV appends. Merge them
# once, under a cooperative lock, only after every worker succeeds.
exec 8>>"${main_registry}.lock"
flock 8
if [[ ! -s "${main_registry}" ]]; then
  head -n 1 "${run_dirs[0]}/registry.csv" >"${main_registry}"
fi
for run_dir in "${run_dirs[@]}"; do
  tail -n +2 "${run_dir}/registry.csv" >>"${main_registry}"
done
flock -u 8

printf 'All eight runs completed.\nAggregate: %s\n' "${batch_root}/aggregate.json"
