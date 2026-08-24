#!/usr/bin/env bash
set -Eeuo pipefail

# Fixed Amazon Science PatchCore baseline. Each task is an independent MVTec
# category/seed run; no DDP is used. The upstream checkout is never modified.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${EVOINSPECT_PATCHCORE_PYTHON:-/home/CuiMinghao/envs/evoinspect-patchcore/bin/python}"
data_root="${EVOINSPECT_MVTEC_ROOT:-/home/CuiMinghao/data/mvtec_ad_official/raw}"
manifest="${EVOINSPECT_MANIFEST:-/home/CuiMinghao/data/mvtec_ad_official/manifests/mvtec_ad_manifest.csv}"
upstream="${EVOINSPECT_PATCHCORE_UPSTREAM:-${repo_root}/third_party/patchcore-inspection-fcaa92f}"
config="${EVOINSPECT_CONFIG:-${repo_root}/configs/baselines/patchcore_upstream_standard_mvtec.yaml}"
gpu_text="${EVOINSPECT_GPU_IDS:-0 1 2 3 4 5 6 7}"
seed_text="${EVOINSPECT_SEEDS:-0}"
category_text="${EVOINSPECT_CATEGORIES:-bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper}"
max_memory_used_mb="${EVOINSPECT_MAX_IDLE_MEMORY_MB:-256}"
max_idle_utilization="${EVOINSPECT_MAX_IDLE_UTILIZATION:-5}"
task_timeout="${EVOINSPECT_TASK_TIMEOUT:-60m}"
dry_run="${EVOINSPECT_DRY_RUN:-0}"
batch_stamp="${EVOINSPECT_BATCH_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
batch_label="${EVOINSPECT_BATCH_LABEL:-upstream-patchcore-standard-mvtec15-s0-8gpu}"
batch_root="${EVOINSPECT_BATCH_ROOT:-${repo_root}/reports/experiments/${batch_label}-${batch_stamp}}"
upstream_commit="fcaa92f124fb1ad74a7acf56726decd4b27cbcad"
weights_checkpoint="${EVOINSPECT_WEIGHTS_CHECKPOINT:-/home/CuiMinghao/.cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth}"

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
  nvidia-smi -i "${gpu}" --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null || true
}

assert_gpu_idle() {
  local gpu="$1"
  local snapshot uuid memory_used utilization processes
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

run_task() {
  local gpu="$1"
  local category="$2"
  local seed="$3"
  local run_id="upstream-pc-mvtec-${category}-s${seed}-${batch_stamp}"
  local run_dir="${batch_root}/runs/${run_id}"
  local group="IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S${seed}"
  local result_file="${run_dir}/upstream_patchcore/${group}/results.csv"
  mkdir -p "${run_dir}"
  {
    printf 'run_id=%s\n' "${run_id}"
    printf 'upstream_commit=%s\n' "${upstream_commit}"
    printf 'physical_gpu=%s\ncategory=%s\nseed=%s\n' "${gpu}" "${category}" "${seed}"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  } >"${run_dir}/run-meta.txt"

  if timeout --signal=TERM --kill-after=30s "${task_timeout}" \
    env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${upstream}/src" \
    "${python_bin}" "${upstream}/bin/run_patchcore.py" \
    --gpu 0 --seed "${seed}" --save_patchcore_model \
    --log_group "${group}" --log_project upstream_patchcore "${run_dir}" \
    patch_core -b wideresnet50 -le layer2 -le layer3 \
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 1 --patchsize 3 \
    sampler -p 0.1 approx_greedy_coreset \
    dataset --resize 256 --imagesize 224 --batch_size 2 --num_workers 4 \
    -d "${category}" mvtec "${data_root}" >"${run_dir}/run.log" 2>&1; then
    [[ -s "${result_file}" ]] || return 93
    printf 'completed_at=%s\nstatus=completed\n' "$(date --iso-8601=seconds)" \
      >>"${run_dir}/run-meta.txt"
    printf 'PASS gpu=%s category=%s seed=%s\n' "${gpu}" "${category}" "${seed}"
    return 0
  fi
  printf 'completed_at=%s\nstatus=failed\n' "$(date --iso-8601=seconds)" \
    >>"${run_dir}/run-meta.txt"
  printf 'FAIL gpu=%s category=%s seed=%s log=%s\n' \
    "${gpu}" "${category}" "${seed}" "${run_dir}/run.log" >&2
  return 1
}

worker() {
  local slot="$1"
  local gpu="$2"
  local task_index=0
  local failures=0
  exec 9>"/tmp/evoinspect-130-gpu-${gpu}.lock"
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

(( ${#gpu_ids[@]} == 8 )) || die "exactly 8 GPU IDs are required"
(( ${#seeds[@]} >= 1 )) || die "at least one seed is required"
for gpu in "${gpu_ids[@]}"; do require_uint "GPU ID" "${gpu}"; done
for seed in "${seeds[@]}"; do require_uint "seed" "${seed}"; done
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
command -v flock >/dev/null || die "flock not found"
command -v timeout >/dev/null || die "timeout not found"
[[ -x "${python_bin}" ]] || die "Python unavailable: ${python_bin}"
[[ -d "${data_root}" ]] || die "MVTec root unavailable: ${data_root}"
[[ -f "${manifest}" ]] || die "manifest unavailable: ${manifest}"
[[ -f "${config}" ]] || die "config unavailable: ${config}"
[[ -f "${upstream}/bin/run_patchcore.py" ]] || die "upstream entrypoint unavailable"
[[ -f "${weights_checkpoint}" ]] || die "backbone weights unavailable"
actual_commit="$(git -C "${upstream}" rev-parse HEAD)"
[[ "${actual_commit}" == "${upstream_commit}" ]] || die "upstream commit mismatch: ${actual_commit}"
[[ -z "$(git -C "${upstream}" status --short)" ]] || die "upstream checkout is modified"

printf 'Preflight: checking all eight physical GPUs.\n'
for gpu in "${gpu_ids[@]}"; do assert_gpu_idle "${gpu}" || die "preflight failed"; done
task_count=$(( ${#categories[@]} * ${#seeds[@]} ))
printf 'Tasks: %s categories x %s seeds = %s\nOutput: %s\n' \
  "${#categories[@]}" "${#seeds[@]}" "${task_count}" "${batch_root}"
if [[ "${dry_run}" == "1" ]]; then
  printf 'Dry run complete; no training directory was created.\n'
  exit 0
fi
[[ ! -e "${batch_root}" ]] || die "batch output exists: ${batch_root}"
mkdir -p "${batch_root}/runs"
{
  printf 'upstream_commit=%s\n' "${upstream_commit}"
  printf 'upstream_license=Apache-2.0\n'
  printf 'manifest_sha256=%s\n' "$(sha256sum "${manifest}" | awk '{print $1}')"
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
completed="$(find "${batch_root}/runs" -name results.csv -type f | wc -l)"
printf 'Completed %s/%s tasks.\n' "${completed}" "${task_count}"
if (( completed != task_count || worker_failures > 0 )); then
  die "batch incomplete; inspect worker/task logs"
fi
printf 'Batch PASS: %s\n' "${batch_root}"
