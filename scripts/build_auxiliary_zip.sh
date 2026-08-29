#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${repo_root}/submission/drafts/auxiliary_material.zip}"
[[ ! -e "${output}" ]] || { printf 'refusing to overwrite: %s\n' "${output}" >&2; exit 2; }
mkdir -p "$(dirname "${output}")"
cd "${repo_root}"

zip -q -r "${output}" \
  AGENTS.md README.md STATUS.md project_spec.yaml pyproject.toml \
  src scripts configs docs evidence \
  artifacts/model_registry.yaml data/dataset_registry.yaml \
  submission/README.md submission/metadata.yaml submission/works_intro.txt submission/works_intro.html \
  submission/project_document.html submission/assets \
  reports/experiments/upstream-patchcore-100-30-mvtec15-5seed-8gpu-20260823T235656Z-29160/aggregate.json \
  reports/experiments/upstream-patchcore-baselines-report-20260824.md \
  reports/experiments/upstream-patchcore-localization-reeval-20260824T172300-keycheck/aggregate.json \
  reports/experiments/rcbr-smoke-20260824T164000Z-rcbr-rawfusion-70k-gpu4-7/analysis.md \
  reports/experiments/rcbr-smoke-20260824T164000Z-rcbr-rawfusion-70k-gpu4-7/smoke-gate.json \
  reports/experiments/guarded-adapt-replay-20260827T194500-cpu/report.json \
  reports/experiments/guarded-adapt-risk-20260829-preregistered-e17419c/report.json \
  reports/experiments/video-demo-20260827T194000-cpu-annotated/report.json \
  -x '*/__pycache__/*' '*.pyc' '*.ckpt' '*.npz' '*.faiss' '*.lock' \
     'reports/experiments/*/annotated/*' 'data/video/*' \
     'evidence/submission_artifact_validation.json'

size="$(stat -c '%s' "${output}")"
(( size <= 209715200 )) || { printf 'ZIP exceeds 200 MiB: %s\n' "${size}" >&2; exit 3; }
unzip -tq "${output}" >/dev/null
printf '%s %s bytes\n' "${output}" "${size}"
