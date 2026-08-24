#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${EVOINSPECT_PYTHON:-/home/CuiMinghao/apps/miniforge3/bin/python}"
run_id="${EVOINSPECT_RUN_ID:-fixture-$(date -u +%Y%m%dT%H%M%SZ)}"
fixture_root="${EVOINSPECT_FIXTURE_ROOT:-/tmp/evoinspect-fixture-${run_id}}"
evidence_root="${EVOINSPECT_EVIDENCE_ROOT:-${repo_root}/reports/experiments/${run_id}}"

mkdir -p "${fixture_root}" "${evidence_root}"
"${python_bin}" "${repo_root}/scripts/generate_fixture.py" --output-dir "${fixture_root}"

export PYTHONPATH="${repo_root}/src"
"${python_bin}" -m evoinspect.cli data validate \
  --manifest "${fixture_root}/manifest.csv" \
  --output "${evidence_root}/validated.csv" \
  --summary "${evidence_root}/validation.json"
"${python_bin}" -m evoinspect.cli data split \
  --manifest "${evidence_root}/validated.csv" \
  --config "${repo_root}/configs/smoke.yaml" \
  --output "${evidence_root}/split.csv" \
  --adaptation-output "${evidence_root}/adaptation_manifest.csv" \
  --test-inputs-output "${evidence_root}/test_inputs.csv" \
  --test-truth-output "${evidence_root}/test_truth.csv" \
  --summary "${evidence_root}/split.json"
"${python_bin}" -m evoinspect.cli adapt product \
  --manifest "${evidence_root}/adaptation_manifest.csv" \
  --config "${repo_root}/configs/smoke.yaml" \
  --output "${evidence_root}/model.json" \
  --summary "${evidence_root}/adaptation.json"
"${python_bin}" -m evoinspect.cli infer image \
  --manifest "${evidence_root}/test_inputs.csv" \
  --model "${evidence_root}/model.json" \
  --roles final_test \
  --output "${evidence_root}/predictions.jsonl" \
  --summary "${evidence_root}/inference.json"
"${python_bin}" -m evoinspect.cli evaluate \
  --manifest "${evidence_root}/test_truth.csv" \
  --predictions "${evidence_root}/predictions.jsonl" \
  --model "${evidence_root}/model.json" \
  --config "${repo_root}/configs/smoke.yaml" \
  --output "${evidence_root}/metrics.json" \
  --registry "${repo_root}/evidence/experiment_registry.csv" \
  --run-id "${run_id}"
"${python_bin}" -m evoinspect.cli report generate \
  --metrics "${evidence_root}/metrics.json" \
  --model "${evidence_root}/model.json" \
  --output "${evidence_root}/report.md"

printf '%s\n' "${evidence_root}"
