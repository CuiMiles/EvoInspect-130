# Latest experiment

Updated 2026-08-27 15:14 CST. The latest completed evidence includes the formal revised RCBR smoke:
12/12 runs completed, but the preregistered gate failed. RCBR performance expansion is stopped;
the report route is fixed PatchCore plus the system/deployment evidence.

## Submission-readiness audit

The 2026-08-27 audit found that research evidence is sufficient for an honest report, but the
work is not ready for official submission. The required summary PDF, template-based project PDF,
MP4 video and auxiliary ZIP are all still `not_started`; there is no selected deployment model,
real-video/feedback validation, GTX 2060/CPU result or clean-environment submission-package test.
The advisor-facing decision document is
`docs/17_ADVISOR_STATUS_AND_SUBMISSION_READINESS_20260827.md`.

## Completed evidence: localization reevaluation

The evaluator was rerun against the pinned upstream PatchCore P1 source aggregate to verify the
earlier `f1_fixed_threshold` loading failure. It completed 75/75 runs with an empty failures file;
no confirmation seeds 138--142 were opened.

- Source: 15 MVTec AD categories × seeds 133--137, 75/75 runs, no failures.
- Integrity: recomputed full-pixel AUROC and AP matched every stored run within `1e-12`.
- AUPRO@0.30: 0.9342, category-bootstrap 95% CI [0.9165, 0.9491].
- Strict AUPRO@0.05: 0.7241, CI [0.6728, 0.7695].
- PRO at global background FPR 0.01: 0.5764, CI [0.5061, 0.6408].
- Small/medium/large-region AUPRO@0.30: 0.9148 / 0.9411 / 0.9376.
- Wood is the weakest strict localization category: AUPRO@0.05 0.4746 and small-region
  AUPRO@0.30 0.7418.
- Evaluator v2 treats tied background-score groups atomically. Across 75 runs, actual FPR never
  exceeded the requested 1% or 5% diagnostic budget.
- Test-derived operating points are diagnostic only and forbidden for model selection or
  deployment calibration.
- The preceding evaluator-v1 output has identical AUPRO values but is superseded for
  operating-point evidence because tied scores could slightly exceed the requested FPR.
- At the final GPU snapshot, another user occupied GPUs 0--5; GPUs 6--7 were idle. This experiment
  used no GPU and did not interrupt any external process.

Main evidence:

- `reports/experiments/upstream-patchcore-localization-report-20260824.md`
- `reports/experiments/upstream-patchcore-localization-reeval-v2-20260824T101000/aggregate.json`
- `reports/experiments/upstream-patchcore-localization-reeval-20260824T172300-keycheck/aggregate.json`
- `reports/experiments/upstream-patchcore-localization-reeval-v2-20260824T101000/failures.json`
- `reports/experiments/upstream-patchcore-100-30-mvtec15-5seed-8gpu-20260823T235656Z-29160/aggregate.json`

## Completed engineering closure

The deterministic video sequence FSM and two-level GuardedAdapt controller are implemented and
covered by CPU tests. Full repository pytest is 54/54; the new modules pass ruff and mypy. This is
an engineering/protocol result only: no real video accuracy, feedback gain, or deployment latency
claim is attached. A six-scenario synthetic sequence fixture identifies 6/6 expected event patterns.

- `docs/15_SYSTEM_CLOSURE.md`
- `src/evoinspect/sequence.py`
- `src/evoinspect/guarded_adapt.py`
- `tests/test_sequence.py`
- `tests/test_guarded_adapt.py`
- `reports/experiments/system-closure-sequence-fixture-20260825T054200Z/report.json`

## Completed evidence: preliminary latency

Using the existing 5000-step wood checkpoint, the repaired benchmark completed 100 warmup and
1,000 measured iterations on physical GPU 3 (RTX 3090), batch size 1, synthetic 2500×2500 input:

- End-to-end p50/p95/max: 687.479 / 844.530 / 1007.035 ms.
- This is an RTX 3090 synthetic-resolution engineering measurement, not native-resolution
  accuracy, GTX 2060 evidence, or a final-model result.
- Evidence: `reports/experiments/rcbr-latency-20260824T170800Z-rtx3090-preliminary/latency-2500-rtx3090-gpu3.json`.

## Completed evidence: post-smoke 70k latency diagnostic

Using the completed but gate-rejected wood-s130 RCBR checkpoint, the benchmark ran on physical
GPU 4 (RTX 3090), batch size 1, synthetic 2500×2500 input, 100 warmups and 1,000 measured repeats.
The normal sample p50/p95/max was 350.153/362.552/383.206 ms; a registered wood scratch sample
was 371.293/386.795/420.942 ms. Both selected zero ROI, so the local-model branch was not
exercised. This is a diagnostic RTX 3090 measurement for a rejected checkpoint, not GTX 2060,
CPU, native high-resolution accuracy, or a final positive deployment claim.

- `reports/experiments/rcbr-latency-20260825T081500Z-rcbr-70k-negative-gpu4/latency-2500-rtx3090-gpu4.json`
- `reports/experiments/rcbr-latency-20260825T081500Z-rcbr-70k-negative-gpu4/latency-2500-rtx3090-gpu4-scratch000.json`
- `evidence/rcbr-latency-20260825.txt`

## Formal RCBR smoke result

- Batch: `reports/experiments/rcbr-smoke-20260824T164000Z-rcbr-rawfusion-70k-gpu4-7`.
- Scope: capsule, hazelnut, transistor, wood × seeds 130--132; 70,000 steps per run.
- Completion: 12/12 `metrics.json`; `smoke-gate.json` exists with `passed=false`.
- Gate values: mean ΔAUPRO@0.05 `+0.015647` (threshold `+0.025`), worst category `−0.105517`,
  ΔOverall F1 `−0.150921`, and ΔUnseen F1 `−0.165300`; all five preregistered checks failed.
- Macro diagnostics: AUPRO@0.30 `−0.056845`, fixed-small AUPRO@0.05 `−0.104910`, Image AUROC
  `−0.060087`, PRO@1% FPR `+0.082341`, mean ROI area `2.1357%`, P95 ROI area `4.5573%`.
- Consequence: no development expansion, freeze manifest, or confirmation seeds 138--142; the
  RCBR revision is retained only as a controlled negative/diagnostic result.
- GPU safety: the batch used only GPUs 4--7; GPUs 0--3 were occupied by another user and were not
  touched. GPUs 4--7 were released after completion.

Main evidence:

- `reports/experiments/rcbr-smoke-20260824T164000Z-rcbr-rawfusion-70k-gpu4-7/analysis.md`
- `reports/experiments/rcbr-smoke-20260824T164000Z-rcbr-rawfusion-70k-gpu4-7/smoke-gate.json`
- `evidence/rcbr-smoke-20260824T164000Z.txt`
- `docs/16_FINAL_REPORT_EVIDENCE_20260825.md`

Next primary action: freeze the PatchCore accuracy baseline and run the final 2500×2500 deployment
latency/package evidence. Do not launch RCBR development or confirmation experiments.
