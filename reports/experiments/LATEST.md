# Latest experiment

Updated 2026-08-25 04:06 CST. The latest completed evidence consists of a CPU localization
reevaluation and an RTX 3090 engineering latency benchmark; the formal revised RCBR smoke is
still running and has not produced performance metrics or a gate.

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

## Completed evidence: preliminary latency

Using the existing 5000-step wood checkpoint, the repaired benchmark completed 100 warmup and
1,000 measured iterations on physical GPU 3 (RTX 3090), batch size 1, synthetic 2500×2500 input:

- End-to-end p50/p95/max: 687.479 / 844.530 / 1007.035 ms.
- This is an RTX 3090 synthetic-resolution engineering measurement, not native-resolution
  accuracy, GTX 2060 evidence, or a final-model result.
- Evidence: `reports/experiments/rcbr-latency-20260824T170800Z-rtx3090-preliminary/latency-2500-rtx3090-gpu3.json`.

## Active RCBR smoke

- Batch: `reports/experiments/rcbr-smoke-20260824T164000Z-rcbr-rawfusion-70k-gpu4-7`.
- Seed-130 wood/capsule/transistor/hazelnut completed 4/4 `metrics.json`; the smoke-s131-132 stage
  has 8 tasks total, with 4 completed metrics (wood-s131/132, capsule-s131/132); transistor-s131/132
  transistor-s131/132 are at 16,000 steps and hazelnut-s131/132 at 14,400 steps in the second
  batch.
  `smoke-gate.json`
  is not present yet,
  so no formal
  smoke conclusion is available.
  GPUs 0--3 are currently occupied by another user and were not touched.
- The 5000-step RCBR pilot failed its pre-registered gate; the current 70k raw-score fusion
  rerun is the single authorized mechanism revision and cannot be called a gain until its gate.

Next primary action: monitor the active smoke until `smoke-gate.json` appears. Only a passed gate
unlocks the remaining 11 categories × 3 development seeds; confirmation seeds 138--142 remain
sealed.
