# Latest experiment

Latest completed experiment: evaluator-v2 localization reevaluation of 75 saved masks from the
pinned upstream PatchCore P1 runs. This was CPU-only and did not open confirmation seeds
138--142.

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
- `reports/experiments/upstream-patchcore-localization-reeval-v2-20260824T101000/failures.json`
- `reports/experiments/upstream-patchcore-100-30-mvtec15-5seed-8gpu-20260823T235656Z-29160/aggregate.json`

RCBR code and isolated EfficientAD environment preparation are now complete; see
`reports/experiments/rcbr-code-readiness-report-20260824.md`. This is not a completed model
experiment and therefore does not replace the verified metrics above.

Next primary action: run the gated RCBR development suite on seeds 130--132 and return its batch
path for review. Confirmation seeds 138--142 remain sealed.
