# Upstream PatchCore localization reevaluation

Date: 2026-08-24

## Scope

This is a CPU-only reevaluation of the 75 saved anomaly masks from the pinned Amazon Science
PatchCore commit `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`. It does not retrain a model and does not
touch confirmation seeds 138--142. The source protocol is MVTec AD, 15 categories, seeds
133--137, with up to 100 normal and 30 seen-anomaly support samples kept separate from final
test data.

The implementation follows the standard PRO convention: each 8-connected ground-truth region
receives equal total weight, the false-positive rate is measured over all background pixels, and
the curve is linearly clipped and normalized at the requested limit. We report both AUPRO@0.30
and the stricter AUPRO@0.05. This is consistent with the open-source
[Anomalib AUPRO implementation](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/metrics/aupro.py)
and the low-FPR reporting emphasis in the [MVTec AD 2 paper](https://arxiv.org/abs/2503.21622).

## Main results

All values are macro means over 15 category means; each category mean contains five seeds.
Confidence intervals bootstrap categories with 10,000 deterministic draws.

| Metric | Mean | Category-bootstrap 95% CI |
|---|---:|---:|
| AUPRO@0.30 | 0.934161 | [0.916532, 0.949062] |
| AUPRO@0.05 | 0.724099 | [0.672840, 0.769526] |
| PRO at global background FPR 0.01 | 0.576406 | [0.506066, 0.640775] |
| Small-region AUPRO@0.30 | 0.914800 | [0.881673, 0.940237] |
| Medium-region AUPRO@0.30 | 0.941085 | [0.926710, 0.953986] |
| Large-region AUPRO@0.30 | 0.937604 | [0.913464, 0.957136] |

The broad AUPRO@0.30 result exceeds the internal 0.90 research target, but the strict
AUPRO@0.05 and PRO at 1% FPR expose a substantial low-false-positive localization gap. The
small-region slice is also the weakest size slice. These results do not support a claim that
pixel localization is solved.

## Per-category localization

| Category | AUPRO@0.30 | AUPRO@0.05 | Small-region AUPRO@0.30 |
|---|---:|---:|---:|
| bottle | 0.954196 | 0.736878 | 0.952534 |
| cable | 0.939300 | 0.712387 | 0.920805 |
| capsule | 0.927164 | 0.628130 | 0.890101 |
| carpet | 0.944206 | 0.769489 | 0.929880 |
| grid | 0.901165 | 0.685639 | 0.834493 |
| hazelnut | 0.931942 | 0.747781 | 0.892477 |
| leather | 0.975980 | 0.856035 | 0.961527 |
| metal_nut | 0.930574 | 0.645715 | 0.905219 |
| pill | 0.943416 | 0.798590 | 0.954230 |
| screw | 0.962616 | 0.802446 | 0.927054 |
| tile | 0.923621 | 0.723290 | 0.954263 |
| toothbrush | 0.973060 | 0.838566 | 0.981405 |
| transistor | 0.902351 | 0.643191 | 0.939838 |
| wood | 0.844396 | 0.474627 | 0.741757 |
| zipper | 0.958421 | 0.798726 | 0.936418 |

Wood is the clearest failure slice, followed by capsule, transistor, and metal_nut in the strict
low-FPR metric. The next localization method should therefore be developed on the existing
development seeds and target high-resolution/local refinement rather than image-score threshold
sweeps.

## Correctness and leakage controls

- 75/75 saved-mask runs completed; `failures.json` is empty.
- Recomputed full-pixel AUROC and AP match every stored source metric within `1e-12`.
- The mask transform deliberately matches the pinned upstream evaluator: bilinear
  `Resize(int)`, `CenterCrop`, `ToTensor`, then integer cast. Changing mask geometry would require
  a separately versioned protocol.
- Conservative diagnostic thresholds include a tied background-score group only when the entire
  group stays within the integer false-positive budget. Across all runs, actual FPR ranges were
  0.009955--0.010000 and 0.049785--0.049999 for the 1% and 5% requests.
- The test-derived operating-point diagnostics are forbidden for threshold selection, model
  selection, early stopping, or deployment calibration. At those diagnostic points only, the
  macro region recall at overlap at least 0.30 and requested FPR 5% was 0.951577; false-positive
  regions per normal image at requested FPR 1% was 0.055200.
- An earlier evaluator output at `upstream-patchcore-localization-reeval-20260824T084500` used a
  non-conservative tied-threshold rule for operating-point diagnostics. Its AUPRO values are
  unchanged, but that output is superseded by evaluator v2 for operating-point evidence.

## Evidence

- Source masks and metrics:
  `reports/experiments/upstream-patchcore-100-30-mvtec15-5seed-8gpu-20260823T235656Z-29160/`
- Corrected aggregate:
  `reports/experiments/upstream-patchcore-localization-reeval-v2-20260824T101000/aggregate.json`
- Per-run outputs and hashes:
  `reports/experiments/upstream-patchcore-localization-reeval-v2-20260824T101000/runs/`
- Failure record:
  `reports/experiments/upstream-patchcore-localization-reeval-v2-20260824T101000/failures.json`
- Evaluator config hash:
  `3773c7b55390f0003a826959d681164f86a6308f9f8463034f881d444df6bb22`

No GPU was used for this reevaluation. At completion, GPUs 0--5 were occupied by another user's
processes and GPUs 6--7 were idle; no process was interrupted or modified.
