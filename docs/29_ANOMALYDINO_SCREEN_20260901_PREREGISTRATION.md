# AnomalyDINO six-category screen (2026-09-01)

This is a bounded exploratory screen requested after the frozen submission package was
completed. It must not overwrite `submission/final` or change claims automatically.

## Fixed protocol

- Categories: `cable`, `capsule`, `screw`, `carpet`, `transistor`, and `wood`.
- Seed: 143 and the already frozen per-category manifests.
- The first 16 `support_normal` rows in each manifest are the only AnomalyDINO memory
  reference images. No optimization or backbone training is performed.
- The remaining support normals are used only for threshold calibration, together with all
  available `support_anomaly` rows. No development or final-test labels are used for this
  calibration.
- DINOv2-Small (`dinov2_vit_small_14`), one nearest neighbour, no masking, no coreset, and
  252x252 preprocessing are fixed before reading final-test truth.
- Every final-test score and map is persisted before `test_truth.csv` is opened.

## Promotion gate

The six-category macro means must satisfy all three thresholds: Overall F1 >= 0.925,
eligible Unseen F1 >= 0.88, and Image AUROC >= 0.985, with six successful categories and
zero test-label leakage. A failure stops this route; no additional seed, category, or
parameter search is authorized under this registration.

AnomalyDINO is an upstream baseline, not an EvoInspect-130 originality claim. Any result is
reported as an exploratory comparison and remains outside `submission/final` unless reviewed
explicitly.
