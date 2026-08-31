# Dinomaly six-category screen (2026-09-01)

This is a bounded exploratory screen. It does not overwrite `submission/final` and does not
promote a model without a separate aggregate review.

## Fixed protocol

- Categories: `cable`, `capsule`, `screw`, `carpet`, `transistor`, and `wood`, seed 143.
- The first 80 `support_normal` rows train the normal-only Dinomaly bottleneck and decoder.
  The remaining 20 support normals and all 30 support anomalies are used only for the fixed
  image-level threshold. No development or final-test labels are used for training or tuning.
- The DINOv2-Small encoder (`dinov2_vit_small_14`) is frozen. Decoder depth 8, 224x224 input,
  500 optimization steps, AdamW learning rate 0.002, weight decay 1e-4, and gradient clipping
  at 0.1 are fixed before the screen.
- Each final-test score and map is persisted before `test_truth.csv` is opened.

## Promotion gate

The six-category macro means must satisfy Overall F1 >= 0.925, eligible Unseen F1 >= 0.88,
Image AUROC >= 0.985, six successful categories, and zero test-label leakage. Failure stops
this route: no additional seed, category, resolution, or parameter search is authorized.

Dinomaly is an upstream baseline comparison, not an EvoInspect-130 originality claim. Any
result remains exploratory and outside `submission/final` unless reviewed explicitly.
