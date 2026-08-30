# Parallel screening preregistration (2026-08-31)

This is an exploratory sprint requested after the frozen submission package was built. The
submission files under `submission/final/` remain the immutable fallback; screening outputs are
written under a new `reports/experiments/parallel-screening-20260831/` directory and are not
promoted automatically.

## Scope

The first screen is six MVTec AD categories (`cable`, `capsule`, `screw`, `carpet`, `transistor`,
`wood`) at seed 143. It reuses the completed EfficientAD-S checkpoints and the already frozen
support/test manifests. No test label is read before all predictions and decisions are durable.
The only predeclared variants are:

1. S-384: one EfficientAD-S forward pass at 384x384;
2. S-512: one EfficientAD-S forward pass at 512x512;
3. StaticTile-S: one global 256x256 pass plus four fixed 2x2 image tiles, merged by pixelwise max.

All thresholds are selected from held-out support normals and available support anomalies only.
Metrics use the same `evaluate_strategy` implementation as the strict v2.1 evaluator, with image
score equal to the spatial maximum and maps resized to the model input shape. This screen is not a
new official quality gate; it is a pre-registered engineering/algorithm screen.

## Promotion and stopping

At most one seed-143 screen is run per category/variant. A route can be expanded only after a
machine-readable aggregate is reviewed; expansion is not automatic and cannot be selected using
final test results. For context, the edge comparison is the frozen EfficientAD-M strict result,
but the existing PatchCore/EfficientAD claims are not changed by this screen. A failed or
incomplete task remains a failure artifact, not a missing result to be silently discarded.

## Resource and sharing rules

The supervisor samples `nvidia-smi` continuously (the running instance polls every 30 seconds and
therefore provides at least a half-hourly audit trail), starts work only on a GPU with at least
4 GiB free and <=5% utilization, and never sends signals to processes not launched by this
experiment. Each process is limited to one CPU thread and 30% of a GPU. Current occupied GPUs are
skipped and rechecked at the next poll. A task has a two-hour timeout.
