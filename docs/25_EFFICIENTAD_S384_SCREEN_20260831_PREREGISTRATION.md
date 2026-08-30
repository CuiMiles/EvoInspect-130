# EfficientAD-S384 training screen (2026-08-31)

This is a bounded exploratory screen requested after the frozen submission package was built. The
machine-validated files under `submission/final/` remain the fallback and are not edited by this
experiment.

## Frozen scope

- Six categories: `cable`, `capsule`, `screw`, `carpet`, `transistor`, `wood`.
- One predeclared seed: `143`.
- EfficientAD-S student/autoencoder are trained from the existing project training entry point at
  384x384; the teacher and ImageNette weights are unchanged.
- 70,000 optimization steps, batch size 1, one data-loader worker per process, and no defect
  backbone fine-tuning.
- Four initially idle GPUs (4--7) are used at most one task per GPU. The queue rechecks every 30
  seconds and never signals a process it did not launch.

## Evaluation contract

The training run writes ordinary diagnostics, then the same strict v2.1 evaluator is run with a
384x384 input configuration. Thresholds and map quantiles use support normals/anomalies only;
development and final-test labels are not used for thresholding or model selection. The evaluator
opens final-test truth only after all predictions and decisions are durable. Results are exploratory
and require aggregate review; they cannot replace EfficientAD-M/S or change the final materials
automatically.

## Stop rule

This is one six-category screen, not an invitation to search over steps, seeds, learning rates or
resolutions. If it does not beat the frozen S-256 six-category aggregate under the same evaluator,
it is recorded as a negative result and no further EfficientAD resolution training is started.
