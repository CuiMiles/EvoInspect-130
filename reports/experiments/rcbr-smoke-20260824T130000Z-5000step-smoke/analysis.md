# RCBR 5000-step smoke pilot analysis

Date: 2026-08-24

## Scope

This batch is an engineering/development pilot, not a final result. It contains the four
pre-registered smoke categories (`wood`, `capsule`, `transistor`, `hazelnut`) at seeds 130--132,
with one EfficientAD-S training per category/seed and six controlled routing strategies evaluated
from the shared checkpoint.

The baseline configuration is `configs/baselines/efficientad_s_5000step_dev.yaml` with 5,000
training steps. The run records report commit `3e040b6d8a472afdcde15647e7083a3fe999f0c0`, a clean
tree, baseline config hash
`391489c724213153f36a73cc32e7f6f32ed829a32aca5676161b3a8005a1e65d`, and RCBR config hash
`2deece2ca9eee62750b80af937cced2e74b72842e07a70fd06bbaee75012b1a3`.

## Gate result

The 12/12 batch completed without task failures, but the pre-registered smoke gate failed all five
checks:

| Check | Value |
|---|---:|
| mean full-RCBR ΔAUPRO@0.05 | -0.170238 |
| mean full-RCBR ΔOverall F1 | -0.200822 |
| mean full-RCBR ΔUnseen F1 | -0.246624 |
| full-RCBR macro AUPRO@0.05 | 0.469380 |
| PatchCore reference macro AUPRO@0.05 | 0.639618 |
| full-RCBR mean ROI area | 0.022626 |
| full-RCBR p95 ROI area | 0.050781 |

The paired category bootstrap intervals for AUPRO@0.05 and Overall F1 remain below zero. These
values are diagnostic evidence only; they are not a final algorithm claim.

## Attribution

The shared EfficientAD pilot was weak on this budget, but the controlled strategies also show a
consistent fusion issue: `risk_calibrated` and `full_rcbr` replaced raw EfficientAD maps with
calibrated CDF-like maps before max fusion. On the pilot, `uniform_downsample` was better than
`full_rcbr` on all four categories for AUPRO@0.05. This is consistent with mixing incompatible
score spaces rather than evidence of a stable RCBR gain.

## Decision

Per the advisor rule, no remaining categories were started after this failed gate and confirmation
seeds were not opened. Exactly one mechanism-level revision was made in commit `a816b32`: risk
calibration remains a routing signal, while global and local anomaly maps stay in the shared raw
EfficientAD score space for fusion. A formal 70,000-step four-category smoke re-evaluation is
running in the sibling batch
`reports/experiments/rcbr-smoke-20260824T143800Z-rcbr-rawfusion-70k`.

## Evidence

- Gate: `smoke-gate.json` (SHA-256
  `02d9d3a30f290c3e535de96b10f9114d3f79b8b5021d65d0429741ccffdc8710`)
- Per-run metrics: `runs/*/result/metrics.json`
- Launcher log: `../rcbr-smoke-20260824T130000Z-5000step-smoke.launcher.log`
- Fixed reference: `../../references/patchcore-dev-130-132-localization-v3/aggregate.json`
