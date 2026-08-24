# RCBR code readiness report

Date: 2026-08-24

## Outcome

RCBR v1 and its complete development launcher are code-ready but untrained. This report contains
engineering verification only and introduces no accuracy or latency claim.

## Implemented evidence

- Pinned Anomalib v2.3.0 commit `091ca6aca92c8d0e416394f79e52f5a3cea3db73`.
- One EfficientAD-S training per category/seed with six shared-weight controlled strategies.
- Held-out normal spatial risk calibration and five-fold support-defect ROI utility estimation.
- Multi-signal ROI candidates, NMS, measured-cost hard budget, maximum four ROIs, area cap,
  monotonic fusion, explicit fallback and per-image routing audit.
- Fixed relative defect areas: Tiny ≤0.1%, Small 0.1%–1%, Large >1%.
- Matching PatchCore seeds 130–132 saved-mask CPU reference generation.
- Four-category and full-15-category gates, paired category bootstrap, and confirmation freeze lock.
- Shared-GPU inventory, busy-card skip, per-card lock, task timeout and per-task recheck.
- Synthetic-resolution 2500×2500 RTX 3090 latency harness with 100 warmups and 1,000 repeats.

## Verification

- `pytest`: 43 passed.
- `ruff check .`: passed.
- strict `mypy src/evoinspect`: passed.
- `bash -n` for setup and launcher: passed.
- development launcher dry-run: 4 + 8 + 33 = 45 tasks; no training files or GPU process created.
- Direct Anomalib API import in the existing project environment was not possible because
  `omegaconf` is intentionally absent. This is expected; the isolated setup script installs the
  pinned upstream dependencies before formal training. Actual train/infer API execution remains
  to be validated by the first smoke task.

## Run boundary

No EfficientAD environment was installed, no upstream weight was downloaded, no GPU training was
started, and seeds 138–142 remain sealed. If the seed-130 functional stage fails, the launcher
stops with evidence rather than continuing all 45 tasks.

## Command

```bash
bash scripts/setup_efficientad_env.sh
bash scripts/run_rcbr_experiment_suite.sh development 2>&1 | tee logs/rcbr-development.log
```

Return the batch path printed by the launcher. Do not run confirmation before development results
are reviewed and a freeze manifest is created.
