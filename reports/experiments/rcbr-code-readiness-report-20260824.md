# RCBR code readiness report

Date: 2026-08-24

## Outcome

RCBR v1, its complete development launcher, and the isolated EfficientAD environment are ready but
untrained. This report contains engineering verification only and introduces no accuracy or
latency claim.

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

- Isolated environment: Python 3.11.16, Torch 2.6.0+cu124, Anomalib 2.3.0.
- `EfficientAd(model_size="small")`, `Engine`, and `Folder`: imported/instantiated successfully.
- `pip check`: no broken requirements.
- EfficientAD teacher weights: two files, 41 MB total, independent SHA-256 recorded.
- Imagenette: downloaded outside the repository, 13,395 files, 1.5 GB parent directory.
- `pytest`: 44 passed in the installed environment.
- `ruff check .`: passed.
- strict `mypy src/evoinspect`: passed.
- `bash -n` for setup and launcher: passed.
- development launcher dry-run: 4 + 8 + 33 = 45 tasks; no training files or GPU process created.
- The first dependency pass failed while compiling `imagecodecs 2026.3.6`, which has no CPython
  3.11 wheel. The installer now pins the newest compatible binary wheel, `2026.1.14`; the rerun
  completed and retained the already installed pinned CUDA Torch packages.
- Actual train/infer execution and resulting metrics remain to be validated by the seed-130 smoke
  stage. Import and construction success is not a training result.

## Run boundary

No GPU training was started and seeds 138–142 remain sealed. At validation time the eight GPUs had
no compute processes; this is only a snapshot and the launcher must recheck before every task. If
the seed-130 functional stage fails, the launcher stops with evidence rather than continuing all 45
tasks.

## Command

```bash
bash scripts/run_rcbr_experiment_suite.sh development 2>&1 | tee logs/rcbr-development.log
```

Return the batch path printed by the launcher. Do not run confirmation before development results
are reviewed and a freeze manifest is created.

Environment evidence: `evidence/environment-efficientad-20260824.txt`.
