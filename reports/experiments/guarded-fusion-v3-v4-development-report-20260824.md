# Upstream PatchCore innovation development gates: v3 and v4

Date: 2026-08-24  
Protocol role: public development experiments only; seeds 130--132.  
Decision: eliminate both variants and keep seeds 138--142 untouched.

## Shared controls

- Baseline core: unchanged Amazon Science PatchCore commit
  `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`, Apache-2.0.
- Data: MVTec AD direct archive, all 15 categories, identical content-hash-isolated 100+30-style
  splits within each paired task.
- Scale: 15 categories x 3 development seeds = 45 paired tasks per variant; both batches
  completed 45/45 with zero failures.
- Thresholds and rescue selection used development partitions only. The public experiment test
  slices are used to judge the development variants; they are not the final hidden simulation.
- Neither variant was allowed to remove a PatchCore-positive decision. Failure of the configured
  development gate produced an exact per-task PatchCore fallback.
- Untouched confirmation seeds 138--142 were not run because neither variant passed its
  development gate.

## v3: global plus top-k residual descriptor

v3 pooled the unchanged upstream PatchCore embeddings into a global mean and a top-32
nearest-neighbour-distance residual mean. A balanced logistic head used support normals and
support defects. A supervised positive could rescue a PatchCore negative only inside a
development-selected PatchCore evidence band.

| Metric | Upstream PatchCore | v3 | Delta |
|---|---:|---:|---:|
| Overall fixed-threshold F1 | 0.9249 | 0.9241 | -0.0008 |
| Overall image AUROC | 0.98446 | 0.98449 | +0.00003 |
| Seen F1 | 0.9276 | 0.9258 | -0.0018 |
| Unseen F1 | 0.8735 | 0.8718 | -0.0016 |

The gate selected exact PatchCore fallback in 44/45 tasks. It selected rescue only for carpet
seed 132, where final development-experiment F1 decreased by 0.0375 and unseen F1 decreased by
0.0693 while AUROC increased by 0.00135. Task wins/ties/losses were 0/44/1.

A diagnostic of the supervised head alone found task-weighted mean deltas of -0.0506 AUROC and
-0.0821 F1 versus PatchCore. This diagnostic is evidence for elimination, not a new model
selection sweep. v3 is therefore eliminated rather than converted into an all-fallback variant.

Operational observations across concurrent workers: adaptation mean 24.09 s, standard deviation
3.39 s and maximum 28.34 s. Concurrent p95 latency is excluded from deployment claims. The
isolated screw seed-130 smoke recorded model-graph p50 138.39 ms and p95 146.63 ms; the current
implementation makes a second embedding pass and excludes decode, preprocessing, loading,
serialization and I/O.

## v4: mask-guided local defect prototypes

v4 used support anomaly masks to select local embeddings from the same upstream PatchCore
feature grid, capped the defect memory at 2,048 prototypes, and used reciprocal nearest-prototype
distance only as a guarded rescue score. Defect prototypes never entered PatchCore's normal
memory.

| Metric | Upstream PatchCore | v4 | Delta |
|---|---:|---:|---:|
| Overall fixed-threshold F1 | 0.9249 | 0.9249 | 0.0000 |
| Overall image AUROC | 0.98446 | 0.98446 | 0.00000 |
| Seen F1 | 0.9276 | 0.9276 | 0.0000 |
| Unseen F1 | 0.8735 | 0.8735 | 0.0000 |

All 45 development gates rejected rescue: 39 because gain/precision constraints were not met and
6 because the development anomaly count was below the predeclared minimum. Task wins/ties/losses
were 0/45/0. The exact equality is a safety property, not evidence of innovation benefit.

The mean stored defect-prototype count was 1,361.2 and the maximum was 2,048. Adaptation mean was
13.30 s, standard deviation 1.66 s and maximum 15.98 s. The isolated grid seed-130 smoke used 609
prototypes and recorded model-graph p50 95.18 ms and p95 98.81 ms; it also excludes decode,
preprocessing, loading, serialization and I/O.

## Decision and next direction

Both variants fail the retention requirement of a stable gain over pinned upstream PatchCore.
They are preserved as controlled negative results and must not be described as retained
innovations. No threshold adjustment is justified: making the guards stricter would merely make
every task equal to the baseline.

The next method-development package should target a weakness demonstrated independently of these
failed image-score fusions: pixel/region localization (full-pixel AP is only 0.5521), AUPRO and
high-resolution routing. It should not consume seeds 138--142 until a new method and its gate are
specified and frozen.

## Evidence

- v3 aggregate: `reports/experiments/guarded-fusion-v3-upstream-dev15-3seed-8gpu-20260824T002208Z-4073/aggregate.json`
- v3 smoke: `reports/experiments/guarded-fusion-v3-upstream-screw-s130-smoke-20260824T002038Z-2516/aggregate.json`
- v4 aggregate: `reports/experiments/masked-prototype-v4-upstream-dev15-3seed-8gpu-20260824T003300Z-26470/aggregate.json`
- v4 smoke: `reports/experiments/masked-prototype-v4-upstream-grid-s130-smoke-20260824T003202Z-25656/aggregate.json`
- Per-task thresholds, selected strategy, model hash and counts: each batch's `runs/*/model/meta.json`.
- Per-task evaluation: each batch's `runs/*/metrics.json`.

