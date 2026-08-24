# Pinned upstream PatchCore baseline report

Date: 2026-08-24  
Decision: retain pinned upstream PatchCore as the current strong baseline; downgrade
GuardedFusion v2-safe from a competitive innovation candidate to an ablation/negative-result
record.

## Provenance and implementation boundary

- Upstream: `amazon-science/patchcore-inspection`.
- Exact commit: `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`.
- License: Apache-2.0; upstream `LICENSE` and `NOTICE` are retained.
- Checkout: `third_party/patchcore-inspection-fcaa92f`; it was clean after all runs and its
  source was not modified.
- Environment: isolated overlay at `/home/CuiMinghao/envs/evoinspect-patchcore`, using
  PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124 and FAISS CPU 1.15.0.
- Model: WideResNet-50, `layer2` + `layer3`, 1024-dimensional pre-adaptation and target
  embeddings, patch size 3, 10% approximate greedy coreset and 1-NN scoring.
- The upstream example enables GPU FAISS. These runs use CPU FAISS so eight independent
  experiments do not reserve avoidable GPU index memory; the PatchCore model, sampler and
  nearest-neighbour algorithm are otherwise the pinned upstream implementation.
- P1 uses a repository adapter only for the sealed 100+30-style split, development-only
  threshold selection, truth isolation, mask export, metrics and provenance. Support anomalies
  are deliberately not inserted into PatchCore's normal memory.

Static hashes:

| Item | P0 standard | P1 100+30-style |
|---|---:|---:|
| Dataset manifest SHA-256 | `6069c761...cbcd9` | `6069c761...cbcd9` |
| Backbone weights SHA-256 | `95faca4d...950b8` | `95faca4d...950b8` |
| Config SHA-256 | `4e197841...55eae` | `3f3e3dae...416a` |

## P0: standard MVTec AD reproduction

All 15 categories completed at seed 0 with zero failures. The category-macro results are:

| Metric | Reproduced | Upstream README reference |
|---|---:|---:|
| Image AUROC | 0.9909 | 0.992 |
| Full pixel AUROC | 0.9813 | 0.981 |
| Anomaly-image pixel AUROC | 0.9737 | not listed |

The close image and full-pixel agreement is sufficient to pass the P0 reproduction gate. This
is a single-seed reproduction check, not a new benchmark claim.

## P1: isolated 100+30-style protocol

The frozen seeds 133--137 produced 75/75 successful tasks over all 15 categories. A threshold
was selected from the development partition only. Final-test labels were unavailable to the
inference process and used only by the evaluator. Metrics below are category-macro means over
five-seed category means.

| Slice | Accuracy | AUROC | AP | Fixed-threshold F1 |
|---|---:|---:|---:|---:|
| Overall | 0.9194 | 0.9817 | 0.9887 | 0.9224 |
| Seen defects | 0.9412 | 0.9816 | 0.9854 | 0.9269 |
| Unseen defects | 0.9184 | 0.9839 | 0.9710 | 0.8715 |

Pixel metrics:

| Metric | Macro mean |
|---|---:|
| Full pixel AUROC | 0.9811 |
| Full pixel AP | 0.5521 |
| Anomaly-image pixel AUROC | 0.9660 |
| Anomaly-image pixel AP | 0.5599 |

Per-category overall fixed-threshold F1 comparison:

| Category | Upstream PatchCore | PatchCore-lite | GuardedFusion v2-safe |
|---|---:|---:|---:|
| bottle | 0.8826 | 0.9062 | 0.9062 |
| cable | 0.8752 | 0.8461 | 0.8461 |
| capsule | 0.9477 | 0.7858 | 0.8704 |
| carpet | 0.9321 | 0.9520 | 0.9520 |
| grid | 0.8119 | 0.8149 | 0.8149 |
| hazelnut | 0.9851 | 0.9630 | 0.9630 |
| leather | 0.9964 | 1.0000 | 1.0000 |
| metal_nut | 0.9522 | 0.9609 | 0.9609 |
| pill | 0.8962 | 0.8386 | 0.8503 |
| screw | 0.8917 | 0.5725 | 0.7174 |
| tile | 0.9460 | 0.9387 | 0.9547 |
| toothbrush | 0.9714 | 0.6509 | 0.6509 |
| transistor | 0.8972 | 0.7903 | 0.7903 |
| wood | 0.8908 | 0.9095 | 0.9095 |
| zipper | 0.9588 | 0.8711 | 0.8711 |

## Controlled comparison and decision

All three methods used the same categories, seeds and content-hash-isolated splits.

| Comparison | Delta F1 | Delta AUROC | Task wins / ties / losses | Paired task bootstrap F1 95% CI |
|---|---:|---:|---:|---:|
| Upstream vs PatchCore-lite | +0.0690 | +0.0503 | 47 / 6 / 22 | [0.0405, 0.1009] |
| Upstream vs GuardedFusion v2-safe | +0.0518 | +0.0443 | 45 / 6 / 24 | [0.0288, 0.0770] |

For upstream versus GuardedFusion, the category-level paired bootstrap mean F1 delta is
+0.0518 with 95% CI [0.0114, 0.1022]. Therefore GuardedFusion v2-safe is not competitive with
the pinned strong baseline. Its earlier positive comparison against PatchCore-lite remains a
valid controlled result, but it is no longer allowed as a current innovation claim.

## Resource observations

Across 75 P1 tasks, adaptation time was 9.88 s on average (standard deviation 1.31 s; maximum
12.35 s), saved model size averaged 30,999,622 bytes (maximum 32,112,915 bytes), and peak
PyTorch allocation averaged 655,009,826 bytes (maximum 667,889,664 bytes).

Only the isolated bottle seed-133 smoke is retained as latency evidence: RTX 3090, FP32,
batch 1, 224x224, warmup 10, model-only p50 70.76 ms, p95 81.33 ms and maximum 83.54 ms.
Concurrent eight-worker timings are excluded from deployment evidence. The isolated result also
excludes image decode, preprocessing, model load, mask serialization and file I/O.

## Limitations and open work

- This is public MVTec AD, not the official hidden test or an assembly-specific second dataset.
- P1 reuses the same images with five deterministic splits; seeds are not independent datasets.
- AUPRO and region-level metrics are not yet implemented. Pixel AP (0.5521) shows that strong
  image AUROC does not imply submission-ready localization.
- No 2500x2500, GTX 2060-or-lower, CPU, end-to-end or power-controlled benchmark exists.
- The P1 adapter is repository code around an unchanged upstream core; it must not be described
  as an unmodified upstream command-line reproduction.
- Repository Git commit is unavailable because the root `.git` entry is a read-only placeholder;
  experiment rows correctly record `UNAVAILABLE` and dirty state.

## Evidence

- P0 aggregate: `reports/experiments/upstream-patchcore-standard-mvtec15-s0-8gpu-20260823T234806Z-22135/aggregate.json`
- P1 aggregate: `reports/experiments/upstream-patchcore-100-30-mvtec15-5seed-8gpu-20260823T235656Z-29160/aggregate.json`
- P1 per-run metrics, masks and models: the `runs/` directory beside the P1 aggregate.
- Isolated latency smoke: `reports/experiments/upstream-patchcore-100-30-bottle-s133-smoke-20260823T235418Z-27196/metrics.json`
- Expected initial dependency failure: `reports/experiments/upstream-patchcore-standard-bottle-s0-smoke-20260823T234410Z-19987/run.log`
- Global experiment rows: `evidence/experiment_registry.csv`

