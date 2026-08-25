# Formal RCBR 70k smoke result

## Decision

The preregistered smoke gate **failed**. All 12 runs (four MVTec AD categories × seeds
130--132) completed and produced `metrics.json`, but the RCBR revision is not retained as a
competitive performance method. The development expansion and confirmation seeds 138--142 remain
locked. This is a verified negative result, not a missing-run result.

## Protocol and provenance

- Batch: `rcbr-smoke-20260824T164000Z-rcbr-rawfusion-70k-gpu4-7`
- Categories: capsule, hazelnut, transistor, wood
- Seeds: 130, 131, 132
- Training budget: 70,000 steps per run
- Baseline config hash: `caf94b98282f37f6491ff2e1bc86aaa48897298ca51113a04b4ed6d3152af9af`
- RCBR config hash: `2deece2ca9eee62750b80af937cced2e74b72842e07a70fd06bbaee75012b1a3`
- Upstream Anomalib commit: `091ca6aca92c8d0e416394f79e52f5a3cea3db73`
- Gate file SHA-256: `b09bb7d5e4b9b03ad91e53df5a3b4e78baba4c44c86b74b940446c3637026af6`
- All 12 metric records have status `completed`.

The recorded Git commits are `f2b2d300`, `b078aef`, `11987ccb`, and `411ec24`. The differences
between these commits are status-document-only updates made while the batch was running; the
baseline/RCBR configuration hashes are identical across all 12 records. The raw per-run records
and the gate JSON remain authoritative.

## Preregistered gate

| Check | Observed | Requirement | Result |
|---|---:|---:|---|
| Mean ΔAUPRO@0.05 | +0.015647 | ≥ +0.025 | FAIL |
| Categories non-decreasing | 2/4 | ≥ 3/4 | FAIL |
| Worst-category ΔAUPRO@0.05 | −0.105517 | ≥ −0.015 | FAIL |
| ΔOverall F1 | −0.150921 | ≥ −0.005 | FAIL |
| ΔUnseen F1 | −0.165300 | ≥ −0.010 | FAIL |

## Macro results: full RCBR minus fixed PatchCore

| Metric | Full RCBR | PatchCore reference | Δ |
|---|---:|---:|---:|
| AUPRO@0.05 | 0.655265 | 0.639618 | +0.015647 |
| AUPRO@0.30 | 0.849750 | 0.906595 | −0.056845 |
| Fixed-small AUPRO@0.05 | 0.670734 | 0.775643 | −0.104910 |
| Image AUROC | 0.927320 | 0.987407 | −0.060087 |
| Overall F1 | 0.770043 | 0.920964 | −0.150921 |
| Unseen F1 | 0.760808 | 0.926108 | −0.165300 |
| PRO@1% FPR | 0.564093 | 0.481752 | +0.082341 |
| Mean ROI area fraction | 0.021357 | 0 | +0.021357 |
| P95 ROI area fraction | 0.045573 | 0 | +0.045573 |

## Category paired deltas

| Category | ΔAUPRO@0.05 | ΔOverall F1 | ΔUnseen F1 |
|---|---:|---:|---:|
| capsule | −0.017404 | −0.151149 | −0.210157 |
| hazelnut | +0.030234 | −0.211303 | −0.188148 |
| transistor | −0.105517 | −0.286067 | −0.324178 |
| wood | +0.155276 | +0.044836 | +0.061283 |

## Consequence

No RCBR gain, real-time claim, or confirmation-seed result may be reported. The report should use
the fixed upstream PatchCore reproduction as the accuracy baseline, describe RCBR as an implemented
and formally rejected revision, and retain this batch as controlled negative/diagnostic evidence.

Raw evidence: `smoke-gate.json`, the 12 per-run `metrics.json` files, and the batch
`experiment_registry.csv` in this directory.
