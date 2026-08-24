# Leakage-safe vertical slice

## Input manifest

CSV columns are `sample_id,path,label,defect_type,product_id,source,license_id`. Labels are
`normal` or `anomaly`; every anomaly needs a defect type. Validation decodes every image, records
dimensions and SHA-256, and rejects duplicate IDs or duplicate content before splitting.

## Split policy

The split configuration predeclares the random seed, sample counts and unseen defect types. The
split fails when counts are insufficient or when final-test cannot retain normal, seen-anomaly
and unseen-anomaly samples. Roles are mutually exclusive by content hash:

- `support_normal`: build the normal centroid;
- `support_anomaly`: build a known-defect reference centroid;
- `development`: select the fixed decision threshold;
- `final_test`: inaccessible to adaptation and read only by inference/evaluation.

The splitter emits separate views: the adapter receives no final-test rows; inference receives
test image paths with labels and defect metadata blanked; evaluation receives a truth manifest
with image paths blanked. Inference writes predictions without labels, and only the later
evaluation process joins them to sealed final-test truth.

## Fixture baseline boundary

`fixture-stat-v1` uses grid intensity statistics and Euclidean distance to the normal support
centroid. It exists to exercise contracts, failure modes, hashes and evidence generation. It is
not EfficientAD, PatchCore, AnomalyDINO, AHL/DRA, RealNet or GLASS, and must never appear in a
scientific comparison.

The retained negative smoke result is intentional: a development threshold fitted to synthetic
seen scratches detects the held-out seen scratches but misses all synthetic unseen dents. This
demonstrates that ranking quality and fixed-threshold generalization are different questions and
motivates, but does not prove, the HeteroMemory hypothesis.
