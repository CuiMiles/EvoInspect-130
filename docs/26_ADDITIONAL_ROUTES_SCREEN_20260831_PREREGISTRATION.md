# Additional three-route screen (2026-08-31)

This bounded experiment completes the previously proposed six-route first stage without changing
the frozen submission package. It covers DefectAdapter-130, the official SuperSimpleNet mixed-
supervision implementation, and the official DRA base route required before AHL can be evaluated.

The fixed scope is six categories (`cable`, `capsule`, `screw`, `carpet`, `transistor`, `wood`) and
seed 143. All model fitting and threshold selection use only the existing adaptation manifest. Test
input manifests contain no labels; test truth may be opened only after predictions are durable.

SuperSimpleNet is pinned to author commit `98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d`
(MIT). DRA is pinned to author commit `3fb0e9cce9bee3c23072caa30c889b905c4830ed`.
AHL commit `7114e08243a8c6ea591a5e74c8dc176c913bddb8` is recorded, but full AHL feature
generation is unlocked only if the DRA six-category base screen passes the fixed quality screen.

The screen gate is Overall F1 >= 0.905, eligible Unseen F1 >= 0.84, and Image AUROC >= 0.97.
No learning-rate, epoch, resolution, threshold, or seed search is allowed after test truth is read.

