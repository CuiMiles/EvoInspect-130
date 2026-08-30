# HeteroResidual-S screen (2026-08-31)

This is a separate exploratory screen. The machine-validated final submission remains unchanged.
The source is the completed EfficientAD-S strict seed-143 checkpoint for each of six categories:
`cable`, `capsule`, `screw`, `carpet`, `transistor`, and `wood`.

HeteroResidual-S keeps the EfficientAD backbone and its teacher/student/autoencoder weights frozen.
One `get_maps(normalize=True)` call yields student-teacher and student-autoencoder maps. A fixed
feature vector contains the fused-map maximum plus four auxiliary residual-evidence heads (top
fractions, branch maxima and branch disagreement). A non-negative residual head is fitted only on
support normals and support anomalies. Leave-one-defect-type-out support validation is a fixed
selection rule; if it fails, the fused maximum is retained. The threshold is selected from support
rows only. Test truth is opened only after all test predictions and decisions are durable.

The screen has six category/seed-143 runs and is not an official quality gate. It is not allowed to
change the final PDF, video, ZIP, claim ledger promotion, or hardware claims automatically.
