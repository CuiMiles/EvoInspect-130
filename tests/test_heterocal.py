from __future__ import annotations

import numpy as np

from evoinspect.heterocal import fit_nonnegative_residual, leave_one_defect_type_out


def test_residual_head_has_bounded_nonnegative_evidence() -> None:
    features = np.asarray([[0.0, 0.0], [0.2, 0.1], [2.0, 2.0], [3.0, 4.0]])
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    head = fit_nonnegative_residual(
        features,
        labels,
        features[:2],
        delta=0.35,
        l2=0.02,
        learning_rate=0.03,
        iterations=100,
        scale_floor=0.001,
    )
    assert np.all(head.weights >= 0)
    base = (features[:, 0] - head.median[0]) / head.scale[0]
    assert np.all(np.abs(head.score(features) - base) <= 0.35 + 1e-12)


def test_single_defect_type_is_never_selected() -> None:
    config = {
        "selection": {"minimum_defect_types": 2},
        "calibrator": {},
    }
    result = leave_one_defect_type_out(
        np.asarray([[0.0, 0.0], [1.0, 1.0]]),
        np.asarray([0, 1], dtype=np.int64),
        ["", "only"],
        config,
    )
    assert result["accepted"] is False
    assert result["reason"] == "insufficient_defect_types"
