from __future__ import annotations

import numpy as np
import pytest

from evoinspect.rcbr import (
    NormalRiskCalibrator,
    Roi,
    RouterLimits,
    cross_fitted_utility_predictions,
    fuse_refinements,
    generate_candidates,
    roi_iou,
    select_under_budget,
)


def test_normal_risk_calibration_is_spatial_and_monotonic() -> None:
    normal = np.stack([np.zeros((4, 4)), np.ones((4, 4))], axis=0)
    calibrator = NormalRiskCalibrator.fit(normal)
    low = calibrator.transform(np.full((4, 4), -1.0))
    high = calibrator.transform(np.full((4, 4), 2.0))
    assert np.all(high > low)
    assert np.all((0.0 < low) & (high <= 1.0))


def test_candidate_coordinates_are_valid_and_deterministic() -> None:
    risk = np.zeros((32, 32), dtype=np.float32)
    risk[4:8, 4:8] = 1.0
    candidates = generate_candidates(risk, risk / 2, risk / 4)
    assert candidates == generate_candidates(risk, risk / 2, risk / 4)
    assert candidates
    assert all(0 <= roi.y0 < roi.y1 <= 32 for roi in candidates)
    assert all(0 <= roi.x0 < roi.x1 <= 32 for roi in candidates)


def test_budget_enforces_cost_area_count_and_overlap() -> None:
    candidates = [
        Roi(0, 0, 4, 4, 1, 1, 1, 0, benefit, cost, str(index))
        for index, (benefit, cost) in enumerate([(0.9, 3.0), (0.8, 3.0), (0.2, 1.0)])
    ]
    selected = select_under_budget(
        candidates, (10, 10), RouterLimits(4.0, max_rois=2, max_total_area_fraction=0.2)
    )
    assert len(selected) == 1
    assert sum(roi.predicted_cost_ms for roi in selected) <= 4.0
    assert sum(roi.area for roi in selected) / 100 <= 0.2


def test_iou_uses_half_open_boxes() -> None:
    left = Roi(0, 0, 2, 2, 0, 0, 0, 0)
    adjacent = Roi(0, 2, 2, 4, 0, 0, 0, 0)
    overlapping = Roi(1, 1, 3, 3, 0, 0, 0, 0)
    assert roi_iou(left, adjacent) == 0.0
    assert roi_iou(left, overlapping) == pytest.approx(1 / 7)


def test_safe_fusion_applies_valid_evidence_and_records_fallback() -> None:
    base = np.zeros((4, 4), dtype=np.float32)
    roi = Roi(1, 1, 3, 3, 1, 1, 1, 0)
    fused, audit = fuse_refinements(base, [(roi, np.ones((2, 2)), 0.8)], minimum_evidence=0.5)
    assert fused.sum() == 4
    assert audit[0]["result"] == "applied"
    fallback, audit = fuse_refinements(base, [(roi, np.ones((1, 1)), 1.0)], minimum_evidence=0.5)
    assert fallback.sum() == 0
    assert audit[0]["result"] == "fallback_global"


def test_five_fold_utility_cross_fit_is_finite() -> None:
    features = np.asarray([[index, index % 2, 0.1, 0.2] for index in range(20)], dtype=float)
    labels = np.asarray([index % 2 for index in range(20)])
    folds = np.asarray([index % 5 for index in range(20)])
    model, predictions = cross_fitted_utility_predictions(features, labels, folds)
    assert np.isfinite(predictions).all()
    assert np.all((0 <= predictions) & (predictions <= 1))
    assert model.predict(features[:1]).shape == (1,)
