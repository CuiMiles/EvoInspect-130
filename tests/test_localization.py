from __future__ import annotations

import numpy as np
import pytest

from evoinspect.localization import compute_localization_diagnostics, connected_region_labels


def test_connected_components_are_unique_and_eight_connected() -> None:
    targets = np.zeros((2, 3, 3), dtype=np.bool_)
    targets[0, 0, 0] = True
    targets[0, 1, 1] = True
    targets[1, 2, 2] = True
    labels, sizes = connected_region_labels(targets)
    assert labels[0, 0, 0] == labels[0, 1, 1]
    assert labels[1, 2, 2] != labels[0, 0, 0]
    assert sizes[1:].tolist() == [2, 1]


def test_perfect_localization_has_unit_aupro() -> None:
    targets = np.zeros((2, 4, 4), dtype=np.bool_)
    targets[1, 1:3, 1:3] = True
    predictions = targets.astype(np.float32)
    result = compute_localization_diagnostics(predictions, targets, np.asarray([True, False]))
    assert result["curves"]["0.05"]["aupro"] == pytest.approx(1.0)
    assert result["curves"]["0.30"]["aupro"] == pytest.approx(1.0)


def test_inverted_localization_has_zero_low_fpr_aupro() -> None:
    targets = np.zeros((2, 4, 4), dtype=np.bool_)
    targets[1, 1:3, 1:3] = True
    predictions = (~targets).astype(np.float32)
    result = compute_localization_diagnostics(predictions, targets, np.asarray([True, False]))
    assert result["curves"]["0.30"]["aupro"] == pytest.approx(0.0)


def test_regions_receive_equal_weight_regardless_of_area() -> None:
    targets = np.zeros((1, 5, 5), dtype=np.bool_)
    targets[0, 0, 0] = True
    targets[0, 2:4, 2:4] = True
    predictions = np.zeros_like(targets, dtype=np.float32)
    predictions[0, 0, 0] = 1.0
    result = compute_localization_diagnostics(
        predictions, targets, np.asarray([False]), fpr_limits=(0.30,)
    )
    assert result["curves"]["0.30"]["pro_at_fpr_0_00"] == pytest.approx(0.5)


def test_fixed_relative_area_slices_use_declared_boundaries() -> None:
    targets = np.zeros((3, 100, 100), dtype=np.bool_)
    targets[0, 0:2, 0:5] = True  # 10 pixels = 0.1%, inclusive tiny boundary.
    targets[1, 0:5, 0:5] = True  # 25 pixels = 0.25%, small.
    targets[2, 0:11, 0:10] = True  # 110 pixels = 1.1%, large.
    predictions = targets.astype(np.float32)
    result = compute_localization_diagnostics(
        predictions, targets, np.asarray([False, False, False])
    )
    slices = result["fixed_relative_area_slices"]
    assert slices["tiny_le_0_001"]["region_count"] == 1
    assert slices["small_0_001_0_01"]["region_count"] == 1
    assert slices["large_gt_0_01"]["region_count"] == 1
    assert slices["small_0_001_0_01"]["aupro_at_0.05"] == pytest.approx(1.0)


def test_empty_fixed_area_slice_is_explicit() -> None:
    targets = np.zeros((1, 10, 10), dtype=np.bool_)
    targets[0, 0:2, 0:2] = True
    result = compute_localization_diagnostics(
        targets.astype(np.float32), targets, np.asarray([False])
    )
    assert result["fixed_relative_area_slices"]["tiny_le_0_001"] == {"region_count": 0}


def test_operating_point_does_not_split_tied_score_group() -> None:
    targets = np.zeros((2, 3, 3), dtype=np.bool_)
    targets[1, 1, 1] = True
    predictions = np.zeros_like(targets, dtype=np.float32)
    predictions[0, 0, 0:2] = 1.0
    predictions[1, 1, 1] = 0.5
    result = compute_localization_diagnostics(predictions, targets, np.asarray([True, False]))
    operating_point = result["test_derived_operating_points_do_not_use_for_model_selection"][
        "fpr_0_05"
    ]
    assert operating_point["actual_fpr"] == 0.0
    assert operating_point["score_threshold_test_derived"] > 1.0


def test_invalid_inputs_are_rejected() -> None:
    targets = np.zeros((1, 2, 2), dtype=np.bool_)
    with pytest.raises(ValueError, match="ground-truth anomaly region"):
        compute_localization_diagnostics(
            np.zeros_like(targets, dtype=np.float32), targets, np.asarray([True])
        )
