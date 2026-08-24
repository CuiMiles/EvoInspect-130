from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore[import-untyped]


def connected_region_labels(
    targets: NDArray[np.bool_],
) -> tuple[NDArray[np.int32], NDArray[np.int64]]:
    """Label 8-connected target regions with IDs unique across all images."""
    if targets.ndim != 3:
        raise ValueError(f"targets must have N,H,W shape, got {targets.shape}")
    labels = np.zeros(targets.shape, dtype=np.int32)
    next_label = 1
    structure = np.ones((3, 3), dtype=np.uint8)
    for index, target in enumerate(targets):
        image_labels, count = ndimage.label(target, structure=structure)
        if count:
            foreground = image_labels > 0
            labels[index][foreground] = image_labels[foreground] + next_label - 1
            next_label += int(count)
    sizes = np.bincount(labels.ravel(), minlength=next_label).astype(np.int64)
    return labels, sizes


def _clip_curve(
    fpr: NDArray[np.float64],
    pro: NDArray[np.float64],
    fpr_limit: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    inside = np.flatnonzero(fpr <= fpr_limit)
    if not len(inside):
        return np.asarray([0.0, fpr_limit]), np.asarray([0.0, 0.0])
    last = int(inside[-1])
    clipped_fpr = fpr[: last + 1]
    clipped_pro = pro[: last + 1]
    if clipped_fpr[-1] < fpr_limit:
        if last + 1 >= len(fpr):
            boundary_pro = clipped_pro[-1]
        else:
            denominator = fpr[last + 1] - fpr[last]
            fraction = 0.0 if denominator == 0 else (fpr_limit - fpr[last]) / denominator
            boundary_pro = pro[last] + fraction * (pro[last + 1] - pro[last])
        clipped_fpr = np.append(clipped_fpr, fpr_limit)
        clipped_pro = np.append(clipped_pro, boundary_pro)
    return clipped_fpr, clipped_pro


def _curve_from_sorted(
    background_sorted: NDArray[np.bool_],
    region_weights_sorted: NDArray[np.float64],
    scores_sorted: NDArray[np.float32],
    background_count: int,
    fpr_limit: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    fpr = np.cumsum(background_sorted, dtype=np.float64) / background_count
    pro = np.cumsum(region_weights_sorted, dtype=np.float64)
    keep = np.ones(len(scores_sorted), dtype=np.bool_)
    keep[:-1] = scores_sorted[:-1] != scores_sorted[1:]
    fpr = np.concatenate([np.asarray([0.0]), fpr[keep]])
    pro = np.concatenate([np.asarray([0.0]), pro[keep]])
    return _clip_curve(fpr, pro, fpr_limit)


def _aupro_from_curve(
    fpr: NDArray[np.float64], pro: NDArray[np.float64], fpr_limit: float
) -> float:
    return float(np.trapezoid(pro, fpr) / fpr_limit)


def _last_values_for_unique_x(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    keep = np.ones(len(x), dtype=np.bool_)
    keep[:-1] = x[:-1] != x[1:]
    return x[keep], y[keep]


def _operating_point(
    predictions: NDArray[np.float32],
    labels: NDArray[np.int32],
    region_sizes: NDArray[np.int64],
    target_fpr: float,
    normal_image_mask: NDArray[np.bool_],
) -> dict[str, float]:
    background_scores = predictions[labels == 0]
    # Scores tied at a threshold must be admitted as one group. Select the lowest score
    # whose complete tied group stays within the integer false-positive budget. If even
    # the maximum-score group is too large, use the next representable float above it.
    allowed_false_positives = int(np.floor(target_fpr * len(background_scores)))
    unique_scores, counts = np.unique(background_scores, return_counts=True)
    descending_scores = unique_scores[::-1]
    cumulative_counts = np.cumsum(counts[::-1])
    eligible = np.flatnonzero(cumulative_counts <= allowed_false_positives)
    if len(eligible):
        threshold = float(descending_scores[int(eligible[-1])])
    else:
        threshold = float(np.nextafter(descending_scores[0], np.float32(np.inf), dtype=np.float32))
    binary = predictions >= threshold
    actual_fpr = float(np.mean(binary[labels == 0]))
    if actual_fpr > target_fpr:
        raise RuntimeError(
            f"conservative threshold exceeded FPR budget: {actual_fpr} > {target_fpr}"
        )
    region_hits = np.bincount(labels[binary], minlength=len(region_sizes)).astype(np.float64)
    overlaps = np.divide(
        region_hits[1:],
        region_sizes[1:],
        out=np.zeros(len(region_sizes) - 1, dtype=np.float64),
        where=region_sizes[1:] > 0,
    )
    false_positive_regions = 0
    normal_count = int(normal_image_mask.sum())
    structure = np.ones((3, 3), dtype=np.uint8)
    for prediction in binary[normal_image_mask]:
        _, count = ndimage.label(prediction, structure=structure)
        false_positive_regions += int(count)
    return {
        "requested_fpr": target_fpr,
        "actual_fpr": actual_fpr,
        "score_threshold_test_derived": threshold,
        "mean_per_region_overlap": float(np.mean(overlaps)),
        "region_recall_overlap_0_10": float(np.mean(overlaps >= 0.10)),
        "region_recall_overlap_0_30": float(np.mean(overlaps >= 0.30)),
        "region_recall_overlap_0_50": float(np.mean(overlaps >= 0.50)),
        "false_positive_regions_per_normal_image": (
            false_positive_regions / normal_count if normal_count else 0.0
        ),
    }


def compute_localization_diagnostics(
    predictions: NDArray[np.floating[Any]],
    targets: NDArray[np.bool_],
    normal_image_mask: NDArray[np.bool_],
    fpr_limits: Sequence[float] = (0.05, 0.30),
) -> dict[str, Any]:
    """Compute exact AU-PRO curves and region-size localization diagnostics.

    AU-PRO gives every 8-connected ground-truth region equal total weight. The curve is
    integrated against global background FPR and linearly clipped at each requested FPR limit.
    """
    if predictions.shape != targets.shape or targets.ndim != 3:
        raise ValueError("predictions and targets must share N,H,W shape")
    if normal_image_mask.shape != (targets.shape[0],):
        raise ValueError("normal_image_mask must have shape N")
    limits = sorted({float(limit) for limit in fpr_limits})
    if not limits or any(not 0 < limit <= 1 for limit in limits):
        raise ValueError("fpr_limits must be non-empty and in (0, 1]")

    scores = np.asarray(predictions, dtype=np.float32)
    labels, region_sizes = connected_region_labels(targets)
    region_count = len(region_sizes) - 1
    if region_count < 1:
        raise ValueError("at least one ground-truth anomaly region is required")
    flat_scores = scores.ravel()
    flat_labels = labels.ravel()
    background = flat_labels == 0
    background_count = int(background.sum())
    if not background_count:
        raise ValueError("at least one background pixel is required")
    order = np.argsort(flat_scores, kind="stable")[::-1]
    scores_sorted = flat_scores[order]
    background_sorted = background[order]

    weights = np.zeros(len(flat_labels), dtype=np.float64)
    foreground = flat_labels > 0
    weights[foreground] = 1.0 / (
        region_sizes[flat_labels[foreground]].astype(np.float64) * region_count
    )
    weights_sorted = weights[order]

    curves: dict[str, Any] = {}
    for limit in limits:
        fpr, pro = _curve_from_sorted(
            background_sorted, weights_sorted, scores_sorted, background_count, limit
        )
        unique_fpr, unique_pro = _last_values_for_unique_x(fpr, pro)
        curves[f"{limit:.2f}"] = {
            "aupro": _aupro_from_curve(fpr, pro, limit),
            "fpr_limit": limit,
            "curve_points": len(fpr),
            "pro_at_fpr_0_00": float(np.interp(0.00, unique_fpr, unique_pro)),
            "pro_at_fpr_0_01": float(np.interp(0.01, unique_fpr, unique_pro)),
            "pro_at_fpr_0_05": float(np.interp(0.05, unique_fpr, unique_pro)),
            "pro_at_fpr_0_10": float(np.interp(0.10, unique_fpr, unique_pro)),
            "pro_at_fpr_0_30": float(np.interp(0.30, unique_fpr, unique_pro)),
        }

    foreground_sizes = region_sizes[1:]
    lower, upper = np.quantile(foreground_sizes, [0.25, 0.75])
    size_masks = {
        "small_le_q25": foreground_sizes <= lower,
        "medium_q25_q75": (foreground_sizes > lower) & (foreground_sizes <= upper),
        "large_gt_q75": foreground_sizes > upper,
    }

    def build_size_slices(
        masks: dict[str, NDArray[np.bool_]],
    ) -> dict[str, Any]:
        slices: dict[str, Any] = {}
        for name, selected in masks.items():
            selected_ids = np.flatnonzero(selected) + 1
            selected_count = len(selected_ids)
            selected_weights = np.zeros(len(flat_labels), dtype=np.float64)
            selected_foreground = np.isin(flat_labels, selected_ids)
            if selected_count:
                selected_weights[selected_foreground] = 1.0 / (
                    region_sizes[flat_labels[selected_foreground]].astype(np.float64)
                    * selected_count
                )
                entry: dict[str, Any] = {
                    "region_count": selected_count,
                    "area_pixels_min": int(foreground_sizes[selected].min()),
                    "area_pixels_max": int(foreground_sizes[selected].max()),
                }
                for limit in limits:
                    fpr, pro = _curve_from_sorted(
                        background_sorted,
                        selected_weights[order],
                        scores_sorted,
                        background_count,
                        limit,
                    )
                    entry[f"aupro_at_{limit:.2f}"] = _aupro_from_curve(fpr, pro, limit)
                slices[name] = entry
            else:
                slices[name] = {"region_count": 0}
        return slices

    # The relative-area definition is fixed across tasks and datasets. Quantile slices
    # are retained only as a backwards-compatible diagnostic.
    region_image_ids = np.zeros(region_count, dtype=np.int64)
    for region_id in range(1, region_count + 1):
        image_ids = np.flatnonzero(np.any(labels == region_id, axis=(1, 2)))
        if len(image_ids) != 1:
            raise RuntimeError(f"region {region_id} must belong to exactly one image")
        region_image_ids[region_id - 1] = int(image_ids[0])
    image_areas = np.asarray([targets[index].size for index in region_image_ids], dtype=np.float64)
    relative_areas = foreground_sizes.astype(np.float64) / image_areas
    fixed_area_masks = {
        "tiny_le_0_001": relative_areas <= 0.001,
        "small_0_001_0_01": (relative_areas > 0.001) & (relative_areas <= 0.01),
        "large_gt_0_01": relative_areas > 0.01,
    }
    quantile_slices = build_size_slices(size_masks)

    return {
        "schema_version": 2,
        "metric": "equal-region-weighted-aupro",
        "connectivity": 8,
        "region_count": region_count,
        "background_pixels": background_count,
        "anomaly_pixels": int(foreground.sum()),
        "region_area_pixels": {
            "min": int(foreground_sizes.min()),
            "q25": float(lower),
            "median": float(np.median(foreground_sizes)),
            "q75": float(upper),
            "max": int(foreground_sizes.max()),
        },
        "curves": curves,
        "fixed_relative_area_slices": build_size_slices(fixed_area_masks),
        "relative_area_definition": {
            "tiny": "area_fraction <= 0.001",
            "small": "0.001 < area_fraction <= 0.01",
            "large": "area_fraction > 0.01",
        },
        "relative_region_area": {
            "min": float(relative_areas.min()),
            "median": float(np.median(relative_areas)),
            "max": float(relative_areas.max()),
        },
        "quantile_size_slices_supplemental": quantile_slices,
        # Compatibility alias for saved-mask tools written against schema v1.
        "size_slices": quantile_slices,
        "test_derived_operating_points_do_not_use_for_model_selection": {
            "fpr_0_01": _operating_point(scores, labels, region_sizes, 0.01, normal_image_mask),
            "fpr_0_05": _operating_point(scores, labels, region_sizes, 0.05, normal_image_mask),
        },
    }
