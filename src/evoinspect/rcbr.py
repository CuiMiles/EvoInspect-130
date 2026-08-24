from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, special  # type: ignore[import-untyped]

Strategy = Literal[
    "uniform_downsample",
    "full_grid",
    "fixed_topk",
    "uncertainty_only",
    "risk_calibrated",
    "full_rcbr",
]


@dataclass(frozen=True)
class Roi:
    """Half-open ROI in anomaly-map coordinates with auditable routing signals."""

    y0: int
    x0: int
    y1: int
    x1: int
    risk: float
    uncertainty: float
    high_frequency: float
    position_rarity: float
    predicted_benefit: float = 0.0
    predicted_cost_ms: float = 0.0
    reason: str = ""

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def area(self) -> int:
        return self.height * self.width

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouterLimits:
    latency_budget_ms: float
    max_rois: int = 4
    max_total_area_fraction: float = 0.35
    nms_iou: float = 0.35


@dataclass(frozen=True)
class UtilityModel:
    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    intercept: float

    def predict(self, features: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        matrix = np.atleast_2d(np.asarray(features, dtype=np.float64))
        normalized = (matrix - self.feature_mean) / self.feature_scale
        logits = np.clip(normalized @ self.coefficients + self.intercept, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))


class NormalRiskCalibrator:
    """Empirical spatial calibration fitted only on held-out normal maps."""

    def __init__(self, sorted_normal_scores: NDArray[np.float32]) -> None:
        if sorted_normal_scores.ndim != 3 or sorted_normal_scores.shape[0] < 2:
            raise ValueError("normal calibration requires at least two N,H,W maps")
        self._scores = sorted_normal_scores

    @classmethod
    def fit(cls, normal_maps: NDArray[np.floating[Any]]) -> NormalRiskCalibrator:
        maps = np.asarray(normal_maps, dtype=np.float32)
        if maps.ndim != 3 or not np.isfinite(maps).all():
            raise ValueError("normal_maps must be finite N,H,W values")
        return cls(np.sort(maps, axis=0))

    @property
    def shape(self) -> tuple[int, int]:
        return self._scores.shape[1], self._scores.shape[2]

    def transform(self, anomaly_map: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
        values = np.asarray(anomaly_map, dtype=np.float32)
        if values.shape != self.shape:
            raise ValueError(f"map shape {values.shape} does not match calibration {self.shape}")
        # A spatial z-score retains continuous ordering beyond the largest calibration sample,
        # unlike a raw empirical CDF with only 20 normals. This is a risk score, not a formal
        # pixel-wise conformal coverage guarantee.
        mean = self._scores.mean(axis=0, dtype=np.float64)
        scale = self._scores.std(axis=0, dtype=np.float64, ddof=1)
        positive_scales = scale[scale > 1e-8]
        fallback = float(np.median(positive_scales)) if len(positive_scales) else 1.0
        scale = np.maximum(scale, max(fallback * 0.05, 1e-8))
        risk = special.ndtr((values - mean) / scale)
        return np.asarray(risk, dtype=np.float32)


def multiscale_disagreement(anomaly_map: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
    values = np.asarray(anomaly_map, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("anomaly_map must be H,W")
    local = ndimage.gaussian_filter(values, sigma=2.0)
    broad = ndimage.gaussian_filter(values, sigma=8.0)
    return np.asarray(np.abs(values - local) + np.abs(local - broad), dtype=np.float32)


def high_frequency_map(image: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim == 3:
        values = values.mean(axis=2)
    if values.ndim != 2:
        raise ValueError("image must be H,W or H,W,C")
    gx = ndimage.sobel(values, axis=1, mode="reflect")
    gy = ndimage.sobel(values, axis=0, mode="reflect")
    magnitude = np.hypot(gx, gy)
    scale = float(np.quantile(magnitude, 0.99))
    return np.asarray(np.clip(magnitude / max(scale, 1e-12), 0.0, 1.0), dtype=np.float32)


def _window_mean(integral: NDArray[np.float64], y0: int, x0: int, y1: int, x1: int) -> float:
    total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    return float(total / ((y1 - y0) * (x1 - x0)))


def generate_candidates(
    risk_map: NDArray[np.floating[Any]],
    uncertainty_map: NDArray[np.floating[Any]],
    frequency_map: NDArray[np.floating[Any]],
    *,
    window_fractions: tuple[float, ...] = (0.125, 0.25),
    stride_fraction: float = 0.5,
    per_scale: int = 8,
) -> list[Roi]:
    """Generate deterministic multi-signal candidates, including sub-threshold texture ROIs."""
    maps = [
        np.asarray(value, dtype=np.float32) for value in (risk_map, uncertainty_map, frequency_map)
    ]
    if maps[0].ndim != 2 or any(value.shape != maps[0].shape for value in maps):
        raise ValueError("candidate maps must share H,W shape")
    height, width = maps[0].shape
    integrals = [np.pad(value.cumsum(0).cumsum(1), ((1, 0), (1, 0))) for value in maps]
    candidates: list[Roi] = []
    for fraction in window_fractions:
        window_h = min(height, max(4, round(height * fraction)))
        window_w = min(width, max(4, round(width * fraction)))
        stride_y = max(1, round(window_h * stride_fraction))
        stride_x = max(1, round(window_w * stride_fraction))
        y_starts = sorted(
            set([*range(0, max(height - window_h + 1, 1), stride_y), height - window_h])
        )
        x_starts = sorted(
            set([*range(0, max(width - window_w + 1, 1), stride_x), width - window_w])
        )
        scale_candidates: list[Roi] = []
        for y0 in y_starts:
            for x0 in x_starts:
                y1, x1 = y0 + window_h, x0 + window_w
                risk = _window_mean(integrals[0], y0, x0, y1, x1)
                uncertainty = _window_mean(integrals[1], y0, x0, y1, x1)
                frequency = _window_mean(integrals[2], y0, x0, y1, x1)
                cy = (y0 + y1) / (2.0 * height)
                cx = (x0 + x1) / (2.0 * width)
                rarity = float(min(1.0, 2.0 * np.hypot(cy - 0.5, cx - 0.5)))
                score = 0.50 * risk + 0.25 * uncertainty + 0.20 * frequency + 0.05 * rarity
                scale_candidates.append(
                    Roi(y0, x0, y1, x1, risk, uncertainty, frequency, rarity, score)
                )
        scale_candidates.sort(key=lambda roi: (-roi.predicted_benefit, roi.area, roi.y0, roi.x0))
        candidates.extend(scale_candidates[:per_scale])
    return nms(candidates, 0.35)


def roi_iou(left: Roi, right: Roi) -> float:
    intersection_h = max(0, min(left.y1, right.y1) - max(left.y0, right.y0))
    intersection_w = max(0, min(left.x1, right.x1) - max(left.x0, right.x0))
    intersection = intersection_h * intersection_w
    union = left.area + right.area - intersection
    return intersection / union if union else 0.0


def nms(candidates: list[Roi], iou_threshold: float) -> list[Roi]:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    ordered = sorted(candidates, key=lambda roi: (-roi.predicted_benefit, roi.area, roi.y0, roi.x0))
    selected: list[Roi] = []
    for candidate in ordered:
        if all(roi_iou(candidate, kept) <= iou_threshold for kept in selected):
            selected.append(candidate)
    return selected


def roi_features(roi: Roi) -> NDArray[np.float64]:
    return np.asarray(
        [roi.risk, roi.uncertainty, roi.high_frequency, roi.position_rarity],
        dtype=np.float64,
    )


def fit_utility_model(
    features: NDArray[np.floating[Any]], labels: NDArray[np.integer[Any]], *, iterations: int = 100
) -> UtilityModel:
    """Fit small deterministic L2 logistic model without an external ML dependency."""
    matrix = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) != len(targets) or len(matrix) < 4:
        raise ValueError("utility training requires aligned 2-D features and >=4 labels")
    if set(np.unique(targets)) != {0.0, 1.0}:
        raise ValueError("utility labels must include both classes")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (matrix - mean) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])
    weights = np.zeros(design.shape[1], dtype=np.float64)
    regularization = np.diag([0.0, *([1e-3] * matrix.shape[1])])
    for _ in range(iterations):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        variance = np.maximum(probabilities * (1.0 - probabilities), 1e-6)
        gradient = design.T @ (probabilities - targets) + regularization @ weights
        hessian = (design.T * variance) @ design + regularization
        update = np.linalg.solve(hessian + np.eye(len(weights)) * 1e-8, gradient)
        weights -= update
        if float(np.max(np.abs(update))) < 1e-8:
            break
    return UtilityModel(mean, scale, weights[1:], float(weights[0]))


def cross_fitted_utility_predictions(
    features: NDArray[np.floating[Any]],
    labels: NDArray[np.integer[Any]],
    folds: NDArray[np.integer[Any]],
) -> tuple[UtilityModel, NDArray[np.float64]]:
    matrix = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    fold_ids = np.asarray(folds, dtype=np.int64)
    if len(matrix) != len(targets) or len(targets) != len(fold_ids):
        raise ValueError("cross-fitting inputs must be aligned")
    predictions = np.zeros(len(targets), dtype=np.float64)
    unique_folds = np.unique(fold_ids)
    if len(unique_folds) != 5:
        raise ValueError("RCBR requires exactly five cross-fitting folds")
    for fold in unique_folds:
        train = fold_ids != fold
        held_out = ~train
        if set(np.unique(targets[train])) != {0, 1}:
            raise ValueError(f"cross-fitting fold {fold} training data lacks a class")
        predictions[held_out] = fit_utility_model(matrix[train], targets[train]).predict(
            matrix[held_out]
        )
    return fit_utility_model(matrix, targets), predictions


def attach_costs_and_utility(
    candidates: list[Roi], utility_model: UtilityModel | None, latency_table: dict[int, float]
) -> list[Roi]:
    if not latency_table or any(size <= 0 or cost <= 0 for size, cost in latency_table.items()):
        raise ValueError("latency_table must contain positive area-pixel and millisecond values")
    sizes = np.asarray(sorted(latency_table), dtype=np.float64)
    costs = np.asarray([latency_table[int(size)] for size in sizes], dtype=np.float64)
    attached: list[Roi] = []
    for roi in candidates:
        benefit = (
            float(utility_model.predict(roi_features(roi))[0])
            if utility_model is not None
            else roi.predicted_benefit
        )
        cost = float(np.interp(roi.area, sizes, costs, left=costs[0], right=costs[-1]))
        attached.append(replace(roi, predicted_benefit=benefit, predicted_cost_ms=cost))
    return attached


def select_under_budget(
    candidates: list[Roi], image_shape: tuple[int, int], limits: RouterLimits
) -> list[Roi]:
    if limits.latency_budget_ms < 0 or limits.max_rois < 0:
        raise ValueError("router budget and max_rois must be non-negative")
    image_area = image_shape[0] * image_shape[1]
    ordered = sorted(
        nms(candidates, limits.nms_iou),
        key=lambda roi: (
            -(roi.predicted_benefit / max(roi.predicted_cost_ms, 1e-9)),
            -roi.predicted_benefit,
            roi.area,
            roi.y0,
            roi.x0,
        ),
    )
    selected: list[Roi] = []
    cost = 0.0
    area = 0
    for roi in ordered:
        if len(selected) >= limits.max_rois:
            break
        next_cost = cost + roi.predicted_cost_ms
        next_area = area + roi.area
        if (
            next_cost <= limits.latency_budget_ms
            and next_area / image_area <= limits.max_total_area_fraction
        ):
            selected.append(roi)
            cost, area = next_cost, next_area
    return selected


def fuse_refinements(
    global_map: NDArray[np.floating[Any]],
    refinements: list[tuple[Roi, NDArray[np.floating[Any]], float]],
    *,
    minimum_evidence: float,
) -> tuple[NDArray[np.float32], list[dict[str, Any]]]:
    """Apply deterministic ROI updates; invalid/weak evidence falls back to global output."""
    output = np.asarray(global_map, dtype=np.float32).copy()
    audit: list[dict[str, Any]] = []
    for roi, refined_map, evidence in refinements:
        local = np.asarray(refined_map, dtype=np.float32)
        applied = bool(
            evidence >= minimum_evidence
            and local.shape == (roi.height, roi.width)
            and np.isfinite(local).all()
        )
        reason = "applied" if applied else "fallback_global"
        if applied:
            # Monotonic max fusion prevents a weak local crop from erasing global evidence.
            output[roi.y0 : roi.y1, roi.x0 : roi.x1] = np.maximum(
                output[roi.y0 : roi.y1, roi.x0 : roi.x1], local
            )
        audit.append({**roi.to_dict(), "evidence": float(evidence), "result": reason})
    return output, audit


def strategy_candidates(strategy: Strategy, candidates: list[Roi]) -> list[Roi]:
    if strategy == "uniform_downsample":
        return []
    if strategy == "full_grid":
        return candidates
    if strategy == "fixed_topk":
        return sorted(candidates, key=lambda roi: (-roi.risk, roi.area))
    if strategy == "uncertainty_only":
        return [replace(roi, predicted_benefit=roi.uncertainty) for roi in candidates]
    if strategy == "risk_calibrated":
        return [replace(roi, predicted_benefit=roi.risk) for roi in candidates]
    if strategy == "full_rcbr":
        return candidates
    raise ValueError(f"unsupported strategy: {strategy}")
