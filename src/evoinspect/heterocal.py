from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .evaluation import binary_metrics

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ResidualHead:
    median: FloatArray
    scale: FloatArray
    bias: float
    weights: FloatArray
    delta: float

    def score(self, features: FloatArray) -> FloatArray:
        values = np.atleast_2d(features).astype(np.float64)
        normalized = (values - self.median) / self.scale
        residual = np.tanh(self.bias + normalized[:, 1:] @ self.weights)
        return normalized[:, 0] + self.delta * residual


def robust_parameters(normal_features: FloatArray, floor: float) -> tuple[FloatArray, FloatArray]:
    values = np.asarray(normal_features, dtype=np.float64)
    median = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, floor)
    return median, scale


def fit_nonnegative_residual(
    features: FloatArray,
    labels: NDArray[np.int64],
    normal_features: FloatArray,
    *,
    delta: float,
    l2: float,
    learning_rate: float,
    iterations: int,
    scale_floor: float,
) -> ResidualHead:
    median, scale = robust_parameters(normal_features, scale_floor)
    normalized = (np.asarray(features, dtype=np.float64) - median) / scale
    base = normalized[:, 0]
    auxiliary = normalized[:, 1:]
    targets = np.asarray(labels, dtype=np.float64)
    weights = np.zeros(auxiliary.shape[1], dtype=np.float64)
    bias = 0.0
    positive = max(1, int(np.sum(targets == 1)))
    negative = max(1, int(np.sum(targets == 0)))
    sample_weights = np.where(
        targets == 1, len(targets) / (2 * positive), len(targets) / (2 * negative)
    )
    for _ in range(iterations):
        activation = bias + auxiliary @ weights
        tanh_value = np.tanh(activation)
        logits = np.clip(base + delta * tanh_value, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        common = sample_weights * (probability - targets) * delta * (1.0 - tanh_value**2)
        gradient_weights = auxiliary.T @ common / len(targets) + 2.0 * l2 * weights
        gradient_bias = float(np.mean(common))
        weights = np.maximum(0.0, weights - learning_rate * gradient_weights)
        bias -= learning_rate * gradient_bias
    return ResidualHead(median=median, scale=scale, bias=bias, weights=weights, delta=delta)


def choose_threshold(scores: FloatArray, labels: NDArray[np.int64]) -> float:
    candidates = np.unique(np.asarray(scores, dtype=np.float64))
    candidates = np.concatenate(([np.nextafter(candidates[0], -np.inf)], candidates))
    best = (-1.0, float(candidates[0]))
    for threshold in candidates:
        predictions = (scores >= threshold).astype(np.int64)
        tp = int(np.sum((labels == 1) & (predictions == 1)))
        fp = int(np.sum((labels == 0) & (predictions == 1)))
        fn = int(np.sum((labels == 1) & (predictions == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, float(threshold))
        if candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] > best[1]):
            best = candidate
    return best[1]


def evaluate_scores(
    labels: NDArray[np.int64], scores: FloatArray, threshold: float
) -> dict[str, float]:
    return binary_metrics(
        labels.tolist(), scores.tolist(), (scores >= threshold).astype(int).tolist()
    )


def leave_one_defect_type_out(
    features: FloatArray,
    labels: NDArray[np.int64],
    defect_types: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    anomaly_types = sorted(
        {kind for label, kind in zip(labels, defect_types, strict=True) if label == 1}
    )
    minimum = int(config["selection"]["minimum_defect_types"])
    if len(anomaly_types) < minimum:
        return {"accepted": False, "reason": "insufficient_defect_types", "folds": []}
    normal_indices = np.flatnonzero(labels == 0)
    normal_fit = normal_indices[::2]
    normal_eval = normal_indices[1::2]
    folds: list[dict[str, Any]] = []
    head_config = config["calibrator"]
    for held_type in anomaly_types:
        fit_anomaly = np.asarray(
            [
                i
                for i, (label, kind) in enumerate(zip(labels, defect_types, strict=True))
                if label == 1 and kind != held_type
            ],
            dtype=np.int64,
        )
        eval_anomaly = np.asarray(
            [
                i
                for i, (label, kind) in enumerate(zip(labels, defect_types, strict=True))
                if label == 1 and kind == held_type
            ],
            dtype=np.int64,
        )
        fit_indices = np.concatenate((normal_fit, fit_anomaly))
        eval_indices = np.concatenate((normal_eval, eval_anomaly))
        head = fit_nonnegative_residual(
            features[fit_indices],
            labels[fit_indices],
            features[normal_fit],
            delta=float(head_config["residual_delta"]),
            l2=float(head_config["l2"]),
            learning_rate=float(head_config["learning_rate"]),
            iterations=int(head_config["iterations"]),
            scale_floor=float(head_config["normal_scale_floor"]),
        )
        calibrated_fit = head.score(features[fit_indices])
        calibrated_eval = head.score(features[eval_indices])
        base_fit = features[fit_indices, 0]
        base_eval = features[eval_indices, 0]
        calibrated_threshold = choose_threshold(calibrated_fit, labels[fit_indices])
        base_threshold = choose_threshold(base_fit, labels[fit_indices])
        calibrated_metrics = evaluate_scores(
            labels[eval_indices], calibrated_eval, calibrated_threshold
        )
        base_metrics = evaluate_scores(labels[eval_indices], base_eval, base_threshold)
        eval_normal = labels[eval_indices] == 0
        calibrated_fpr = float(np.mean(calibrated_eval[eval_normal] >= calibrated_threshold))
        base_fpr = float(np.mean(base_eval[eval_normal] >= base_threshold))
        folds.append(
            {
                "held_defect_type": held_type,
                "auroc_gain": calibrated_metrics["auroc"] - base_metrics["auroc"],
                "f1_gain": calibrated_metrics["f1_fixed_threshold"]
                - base_metrics["f1_fixed_threshold"],
                "normal_fpr_increase": calibrated_fpr - base_fpr,
            }
        )
    mean_auroc = float(np.mean([fold["auroc_gain"] for fold in folds]))
    mean_f1 = float(np.mean([fold["f1_gain"] for fold in folds]))
    max_fpr = float(np.max([fold["normal_fpr_increase"] for fold in folds]))
    selection = config["selection"]
    accepted = (
        mean_auroc > float(selection["require_mean_heldout_auroc_gain_gt"])
        and mean_f1 > float(selection["require_mean_heldout_f1_gain_gt"])
        and max_fpr <= float(selection["maximum_normal_fpr_increase"])
    )
    return {
        "accepted": accepted,
        "reason": "passed_support_loo" if accepted else "failed_support_loo",
        "mean_heldout_auroc_gain": mean_auroc,
        "mean_heldout_f1_gain": mean_f1,
        "maximum_normal_fpr_increase": max_fpr,
        "folds": folds,
    }
