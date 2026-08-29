"""Risk-controlled primitives for GuardedAdapt-Risk.

The functions in this module are independent of a particular anomaly detector.
They implement the frozen drift trigger, bounded memory replacement, paired
bootstrap risk gate, and exact decision-level rollback used by the replay
benchmark.  Model-specific feature extraction remains in the experiment runner.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ks_2samp  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _as_finite_1d(values: list[float] | FloatArray, name: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    return result


@dataclass(frozen=True)
class DriftPolicy:
    window_size: int = 32
    p_value_max: float = 0.01
    median_shift_iqr_fraction: float = 0.25
    consecutive_windows: int = 2

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least two")
        if not 0.0 < self.p_value_max < 1.0:
            raise ValueError("p_value_max must be in (0, 1)")
        if self.median_shift_iqr_fraction < 0.0:
            raise ValueError("median_shift_iqr_fraction must be non-negative")
        if self.consecutive_windows < 1:
            raise ValueError("consecutive_windows must be positive")


@dataclass(frozen=True)
class DriftWindowResult:
    p_value: float
    median_shift: float
    required_shift: float
    shifted: bool
    consecutive_shifted: int
    triggered: bool


class ConsecutiveDriftDetector:
    """Two-sample KS and robust location-shift drift trigger."""

    def __init__(self, reference_scores: list[float] | FloatArray, policy: DriftPolicy) -> None:
        self.reference = _as_finite_1d(reference_scores, "reference_scores")
        if len(self.reference) < policy.window_size:
            raise ValueError("reference_scores must contain at least one full window")
        self.policy = policy
        self._consecutive = 0

    def update(self, window_scores: list[float] | FloatArray) -> DriftWindowResult:
        values = _as_finite_1d(window_scores, "window_scores")
        if len(values) != self.policy.window_size:
            raise ValueError(f"window_scores must contain {self.policy.window_size} values")
        p_value = float(ks_2samp(self.reference, values, alternative="two-sided").pvalue)
        median_shift = abs(float(np.median(values) - np.median(self.reference)))
        reference_iqr = float(
            np.quantile(self.reference, 0.75) - np.quantile(self.reference, 0.25)
        )
        required_shift = self.policy.median_shift_iqr_fraction * max(reference_iqr, 1e-12)
        shifted = p_value < self.policy.p_value_max and median_shift > required_shift
        self._consecutive = self._consecutive + 1 if shifted else 0
        return DriftWindowResult(
            p_value=p_value,
            median_shift=median_shift,
            required_shift=required_shift,
            shifted=shifted,
            consecutive_shifted=self._consecutive,
            triggered=self._consecutive >= self.policy.consecutive_windows,
        )


@dataclass(frozen=True)
class RiskBudget:
    target_f1_gain_lcb_min: float = 0.0
    historical_normal_fpr_ucb_max: float = 0.01
    seen_anomaly_fnr_ucb_max: float = 0.01
    unseen_anomaly_fnr_ucb_max: float = 0.02
    confidence: float = 0.95
    bootstrap_draws: int = 2000
    multiplicity: int = 4

    def __post_init__(self) -> None:
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0.5, 1)")
        if self.bootstrap_draws < 100:
            raise ValueError("bootstrap_draws must be at least 100")
        if self.multiplicity < 1:
            raise ValueError("multiplicity must be positive")
        for name in (
            "historical_normal_fpr_ucb_max",
            "seen_anomaly_fnr_ucb_max",
            "unseen_anomaly_fnr_ucb_max",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class PairedDecisions:
    labels: IntArray
    champion: IntArray
    candidate: IntArray

    @classmethod
    def from_values(
        cls, labels: list[int] | IntArray, champion: list[int] | IntArray,
        candidate: list[int] | IntArray,
    ) -> PairedDecisions:
        arrays = tuple(np.asarray(value, dtype=np.int64) for value in (labels, champion, candidate))
        if any(array.ndim != 1 for array in arrays) or len({len(array) for array in arrays}) != 1:
            raise ValueError("paired decision arrays must be aligned and one-dimensional")
        if len(arrays[0]) == 0:
            raise ValueError("paired decision arrays must be non-empty")
        if any(not set(array.tolist()) <= {0, 1} for array in arrays):
            raise ValueError("labels and decisions must be binary")
        return cls(*arrays)


@dataclass(frozen=True)
class RiskGateResult:
    accepted: bool
    target_f1_gain: float
    target_f1_gain_lcb: float
    historical_normal_fpr_delta: float
    historical_normal_fpr_delta_ucb: float
    seen_anomaly_fnr_delta: float
    seen_anomaly_fnr_delta_ucb: float
    unseen_anomaly_fnr_delta: float
    unseen_anomaly_fnr_delta_ucb: float
    corrected_one_sided_alpha: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def _f1(labels: IntArray, decisions: IntArray) -> float:
    true_positive = int(np.sum((labels == 1) & (decisions == 1)))
    false_positive = int(np.sum((labels == 0) & (decisions == 1)))
    false_negative = int(np.sum((labels == 1) & (decisions == 0)))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else float(2 * true_positive / denominator)


def _rate(decisions: IntArray, positive_value: int) -> float:
    return float(np.mean(decisions == positive_value))


def paired_bootstrap_risk_gate(
    *,
    target: PairedDecisions,
    historical_normal: PairedDecisions,
    seen_anomaly: PairedDecisions,
    unseen_anomaly: PairedDecisions,
    budget: RiskBudget,
    seed: int,
) -> RiskGateResult:
    """Apply a family-wise corrected one-sided paired bootstrap gate."""

    if set(historical_normal.labels.tolist()) != {0}:
        raise ValueError("historical_normal must contain only normal labels")
    if set(seen_anomaly.labels.tolist()) != {1}:
        raise ValueError("seen_anomaly must contain only anomaly labels")
    if set(unseen_anomaly.labels.tolist()) != {1}:
        raise ValueError("unseen_anomaly must contain only anomaly labels")
    if set(target.labels.tolist()) != {0, 1}:
        raise ValueError("target must contain both labels")

    generator = np.random.default_rng(seed)
    draws = budget.bootstrap_draws
    target_delta = np.empty(draws, dtype=np.float64)
    normal_delta = np.empty(draws, dtype=np.float64)
    seen_delta = np.empty(draws, dtype=np.float64)
    unseen_delta = np.empty(draws, dtype=np.float64)

    groups = (target, historical_normal, seen_anomaly, unseen_anomaly)
    for draw in range(draws):
        sampled = [
            generator.integers(0, len(group.labels), size=len(group.labels)) for group in groups
        ]
        target_index, normal_index, seen_index, unseen_index = sampled
        target_delta[draw] = _f1(
            target.labels[target_index], target.candidate[target_index]
        ) - _f1(target.labels[target_index], target.champion[target_index])
        normal_delta[draw] = _rate(
            historical_normal.candidate[normal_index], 1
        ) - _rate(historical_normal.champion[normal_index], 1)
        seen_delta[draw] = _rate(
            seen_anomaly.candidate[seen_index], 0
        ) - _rate(seen_anomaly.champion[seen_index], 0)
        unseen_delta[draw] = _rate(
            unseen_anomaly.candidate[unseen_index], 0
        ) - _rate(unseen_anomaly.champion[unseen_index], 0)

    alpha = (1.0 - budget.confidence) / budget.multiplicity
    lcb = float(np.quantile(target_delta, alpha))
    normal_ucb = float(np.quantile(normal_delta, 1.0 - alpha))
    seen_ucb = float(np.quantile(seen_delta, 1.0 - alpha))
    unseen_ucb = float(np.quantile(unseen_delta, 1.0 - alpha))
    target_point = _f1(target.labels, target.candidate) - _f1(target.labels, target.champion)
    normal_point = _rate(historical_normal.candidate, 1) - _rate(
        historical_normal.champion, 1
    )
    seen_point = _rate(seen_anomaly.candidate, 0) - _rate(seen_anomaly.champion, 0)
    unseen_point = _rate(unseen_anomaly.candidate, 0) - _rate(
        unseen_anomaly.champion, 0
    )
    reasons: list[str] = []
    if lcb <= budget.target_f1_gain_lcb_min:
        reasons.append("target F1 gain lower bound did not exceed zero")
    if normal_ucb > budget.historical_normal_fpr_ucb_max:
        reasons.append("historical normal FPR risk budget exceeded")
    if seen_ucb > budget.seen_anomaly_fnr_ucb_max:
        reasons.append("seen anomaly FNR risk budget exceeded")
    if unseen_ucb > budget.unseen_anomaly_fnr_ucb_max:
        reasons.append("unseen anomaly FNR risk budget exceeded")
    return RiskGateResult(
        accepted=not reasons,
        target_f1_gain=target_point,
        target_f1_gain_lcb=lcb,
        historical_normal_fpr_delta=normal_point,
        historical_normal_fpr_delta_ucb=normal_ucb,
        seen_anomaly_fnr_delta=seen_point,
        seen_anomaly_fnr_delta_ucb=seen_ucb,
        unseen_anomaly_fnr_delta=unseen_point,
        unseen_anomaly_fnr_delta_ucb=unseen_ucb,
        corrected_one_sided_alpha=alpha,
        reasons=tuple(reasons),
    )


def bounded_memory_replace(
    original: NDArray[np.floating[Any]],
    proposed: NDArray[np.floating[Any]],
    *,
    nearest_original_distance: list[float] | FloatArray,
    original_redundancy_distance: list[float] | FloatArray,
    max_replace_fraction: float,
    max_new_features: int,
    candidate_pool_multiplier: int = 4,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Replace redundant original features with diverse far-from-memory features."""

    base = np.asarray(original, dtype=np.float32)
    additions = np.asarray(proposed, dtype=np.float32)
    if base.ndim != 2 or additions.ndim != 2 or base.shape[1] != additions.shape[1]:
        raise ValueError("original and proposed must be aligned two-dimensional feature arrays")
    if not len(base) or not len(additions):
        raise ValueError("original and proposed features must be non-empty")
    if not 0.0 < max_replace_fraction <= 0.05:
        raise ValueError("max_replace_fraction must be in (0, 0.05]")
    if max_new_features < 1 or candidate_pool_multiplier < 1:
        raise ValueError("feature limits must be positive")
    new_distance = _as_finite_1d(nearest_original_distance, "nearest_original_distance")
    redundancy = _as_finite_1d(original_redundancy_distance, "original_redundancy_distance")
    if len(new_distance) != len(additions) or len(redundancy) != len(base):
        raise ValueError("distance arrays do not match feature arrays")

    limit = min(max_new_features, max(1, int(len(base) * max_replace_fraction)))
    count = min(limit, len(additions))
    pool_count = min(len(additions), max(count, count * candidate_pool_multiplier))
    pool_indices = np.argsort(new_distance, kind="stable")[-pool_count:]
    pool = additions[pool_indices]
    first = int(np.argmax(new_distance[pool_indices]))
    selected_local = [first]
    minimum_selected_distance = np.sum((pool - pool[first]) ** 2, axis=1)
    minimum_selected_distance[first] = -np.inf
    while len(selected_local) < count:
        index = int(np.argmax(minimum_selected_distance))
        selected_local.append(index)
        distance = np.sum((pool - pool[index]) ** 2, axis=1)
        minimum_selected_distance = np.minimum(minimum_selected_distance, distance)
        minimum_selected_distance[selected_local] = -np.inf
    selected_indices = pool_indices[np.asarray(selected_local, dtype=np.int64)]
    removal_indices = np.argsort(redundancy, kind="stable")[:count]
    candidate = base.copy()
    candidate[removal_indices] = additions[selected_indices]
    digest = hashlib.sha256(np.ascontiguousarray(candidate).tobytes()).hexdigest()
    return candidate, {
        "original_features": len(base),
        "proposed_features": len(additions),
        "replaced_features": count,
        "replacement_fraction": float(count / len(base)),
        "selected_proposed_indices": selected_indices.tolist(),
        "removed_original_indices": removal_indices.tolist(),
        "candidate_features_sha256": digest,
    }


def exact_rollback(champion: IntArray, restored: IntArray) -> bool:
    champion_array = np.asarray(champion, dtype=np.int64)
    restored_array = np.asarray(restored, dtype=np.int64)
    return champion_array.shape == restored_array.shape and bool(
        np.array_equal(champion_array, restored_array)
    )
