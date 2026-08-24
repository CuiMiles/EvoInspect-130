from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .evaluation import binary_metrics


def normalized_score(score: float, threshold: float) -> float:
    return score / max(abs(threshold), 1e-12)


def apply_guarded_fusion(
    patch_score: float,
    supervised_score: float,
    patch_threshold: float,
    supervised_threshold: float,
    strategy: str,
) -> tuple[float, int]:
    patch_normalized = normalized_score(patch_score, patch_threshold)
    supervised_normalized = normalized_score(supervised_score, supervised_threshold)
    if strategy == "patchcore_only":
        value = patch_normalized
    elif strategy == "guarded_or":
        value = max(patch_normalized, supervised_normalized)
    else:
        raise ValueError(f"unsupported guarded fusion strategy: {strategy}")
    return value, int(value >= 1.0)


def calibrate_guarded_fusion(
    patch_scores: Sequence[float],
    supervised_scores: Sequence[float],
    labels: Sequence[int],
    patch_threshold: float,
    supervised_threshold: float,
    min_development_f1_gain: float,
    min_development_anomalies: int = 1,
) -> dict[str, Any]:
    if not (
        len(patch_scores) == len(supervised_scores) == len(labels) and set(labels) == {0, 1}
    ):
        raise ValueError("fusion calibration inputs must be aligned and contain both labels")
    if min_development_f1_gain < 0:
        raise ValueError("minimum development gain must be non-negative")
    if min_development_anomalies < 1:
        raise ValueError("minimum development anomalies must be positive")

    metrics: dict[str, dict[str, float]] = {}
    for strategy in ("patchcore_only", "guarded_or"):
        values_and_decisions = [
            apply_guarded_fusion(
                patch_score,
                supervised_score,
                patch_threshold,
                supervised_threshold,
                strategy,
            )
            for patch_score, supervised_score in zip(
                patch_scores, supervised_scores, strict=True
            )
        ]
        values = [value for value, _ in values_and_decisions]
        decisions = [decision for _, decision in values_and_decisions]
        metrics[strategy] = binary_metrics(labels, values, decisions)

    patch_f1 = metrics["patchcore_only"]["f1_fixed_threshold"]
    or_f1 = metrics["guarded_or"]["f1_fixed_threshold"]
    enough_anomalies = sum(labels) >= min_development_anomalies
    strategy = (
        "guarded_or"
        if enough_anomalies and or_f1 > patch_f1 + min_development_f1_gain
        else "patchcore_only"
    )
    return {
        "strategy": strategy,
        "threshold": 1.0,
        "min_development_f1_gain": min_development_f1_gain,
        "min_development_anomalies": min_development_anomalies,
        "development_anomalies": sum(labels),
        "fallback_reason": (
            "insufficient_development_anomalies"
            if not enough_anomalies
            else "insufficient_f1_gain"
            if strategy == "patchcore_only"
            else ""
        ),
        "development_metrics": metrics,
    }


def apply_selective_rescue(
    patch_score: float,
    supervised_score: float,
    patch_threshold: float,
    supervised_threshold: float,
    min_patch_ratio: float,
    strategy: str,
) -> tuple[float, int]:
    """Fuse a supervised score without overriding distant PatchCore negatives.

    The rescue score reaches one only when both the PatchCore evidence reaches
    ``min_patch_ratio`` and the supervised evidence reaches its threshold. This makes the
    development-selected rescue reversible: ``patchcore_only`` is an exact fallback.
    """
    if not 0 < min_patch_ratio <= 1:
        raise ValueError("minimum PatchCore ratio must be in (0, 1]")
    patch_normalized = normalized_score(patch_score, patch_threshold)
    supervised_normalized = normalized_score(supervised_score, supervised_threshold)
    if strategy == "patchcore_only":
        value = patch_normalized
    elif strategy == "selective_rescue":
        rescue_value = min(patch_normalized / min_patch_ratio, supervised_normalized)
        value = max(patch_normalized, rescue_value)
    else:
        raise ValueError(f"unsupported selective rescue strategy: {strategy}")
    return value, int(value >= 1.0)


def calibrate_selective_rescue(
    patch_scores: Sequence[float],
    supervised_scores: Sequence[float],
    labels: Sequence[int],
    patch_threshold: float,
    supervised_threshold: float,
    min_patch_ratio_candidates: Sequence[float],
    min_development_f1_gain: float,
    max_development_precision_drop: float,
    min_development_anomalies: int,
) -> dict[str, Any]:
    """Select a conservative rescue band using development data only."""
    if not (
        len(patch_scores) == len(supervised_scores) == len(labels) and set(labels) == {0, 1}
    ):
        raise ValueError("selective rescue inputs must be aligned and contain both labels")
    if min_development_f1_gain < 0 or max_development_precision_drop < 0:
        raise ValueError("development constraints must be non-negative")
    if min_development_anomalies < 1:
        raise ValueError("minimum development anomalies must be positive")
    candidates = sorted({float(value) for value in min_patch_ratio_candidates})
    if not candidates or any(not 0 < value <= 1 for value in candidates):
        raise ValueError("rescue ratio candidates must be non-empty and in (0, 1]")

    patch_values_and_decisions = [
        apply_selective_rescue(
            patch_score,
            supervised_score,
            patch_threshold,
            supervised_threshold,
            1.0,
            "patchcore_only",
        )
        for patch_score, supervised_score in zip(
            patch_scores, supervised_scores, strict=True
        )
    ]
    patch_metrics = binary_metrics(
        labels,
        [value for value, _ in patch_values_and_decisions],
        [decision for _, decision in patch_values_and_decisions],
    )
    candidate_metrics: dict[str, dict[str, float]] = {}
    eligible: list[tuple[float, float, float]] = []
    enough_anomalies = sum(labels) >= min_development_anomalies
    for ratio in candidates:
        values_and_decisions = [
            apply_selective_rescue(
                patch_score,
                supervised_score,
                patch_threshold,
                supervised_threshold,
                ratio,
                "selective_rescue",
            )
            for patch_score, supervised_score in zip(
                patch_scores, supervised_scores, strict=True
            )
        ]
        metrics = binary_metrics(
            labels,
            [value for value, _ in values_and_decisions],
            [decision for _, decision in values_and_decisions],
        )
        candidate_metrics[f"{ratio:.6g}"] = metrics
        if (
            enough_anomalies
            and metrics["f1_fixed_threshold"]
            > patch_metrics["f1_fixed_threshold"] + min_development_f1_gain
            and metrics["precision"]
            >= patch_metrics["precision"] - max_development_precision_drop
        ):
            eligible.append((metrics["f1_fixed_threshold"], metrics["precision"], ratio))

    if eligible:
        _, _, selected_ratio = max(eligible)
        strategy = "selective_rescue"
        fallback_reason = ""
    else:
        selected_ratio = max(candidates)
        strategy = "patchcore_only"
        fallback_reason = (
            "insufficient_development_anomalies"
            if not enough_anomalies
            else "development_constraints_not_met"
        )
    return {
        "strategy": strategy,
        "threshold": 1.0,
        "selected_min_patch_ratio": selected_ratio,
        "min_patch_ratio_candidates": candidates,
        "min_development_f1_gain": min_development_f1_gain,
        "max_development_precision_drop": max_development_precision_drop,
        "min_development_anomalies": min_development_anomalies,
        "development_anomalies": sum(labels),
        "fallback_reason": fallback_reason,
        "development_metrics": {
            "patchcore_only": patch_metrics,
            "selective_rescue_candidates": candidate_metrics,
        },
    }
