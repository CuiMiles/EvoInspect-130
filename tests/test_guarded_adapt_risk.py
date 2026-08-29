from __future__ import annotations

import numpy as np

from evoinspect.guarded_adapt_risk import (
    ConsecutiveDriftDetector,
    DriftPolicy,
    PairedDecisions,
    RiskBudget,
    bounded_memory_replace,
    exact_rollback,
    paired_bootstrap_risk_gate,
)


def test_drift_requires_two_ks_and_median_shift_windows() -> None:
    reference = np.linspace(0.0, 1.0, 32)
    detector = ConsecutiveDriftDetector(
        reference,
        DriftPolicy(p_value_max=0.01, median_shift_iqr_fraction=0.25),
    )
    first = detector.update(reference + 2.0)
    second = detector.update(reference + 2.0)
    assert first.shifted is True and first.triggered is False
    assert second.shifted is True and second.triggered is True
    reset = detector.update(reference)
    assert reset.shifted is False and reset.consecutive_shifted == 0


def test_paired_bootstrap_gate_accepts_gain_without_group_regression() -> None:
    target = PairedDecisions.from_values(
        [0] * 20 + [1] * 20,
        [0] * 20 + [0] * 20,
        [0] * 20 + [1] * 20,
    )
    normal = PairedDecisions.from_values([0] * 20, [0] * 20, [0] * 20)
    seen = PairedDecisions.from_values([1] * 20, [1] * 20, [1] * 20)
    unseen = PairedDecisions.from_values([1] * 20, [1] * 20, [1] * 20)
    result = paired_bootstrap_risk_gate(
        target=target,
        historical_normal=normal,
        seen_anomaly=seen,
        unseen_anomaly=unseen,
        budget=RiskBudget(bootstrap_draws=500),
        seed=130,
    )
    assert result.accepted is True
    assert result.target_f1_gain_lcb > 0.0


def test_paired_bootstrap_gate_rejects_unseen_harm() -> None:
    target = PairedDecisions.from_values(
        [0] * 20 + [1] * 20,
        [0] * 20 + [0] * 20,
        [0] * 20 + [1] * 20,
    )
    normal = PairedDecisions.from_values([0] * 20, [0] * 20, [0] * 20)
    seen = PairedDecisions.from_values([1] * 20, [1] * 20, [1] * 20)
    unseen = PairedDecisions.from_values([1] * 20, [1] * 20, [0] * 20)
    result = paired_bootstrap_risk_gate(
        target=target,
        historical_normal=normal,
        seen_anomaly=seen,
        unseen_anomaly=unseen,
        budget=RiskBudget(bootstrap_draws=500),
        seed=131,
    )
    assert result.accepted is False
    assert "unseen anomaly" in " ".join(result.reasons)


def test_memory_replacement_is_bounded_and_changes_real_features() -> None:
    original = np.arange(200 * 4, dtype=np.float32).reshape(200, 4)
    proposed = np.arange(40 * 4, dtype=np.float32).reshape(40, 4) + 10_000
    candidate, audit = bounded_memory_replace(
        original,
        proposed,
        nearest_original_distance=np.linspace(1, 40, 40),
        original_redundancy_distance=np.linspace(0, 1, 200),
        max_replace_fraction=0.05,
        max_new_features=256,
    )
    assert candidate.shape == original.shape
    assert audit["replaced_features"] == 10
    assert audit["replacement_fraction"] == 0.05
    assert np.count_nonzero(np.any(candidate != original, axis=1)) == 10


def test_exact_rollback_requires_identical_decisions() -> None:
    assert exact_rollback(np.asarray([0, 1]), np.asarray([0, 1])) is True
    assert exact_rollback(np.asarray([0, 1]), np.asarray([1, 0])) is False
