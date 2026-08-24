from __future__ import annotations

from pathlib import Path

from evoinspect.guarded_adapt import (
    CandidateUpdate,
    GatePolicy,
    GuardedAdaptController,
    ModelVersion,
)


def controller(tmp_path: Path) -> GuardedAdaptController:
    base = tmp_path / "base.bin"
    base.write_bytes(b"base")
    return GuardedAdaptController(
        ModelVersion(
            "base", str(base), "base-hash", "support-hash", None, "active", "2026-01-01T00:00:00Z"
        ),
        policy=GatePolicy(min_feedback_gain=0.02, max_anchor_regression=0.01),
        state_path=tmp_path / "state.json",
    )


def test_immediate_threshold_and_memory_updates_are_reversible(tmp_path: Path) -> None:
    instance = controller(tmp_path)
    threshold = instance.update_threshold("global", 0.7, feedback_data_hash="f1", reason="operator")
    memory = instance.add_memory("roi-1", feedback_data_hash="f2", reason="operator")
    assert instance.thresholds == {"global": 0.7}
    assert instance.memory == ("roi-1",)
    instance.revert_immediate(threshold.update_id)
    instance.revert_immediate(memory.update_id)
    assert instance.thresholds == {}
    assert instance.memory == ()
    assert instance.to_dict()["immediate_updates"][0]["reverted"] is True


def test_candidate_requires_gain_anchor_and_shadow(tmp_path: Path) -> None:
    instance = controller(tmp_path)
    artifact = tmp_path / "candidate.bin"
    artifact.write_bytes(b"candidate")
    candidate = CandidateUpdate(
        "candidate-1",
        "base",
        str(artifact),
        "feedback-hash",
        0.70,
        0.74,
        0.90,
        0.895,
        True,
    )
    decision = instance.publish_candidate(candidate)
    assert decision.accepted is True
    assert instance.active_version.version_id == "candidate-1"
    assert instance.active_version.artifact_sha256

    rejected = CandidateUpdate(
        "candidate-2",
        "candidate-1",
        str(artifact),
        "feedback-hash-2",
        0.74,
        0.741,
        0.895,
        0.80,
        False,
    )
    rejected_decision = instance.publish_candidate(rejected)
    assert rejected_decision.accepted is False
    assert "anchor regression" in " ".join(rejected_decision.reasons)
    assert "shadow validation" in " ".join(rejected_decision.reasons)


def test_model_rollback_persists_state(tmp_path: Path) -> None:
    instance = controller(tmp_path)
    artifact = tmp_path / "candidate.bin"
    artifact.write_bytes(b"candidate")
    instance.publish_candidate(
        CandidateUpdate("candidate-1", "base", str(artifact), "hash", 0.7, 0.75, 0.9, 0.9, True)
    )
    active = instance.rollback_model("base")
    assert active.version_id == "base"
    assert active.status == "active"
    assert instance.state_path is not None and instance.state_path.is_file()
