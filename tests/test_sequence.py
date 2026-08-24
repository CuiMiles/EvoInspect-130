from __future__ import annotations

import pytest

from evoinspect.sequence import AssemblySequenceFSM, FrameObservation, SequenceRule


def observation(index: int, step: str | None, score: float = 0.0) -> FrameObservation:
    return FrameObservation(index, float(index) * 0.1, step=step, anomaly_score=score)


def test_sequence_emits_completed_steps_and_anomaly_interval() -> None:
    fsm = AssemblySequenceFSM(SequenceRule(("screen", "battery"), anomaly_threshold=0.8))
    events = fsm.run(
        [
            observation(0, "screen"),
            observation(1, "screen", 0.9),
            observation(2, "screen", 0.95),
            observation(3, "battery"),
            observation(4, None),
        ]
    )
    assert [event.kind for event in events] == [
        "step_completed",
        "anomaly",
        "step_completed",
    ]
    anomaly = events[1]
    assert anomaly.start_frame == 1
    assert anomaly.end_frame == 2
    assert anomaly.step == "screen"


def test_later_step_reports_missing_and_reordered() -> None:
    fsm = AssemblySequenceFSM(SequenceRule(("screen", "battery", "cover")))
    events = fsm.run([observation(0, "screen"), observation(1, "cover")])
    assert [event.kind for event in events] == [
        "step_completed",
        "missing_step",
        "reordered_step",
    ]
    assert events[1].step == "battery"
    assert events[-1].step == "cover"


def test_repeated_and_unexpected_steps_are_explicit() -> None:
    fsm = AssemblySequenceFSM(SequenceRule(("screen", "battery")))
    events = fsm.run(
        [
            observation(0, "screen"),
            observation(1, "battery"),
            observation(2, "screen"),
            observation(3, "robot"),
        ]
    )
    assert [event.kind for event in events] == [
        "step_completed",
        "step_completed",
        "repeated_step",
        "unexpected_step",
    ]


def test_stream_gap_and_monotonicity_are_checked() -> None:
    fsm = AssemblySequenceFSM(SequenceRule(("screen", "battery"), max_gap_s=0.2))
    events = fsm.run(
        [
            FrameObservation(0, 0.0, step="screen"),
            FrameObservation(1, 1.0, step=None),
        ]
    )
    assert any(event.kind == "missing_step" and event.step == "battery" for event in events)
    with pytest.raises(ValueError, match="frame_index"):
        fsm.process(FrameObservation(1, 1.1, step=None))
