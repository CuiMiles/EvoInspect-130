"""Deterministic event-level sequence checks for assembly video streams.

The FSM consumes observations produced by an image model or an operator-facing
component detector.  It deliberately does not contain a neural network: its
job is to make missing, repeated, and reordered process steps auditable and
to return frame/time intervals that can be rendered in a video report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

EventKind = Literal[
    "step_completed",
    "missing_step",
    "reordered_step",
    "repeated_step",
    "unexpected_step",
    "anomaly",
]


@dataclass(frozen=True)
class FrameObservation:
    """One frame-level observation from the visual front end."""

    frame_index: int
    timestamp_s: float
    step: str | None = None
    component: str | None = None
    anomaly_score: float = 0.0
    anomaly: bool = False

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if self.step is not None and not self.step.strip():
            raise ValueError("step must be non-empty when provided")
        if self.component is not None and not self.component.strip():
            raise ValueError("component must be non-empty when provided")
        if not math.isfinite(self.anomaly_score) or self.anomaly_score < 0:
            raise ValueError("anomaly_score must be finite and non-negative")


@dataclass(frozen=True)
class SequenceRule:
    """Expected process sequence and deterministic temporal thresholds."""

    expected_steps: tuple[str, ...]
    anomaly_threshold: float = 0.5
    max_gap_s: float = 2.0
    min_persistence: int = 1

    def __post_init__(self) -> None:
        if not self.expected_steps or any(not step.strip() for step in self.expected_steps):
            raise ValueError("expected_steps must contain non-empty names")
        if len(set(self.expected_steps)) != len(self.expected_steps):
            raise ValueError("expected_steps must be unique")
        if not math.isfinite(self.anomaly_threshold) or self.anomaly_threshold < 0:
            raise ValueError("anomaly_threshold must be finite and non-negative")
        if not math.isfinite(self.max_gap_s) or self.max_gap_s <= 0:
            raise ValueError("max_gap_s must be finite and positive")
        if self.min_persistence < 1:
            raise ValueError("min_persistence must be at least one")


@dataclass(frozen=True)
class SequenceEvent:
    """An auditable event interval and explanation."""

    kind: EventKind
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    step: str | None
    confidence: float
    reason: str
    component: str | None = None

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("event frame interval is invalid")
        if self.start_time_s < 0 or self.end_time_s < self.start_time_s:
            raise ValueError("event time interval is invalid")
        if not 0.0 <= self.confidence <= 1.0 or not math.isfinite(self.confidence):
            raise ValueError("event confidence must be finite in [0, 1]")

    def to_dict(self) -> dict[str, int | float | str | None]:
        """Return a JSON-friendly representation for structured output."""

        return {
            "kind": self.kind,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "step": self.step,
            "confidence": self.confidence,
            "reason": self.reason,
            "component": self.component,
        }


class AssemblySequenceFSM:
    """State machine that turns frame observations into process events.

    A later expected step causes explicit ``missing_step`` events for skipped
    steps.  A previously completed step is reported as ``repeated_step`` and
    an unknown step as ``unexpected_step``.  Anomaly frames are coalesced into
    one interval until a normal frame closes the interval.
    """

    def __init__(self, rule: SequenceRule) -> None:
        self.rule = rule
        self.reset()

    def reset(self) -> None:
        self._next_index = 0
        self._last_observation: FrameObservation | None = None
        self._pending_step: str | None = None
        self._pending_count = 0
        self._pending_start: FrameObservation | None = None
        self._reported_missing: set[int] = set()
        self._observed_steps: set[str] = set()
        self._active_anomaly_start: FrameObservation | None = None
        self._active_anomaly_last: FrameObservation | None = None
        self._last_committed_step: str | None = None

    @property
    def next_expected_step(self) -> str | None:
        if self._next_index >= len(self.rule.expected_steps):
            return None
        return self.rule.expected_steps[self._next_index]

    @property
    def completed_steps(self) -> tuple[str, ...]:
        return self.rule.expected_steps[: self._next_index]

    def _event(
        self,
        kind: EventKind,
        start: FrameObservation,
        end: FrameObservation,
        step: str | None,
        confidence: float,
        reason: str,
    ) -> SequenceEvent:
        return SequenceEvent(
            kind=kind,
            start_frame=start.frame_index,
            end_frame=end.frame_index,
            start_time_s=start.timestamp_s,
            end_time_s=end.timestamp_s,
            step=step,
            confidence=max(0.0, min(1.0, float(confidence))),
            reason=reason,
            component=end.component,
        )

    def _close_anomaly(self) -> SequenceEvent | None:
        start = self._active_anomaly_start
        end = self._active_anomaly_last
        if start is None or end is None:
            return None
        confidence = max(start.anomaly_score, end.anomaly_score)
        event = self._event(
            "anomaly",
            start,
            end,
            end.step,
            min(1.0, confidence),
            "consecutive anomaly observations were coalesced into one event interval",
        )
        self._active_anomaly_start = None
        self._active_anomaly_last = None
        return event

    def _anomaly_event(self, observation: FrameObservation) -> SequenceEvent | None:
        is_anomaly = observation.anomaly or observation.anomaly_score >= self.rule.anomaly_threshold
        if is_anomaly:
            if self._active_anomaly_start is None:
                self._active_anomaly_start = observation
            self._active_anomaly_last = observation
            return None
        return self._close_anomaly()

    def _commit_step(self, observation: FrameObservation) -> list[SequenceEvent]:
        step = self._pending_step
        start = self._pending_start or observation
        if step is None:
            return []
        self._pending_step = None
        self._pending_start = None
        self._pending_count = 0
        try:
            index = self.rule.expected_steps.index(step)
        except ValueError:
            return [
                self._event(
                    "unexpected_step",
                    start,
                    observation,
                    step,
                    0.0,
                    f"step {step!r} is not present in the declared process sequence",
                )
            ]

        if index < self._next_index:
            if step in self._observed_steps:
                return [
                    self._event(
                        "repeated_step",
                        start,
                        observation,
                        step,
                        1.0,
                        f"step {step!r} was already observed before the current expected step",
                    )
                ]
            self._observed_steps.add(step)
            return [
                self._event(
                    "reordered_step",
                    start,
                    observation,
                    step,
                    1.0,
                    f"previously skipped step {step!r} was observed late for the first time",
                )
            ]

        events: list[SequenceEvent] = []
        was_later = index > self._next_index
        for missing_index in range(self._next_index, index):
            missing_step = self.rule.expected_steps[missing_index]
            events.append(
                self._event(
                    "missing_step",
                    start,
                    observation,
                    missing_step,
                    1.0,
                    f"step {missing_step!r} was skipped before observed step {step!r}",
                )
            )
        self._next_index = index + 1
        self._last_committed_step = step
        self._observed_steps.add(step)
        self._reported_missing.clear()
        events.append(
            self._event(
                "reordered_step" if was_later else "step_completed",
                start,
                observation,
                step,
                1.0,
                "observed step matched the declared sequence"
                if not was_later
                else "observed a later step after one or more missing steps",
            )
        )
        return events

    def process(self, observation: FrameObservation) -> tuple[SequenceEvent, ...]:
        """Consume one observation and return newly closed events."""

        previous = self._last_observation
        if previous is not None:
            if observation.frame_index <= previous.frame_index:
                raise ValueError("frame_index must increase strictly")
            if observation.timestamp_s < previous.timestamp_s:
                raise ValueError("timestamp_s must be non-decreasing")
        self._last_observation = observation
        events: list[SequenceEvent] = []
        anomaly_event = self._anomaly_event(observation)
        if anomaly_event is not None:
            events.append(anomaly_event)

        expected = self.next_expected_step
        if expected is not None and previous is not None:
            if (
                observation.timestamp_s - previous.timestamp_s > self.rule.max_gap_s
                and self._next_index not in self._reported_missing
            ):
                self._reported_missing.add(self._next_index)
                events.append(
                    self._event(
                        "missing_step",
                        previous,
                        observation,
                        expected,
                        1.0,
                        f"no observation for expected step {expected!r} within max_gap_s",
                    )
                )

        if observation.step is None:
            self._pending_step = None
            self._pending_start = None
            self._pending_count = 0
            return tuple(events)

        if observation.step == self._pending_step:
            self._pending_count += 1
        else:
            self._pending_step = observation.step
            self._pending_count = 1
            self._pending_start = observation
        continuous_frame = (
            previous is not None
            and observation.step == previous.step == self._last_committed_step
        )
        if self._pending_count >= self.rule.min_persistence and not continuous_frame:
            events.extend(self._commit_step(observation))
        elif continuous_frame:
            # Consecutive frames of the same completed step extend its visual
            # persistence; they are not a second process-step event.
            self._pending_step = None
            self._pending_start = None
            self._pending_count = 0
        return tuple(events)

    def finalize(self) -> tuple[SequenceEvent, ...]:
        """Close open anomaly intervals and report an unfinished expected step."""

        events: list[SequenceEvent] = []
        anomaly_event = self._close_anomaly()
        if anomaly_event is not None:
            events.append(anomaly_event)
        last = self._last_observation
        expected = self.next_expected_step
        if (
            last is not None
            and expected is not None
            and self._next_index not in self._reported_missing
        ):
            self._reported_missing.add(self._next_index)
            events.append(
                self._event(
                    "missing_step",
                    last,
                    last,
                    expected,
                    1.0,
                    f"stream ended before expected step {expected!r} was observed",
                )
            )
        return tuple(events)

    def run(
        self, observations: list[FrameObservation] | tuple[FrameObservation, ...]
    ) -> tuple[SequenceEvent, ...]:
        """Process a finite stream and return all closed events."""

        events: list[SequenceEvent] = []
        for observation in observations:
            events.extend(self.process(observation))
        events.extend(self.finalize())
        return tuple(events)
