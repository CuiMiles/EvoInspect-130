#!/usr/bin/env python3
"""Run a deterministic, synthetic event-level sequence FSM smoke evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from evoinspect.sequence import (
    AssemblySequenceFSM,
    FrameObservation,
    SequenceEvent,
    SequenceRule,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    observations: tuple[FrameObservation, ...]
    required_event_kinds: frozenset[str]


def _obs(index: int, step: str | None, score: float = 0.0) -> FrameObservation:
    return FrameObservation(index, index / 10.0, step=step, anomaly_score=score)


def scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario(
            "normal",
            (_obs(0, "screen"), _obs(1, "battery"), _obs(2, "cover")),
            frozenset(),
        ),
        Scenario(
            "missing_battery",
            (_obs(0, "screen"), _obs(1, "cover")),
            frozenset({"missing_step", "reordered_step"}),
        ),
        Scenario(
            "reordered_battery_first",
            (_obs(0, "battery"), _obs(1, "screen"), _obs(2, "cover")),
            frozenset({"missing_step", "reordered_step", "repeated_step"}),
        ),
        Scenario(
            "repeated_screen",
            (_obs(0, "screen"), _obs(1, "battery"), _obs(2, "screen"), _obs(3, "cover")),
            frozenset({"repeated_step"}),
        ),
        Scenario(
            "unexpected_robot_step",
            (_obs(0, "screen"), _obs(1, "robot"), _obs(2, "battery"), _obs(3, "cover")),
            frozenset({"unexpected_step"}),
        ),
        Scenario(
            "anomaly_burst",
            (
                _obs(0, "screen"),
                _obs(1, "battery", 0.95),
                _obs(2, "battery", 0.9),
                _obs(3, "cover"),
            ),
            frozenset({"anomaly"}),
        ),
    )


def evaluate(scenario: Scenario) -> tuple[bool, tuple[SequenceEvent, ...]]:
    events = AssemblySequenceFSM(SequenceRule(("screen", "battery", "cover"))).run(
        list(scenario.observations)
    )
    observed = {event.kind for event in events}
    issue_kinds = {
        "missing_step",
        "reordered_step",
        "repeated_step",
        "unexpected_step",
        "anomaly",
    }
    correct = scenario.required_event_kinds <= observed
    if not scenario.required_event_kinds:
        correct = not (observed & issue_kinds)
    return correct, events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for scenario in scenarios():
        correct, events = evaluate(scenario)
        rows.append(
            {
                "scenario": scenario.name,
                "required_event_kinds": sorted(scenario.required_event_kinds),
                "observed_event_kinds": sorted({event.kind for event in events}),
                "correct": correct,
                "events": [event.to_dict() for event in events],
            }
        )
    correct_count = sum(bool(row["correct"]) for row in rows)
    payload = {
        "protocol": "deterministic synthetic sequence fixture",
        "expected_steps": ["screen", "battery", "cover"],
        "scenario_count": len(rows),
        "correct_scenarios": correct_count,
        "scenario_accuracy": correct_count / len(rows),
        "rows": rows,
        "limitations": [
            "synthetic observations, not a real video dataset",
            "no claim about official accuracy, latency, or hidden-test behavior",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"scenarios={len(rows)} correct={correct_count} accuracy={correct_count / len(rows):.6f}")
    return 0 if correct_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
