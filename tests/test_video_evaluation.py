from __future__ import annotations

from evoinspect.video_evaluation import aggregate_metrics, evaluate_clip


def event(kind: str, step: str, time: float, public: str | None = None) -> dict[str, object]:
    return {"kind": kind, "step": step, "start_time_s": time, "public_kind": public}


def test_bipartite_event_matching_is_one_to_one() -> None:
    result = evaluate_clip(
        [event("step_completed", "cup", 2.0), event("step_completed", "cup", 2.1)],
        [
            {
                "event_id": 1,
                "gt_start": 1.8,
                "gt_end": 2.2,
                "component": "cup",
                "event_type": "step_completed",
            }
        ],
        duration_seconds=5.0,
        tolerance_seconds=0.5,
    )
    assert result["metrics"]["true_positive"] == 1
    assert result["metrics"]["predicted"] == 2
    assert len(result["false_positives"]) == 1


def test_missing_uses_stream_end_window_and_aggregate_counts_types() -> None:
    result = evaluate_clip(
        [event("missing_step", "mouse", 9.9, "missing")],
        [
            {
                "event_id": 1,
                "gt_start": 8.0,
                "gt_end": 10.0,
                "component": "mouse",
                "event_type": "missing",
            }
        ],
        duration_seconds=10.0,
        tolerance_seconds=0.5,
    )
    aggregate = aggregate_metrics([result])
    assert aggregate["micro"]["f1"] == 1.0
    assert aggregate["by_event_type"]["missing"]["true_positive"] == 1
