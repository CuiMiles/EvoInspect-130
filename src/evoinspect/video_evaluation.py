"""Event-level evaluation for the fixed-camera assembly functional demo."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

ALLOWED_EVENT_TYPES = frozenset(
    {"step_completed", "skip", "reorder", "repeat", "missing", "unknown"}
)


def normalize_prediction(event: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an internal FSM event into the frozen public event vocabulary."""

    kind = str(event["kind"])
    if kind == "step_completed":
        event_type = "step_completed"
    else:
        public = event.get("public_kind")
        if public not in ALLOWED_EVENT_TYPES:
            return None
        event_type = str(public)
    component = event.get("step")
    if not isinstance(component, str) or not component:
        return None
    return {
        "prediction_index": int(event.get("prediction_index", -1)),
        "time": float(event["start_time_s"]),
        "component": component,
        "event_type": event_type,
    }


def _maximum_matching(edges: list[list[int]], ground_truth_count: int) -> list[tuple[int, int]]:
    """Return a deterministic maximum-cardinality bipartite matching."""

    matched_prediction = [-1] * ground_truth_count

    def augment(prediction_index: int, visited: set[int]) -> bool:
        for ground_truth_index in edges[prediction_index]:
            if ground_truth_index in visited:
                continue
            visited.add(ground_truth_index)
            previous = matched_prediction[ground_truth_index]
            if previous == -1 or augment(previous, visited):
                matched_prediction[ground_truth_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(edges)):
        augment(prediction_index, set())
    return sorted(
        (prediction_index, ground_truth_index)
        for ground_truth_index, prediction_index in enumerate(matched_prediction)
        if prediction_index >= 0
    )


def _metrics(true_positive: int, predicted: int, ground_truth: int) -> dict[str, float | int]:
    precision = true_positive / predicted if predicted else float(ground_truth == 0)
    recall = true_positive / ground_truth if ground_truth else float(predicted == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_clip(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    *,
    duration_seconds: float,
    tolerance_seconds: float,
) -> dict[str, Any]:
    """Match one clip using event type, component and the frozen time window."""

    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(predictions):
        item = normalize_prediction({**raw, "prediction_index": index})
        if item is not None:
            normalized.append(item)
    edges: list[list[int]] = []
    for prediction in normalized:
        compatible: list[int] = []
        for index, truth in enumerate(ground_truth):
            event_type = str(truth["event_type"])
            if event_type not in ALLOWED_EVENT_TYPES:
                raise ValueError(f"unsupported GT event type: {event_type}")
            if prediction["event_type"] != event_type:
                continue
            if prediction["component"] != truth["component"]:
                continue
            if event_type == "missing":
                lower = float(truth["gt_start"])
                upper = duration_seconds + tolerance_seconds
            else:
                lower = float(truth["gt_start"]) - tolerance_seconds
                upper = float(truth["gt_end"]) + tolerance_seconds
            if lower <= prediction["time"] <= upper:
                compatible.append(index)
        edges.append(compatible)
    matches = _maximum_matching(edges, len(ground_truth))
    matched_predictions = {prediction for prediction, _ in matches}
    matched_truth = {truth for _, truth in matches}
    return {
        "metrics": _metrics(len(matches), len(normalized), len(ground_truth)),
        "matches": [
            {
                "prediction": normalized[prediction],
                "ground_truth": ground_truth[truth],
            }
            for prediction, truth in matches
        ],
        "false_positives": [
            prediction
            for index, prediction in enumerate(normalized)
            if index not in matched_predictions
        ],
        "false_negatives": [
            truth for index, truth in enumerate(ground_truth) if index not in matched_truth
        ],
    }


def aggregate_metrics(clip_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate micro metrics and per-event-type counts across clips."""

    true_positive = sum(int(item["metrics"]["true_positive"]) for item in clip_results)
    predicted = sum(int(item["metrics"]["predicted"]) for item in clip_results)
    ground_truth = sum(int(item["metrics"]["ground_truth"]) for item in clip_results)
    by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"true_positive": 0, "predicted": 0, "ground_truth": 0}
    )
    for result in clip_results:
        for match in result["matches"]:
            by_type[match["ground_truth"]["event_type"]]["true_positive"] += 1
        for match in result["matches"]:
            by_type[match["prediction"]["event_type"]]["predicted"] += 1
            by_type[match["ground_truth"]["event_type"]]["ground_truth"] += 1
        for item in result["false_positives"]:
            by_type[item["event_type"]]["predicted"] += 1
        for item in result["false_negatives"]:
            by_type[item["event_type"]]["ground_truth"] += 1
    return {
        "micro": _metrics(true_positive, predicted, ground_truth),
        "by_event_type": {
            event_type: _metrics(
                counts["true_positive"], counts["predicted"], counts["ground_truth"]
            )
            for event_type, counts in sorted(by_type.items())
        },
    }
