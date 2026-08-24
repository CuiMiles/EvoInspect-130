from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .data import read_manifest
from .errors import EvoInspectError
from .provenance import file_sha256, utc_now, write_json


def binary_metrics(
    labels: Sequence[int], scores: Sequence[float], predictions: Sequence[int]
) -> dict[str, float]:
    if not labels or not (len(labels) == len(scores) == len(predictions)):
        raise EvoInspectError("metrics inputs must be non-empty and aligned")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise EvoInspectError("AUROC/AP require both classes")
    tp = sum(
        label == 1 and prediction == 1
        for label, prediction in zip(labels, predictions, strict=False)
    )
    tn = sum(
        label == 0 and prediction == 0
        for label, prediction in zip(labels, predictions, strict=False)
    )
    fp = sum(
        label == 0 and prediction == 1
        for label, prediction in zip(labels, predictions, strict=False)
    )
    fn = sum(
        label == 1 and prediction == 0
        for label, prediction in zip(labels, predictions, strict=False)
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": float(len(labels)),
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1_fixed_threshold": f1,
        "auroc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "true_positive": float(tp),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
    }


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=False) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=False) if label == 0]
    if not positives or not negatives:
        raise EvoInspectError("AUROC requires both classes")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    ranked = sorted(zip(scores, labels, strict=False), key=lambda pair: pair[0], reverse=True)
    positives = sum(labels)
    if positives == 0:
        raise EvoInspectError("average precision requires positives")
    true_positive = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positives


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvoInspectError(f"invalid prediction JSON at line {line_number}") from exc
        predictions.append(value)
    return predictions


def _slice_metrics(
    rows: Iterable[tuple[dict[str, str], dict[str, Any]]],
) -> dict[str, float] | None:
    pairs = list(rows)
    if not pairs:
        return None
    labels = [int(truth["label"] == "anomaly") for truth, _ in pairs]
    if set(labels) != {0, 1}:
        return None
    scores = [float(prediction["anomaly_score"]) for _, prediction in pairs]
    decisions = [int(prediction["binary_decision"] == "anomaly") for _, prediction in pairs]
    return binary_metrics(labels, scores, decisions)


def evaluate_predictions(
    split_manifest: Path,
    predictions_path: Path,
    output_path: Path,
    model_hash: str,
) -> dict[str, Any]:
    truth_rows = [row for row in read_manifest(split_manifest) if row.get("role") == "final_test"]
    truth = {row["sample_id"]: row for row in truth_rows}
    predictions = _load_predictions(predictions_path)
    if any("label" in row or "defect_type" in row for row in predictions):
        raise EvoInspectError("predictions must not contain ground-truth fields")
    if any(str(row.get("model_version")) != model_hash for row in predictions):
        raise EvoInspectError("prediction model_version does not match the evaluated model")
    prediction_map = {str(row.get("sample_id")): row for row in predictions}
    if len(prediction_map) != len(predictions):
        raise EvoInspectError("duplicate sample_id in predictions")
    if set(prediction_map) != set(truth):
        missing = sorted(set(truth) - set(prediction_map))
        extra = sorted(set(prediction_map) - set(truth))
        raise EvoInspectError(f"prediction coverage mismatch; missing={missing}, extra={extra}")
    pairs = [(truth[sample_id], prediction_map[sample_id]) for sample_id in sorted(truth)]
    labels = [int(row[0]["label"] == "anomaly") for row in pairs]
    scores = [float(row[1]["anomaly_score"]) for row in pairs]
    decisions = [int(row[1]["binary_decision"] == "anomaly") for row in pairs]
    overall = binary_metrics(labels, scores, decisions)
    seen_pairs = [
        pair
        for pair in pairs
        if pair[0]["label"] == "normal" or pair[0].get("defect_visibility") == "seen"
    ]
    unseen_pairs = [
        pair
        for pair in pairs
        if pair[0]["label"] == "normal" or pair[0].get("defect_visibility") == "unseen"
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "protocol": "fixture_vertical_slice",
        "dataset": "generated_smoke_fixture",
        "status": "engineering_test_only",
        "overall": overall,
        "seen_slice": _slice_metrics(seen_pairs),
        "unseen_slice": _slice_metrics(unseen_pairs),
        "model_hash": model_hash,
        "split_hash": file_sha256(split_manifest),
        "predictions_hash": file_sha256(predictions_path),
        "warning": (
            "Synthetic fixture metrics are forbidden as algorithm, public benchmark, "
            "competition, or deployment evidence."
        ),
    }
    write_json(output_path, result)
    return result
