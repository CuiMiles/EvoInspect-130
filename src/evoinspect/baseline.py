from __future__ import annotations

import json
import time
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from .data import read_manifest, sample_path
from .errors import EvoInspectError
from .images import centroid, euclidean, image_features
from .provenance import canonical_hash, file_sha256, utc_now, write_json


def _f1(labels: Sequence[int], predictions: Sequence[int]) -> tuple[float, float, float]:
    true_positive = sum(a == 1 and b == 1 for a, b in zip(labels, predictions, strict=False))
    false_positive = sum(a == 0 and b == 1 for a, b in zip(labels, predictions, strict=False))
    false_negative = sum(a == 1 and b == 0 for a, b in zip(labels, predictions, strict=False))
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return score, precision, recall


def select_threshold(scores: Sequence[float], labels: Sequence[int]) -> dict[str, float]:
    if not scores or len(scores) != len(labels) or set(labels) != {0, 1}:
        raise EvoInspectError("development calibration requires both labels and aligned scores")
    unique = sorted(set(scores))
    candidates = [unique[0] - 1e-12]
    candidates.extend((left + right) / 2 for left, right in pairwise(unique))
    candidates.append(unique[-1] + 1e-12)
    best: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        predictions = [int(score >= threshold) for score in scores]
        f1, precision, recall = _f1(labels, predictions)
        candidate = (f1, precision, recall, -threshold)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return {
        "threshold": -best[3],
        "development_f1": best[0],
        "development_precision": best[1],
        "development_recall": best[2],
    }


def anomaly_score(vector: Sequence[float], normal_centroid: Sequence[float]) -> float:
    return euclidean(vector, normal_centroid)


def adapt_fixture_baseline(
    split_manifest: Path, config: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    records = read_manifest(split_manifest)
    allowed_roles = {"support_normal", "support_anomaly", "development"}
    adaptation = [row for row in records if row.get("role") in allowed_roles]
    forbidden = [row for row in adaptation if row.get("role") == "final_test"]
    if forbidden:
        raise EvoInspectError("internal error: final_test entered adaptation")
    grid = int(config["feature_grid"])
    normal_vectors = [
        image_features(sample_path(row, split_manifest), grid)
        for row in adaptation
        if row["role"] == "support_normal"
    ]
    anomaly_vectors = [
        image_features(sample_path(row, split_manifest), grid)
        for row in adaptation
        if row["role"] == "support_anomaly"
    ]
    if not normal_vectors or not anomaly_vectors:
        raise EvoInspectError("adaptation needs normal and anomaly support samples")
    normal_center = centroid(normal_vectors)
    anomaly_center = centroid(anomaly_vectors)
    development = [row for row in adaptation if row["role"] == "development"]
    development_vectors = [
        image_features(sample_path(row, split_manifest), grid) for row in development
    ]
    scores = [anomaly_score(vector, normal_center) for vector in development_vectors]
    labels = [int(row["label"] == "anomaly") for row in development]
    calibration = select_threshold(scores, labels)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "model_id": "fixture-stat-v1",
        "status": "engineering_test_only",
        "created_at": utc_now(),
        "feature_grid": grid,
        "normal_centroid": normal_center,
        "anomaly_centroid": anomaly_center,
        "threshold": calibration["threshold"],
        "calibration": calibration,
        "calibration_samples": len(development),
        "normal_support_samples": len(normal_vectors),
        "anomaly_support_samples": len(anomaly_vectors),
        "split_hash": file_sha256(split_manifest),
        "config_hash": canonical_hash(config),
        "warning": "Fixture engineering baseline; not valid scientific or competition evidence.",
    }
    hash_payload = dict(payload)
    hash_payload.pop("created_at")
    payload["model_hash"] = canonical_hash(hash_payload)
    write_json(output_path, payload)
    return payload


def load_model(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("model_id") != "fixture-stat-v1":
        raise EvoInspectError(f"unsupported model: {path}")
    expected_hash = value.get("model_hash")
    unhashed = dict(value)
    unhashed.pop("model_hash", None)
    unhashed.pop("created_at", None)
    if expected_hash != canonical_hash(unhashed):
        raise EvoInspectError(f"model hash mismatch: {path}")
    return cast(dict[str, Any], value)


def infer_manifest(
    split_manifest: Path,
    model_path: Path,
    output_path: Path,
    roles: set[str],
) -> dict[str, Any]:
    model = load_model(model_path)
    records = [row for row in read_manifest(split_manifest) if row.get("role") in roles]
    if not records:
        raise EvoInspectError(f"no records found for roles {sorted(roles)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    latencies: list[float] = []
    with temporary.open("w", encoding="utf-8") as stream:
        for row in records:
            started = time.perf_counter()
            preprocess_started = time.perf_counter()
            vector = image_features(sample_path(row, split_manifest), int(model["feature_grid"]))
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
            inference_started = time.perf_counter()
            score = anomaly_score(vector, model["normal_centroid"])
            known_distance = euclidean(vector, model["anomaly_centroid"])
            inference_ms = (time.perf_counter() - inference_started) * 1000
            post_started = time.perf_counter()
            prediction = "anomaly" if score >= float(model["threshold"]) else "normal"
            if prediction == "normal":
                defect_tag = "normal"
            else:
                defect_tag = "known_like" if known_distance < score else "unknown_like"
            confidence = min(1.0, abs(score - float(model["threshold"])) / max(score, 1e-12))
            postprocess_ms = (time.perf_counter() - post_started) * 1000
            prediction_record: dict[str, Any] = {
                "sample_id": row["sample_id"],
                "anomaly_score": score,
                "binary_decision": prediction,
                "known_or_unknown_tag": defect_tag,
                "confidence": confidence,
                "model_version": model["model_hash"],
                "latency_ms": {
                    "preprocess": preprocess_ms,
                    "model_score": inference_ms,
                    "postprocess": postprocess_ms,
                },
            }
            serialization_started = time.perf_counter()
            json.dumps(prediction_record, ensure_ascii=False, sort_keys=True)
            serialization_ms = (time.perf_counter() - serialization_started) * 1000
            prediction_record["latency_ms"]["serialization_encode"] = serialization_ms
            prediction_record["latency_ms"]["end_to_end_without_file_io"] = (
                time.perf_counter() - started
            ) * 1000
            stream.write(json.dumps(prediction_record, ensure_ascii=False, sort_keys=True) + "\n")
            latencies.append(prediction_record["latency_ms"]["end_to_end_without_file_io"])
    temporary.replace(output_path)
    ordered = sorted(latencies)
    return {
        "predictions": len(records),
        "roles": sorted(roles),
        "model_hash": model["model_hash"],
        "predictions_hash": file_sha256(output_path),
        "fixture_latency_ms": {
            "p50": _percentile(ordered, 0.50),
            "p95": _percentile(ordered, 0.95),
            "max": ordered[-1],
            "scope": "CPU fixture, end-to-end excluding file I/O; not deployment evidence",
        },
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("empty percentile")
    index = (len(values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction
