#!/usr/bin/env python3
"""Train and evaluate frozen EfficientAD-M/S under the isolated 100+30 protocol."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from evoinspect.baseline import select_threshold
from evoinspect.provenance import (
    append_csv,
    canonical_hash,
    file_sha256,
    git_state,
    utc_now,
    write_json,
)
from scripts.efficientad_rcbr_100_30 import (
    REGISTRY_COLUMNS,
    evaluate_strategy,
    infer_array,
    load_rgb,
    load_yaml,
    read_csv,
    score_map,
    train_efficientad,
)


def zero_audit() -> dict[str, Any]:
    return {
        "rois": [],
        "selected_roi_count": 0,
        "applied_roi_count": 0,
        "fallback_count": 0,
        "roi_area_fraction": 0.0,
        "predicted_roi_cost_ms": 0.0,
        "measured_unique_roi_inference_ms": 0.0,
    }


def run_task(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    adaptation = read_csv(args.adaptation)
    support_normals = [row for row in adaptation if row["role"] == "support_normal"]
    development = [row for row in adaptation if row["role"] == "development"]
    if (
        not support_normals
        or not development
        or {row["label"] for row in development}
        != {
            "normal",
            "anomaly",
        }
    ):
        raise RuntimeError("adaptation manifest lacks normal support or two-class development data")
    requested_train = int(config["normal_support_train"])
    train_count = min(requested_train, max(1, int(len(support_normals) * 0.8)))
    train_normals = support_normals[:train_count]
    calibration_normals = support_normals[train_count:]
    if len(calibration_normals) < 2:
        raise RuntimeError("at least two held-out support normals are required")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    model, training = train_efficientad(
        train_normals,
        calibration_normals,
        [row for row in development if row["label"] == "anomaly"],
        args.output_dir,
        config,
        args.seed,
    )
    shape = tuple(int(value) for value in config["input_resolution"])
    quantile = float(config["inference"]["score_quantile"])
    development_maps = []
    development_scores = []
    for row in development:
        anomaly_map, _ = infer_array(model, load_rgb(row["path"]), shape)
        development_maps.append(anomaly_map)
        development_scores.append(score_map(anomaly_map, quantile))
    threshold = select_threshold(
        development_scores, [int(row["label"] == "anomaly") for row in development]
    )

    # Predict every test input before opening the separately stored truth file.
    test_inputs = read_csv(args.test_inputs)
    final_maps = []
    final_scores = []
    inference_times = []
    for row in test_inputs:
        anomaly_map, elapsed = infer_array(model, load_rgb(row["path"]), shape)
        final_maps.append(anomaly_map)
        final_scores.append(score_map(anomaly_map, quantile))
        inference_times.append(elapsed)
    prediction_path = args.output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as stream:
        for row, score, elapsed in zip(test_inputs, final_scores, inference_times, strict=True):
            stream.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "score": score,
                        "decision": int(score >= float(threshold["threshold"])),
                        "model_ms": elapsed,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    truth_by_id = {row["sample_id"]: row for row in read_csv(args.test_truth)}
    if set(truth_by_id) != {row["sample_id"] for row in test_inputs}:
        raise RuntimeError("test truth/input coverage mismatch")
    truth_rows = [truth_by_id[row["sample_id"]] for row in test_inputs]
    result = evaluate_strategy(
        truth_rows,
        final_maps,
        final_scores,
        float(threshold["threshold"]),
        [zero_audit() for _ in final_scores],
    )
    np.savez_compressed(args.output_dir / "prediction_maps.npz", predictions=np.stack(final_maps))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    commit, dirty = git_state(Path.cwd())
    metrics = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "completed",
        "started_at": started_at,
        "ended_at": utc_now(),
        "category": split["category"],
        "seed": args.seed,
        "protocol": split["protocol"],
        "split_hash": split["split_hash"],
        "adaptation_sha256": file_sha256(args.adaptation),
        "test_inputs_sha256": file_sha256(args.test_inputs),
        "test_truth_sha256": file_sha256(args.test_truth),
        "config_hash": canonical_hash(config),
        "git_commit": commit,
        "dirty": dirty,
        "model_id": config["model_id"],
        "model_size": config["model_size"],
        "training": training,
        "calibration": {
            "normal_train_count": len(train_normals),
            "normal_calibration_count": len(calibration_normals),
            "development_count": len(development),
            "threshold_development_only": threshold,
        },
        "result": result,
        "latency_model_segment_ms": {
            "p50": float(np.quantile(inference_times, 0.50)),
            "p95": float(np.quantile(inference_times, 0.95)),
            "p99": float(np.quantile(inference_times, 0.99)),
            "max": float(np.max(inference_times)),
            "scope": "256x256 model segment diagnostic only",
        },
        "leakage_audit": {
            "test_label_reads_before_all_predictions_fixed": 0,
            "test_labels_used_for_training": 0,
            "test_labels_used_for_threshold": 0,
            "test_labels_used_for_model_selection": 0,
        },
        "hardware": platform.platform(),
        "physical_gpu": os.environ.get("EVOINSPECT_PHYSICAL_GPU", "UNRECORDED"),
        "warnings": [
            "Test truth was opened only after all test scores and decisions were fixed.",
            (
                "Per-run timing is a 256x256 model-segment diagnostic, not the frozen "
                "2500x2500 benchmark."
            ),
        ],
    }
    write_json(args.output_dir / "metrics.json", metrics)
    return metrics


def register(args: argparse.Namespace, start: str, status: str, error: str = "") -> None:
    split = json.loads(args.split.read_text(encoding="utf-8"))
    config = load_yaml(args.config)
    metrics_path = args.output_dir / "metrics.json"
    commit, dirty = git_state(Path.cwd())
    row = {
        "run_id": args.run_id,
        "status": status,
        "start_time": start,
        "end_time": utc_now(),
        "git_commit": commit,
        "dirty": str(dirty).lower(),
        "config_hash": canonical_hash(config),
        "data_hash": split.get("source_manifest_sha256", "UNAVAILABLE"),
        "split_hash": split.get("split_hash", "UNAVAILABLE"),
        "seed": args.seed,
        "hardware": f"physical GPU {os.environ.get('EVOINSPECT_PHYSICAL_GPU', 'UNRECORDED')}",
        "model": config["model_id"],
        "protocol": split.get("protocol", "UNAVAILABLE"),
        "metrics_path": str(metrics_path) if metrics_path.is_file() else "",
        "artifact_path": str(args.output_dir / "model.ckpt")
        if (args.output_dir / "model.ckpt").is_file()
        else "",
        "failure_reason": error,
        "notes": f"category={split.get('category')}; frozen EfficientAD baseline",
    }
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.registry.with_suffix(args.registry.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        append_csv(args.registry, REGISTRY_COLUMNS, row)
        fcntl.flock(lock, fcntl.LOCK_UN)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    for name in (
        "adaptation",
        "test-inputs",
        "test-truth",
        "split",
        "config",
        "output-dir",
        "registry",
    ):
        value.add_argument(f"--{name}", required=True, type=Path)
    value.add_argument("--run-id", required=True)
    value.add_argument("--seed", required=True, type=int)
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    started = utc_now()
    try:
        run_task(args)
        register(args, started, "completed_efficientad_frozen")
    except Exception as exception:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        error = f"{type(exception).__name__}: {exception}"
        write_json(
            args.output_dir / "failure.json", {"error": error, "traceback": traceback.format_exc()}
        )
        register(args, started, "failed_efficientad_frozen", error)
        raise
