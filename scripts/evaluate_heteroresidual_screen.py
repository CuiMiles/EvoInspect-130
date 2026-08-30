#!/usr/bin/env python3
"""Evaluate the pre-registered frozen-backbone HeteroResidual-S screen."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from evoinspect.baseline import select_threshold
from evoinspect.heterocal import fit_nonnegative_residual, leave_one_defect_type_out
from evoinspect.provenance import file_sha256, git_state, utc_now, write_json
from scripts.efficientad_rcbr_100_30 import evaluate_strategy, read_csv
from scripts.evaluate_efficientad_strict_100_30 import (
    image_tensor,
    load_fixed_model,
    strict_calibration_rows,
)

FloatArray = NDArray[np.float64]


def top_mean(values: NDArray[np.float32], fraction: float) -> float:
    flat = values.reshape(-1)
    count = max(1, int(np.ceil(flat.size * fraction)))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


def feature_vector(st: NDArray[np.float32], ae: NDArray[np.float32]) -> list[float]:
    fused = 0.5 * st + 0.5 * ae
    disagreement = np.abs(st - ae)
    # Four residual evidence heads: local peak, branch asymmetry, texture tail, and disagreement.
    return [
        float(np.max(fused)),
        float(np.max(st)),
        float(np.max(ae)),
        top_mean(fused, 0.001),
        top_mean(st, 0.005),
        top_mean(ae, 0.005),
        top_mean(fused, 0.01),
        float(np.mean(disagreement)),
        float(np.max(disagreement)),
        float(np.mean(fused > 0.1)),
    ]


def infer(
    model: Any, path: str | Path, shape: tuple[int, int], device: torch.device
) -> tuple[FloatArray, NDArray[np.float32], float]:
    tensor = image_tensor(path, shape, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        st, ae = model.model.get_maps(tensor, normalize=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - started) * 1000.0
    st_np = st.float().cpu().squeeze().numpy().astype(np.float32)
    ae_np = ae.float().cpu().squeeze().numpy().astype(np.float32)
    return (
        np.asarray(feature_vector(st_np, ae_np), dtype=np.float64),
        (0.5 * st_np + 0.5 * ae_np),
        elapsed,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = args.source_run.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    old_metrics = json.loads((source / "result" / "metrics.json").read_text(encoding="utf-8"))
    adaptation_path = source / "adaptation.csv"
    adaptation = read_csv(adaptation_path)
    normal_rows, anomaly_rows = strict_calibration_rows(
        adaptation, int(old_metrics["calibration"]["normal_train_count"])
    )
    support_rows = [*normal_rows, *anomaly_rows]
    labels = np.asarray([int(row["label"] == "anomaly") for row in support_rows], dtype=np.int64)
    defect_types = [row["defect_type"] for row in support_rows]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    shape = (256, 256)
    checkpoint = source / "result" / "model.ckpt"
    model = load_fixed_model(
        checkpoint,
        device,
        str(old_metrics["training"]["model_sha256"]),
        normal_rows,
        shape,
    )
    support_features: list[FloatArray] = []
    support_maps: list[NDArray[np.float32]] = []
    support_times: list[float] = []
    for row in support_rows:
        features, anomaly_map, elapsed = infer(model, row["path"], shape, device)
        support_features.append(features)
        support_maps.append(anomaly_map)
        support_times.append(elapsed)
    support_array = np.stack(support_features)
    base_scores = support_array[:, 0]
    head = fit_nonnegative_residual(
        support_array,
        labels,
        support_array[labels == 0],
        delta=float(config["calibrator"]["residual_delta"]),
        l2=float(config["calibrator"]["l2"]),
        learning_rate=float(config["calibrator"]["learning_rate"]),
        iterations=int(config["calibrator"]["iterations"]),
        scale_floor=float(config["calibrator"]["normal_scale_floor"]),
    )
    selection = leave_one_defect_type_out(support_array, labels, defect_types, config)
    support_scores = head.score(support_array) if selection["accepted"] else base_scores
    threshold = select_threshold(support_scores.tolist(), labels.tolist())

    test_inputs = read_csv(source / "test_inputs.csv")
    if any(row["label"] or row["defect_type"] for row in test_inputs):
        raise RuntimeError("test input manifest exposes labels")
    test_features: list[FloatArray] = []
    test_maps: list[NDArray[np.float32]] = []
    latencies: list[float] = []
    for row in test_inputs:
        features, anomaly_map, elapsed = infer(model, row["path"], shape, device)
        test_features.append(features)
        test_maps.append(anomaly_map)
        latencies.append(elapsed)
    test_array = np.stack(test_features)
    test_scores = head.score(test_array) if selection["accepted"] else test_array[:, 0]
    prediction_path = output / "predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8") as stream:
        for row, score in zip(test_inputs, test_scores, strict=True):
            stream.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "score": float(score),
                        "decision": int(score >= threshold["threshold"]),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    np.savez_compressed(output / "prediction_maps.npz", predictions=np.stack(test_maps))

    truth_by_id = {row["sample_id"]: row for row in read_csv(source / "test_truth.csv")}
    if set(truth_by_id) != {row["sample_id"] for row in test_inputs}:
        raise RuntimeError("test truth/input coverage mismatch")
    truth_rows = [truth_by_id[row["sample_id"]] for row in test_inputs]
    audits = [
        {
            "rois": [],
            "selected_roi_count": 0,
            "applied_roi_count": 0,
            "fallback_count": 0,
            "roi_area_fraction": 0.0,
            "predicted_roi_cost_ms": 0.0,
            "measured_unique_roi_inference_ms": 0.0,
        }
        for _ in test_maps
    ]
    result = evaluate_strategy(
        truth_rows, test_maps, test_scores.tolist(), threshold["threshold"], audits
    )
    split = json.loads((source / "split.json").read_text(encoding="utf-8"))
    commit, dirty = git_state(Path.cwd())
    metrics = {
        "schema_version": 1,
        "status": "completed_heteroresidual_screen",
        "run_id": args.run_id,
        "started_at": args.started_at,
        "ended_at": utc_now(),
        "category": split["category"],
        "seed": split["seed"],
        "variant": "heteroresidual_s",
        "source_run": str(source),
        "source_checkpoint_sha256": file_sha256(checkpoint),
        "source_adaptation_sha256": file_sha256(adaptation_path),
        "source_test_inputs_sha256": file_sha256(source / "test_inputs.csv"),
        "git_commit": commit,
        "dirty": dirty,
        "support": {
            "normal_count": len(normal_rows),
            "anomaly_count": len(anomaly_rows),
            "threshold_support_only": threshold,
            "selection": selection,
            "feature_count": int(support_array.shape[1]),
        },
        "result": result,
        "latency_model_segment_ms": {
            "p50": float(np.quantile(latencies, 0.50)),
            "p95": float(np.quantile(latencies, 0.95)),
            "p99": float(np.quantile(latencies, 0.99)),
            "max": float(np.max(latencies)),
            "scope": f"{device}, one get_maps call, 256x256 model segment",
        },
        "leakage_audit": {
            "test_label_reads_before_all_predictions_fixed": 0,
            "test_labels_used_for_training": 0,
            "test_labels_used_for_threshold": 0,
            "test_labels_used_for_model_selection": 0,
        },
        "hardware": platform.platform(),
        "physical_gpu": os.environ.get("EVOINSPECT_PHYSICAL_GPU", "CPU_OR_UNRECORDED"),
        "warnings": [
            "Exploratory screen only; not an official gate or final submission claim.",
            (
                "The EfficientAD backbone is frozen; only a support-fitted residual score "
                "head changes."
            ),
        ],
    }
    np.savez_compressed(output / "support_features.npz", features=support_array, labels=labels)
    write_json(output / "metrics.json", metrics)
    return metrics


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--source-run", required=True, type=Path)
    value.add_argument("--output-dir", required=True, type=Path)
    value.add_argument(
        "--config",
        default=Path("configs/experiments/heteroresidual_screen_20260831.yaml"),
        type=Path,
    )
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--run-id", required=True)
    value.add_argument("--started-at", default=utc_now())
    return value


if __name__ == "__main__":
    payload = run(parser().parse_args())
    print(
        json.dumps(
            {"run_id": payload["run_id"], "result": payload["result"]["overall"]}, sort_keys=True
        )
    )
