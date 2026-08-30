#!/usr/bin/env python3
"""Evaluate a pre-registered, train-free EfficientAD resolution/tile variant.

The source checkpoint and split are fixed by an existing strict 100+30 run. This script is an
exploratory screen; it never opens ``test_truth.csv`` until every test prediction is written.
"""

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
from PIL import Image

from evoinspect.baseline import select_threshold
from evoinspect.provenance import file_sha256, git_state, utc_now, write_json
from scripts.efficientad_rcbr_100_30 import evaluate_strategy, read_csv
from scripts.evaluate_efficientad_strict_100_30 import load_fixed_model, strict_calibration_rows


def tensor_from_array(
    array: np.ndarray[Any, Any], shape: tuple[int, int], device: torch.device
) -> torch.Tensor:
    tensor = torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).float().div_(255.0)[None]
    return torch.nn.functional.interpolate(
        tensor, size=shape, mode="bilinear", align_corners=False, antialias=True
    ).to(device)


def load_array(path: str | Path) -> np.ndarray[Any, Any]:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def forward(
    model: Any, array: np.ndarray[Any, Any], shape: tuple[int, int], device: torch.device
) -> tuple[np.ndarray[Any, Any], float]:
    tensor = tensor_from_array(array, shape, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = model.model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    anomaly_map = np.asarray(
        prediction.anomaly_map.detach().float().cpu().squeeze().numpy(),
        dtype=np.float32,
    )
    return anomaly_map, elapsed_ms


def tile_arrays(array: np.ndarray[Any, Any]) -> list[np.ndarray[Any, Any]]:
    height, width = array.shape[:2]
    mid_y, mid_x = height // 2, width // 2
    # Fixed, non-overlapping quadrants are deliberately simple and deterministic.
    return [
        array[:mid_y, :mid_x],
        array[:mid_y, mid_x:],
        array[mid_y:, :mid_x],
        array[mid_y:, mid_x:],
    ]


def infer_variant(
    model: Any,
    path: str | Path,
    shape: tuple[int, int],
    variant: str,
    device: torch.device,
) -> tuple[np.ndarray[Any, Any], float, int]:
    array = load_array(path)
    global_map, elapsed = forward(model, array, shape, device)
    maps = [global_map]
    forwards = 1
    if variant == "static_tile_efficientad_s":
        for tile in tile_arrays(array):
            tile_map, tile_elapsed = forward(model, tile, shape, device)
            maps.append(tile_map)
            elapsed += tile_elapsed
            forwards += 1
    elif variant != "global_single_forward":
        raise ValueError(f"unsupported variant: {variant}")
    merged = np.maximum.reduce(maps).astype(np.float32, copy=False)
    return merged, elapsed, forwards


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_run.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    old_metrics_path = source / "result" / "metrics.json"
    old_metrics = json.loads(old_metrics_path.read_text(encoding="utf-8"))
    adaptation_path = source / "adaptation.csv"
    adaptation = read_csv(adaptation_path)
    normal_rows, anomaly_rows = strict_calibration_rows(
        adaptation, int(old_metrics["calibration"]["normal_train_count"])
    )
    support_rows = [*normal_rows, *anomaly_rows]
    support_labels = [int(row["label"] == "anomaly") for row in support_rows]
    shape = (int(args.resolution), int(args.resolution))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = load_fixed_model(
        source / "result" / "model.ckpt",
        device,
        str(old_metrics["training"]["model_sha256"]),
        normal_rows,
        shape,
    )

    support_maps: list[np.ndarray[Any, Any]] = []
    support_scores: list[float] = []
    support_forward_counts: list[int] = []
    for row in support_rows:
        anomaly_map, _, forwards = infer_variant(model, row["path"], shape, args.inference, device)
        support_maps.append(anomaly_map)
        support_scores.append(float(np.max(anomaly_map)))
        support_forward_counts.append(forwards)
    threshold = select_threshold(support_scores, support_labels)

    test_inputs = read_csv(source / "test_inputs.csv")
    if any(row["label"] or row["defect_type"] for row in test_inputs):
        raise RuntimeError("test input manifest exposes labels")
    test_maps: list[np.ndarray[Any, Any]] = []
    test_scores: list[float] = []
    test_times: list[float] = []
    test_forwards: list[int] = []
    for row in test_inputs:
        anomaly_map, elapsed, forwards = infer_variant(
            model, row["path"], shape, args.inference, device
        )
        test_maps.append(anomaly_map)
        test_scores.append(float(np.max(anomaly_map)))
        test_times.append(elapsed)
        test_forwards.append(forwards)

    prediction_path = output / "predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8") as stream:
        for row, score in zip(test_inputs, test_scores, strict=True):
            stream.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "score": score,
                        "decision": int(score >= float(threshold["threshold"])),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    np.savez_compressed(output / "prediction_maps.npz", predictions=np.stack(test_maps))

    # Deliberately first read truth after predictions are durable.
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
    result = evaluate_strategy(truth_rows, test_maps, test_scores, threshold["threshold"], audits)
    commit, dirty = git_state(Path.cwd())
    checkpoint = source / "result" / "model.ckpt"
    metrics = {
        "schema_version": 1,
        "status": "completed_parallel_screen_variant",
        "run_id": args.run_id,
        "started_at": args.started_at,
        "ended_at": utc_now(),
        "category": json.loads((source / "split.json").read_text(encoding="utf-8"))["category"],
        "seed": json.loads((source / "split.json").read_text(encoding="utf-8"))["seed"],
        "variant": args.inference,
        "variant_id": (
            f"efficientad_s_{args.resolution}"
            if args.inference == "global_single_forward"
            else "static_tile_efficientad_s"
        ),
        "resolution": list(shape),
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
            "forward_count_distribution": sorted(set(support_forward_counts)),
        },
        "result": result,
        "latency_model_segment_ms": {
            "p50": float(np.quantile(test_times, 0.50)),
            "p95": float(np.quantile(test_times, 0.95)),
            "p99": float(np.quantile(test_times, 0.99)),
            "max": float(np.max(test_times)),
            "mean_forward_count": float(np.mean(test_forwards)),
            "scope": f"{device}, model segment only; source images resized to {shape}",
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
            "Exploratory screen only; not an official quality gate or deployment benchmark.",
            (
                "Static tiles are fixed non-overlapping quadrants and do not imply native "
                "2500px accuracy."
            ),
            "Existing submission files are not modified by this evaluator.",
        ],
    }
    write_json(output / "metrics.json", metrics)
    return metrics


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--source-run", required=True, type=Path)
    value.add_argument("--output-dir", required=True, type=Path)
    value.add_argument(
        "--inference", required=True, choices=["global_single_forward", "static_tile_efficientad_s"]
    )
    value.add_argument("--resolution", required=True, type=int)
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
