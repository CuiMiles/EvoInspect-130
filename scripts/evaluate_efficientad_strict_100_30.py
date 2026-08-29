#!/usr/bin/env python3
"""Re-evaluate a fixed EfficientAD checkpoint under strict 100+30 protocol v2."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from evoinspect.baseline import select_threshold
from evoinspect.provenance import canonical_hash, file_sha256, git_state, utc_now, write_json
from scripts.efficientad_rcbr_100_30 import evaluate_strategy, read_csv


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


def strict_calibration_rows(
    adaptation: list[dict[str, str]], normal_train_count: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    normals = [row for row in adaptation if row["role"] == "support_normal"]
    anomalies = [row for row in adaptation if row["role"] == "support_anomaly"]
    if len(normals) <= normal_train_count or not anomalies:
        raise RuntimeError(
            "strict evaluator requires held-out support normals and support anomalies"
        )
    return normals[normal_train_count:], anomalies


def image_tensor(path: str | Path, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).float().div_(255.0)[None]
    return torch.nn.functional.interpolate(
        tensor, size=shape, mode="bilinear", align_corners=False, antialias=True
    ).to(device)


def normalization_batches(
    rows: list[dict[str, str]], shape: tuple[int, int], device: torch.device
) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            image=image_tensor(row["path"], shape, device),
            gt_label=torch.zeros(1, dtype=torch.int64, device=device),
        )
        for row in rows
    ]


def infer_row(
    model: Any, row: dict[str, str], shape: tuple[int, int], device: torch.device
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float, float]:
    tensor = image_tensor(row["path"], shape, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = model.model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    anomaly_map = np.asarray(
        prediction.anomaly_map.detach().float().cpu().squeeze().numpy(), dtype=np.float32
    )
    score = float(prediction.pred_score.detach().float().cpu().reshape(-1)[0])
    map_max = float(np.max(anomaly_map))
    if not np.isclose(score, map_max, rtol=1e-6, atol=1e-7):
        raise RuntimeError("upstream pred_score is not the anomaly-map amax")
    return anomaly_map, score, elapsed_ms


def load_fixed_model(
    checkpoint: Path, device: torch.device, expected_sha256: str, normal_rows: list[dict[str, str]],
    shape: tuple[int, int],
) -> Any:
    from anomalib.models import EfficientAd

    if file_sha256(checkpoint) != expected_sha256:
        raise RuntimeError("checkpoint hash differs from the completed training record")
    model = EfficientAd.load_from_checkpoint(
        checkpoint, map_location=device, weights_only=False
    ).to(device)
    model.eval()
    final_quantiles = model.map_norm_quantiles(normalization_batches(normal_rows, shape, device))
    model.model.quantiles.update(final_quantiles)
    return model


def run(args: argparse.Namespace) -> dict[str, Any]:
    evaluator = yaml.safe_load(args.evaluator_config.read_text(encoding="utf-8"))
    training_config = yaml.safe_load(args.training_config.read_text(encoding="utf-8"))
    if canonical_hash(training_config) != evaluator["source_training_config_hash"]:
        raise RuntimeError("source training config changed after evaluator preregistration")
    old_metrics_path = args.run_dir / "result" / "metrics.json"
    old_metrics = json.loads(old_metrics_path.read_text(encoding="utf-8"))
    checkpoint = args.run_dir / "result" / "model.ckpt"
    if old_metrics["training"]["checkpoint"] != str(checkpoint.resolve()):
        raise RuntimeError("training record points to a different checkpoint")
    adaptation_path = args.run_dir / "adaptation.csv"
    adaptation = read_csv(adaptation_path)
    normal_rows, anomaly_rows = strict_calibration_rows(
        adaptation, int(training_config["normal_support_train"])
    )
    expected_normal = int(evaluator["map_normalization"]["count"])
    if len(normal_rows) != expected_normal:
        raise RuntimeError(
            f"expected {expected_normal} calibration normals, found {len(normal_rows)}"
        )
    if any(row["role"] == "development" for row in [*normal_rows, *anomaly_rows]):
        raise RuntimeError("development rows entered strict calibration")
    output = args.run_dir / str(evaluator["test_policy"]["output_directory_name"])
    output.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    shape = tuple(int(value) for value in training_config["input_resolution"])
    model = load_fixed_model(
        checkpoint,
        device,
        str(old_metrics["training"]["model_sha256"]),
        normal_rows,
        shape,
    )
    calibration_rows = [*normal_rows, *anomaly_rows]
    calibration_scores = [infer_row(model, row, shape, device)[1] for row in calibration_rows]
    threshold = select_threshold(
        calibration_scores, [int(row["label"] == "anomaly") for row in calibration_rows]
    )

    test_inputs_path = args.run_dir / "test_inputs.csv"
    test_inputs = read_csv(test_inputs_path)
    if any(row["label"] or row["defect_type"] for row in test_inputs):
        raise RuntimeError("test input manifest exposes labels")
    maps: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    scores: list[float] = []
    latencies: list[float] = []
    for row in test_inputs:
        anomaly_map, score, elapsed_ms = infer_row(model, row, shape, device)
        maps.append(anomaly_map)
        scores.append(score)
        latencies.append(elapsed_ms)
    prediction_path = output / "predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8") as stream:
        for row, score, elapsed_ms in zip(test_inputs, scores, latencies, strict=True):
            stream.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "score": score,
                        "decision": int(score >= float(threshold["threshold"])),
                        "model_ms": elapsed_ms,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    np.savez_compressed(output / "prediction_maps.npz", predictions=np.stack(maps))

    # The truth file is deliberately opened only after every score and decision is durable.
    truth_path = args.run_dir / "test_truth.csv"
    truth_by_id = {row["sample_id"]: row for row in read_csv(truth_path)}
    if set(truth_by_id) != {row["sample_id"] for row in test_inputs}:
        raise RuntimeError("test truth/input coverage mismatch")
    truth_rows = [truth_by_id[row["sample_id"]] for row in test_inputs]
    result = evaluate_strategy(
        truth_rows,
        maps,
        scores,
        float(threshold["threshold"]),
        [zero_audit() for _ in maps],
    )
    split = json.loads((args.run_dir / "split.json").read_text(encoding="utf-8"))
    commit, dirty = git_state(Path.cwd())
    unseen_eligible = result["unseen"] is not None
    report = {
        "schema_version": 2,
        "status": "completed_strict_100_30_evaluator_v2",
        "run_id": f"{old_metrics['run_id']}-strict-v2",
        "started_at": started_at,
        "ended_at": utc_now(),
        "category": split["category"],
        "seed": split["seed"],
        "protocol": "strict_100_normal_30_or_available_anomaly_support",
        "split_hash": split["split_hash"],
        "evaluator_config": str(args.evaluator_config),
        "evaluator_config_sha256": file_sha256(args.evaluator_config),
        "training_config_sha256": file_sha256(args.training_config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "git_commit": commit,
        "dirty": dirty,
        "calibration": {
            "normal_count": len(normal_rows),
            "support_anomaly_count": len(anomaly_rows),
            "development_count": 0,
            "threshold": threshold,
            "image_score": "upstream_amax",
            "map_quantiles_recomputed_from_final_weights": True,
        },
        "result": result,
        "unseen_eligible": unseen_eligible,
        "latency_model_segment_ms": {
            "p50": float(np.quantile(latencies, 0.50)),
            "p95": float(np.quantile(latencies, 0.95)),
            "p99": float(np.quantile(latencies, 0.99)),
            "max": float(np.max(latencies)),
            "scope": f"{device}, 256x256 model segment diagnostic only",
        },
        "leakage_audit": {
            "test_label_reads_before_all_predictions_fixed": 0,
            "test_labels_used_for_training": 0,
            "test_labels_used_for_threshold": 0,
            "test_labels_used_for_model_selection": 0,
            "development_anomaly_rows_used_for_threshold": 0,
        },
        "source_hashes": {
            "adaptation": file_sha256(adaptation_path),
            "test_inputs": file_sha256(test_inputs_path),
            "test_truth": file_sha256(truth_path),
            "old_diagnostic_metrics": file_sha256(old_metrics_path),
        },
        "hardware": platform.platform(),
        "physical_gpu": os.environ.get("EVOINSPECT_PHYSICAL_GPU", "CPU_OR_UNRECORDED"),
        "warnings": [
            "Checkpoint weights are reused without retraining or test-driven model selection.",
            "Support anomalies calibrate only the fixed threshold and never update model weights.",
            "This 256x256 timing is not GTX2060 or 2500x2500 evidence.",
        ],
    }
    write_json(output / "metrics.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--run-dir", required=True, type=Path)
    value.add_argument("--training-config", required=True, type=Path)
    value.add_argument("--evaluator-config", required=True, type=Path)
    value.add_argument("--device", default="cpu")
    return value


if __name__ == "__main__":
    payload = run(parser().parse_args())
    print(json.dumps({"run_id": payload["run_id"], "result": payload["result"]["overall"]}))
