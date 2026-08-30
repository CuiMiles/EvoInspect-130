#!/usr/bin/env python3
"""Evaluate preregistered HeteroCal-130 using frozen EfficientAD-M checkpoints."""

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

from evoinspect.evaluation import binary_metrics
from evoinspect.heterocal import (
    FloatArray,
    choose_threshold,
    fit_nonnegative_residual,
    leave_one_defect_type_out,
)
from evoinspect.provenance import file_sha256, git_state, utc_now, write_json
from scripts.efficientad_rcbr_100_30 import read_csv
from scripts.evaluate_efficientad_strict_100_30 import (
    image_tensor,
    load_fixed_model,
    strict_calibration_rows,
)


def top_mean(values: NDArray[np.float32], fraction: float) -> float:
    flat = values.reshape(-1)
    count = max(1, int(np.ceil(flat.size * fraction)))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


def map_features(
    map_st: NDArray[np.float32],
    map_ae: NDArray[np.float32],
    fractions: list[float],
    area_threshold: float,
) -> list[float]:
    fused = 0.5 * map_st + 0.5 * map_ae
    features = [float(np.max(fused)), float(np.max(map_st)), float(np.max(map_ae))]
    for anomaly_map in (fused, map_st, map_ae):
        features.extend(top_mean(anomaly_map, fraction) for fraction in fractions)
    features.extend(
        [
            float(np.mean(fused > area_threshold)),
            float(np.mean(map_st > area_threshold)),
            float(np.mean(map_ae > area_threshold)),
            float(np.mean((map_st > area_threshold) & (map_ae > area_threshold))),
            float(np.mean(np.abs(map_st - map_ae))),
            float(np.max(np.abs(map_st - map_ae))),
        ]
    )
    return features


def infer_features(
    model: Any,
    row: dict[str, str],
    shape: tuple[int, int],
    device: torch.device,
    fractions: list[float],
    area_threshold: float,
) -> tuple[FloatArray, FloatArray, float]:
    tensor = image_tensor(row["path"], shape, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        original_st, original_ae = model.model.get_maps(tensor, normalize=True)
        flipped_st, flipped_ae = model.model.get_maps(
            torch.flip(tensor, dims=(-1,)), normalize=True
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    original_st_np = original_st.float().cpu().squeeze().numpy().astype(np.float32)
    original_ae_np = original_ae.float().cpu().squeeze().numpy().astype(np.float32)
    flipped_st_np = (
        np.flip(flipped_st.float().cpu().squeeze().numpy(), axis=-1).copy().astype(np.float32)
    )
    flipped_ae_np = (
        np.flip(flipped_ae.float().cpu().squeeze().numpy(), axis=-1).copy().astype(np.float32)
    )
    original = map_features(original_st_np, original_ae_np, fractions, area_threshold)
    aligned_st = 0.5 * (original_st_np + flipped_st_np)
    aligned_ae = 0.5 * (original_ae_np + flipped_ae_np)
    flip = map_features(aligned_st, aligned_ae, fractions, area_threshold)
    flip.extend(
        [
            float(np.mean(np.abs(original_st_np - flipped_st_np))),
            float(np.mean(np.abs(original_ae_np - flipped_ae_np))),
        ]
    )
    return np.asarray(original, dtype=np.float64), np.asarray(flip, dtype=np.float64), elapsed_ms


def slice_metrics(
    rows: list[dict[str, str]], scores: FloatArray, threshold: float
) -> dict[str, float] | None:
    labels = np.asarray([int(row["label"] == "anomaly") for row in rows], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        return None
    return binary_metrics(
        labels.tolist(), scores.tolist(), (scores >= threshold).astype(int).tolist()
    )


def image_result(
    rows: list[dict[str, str]], scores: FloatArray, threshold: float
) -> dict[str, Any]:
    overall = slice_metrics(rows, scores, threshold)
    if overall is None:
        raise RuntimeError("overall test slice lacks both classes")
    seen_indexes = [
        index
        for index, row in enumerate(rows)
        if row["label"] == "normal" or row.get("defect_visibility") == "seen"
    ]
    unseen_indexes = [
        index
        for index, row in enumerate(rows)
        if row["label"] == "normal" or row.get("defect_visibility") == "unseen"
    ]
    return {
        "overall": overall,
        "seen": slice_metrics([rows[i] for i in seen_indexes], scores[seen_indexes], threshold),
        "unseen": slice_metrics(
            [rows[i] for i in unseen_indexes], scores[unseen_indexes], threshold
        ),
    }


def fit_head(features: FloatArray, labels: NDArray[np.int64], config: dict[str, Any]) -> Any:
    head_config = config["calibrator"]
    return fit_nonnegative_residual(
        features,
        labels,
        features[labels == 0],
        delta=float(head_config["residual_delta"]),
        l2=float(head_config["l2"]),
        learning_rate=float(head_config["learning_rate"]),
        iterations=int(head_config["iterations"]),
        scale_floor=float(head_config["normal_scale_floor"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = utc_now()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evaluator = yaml.safe_load(args.evaluator_config.read_text(encoding="utf-8"))
    training = yaml.safe_load(args.training_config.read_text(encoding="utf-8"))
    if evaluator["protocol_id"] != "efficientad_strict_100_30_evaluator_v2_1":
        raise RuntimeError("HeteroCal requires the frozen strict-v2.1 source evaluator")
    old_metrics_path = args.run_dir / "result" / "metrics.json"
    old_metrics = json.loads(old_metrics_path.read_text(encoding="utf-8"))
    checkpoint = args.run_dir / "result" / "model.ckpt"
    adaptation = read_csv(args.run_dir / "adaptation.csv")
    normal_rows, anomaly_rows = strict_calibration_rows(
        adaptation, int(old_metrics["calibration"]["normal_train_count"])
    )
    support_rows = [*normal_rows, *anomaly_rows]
    support_labels = np.asarray(
        [int(row["label"] == "anomaly") for row in support_rows], dtype=np.int64
    )
    defect_types = [row["defect_type"] for row in support_rows]
    output = args.run_dir / args.output_name
    if output.exists():
        raise FileExistsError(output)
    temporary = args.run_dir / f".{args.output_name}.inprogress-{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    shape = tuple(int(value) for value in training["input_resolution"])
    model = load_fixed_model(
        checkpoint,
        device,
        str(old_metrics["training"]["model_sha256"]),
        normal_rows,
        shape,
    )
    fractions = [float(value) for value in config["features"]["top_fractions"]]
    area_threshold = float(config["features"]["normalized_map_area_threshold"])
    support_original: list[FloatArray] = []
    support_flip: list[FloatArray] = []
    for row in support_rows:
        original, flip, _ = infer_features(model, row, shape, device, fractions, area_threshold)
        support_original.append(original)
        support_flip.append(flip)
    support_original_array = np.stack(support_original)
    support_flip_array = np.stack(support_flip)
    support_full = np.concatenate((support_original_array, support_flip_array), axis=1)
    original_selection = leave_one_defect_type_out(
        support_original_array, support_labels, defect_types, config
    )
    full_selection = leave_one_defect_type_out(support_full, support_labels, defect_types, config)
    original_head = fit_head(support_original_array, support_labels, config)
    full_head = fit_head(support_full, support_labels, config)

    def support_and_thresholds() -> tuple[dict[str, FloatArray], dict[str, float]]:
        base = support_original_array[:, 0]
        dual = np.maximum(support_original_array[:, 1], support_original_array[:, 2])
        flip = support_flip_array[:, 0]
        calibrated = original_head.score(support_original_array)
        complete = full_head.score(support_full)
        values = {
            "efficientad_m": base,
            "dual_map_fixed": dual,
            "defect_type_loo_calibration": calibrated if original_selection["accepted"] else base,
            "flip_consistency": flip,
            "heterocal_130": complete if full_selection["accepted"] else base,
        }
        return values, {
            name: choose_threshold(scores, support_labels) for name, scores in values.items()
        }

    support_scores, thresholds = support_and_thresholds()
    test_inputs = read_csv(args.run_dir / "test_inputs.csv")
    if any(row["label"] or row["defect_type"] for row in test_inputs):
        raise RuntimeError("test input manifest exposes labels")
    test_original: list[FloatArray] = []
    test_flip: list[FloatArray] = []
    latencies: list[float] = []
    for row in test_inputs:
        original, flip, elapsed = infer_features(
            model, row, shape, device, fractions, area_threshold
        )
        test_original.append(original)
        test_flip.append(flip)
        latencies.append(elapsed)
    test_original_array = np.stack(test_original)
    test_flip_array = np.stack(test_flip)
    test_full = np.concatenate((test_original_array, test_flip_array), axis=1)
    base = test_original_array[:, 0]
    test_scores = {
        "efficientad_m": base,
        "dual_map_fixed": np.maximum(test_original_array[:, 1], test_original_array[:, 2]),
        "defect_type_loo_calibration": original_head.score(test_original_array)
        if original_selection["accepted"]
        else base,
        "flip_consistency": test_flip_array[:, 0],
        "heterocal_130": full_head.score(test_full) if full_selection["accepted"] else base,
    }
    predictions_path = temporary / "predictions.jsonl"
    with predictions_path.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(test_inputs):
            stream.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "scores": {
                            name: float(scores[index]) for name, scores in test_scores.items()
                        },
                        "decisions": {
                            name: int(scores[index] >= thresholds[name])
                            for name, scores in test_scores.items()
                        },
                        "dual_forward_model_ms": latencies[index],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    # Truth is opened only after all five strategies' test scores and decisions are durable.
    truth_path = args.run_dir / "test_truth.csv"
    truth_by_id = {row["sample_id"]: row for row in read_csv(truth_path)}
    truth_rows = [truth_by_id[row["sample_id"]] for row in test_inputs]
    results = {
        name: image_result(truth_rows, scores, thresholds[name])
        for name, scores in test_scores.items()
    }
    split = json.loads((args.run_dir / "split.json").read_text(encoding="utf-8"))
    commit, dirty = git_state(Path.cwd())
    report = {
        "schema_version": 1,
        "status": "completed_heterocal_130_preregistered_v1",
        "run_id": f"{old_metrics['run_id']}-heterocal-v1",
        "started_at": started_at,
        "ended_at": utc_now(),
        "category": split["category"],
        "seed": split["seed"],
        "split_hash": split["split_hash"],
        "git_commit": commit,
        "dirty": dirty,
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "checkpoint_sha256": file_sha256(checkpoint),
        "support": {
            "normal_count": len(normal_rows),
            "anomaly_count": len(anomaly_rows),
            "defect_types": sorted(set(row["defect_type"] for row in anomaly_rows)),
            "thresholds": thresholds,
            "original_selection": original_selection,
            "full_selection": full_selection,
            "scores_sha256": "stored_in_report_only",
        },
        "results": results,
        "dual_forward_model_segment_ms": {
            "p50": float(np.quantile(latencies, 0.50)),
            "p95": float(np.quantile(latencies, 0.95)),
            "p99": float(np.quantile(latencies, 0.99)),
            "max": float(np.max(latencies)),
            "scope": f"{device}, two 256x256 forward passes, model segment diagnostic",
        },
        "leakage_audit": {
            "test_label_reads_before_all_predictions_fixed": 0,
            "test_labels_used_for_training": 0,
            "test_labels_used_for_threshold": 0,
            "test_labels_used_for_model_selection": 0,
        },
        "source_hashes": {
            "adaptation": file_sha256(args.run_dir / "adaptation.csv"),
            "test_inputs": file_sha256(args.run_dir / "test_inputs.csv"),
            "test_truth": file_sha256(truth_path),
            "old_metrics": file_sha256(old_metrics_path),
        },
        "hardware": platform.platform(),
        "physical_gpu": os.environ.get("EVOINSPECT_PHYSICAL_GPU", "CPU_OR_UNRECORDED"),
        "warnings": [
            "HeteroCal is support-only calibration over a frozen upstream checkpoint.",
            "Per-run timing is not the final GTX2060 2500x2500 end-to-end benchmark.",
        ],
    }
    # Keep support scores for reproducible threshold auditing, never test truth in this archive.
    np.savez_compressed(
        temporary / "support_features.npz",
        original=support_original_array,
        flip=support_flip_array,
        labels=support_labels,
        **{f"score_{name}": value for name, value in support_scores.items()},
    )
    write_json(temporary / "metrics.json", report)
    temporary.rename(output)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--run-dir", required=True, type=Path)
    value.add_argument("--config", required=True, type=Path)
    value.add_argument("--training-config", required=True, type=Path)
    value.add_argument("--evaluator-config", required=True, type=Path)
    value.add_argument("--output-name", default="heterocal_result")
    value.add_argument("--device", default="cuda:0")
    return value


if __name__ == "__main__":
    payload = run(parser().parse_args())
    print(json.dumps({"run_id": payload["run_id"], "results": payload["results"]}))
