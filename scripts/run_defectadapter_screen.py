#!/usr/bin/env python3
"""Run bounded EfficientAD-S last-layer defect adaptation on frozen manifests."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from evoinspect.baseline import select_threshold
from evoinspect.provenance import file_sha256, git_state, utc_now, write_json
from scripts.efficientad_rcbr_100_30 import evaluate_strategy, read_csv
from scripts.evaluate_efficientad_strict_100_30 import (
    image_tensor,
    load_fixed_model,
    strict_calibration_rows,
    zero_audit,
)


def differentiable_map(model: Any, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        teacher = model.model.teacher(images)
        if model.model.is_set(model.model.mean_std):
            teacher = (teacher - model.model.mean_std["mean"]) / model.model.mean_std["std"]
        autoencoder = model.model.ae(images, images.shape[-2:])
    student = model.model.student(images)
    channels = model.model.teacher_out_channels
    map_st = torch.mean((teacher - student[:, :channels]) ** 2, dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder - student[:, channels:]) ** 2, dim=1, keepdim=True)
    if model.model.pad_maps:
        map_st = F.pad(map_st, (4, 4, 4, 4))
        map_ae = F.pad(map_ae, (4, 4, 4, 4))
    map_st = F.interpolate(map_st, images.shape[-2:], mode="bilinear")
    map_ae = F.interpolate(map_ae, images.shape[-2:], mode="bilinear")
    quantiles = model.model.quantiles
    map_st = 0.1 * (map_st - quantiles["qa_st"]) / (quantiles["qb_st"] - quantiles["qa_st"])
    map_ae = 0.1 * (map_ae - quantiles["qa_ae"]) / (quantiles["qb_ae"] - quantiles["qa_ae"])
    return 0.5 * map_st + 0.5 * map_ae


def image_score(anomaly_map: torch.Tensor) -> torch.Tensor:
    flat = anomaly_map.flatten(1)
    count = max(1, int(flat.shape[1] * 0.001))
    return torch.topk(flat, count, dim=1).values.mean(dim=1)


def balanced_batches(
    normals: list[dict[str, str]],
    anomalies: list[dict[str, str]],
    batch_size: int,
    generator: random.Random,
) -> list[list[dict[str, str]]]:
    steps = max(1, int(np.ceil((len(normals) + len(anomalies)) / batch_size)))
    half = batch_size // 2
    return [
        generator.choices(normals, k=half) + generator.choices(anomalies, k=batch_size - half)
        for _ in range(steps)
    ]


def infer(
    model: Any, rows: list[dict[str, str]], shape: tuple[int, int], device: torch.device
) -> tuple[list[float], list[np.ndarray], list[float]]:
    model.eval()
    scores: list[float] = []
    maps: list[np.ndarray] = []
    latencies: list[float] = []
    with torch.inference_mode():
        for row in rows:
            image = image_tensor(row["path"], shape, device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            anomaly_map = differentiable_map(model, image)
            torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000)
            scores.append(float(image_score(anomaly_map)[0].cpu()))
            maps.append(anomaly_map[0, 0].float().cpu().numpy())
    return scores, maps, latencies


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    route = config["routes"]["defectadapter"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    old_metrics = json.loads(
        (args.source_run / "result" / "metrics.json").read_text(encoding="utf-8")
    )
    adaptation = read_csv(args.source_run / "adaptation.csv")
    train_count = int(old_metrics["calibration"]["normal_train_count"])
    train_normals = [row for row in adaptation if row["role"] == "support_normal"][:train_count]
    calibration_normals, anomalies = strict_calibration_rows(adaptation, train_count)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    shape = tuple(int(value) for value in config["input_resolution"])
    checkpoint = args.source_run / "result" / "model.ckpt"
    model = load_fixed_model(
        checkpoint, device, str(old_metrics["training"]["model_sha256"]), calibration_normals, shape
    )
    original_student = copy.deepcopy(model.model.student).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.model.student.conv4.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        model.model.student.conv4.parameters(), lr=float(route["learning_rate"])
    )
    with torch.inference_mode():
        base_scores, _, _ = infer(model, calibration_normals, shape, device)
    center = float(np.median(base_scores))
    scale = max(float(np.subtract(*np.percentile(base_scores, [75, 25]))), 1e-4)
    generator = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model.train()
    for _epoch in range(int(route["epochs"])):
        for rows in balanced_batches(train_normals, anomalies, int(route["batch_size"]), generator):
            images = torch.cat([image_tensor(row["path"], shape, device) for row in rows])
            labels = torch.tensor(
                [row["label"] == "anomaly" for row in rows], device=device
            ).float()
            anomaly_map = differentiable_map(model, images)
            scores = image_score(anomaly_map)
            classification = F.binary_cross_entropy_with_logits((scores - center) / scale, labels)
            normal_mask = labels == 0
            with torch.no_grad():
                anchor = original_student(images[normal_mask])
            current = model.model.student(images[normal_mask])
            retention = F.mse_loss(current, anchor)
            loss = classification + float(route["anchor_weight"]) * retention
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.student.conv4.parameters(), 1.0)
            optimizer.step()
    calibration_rows = [*calibration_normals, *anomalies]
    calibration_scores, _, _ = infer(model, calibration_rows, shape, device)
    threshold = select_threshold(
        calibration_scores, [int(row["label"] == "anomaly") for row in calibration_rows]
    )
    test_inputs = read_csv(args.source_run / "test_inputs.csv")
    if any(row["label"] or row["defect_type"] for row in test_inputs):
        raise RuntimeError("test input manifest exposes labels")
    test_scores, test_maps, latencies = infer(model, test_inputs, shape, device)
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row, score in zip(test_inputs, test_scores, strict=True):
            stream.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "score": score,
                        "decision": int(score >= threshold["threshold"]),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    np.savez_compressed(output / "prediction_maps.npz", predictions=np.stack(test_maps))
    truth_by_id = {row["sample_id"]: row for row in read_csv(args.source_run / "test_truth.csv")}
    truth_rows = [truth_by_id[row["sample_id"]] for row in test_inputs]
    result = evaluate_strategy(
        truth_rows,
        test_maps,
        test_scores,
        float(threshold["threshold"]),
        [zero_audit() for _ in test_maps],
    )
    split = json.loads((args.source_run / "split.json").read_text(encoding="utf-8"))
    commit, dirty = git_state(Path.cwd())
    metrics = {
        "schema_version": 1,
        "status": "completed_additional_route_screen",
        "route": "defectadapter",
        "run_id": args.run_id,
        "started_at": args.started_at,
        "ended_at": utc_now(),
        "category": split["category"],
        "seed": args.seed,
        "git_commit": commit,
        "dirty": dirty,
        "source_run": str(args.source_run.resolve()),
        "source_split_hash": split["split_hash"],
        "source_checkpoint_sha256": file_sha256(checkpoint),
        "threshold_support_only": threshold,
        "result": result,
        "latency_model_segment_ms": {
            "p50": float(np.quantile(latencies, 0.5)),
            "p95": float(np.quantile(latencies, 0.95)),
            "max": float(np.max(latencies)),
        },
        "leakage_audit": {
            "test_label_reads_before_all_predictions_fixed": 0,
            "test_labels_used_for_training": 0,
            "test_labels_used_for_threshold": 0,
            "test_labels_used_for_model_selection": 0,
        },
        "warnings": [
            "Exploratory six-category seed143 screen only.",
            (
                "Only EfficientAD-S student conv4 was adapted; teacher and autoencoder "
                "remained frozen."
            ),
        ],
    }
    torch.save(model.model.student.conv4.state_dict(), output / "adapter.pt")
    write_json(output / "metrics.json", metrics)
    return metrics


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--source-run", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/additional_routes_screen_20260831.yaml"),
    )
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--seed", type=int, default=143)
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
