#!/usr/bin/env python3
"""Train and evaluate one bounded Dinomaly category under frozen support/test manifests."""

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
from PIL import Image

from evoinspect.baseline import select_threshold
from evoinspect.provenance import file_sha256, git_state, utc_now, write_json
from scripts.efficientad_rcbr_100_30 import evaluate_strategy, read_csv
from scripts.evaluate_efficientad_strict_100_30 import zero_audit


def find_run(source_batch: Path, category: str, seed: int) -> Path:
    candidates = sorted(
        path
        for path in source_batch.iterdir()
        if path.is_dir() and f"-{category}-s{seed}-" in path.name
    )
    if len(candidates) != 1:
        raise RuntimeError(f"expected one frozen run for {category}/seed{seed}, found {candidates}")
    return candidates[0]


def load_image(path: str | Path, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).float().div_(255.0)
    tensor = torch.nn.functional.interpolate(
        tensor[None], size=shape, mode="bicubic", align_corners=False, antialias=True
    )
    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((tensor - mean) / std).to(device)


def infer(
    model: Any,
    row: dict[str, str],
    shape: tuple[int, int],
    device: torch.device,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float, float]:
    image = load_image(row["path"], shape, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = model.model(image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    score = float(prediction.pred_score.detach().float().cpu().reshape(-1)[0])
    anomaly_map = np.asarray(
        prediction.anomaly_map.detach().float().cpu().squeeze().numpy(), dtype=np.float32
    )
    if not np.isfinite(score) or not np.isfinite(anomaly_map).all():
        raise RuntimeError("Dinomaly produced a non-finite score or map")
    return anomaly_map, score, elapsed_ms


def create_model(config: dict[str, Any], device: torch.device) -> Any:
    from anomalib.models.image.dinomaly.lightning_model import Dinomaly

    model_config = config["model"]
    model = Dinomaly(
        encoder_name=str(model_config["encoder_name"]),
        decoder_depth=int(model_config["decoder_depth"]),
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    ).to(device)
    return model


def train_model(
    model: Any,
    rows: list[dict[str, str]],
    shape: tuple[int, int],
    device: torch.device,
    config: dict[str, Any],
    log_path: Path,
) -> None:
    model.train()
    model.model.encoder.eval()
    trainable = [
        parameter
        for parameter in model.trainable_modules.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    steps = int(config["model"]["max_steps"])
    clip_norm = float(config["model"]["gradient_clip_norm"])
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(steps):
            image = load_image(rows[step % len(rows)]["path"], shape, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.model(image, global_step=step)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite Dinomaly loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, clip_norm)
            optimizer.step()
            if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
                log.write(json.dumps({"step": step + 1, "loss": float(loss.detach().cpu())}) + "\n")
                log.flush()
    model.eval()
    model.model.encoder.eval()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    category = str(args.category)
    seed = int(config["seed"])
    if category not in set(config["categories"]):
        raise RuntimeError(f"category {category} is outside preregistered screen")
    torch.manual_seed(seed)
    np.random.seed(seed)
    run_dir = find_run(Path(config["source_batch"]), category, seed)
    adaptation_path = run_dir / "adaptation.csv"
    test_inputs_path = run_dir / "test_inputs.csv"
    truth_path = run_dir / "test_truth.csv"
    adaptation = read_csv(adaptation_path)
    support_normals = [row for row in adaptation if row["role"] == "support_normal"]
    support_anomalies = [row for row in adaptation if row["role"] == "support_anomaly"]
    train_count = int(config["normal_train_count"])
    if len(support_normals) <= train_count or not support_anomalies:
        raise RuntimeError("frozen manifest lacks required support rows")
    train_normals = support_normals[:train_count]
    calibration_normals = support_normals[train_count:]
    test_inputs = read_csv(test_inputs_path)
    if any(row["label"] or row["defect_type"] for row in test_inputs):
        raise RuntimeError("test input manifest exposes labels")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_root = args.output_root / category
    if output_root.exists():
        raise FileExistsError(output_root)
    inprogress = args.output_root / f".{category}.inprogress-{os.getpid()}"
    inprogress.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    shape = tuple(int(value) for value in config["model"]["image_size"])
    model = create_model(config, device)
    train_model(model, train_normals, shape, device, config, inprogress / "train.log")
    torch.save(model.model.state_dict(), inprogress / "model_state.pt")

    calibration_rows = [*calibration_normals, *support_anomalies]
    calibration_scores = [infer(model, row, shape, device)[1] for row in calibration_rows]
    threshold = select_threshold(
        calibration_scores, [int(row["label"] == "anomaly") for row in calibration_rows]
    )
    maps: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    scores: list[float] = []
    latencies: list[float] = []
    for row in test_inputs:
        anomaly_map, score, elapsed_ms = infer(model, row, shape, device)
        maps.append(anomaly_map)
        scores.append(score)
        latencies.append(elapsed_ms)

    with (inprogress / "predictions.jsonl").open("x", encoding="utf-8") as stream:
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
    np.savez_compressed(inprogress / "prediction_maps.npz", predictions=np.stack(maps))

    # Open final truth only after all predictions and maps have been persisted.
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
    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    commit, dirty = git_state(Path.cwd())
    report = {
        "schema_version": 1,
        "status": "completed_dinomaly_six_category_screen",
        "run_id": f"dinomaly-{category}-s{seed}-20260901",
        "started_at": started_at,
        "ended_at": utc_now(),
        "category": split["category"],
        "seed": split["seed"],
        "protocol": "frozen_80_normal_train_plus_20_normal_and_30_anomaly_threshold_calibration",
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "source_run": str(run_dir),
        "source_split_hash": split["split_hash"],
        "source_adaptation_sha256": file_sha256(adaptation_path),
        "source_test_inputs_sha256": file_sha256(test_inputs_path),
        "source_test_truth_sha256": file_sha256(truth_path),
        "normal_train_count": len(train_normals),
        "normal_calibration_count": len(calibration_normals),
        "anomaly_calibration_count": len(support_anomalies),
        "threshold": threshold,
        "result": result,
        "latency_model_segment_ms": {
            "p50": float(np.quantile(latencies, 0.50)),
            "p95": float(np.quantile(latencies, 0.95)),
            "p99": float(np.quantile(latencies, 0.99)),
            "max": float(np.max(latencies)),
            "scope": f"{device}, {shape[0]}x{shape[1]} Dinomaly model segment",
        },
        "git_commit": commit,
        "dirty": dirty,
        "hardware": platform.platform(),
        "physical_gpu": os.environ.get("EVOINSPECT_PHYSICAL_GPU", "CPU_OR_UNRECORDED"),
        "leakage_audit": {
            "test_label_reads_before_all_predictions_fixed": 0,
            "test_labels_used_for_training": 0,
            "test_labels_used_for_threshold": 0,
            "test_labels_used_for_model_selection": 0,
            "development_rows_used": 0,
        },
        "claim_limit": (
            "Upstream Dinomaly exploratory comparison with a bounded small encoder; "
            "not an originality claim or production benchmark."
        ),
    }
    write_json(inprogress / "metrics.json", report)
    inprogress.rename(output_root)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, required=True)
    report = run(parser.parse_args())
    print(
        json.dumps(
            {"category": report["category"], "result": report["result"]["overall"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
