#!/usr/bin/env python3
"""Run bounded SuperSimpleNet or DRA screens on frozen 100+30 manifests."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from evoinspect.baseline import select_threshold
from evoinspect.provenance import file_sha256, git_state, utc_now, write_json
from scripts.efficientad_rcbr_100_30 import evaluate_strategy, read_csv
from scripts.evaluate_efficientad_strict_100_30 import strict_calibration_rows, zero_audit


def image_tensor(path: str | Path, size: int, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    value = torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).float().div_(255.0)
    value = F.interpolate(
        value[None], (size, size), mode="bilinear", align_corners=False, antialias=True
    )[0]
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((value - mean) / std).to(device)


def mask_tensor(row: dict[str, str], size: int, device: torch.device) -> torch.Tensor:
    if row["label"] != "anomaly" or not row.get("mask_path"):
        return torch.zeros(1, size, size, device=device)
    with Image.open(row["mask_path"]) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    value = torch.from_numpy(np.array(array, copy=True)).float().div_(255.0)[None, None]
    return (F.interpolate(value, (size, size), mode="nearest")[0] > 0.5).float().to(device)


def support_rows(
    source: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    metrics = json.loads((source / "result" / "metrics.json").read_text(encoding="utf-8"))
    rows = read_csv(source / "adaptation.csv")
    train_count = int(metrics["calibration"]["normal_train_count"])
    all_normals = [row for row in rows if row["role"] == "support_normal"]
    calibration_normals, anomalies = strict_calibration_rows(rows, train_count)
    return all_normals[:train_count], calibration_normals, anomalies


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, gamma: float = 4.0) -> torch.Tensor:
    ce = F.binary_cross_entropy(inputs.float(), targets.float(), reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    return (ce * ((1 - p_t) ** gamma)).mean()


def train_ssn(
    train_normals: list[dict[str, str]],
    anomalies: list[dict[str, str]],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> Any:
    upstream = Path("/home/CuiMinghao/models/upstream/SuperSimpleNet")
    sys.path.insert(0, str(upstream))
    from model.supersimplenet import SuperSimpleNet  # type: ignore[import-not-found]

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    size = int(config["input_resolution"][0])
    route = config["routes"]["supersimplenet"]
    model_config = {
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "patch_size": 3,
        "noise": True,
        "perlin": True,
        "no_anomaly": "empty",
        "bad": True,
        "overlap": False,
        "adapt_cls_feat": False,
        "noise_std": 0.015,
        "perlin_thr": 0.6,
        "seg_lr": 0.0002,
        "dec_lr": 0.0002,
        "adapt_lr": 0.0001,
        "gamma": 0.4,
        "stop_grad": False,
        "epochs": int(route["epochs"]),
    }
    model = SuperSimpleNet((size, size), model_config).to(device)
    optimizer, scheduler = model.get_optimizers()
    rows = [*train_normals, *anomalies]
    batch_size = int(route["batch_size"])
    for epoch in range(int(route["epochs"])):
        model.train()
        generator = random.Random(seed + epoch)
        epoch_rows = rows.copy()
        generator.shuffle(epoch_rows)
        for offset in range(0, len(epoch_rows), batch_size):
            selected = epoch_rows[offset : offset + batch_size]
            images = torch.stack([image_tensor(row["path"], size, device) for row in selected])
            masks = torch.stack([mask_tensor(row, size, device) for row in selected])
            labels = torch.tensor(
                [row["label"] == "anomaly" for row in selected], device=device
            ).float()
            feature_masks = F.interpolate(masks, size=(model.fh, model.fw), mode="nearest")
            anomaly_map, score, target_masks, target_labels = model(images, feature_masks, labels)
            seg = focal_loss(torch.sigmoid(anomaly_map), target_masks)
            normal = anomaly_map[target_masks == 0]
            bad = anomaly_map[target_masks > 0]
            margin = torch.clamp(normal + 0.5, min=0).mean()
            if bad.numel():
                margin = margin + torch.clamp(-bad + 0.5, min=0).mean()
            loss = seg + margin + focal_loss(torch.sigmoid(score), target_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    return model


@torch.inference_mode()
def infer_ssn(
    model: Any, rows: list[dict[str, str]], size: int, device: torch.device
) -> tuple[list[float], list[np.ndarray], list[float]]:
    model.eval()
    scores: list[float] = []
    maps: list[np.ndarray] = []
    times: list[float] = []
    for row in rows:
        image = image_tensor(row["path"], size, device)[None]
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        anomaly_map, score = model(image)
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - started) * 1000)
        scores.append(float(torch.sigmoid(score)[0].cpu()))
        maps.append(torch.sigmoid(anomaly_map)[0, 0].float().cpu().numpy())
    return scores, maps, times


def pseudo_image(image: torch.Tensor, generator: random.Random) -> torch.Tensor:
    _, height, width = image.shape
    h = max(8, height // 4)
    w = max(8, width // 4)
    y = generator.randrange(0, height - h + 1)
    x = generator.randrange(0, width - w + 1)
    output = image.clone()
    output[:, y : y + h, x : x + w] = torch.flip(output[:, y : y + h, x : x + w], dims=[2])
    return output


def train_dra(
    train_normals: list[dict[str, str]],
    anomalies: list[dict[str, str]],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[Any, torch.Tensor]:
    upstream = Path("/home/CuiMinghao/models/upstream/DRA")
    sys.path.insert(0, str(upstream))
    from modeling.net import DRA  # type: ignore[import-not-found]

    route = config["routes"]["dra"]
    size = int(config["input_resolution"][0])
    cfg = SimpleNamespace(
        img_size=size,
        total_heads=4,
        n_scales=2,
        nRef=int(route["n_ref"]),
        topk=float(route["topk"]),
    )
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = DRA(cfg, backbone="resnet18").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    refs = torch.stack(
        [image_tensor(row["path"], size, device) for row in train_normals[: cfg.nRef]]
    )
    generator = random.Random(seed)
    batch_size = int(route["batch_size"])
    normal_n = batch_size // 2
    anomaly_n = batch_size // 4
    for _epoch in range(int(route["epochs"])):
        model.train()
        for _ in range(int(route["steps_per_epoch"])):
            normal_rows = generator.choices(
                train_normals, k=normal_n + (batch_size - normal_n - anomaly_n)
            )
            anomaly_rows = generator.choices(anomalies, k=anomaly_n)
            normal_images = [image_tensor(row["path"], size, device) for row in normal_rows]
            real = [image_tensor(row["path"], size, device) for row in anomaly_rows]
            pseudo_count = batch_size - normal_n - anomaly_n
            targets = (
                normal_images[:normal_n]
                + real
                + [
                    pseudo_image(value, generator)
                    for value in normal_images[normal_n : normal_n + pseudo_count]
                ]
            )
            labels = torch.tensor(
                [0] * normal_n + [1] * anomaly_n + [2] * pseudo_count, device=device
            )
            outputs = model(torch.cat([refs, torch.stack(targets)]), labels)
            normal_target = (labels == 0).float()
            anomaly_target = (labels != 0).float()
            losses = []
            for index, output in enumerate(outputs):
                prediction = -output if index == 0 else output
                target = normal_target if index == 0 else anomaly_target
                if index == 1:
                    keep = labels != 2
                    prediction = prediction[keep]
                    target = anomaly_target[keep]
                elif index == 2:
                    keep = labels != 1
                    prediction = prediction[keep]
                    target = anomaly_target[keep]
                losses.append(F.binary_cross_entropy_with_logits(prediction, target))
            loss = sum(losses)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    return model, refs


@torch.inference_mode()
def infer_dra(
    model: Any, refs: torch.Tensor, rows: list[dict[str, str]], size: int, device: torch.device
) -> tuple[list[float], list[np.ndarray], list[float]]:
    model.eval()
    scores = []
    maps = []
    times = []
    for row in rows:
        image = image_tensor(row["path"], size, device)[None]
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        outputs = model(torch.cat([refs, image]), torch.zeros(1, dtype=torch.long, device=device))
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - started) * 1000)
        scores.append(float((-outputs[0] + outputs[1] + outputs[2] + outputs[3])[0].cpu()))
        maps.append(np.zeros((size, size), dtype=np.float32))
    return scores, maps, times


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    train_normals, calibration_normals, anomalies = support_rows(args.source_run)
    if args.route == "supersimplenet":
        model = train_ssn(train_normals, anomalies, config, device, args.seed)

        def infer(rows: list[dict[str, str]]) -> tuple[list[float], list[np.ndarray], list[float]]:
            return infer_ssn(model, rows, int(config["input_resolution"][0]), device)

    else:
        model, refs = train_dra(train_normals, anomalies, config, device, args.seed)

        def infer(rows: list[dict[str, str]]) -> tuple[list[float], list[np.ndarray], list[float]]:
            return infer_dra(model, refs, rows, int(config["input_resolution"][0]), device)

    calibration_rows = [*calibration_normals, *anomalies]
    calibration_scores, _, _ = infer(calibration_rows)
    threshold = select_threshold(
        calibration_scores, [int(row["label"] == "anomaly") for row in calibration_rows]
    )
    test_inputs = read_csv(args.source_run / "test_inputs.csv")
    if any(row["label"] or row["defect_type"] for row in test_inputs):
        raise RuntimeError("test input manifest exposes labels")
    test_scores, test_maps, latencies = infer(test_inputs)
    prediction_path = output / "predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8") as stream:
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
        "route": args.route,
        "run_id": args.run_id,
        "started_at": args.started_at,
        "ended_at": utc_now(),
        "category": split["category"],
        "seed": args.seed,
        "git_commit": commit,
        "dirty": dirty,
        "source_run": str(args.source_run.resolve()),
        "source_split_hash": split["split_hash"],
        "source_adaptation_sha256": file_sha256(args.source_run / "adaptation.csv"),
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
            "Six-category seed143 exploratory screen only.",
            (
                "DRA produces image scores only; localization maps are zero and localization "
                "metrics are not claim eligible."
            )
            if args.route == "dra"
            else "Official SuperSimpleNet architecture with project manifest adapter.",
        ],
    }
    torch.save(model.state_dict(), output / "model.pt")
    write_json(output / "metrics.json", metrics)
    return metrics


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--route", choices=["supersimplenet", "dra"], required=True)
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
