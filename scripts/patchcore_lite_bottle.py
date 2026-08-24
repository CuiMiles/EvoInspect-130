from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from PIL import Image
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from torch.nn import functional as functional
from torchvision.models import (  # type: ignore[import-untyped]
    Wide_ResNet50_2_Weights,
    wide_resnet50_2,
)
from torchvision.models.feature_extraction import (  # type: ignore[import-untyped]
    create_feature_extractor,
)

from evoinspect.baseline import select_threshold
from evoinspect.evaluation import binary_metrics
from evoinspect.fusion import apply_guarded_fusion, calibrate_guarded_fusion
from evoinspect.heteromemory import build_direction_bank, heteromemory_scores, spatial_grid
from evoinspect.provenance import append_csv, canonical_hash, file_sha256, git_state, write_json

REGISTRY_COLUMNS = [
    "run_id",
    "status",
    "start_time",
    "end_time",
    "git_commit",
    "dirty",
    "config_hash",
    "data_hash",
    "split_hash",
    "seed",
    "hardware",
    "model",
    "protocol",
    "metrics_path",
    "artifact_path",
    "failure_reason",
    "notes",
]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def cached_file_sha256(path: Path, environment_name: str) -> str:
    cached = os.environ.get(environment_name, "").lower()
    if cached:
        if len(cached) != 64 or any(character not in "0123456789abcdef" for character in cached):
            raise RuntimeError(f"invalid cached SHA-256 in {environment_name}")
        return cached
    return file_sha256(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def write_csv(path: Path, rows: Sequence[dict[str, str]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError(f"invalid baseline config: {path}")
    if value.get("backbone") != "torchvision.wide_resnet50_2":
        raise RuntimeError(f"unsupported backbone in {path}")
    if value.get("weights") != "IMAGENET1K_V2":
        raise RuntimeError(f"unsupported weights in {path}")
    if value.get("input_resolution") != [224, 224]:
        raise RuntimeError(f"unsupported input resolution in {path}")
    if value.get("inference", {}).get("batch_size") != 1:
        raise RuntimeError(f"official-style inference requires batch size 1: {path}")
    heteromemory = value.get("heteromemory")
    if heteromemory is not None and not isinstance(heteromemory, dict):
        raise RuntimeError(f"heteromemory config must be a mapping: {path}")
    guarded_fusion = value.get("guarded_fusion")
    if guarded_fusion is not None and not isinstance(guarded_fusion, dict):
        raise RuntimeError(f"guarded_fusion config must be a mapping: {path}")
    return value


def resolve_category(rows: Sequence[dict[str, str]], requested: str) -> str:
    categories = {row["product_id"] for row in rows}
    candidates = [requested, f"mvtec_ad_{requested}"]
    matches = [candidate for candidate in candidates if candidate in categories]
    if len(matches) != 1:
        raise RuntimeError(
            f"unknown or ambiguous category {requested!r}; available={sorted(categories)}"
        )
    return matches[0]


def protocol_counts(normal_count: int, seen_anomaly_count: int) -> dict[str, int]:
    if normal_count < 2:
        raise RuntimeError("category needs at least two official train normal images")
    if seen_anomaly_count < 4:
        raise RuntimeError("category needs at least four seen anomaly images")
    if normal_count >= 120:
        normal_support = 100
        development_normal = 20
    else:
        normal_support = max(1, math.floor(normal_count * 0.8))
        development_normal = normal_count - normal_support
    if seen_anomaly_count >= 37:
        anomaly_support = 30
        development_anomaly = 6
    else:
        anomaly_support = max(1, math.floor(seen_anomaly_count * 0.75))
        remaining = seen_anomaly_count - anomaly_support
        development_anomaly = max(1, remaining // 2)
    final_seen_anomaly = seen_anomaly_count - anomaly_support - development_anomaly
    if development_normal < 1 or development_anomaly < 1 or final_seen_anomaly < 1:
        raise RuntimeError("category cannot retain development and final-test samples")
    return {
        "normal_support": normal_support,
        "development_normal": development_normal,
        "anomaly_support": anomaly_support,
        "development_anomaly": development_anomaly,
        "final_seen_anomaly": final_seen_anomaly,
    }


def prepare(manifest: Path, output_dir: Path, seed: int, requested_category: str) -> None:
    source_rows = read_csv(manifest)
    if len(source_rows) != len({row["content_sha256"] for row in source_rows}):
        raise RuntimeError("content hash duplicate in source manifest")
    category = resolve_category(source_rows, requested_category)
    rows = [dict(row) for row in source_rows if row["product_id"] == category]
    randomizer = random.Random(seed)
    train_normal = [
        dict(row) for row in rows if row["official_split"] == "train" and row["label"] == "normal"
    ]
    test_normal = [
        dict(row) for row in rows if row["official_split"] == "test" and row["label"] == "normal"
    ]
    anomaly_rows = [dict(row) for row in rows if row["label"] == "anomaly"]
    anomaly_types = sorted({row["defect_type"] for row in anomaly_rows})
    if not anomaly_types:
        raise RuntimeError(f"category has no anomaly types: {category}")
    # Predeclared deterministic holdout: the lexicographically last defect type.
    # A one-type category cannot support an unseen-defect slice and records that limit.
    unseen_types = {anomaly_types[-1]} if len(anomaly_types) > 1 else set()
    seen_types = set(anomaly_types) - unseen_types if unseen_types else set(anomaly_types)
    seen = [dict(row) for row in anomaly_rows if row["defect_type"] in seen_types]
    unseen = [dict(row) for row in anomaly_rows if row["defect_type"] in unseen_types]
    randomizer.shuffle(train_normal)
    randomizer.shuffle(seen)
    counts = protocol_counts(len(train_normal), len(seen))
    support_normal = train_normal[: counts["normal_support"]]
    development_normal = train_normal[
        counts["normal_support"] : counts["normal_support"] + counts["development_normal"]
    ]
    support_anomaly = seen[: counts["anomaly_support"]]
    development_anomaly = seen[
        counts["anomaly_support"] : counts["anomaly_support"]
        + counts["development_anomaly"]
    ]
    final_seen = seen[counts["anomaly_support"] + counts["development_anomaly"] :]
    for row in support_normal:
        row.update(role="support_normal", defect_visibility="normal")
    for row in support_anomaly:
        row.update(role="support_anomaly", defect_visibility="seen")
    for row in development_normal:
        row.update(role="development", defect_visibility="normal")
    for row in development_anomaly:
        row.update(role="development", defect_visibility="seen")
    final_rows = test_normal + final_seen + unseen
    for row in final_rows:
        row["role"] = "final_test"
        row["defect_visibility"] = (
            "normal"
            if row["label"] == "normal"
            else "unseen"
            if row["defect_type"] in unseen_types
            else "seen"
        )
    adaptation = support_normal + support_anomaly + development_normal + development_anomaly
    columns = [*list(rows[0]), "role", "defect_visibility"]
    test_inputs = []
    test_truth = []
    for row in final_rows:
        input_row = dict(row)
        input_row.update(label="", defect_type="", defect_visibility="")
        test_inputs.append(input_row)
        truth_row = dict(row)
        truth_row["path"] = ""
        test_truth.append(truth_row)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "adaptation.csv", adaptation, columns)
    write_csv(output_dir / "test_inputs.csv", test_inputs, columns)
    write_csv(output_dir / "test_truth.csv", test_truth, columns)
    split = {
        "seed": seed,
        "category": category,
        "normal_support": len(support_normal),
        "anomaly_support": len(support_anomaly),
        "development_normal": len(development_normal),
        "development_anomaly": len(development_anomaly),
        "final_test_normal": len(test_normal),
        "final_test_seen_anomaly": len(final_seen),
        "final_test_unseen_anomaly": len(unseen),
        "seen_types": sorted(seen_types),
        "unseen_types": sorted(unseen_types),
        "source_manifest_sha256": cached_file_sha256(
            manifest, "EVOINSPECT_MANIFEST_SHA256"
        ),
        "source_values": sorted({row["source"] for row in rows}),
        "dataset": "MVTec_AD_direct_archive",
        "protocol": "official_style_up_to_100_normal_30_seen_anomaly",
        "protocol_limits": [
            message
            for condition, message in (
                (
                    len(support_normal) < 100,
                    "normal support reduced because category has <120 train normals",
                ),
                (
                    len(support_anomaly) < 30,
                    "anomaly support reduced to retain development and final-test anomalies",
                ),
                (
                    not unseen,
                    "unseen-defect holdout unavailable because category has one defect type",
                ),
            )
            if condition
        ],
        "adaptation_manifest_sha256": file_sha256(output_dir / "adaptation.csv"),
        "test_inputs_manifest_sha256": file_sha256(output_dir / "test_inputs.csv"),
        "test_truth_manifest_sha256": file_sha256(output_dir / "test_truth.csv"),
    }
    split["split_hash"] = canonical_hash(
        sorted((row["sample_id"], row["role"]) for row in adaptation + final_rows)
    )
    write_json(output_dir / "split.json", split)


def build_extractor(device: torch.device) -> tuple[torch.nn.Module, Any, str, str]:
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    backbone = wide_resnet50_2(weights=weights)
    extractor = (
        create_feature_extractor(
            backbone,
            return_nodes={"layer2": "layer2", "layer3": "layer3", "avgpool": "global"},
        )
        .eval()
        .to(device)
    )
    transform = weights.transforms()
    filename = Path(urlparse(weights.url).path).name
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / filename
    weight_hash = (
        cached_file_sha256(checkpoint, "EVOINSPECT_WEIGHTS_SHA256")
        if checkpoint.is_file()
        else "MISSING"
    )
    return extractor, transform, weights.url, weight_hash


@torch.inference_mode()
def extract_batch(
    extractor: torch.nn.Module,
    transform: Any,
    rows: Sequence[dict[str, str]],
    device: torch.device,
    batch_size: int,
    patch_layer: str,
) -> tuple[list[torch.Tensor], NDArray[np.float64]]:
    patch_vectors: list[torch.Tensor] = []
    global_vectors: list[NDArray[np.float64]] = []
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        tensors = []
        for row in batch_rows:
            with Image.open(row["path"]) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        outputs = extractor(batch)
        if patch_layer not in {"layer2", "layer3"}:
            raise RuntimeError(f"unsupported patch feature layer: {patch_layer}")
        patch = functional.normalize(outputs[patch_layer], dim=1)
        patch = patch.permute(0, 2, 3, 1).reshape(patch.shape[0], -1, patch.shape[1])
        patch_vectors.extend(vector.cpu() for vector in patch)
        global_vectors.extend(outputs["global"].flatten(1).cpu().numpy())
    return patch_vectors, np.asarray(global_vectors)


def patch_score(patches: torch.Tensor, memory: torch.Tensor) -> float:
    distances = torch.cdist(patches.unsqueeze(0), memory.unsqueeze(0)).squeeze(0)
    return float(distances.min(dim=1).values.max().item())


def logistic_scores(global_vectors: NDArray[np.float64], model: dict[str, Any]) -> list[float]:
    mean = np.asarray(model["head_mean"], dtype=np.float64)
    scale = np.asarray(model["head_scale"], dtype=np.float64)
    coefficient = np.asarray(model["head_coefficient"], dtype=np.float64)
    logits = ((global_vectors - mean) / scale) @ coefficient + float(model["head_intercept"])
    return [float(1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, value))))) for value in logits]


def train(
    adaptation: Path,
    split_path: Path,
    output_model: Path,
    seed: int,
    config_path: Path,
) -> None:
    started_at = now()
    started = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = load_config(config_path)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.init()  # type: ignore[no-untyped-call]
    torch.cuda.reset_peak_memory_stats()
    rows = read_csv(adaptation)
    support_normal = [row for row in rows if row["role"] == "support_normal"]
    support_anomaly = [row for row in rows if row["role"] == "support_anomaly"]
    development = [row for row in rows if row["role"] == "development"]
    extractor, transform, weights_url, weights_hash = build_extractor(device)
    patch_layer = str(config.get("patch_layer", "layer3"))
    normal_patches, normal_global = extract_batch(
        extractor,
        transform,
        support_normal,
        device,
        batch_size=int(config["train_batch_size"]),
        patch_layer=patch_layer,
    )
    anomaly_patches, anomaly_global = extract_batch(
        extractor,
        transform,
        support_anomaly,
        device,
        batch_size=int(config["train_batch_size"]),
        patch_layer=patch_layer,
    )
    all_normal_patches = torch.cat(normal_patches, dim=0)
    patches_per_image = normal_patches[0].shape[0]
    grid_side = math.isqrt(patches_per_image)
    if grid_side * grid_side != patches_per_image:
        raise RuntimeError(f"non-square patch grid: {patches_per_image}")
    patch_coordinates = spatial_grid(grid_side, grid_side)
    all_normal_coordinates = patch_coordinates.repeat(len(normal_patches), 1)
    generator = torch.Generator().manual_seed(seed)
    coreset_size = min(int(config["coreset_size"]), len(all_normal_patches))
    indices = torch.randperm(len(all_normal_patches), generator=generator)[:coreset_size]
    memory = all_normal_patches[indices].contiguous()
    memory_coordinates = all_normal_coordinates[indices].contiguous()

    support_global = np.concatenate([normal_global, anomaly_global], axis=0)
    support_labels = np.asarray([0] * len(normal_global) + [1] * len(anomaly_global))
    scaler = StandardScaler().fit(support_global)
    logistic_config = config["logistic_regression"]
    head = LogisticRegression(
        C=float(logistic_config["c"]),
        max_iter=int(logistic_config["max_iter"]),
        random_state=seed,
    )
    head.fit(scaler.transform(support_global), support_labels)

    development_patches, development_global = extract_batch(
        extractor,
        transform,
        development,
        device,
        batch_size=int(config["train_batch_size"]),
        patch_layer=patch_layer,
    )
    memory_gpu = memory.to(device)
    patch_scores = [patch_score(patches.to(device), memory_gpu) for patches in development_patches]
    heteromemory_config = config.get("heteromemory", {})
    heteromemory_enabled = bool(heteromemory_config.get("enabled", False))
    direction_bank_cpu: torch.Tensor | None = None
    heteromemory_threshold: dict[str, float] | None = None
    direction_threshold: dict[str, float] | None = None
    direction_real_count = 0
    direction_synthetic_count = 0
    heteromemory_development_scores: list[float] = []
    direction_development_scores: list[float] = []
    if heteromemory_enabled:
        memory_coordinates_gpu = memory_coordinates.to(device)
        patch_coordinates_gpu = patch_coordinates.to(device)
        direction_bank = build_direction_bank(
            [patches.to(device) for patches in anomaly_patches],
            memory_gpu,
            memory_coordinates_gpu,
            patch_coordinates_gpu,
            top_k_per_image=int(heteromemory_config["top_k_per_image"]),
            spatial_weight=float(heteromemory_config["spatial_weight"]),
            synthetic_count=int(heteromemory_config["synthetic_count"]),
            jitter_strength=float(heteromemory_config["jitter_strength"]),
            seed=seed,
        )
        development_score_triples = [
            heteromemory_scores(
                patches.to(device),
                memory_gpu,
                memory_coordinates_gpu,
                patch_coordinates_gpu,
                direction_bank.vectors,
                spatial_weight=float(heteromemory_config["spatial_weight"]),
                direction_weight=float(heteromemory_config["direction_weight"]),
                query_top_k=int(heteromemory_config.get("query_top_k", patches_per_image)),
            )
            for patches in development_patches
        ]
        direction_development_scores = [scores[1] for scores in development_score_triples]
        heteromemory_development_scores = [scores[2] for scores in development_score_triples]
        direction_bank_cpu = direction_bank.vectors.cpu()
        direction_real_count = direction_bank.real_count
        direction_synthetic_count = direction_bank.synthetic_count
    head_model = {
        "head_mean": scaler.mean_.tolist(),
        "head_scale": scaler.scale_.tolist(),
        "head_coefficient": head.coef_[0].tolist(),
        "head_intercept": float(head.intercept_[0]),
    }
    supervised_scores = logistic_scores(development_global, head_model)
    development_labels = [int(row["label"] == "anomaly") for row in development]
    patch_threshold = select_threshold(patch_scores, development_labels)
    supervised_threshold = select_threshold(supervised_scores, development_labels)
    guarded_fusion_config = config.get("guarded_fusion", {})
    guarded_fusion_enabled = bool(guarded_fusion_config.get("enabled", False))
    guarded_fusion_calibration: dict[str, Any] | None = None
    if guarded_fusion_enabled:
        guarded_fusion_calibration = calibrate_guarded_fusion(
            patch_scores,
            supervised_scores,
            development_labels,
            float(patch_threshold["threshold"]),
            float(supervised_threshold["threshold"]),
            float(guarded_fusion_config["min_development_f1_gain"]),
            int(guarded_fusion_config.get("min_development_anomalies", 1)),
        )
    if heteromemory_enabled:
        fusion = str(heteromemory_config.get("fusion", "additive"))
        if fusion == "additive":
            heteromemory_threshold = select_threshold(
                heteromemory_development_scores, development_labels
            )
        elif fusion == "guarded_or":
            direction_threshold = select_threshold(
                direction_development_scores, development_labels
            )
            patch_scale = max(float(patch_threshold["threshold"]), 1e-12)
            direction_scale = max(float(direction_threshold["threshold"]), 1e-12)
            guarded_scores = [
                max(patch / patch_scale, direction / direction_scale)
                for patch, direction in zip(
                    patch_scores, direction_development_scores, strict=True
                )
            ]
            guarded_decisions = [int(score >= 1.0) for score in guarded_scores]
            guarded_metrics = binary_metrics(
                development_labels, guarded_scores, guarded_decisions
            )
            heteromemory_threshold = {
                "threshold": 1.0,
                "development_f1": guarded_metrics["f1_fixed_threshold"],
                "development_precision": guarded_metrics["precision"],
                "development_recall": guarded_metrics["recall"],
            }
        else:
            raise RuntimeError(f"unsupported heteromemory fusion: {fusion}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "model_id": config["model_id"],
        "status": "mvtec_category_preliminary_baseline",
        "seed": seed,
        "created_at": now(),
        "backbone": "torchvision.wide_resnet50_2",
        "weights": "IMAGENET1K_V2",
        "weights_url": weights_url,
        "weights_sha256": weights_hash,
        "image_transform": repr(transform),
        "memory": memory,
        "memory_coordinates": memory_coordinates,
        "coreset_strategy": config["coreset_strategy"],
        "coreset_size": coreset_size,
        "patch_threshold": patch_threshold,
        "supervised_threshold": supervised_threshold,
        "guarded_fusion_enabled": guarded_fusion_enabled,
        "guarded_fusion_calibration": guarded_fusion_calibration,
        "heteromemory_enabled": heteromemory_enabled,
        "heteromemory_threshold": heteromemory_threshold,
        "direction_threshold": direction_threshold,
        "direction_bank": direction_bank_cpu,
        "direction_real_count": direction_real_count,
        "direction_synthetic_count": direction_synthetic_count,
        "support_normal_ids": [row["sample_id"] for row in support_normal],
        "support_anomaly_ids": [row["sample_id"] for row in support_anomaly],
        "development_ids": [row["sample_id"] for row in development],
        "split_hash": split["split_hash"],
        "adaptation_manifest_sha256": file_sha256(adaptation),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "config_hash": canonical_hash(config),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "adaptation_seconds": time.perf_counter() - started,
        **head_model,
    }
    output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_model)
    summary = {
        key: value
        for key, value in artifact.items()
        if key not in {"memory", "memory_coordinates", "direction_bank"}
    }
    summary.update(
        {
            "model_sha256": file_sha256(output_model),
            "model_bytes": output_model.stat().st_size,
            "started_at": started_at,
            "completed_at": now(),
        }
    )
    write_json(output_model.with_suffix(".summary.json"), summary)


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


@torch.inference_mode()
def infer(test_inputs: Path, model_path: Path, output_path: Path) -> None:
    rows = read_csv(test_inputs)
    if any(row["label"] or row["defect_type"] for row in rows):
        raise RuntimeError("test input manifest exposes labels")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.init()  # type: ignore[no-untyped-call]
    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    model_hash = file_sha256(model_path)
    extractor, transform, _, weights_hash = build_extractor(device)
    if weights_hash != artifact["weights_sha256"]:
        raise RuntimeError("backbone weight hash mismatch")
    memory = artifact["memory"].to(device)
    heteromemory_enabled = bool(artifact.get("heteromemory_enabled", False))
    memory_coordinates = (
        artifact["memory_coordinates"].to(device) if heteromemory_enabled else None
    )
    direction_bank = artifact["direction_bank"].to(device) if heteromemory_enabled else None
    guarded_fusion_enabled = bool(artifact.get("guarded_fusion_enabled", False))
    zero = torch.zeros(1, 3, 224, 224, device=device)
    warmup_iterations = int(artifact["config"]["inference"]["warmup_iterations"])
    for _ in range(warmup_iterations):
        extractor(zero)
    torch.cuda.synchronize(device)
    predictions: list[dict[str, Any]] = []
    timings: dict[str, list[float]] = {
        "preprocess": [],
        "backbone": [],
        "postprocess": [],
        "end_to_end_without_file_io": [],
    }
    for row in rows:
        total_started = time.perf_counter()
        preprocess_started = time.perf_counter()
        with Image.open(row["path"]) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
        torch.cuda.synchronize(device)
        backbone_started = time.perf_counter()
        outputs = extractor(tensor)
        torch.cuda.synchronize(device)
        backbone_ms = (time.perf_counter() - backbone_started) * 1000
        postprocess_started = time.perf_counter()
        patch_layer = str(artifact["config"].get("patch_layer", "layer3"))
        patches = functional.normalize(outputs[patch_layer], dim=1)
        patches = patches.permute(0, 2, 3, 1).reshape(
            -1, outputs[patch_layer].shape[1]
        )
        patch_value = patch_score(patches, memory)
        heteromemory_value: float | None = None
        heteromemory_direction_value: float | None = None
        heteromemory_decision: int | None = None
        if heteromemory_enabled:
            grid_side = math.isqrt(len(patches))
            if grid_side * grid_side != len(patches):
                raise RuntimeError(f"non-square inference patch grid: {len(patches)}")
            _, heteromemory_direction_value, additive_value = heteromemory_scores(
                patches,
                memory,
                memory_coordinates,
                spatial_grid(grid_side, grid_side, device=device),
                direction_bank,
                spatial_weight=float(artifact["config"]["heteromemory"]["spatial_weight"]),
                direction_weight=float(
                    artifact["config"]["heteromemory"]["direction_weight"]
                ),
                query_top_k=int(
                    artifact["config"]["heteromemory"].get("query_top_k", len(patches))
                ),
            )
            fusion = str(artifact["config"]["heteromemory"].get("fusion", "additive"))
            if fusion == "additive":
                heteromemory_value = additive_value
                heteromemory_decision = int(
                    heteromemory_value >= artifact["heteromemory_threshold"]["threshold"]
                )
            elif fusion == "guarded_or":
                patch_scale = max(float(artifact["patch_threshold"]["threshold"]), 1e-12)
                direction_scale = max(
                    float(artifact["direction_threshold"]["threshold"]), 1e-12
                )
                heteromemory_value = max(
                    patch_value / patch_scale,
                    heteromemory_direction_value / direction_scale,
                )
                heteromemory_decision = int(heteromemory_value >= 1.0)
            else:
                raise RuntimeError(f"unsupported heteromemory fusion: {fusion}")
        global_vector = outputs["global"].flatten(1).cpu().numpy()
        supervised_value = logistic_scores(global_vector, artifact)[0]
        torch.cuda.synchronize(device)
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
        total_ms = (time.perf_counter() - total_started) * 1000
        latency = {
            "preprocess": preprocess_ms,
            "backbone": backbone_ms,
            "postprocess": postprocess_ms,
            "end_to_end_without_file_io": total_ms,
        }
        for key, value in latency.items():
            timings[key].append(value)
        prediction = {
            "sample_id": row["sample_id"],
            "model_version": model_hash,
            "patchcore_score": patch_value,
            "patchcore_decision": int(patch_value >= artifact["patch_threshold"]["threshold"]),
            "supervised_score": supervised_value,
            "supervised_decision": int(
                supervised_value >= artifact["supervised_threshold"]["threshold"]
            ),
            "latency_ms": latency,
        }
        if guarded_fusion_enabled:
            guarded_fusion_value, guarded_fusion_decision = apply_guarded_fusion(
                patch_value,
                supervised_value,
                float(artifact["patch_threshold"]["threshold"]),
                float(artifact["supervised_threshold"]["threshold"]),
                str(artifact["guarded_fusion_calibration"]["strategy"]),
            )
            prediction.update(
                {
                    "guarded_fusion_score": guarded_fusion_value,
                    "guarded_fusion_decision": guarded_fusion_decision,
                }
            )
        if heteromemory_value is not None:
            prediction.update(
                {
                    "heteromemory_score": heteromemory_value,
                    "heteromemory_direction_score": heteromemory_direction_value,
                    "heteromemory_decision": heteromemory_decision,
                }
            )
        predictions.append(prediction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, sort_keys=True) + "\n")
    latency_summary: dict[str, Any] = {
        key: {
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }
        for key, values in timings.items()
    }
    latency_summary["scope"] = (
        f"RTX 3090 batch=1, 224x224, warmup={warmup_iterations}, "
        "excludes model loading and output file I/O"
    )
    write_json(output_path.with_suffix(".latency.json"), latency_summary)


def metric_slices(
    truth: Sequence[dict[str, str]], predictions: dict[str, dict[str, Any]], method: str
) -> dict[str, Any]:
    def calculate(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
        labels = [int(row["label"] == "anomaly") for row in rows]
        if set(labels) != {0, 1}:
            return {
                "available": False,
                "samples": len(labels),
                "reason": "slice does not contain both normal and anomaly samples",
            }
        scores = [float(predictions[row["sample_id"]][f"{method}_score"]) for row in rows]
        decisions = [int(predictions[row["sample_id"]][f"{method}_decision"]) for row in rows]
        return {"available": True, **binary_metrics(labels, scores, decisions)}

    overall = list(truth)
    seen = [row for row in truth if row["label"] == "normal" or row["defect_visibility"] == "seen"]
    unseen = [
        row for row in truth if row["label"] == "normal" or row["defect_visibility"] == "unseen"
    ]
    return {"overall": calculate(overall), "seen": calculate(seen), "unseen": calculate(unseen)}


def evaluate(
    truth_path: Path,
    predictions_path: Path,
    model_path: Path,
    split_path: Path,
    output_path: Path,
    registry: Path,
) -> None:
    truth = read_csv(truth_path)
    if any(row["path"] for row in truth):
        raise RuntimeError("test truth manifest exposes image paths")
    prediction_rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    predictions = {row["sample_id"]: row for row in prediction_rows}
    if set(predictions) != {row["sample_id"] for row in truth}:
        raise RuntimeError("prediction/truth coverage mismatch")
    artifact_summary = json.loads(
        model_path.with_suffix(".summary.json").read_text(encoding="utf-8")
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    metrics = {
        "schema_version": 1,
        "created_at": now(),
        "status": "mvtec_category_preliminary_baseline",
        "dataset": split.get("dataset", "MVTec_AD_bottle_community_mirror"),
        "category": split["category"],
        "protocol": split["protocol"],
        "protocol_limits": split["protocol_limits"],
        "seed": artifact_summary["seed"],
        "patchcore_lite": metric_slices(truth, predictions, "patchcore"),
        "supervised_linear_head": metric_slices(truth, predictions, "supervised"),
        "latency": json.loads(
            predictions_path.with_suffix(".latency.json").read_text(encoding="utf-8")
        ),
        "model_sha256": file_sha256(model_path),
        "model_bytes": model_path.stat().st_size,
        "weights_sha256": artifact_summary["weights_sha256"],
        "split_hash": split["split_hash"],
        "source_manifest_sha256": split["source_manifest_sha256"],
        "warning": (
            "PatchCore-lite category result; not the formal upstream PatchCore baseline. "
            f"Sources: {split.get('source_values', ['legacy_unknown'])}."
        ),
    }
    if prediction_rows and "heteromemory_score" in prediction_rows[0]:
        metrics["heteromemory"] = metric_slices(truth, predictions, "heteromemory")
    if prediction_rows and "guarded_fusion_score" in prediction_rows[0]:
        metrics["guarded_fusion"] = metric_slices(truth, predictions, "guarded_fusion")
    write_json(output_path, metrics)
    commit, dirty = git_state(Path.cwd())
    run_id = output_path.parent.name
    append_csv(
        registry,
        REGISTRY_COLUMNS,
        {
            "run_id": run_id,
            "status": "completed_preliminary_single_category",
            "start_time": artifact_summary["started_at"],
            "end_time": now(),
            "git_commit": commit,
            "dirty": str(dirty).lower(),
            "config_hash": artifact_summary["config_hash"],
            "data_hash": split["source_manifest_sha256"],
            "split_hash": split["split_hash"],
            "seed": artifact_summary["seed"],
            "hardware": (
                f"RTX 3090 physical GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}; "
                f"{platform.platform()}; torch {torch.__version__}"
            ),
            "model": artifact_summary["model_id"],
            "protocol": split["protocol"],
            "metrics_path": str(output_path),
            "artifact_path": str(model_path),
            "failure_reason": "",
            "notes": (
                f"Dataset {metrics['dataset']}, category {split['category']}, preliminary; "
                f"limits={split['protocol_limits']}; not formal upstream PatchCore"
            ),
        },
    )


def aggregate(run_dirs: Sequence[Path], output_path: Path) -> None:
    runs = [json.loads((path / "metrics.json").read_text(encoding="utf-8")) for path in run_dirs]
    categories = sorted({run["category"] for run in runs})
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at": now(),
        "status": (
            "mvtec_15_category_preliminary_baseline"
            if len(categories) == 15
            else "partial_mvtec_category_preliminary_baseline"
        ),
        "seeds": sorted({run["seed"] for run in runs}),
        "categories": categories,
        "runs": [str(path) for path in run_dirs],
        "datasets": sorted({run["dataset"] for run in runs}),
        "warning": "PatchCore-lite full-category study; not formal upstream PatchCore.",
        "per_category": {},
        "macro_category_mean": {},
    }
    candidate_methods = (
        "patchcore_lite",
        "supervised_linear_head",
        "heteromemory",
        "guarded_fusion",
    )
    methods = tuple(method for method in candidate_methods if all(method in run for run in runs))
    slice_names = ("overall", "seen", "unseen")
    metric_names = ("auroc", "average_precision", "f1_fixed_threshold", "accuracy")
    for category in categories:
        category_runs = [run for run in runs if run["category"] == category]
        summary["per_category"][category] = {}
        for method in methods:
            summary["per_category"][category][method] = {}
            for slice_name in slice_names:
                summary["per_category"][category][method][slice_name] = {}
                for metric in metric_names:
                    values = [
                        float(run[method][slice_name][metric])
                        for run in category_runs
                        if run[method][slice_name].get("available", True)
                        and metric in run[method][slice_name]
                    ]
                    summary["per_category"][category][method][slice_name][metric] = (
                        {
                            "mean": statistics.mean(values),
                            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                            "values": values,
                        }
                        if values
                        else {"available": False}
                    )
    for method in methods:
        summary["macro_category_mean"][method] = {}
        for slice_name in slice_names:
            summary["macro_category_mean"][method][slice_name] = {}
            for metric in metric_names:
                category_means = [
                    value
                    for category in categories
                    if (
                        value := summary["per_category"][category][method][slice_name][metric].get(
                            "mean"
                        )
                    )
                    is not None
                ]
                summary["macro_category_mean"][method][slice_name][metric] = (
                    {
                        "mean": statistics.mean(category_means),
                        "category_count": len(category_means),
                    }
                    if category_means
                    else {"available": False, "category_count": 0}
                )
    write_json(output_path, summary)


def record_failure(
    registry: Path,
    run_id: str,
    seed: int,
    category: str,
    physical_gpu: str,
    config_path: Path,
    manifest: Path,
    run_dir: Path,
) -> None:
    split_path = run_dir / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8")) if split_path.is_file() else {}
    config = load_config(config_path)
    append_csv(
        registry,
        REGISTRY_COLUMNS,
        {
            "run_id": run_id,
            "status": "failed",
            "start_time": "UNKNOWN",
            "end_time": now(),
            "git_commit": git_state(Path.cwd())[0],
            "dirty": "true",
            "config_hash": canonical_hash(config),
            "data_hash": cached_file_sha256(manifest, "EVOINSPECT_MANIFEST_SHA256"),
            "split_hash": split.get("split_hash", "UNAVAILABLE"),
            "seed": seed,
            "hardware": (
                f"physical GPU {physical_gpu}; {platform.platform()}; torch {torch.__version__}"
            ),
            "model": config["model_id"],
            "protocol": "official_style_up_to_100_normal_30_seen_anomaly",
            "metrics_path": "",
            "artifact_path": str(run_dir / "model.pt") if (run_dir / "model.pt").is_file() else "",
            "failure_reason": "pipeline failed; inspect run.log",
            "notes": f"category={category}; preliminary PatchCore-lite full-category batch",
        },
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--manifest", required=True, type=Path)
    prepare_parser.add_argument("--output-dir", required=True, type=Path)
    prepare_parser.add_argument("--seed", required=True, type=int)
    prepare_parser.add_argument("--category", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--adaptation", required=True, type=Path)
    train_parser.add_argument("--split", required=True, type=Path)
    train_parser.add_argument("--output-model", required=True, type=Path)
    train_parser.add_argument("--seed", required=True, type=int)
    train_parser.add_argument("--config", required=True, type=Path)
    infer_parser = commands.add_parser("infer")
    infer_parser.add_argument("--test-inputs", required=True, type=Path)
    infer_parser.add_argument("--model", required=True, type=Path)
    infer_parser.add_argument("--output", required=True, type=Path)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--truth", required=True, type=Path)
    evaluate_parser.add_argument("--predictions", required=True, type=Path)
    evaluate_parser.add_argument("--model", required=True, type=Path)
    evaluate_parser.add_argument("--split", required=True, type=Path)
    evaluate_parser.add_argument("--output", required=True, type=Path)
    evaluate_parser.add_argument("--registry", required=True, type=Path)
    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--run-dirs", required=True, nargs="+", type=Path)
    aggregate_parser.add_argument("--output", required=True, type=Path)
    failure_parser = commands.add_parser("record-failure")
    failure_parser.add_argument("--registry", required=True, type=Path)
    failure_parser.add_argument("--run-id", required=True)
    failure_parser.add_argument("--seed", required=True, type=int)
    failure_parser.add_argument("--category", required=True)
    failure_parser.add_argument("--physical-gpu", required=True)
    failure_parser.add_argument("--config", required=True, type=Path)
    failure_parser.add_argument("--manifest", required=True, type=Path)
    failure_parser.add_argument("--run-dir", required=True, type=Path)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.output_dir, args.seed, args.category)
    elif args.command == "train":
        train(args.adaptation, args.split, args.output_model, args.seed, args.config)
    elif args.command == "infer":
        infer(args.test_inputs, args.model, args.output)
    elif args.command == "evaluate":
        evaluate(
            args.truth,
            args.predictions,
            args.model,
            args.split,
            args.output,
            args.registry,
        )
    elif args.command == "aggregate":
        aggregate(args.run_dirs, args.output)
    elif args.command == "record-failure":
        record_failure(
            args.registry,
            args.run_id,
            args.seed,
            args.category,
            args.physical_gpu,
            args.config,
            args.manifest,
            args.run_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
