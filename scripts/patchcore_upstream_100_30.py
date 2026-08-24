from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler
import torch
import torch.nn.functional as torch_functional
import yaml  # type: ignore[import-untyped]
from PIL import Image
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from evoinspect.baseline import select_threshold
from evoinspect.evaluation import binary_metrics
from evoinspect.fusion import apply_selective_rescue, calibrate_selective_rescue
from evoinspect.provenance import append_csv, canonical_hash, file_sha256, git_state, write_json

UPSTREAM_COMMIT = "fcaa92f124fb1ad74a7acf56726decd4b27cbcad"
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError(f"invalid config: {path}")
    if value.get("upstream", {}).get("commit") != UPSTREAM_COMMIT:
        raise RuntimeError("upstream commit mismatch in config")
    return value


def image_transform(config: dict[str, Any]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(config["input"]["resize"])),
            transforms.CenterCrop(int(config["input"]["crop"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def mask_transform(config: dict[str, Any]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(config["input"]["resize"])),
            transforms.CenterCrop(int(config["input"]["crop"])),
            transforms.ToTensor(),
        ]
    )


class CsvDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[dict[str, str]], transform: transforms.Compose) -> None:
        self.rows = list(rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "mask": torch.zeros((1, tensor.shape[-2], tensor.shape[-1])),
            "is_anomaly": int(row.get("label") == "anomaly"),
            "sample_id": row["sample_id"],
        }


def build_model(device: torch.device, config: dict[str, Any]) -> patchcore.patchcore.PatchCore:
    backbone = patchcore.backbones.load(str(config["backbone"]))
    backbone.name = str(config["backbone"])
    backbone.seed = None
    sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
        float(config["patchcore"]["coreset_fraction"]), device
    )
    model = patchcore.patchcore.PatchCore(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=list(config["layers"]),
        device=device,
        input_shape=(3, int(config["input"]["crop"]), int(config["input"]["crop"])),
        pretrain_embed_dimension=int(config["patchcore"]["pretrain_embed_dimension"]),
        target_embed_dimension=int(config["patchcore"]["target_embed_dimension"]),
        patchsize=int(config["patchcore"]["patch_size"]),
        featuresampler=sampler,
        anomaly_scorer_num_nn=int(config["patchcore"]["anomaly_scorer_num_nn"]),
        nn_method=patchcore.common.FaissNN(False, 4),
    )
    return model


def extract_descriptors(
    model: patchcore.patchcore.PatchCore,
    rows: Sequence[dict[str, str]],
    transform: transforms.Compose,
    batch_size: int,
    top_k_patches: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Pool upstream patch embeddings into global and defect-focused descriptors."""
    if top_k_patches < 1:
        raise ValueError("top_k_patches must be positive")
    loader = DataLoader(
        CsvDataset(rows, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    descriptors: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for batch in loader:
        images = batch["image"].to(torch.float).to(model.device)
        features = np.asarray(model._embed(images), dtype=np.float64)
        image_count = int(images.shape[0])
        if len(features) % image_count:
            raise RuntimeError("PatchCore embedding count is not divisible by batch size")
        features = features.reshape(image_count, -1, features.shape[-1])
        flat_features = features.reshape(-1, features.shape[-1])
        _, distances, _ = model.anomaly_scorer.predict([flat_features])
        distances = np.asarray(distances, dtype=np.float64).reshape(image_count, -1)
        for image_features, image_distances in zip(features, distances, strict=True):
            count = min(top_k_patches, len(image_features))
            top_indices = np.argpartition(image_distances, -count)[-count:]
            global_mean = image_features.mean(axis=0)
            top_mean = image_features[top_indices].mean(axis=0)
            descriptors.append(np.concatenate([global_mean, top_mean - global_mean]))
    if len(descriptors) != len(rows):
        raise RuntimeError("descriptor/sample coverage mismatch")
    return np.asarray(descriptors, dtype=np.float64)


def supervised_scores(
    descriptors: np.ndarray[Any, np.dtype[np.float64]], head: dict[str, Any]
) -> list[float]:
    mean = np.asarray(head["mean"], dtype=np.float64)
    scale = np.asarray(head["scale"], dtype=np.float64)
    coefficient = np.asarray(head["coefficient"], dtype=np.float64)
    logits = ((descriptors - mean) / scale) @ coefficient + float(head["intercept"])
    logits = np.clip(logits, -50.0, 50.0)
    return [float(value) for value in 1.0 / (1.0 + np.exp(-logits))]


def extract_masked_defect_prototypes(
    model: patchcore.patchcore.PatchCore,
    rows: Sequence[dict[str, str]],
    transform: transforms.Compose,
    target_mask_transform: transforms.Compose,
    max_prototypes: int,
    seed: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Extract only patch embeddings touched by support defect masks."""
    if max_prototypes < 1:
        raise ValueError("max_prototypes must be positive")
    prototype_chunks: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    for row in rows:
        if not row.get("mask_path"):
            raise RuntimeError(f"support anomaly lacks a mask: {row['sample_id']}")
        with Image.open(row["path"]) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(model.device)
        features, patch_shapes = model._embed(tensor, provide_patch_shapes=True)
        feature_grid = np.asarray(features, dtype=np.float32).reshape(
            patch_shapes[0][0], patch_shapes[0][1], -1
        )
        with Image.open(row["mask_path"]) as image:
            target = target_mask_transform(image).unsqueeze(0).to(model.device)
        pooled = torch_functional.adaptive_max_pool2d(
            target, (patch_shapes[0][0], patch_shapes[0][1])
        )[0, 0]
        selected = feature_grid[pooled.detach().cpu().numpy() > 0.0]
        if len(selected):
            prototype_chunks.append(selected)
    if not prototype_chunks:
        raise RuntimeError("support masks selected no defect prototypes")
    prototypes = np.concatenate(prototype_chunks, axis=0)
    if len(prototypes) > max_prototypes:
        generator = np.random.default_rng(seed)
        indices = generator.choice(len(prototypes), size=max_prototypes, replace=False)
        prototypes = prototypes[indices]
    return np.asarray(prototypes, dtype=np.float32)


def masked_prototype_score(
    model: patchcore.patchcore.PatchCore,
    tensor: torch.Tensor,
    prototypes: torch.Tensor,
) -> float:
    features = np.asarray(model._embed(tensor.to(model.device)), dtype=np.float32)
    query = torch.from_numpy(features).to(model.device)
    minimum_distance = torch.cdist(query.unsqueeze(0), prototypes.unsqueeze(0))[0].min()
    return float(1.0 / (minimum_distance.item() + 1e-6))


def masked_prototype_scores_for_rows(
    model: patchcore.patchcore.PatchCore,
    rows: Sequence[dict[str, str]],
    transform: transforms.Compose,
    prototypes: np.ndarray[Any, np.dtype[np.float32]],
) -> list[float]:
    prototypes_tensor = torch.from_numpy(prototypes).to(model.device)
    scores: list[float] = []
    for row in rows:
        with Image.open(row["path"]) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0)
        scores.append(masked_prototype_score(model, tensor, prototypes_tensor))
    return scores


def model_files(model_dir: Path) -> list[Path]:
    return sorted(
        path for path in model_dir.iterdir() if path.is_file() and path.name != "meta.json"
    )


def combined_model_hash(model_dir: Path) -> str:
    return canonical_hash([(path.name, file_sha256(path)) for path in model_files(model_dir)])


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
    split = json.loads(split_path.read_text(encoding="utf-8"))
    rows = read_csv(adaptation)
    support_normal = [row for row in rows if row["role"] == "support_normal"]
    development = [row for row in rows if row["role"] == "development"]
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.init()  # type: ignore[no-untyped-call]
    torch.cuda.reset_peak_memory_stats()
    transform = image_transform(config)
    model = build_model(device, config)
    train_loader = DataLoader(
        CsvDataset(support_normal, transform),
        batch_size=int(config["input"]["train_batch_size"]),
        shuffle=False,
        num_workers=int(config["input"]["num_workers"]),
        pin_memory=True,
    )
    model.fit(train_loader)
    development_loader = DataLoader(
        CsvDataset(development, transform),
        batch_size=int(config["input"]["train_batch_size"]),
        shuffle=False,
        num_workers=int(config["input"]["num_workers"]),
        pin_memory=True,
    )
    scores, _, labels, _ = model.predict(development_loader)
    threshold = select_threshold([float(score) for score in scores], [int(x) for x in labels])
    fusion_config = config.get("guarded_fusion_v3", {})
    fusion_enabled = bool(fusion_config.get("enabled", False))
    prototype_config = config.get("masked_prototype_v4", {})
    prototype_enabled = bool(prototype_config.get("enabled", False))
    if fusion_enabled and prototype_enabled:
        raise RuntimeError("GuardedFusion v3 and MaskedPrototype v4 are mutually exclusive")
    fusion_head: dict[str, Any] | None = None
    supervised_threshold: dict[str, float] | None = None
    fusion_calibration: dict[str, Any] | None = None
    if fusion_enabled:
        support_anomaly = [row for row in rows if row["role"] == "support_anomaly"]
        if not support_anomaly:
            raise RuntimeError("GuardedFusion v3 requires support anomalies")
        top_k_patches = int(fusion_config["descriptor_top_k_patches"])
        support_rows = [*support_normal, *support_anomaly]
        support_descriptors = extract_descriptors(
            model,
            support_rows,
            transform,
            int(config["input"]["train_batch_size"]),
            top_k_patches,
        )
        support_labels = np.asarray(
            [0] * len(support_normal) + [1] * len(support_anomaly), dtype=np.int64
        )
        scaler = StandardScaler().fit(support_descriptors)
        classifier = LogisticRegression(
            C=float(fusion_config["logistic_regression_c"]),
            max_iter=int(fusion_config["logistic_regression_max_iter"]),
            class_weight="balanced",
            random_state=seed,
        )
        classifier.fit(scaler.transform(support_descriptors), support_labels)
        fusion_head = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coefficient": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
            "descriptor_top_k_patches": top_k_patches,
            "descriptor": "concat(global_mean, top_k_nn_distance_mean_minus_global_mean)",
        }
        development_descriptors = extract_descriptors(
            model,
            development,
            transform,
            int(config["input"]["train_batch_size"]),
            top_k_patches,
        )
        development_supervised_scores = supervised_scores(
            development_descriptors, fusion_head
        )
        supervised_threshold = select_threshold(
            development_supervised_scores, [int(x) for x in labels]
        )
        fusion_calibration = calibrate_selective_rescue(
            [float(score) for score in scores],
            development_supervised_scores,
            [int(x) for x in labels],
            float(threshold["threshold"]),
            float(supervised_threshold["threshold"]),
            [float(value) for value in fusion_config["min_patch_ratio_candidates"]],
            float(fusion_config["min_development_f1_gain"]),
            float(fusion_config["max_development_precision_drop"]),
            int(fusion_config["min_development_anomalies"]),
        )
    defect_prototypes: np.ndarray[Any, np.dtype[np.float32]] | None = None
    prototype_threshold: dict[str, float] | None = None
    prototype_calibration: dict[str, Any] | None = None
    if prototype_enabled:
        support_anomaly = [row for row in rows if row["role"] == "support_anomaly"]
        if not support_anomaly:
            raise RuntimeError("MaskedPrototype v4 requires support anomalies")
        defect_prototypes = extract_masked_defect_prototypes(
            model,
            support_anomaly,
            transform,
            mask_transform(config),
            int(prototype_config["max_prototypes"]),
            seed,
        )
        development_prototype_scores = masked_prototype_scores_for_rows(
            model, development, transform, defect_prototypes
        )
        prototype_threshold = select_threshold(
            development_prototype_scores, [int(x) for x in labels]
        )
        prototype_calibration = calibrate_selective_rescue(
            [float(score) for score in scores],
            development_prototype_scores,
            [int(x) for x in labels],
            float(threshold["threshold"]),
            float(prototype_threshold["threshold"]),
            [float(value) for value in prototype_config["min_patch_ratio_candidates"]],
            float(prototype_config["min_development_f1_gain"]),
            float(prototype_config["max_development_precision_drop"]),
            int(prototype_config["min_development_anomalies"]),
        )
    output_model.mkdir(parents=True, exist_ok=False)
    model.save_to_path(str(output_model))
    if defect_prototypes is not None:
        np.save(output_model / "masked_defect_prototypes.npy", defect_prototypes)
    upstream_model_hash = combined_model_hash(output_model)
    fusion_payload = {
        "enabled": fusion_enabled,
        "head": fusion_head,
        "supervised_threshold": supervised_threshold,
        "calibration": fusion_calibration,
    }
    model_hash = (
        canonical_hash(
            {"upstream_model_hash": upstream_model_hash, "fusion": fusion_payload}
        )
        if fusion_enabled
        else upstream_model_hash
    )
    prototype_payload = {
        "enabled": prototype_enabled,
        "prototype_threshold": prototype_threshold,
        "calibration": prototype_calibration,
        "prototype_count": 0 if defect_prototypes is None else len(defect_prototypes),
    }
    if prototype_enabled:
        model_hash = canonical_hash(
            {"upstream_model_hash": upstream_model_hash, "masked_prototype_v4": prototype_payload}
        )
    meta = {
        "schema_version": 1,
        "status": (
            "upstream_patchcore_masked_prototype_v4_development"
            if prototype_enabled
            else "upstream_patchcore_guarded_fusion_v3_development"
            if fusion_enabled
            else "upstream_patchcore_official_style_baseline"
        ),
        "model_id": config["model_id"],
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_license": config["upstream"]["license"],
        "seed": seed,
        "category": split["category"],
        "split_hash": split["split_hash"],
        "source_manifest_sha256": split["source_manifest_sha256"],
        "support_normal": len(support_normal),
        "support_anomaly_unused_by_patchcore": sum(
            row["role"] == "support_anomaly" for row in rows
        ),
        "development_samples": len(development),
        "threshold": threshold,
        "guarded_fusion_v3_enabled": fusion_enabled,
        "guarded_fusion_v3_head": fusion_head,
        "guarded_fusion_v3_supervised_threshold": supervised_threshold,
        "guarded_fusion_v3_calibration": fusion_calibration,
        "guarded_fusion_v3_hash": canonical_hash(fusion_payload),
        "masked_prototype_v4_enabled": prototype_enabled,
        "masked_prototype_v4_threshold": prototype_threshold,
        "masked_prototype_v4_calibration": prototype_calibration,
        "masked_prototype_v4_count": prototype_payload["prototype_count"],
        "masked_prototype_v4_hash": canonical_hash(prototype_payload),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "adaptation_seconds": time.perf_counter() - started,
        "started_at": started_at,
        "completed_at": now(),
        "model_bytes": sum(path.stat().st_size for path in model_files(output_model)),
        "upstream_model_hash": upstream_model_hash,
        "model_hash": model_hash,
    }
    write_json(output_model / "meta.json", meta)


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def infer(test_inputs: Path, model_dir: Path, output_path: Path) -> None:
    rows = read_csv(test_inputs)
    if any(row["label"] or row["defect_type"] for row in rows):
        raise RuntimeError("test input manifest exposes labels")
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    if os.environ.get("EVOINSPECT_VERIFY_MODEL_HASH", "0") == "1":
        upstream_model_hash = combined_model_hash(model_dir)
        if upstream_model_hash != meta.get("upstream_model_hash", meta["model_hash"]):
            raise RuntimeError("saved upstream model hash mismatch")
        if meta.get("guarded_fusion_v3_enabled", False):
            fusion_payload = {
                "enabled": True,
                "head": meta["guarded_fusion_v3_head"],
                "supervised_threshold": meta["guarded_fusion_v3_supervised_threshold"],
                "calibration": meta["guarded_fusion_v3_calibration"],
            }
            if canonical_hash(fusion_payload) != meta["guarded_fusion_v3_hash"]:
                raise RuntimeError("saved GuardedFusion v3 hash mismatch")
            expected = canonical_hash(
                {"upstream_model_hash": upstream_model_hash, "fusion": fusion_payload}
            )
            if expected != meta["model_hash"]:
                raise RuntimeError("saved combined model hash mismatch")
        if meta.get("masked_prototype_v4_enabled", False):
            prototype_payload = {
                "enabled": True,
                "prototype_threshold": meta["masked_prototype_v4_threshold"],
                "calibration": meta["masked_prototype_v4_calibration"],
                "prototype_count": meta["masked_prototype_v4_count"],
            }
            if canonical_hash(prototype_payload) != meta["masked_prototype_v4_hash"]:
                raise RuntimeError("saved MaskedPrototype v4 hash mismatch")
            expected = canonical_hash(
                {
                    "upstream_model_hash": upstream_model_hash,
                    "masked_prototype_v4": prototype_payload,
                }
            )
            if expected != meta["model_hash"]:
                raise RuntimeError("saved combined model hash mismatch")
    config = meta["config"]
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.init()  # type: ignore[no-untyped-call]
    model = patchcore.patchcore.PatchCore(device)
    model.load_from_path(
        str(model_dir), device=device, nn_method=patchcore.common.FaissNN(False, 4)
    )
    transform = image_transform(config)
    fusion_enabled = bool(meta.get("guarded_fusion_v3_enabled", False))
    prototype_enabled = bool(meta.get("masked_prototype_v4_enabled", False))
    prototype_tensor = (
        torch.from_numpy(
            np.load(model_dir / "masked_defect_prototypes.npy").astype(np.float32)
        ).to(device)
        if prototype_enabled
        else None
    )
    zero = torch.zeros((1, 3, int(config["input"]["crop"]), int(config["input"]["crop"])))
    for _ in range(10):
        model._predict(zero)
        if fusion_enabled:
            zero_features = np.asarray(model._embed(zero.to(device)), dtype=np.float64)
            _, zero_distances, _ = model.anomaly_scorer.predict([zero_features])
            _ = zero_distances
        if prototype_tensor is not None:
            _ = masked_prototype_score(model, zero, prototype_tensor)
    torch.cuda.synchronize(device)
    predictions: list[dict[str, Any]] = []
    masks: list[np.ndarray[Any, Any]] = []
    latencies: list[float] = []
    for row in rows:
        with Image.open(row["path"]) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        scores, image_masks = model._predict(tensor)
        supervised_value: float | None = None
        fusion_value: float | None = None
        fusion_decision: int | None = None
        prototype_value: float | None = None
        prototype_fusion_value: float | None = None
        prototype_fusion_decision: int | None = None
        if fusion_enabled:
            features = np.asarray(model._embed(tensor.to(device)), dtype=np.float64)
            _, distances, _ = model.anomaly_scorer.predict([features])
            distances = np.asarray(distances, dtype=np.float64).reshape(-1)
            count = min(
                int(meta["guarded_fusion_v3_head"]["descriptor_top_k_patches"]),
                len(features),
            )
            top_indices = np.argpartition(distances, -count)[-count:]
            global_mean = features.mean(axis=0)
            top_mean = features[top_indices].mean(axis=0)
            descriptor = np.concatenate([global_mean, top_mean - global_mean])[None, :]
            supervised_value = supervised_scores(
                descriptor, meta["guarded_fusion_v3_head"]
            )[0]
            calibration = meta["guarded_fusion_v3_calibration"]
            fusion_value, fusion_decision = apply_selective_rescue(
                float(scores[0]),
                supervised_value,
                float(meta["threshold"]["threshold"]),
                float(meta["guarded_fusion_v3_supervised_threshold"]["threshold"]),
                float(calibration["selected_min_patch_ratio"]),
                str(calibration["strategy"]),
            )
        if prototype_tensor is not None:
            prototype_value = masked_prototype_score(model, tensor, prototype_tensor)
            calibration = meta["masked_prototype_v4_calibration"]
            prototype_fusion_value, prototype_fusion_decision = apply_selective_rescue(
                float(scores[0]),
                prototype_value,
                float(meta["threshold"]["threshold"]),
                float(meta["masked_prototype_v4_threshold"]["threshold"]),
                float(calibration["selected_min_patch_ratio"]),
                str(calibration["strategy"]),
            )
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        score = float(scores[0])
        prediction = {
                "sample_id": row["sample_id"],
                "model_version": meta["model_hash"],
                "upstream_patchcore_score": score,
                "upstream_patchcore_decision": int(score >= meta["threshold"]["threshold"]),
                "model_only_latency_ms": elapsed_ms,
            }
        if fusion_enabled:
            prediction.update(
                {
                    "guarded_fusion_v3_score": fusion_value,
                    "guarded_fusion_v3_decision": fusion_decision,
                    "supervised_score": supervised_value,
                }
            )
        if prototype_enabled:
            prediction.update(
                {
                    "masked_prototype_v4_score": prototype_fusion_value,
                    "masked_prototype_v4_decision": prototype_fusion_decision,
                    "defect_prototype_score": prototype_value,
                }
            )
        predictions.append(prediction)
        masks.append(np.asarray(image_masks[0], dtype=np.float16))
        latencies.append(elapsed_ms)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, sort_keys=True) + "\n")
    np.savez_compressed(
        output_path.with_suffix(".masks.npz"),
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        masks=np.stack(masks),
    )
    write_json(
        output_path.with_suffix(".latency.json"),
        {
            "model_only": {
                "p50": percentile(latencies, 0.5),
                "p95": percentile(latencies, 0.95),
                "max": max(latencies),
            },
            "scope": (
                "RTX 3090, batch=1, 224x224, warmup=10, model online graph only; "
                + (
                    "includes a second embedding pass for GuardedFusion v3; "
                    if fusion_enabled
                    else "includes a second embedding pass for MaskedPrototype v4; "
                    if prototype_enabled
                    else ""
                )
                + "excludes image decode, preprocessing, model load, mask serialization "
                "and file I/O"
            ),
        },
    )


def metric_slices(
    truth: Sequence[dict[str, str]],
    predictions: dict[str, dict[str, Any]],
    prefix: str = "upstream_patchcore",
) -> dict[str, Any]:
    def calculate(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
        labels = [int(row["label"] == "anomaly") for row in rows]
        if set(labels) != {0, 1}:
            return {"available": False, "samples": len(labels)}
        scores = [float(predictions[row["sample_id"]][f"{prefix}_score"]) for row in rows]
        decisions = [
            int(predictions[row["sample_id"]][f"{prefix}_decision"]) for row in rows
        ]
        return {"available": True, **binary_metrics(labels, scores, decisions)}

    return {
        "overall": calculate(truth),
        "seen": calculate(
            [row for row in truth if row["label"] == "normal" or row["defect_visibility"] == "seen"]
        ),
        "unseen": calculate(
            [
                row
                for row in truth
                if row["label"] == "normal" or row["defect_visibility"] == "unseen"
            ]
        ),
    }


def pixel_metrics(
    truth: Sequence[dict[str, str]], masks_path: Path, config: dict[str, Any]
) -> dict[str, float]:
    archive = np.load(masks_path)
    sample_ids = [str(value) for value in archive["sample_ids"]]
    predicted = np.asarray(archive["masks"], dtype=np.float32)
    predicted_by_id = dict(zip(sample_ids, predicted, strict=True))
    transform = mask_transform(config)
    ground_truth: list[np.ndarray[Any, Any]] = []
    predicted_ordered: list[np.ndarray[Any, Any]] = []
    anomaly_ground_truth: list[np.ndarray[Any, Any]] = []
    anomaly_predicted: list[np.ndarray[Any, Any]] = []
    for row in truth:
        if row["label"] == "anomaly":
            with Image.open(row["mask_path"]) as image:
                target = transform(image).numpy()[0]
            anomaly_ground_truth.append(target)
            anomaly_predicted.append(predicted_by_id[row["sample_id"]])
        else:
            target = np.zeros_like(predicted_by_id[row["sample_id"]])
        ground_truth.append(target)
        predicted_ordered.append(predicted_by_id[row["sample_id"]])
    flat_truth = np.stack(ground_truth).ravel().astype(np.uint8)
    flat_prediction = np.stack(predicted_ordered).ravel()
    anomaly_truth = np.stack(anomaly_ground_truth).ravel().astype(np.uint8)
    anomaly_prediction = np.stack(anomaly_predicted).ravel()
    return {
        "full_pixel_auroc": float(roc_auc_score(flat_truth, flat_prediction)),
        "full_pixel_average_precision": float(average_precision_score(flat_truth, flat_prediction)),
        "anomaly_pixel_auroc": float(roc_auc_score(anomaly_truth, anomaly_prediction)),
        "anomaly_pixel_average_precision": float(
            average_precision_score(anomaly_truth, anomaly_prediction)
        ),
    }


def evaluate(
    truth_path: Path,
    predictions_path: Path,
    model_dir: Path,
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
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    metrics = {
        "schema_version": 1,
        "created_at": now(),
        "status": "upstream_patchcore_official_style_baseline",
        "dataset": split["dataset"],
        "category": split["category"],
        "protocol": split["protocol"],
        "protocol_limits": split["protocol_limits"],
        "seed": meta["seed"],
        "upstream_patchcore": metric_slices(truth, predictions),
        "pixel": pixel_metrics(truth, predictions_path.with_suffix(".masks.npz"), meta["config"]),
        "latency": json.loads(
            predictions_path.with_suffix(".latency.json").read_text(encoding="utf-8")
        ),
        "model_hash": meta["model_hash"],
        "model_bytes": meta["model_bytes"],
        "split_hash": split["split_hash"],
        "source_manifest_sha256": split["source_manifest_sha256"],
        "upstream_commit": UPSTREAM_COMMIT,
        "warning": (
            "Pinned upstream PatchCore core with a repository adapter for the isolated 100+30 "
            "protocol and development-only image threshold."
        ),
    }
    if prediction_rows and "guarded_fusion_v3_score" in prediction_rows[0]:
        metrics["guarded_fusion_v3"] = metric_slices(
            truth, predictions, "guarded_fusion_v3"
        )
    if prediction_rows and "masked_prototype_v4_score" in prediction_rows[0]:
        metrics["masked_prototype_v4"] = metric_slices(
            truth, predictions, "masked_prototype_v4"
        )
    write_json(output_path, metrics)
    commit, dirty = git_state(Path.cwd())
    append_csv(
        registry,
        REGISTRY_COLUMNS,
        {
            "run_id": output_path.parent.name,
            "status": (
                "completed_development_innovation"
                if meta.get("guarded_fusion_v3_enabled", False)
                or meta.get("masked_prototype_v4_enabled", False)
                else "completed_official_style_baseline"
            ),
            "start_time": meta["started_at"],
            "end_time": now(),
            "git_commit": commit,
            "dirty": str(dirty).lower(),
            "config_hash": meta["config_sha256"],
            "data_hash": split["source_manifest_sha256"],
            "split_hash": split["split_hash"],
            "seed": meta["seed"],
            "hardware": (
                f"RTX 3090 physical GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}; "
                f"{platform.platform()}; torch {torch.__version__}; FAISS CPU"
            ),
            "model": meta["model_id"],
            "protocol": split["protocol"],
            "metrics_path": str(output_path),
            "artifact_path": str(model_dir),
            "failure_reason": "",
            "notes": (
                f"Pinned upstream {UPSTREAM_COMMIT}; category={split['category']}; "
                "support anomalies unused by PatchCore memory; "
                + (
                    "used only by GuardedFusion v3 supervised rescue head"
                    if meta.get("guarded_fusion_v3_enabled", False)
                    else "used only by MaskedPrototype v4 defect memory"
                    if meta.get("masked_prototype_v4_enabled", False)
                    else "not used by baseline"
                )
            ),
        },
    )


def aggregate(run_dirs: Sequence[Path], output_path: Path) -> None:
    runs = [json.loads((path / "metrics.json").read_text(encoding="utf-8")) for path in run_dirs]
    categories = sorted({run["category"] for run in runs})
    slice_names = ("overall", "seen", "unseen")
    metric_names = ("accuracy", "auroc", "average_precision", "f1_fixed_threshold")
    optional_names = ("guarded_fusion_v3", "masked_prototype_v4")
    optional_counts = {name: sum(name in run for run in runs) for name in optional_names}
    if any(count not in {0, len(runs)} for count in optional_counts.values()):
        raise RuntimeError("cannot aggregate a mixture of baseline and innovation runs")
    enabled_optional = tuple(
        name for name in optional_names if optional_counts[name] == len(runs)
    )
    if len(enabled_optional) > 1:
        raise RuntimeError("cannot aggregate multiple innovation variants together")
    model_names = ("upstream_patchcore", *enabled_optional)
    per_category: dict[str, Any] = {}
    for category in categories:
        category_runs = [run for run in runs if run["category"] == category]
        per_category[category] = {model_name: {} for model_name in model_names}
        for model_name in model_names:
            for slice_name in slice_names:
                per_category[category][model_name][slice_name] = {}
                for metric in metric_names:
                    values = [
                        float(run[model_name][slice_name][metric])
                        for run in category_runs
                        if run[model_name][slice_name].get("available", True)
                    ]
                    per_category[category][model_name][slice_name][metric] = (
                        {
                            "mean": statistics.mean(values),
                            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                            "values": values,
                        }
                        if values
                        else {"available": False}
                    )
        for metric in (
            "full_pixel_auroc",
            "full_pixel_average_precision",
            "anomaly_pixel_auroc",
            "anomaly_pixel_average_precision",
        ):
            values = [float(run["pixel"][metric]) for run in category_runs]
            per_category[category].setdefault("pixel", {})[metric] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
    macro: dict[str, Any] = {model_name: {} for model_name in model_names}
    for model_name in model_names:
        for slice_name in slice_names:
            macro[model_name][slice_name] = {}
            for metric in metric_names:
                values = [
                    per_category[category][model_name][slice_name][metric].get("mean")
                    for category in categories
                ]
                available = [float(value) for value in values if value is not None]
                macro[model_name][slice_name][metric] = {
                    "mean": statistics.mean(available),
                    "category_count": len(available),
                }
    macro["pixel"] = {}
    for metric in (
        "full_pixel_auroc",
        "full_pixel_average_precision",
        "anomaly_pixel_auroc",
        "anomaly_pixel_average_precision",
    ):
        macro["pixel"][metric] = {
            "mean": statistics.mean(
                per_category[category]["pixel"][metric]["mean"] for category in categories
            ),
            "category_count": len(categories),
        }
    write_json(
        output_path,
        {
            "schema_version": 1,
            "status": (
                "upstream_patchcore_masked_prototype_v4_development_aggregate"
                if "masked_prototype_v4" in enabled_optional
                else "upstream_patchcore_guarded_fusion_v3_development_aggregate"
                if "guarded_fusion_v3" in enabled_optional
                else "upstream_patchcore_official_style_aggregate"
            ),
            "dataset": "MVTec_AD_direct_archive",
            "protocol": "official_style_up_to_100_normal_30_seen_anomaly",
            "upstream_commit": UPSTREAM_COMMIT,
            "seeds": sorted({run["seed"] for run in runs}),
            "categories": categories,
            "runs": [str(path) for path in run_dirs],
            "per_category": per_category,
            "macro_category_mean": macro,
            "warning": (
                "Pinned upstream PatchCore core; repository adapter controls split and evidence; "
                + (
                    f"development-only {enabled_optional[0]} enabled."
                    if enabled_optional
                    else "baseline only."
                )
            ),
        },
    )


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
    commit, dirty = git_state(Path.cwd())
    append_csv(
        registry,
        REGISTRY_COLUMNS,
        {
            "run_id": run_id,
            "status": "failed",
            "start_time": "UNKNOWN",
            "end_time": now(),
            "git_commit": commit,
            "dirty": str(dirty).lower(),
            "config_hash": file_sha256(config_path),
            "data_hash": file_sha256(manifest),
            "split_hash": split.get("split_hash", "UNAVAILABLE"),
            "seed": seed,
            "hardware": (
                f"physical GPU {physical_gpu}; {platform.platform()}; torch {torch.__version__}"
            ),
            "model": config["model_id"],
            "protocol": "official_style_up_to_100_normal_30_seen_anomaly",
            "metrics_path": "",
            "artifact_path": str(run_dir / "model") if (run_dir / "model").is_dir() else "",
            "failure_reason": "pipeline failed; inspect run.log",
            "notes": f"Pinned upstream {UPSTREAM_COMMIT}; category={category}",
        },
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
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
    if args.command == "train":
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
