from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import platform
import random
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from PIL import Image

from evoinspect.baseline import select_threshold
from evoinspect.evaluation import binary_metrics
from evoinspect.localization import compute_localization_diagnostics
from evoinspect.provenance import (
    append_csv,
    canonical_hash,
    file_sha256,
    git_state,
    utc_now,
    write_json,
)
from evoinspect.rcbr import (
    NormalRiskCalibrator,
    Roi,
    RouterLimits,
    Strategy,
    attach_costs_and_utility,
    cross_fitted_utility_predictions,
    fuse_refinements,
    generate_candidates,
    high_frequency_map,
    multiscale_disagreement,
    roi_features,
    select_under_budget,
    strategy_candidates,
)

STRATEGIES: tuple[Strategy, ...] = (
    "uniform_downsample",
    "full_grid",
    "fixed_topk",
    "uncertainty_only",
    "risk_calibrated",
    "full_rcbr",
)
REGISTRY_COLUMNS = (
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
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError(f"invalid config: {path}")
    return value


def symlink_rows(rows: list[dict[str, str]], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        source = Path(row["path"])
        target = destination / f"{index:04d}{source.suffix.lower()}"
        if not target.exists():
            target.symlink_to(source)


def symlink_anomalies_with_masks(
    rows: list[dict[str, str]], image_destination: Path, mask_destination: Path
) -> None:
    image_destination.mkdir(parents=True, exist_ok=True)
    mask_destination.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        image_source = Path(row["path"])
        mask_source = Path(row["mask_path"])
        suffix = image_source.suffix.lower()
        image_target = image_destination / f"{index:04d}{suffix}"
        mask_target = mask_destination / f"{index:04d}{mask_source.suffix.lower()}"
        if not image_target.exists():
            image_target.symlink_to(image_source)
        if not mask_target.exists():
            mask_target.symlink_to(mask_source)


def train_efficientad(
    train_rows: list[dict[str, str]],
    calibration_rows: list[dict[str, str]],
    development_anomalies: list[dict[str, str]],
    output_dir: Path,
    config: dict[str, Any],
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_per_process_memory_fraction(
        float(os.environ.get("EVOINSPECT_GPU_MEMORY_FRACTION", "0.35")), device=0
    )
    torch.set_float32_matmul_precision("high")
    data_root = output_dir / "derived_training"
    symlink_rows(train_rows, data_root / "train" / "good")
    symlink_rows(calibration_rows, data_root / "calibration" / "good")
    symlink_anomalies_with_masks(
        development_anomalies,
        data_root / "development" / "anomaly",
        data_root / "development" / "masks",
    )
    training = config["training"]
    model_size = str(config["model_size"])
    if model_size not in {"small", "medium"}:
        raise RuntimeError(f"unsupported EfficientAD model_size: {model_size}")
    datamodule = Folder(
        name=f"efficientad-{model_size}-{seed}",
        root=data_root,
        normal_dir="train/good",
        normal_test_dir="calibration/good",
        abnormal_dir="development/anomaly",
        mask_dir="development/masks",
        train_batch_size=1,
        eval_batch_size=int(training["eval_batch_size"]),
        num_workers=int(training["num_workers"]),
        val_split_mode="same_as_test",
        seed=seed,
    )
    model = EfficientAd(
        imagenet_dir=Path(config["imagenette_dir"]),
        model_size=model_size,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    engine = Engine(
        max_steps=int(training["max_steps"]),
        max_epochs=int(training["max_epochs"]),
        # EfficientAD's validation loop only refreshes map-normalization
        # quantiles; it is not part of the optimization loss. Refreshing less
        # often avoids thousands of redundant passes over the held-out support
        # set during long runs. We recompute once after fit below so inference
        # always uses quantiles from the final weights.
        check_val_every_n_epoch=int(training.get("validation_every_n_epochs", 1)),
        accelerator="gpu",
        devices=1,
        precision=str(training["precision"]),
        default_root_dir=output_dir / "training",
        enable_progress_bar=True,
        # Anomalib 2.3.0 always installs its ModelCheckpoint callback. Lightning
        # rejects that callback when checkpointing is explicitly disabled.
        enable_checkpointing=True,
        deterministic=True,
    )
    started = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    elapsed = time.perf_counter() - started
    checkpoint = output_dir / "model.ckpt"
    engine.trainer.save_checkpoint(checkpoint, weights_only=False)
    # Lightning's strategy teardown moves the module back to CPU after fit. All
    # calibration, router fitting and controlled test inference below must run on
    # the assigned GPU rather than silently consuming many shared CPU cores.
    model.to(torch.device("cuda", 0))
    model.eval()
    if model.device.type != "cuda":
        raise RuntimeError("EfficientAD model was not restored to CUDA after training")
    final_quantiles = model.map_norm_quantiles(datamodule.val_dataloader())
    model.model.quantiles.update(final_quantiles)
    return model, {
        "training_seconds": elapsed,
        "checkpoint": str(checkpoint),
        "model_sha256": file_sha256(checkpoint),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def load_rgb(path: str | Path) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def resize_map(values: NDArray[np.floating[Any]], shape: tuple[int, int]) -> NDArray[np.float32]:
    import torch

    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))[None, None]
    output = torch.nn.functional.interpolate(
        tensor, size=shape, mode="bilinear", align_corners=False
    )
    return np.asarray(output[0, 0].numpy(), dtype=np.float32)


def infer_array(
    model: Any, image: NDArray[np.uint8], input_shape: tuple[int, int]
) -> tuple[NDArray[np.float32], float]:
    import torch

    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div_(255.0)[None]
    tensor = torch.nn.functional.interpolate(
        tensor, size=input_shape, mode="bilinear", align_corners=False, antialias=True
    ).to(model.device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = model.model(tensor)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    anomaly_map = prediction.anomaly_map.detach().float().cpu().squeeze().numpy()
    return resize_map(anomaly_map, input_shape), elapsed_ms


def mask_for_row(row: dict[str, str], shape: tuple[int, int]) -> NDArray[np.bool_]:
    if row.get("label") != "anomaly":
        return np.zeros(shape, dtype=np.bool_)
    mask_path = row.get("mask_path", "")
    if not mask_path:
        raise RuntimeError(f"anomaly row lacks mask: {row.get('sample_id')}")
    with Image.open(mask_path) as mask:
        resized = mask.convert("L").resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
        return np.asarray(resized, dtype=np.uint8) > 0


def full_grid(shape: tuple[int, int], rows: int, columns: int, cost_ms: float) -> list[Roi]:
    height, width = shape
    result: list[Roi] = []
    for row in range(rows):
        y0, y1 = round(row * height / rows), round((row + 1) * height / rows)
        for column in range(columns):
            x0, x1 = round(column * width / columns), round((column + 1) * width / columns)
            result.append(Roi(y0, x0, y1, x1, 1, 1, 1, 0, 1, cost_ms, "full_grid"))
    return result


def roi_union_fraction(rois: list[Roi], shape: tuple[int, int]) -> float:
    union = np.zeros(shape, dtype=np.bool_)
    for roi in rois:
        union[roi.y0 : roi.y1, roi.x0 : roi.x1] = True
    return float(union.mean())


def fit_router(
    model: Any,
    support_rows: list[dict[str, str]],
    calibrator: NormalRiskCalibrator,
    normal_maps: NDArray[np.float32],
    input_shape: tuple[int, int],
    router_config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    features: list[NDArray[np.float64]] = []
    labels: list[int] = []
    folds: list[int] = []
    class_offsets = {"normal": 0, "anomaly": 0}
    for row in support_rows:
        image = load_rgb(row["path"])
        raw_map, _ = infer_array(model, image, input_shape)
        risk = calibrator.transform(raw_map)
        image_small = np.asarray(
            Image.fromarray(image).resize((input_shape[1], input_shape[0])), dtype=np.float32
        )
        candidates = generate_candidates(
            risk,
            multiscale_disagreement(raw_map),
            high_frequency_map(image_small),
            window_fractions=tuple(
                float(value) for value in router_config["candidate_window_fractions"]
            ),
            stride_fraction=float(router_config["candidate_stride_fraction"]),
            per_scale=int(router_config["candidates_per_scale"]),
        )
        target = mask_for_row(row, input_shape)
        class_name = row["label"]
        fold = class_offsets[class_name] % 5
        class_offsets[class_name] += 1
        for roi in candidates:
            features.append(roi_features(roi))
            overlap = float(target[roi.y0 : roi.y1, roi.x0 : roi.x1].sum()) / max(
                1, int(target.sum())
            )
            labels.append(int(overlap >= float(router_config["utility_positive_min_gt_coverage"])))
            folds.append(fold)
    if len(set(labels)) != 2:
        raise RuntimeError("ROI utility training has only one class")
    utility_model, held_out = cross_fitted_utility_predictions(
        np.stack(features), np.asarray(labels), np.asarray(folds)
    )
    targets = np.asarray(labels, dtype=np.float64)
    return utility_model, {
        "samples": len(labels),
        "positive_fraction": float(targets.mean()),
        "cross_fitted_brier": float(np.mean((held_out - targets) ** 2)),
        "folds": 5,
        "normal_maps_shape": list(normal_maps.shape),
    }


def build_image_outputs(
    model: Any,
    row: dict[str, str],
    calibrator: NormalRiskCalibrator,
    normal_maps: NDArray[np.float32],
    utility_model: Any,
    input_shape: tuple[int, int],
    config: dict[str, Any],
    roi_cost_ms: float,
) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
    router = config["router"]
    image = load_rgb(row["path"])
    image_small = np.asarray(
        Image.fromarray(image).resize((input_shape[1], input_shape[0])), dtype=np.float32
    )
    global_map, global_ms = infer_array(model, image, input_shape)
    calibrated = calibrator.transform(global_map)
    candidates = generate_candidates(
        calibrated,
        multiscale_disagreement(global_map),
        high_frequency_map(image_small),
        window_fractions=tuple(float(value) for value in router["candidate_window_fractions"]),
        stride_fraction=float(router["candidate_stride_fraction"]),
        per_scale=int(router["candidates_per_scale"]),
    )
    latency_table = {roi.area: roi_cost_ms for roi in candidates}
    if not latency_table:
        latency_table = {input_shape[0] * input_shape[1]: roi_cost_ms}
    candidates = attach_costs_and_utility(
        candidates,
        utility_model,
        latency_table,
        false_positive_penalty=float(router["utility_false_positive_penalty"]),
    )
    limits = RouterLimits(
        latency_budget_ms=float(router["latency_budget_ms"]),
        max_rois=int(router["max_rois"]),
        max_total_area_fraction=float(router["max_single_image_roi_area_fraction"]),
        nms_iou=float(router["nms_iou"]),
    )
    selected: dict[str, list[Roi]] = {"uniform_downsample": []}
    grid_config = config["full_grid"]
    selected["full_grid"] = full_grid(
        input_shape, int(grid_config["rows"]), int(grid_config["columns"]), roi_cost_ms
    )
    fixed = strategy_candidates("fixed_topk", candidates)
    selected["fixed_topk"] = fixed[: int(router["fixed_topk"])]
    for strategy in ("uncertainty_only", "risk_calibrated", "full_rcbr"):
        routed = strategy_candidates(strategy, candidates)
        selected[strategy] = select_under_budget(routed, input_shape, limits)

    coordinates = sorted(
        {(roi.y0, roi.x0, roi.y1, roi.x1) for rois in selected.values() for roi in rois}
    )
    local_maps: dict[tuple[int, int, int, int], NDArray[np.float32]] = {}
    local_times: dict[tuple[int, int, int, int], float] = {}
    height, width = image.shape[:2]
    for coordinate in coordinates:
        y0, x0, y1, x1 = coordinate
        source_y0, source_y1 = (
            round(y0 * height / input_shape[0]),
            round(y1 * height / input_shape[0]),
        )
        source_x0, source_x1 = (
            round(x0 * width / input_shape[1]),
            round(x1 * width / input_shape[1]),
        )
        crop = image[source_y0:source_y1, source_x0:source_x1]
        local, elapsed = infer_array(model, crop, input_shape)
        local_maps[coordinate] = resize_map(local, (y1 - y0, x1 - x0))
        local_times[coordinate] = elapsed

    outputs: dict[str, NDArray[np.float32]] = {"uniform_downsample": global_map}
    audit: dict[str, Any] = {}
    for strategy in STRATEGIES[1:]:
        # Risk calibration is a routing signal, not a replacement anomaly-score space.
        # The global and local EfficientAD outputs must remain comparable before max fusion;
        # mixing a spatial CDF with raw model scores caused broad false-positive inflation in
        # the first smoke gate. The calibrated map still drives candidate generation above.
        base = global_map
        refinements = []
        for roi in selected[strategy]:
            local = local_maps[(roi.y0, roi.x0, roi.y1, roi.x1)]
            evidence = roi.predicted_benefit if strategy == "full_rcbr" else 1.0
            refinements.append((roi, local, evidence))
        output, records = fuse_refinements(
            base,
            refinements,
            minimum_evidence=(
                float(router["minimum_refinement_evidence"]) if strategy == "full_rcbr" else 0.0
            ),
        )
        outputs[strategy] = output
        applied = [record for record in records if record["result"] == "applied"]
        audit[strategy] = {
            "rois": records,
            "selected_roi_count": len(selected[strategy]),
            "applied_roi_count": len(applied),
            "fallback_count": len(records) - len(applied),
            "roi_area_fraction": roi_union_fraction(selected[strategy], input_shape),
            "predicted_roi_cost_ms": float(
                sum(roi.predicted_cost_ms for roi in selected[strategy])
            ),
            "measured_unique_roi_inference_ms": float(
                sum(local_times[(roi.y0, roi.x0, roi.y1, roi.x1)] for roi in selected[strategy])
            ),
        }
    audit["uniform_downsample"] = {
        "rois": [],
        "selected_roi_count": 0,
        "applied_roi_count": 0,
        "fallback_count": 0,
        "roi_area_fraction": 0.0,
        "predicted_roi_cost_ms": 0.0,
        "measured_unique_roi_inference_ms": 0.0,
    }
    audit["global_inference_ms"] = global_ms
    return outputs, audit


def score_map(anomaly_map: NDArray[np.float32], quantile: float) -> float:
    return float(np.quantile(anomaly_map, quantile))


def safe_binary_metrics(
    rows: list[dict[str, str]], scores: list[float], threshold: float
) -> dict[str, float] | None:
    labels = [int(row["label"] == "anomaly") for row in rows]
    if set(labels) != {0, 1}:
        return None
    return binary_metrics(labels, scores, [int(value >= threshold) for value in scores])


def evaluate_strategy(
    rows: list[dict[str, str]],
    maps: list[NDArray[np.float32]],
    scores: list[float],
    threshold: float,
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    labels = [int(row["label"] == "anomaly") for row in rows]
    decisions = [int(score >= threshold) for score in scores]
    overall = binary_metrics(labels, scores, decisions)
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
    seen = safe_binary_metrics(
        [rows[i] for i in seen_indexes], [scores[i] for i in seen_indexes], threshold
    )
    unseen = safe_binary_metrics(
        [rows[i] for i in unseen_indexes], [scores[i] for i in unseen_indexes], threshold
    )
    targets = np.stack([mask_for_row(row, maps[0].shape) for row in rows])
    normal_mask = np.asarray([row["label"] == "normal" for row in rows])
    localization = compute_localization_diagnostics(np.stack(maps), targets, normal_mask)
    area = np.asarray([float(audit["roi_area_fraction"]) for audit in audits])
    roi_count = np.asarray([int(audit["selected_roi_count"]) for audit in audits])
    return {
        "overall": overall,
        "seen": seen,
        "unseen": unseen,
        "localization": localization,
        "routing": {
            "mean_roi_area_fraction": float(area.mean()),
            "p95_roi_area_fraction": float(np.quantile(area, 0.95)),
            "max_roi_area_fraction": float(area.max()),
            "mean_roi_count": float(roi_count.mean()),
            "max_roi_count": int(roi_count.max()),
            "hidden_fallback": False,
        },
    }


def run_task(args: argparse.Namespace) -> dict[str, Any]:
    baseline_config = load_yaml(args.baseline_config)
    rcbr_config = load_yaml(args.rcbr_config)
    if tuple(rcbr_config["strategies"]) != STRATEGIES:
        raise RuntimeError(
            "RCBR strategy set/order changed; update the controlled protocol explicitly"
        )
    adaptation = read_csv(args.adaptation)
    support_normals = [row for row in adaptation if row["role"] == "support_normal"]
    support_anomalies = [row for row in adaptation if row["role"] == "support_anomaly"]
    development = [row for row in adaptation if row["role"] == "development"]
    if not support_normals or not support_anomalies or not development:
        raise RuntimeError("adaptation manifest lacks required roles")
    requested_train = int(baseline_config["normal_support_train"])
    train_count = min(requested_train, max(1, int(len(support_normals) * 0.8)))
    train_normals, calibration_normals = (
        support_normals[:train_count],
        support_normals[train_count:],
    )
    if len(calibration_normals) < 2:
        raise RuntimeError("at least two held-out support normals are required for calibration")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    model, training_info = train_efficientad(
        train_normals,
        calibration_normals,
        [row for row in development if row["label"] == "anomaly"],
        args.output_dir,
        baseline_config,
        args.seed,
    )
    input_shape = tuple(int(value) for value in baseline_config["input_resolution"])
    calibration_maps: list[NDArray[np.float32]] = []
    calibration_times: list[float] = []
    for row in calibration_normals:
        anomaly_map, elapsed = infer_array(model, load_rgb(row["path"]), input_shape)
        calibration_maps.append(anomaly_map)
        calibration_times.append(elapsed)
    normal_maps = np.stack(calibration_maps)
    calibrator = NormalRiskCalibrator.fit(normal_maps)
    utility_rows = (
        support_anomalies + support_normals[: min(len(support_anomalies), len(support_normals))]
    )
    utility_model, utility_info = fit_router(
        model,
        utility_rows,
        calibrator,
        normal_maps,
        input_shape,
        rcbr_config["router"],
    )
    roi_cost_ms = float(np.median(calibration_times))
    np.savez_compressed(
        args.output_dir / "router_state.npz",
        normal_maps=normal_maps,
        utility_feature_mean=utility_model.feature_mean,
        utility_feature_scale=utility_model.feature_scale,
        utility_coefficients=utility_model.coefficients,
        utility_intercept=np.asarray([utility_model.intercept], dtype=np.float64),
        roi_cost_ms=np.asarray([roi_cost_ms], dtype=np.float64),
    )

    quantile = float(baseline_config["inference"]["score_quantile"])
    dev_maps: dict[str, list[NDArray[np.float32]]] = {strategy: [] for strategy in STRATEGIES}
    dev_scores: dict[str, list[float]] = {strategy: [] for strategy in STRATEGIES}
    for row in development:
        outputs, _ = build_image_outputs(
            model,
            row,
            calibrator,
            normal_maps,
            utility_model,
            input_shape,
            rcbr_config,
            roi_cost_ms,
        )
        for strategy, anomaly_map in outputs.items():
            dev_maps[strategy].append(anomaly_map)
            dev_scores[strategy].append(score_map(anomaly_map, quantile))
    dev_labels = [int(row["label"] == "anomaly") for row in development]
    thresholds = {
        strategy: select_threshold(dev_scores[strategy], dev_labels) for strategy in STRATEGIES
    }

    # Test inputs have paths but no labels; truth is deliberately opened only after every
    # prediction map and routing decision has been fixed.
    test_inputs = read_csv(args.test_inputs)
    final_maps: dict[str, list[NDArray[np.float32]]] = {strategy: [] for strategy in STRATEGIES}
    final_scores: dict[str, list[float]] = {strategy: [] for strategy in STRATEGIES}
    final_audits: dict[str, list[dict[str, Any]]] = {strategy: [] for strategy in STRATEGIES}
    for row in test_inputs:
        outputs, audit = build_image_outputs(
            model,
            row,
            calibrator,
            normal_maps,
            utility_model,
            input_shape,
            rcbr_config,
            roi_cost_ms,
        )
        for strategy, anomaly_map in outputs.items():
            final_maps[strategy].append(anomaly_map)
            final_scores[strategy].append(score_map(anomaly_map, quantile))
            final_audits[strategy].append(audit[strategy])

    truth_by_id = {row["sample_id"]: row for row in read_csv(args.test_truth)}
    if set(truth_by_id) != {row["sample_id"] for row in test_inputs}:
        raise RuntimeError("test truth/input coverage mismatch")
    truth_rows = [truth_by_id[row["sample_id"]] for row in test_inputs]
    results: dict[str, Any] = {}
    mask_dir = args.output_dir / "masks"
    mask_dir.mkdir()
    for strategy in STRATEGIES:
        results[strategy] = evaluate_strategy(
            truth_rows,
            final_maps[strategy],
            final_scores[strategy],
            float(thresholds[strategy]["threshold"]),
            final_audits[strategy],
        )
        np.savez_compressed(
            mask_dir / f"{strategy}.npz", predictions=np.stack(final_maps[strategy])
        )
    audit_path = args.output_dir / "routing_audit.json"
    write_json(audit_path, {strategy: final_audits[strategy] for strategy in STRATEGIES})
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
        "baseline_config_hash": canonical_hash(baseline_config),
        "rcbr_config_hash": canonical_hash(rcbr_config),
        "git_commit": commit,
        "dirty": dirty,
        "upstream_anomalib_commit": baseline_config["upstream"]["commit"],
        "training": training_info,
        "calibration": {
            "normal_train_count": len(train_normals),
            "normal_calibration_count": len(calibration_normals),
            "development_count": len(development),
            "roi_cost_lookup_median_ms": roi_cost_ms,
        },
        "utility_cross_fit": utility_info,
        "thresholds_development_only": thresholds,
        "strategies": results,
        "routing_audit_path": str(audit_path),
        "hardware": platform.platform(),
        "physical_gpu": os.environ.get("EVOINSPECT_PHYSICAL_GPU", "UNRECORDED"),
        "warnings": [
            (
                "No test labels were used for training, routing, threshold selection, "
                "or early stopping."
            ),
            (
                "Latency in this task is RTX-3090 model-segment diagnostic, not 2500px "
                "end-to-end or RTX-2060 evidence."
            ),
        ],
    }
    write_json(args.output_dir / "metrics.json", metrics)
    return metrics


def register(args: argparse.Namespace, started_at: str, status: str, error: str = "") -> None:
    split = json.loads(args.split.read_text(encoding="utf-8"))
    baseline_config = load_yaml(args.baseline_config)
    rcbr_config = load_yaml(args.rcbr_config)
    metrics_path = args.output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    commit, dirty = git_state(Path.cwd())
    row = {
        "run_id": args.run_id,
        "status": status,
        "start_time": started_at,
        "end_time": utc_now(),
        "git_commit": commit,
        "dirty": str(dirty).lower(),
        "config_hash": canonical_hash({"baseline": baseline_config, "rcbr": rcbr_config}),
        "data_hash": split.get("source_manifest_sha256", "UNAVAILABLE"),
        "split_hash": split.get("split_hash", "UNAVAILABLE"),
        "seed": args.seed,
        "hardware": (
            f"physical GPU {os.environ.get('EVOINSPECT_PHYSICAL_GPU', 'UNRECORDED')}; "
            f"{platform.platform()}"
        ),
        "model": "efficientad-s-anomalib-2.3.0 + evoinspect-rcbr-v1",
        "protocol": split.get("protocol", "UNAVAILABLE"),
        "metrics_path": str(metrics_path) if metrics_path.is_file() else "",
        "artifact_path": str(args.output_dir / "model.ckpt")
        if (args.output_dir / "model.ckpt").is_file()
        else "",
        "failure_reason": error,
        "notes": (
            f"category={split.get('category')}; model_sha256="
            f"{metrics.get('training', {}).get('model_sha256', 'UNAVAILABLE')}"
        ),
    }
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.registry.with_suffix(args.registry.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        append_csv(args.registry, REGISTRY_COLUMNS, row)
        fcntl.flock(lock, fcntl.LOCK_UN)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Train one EfficientAD-S model and evaluate six RCBR controls"
    )
    root.add_argument("--adaptation", required=True, type=Path)
    root.add_argument("--test-inputs", required=True, type=Path)
    root.add_argument("--test-truth", required=True, type=Path)
    root.add_argument("--split", required=True, type=Path)
    root.add_argument("--baseline-config", required=True, type=Path)
    root.add_argument("--rcbr-config", required=True, type=Path)
    root.add_argument("--output-dir", required=True, type=Path)
    root.add_argument("--run-id", required=True)
    root.add_argument("--seed", required=True, type=int)
    root.add_argument("--registry", required=True, type=Path)
    return root


if __name__ == "__main__":
    parsed = parser().parse_args()
    command_started = utc_now()
    try:
        run_task(parsed)
        register(parsed, command_started, "completed_development_rcbr")
    except Exception as exception:
        parsed.output_dir.mkdir(parents=True, exist_ok=True)
        error_text = f"{type(exception).__name__}: {exception}"
        write_json(
            parsed.output_dir / "failure.json",
            {"status": "failed", "error": error_text, "traceback": traceback.format_exc()},
        )
        register(parsed, command_started, "failed_development_rcbr", error_text)
        raise
