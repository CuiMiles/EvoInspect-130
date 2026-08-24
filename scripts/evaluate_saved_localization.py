from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]
from torchvision import transforms

from evoinspect.localization import compute_localization_diagnostics
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

EVALUATOR_CONFIG = {
    "schema_version": 3,
    "metric": "equal-region-weighted-aupro",
    "fpr_limits": [0.05, 0.30],
    "connectivity": 8,
    "curve": "exact global score sort; equal total weight per ground-truth region",
    "clip": "linear interpolation at FPR limit; normalized trapezoidal area",
    "mask_convention": (
        "pinned-upstream-compatible bilinear Resize(int), CenterCrop, ToTensor, uint8 cast"
    ),
    "operating_points": (
        "test-derived diagnostics only; forbidden for model selection; tied background "
        "score groups are included only when the complete group stays within the integer "
        "false-positive budget"
    ),
    "fixed_relative_area_slices": {
        "tiny": "area_fraction <= 0.001",
        "small": "0.001 < area_fraction <= 0.01",
        "large": "area_fraction > 0.01",
    },
}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def target_transform(config: dict[str, Any]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(config["input"]["resize"])),
            transforms.CenterCrop(int(config["input"]["crop"])),
            transforms.ToTensor(),
        ]
    )


def evaluate_run(run_dir_text: str) -> dict[str, Any]:
    run_dir = Path(run_dir_text)
    started = time.perf_counter()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "model" / "meta.json").read_text(encoding="utf-8"))
    truth = {row["sample_id"]: row for row in read_csv(run_dir / "test_truth.csv")}
    masks_path = run_dir / "predictions.masks.npz"
    with np.load(masks_path) as archive:
        sample_ids = [str(value) for value in archive["sample_ids"]]
        predictions = np.asarray(archive["masks"], dtype=np.float32)
    if set(sample_ids) != set(truth):
        raise RuntimeError(f"prediction/truth coverage mismatch: {run_dir}")
    transform = target_transform(meta["config"])
    targets: list[np.ndarray[Any, np.dtype[np.bool_]]] = []
    normal_image_mask: list[bool] = []
    for sample_id, prediction in zip(sample_ids, predictions, strict=True):
        row = truth[sample_id]
        is_normal = row["label"] == "normal"
        normal_image_mask.append(is_normal)
        if is_normal:
            target = np.zeros_like(prediction, dtype=np.bool_)
        else:
            with Image.open(row["mask_path"]) as image:
                transformed = transform(image).numpy()[0]
            # This deliberately matches the pinned upstream integer-cast convention used by
            # the existing P0/P1 pixel metrics rather than silently changing mask geometry.
            target = transformed.astype(np.uint8).astype(np.bool_)
        targets.append(target)
    target_array = np.stack(targets)
    normal_array = np.asarray(normal_image_mask, dtype=np.bool_)
    diagnostics = compute_localization_diagnostics(predictions, target_array, normal_array)
    flat_target = target_array.ravel().astype(np.uint8)
    flat_prediction = predictions.ravel()
    recomputed_pixel = {
        "full_pixel_auroc": float(roc_auc_score(flat_target, flat_prediction)),
        "full_pixel_average_precision": float(
            average_precision_score(flat_target, flat_prediction)
        ),
    }
    for key, value in recomputed_pixel.items():
        if abs(value - float(metrics["pixel"][key])) > 1e-12:
            raise RuntimeError(f"stored pixel metric mismatch for {run_dir}: {key}")
    return {
        "schema_version": 1,
        "status": "completed_saved_mask_localization_reevaluation",
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "category": metrics["category"],
        "seed": metrics["seed"],
        "model_hash": metrics["model_hash"],
        "split_hash": metrics["split_hash"],
        "source_masks_sha256": file_sha256(masks_path),
        "evaluator_config": EVALUATOR_CONFIG,
        "evaluator_config_hash": canonical_hash(EVALUATOR_CONFIG),
        "stored_pixel_metrics_verified": recomputed_pixel,
        "source_image_metrics": metrics.get("upstream_patchcore"),
        "localization": diagnostics,
        "evaluation_seconds": time.perf_counter() - started,
    }


def mean_metric(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return statistics.mean(values)


def _fixed_slice_mean(
    rows: list[dict[str, Any]], slice_name: str, metric_name: str
) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row["localization"]["fixed_relative_area_slices"][slice_name]
        if int(value.get("region_count", 0)) and metric_name in value:
            values.append(float(value[metric_name]))
    return statistics.mean(values) if values else None


def bootstrap_category_ci(values: list[float], seed: int = 130, draws: int = 10_000) -> list[float]:
    generator = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    samples = generator.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def aggregate(results: list[dict[str, Any]], source_aggregate: Path) -> dict[str, Any]:
    source = json.loads(source_aggregate.read_text(encoding="utf-8"))
    categories = sorted({str(row["category"]) for row in results})
    per_category: dict[str, Any] = {}
    for category in categories:
        rows = [row for row in results if row["category"] == category]
        per_category[category] = {
            "run_count": len(rows),
            "aupro_at_0_30": mean_metric(rows, ("localization", "curves", "0.30", "aupro")),
            "aupro_at_0_05": mean_metric(rows, ("localization", "curves", "0.05", "aupro")),
            "pro_at_fpr_0_01": mean_metric(
                rows, ("localization", "curves", "0.30", "pro_at_fpr_0_01")
            ),
            "small_region_aupro_at_0_30": mean_metric(
                rows,
                ("localization", "size_slices", "small_le_q25", "aupro_at_0.30"),
            ),
            "medium_region_aupro_at_0_30": mean_metric(
                rows,
                ("localization", "size_slices", "medium_q25_q75", "aupro_at_0.30"),
            ),
            "large_region_aupro_at_0_30": mean_metric(
                rows,
                ("localization", "size_slices", "large_gt_q75", "aupro_at_0.30"),
            ),
            "fixed_tiny_aupro_at_0_05": _fixed_slice_mean(rows, "tiny_le_0_001", "aupro_at_0.05"),
            "fixed_small_aupro_at_0_05": _fixed_slice_mean(
                rows, "small_0_001_0_01", "aupro_at_0.05"
            ),
            "fixed_large_aupro_at_0_05": _fixed_slice_mean(rows, "large_gt_0_01", "aupro_at_0.05"),
            "region_recall_overlap_0_30_at_test_fpr_0_05": mean_metric(
                rows,
                (
                    "localization",
                    "test_derived_operating_points_do_not_use_for_model_selection",
                    "fpr_0_05",
                    "region_recall_overlap_0_30",
                ),
            ),
            "false_positive_regions_per_normal_at_test_fpr_0_01": mean_metric(
                rows,
                (
                    "localization",
                    "test_derived_operating_points_do_not_use_for_model_selection",
                    "fpr_0_01",
                    "false_positive_regions_per_normal_image",
                ),
            ),
            "overall_f1": mean_metric(
                rows, ("source_image_metrics", "overall", "f1_fixed_threshold")
            ),
            "unseen_f1": mean_metric(
                rows, ("source_image_metrics", "unseen", "f1_fixed_threshold")
            ),
            "image_auroc": mean_metric(rows, ("source_image_metrics", "overall", "auroc")),
        }
    macro: dict[str, Any] = {}
    for metric in (
        "aupro_at_0_30",
        "aupro_at_0_05",
        "pro_at_fpr_0_01",
        "small_region_aupro_at_0_30",
        "medium_region_aupro_at_0_30",
        "large_region_aupro_at_0_30",
        "fixed_tiny_aupro_at_0_05",
        "fixed_small_aupro_at_0_05",
        "fixed_large_aupro_at_0_05",
        "region_recall_overlap_0_30_at_test_fpr_0_05",
        "false_positive_regions_per_normal_at_test_fpr_0_01",
        "overall_f1",
        "unseen_f1",
        "image_auroc",
    ):
        values = [
            float(per_category[category][metric])
            for category in categories
            if per_category[category][metric] is not None
        ]
        macro[metric] = {
            "mean": statistics.mean(values),
            "category_count": len(values),
            "category_bootstrap_95_ci": bootstrap_category_ci(values),
        }
    return {
        "schema_version": 1,
        "status": "completed_upstream_patchcore_saved_mask_localization_reevaluation",
        "created_at": now(),
        "protocol": source.get("protocol", "official_style_up_to_100_normal_30_seen_anomaly"),
        "dataset": "MVTec_AD_direct_archive",
        "source_aggregate": str(source_aggregate),
        "source_aggregate_sha256": file_sha256(source_aggregate),
        "evaluator_config": EVALUATOR_CONFIG,
        "evaluator_config_hash": canonical_hash(EVALUATOR_CONFIG),
        "categories": categories,
        "seeds": sorted({int(row["seed"]) for row in results}),
        "run_count": len(results),
        "per_category": per_category,
        "macro_category_mean": macro,
        "runs": [row["run_id"] for row in results],
        "warning": (
            "AUPRO is threshold-independent. Operating-point diagnostics use final-test truth "
            "and are forbidden for threshold tuning or model selection."
        ),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--source-aggregate", required=True, type=Path)
    value.add_argument("--output-dir", required=True, type=Path)
    value.add_argument("--registry", required=True, type=Path)
    value.add_argument("--run-id", required=True)
    value.add_argument("--workers", type=int, default=4)
    value.add_argument("--expected-runs", type=int, default=75)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.workers < 1 or args.workers > 8:
        raise RuntimeError("workers must be in [1, 8]")
    source = json.loads(args.source_aggregate.read_text(encoding="utf-8"))
    run_dirs = [str(path) for path in source["runs"]]
    if len(run_dirs) != args.expected_runs:
        raise RuntimeError(f"expected {args.expected_runs} source runs, got {len(run_dirs)}")
    if args.output_dir.exists():
        raise RuntimeError(f"output directory exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started_at = now()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_run = {executor.submit(evaluate_run, path): path for path in run_dirs}
        for future in as_completed(future_to_run):
            run_dir = future_to_run[future]
            try:
                result = future.result()
                results.append(result)
                write_json(args.output_dir / "runs" / f"{result['run_id']}.json", result)
                print(f"PASS {len(results)}/{len(run_dirs)} {result['run_id']}", flush=True)
            except Exception as error:
                failures.append({"run_dir": run_dir, "error": repr(error)})
                print(f"FAIL {run_dir}: {error!r}", flush=True)
    write_json(args.output_dir / "failures.json", failures)
    if failures or len(results) != len(run_dirs):
        raise RuntimeError(f"localization reevaluation incomplete: {len(failures)} failures")
    results.sort(key=lambda row: (str(row["category"]), int(row["seed"])))
    summary = aggregate(results, args.source_aggregate)
    write_json(args.output_dir / "aggregate.json", summary)
    commit, dirty = git_state(Path.cwd())
    append_csv(
        args.registry,
        REGISTRY_COLUMNS,
        {
            "run_id": args.run_id,
            "status": "completed_localization_reevaluation",
            "start_time": started_at,
            "end_time": now(),
            "git_commit": commit,
            "dirty": str(dirty).lower(),
            "config_hash": canonical_hash(EVALUATOR_CONFIG),
            "data_hash": file_sha256(args.source_aggregate),
            "split_hash": canonical_hash(sorted(row["split_hash"] for row in results)),
            "seed": "133-137",
            "hardware": f"CPU reevaluation; {platform.platform()}; workers={args.workers}",
            "model": "upstream-patchcore-fcaa92f-saved-masks",
            "protocol": summary["protocol"],
            "metrics_path": str(args.output_dir / "aggregate.json"),
            "artifact_path": str(args.output_dir),
            "failure_reason": "",
            "notes": (
                "AUPRO@0.30/@0.05 and region diagnostics; final-test-derived operating points "
                "are explicitly forbidden for model selection"
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
