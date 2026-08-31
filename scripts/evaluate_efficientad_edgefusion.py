#!/usr/bin/env python3
"""Evaluate the preregistered fixed EfficientAD-S+M EdgeFusion.

The script consumes only completed strict-v2.1 prediction artifacts.  Model
weights are not loaded and no GPU is required.  Each fused prediction file is
durably written before the corresponding test-truth file is opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evoinspect.provenance import utc_now, write_json
from scripts.efficientad_rcbr_100_30 import evaluate_strategy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def metric_paths(batch: Path, seed: int) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(batch.glob("runs/*/strict_result_v2/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if int(metrics["seed"]) == seed:
            category = str(metrics["category"])
            if category in result:
                raise RuntimeError(f"duplicate seed/category in {batch}: {category}")
            result[category] = path
    return result


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


def category_run(
    m_metric_path: Path,
    s_metric_path: Path,
    output_dir: Path,
    alpha_m: float,
    alpha_s: float,
) -> dict[str, Any]:
    m_metrics = json.loads(m_metric_path.read_text(encoding="utf-8"))
    s_metrics = json.loads(s_metric_path.read_text(encoding="utf-8"))
    if m_metrics["category"] != s_metrics["category"] or m_metrics["seed"] != s_metrics["seed"]:
        raise RuntimeError("M/S category or seed mismatch")
    m_run = m_metric_path.parent.parent
    m_pred_path = m_metric_path.parent / "predictions.jsonl"
    s_pred_path = s_metric_path.parent / "predictions.jsonl"
    m_map_path = m_metric_path.parent / "prediction_maps.npz"
    s_map_path = s_metric_path.parent / "prediction_maps.npz"
    m_predictions = read_jsonl(m_pred_path)
    s_predictions = read_jsonl(s_pred_path)
    m_maps = np.asarray(np.load(m_map_path)["predictions"], dtype=np.float32)
    s_maps = np.asarray(np.load(s_map_path)["predictions"], dtype=np.float32)
    m_by_id = {str(row["sample_id"]): row for row in m_predictions}
    s_by_id = {str(row["sample_id"]): row for row in s_predictions}
    if len(m_by_id) != len(m_predictions) or len(s_by_id) != len(s_predictions):
        raise RuntimeError("duplicate sample_id in source predictions")
    if (
        set(m_by_id) != set(s_by_id)
        or len(m_predictions) != len(m_maps)
        or len(s_predictions) != len(s_maps)
    ):
        raise RuntimeError("M/S prediction and map coverage mismatch")
    ordered_ids = [str(row["sample_id"]) for row in m_predictions]
    s_index = {str(row["sample_id"]): index for index, row in enumerate(s_predictions)}
    s_maps_ordered = np.stack([s_maps[s_index[sample_id]] for sample_id in ordered_ids])
    m_threshold = float(m_metrics["calibration"]["threshold"]["threshold"])
    s_threshold = float(s_metrics["calibration"]["threshold"]["threshold"])
    if not (
        np.isfinite(m_threshold)
        and np.isfinite(s_threshold)
        and m_threshold > 0
        and s_threshold > 0
    ):
        raise RuntimeError("source thresholds must be positive finite support-derived values")
    fused_maps = (
        alpha_m * (m_maps / np.float32(m_threshold))
        + alpha_s * (s_maps_ordered / np.float32(s_threshold))
    ).astype(np.float32)
    fused_scores = np.max(fused_maps.reshape(len(fused_maps), -1), axis=1).astype(np.float64)

    test_inputs_path = m_run / "test_inputs.csv"
    test_inputs = read_csv(test_inputs_path)
    if [str(row["sample_id"]) for row in test_inputs] != ordered_ids:
        raise RuntimeError("source prediction order does not match test input manifest")
    if any(row.get("label") or row.get("defect_type") for row in test_inputs):
        raise RuntimeError("source test input manifest exposes labels")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.inprogress-", dir=output_dir.parent)
    )
    try:
        prediction_path = temporary / "predictions.jsonl"
        with prediction_path.open("x", encoding="utf-8") as stream:
            for sample_id, score in zip(ordered_ids, fused_scores, strict=True):
                stream.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "score": float(score),
                            "decision": int(score >= 1.0),
                            "model_ms": None,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        map_path = temporary / "prediction_maps.npz"
        np.savez_compressed(map_path, predictions=fused_maps)

        # The truth file is deliberately opened only after every fused prediction
        # and map is durable on disk.
        truth_path = m_run / "test_truth.csv"
        truth_by_id = {row["sample_id"]: row for row in read_csv(truth_path)}
        if set(truth_by_id) != set(ordered_ids):
            raise RuntimeError("test truth/input coverage mismatch")
        truth_rows = [truth_by_id[sample_id] for sample_id in ordered_ids]
        result = evaluate_strategy(
            truth_rows,
            [fused_maps[index] for index in range(len(fused_maps))],
            [float(score) for score in fused_scores],
            1.0,
            [zero_audit() for _ in fused_scores],
        )
        report = {
            "schema_version": 1,
            "status": "completed_efficientad_sm_edgefusion",
            "run_id": f"efficientad-sm-edgefusion-{m_metrics['category']}-s{m_metrics['seed']}",
            "created_at": utc_now(),
            "category": m_metrics["category"],
            "seed": int(m_metrics["seed"]),
            "protocol": "strict_100_30_evaluator_v2_1_fixed_prediction_fusion",
            "fusion": {
                "alpha_m": alpha_m,
                "alpha_s": alpha_s,
                "normalization": "divide_each_map_by_its_support_threshold",
                "m_support_threshold": m_threshold,
                "s_support_threshold": s_threshold,
                "score": "amax_of_fused_map",
                "threshold": 1.0,
            },
            "result": result,
            "leakage_audit": {
                "test_label_reads_before_all_predictions_fixed": 0,
                "test_labels_used_for_training": 0,
                "test_labels_used_for_threshold": 0,
                "test_labels_used_for_model_selection": 0,
            },
            "source": {
                "m_metrics": str(m_metric_path),
                "m_metrics_sha256": sha256(m_metric_path),
                "s_metrics": str(s_metric_path),
                "s_metrics_sha256": sha256(s_metric_path),
                "m_prediction_maps_sha256": sha256(m_map_path),
                "s_prediction_maps_sha256": sha256(s_map_path),
                "test_inputs_sha256": sha256(test_inputs_path),
                "test_truth_sha256": sha256(truth_path),
            },
            "artifacts": {
                "predictions": str(output_dir / "predictions.jsonl"),
                "prediction_maps": str(output_dir / "prediction_maps.npz"),
            },
            "hardware": platform.platform(),
            "warnings": [
                (
                    "This is a fixed fusion of completed source predictions; "
                    "no source checkpoint was retrained."
                ),
                (
                    "Support-derived thresholds and alpha=0.5 were fixed in the "
                    "preregistration before truth evaluation."
                ),
                (
                    "No latency is claimed until the fixed representative pair is "
                    "measured on the GTX2060 protocol."
                ),
            ],
        }
        write_json(temporary / "metrics.json", report)
        os.replace(temporary, output_dir)
        return report
    except Exception:
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temporary.rmdir()
        raise


def aggregate(reports: list[dict[str, Any]], config_path: Path) -> dict[str, Any]:
    runs = [report["result"] for report in reports]
    eligible = [item for item in runs if item["unseen"] is not None]
    metrics = {
        "overall_f1": float(np.mean([item["overall"]["f1_fixed_threshold"] for item in runs])),
        "unseen_f1_eligible_only": float(
            np.mean([item["unseen"]["f1_fixed_threshold"] for item in eligible])
        ),
        "image_auroc": float(np.mean([item["overall"]["auroc"] for item in runs])),
    }
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gate = config["quality_gate"]
    leakage = sum(
        int(value)
        for report in reports
        for value in report["leakage_audit"].values()
    )
    checks = {
        "all_expected_runs_completed": len(reports) == int(gate["expected_runs"]),
        "all_15_categories_present": len({report["category"] for report in reports})
        == int(gate["required_categories"]),
        "overall_f1": metrics["overall_f1"] >= float(gate["overall_f1_min"]),
        "unseen_f1": metrics["unseen_f1_eligible_only"] >= float(gate["unseen_f1_min"]),
        "image_auroc": metrics["image_auroc"] >= float(gate["image_auroc_min"]),
        "test_label_leakage": leakage <= int(gate["test_label_leakage_max"]),
    }
    return {
        "schema_version": 1,
        "status": "efficientad_sm_edgefusion_quality_gate",
        "created_at": utc_now(),
        "config": str(config_path),
        "completed_runs": len(reports),
        "categories": sorted(report["category"] for report in reports),
        "unseen_eligible_runs": len(eligible),
        "metrics_macro_over_category_runs": metrics,
        "test_label_leakage_events": leakage,
        "checks": checks,
        "passed": all(checks.values()),
        "metric_paths": [
            report["artifacts"]["predictions"].replace("predictions.jsonl", "metrics.json")
            for report in reports
        ],
        "promotion": "not_promoted_until_quality_and_gtx2060_speed_both_pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    m_paths = metric_paths(Path(config["m_batch"]), seed)
    s_paths = metric_paths(Path(config["s_batch"]), seed)
    categories = [str(value) for value in config["categories"]]
    expected = {f"mvtec_ad_{category}" for category in categories}
    if set(m_paths) != expected or set(s_paths) != expected:
        raise RuntimeError(
            f"source category mismatch; M missing={sorted(expected - set(m_paths))}, "
            f"S missing={sorted(expected - set(s_paths))}"
        )
    alpha_m = float(config["fusion"]["alpha_m"])
    alpha_s = float(config["fusion"]["alpha_s"])
    if abs(alpha_m + alpha_s - 1.0) > 1e-12:
        raise RuntimeError("fusion weights must sum to one")
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for category in sorted(expected):
        report = category_run(
            m_paths[category],
            s_paths[category],
            args.output_root / category,
            alpha_m,
            alpha_s,
        )
        reports.append(report)
        print(category, json.dumps(report["result"]["overall"], sort_keys=True))
    summary = aggregate(reports, args.config)
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
