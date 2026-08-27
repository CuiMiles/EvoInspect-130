#!/usr/bin/env python3
"""Aggregate the frozen 15-category EfficientAD quality gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evoinspect.provenance import file_sha256, utc_now, write_json


def mean_metric(runs: list[dict[str, Any]], slice_name: str, key: str) -> float:
    values = [run["result"][slice_name][key] for run in runs]
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-runs", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gate = config["quality_gate"]
    metric_paths = sorted(args.batch_root.glob("runs/*/result/metrics.json"))
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    failures = sorted(str(path) for path in args.batch_root.glob("runs/*/result/failure.json"))
    categories = Counter(str(run["category"]) for run in runs)
    leakage = sum(int(value) for run in runs for value in run["leakage_audit"].values())
    metrics = {
        "overall_f1": mean_metric(runs, "overall", "f1_fixed_threshold") if runs else None,
        "unseen_f1": mean_metric(runs, "unseen", "f1_fixed_threshold") if runs else None,
        "image_auroc": mean_metric(runs, "overall", "auroc") if runs else None,
    }
    checks = {
        "all_expected_runs_completed": len(runs) == args.expected_runs and not failures,
        "all_15_categories_present": len(categories) == int(gate["required_categories"]),
        "overall_f1": metrics["overall_f1"] is not None
        and metrics["overall_f1"] >= float(gate["overall_f1_min"]),
        "unseen_f1": metrics["unseen_f1"] is not None
        and metrics["unseen_f1"] >= float(gate["unseen_f1_min"]),
        "image_auroc": metrics["image_auroc"] is not None
        and metrics["image_auroc"] >= float(gate["image_auroc_min"]),
        "test_label_leakage": leakage <= int(gate["test_label_leakage_max"]),
    }
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "model_id": config["model_id"],
        "config_path": str(args.config),
        "config_sha256": file_sha256(args.config),
        "expected_runs": args.expected_runs,
        "completed_runs": len(runs),
        "categories": dict(sorted(categories.items())),
        "failures": failures,
        "metrics_macro_over_category_seed_runs": metrics,
        "test_label_leakage_events": leakage,
        "quality_gate": gate,
        "checks": checks,
        "passed": all(checks.values()),
        "metric_paths": [str(path) for path in metric_paths],
    }
    write_json(args.output, report)
    print(json.dumps({"passed": report["passed"], **metrics}, sort_keys=True))
    return 0 if report["passed"] or not args.enforce else 2


if __name__ == "__main__":
    raise SystemExit(main())
