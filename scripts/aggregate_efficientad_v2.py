#!/usr/bin/env python3
"""Aggregate strict EfficientAD evaluator-v2 results and enforce frozen gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evoinspect.provenance import file_sha256, utc_now, write_json


def mean(values: list[float]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gate = config["quality_gate"]
    unseen_policy = config["unseen_policy"]
    paths = sorted(args.batch_root.glob("runs/*/strict_result_v2/metrics.json"))
    runs: list[dict[str, Any]] = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    categories = Counter(str(run["category"]) for run in runs)
    eligible = [run for run in runs if run["unseen_eligible"]]
    eligible_categories = sorted({str(run["category"]) for run in eligible})
    leakage = sum(int(value) for run in runs for value in run["leakage_audit"].values())
    metrics = {
        "overall_f1": mean(
            [float(run["result"]["overall"]["f1_fixed_threshold"]) for run in runs]
        ),
        "unseen_f1_eligible_only": mean(
            [float(run["result"]["unseen"]["f1_fixed_threshold"]) for run in eligible]
        ),
        "image_auroc": mean([float(run["result"]["overall"]["auroc"]) for run in runs]),
    }
    checks = {
        "all_expected_runs_completed": len(runs) == int(gate["expected_runs"]),
        "all_15_categories_present": len(categories) == int(gate["required_categories"]),
        "unseen_coverage": len(eligible) == int(unseen_policy["expected_eligible_runs"])
        and len(eligible_categories) == int(unseen_policy["expected_eligible_categories"]),
        "overall_f1": metrics["overall_f1"] is not None
        and metrics["overall_f1"] >= float(gate["overall_f1_min"]),
        "unseen_f1": metrics["unseen_f1_eligible_only"] is not None
        and metrics["unseen_f1_eligible_only"] >= float(gate["unseen_f1_min"]),
        "image_auroc": metrics["image_auroc"] is not None
        and metrics["image_auroc"] >= float(gate["image_auroc_min"]),
        "test_label_leakage": leakage <= int(gate["test_label_leakage_max"]),
    }
    report = {
        "schema_version": 2,
        "status": "strict_100_30_efficientad_quality_gate",
        "created_at": utc_now(),
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "completed_runs": len(runs),
        "categories": dict(sorted(categories.items())),
        "unseen_coverage": {
            "eligible_runs": len(eligible),
            "eligible_categories": eligible_categories,
            "ineligible_runs": len(runs) - len(eligible),
        },
        "metrics_macro_over_category_seed_runs": metrics,
        "test_label_leakage_events": leakage,
        "checks": checks,
        "passed": all(checks.values()),
        "metric_paths": [str(path) for path in paths],
    }
    write_json(args.output, report)
    print(json.dumps({"passed": report["passed"], **metrics}, sort_keys=True))
    return 0 if report["passed"] or not args.enforce else 2


if __name__ == "__main__":
    raise SystemExit(main())
