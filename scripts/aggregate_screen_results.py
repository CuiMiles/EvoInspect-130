#!/usr/bin/env python3
"""Aggregate a bounded six-category screen without selecting on test outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evoinspect.provenance import file_sha256, utc_now, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    categories = [str(value) for value in config["categories"]]
    reports: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for category in categories:
        path = args.results_root / category / "metrics.json"
        if not path.is_file():
            missing.append(category)
            continue
        reports[category] = json.loads(path.read_text(encoding="utf-8"))
    completed = len(reports)
    leakage = sum(
        int(report.get("leakage_audit", {}).get("test_label_reads_before_all_predictions_fixed", 0))
        + int(report.get("leakage_audit", {}).get("test_labels_used_for_training", 0))
        + int(report.get("leakage_audit", {}).get("test_labels_used_for_threshold", 0))
        + int(report.get("leakage_audit", {}).get("test_labels_used_for_model_selection", 0))
        for report in reports.values()
    )
    overall = [
        float(report["result"]["overall"]["f1_fixed_threshold"])
        for report in reports.values()
    ]
    auroc = [float(report["result"]["overall"]["auroc"]) for report in reports.values()]
    unseen = [
        float(report["result"]["unseen"]["f1_fixed_threshold"])
        for report in reports.values()
        if report["result"].get("unseen") is not None
    ]
    gate = config["quality_gate"]
    means = {
        "overall_f1": float(np.mean(overall)) if overall else None,
        "unseen_f1_eligible_only": float(np.mean(unseen)) if unseen else None,
        "image_auroc": float(np.mean(auroc)) if auroc else None,
    }
    checks = {
        "all_categories_complete": completed == len(categories) and not missing,
        "overall_f1": means["overall_f1"] is not None
        and means["overall_f1"] >= float(gate["overall_f1_min"]),
        "unseen_f1": means["unseen_f1_eligible_only"] is not None
        and means["unseen_f1_eligible_only"] >= float(gate["unseen_f1_min"]),
        "image_auroc": means["image_auroc"] is not None
        and means["image_auroc"] >= float(gate["image_auroc_min"]),
        "test_label_leakage": leakage <= int(gate["test_label_leakage_max"]),
    }
    report = {
        "schema_version": 1,
        "status": "six_category_screen_aggregate",
        "created_at": utc_now(),
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "results_root": str(args.results_root),
        "categories": categories,
        "completed_categories": sorted(reports),
        "missing_categories": missing,
        "completed_runs": completed,
        "test_label_leakage_events": leakage,
        "metrics_macro_over_category_runs": means,
        "checks": checks,
        "passed": all(checks.values()),
        "per_category": {
            category: {
                "overall": report["result"]["overall"],
                "unseen": report["result"].get("unseen"),
                "latency_model_segment_ms": report.get("latency_model_segment_ms"),
                "leakage_audit": report.get("leakage_audit"),
            }
            for category, report in reports.items()
        },
        "claim_limit": (
            "Exploratory six-category screen only; no promotion to the final package "
            "without explicit review."
        ),
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "completed": completed,
                "missing": missing,
                "means": means,
                "passed": report["passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
