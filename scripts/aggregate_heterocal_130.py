#!/usr/bin/env python3
"""Aggregate the five preregistered HeteroCal-130 ablations."""

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
    return float(np.mean(values)) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = sorted(args.batch_root.glob("runs/*/heterocal_result/metrics.json"))
    runs: list[dict[str, Any]] = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    categories = Counter(str(run["category"]) for run in runs)
    strategies = list(config["ablations"])
    summaries: dict[str, Any] = {}
    for strategy in strategies:
        eligible = [run for run in runs if run["results"][strategy]["unseen"] is not None]
        summaries[strategy] = {
            "overall_f1": mean(
                [float(run["results"][strategy]["overall"]["f1_fixed_threshold"]) for run in runs]
            ),
            "unseen_f1": mean(
                [
                    float(run["results"][strategy]["unseen"]["f1_fixed_threshold"])
                    for run in eligible
                ]
            ),
            "image_auroc": mean(
                [float(run["results"][strategy]["overall"]["auroc"]) for run in runs]
            ),
            "eligible_unseen_runs": len(eligible),
        }
    complete = summaries["heterocal_130"]
    baseline_by_category = {
        category: mean(
            [
                float(run["results"]["efficientad_m"]["overall"]["f1_fixed_threshold"])
                for run in runs
                if run["category"] == category
            ]
        )
        for category in categories
    }
    complete_by_category = {
        category: mean(
            [
                float(run["results"]["heterocal_130"]["overall"]["f1_fixed_threshold"])
                for run in runs
                if run["category"] == category
            ]
        )
        for category in categories
    }
    drops = [
        float(baseline_by_category[key]) - float(complete_by_category[key])
        for key in categories
        if baseline_by_category[key] is not None and complete_by_category[key] is not None
    ]
    leakage = sum(int(value) for run in runs for value in run["leakage_audit"].values())
    accepted_runs = sum(bool(run["support"]["full_selection"]["accepted"]) for run in runs)
    eligible_selection_runs = sum(
        run["support"]["full_selection"]["reason"] != "insufficient_defect_types" for run in runs
    )
    gate = config["quality_gate"]
    checks = {
        "all_expected_runs_completed": len(runs) == int(config["model_policy"]["expected_runs"]),
        "all_categories_present": len(categories)
        == int(config["model_policy"]["required_categories"]),
        "overall_f1": complete["overall_f1"] is not None
        and complete["overall_f1"] >= float(gate["overall_f1_min"]),
        "unseen_f1": complete["unseen_f1"] is not None
        and complete["unseen_f1"] >= float(gate["unseen_f1_min"]),
        "image_auroc": complete["image_auroc"] is not None
        and complete["image_auroc"] >= float(gate["image_auroc_min"]),
        "test_label_leakage": leakage <= int(gate["test_label_leakage_max"]),
        "worst_category_f1_drop": bool(drops)
        and max(drops) <= float(gate["worst_category_f1_drop_max"]),
        "support_loo_positive": accepted_runs > 0 if gate["support_loo_must_be_positive"] else True,
    }
    report = {
        "schema_version": 1,
        "status": "heterocal_130_preregistered_quality_gate",
        "created_at": utc_now(),
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "completed_runs": len(runs),
        "categories": dict(sorted(categories.items())),
        "ablations": summaries,
        "support_loo": {
            "eligible_runs": eligible_selection_runs,
            "accepted_runs": accepted_runs,
            "accepted_rate_all_runs": accepted_runs / len(runs) if runs else 0.0,
        },
        "worst_category_f1_drop": max(drops) if drops else None,
        "test_label_leakage_events": leakage,
        "checks": checks,
        "passed": all(checks.values()),
        "metric_paths": [str(path) for path in paths],
    }
    write_json(args.output, report)
    print(json.dumps({"passed": report["passed"], "complete": complete}, sort_keys=True))
    return 0 if report["passed"] or not args.enforce else 2


if __name__ == "__main__":
    raise SystemExit(main())
