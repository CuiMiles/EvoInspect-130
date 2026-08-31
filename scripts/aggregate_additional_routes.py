#!/usr/bin/env python3
"""Aggregate the bounded additional-route six-category screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evoinspect.provenance import utc_now, write_json

CATEGORIES = ("cable", "capsule", "screw", "carpet", "transistor", "wood")
ROUTES = ("defectadapter", "supersimplenet", "dra")


def metric_path(root: Path, route: str, category: str) -> Path:
    scope = "smoke" if category == "cable" else "runs"
    return root / scope / f"{route}-{category}" / "metrics.json"


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate(root: Path, config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gate = config["quality_screen"]
    routes: dict[str, Any] = {}
    for route in ROUTES:
        runs = []
        for category in CATEGORIES:
            path = metric_path(root, route, category)
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if metrics["route"] != route or metrics["category"] != f"mvtec_ad_{category}":
                raise RuntimeError(f"identity mismatch: {path}")
            if metrics["dirty"]:
                raise RuntimeError(f"dirty experiment: {path}")
            if any(metrics["leakage_audit"].values()):
                raise RuntimeError(f"leakage audit failed: {path}")
            runs.append(metrics)
        overall_f1 = mean([run["result"]["overall"]["f1_fixed_threshold"] for run in runs])
        unseen_runs = [run for run in runs if run["result"].get("unseen") is not None]
        unseen_f1 = mean([run["result"]["unseen"]["f1_fixed_threshold"] for run in unseen_runs])
        image_auroc = mean([run["result"]["overall"]["auroc"] for run in runs])
        checks = {
            "all_six_categories_complete": len(runs) == len(CATEGORIES),
            "overall_f1": overall_f1 >= float(gate["overall_f1_min"]),
            "unseen_f1": unseen_f1 >= float(gate["unseen_f1_min"]),
            "image_auroc": image_auroc >= float(gate["image_auroc_min"]),
            "test_label_leakage_zero": True,
        }
        routes[route] = {
            "run_count": len(runs),
            "categories": list(CATEGORIES),
            "overall_f1_mean": overall_f1,
            "unseen_f1_mean": unseen_f1,
            "image_auroc_mean": image_auroc,
            "model_p95_ms_mean": mean([run["latency_model_segment_ms"]["p95"] for run in runs]),
            "checks": checks,
            "passed": all(checks.values()),
            "metrics_paths": [str(metric_path(root, route, category)) for category in CATEGORIES],
        }
    return {
        "schema_version": 1,
        "status": "completed",
        "generated_at": utc_now(),
        "scope": "six categories, seed143, exploratory strict 100+30 screen",
        "routes": routes,
        "ahl_unlocked": bool(routes["dra"]["passed"]),
        "decision": (
            "AHL feature pipeline unlocked because DRA passed."
            if routes["dra"]["passed"]
            else "AHL feature pipeline not unlocked because DRA base failed the preregistered gate."
        ),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("reports/experiments/additional-routes-screen-20260831")
    )
    value.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/additional_routes_screen_20260831.yaml"),
    )
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    summary = aggregate(args.root, args.config)
    output = args.root / "additional-routes-summary.json"
    write_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
