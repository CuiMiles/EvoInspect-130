#!/usr/bin/env python3
"""Aggregate the pre-registered six-category HeteroResidual screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evoinspect.provenance import canonical_hash, utc_now, write_json

EXPECTED_CATEGORIES = ("cable", "capsule", "screw", "carpet", "transistor", "wood")


def average(values: list[float]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


def category_from_path(path: Path) -> str:
    name = path.parent.name
    prefix = "heteroresidual_s-"
    suffix = "-s143"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.batch_root.glob("runs/heteroresidual_s-*-s143/metrics.json"))
    records: list[tuple[Path, dict[str, Any]]] = [
        (path, json.loads(path.read_text(encoding="utf-8"))) for path in paths
    ]
    categories = sorted(category_from_path(path) for path, _ in records)
    overall = [float(record["result"]["overall"]["f1_fixed_threshold"]) for _, record in records]
    unseen = [
        float(record["result"]["unseen"]["f1_fixed_threshold"])
        for _, record in records
        if record["result"].get("unseen") is not None
    ]
    auroc = [float(record["result"]["overall"]["auroc"]) for _, record in records]
    payload = {
        "schema_version": 1,
        "status": "heteroresidual_screen_exploratory_aggregate",
        "created_at": utc_now(),
        "batch_root": str(args.batch_root.resolve()),
        "preregistration": "configs/experiments/heteroresidual_screen_20260831.yaml",
        "completed_runs": len(records),
        "expected_runs": len(EXPECTED_CATEGORIES),
        "categories": categories,
        "all_expected_runs_completed": categories == sorted(EXPECTED_CATEGORIES),
        "overall_f1_mean": average(overall),
        "unseen_f1_mean": average(unseen),
        "image_auroc_mean": average(auroc),
        "model_p95_ms_mean": average(
            [float(record["latency_model_segment_ms"]["p95"]) for _, record in records]
        ),
        "leakage_events": sum(
            int(value) for _, record in records for value in record["leakage_audit"].values()
        ),
        "metric_paths": [str(path) for path, _ in records],
        "promotion": {
            "automatic": False,
            "note": "Exploratory only; final submission files remain unchanged.",
        },
    }
    config = yaml.safe_load(
        Path("configs/experiments/heteroresidual_screen_20260831.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["config_hash"] = canonical_hash(config)
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
