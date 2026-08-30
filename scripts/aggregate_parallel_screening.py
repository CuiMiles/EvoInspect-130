#!/usr/bin/env python3
"""Aggregate the fixed six-category exploratory screen without promoting claims."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evoinspect.provenance import canonical_hash, utc_now, write_json


def mean(values: list[float]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.batch_root.glob("runs/*/metrics.json"))
    path_records = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in paths
        if "check" not in path.parent.name
    ]
    records = [record for _, record in path_records]
    by_variant: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path, record in path_records:
        variant_id = str(
            record.get(
                "variant_id",
                f"efficientad_s_{record['resolution'][0]}"
                if record["variant"] == "global_single_forward"
                else record["variant"],
            )
        )
        by_variant[variant_id].append((path, record))
    summaries: dict[str, Any] = {}
    for variant, pairs in sorted(by_variant.items()):
        items = [record for _, record in pairs]
        overall = [float(item["result"]["overall"]["f1_fixed_threshold"]) for item in items]
        unseen = [
            float(item["result"]["unseen"]["f1_fixed_threshold"])
            for item in items
            if item["result"].get("unseen") is not None
        ]
        auroc = [float(item["result"]["overall"]["auroc"]) for item in items]
        summaries[variant] = {
            "completed_runs": len(items),
            "categories": sorted(str(item["category"]) for item in items),
            "overall_f1_mean": mean(overall),
            "unseen_f1_mean": mean(unseen),
            "image_auroc_mean": mean(auroc),
            "model_p95_ms_mean": mean(
                [float(item["latency_model_segment_ms"]["p95"]) for item in items]
            ),
            "leakage_events": sum(
                int(value) for item in items for value in item["leakage_audit"].values()
            ),
            "metric_paths": [str(path) for path, _ in pairs],
        }
    config = yaml.safe_load(
        Path("configs/experiments/parallel_screening_20260831.yaml").read_text(encoding="utf-8")
    )
    payload = {
        "schema_version": 1,
        "status": "parallel_screening_exploratory_aggregate",
        "created_at": utc_now(),
        "batch_root": str(args.batch_root.resolve()),
        "preregistration": "configs/experiments/parallel_screening_20260831.yaml",
        "completed_runs": len(records),
        "expected_runs": 18,
        "variants": summaries,
        "all_expected_runs_completed": len(records) == 18,
        "total_test_label_leakage_events": sum(
            int(value) for item in records for value in item["leakage_audit"].values()
        ),
        "promotion": {
            "automatic": False,
            "note": "Screening results require review; final submission files are unchanged.",
        },
        "config_hash": canonical_hash(config),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
