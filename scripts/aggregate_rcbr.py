from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from evoinspect.provenance import file_sha256, utc_now, write_json


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_strategy(metrics: dict[str, Any], strategy: str) -> dict[str, float | None]:
    result = metrics["strategies"][strategy]
    localization = result["localization"]
    small = localization["fixed_relative_area_slices"]["small_0_001_0_01"]
    return {
        "aupro_0_05": float(localization["curves"]["0.05"]["aupro"]),
        "aupro_0_30": float(localization["curves"]["0.30"]["aupro"]),
        "pro_at_fpr_0_01": float(localization["curves"]["0.05"]["pro_at_fpr_0_01"]),
        "fixed_small_aupro_0_05": (
            float(small["aupro_at_0.05"]) if int(small["region_count"]) else None
        ),
        "overall_f1": float(result["overall"]["f1_fixed_threshold"]),
        "unseen_f1": (float(result["unseen"]["f1_fixed_threshold"]) if result["unseen"] else None),
        "image_auroc": float(result["overall"]["auroc"]),
        "mean_roi_area_fraction": float(result["routing"]["mean_roi_area_fraction"]),
        "p95_roi_area_fraction": float(result["routing"]["p95_roi_area_fraction"]),
    }


def mean_present(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.mean(present) if present else None


def bootstrap_ci(values: list[float], seed: int, draws: int) -> list[float]:
    generator = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    samples = generator.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def reference_row(reference: dict[str, Any], category: str) -> dict[str, float | None]:
    row = reference["per_category"][category]
    return {
        "aupro_0_05": float(row["aupro_at_0_05"]),
        "aupro_0_30": float(row["aupro_at_0_30"]),
        "pro_at_fpr_0_01": float(row["pro_at_fpr_0_01"]),
        "fixed_small_aupro_0_05": (
            float(row["fixed_small_aupro_at_0_05"])
            if row["fixed_small_aupro_at_0_05"] is not None
            else None
        ),
        "overall_f1": float(row["overall_f1"]),
        "unseen_f1": float(row["unseen_f1"]),
        "image_auroc": float(row["image_auroc"]),
        "mean_roi_area_fraction": 0.0,
        "p95_roi_area_fraction": 0.0,
    }


def main() -> int:
    args = parser().parse_args()
    config = __import__("yaml").safe_load(args.config.read_text(encoding="utf-8"))
    metric_files = sorted(args.batch_root.glob("runs/*/result/metrics.json"))
    if len(metric_files) != args.expected_runs:
        raise RuntimeError(
            f"expected {args.expected_runs} completed runs, found {len(metric_files)}"
        )
    runs = [load(path) for path in metric_files]
    identities = [(row["category"], int(row["seed"])) for row in runs]
    if len(set(identities)) != len(identities):
        raise RuntimeError("duplicate category/seed RCBR runs")
    reference = load(args.patchcore_reference)
    categories = sorted({str(row["category"]) for row in runs})
    seeds = sorted({int(row["seed"]) for row in runs})
    expected_per_category = len(seeds)
    per_category: dict[str, Any] = {}
    for category in categories:
        category_runs = [row for row in runs if row["category"] == category]
        if len(category_runs) != expected_per_category:
            raise RuntimeError(f"incomplete category {category}")
        strategies: dict[str, Any] = {}
        for strategy in config["strategies"]:
            rows = [extract_strategy(run, strategy) for run in category_runs]
            strategies[strategy] = {
                key: mean_present([row[key] for row in rows]) for key in rows[0]
            }
        patchcore = reference_row(reference, category)
        delta = {
            key: (
                float(strategies["full_rcbr"][key]) - float(patchcore[key])
                if strategies["full_rcbr"][key] is not None and patchcore[key] is not None
                else None
            )
            for key in patchcore
        }
        per_category[category] = {
            "run_count": len(category_runs),
            "strategies": strategies,
            "patchcore_reference": patchcore,
            "full_rcbr_minus_patchcore": delta,
        }

    metrics = [
        "aupro_0_05",
        "aupro_0_30",
        "pro_at_fpr_0_01",
        "fixed_small_aupro_0_05",
        "overall_f1",
        "unseen_f1",
        "image_auroc",
        "mean_roi_area_fraction",
        "p95_roi_area_fraction",
    ]
    macro: dict[str, Any] = {}
    for metric in metrics:
        values = [
            per_category[category]["strategies"]["full_rcbr"][metric] for category in categories
        ]
        reference_values = [
            per_category[category]["patchcore_reference"][metric] for category in categories
        ]
        deltas = [
            float(value) - float(reference_value)
            for value, reference_value in zip(values, reference_values, strict=True)
            if value is not None and reference_value is not None
        ]
        macro[metric] = {
            "full_rcbr": mean_present(values),
            "patchcore_reference": mean_present(reference_values),
            "delta": statistics.mean(deltas) if deltas else None,
            "paired_category_bootstrap_95_ci": (
                bootstrap_ci(
                    deltas,
                    int(config["gates"]["bootstrap"]["seed"]),
                    int(config["gates"]["bootstrap"]["draws"]),
                )
                if deltas
                else None
            ),
        }

    gate_config = config["gates"][args.gate]
    delta_aupro = float(macro["aupro_0_05"]["delta"])
    category_deltas = [
        float(per_category[category]["full_rcbr_minus_patchcore"]["aupro_0_05"])
        for category in categories
    ]
    if args.gate == "smoke":
        checks = {
            "mean_delta_aupro_0_05": delta_aupro >= float(gate_config["mean_delta_aupro_0_05_min"]),
            "categories_non_decreasing": sum(value >= 0 for value in category_deltas)
            >= int(gate_config["categories_non_decreasing_min"]),
            "worst_category_delta_aupro_0_05": min(category_deltas)
            >= float(gate_config["worst_category_delta_aupro_0_05_min"]),
            "overall_f1_noninferiority": float(macro["overall_f1"]["delta"])
            >= float(gate_config["delta_overall_f1_min"]),
            "unseen_f1_noninferiority": float(macro["unseen_f1"]["delta"])
            >= float(gate_config["delta_unseen_f1_min"]),
        }
    else:
        checks = {
            "mean_delta_aupro_0_05": delta_aupro >= float(gate_config["mean_delta_aupro_0_05_min"]),
            "mean_delta_pro_at_fpr_0_01": float(macro["pro_at_fpr_0_01"]["delta"])
            >= float(gate_config["mean_delta_pro_at_fpr_0_01_min"]),
            "mean_delta_fixed_small_aupro": float(macro["fixed_small_aupro_0_05"]["delta"])
            >= float(gate_config["mean_delta_fixed_small_aupro_min"]),
            "overall_f1_noninferiority": float(macro["overall_f1"]["delta"])
            >= float(gate_config["delta_overall_f1_min"]),
            "unseen_f1_noninferiority": float(macro["unseen_f1"]["delta"])
            >= float(gate_config["delta_unseen_f1_min"]),
            "image_auroc_noninferiority": float(macro["image_auroc"]["delta"])
            >= float(gate_config["delta_image_auroc_min"]),
            "mean_roi_area": float(macro["mean_roi_area_fraction"]["full_rcbr"])
            <= float(gate_config["mean_roi_area_fraction_max"]),
            "p95_roi_area": max(
                float(per_category[c]["strategies"]["full_rcbr"]["p95_roi_area_fraction"])
                for c in categories
            )
            <= float(gate_config["p95_roi_area_fraction_max"]),
        }
    gate = {"name": args.gate, "passed": all(checks.values()), "checks": checks}
    output = {
        "schema_version": 1,
        "status": "completed",
        "created_at": utc_now(),
        "batch_root": str(args.batch_root),
        "patchcore_reference": str(args.patchcore_reference),
        "patchcore_reference_sha256": file_sha256(args.patchcore_reference),
        "categories": categories,
        "seeds": seeds,
        "run_count": len(runs),
        "per_category": per_category,
        "macro": macro,
        "gate": gate,
    }
    write_json(args.output, output)
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if gate["passed"] or not args.enforce else 2


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--batch-root", required=True, type=Path)
    value.add_argument("--patchcore-reference", required=True, type=Path)
    value.add_argument("--config", required=True, type=Path)
    value.add_argument("--gate", required=True, choices=("smoke", "full_development"))
    value.add_argument("--expected-runs", required=True, type=int)
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--enforce", action="store_true")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
