#!/usr/bin/env python3
"""Aggregate the preregistered 219 GuardedAdapt-Risk replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evoinspect.provenance import file_sha256, utc_now, write_json


def mean(items: list[float]) -> float | None:
    return float(np.mean(np.asarray(items, dtype=np.float64))) if items else None


def upper_mean_bound(values: list[float], draws: int = 2000, seed: int = 130) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.asarray(
        [np.mean(array[generator.integers(0, len(array), size=len(array))]) for _ in range(draws)]
    )
    return float(np.quantile(means, 0.95))


def normalized_strategy(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_gain": float(item.get("target_gain", 0.0)),
        "harmful_update": bool(item.get("harmful_update", False)),
        "accepted_update": bool(item.get("accepted_update", False)),
        "rollback_success": item.get("rollback_success"),
        "normal_fpr_regression": item.get("normal_fpr_regression"),
        "seen_fnr_regression": item.get("seen_fnr_regression"),
        "unseen_fnr_regression": item.get("unseen_fnr_regression"),
    }


def strategy_summary(replays: list[dict[str, Any]], name: str) -> dict[str, Any]:
    items = [normalized_strategy(replay["strategies"][name]) for replay in replays]
    accepted = [item for item in items if item["accepted_update"]]
    rollbacks = [
        bool(item["rollback_success"])
        for item in items
        if item["rollback_success"] is not None
    ]
    image_accepted = [
        item
        for replay, item in zip(replays, items, strict=True)
        if replay["source"] == "real_image_drift_replay" and item["accepted_update"]
    ]
    risk_bounds = {}
    for key in (
        "normal_fpr_regression",
        "seen_fnr_regression",
        "unseen_fnr_regression",
    ):
        values = [float(item[key]) for item in image_accepted if item[key] is not None]
        risk_bounds[f"{key}_mean"] = mean(values)
        risk_bounds[f"{key}_bootstrap_95_ucb"] = upper_mean_bound(values)
    return {
        "replays": len(items),
        "target_gain_mean": mean([item["target_gain"] for item in items]),
        "accepted_target_gain_mean": mean([item["target_gain"] for item in accepted]),
        "harmful_update_rate": mean([float(item["harmful_update"]) for item in items]),
        "accepted_update_rate": mean([float(item["accepted_update"]) for item in items]),
        "rollback_success_rate": mean([float(value) for value in rollbacks]),
        "rollback_trials": len(rollbacks),
        **risk_bounds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    legacy = json.loads(args.legacy.read_text(encoding="utf-8"))
    image_reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.image_dir.glob("*.json"))
        if path.name != args.output.name
    ]
    image_replays = []
    for report in image_reports:
        for replay in report["replays"]:
            image_replays.append({**replay, "source": "real_image_drift_replay"})
    legacy_replays = legacy["replays"]
    replays = [*legacy_replays, *image_replays]
    strategies = {
        name: strategy_summary(replays, name) for name in config["strategies"]
    }
    risk = strategies["GuardedAdapt-Risk"]
    v1 = strategies["GuardedAdapt-v1"]
    v1_harmful = float(v1["harmful_update_rate"] or 0.0)
    risk_harmful = float(risk["harmful_update_rate"] or 0.0)
    relative_reduction = (
        (v1_harmful - risk_harmful) / v1_harmful if v1_harmful > 0.0 else None
    )
    harmful_v1_candidates = [
        replay
        for replay in replays
        if normalized_strategy(replay["strategies"]["GuardedAdapt-v1"])["harmful_update"]
    ]
    blocked = [
        replay
        for replay in harmful_v1_candidates
        if not normalized_strategy(replay["strategies"]["GuardedAdapt-Risk"])[
            "accepted_update"
        ]
    ]
    block_rate = len(blocked) / len(harmful_v1_candidates) if harmful_v1_candidates else None
    leakage = sum(
        int(value)
        for replay in replays
        for key, value in replay.get("leakage_audit", {}).items()
        if key != "test_predictions_fixed_before_truth_open"
    )
    gate = config["quality_gate"]
    checks = {
        "minimum_total_replays": len(replays) >= int(gate["minimum_total_replays"]),
        "harmful_update_rate": risk_harmful <= float(gate["harmful_update_rate_max"]),
        "harmful_relative_reduction_vs_v1": relative_reduction is not None
        and relative_reduction >= float(gate["harmful_relative_reduction_vs_v1_min"]),
        "accepted_update_rate": float(risk["accepted_update_rate"] or 0.0)
        >= float(gate["accepted_update_rate_min"]),
        "accepted_target_f1_gain": risk["accepted_target_gain_mean"] is not None
        and float(risk["accepted_target_gain_mean"])
        >= float(gate["accepted_target_f1_gain_min"]),
        "normal_fpr_regression_ucb": risk["normal_fpr_regression_bootstrap_95_ucb"]
        is not None
        and float(risk["normal_fpr_regression_bootstrap_95_ucb"])
        <= float(gate["normal_fpr_regression_ucb_max"]),
        "seen_fnr_regression_ucb": risk["seen_fnr_regression_bootstrap_95_ucb"]
        is not None
        and float(risk["seen_fnr_regression_bootstrap_95_ucb"])
        <= float(gate["seen_fnr_regression_ucb_max"]),
        "unseen_fnr_regression_ucb": risk["unseen_fnr_regression_bootstrap_95_ucb"]
        is not None
        and float(risk["unseen_fnr_regression_bootstrap_95_ucb"])
        <= float(gate["unseen_fnr_regression_ucb_max"]),
        "harmful_candidate_block_rate": block_rate is not None
        and block_rate >= float(gate["harmful_candidate_block_rate_min"]),
        "rollback_success_rate": risk["rollback_success_rate"] is not None
        and float(risk["rollback_success_rate"]) == float(gate["rollback_success_rate"]),
        "leakage_events": leakage <= int(gate["leakage_events_max"]),
    }
    report = {
        "schema_version": 1,
        "status": "guarded_adapt_risk_preregistered_quality_gate",
        "created_at": utc_now(),
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "legacy_replays": len(legacy_replays),
        "image_drift_replays": len(
            [item for item in image_replays if not item["feedback_corrupted"]]
        ),
        "corrupted_feedback_replays": len(
            [item for item in image_replays if item["feedback_corrupted"]]
        ),
        "total_replays": len(replays),
        "strategies": strategies,
        "safety_effect": {
            "harmful_relative_reduction_vs_v1": relative_reduction,
            "harmful_v1_candidates": len(harmful_v1_candidates),
            "harmful_candidates_blocked": len(blocked),
            "harmful_candidate_block_rate": block_rate,
        },
        "leakage_events": leakage,
        "checks": checks,
        "passed": all(checks.values()),
        "image_reports": [str(path) for path in sorted(args.image_dir.glob("*.json"))],
        "claim_limit": (
            "Offline MVTec replay only; not a production user study, production accuracy "
            "claim, or novelty claim for incremental PatchCore memory."
        ),
    }
    write_json(args.output, report)
    print(json.dumps({"passed": report["passed"], "total_replays": len(replays)}))
    return 0 if report["passed"] or not args.enforce else 2


if __name__ == "__main__":
    raise SystemExit(main())
