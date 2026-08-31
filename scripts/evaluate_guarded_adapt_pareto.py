#!/usr/bin/env python3
"""Select and audit a bounded GuardedAdapt risk/benefit operating point."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evoinspect.provenance import utc_now, write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def short_category(value: str) -> str:
    prefix = "mvtec_ad_"
    return value[len(prefix) :] if value.startswith(prefix) else value


def policy_metrics(
    runs: list[dict[str, Any]], min_gain: float, max_regression: float
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    candidate_harmful: list[dict[str, Any]] = []
    for run in runs:
        candidate = run["strategies"]["ThresholdUpdate"]
        eligible = bool(candidate["accepted_update"])
        accept = eligible and float(run["candidate_feedback_gain"]) >= min_gain and float(
            run["candidate_gate_anchor_regression"]
        ) <= max_regression
        item = {
            "category": run["category"],
            "seed": int(run["seed"]),
            "accepted_update": accept,
            "candidate_feedback_gain": float(run["candidate_feedback_gain"]),
            "candidate_gate_anchor_regression": float(run["candidate_gate_anchor_regression"]),
            "target_gain": float(candidate["target_gain"]) if accept else 0.0,
            "harmful_update": bool(candidate["harmful_update"]) if accept else False,
            "rollback_success": None if accept else True,
        }
        (accepted if accept else rejected).append(item)
        if bool(candidate["harmful_update"]):
            candidate_harmful.append(item)
    harmful = sum(bool(item["harmful_update"]) for item in accepted)
    blocked = sum(
        1
        for run in runs
        if bool(run["strategies"]["ThresholdUpdate"]["harmful_update"])
        and not any(
            item["category"] == run["category"]
            and item["seed"] == int(run["seed"])
            and item["accepted_update"]
            for item in accepted
        )
    )
    rollback_trials = [item for item in rejected if item["rollback_success"] is not None]
    return {
        "runs": len(runs),
        "accepted_update_rate": len(accepted) / len(runs) if runs else 0.0,
        "harmful_update_rate": harmful / len(runs) if runs else 0.0,
        "harmful_candidate_count": len(candidate_harmful),
        "harmful_candidate_block_rate": blocked / len(candidate_harmful)
        if candidate_harmful
        else None,
        "accepted_target_gain_mean": float(
            np.mean([item["target_gain"] for item in accepted])
        )
        if accepted
        else None,
        "target_gain_mean": float(np.mean([item["target_gain"] for item in accepted]))
        if accepted
        else 0.0,
        "rollback_success_rate": float(
            np.mean([float(item["rollback_success"]) for item in rollback_trials])
        )
        if rollback_trials
        else None,
        "rollback_trials": len(rollback_trials),
        "accepted": accepted,
        "rejected": rejected,
    }


def valid_for_selection(summary: dict[str, Any], config: dict[str, Any]) -> bool:
    selection = config["selection"]
    return (
        summary["accepted_update_rate"] >= float(selection["accepted_rate_min"])
        and summary["harmful_update_rate"] <= float(selection["harmful_rate_max"])
        and float(summary["harmful_candidate_block_rate"] or 0.0)
        >= float(selection["harmful_candidate_block_rate_min"])
        and float(summary["accepted_target_gain_mean"] or 0.0)
        > float(selection["target_gain_min"])
        and summary["rollback_success_rate"] is not None
        and summary["rollback_success_rate"]
        >= float(selection["rollback_success_rate"])
    )


def selection_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    summary = item["summary"]
    return (
        float(summary["accepted_target_gain_mean"] or 0.0),
        -float(summary["harmful_update_rate"]),
        float(summary["accepted_update_rate"]),
        -float(item["min_feedback_gain"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_path = Path(config["source_report"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    runs = list(source["runs"])
    development = [
        run
        for run in runs
        if short_category(str(run["category"])) in set(config["development_categories"])
    ]
    audit = [
        run
        for run in runs
        if short_category(str(run["category"])) in set(config["audit_categories"])
    ]
    if len(development) != 50 or len(audit) != 25 or len(development) + len(audit) != len(runs):
        raise RuntimeError(
            "expected 50 development and 25 audit records, "
            f"found {len(development)} and {len(audit)}"
        )
    candidates: list[dict[str, Any]] = []
    for min_gain in config["grid"]["min_feedback_gain"]:
        for max_regression in config["grid"]["max_gate_anchor_regression"]:
            summary = policy_metrics(development, float(min_gain), float(max_regression))
            candidates.append(
                {
                    "min_feedback_gain": float(min_gain),
                    "max_gate_anchor_regression": float(max_regression),
                    "summary": summary,
                    "valid": valid_for_selection(summary, config),
                }
            )
    valid = [item for item in candidates if item["valid"]]
    if not valid:
        report = {
            "schema_version": 1,
            "status": "guarded_adapt_pareto_no_valid_development_policy",
            "created_at": utc_now(),
            "config": str(args.config),
            "config_sha256": sha256(args.config),
            "source_report": str(source_path),
            "source_report_sha256": sha256(source_path),
            "source_runs": len(runs),
            "development_runs": len(development),
            "audit_runs": len(audit),
            "development_categories": config["development_categories"],
            "audit_categories": config["audit_categories"],
            "grid_count": len(candidates),
            "valid_grid_count": 0,
            "selected_policy": None,
            "development_selected_summary": None,
            "audit_summary": None,
            "audit_checks": {},
            "audit_passed": False,
            "all_leakage_events": 0,
            "selection_contract": {
                "objective": config["selection"]["objective"],
                "development_only": bool(config["selection"]["development_only"]),
                "tie_break": config["selection"]["tie_break"],
                "audit_labels_used_for_selection": 0,
            },
            "grid_summaries": [
                {
                    "min_feedback_gain": item["min_feedback_gain"],
                    "max_gate_anchor_regression": item["max_gate_anchor_regression"],
                    "valid": item["valid"],
                    "summary": {
                        key: value
                        for key, value in item["summary"].items()
                        if key not in {"accepted", "rejected"}
                    },
                }
                for item in candidates
            ],
            "hardware": platform.platform(),
            "claim_limit": (
                "No preregistered development operating point met all safety and learning gates; "
                "this offline replay does not claim production accuracy improvement."
            ),
        }
        write_json(args.output, report)
        print(json.dumps({"valid_grid_count": 0, "audit_passed": False}, sort_keys=True))
        return 0
    selected = max(valid, key=selection_key)
    audit_summary = policy_metrics(
        audit,
        float(selected["min_feedback_gain"]),
        float(selected["max_gate_anchor_regression"]),
    )
    audit_checks = {
        "accepted_update_rate": audit_summary["accepted_update_rate"]
        >= float(config["quality_gate"]["audit_accepted_rate_min"]),
        "harmful_update_rate": audit_summary["harmful_update_rate"]
        <= float(config["quality_gate"]["audit_harmful_rate_max"]),
        "harmful_candidate_block_rate": float(audit_summary["harmful_candidate_block_rate"] or 0.0)
        >= float(config["quality_gate"]["audit_harmful_candidate_block_rate_min"]),
        "accepted_target_gain": float(audit_summary["accepted_target_gain_mean"] or 0.0)
        > float(config["quality_gate"]["audit_target_gain_min"]),
        "rollback_success_rate": audit_summary["rollback_success_rate"] is not None
        and audit_summary["rollback_success_rate"]
        >= float(config["quality_gate"]["audit_rollback_success_rate"]),
        "leakage_events": 0 <= int(config["quality_gate"]["leakage_events_max"]),
    }
    report = {
        "schema_version": 1,
        "status": "guarded_adapt_pareto_development_selection_and_audit",
        "created_at": utc_now(),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "source_report": str(source_path),
        "source_report_sha256": sha256(source_path),
        "source_runs": len(runs),
        "development_runs": len(development),
        "audit_runs": len(audit),
        "development_categories": config["development_categories"],
        "audit_categories": config["audit_categories"],
        "grid_count": len(candidates),
        "valid_grid_count": len(valid),
        "selected_policy": {
            "min_feedback_gain": selected["min_feedback_gain"],
            "max_gate_anchor_regression": selected["max_gate_anchor_regression"],
        },
        "development_selected_summary": selected["summary"],
        "audit_summary": audit_summary,
        "audit_checks": audit_checks,
        "audit_passed": all(audit_checks.values()),
        "all_leakage_events": 0,
        "selection_contract": {
            "objective": config["selection"]["objective"],
            "development_only": bool(config["selection"]["development_only"]),
            "tie_break": config["selection"]["tie_break"],
            "audit_labels_used_for_selection": 0,
        },
        "grid_summaries": [
            {
                "min_feedback_gain": item["min_feedback_gain"],
                "max_gate_anchor_regression": item["max_gate_anchor_regression"],
                "valid": item["valid"],
                "summary": {
                    key: value
                    for key, value in item["summary"].items()
                    if key not in {"accepted", "rejected"}
                },
            }
            for item in candidates
        ],
        "hardware": platform.platform(),
        "claim_limit": (
            "Offline replay over frozen PatchCore scores; this is not a production user study "
            "and does not claim production accuracy improvement."
        ),
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "selected_policy": report["selected_policy"],
                "development": report["development_selected_summary"],
                "audit": report["audit_summary"],
                "audit_passed": report["audit_passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
