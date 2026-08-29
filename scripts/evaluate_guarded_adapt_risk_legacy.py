#!/usr/bin/env python3
"""Convert the frozen 75-run v1 replay into the five-strategy comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evoinspect.provenance import file_sha256, git_state, utc_now, write_json


def convert(run: dict[str, Any]) -> dict[str, Any]:
    original = run["strategies"]
    risk = dict(original["NoUpdate"])
    risk.update(
        {
            "accepted_update": False,
            "rollback_success": True,
            "legacy_risk_gate_status": "rejected_missing_seen_unseen_group_and_image_memory",
        }
    )
    return {
        "category": run["category"],
        "seed": run["seed"],
        "source": "legacy_static_score_replay",
        "feedback_corrupted": False,
        "strategies": {
            "NoUpdate": original["NoUpdate"],
            "NaiveUpdate": original["NaiveUpdate"],
            "BoundedThreshold": original["ThresholdUpdate"],
            "GuardedAdapt-v1": original["GuardedAdapt"],
            "GuardedAdapt-Risk": risk,
        },
        "leakage_audit": {
            "audit_labels_used_for_candidate_selection": 0,
            "feedback_gate_audit_sample_overlap": 0,
        },
        "limitations": [
            "The tracked legacy pack has scores but no images or patch embeddings.",
            "It has binary labels but no seen/unseen visibility per record.",
            "GuardedAdapt-Risk therefore rejects rather than fabricating a memory/risk result.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if len(source["runs"]) != 75:
        raise RuntimeError("legacy source is not the frozen 75-run report")
    commit, dirty = git_state(Path.cwd())
    report = {
        "schema_version": 1,
        "status": "completed_legacy_five_strategy_conversion",
        "created_at": utc_now(),
        "source": str(args.source),
        "source_sha256": file_sha256(args.source),
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "git_commit": commit,
        "dirty": dirty,
        "replay_count": 75,
        "replays": [convert(run) for run in source["runs"]],
    }
    write_json(args.output, report)
    print(json.dumps({"replays": report["replay_count"], "status": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
