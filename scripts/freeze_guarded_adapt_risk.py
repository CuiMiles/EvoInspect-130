#!/usr/bin/env python3
"""Freeze GuardedAdapt-Risk sample partitions before running the benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from evoinspect.provenance import canonical_hash, file_sha256, git_state, utc_now, write_json
from scripts.efficientad_rcbr_100_30 import read_csv


def hash_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def one_run(root: Path, category: str, seed: int) -> Path:
    matches = sorted(root.glob(f"upstream-pc-100-30-{category}-s{seed}-*"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one source run for {category}/s{seed}, found {len(matches)}")
    return matches[0]


def take_ordered(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: hash_key(row["sample_id"]))


def support_partitions(
    rows: list[dict[str, str]], config: dict[str, Any]
) -> dict[str, list[str]]:
    normals = take_ordered([row for row in rows if row["role"] == "support_normal"])
    anomalies = take_ordered([row for row in rows if row["role"] == "support_anomaly"])
    trigger = config["drift_trigger"]
    feedback = config["feedback"]
    reference_count = int(trigger["reference_normal_samples"])
    window = int(trigger["window_size"])
    required_normals = reference_count + 2 * window
    if len(normals) < required_normals:
        raise RuntimeError("support normal set cannot supply isolated drift windows")
    anomaly_total = sum(
        int(feedback[key]) for key in ("anomaly_train", "anomaly_validation", "shadow_anomaly")
    )
    if len(anomalies) < anomaly_total:
        raise RuntimeError("support anomaly set cannot supply isolated feedback/shadow groups")
    first_window = normals[reference_count : reference_count + window]
    second_window = normals[reference_count + window : required_normals]
    normal_train = int(feedback["normal_train"])
    normal_validation = int(feedback["normal_validation"])
    shadow_normal = int(feedback["shadow_normal"])
    if normal_train + normal_validation != len(first_window):
        raise RuntimeError("feedback normal train/validation counts must consume first window")
    if shadow_normal > len(second_window):
        raise RuntimeError("shadow normal count exceeds second drift window")
    anomaly_train = int(feedback["anomaly_train"])
    anomaly_validation = int(feedback["anomaly_validation"])
    return {
        "drift_reference_normal": [row["sample_id"] for row in normals[:reference_count]],
        "drift_window_1": [row["sample_id"] for row in first_window],
        "drift_window_2": [row["sample_id"] for row in second_window],
        "feedback_normal_train": [row["sample_id"] for row in first_window[:normal_train]],
        "feedback_normal_validation": [
            row["sample_id"] for row in first_window[normal_train:]
        ],
        "feedback_anomaly_train": [row["sample_id"] for row in anomalies[:anomaly_train]],
        "feedback_anomaly_validation": [
            row["sample_id"]
            for row in anomalies[anomaly_train : anomaly_train + anomaly_validation]
        ],
        "shadow_normal": [row["sample_id"] for row in second_window[:shadow_normal]],
        "shadow_anomaly": [
            row["sample_id"] for row in anomalies[-int(feedback["shadow_anomaly"]) :]
        ],
    }


def test_partitions(rows: list[dict[str, str]], roles: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group = "normal" if row["label"] == "normal" else row["defect_visibility"]
        if group not in {"normal", "seen", "unseen"}:
            raise RuntimeError(f"unexpected truth group: {group!r}")
        grouped[group].append(row)
    result = {role: [] for role in roles}
    for group in ("normal", "seen", "unseen"):
        ordered = take_ordered(grouped[group])
        if len(ordered) < len(roles):
            raise RuntimeError(f"{group} has fewer samples than frozen roles")
        for index, row in enumerate(ordered):
            result[roles[index % len(roles)]].append(row["sample_id"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["source_batch"]) / "runs"
    roles = [str(value) for value in config["partition"]["roles"]]
    runs = []
    for category in config["categories"]:
        for seed in config["seeds"]:
            run_dir = one_run(root, str(category), int(seed))
            adaptation = run_dir / "adaptation.csv"
            truth = run_dir / "test_truth.csv"
            support = support_partitions(read_csv(adaptation), config)
            final = test_partitions(read_csv(truth), roles)
            all_ids = [sample_id for values in final.values() for sample_id in values]
            if len(all_ids) != len(set(all_ids)):
                raise RuntimeError("test role overlap detected")
            runs.append(
                {
                    "category": str(category),
                    "seed": int(seed),
                    "source_run": str(run_dir),
                    "adaptation_sha256": file_sha256(adaptation),
                    "test_inputs_sha256": file_sha256(run_dir / "test_inputs.csv"),
                    "test_truth_sha256": file_sha256(truth),
                    "support": support,
                    "test": final,
                    "partition_hash": canonical_hash({"support": support, "test": final}),
                }
            )
    commit, dirty = git_state(Path.cwd())
    report = {
        "schema_version": 1,
        "status": "preregistered_before_guarded_adapt_risk_metric_evaluation",
        "created_at": utc_now(),
        "config_path": str(args.config),
        "config_sha256": file_sha256(args.config),
        "source_batch": str(root.parent),
        "git_commit": commit,
        "dirty": dirty,
        "runs": runs,
        "run_count": len(runs),
        "partition_hash": canonical_hash([run["partition_hash"] for run in runs]),
        "leakage_policy": {
            "candidate_feedback_vs_gate_vs_audit_sample_overlap": 0,
            "audit_labels_available_to_candidate_gate": False,
            "test_predictions_fixed_before_role_labels_are_evaluated": True,
        },
    }
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("partition_hash") != report["partition_hash"]:
            raise RuntimeError("refusing to overwrite a different frozen partition manifest")
        print(json.dumps({"status": "unchanged", "partition_hash": report["partition_hash"]}))
        return 0
    write_json(args.output, report)
    print(json.dumps({"runs": len(runs), "partition_hash": report["partition_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
