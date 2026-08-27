#!/usr/bin/env python3
"""Real-score MVTec feedback replay for four frozen adaptation strategies."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evoinspect.baseline import select_threshold
from evoinspect.evaluation import binary_metrics
from evoinspect.provenance import file_sha256, git_state, utc_now, write_json


def read_truth(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["sample_id"]: int(row["label"] == "anomaly") for row in csv.DictReader(stream)}


def f1(scores: list[float], labels: list[int], threshold: float) -> float:
    return float(
        binary_metrics(labels, scores, [int(score >= threshold) for score in scores])[
            "f1_fixed_threshold"
        ]
    )


def partitions(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    names = ("feedback", "target", "gate_anchor", "audit_anchor")
    result = {name: [] for name in names}
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[int(record["label"])].append(record)
    for label_records in by_label.values():
        ordered = sorted(
            label_records,
            key=lambda item: hashlib.sha256(item["sample_id"].encode()).hexdigest(),
        )
        slots = (
            ("feedback", "feedback", "target", "gate_anchor", "audit_anchor")
            if len(ordered) >= 5
            else names
        )
        for index, record in enumerate(ordered):
            # 2:1:1:1 allocation approximates the frozen 40/20/20/20 split.
            result[slots[index % 5]].append(record)
    if any({int(item["label"]) for item in values} != {0, 1} for values in result.values()):
        raise RuntimeError("replay partition lacks both labels")
    return result


def scores_labels(records: list[dict[str, Any]]) -> tuple[list[float], list[int]]:
    return [float(item["score"]) for item in records], [int(item["label"]) for item in records]


def evaluate_run(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    truth = read_truth(run_dir / "test_truth.csv")
    records = []
    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            records.append(
                {
                    "sample_id": item["sample_id"],
                    "score": float(item["upstream_patchcore_score"]),
                    "label": truth[item["sample_id"]],
                }
            )
    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "model" / "meta.json").read_text(encoding="utf-8"))
    return evaluate_records(
        records,
        split,
        float(meta["threshold"]["threshold"]),
        {
            "predictions_sha256": file_sha256(run_dir / "predictions.jsonl"),
            "truth_sha256": file_sha256(run_dir / "test_truth.csv"),
        },
        config,
    )


def evaluate_records(
    records: list[dict[str, Any]],
    split: dict[str, Any],
    initial: float,
    evidence: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    grouped = partitions(records)
    feedback_scores, feedback_labels = scores_labels(grouped["feedback"])
    target_scores, target_labels = scores_labels(grouped["target"])
    gate_scores, gate_labels = scores_labels(grouped["gate_anchor"])
    audit_scores, audit_labels = scores_labels(grouped["audit_anchor"])
    base = {
        "target_f1": f1(target_scores, target_labels, initial),
        "gate_anchor_f1": f1(gate_scores, gate_labels, initial),
        "audit_anchor_f1": f1(audit_scores, audit_labels, initial),
    }
    started = time.perf_counter_ns()
    raw_candidate = float(select_threshold(feedback_scores, feedback_labels)["threshold"])
    naive_latency = (time.perf_counter_ns() - started) / 1e6
    iqr = float(np.quantile(feedback_scores, 0.75) - np.quantile(feedback_scores, 0.25))
    limit = float(config["threshold_update"]["max_shift_iqr_fraction"]) * max(iqr, 1e-12)
    bounded = float(np.clip(raw_candidate, initial - limit, initial + limit))

    thresholds = {"NoUpdate": initial, "NaiveUpdate": raw_candidate, "ThresholdUpdate": bounded}
    accepted = {"NoUpdate": False, "NaiveUpdate": True, "ThresholdUpdate": bounded != initial}
    latencies = {"NoUpdate": 0.0, "NaiveUpdate": naive_latency, "ThresholdUpdate": naive_latency}
    started = time.perf_counter_ns()
    candidate_feedback_gain = f1(feedback_scores, feedback_labels, bounded) - f1(
        feedback_scores, feedback_labels, initial
    )
    candidate_gate_regression = base["gate_anchor_f1"] - f1(gate_scores, gate_labels, bounded)
    guarded_accepted = candidate_feedback_gain >= float(
        config["guard"]["min_feedback_gain"]
    ) and candidate_gate_regression <= float(config["guard"]["max_gate_anchor_regression"])
    thresholds["GuardedAdapt"] = bounded if guarded_accepted else initial
    accepted["GuardedAdapt"] = guarded_accepted and bounded != initial
    latencies["GuardedAdapt"] = naive_latency + (time.perf_counter_ns() - started) / 1e6

    outputs: dict[str, Any] = {}
    for strategy, threshold in thresholds.items():
        target_f1 = f1(target_scores, target_labels, threshold)
        audit_f1 = f1(audit_scores, audit_labels, threshold)
        regression = base["audit_anchor_f1"] - audit_f1
        harmful = regression > float(config["harmful_update"]["audit_anchor_regression_gt"])
        rollback_success = None
        if strategy == "GuardedAdapt" and not accepted[strategy]:
            rollback_success = all(
                (score >= threshold) == (score >= initial) for score in audit_scores
            )
        outputs[strategy] = {
            "threshold_before": initial,
            "threshold_after": threshold,
            "target_gain": target_f1 - base["target_f1"],
            "anchor_regression": regression,
            "harmful_update": harmful,
            "accepted_update": accepted[strategy],
            "rollback_success": rollback_success,
            "adapt_latency_ms": latencies[strategy],
        }
    return {
        "category": split["category"],
        "seed": split["seed"],
        "split_hash": split["split_hash"],
        "counts": {name: len(values) for name, values in grouped.items()},
        "base": base,
        "candidate_feedback_gain": candidate_feedback_gain,
        "candidate_gate_anchor_regression": candidate_gate_regression,
        "strategies": outputs,
        "evidence": evidence,
    }


def read_pack(path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    runs = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            runs.append(
                evaluate_records(
                    item["records"],
                    item["split"],
                    float(item["initial_threshold"]),
                    item["evidence"],
                    config,
                )
            )
    return runs


def aggregate(runs: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    items = [run["strategies"][strategy] for run in runs]
    rollbacks = [item["rollback_success"] for item in items if item["rollback_success"] is not None]
    return {
        "runs": len(items),
        "target_gain_mean": float(np.mean([item["target_gain"] for item in items])),
        "anchor_regression_mean": float(np.mean([item["anchor_regression"] for item in items])),
        "harmful_update_rate": float(np.mean([item["harmful_update"] for item in items])),
        "accepted_update_rate": float(np.mean([item["accepted_update"] for item in items])),
        "rollback_success_rate": float(np.mean(rollbacks)) if rollbacks else None,
        "rollback_trials": len(rollbacks),
        "adapt_latency_ms": {
            "p50": float(np.quantile([item["adapt_latency_ms"] for item in items], 0.50)),
            "p95": float(np.quantile([item["adapt_latency_ms"] for item in items], 0.95)),
            "p99": float(np.quantile([item["adapt_latency_ms"] for item in items], 0.99)),
        },
    }


def safety_effect(runs: list[dict[str, Any]]) -> dict[str, Any]:
    harmful_candidates = [
        run
        for run in runs
        if run["strategies"]["ThresholdUpdate"]["harmful_update"]
    ]
    blocked = [
        run
        for run in harmful_candidates
        if not run["strategies"]["GuardedAdapt"]["accepted_update"]
    ]
    return {
        "harmful_bounded_candidates": len(harmful_candidates),
        "harmful_candidates_blocked": len(blocked),
        "harmful_candidate_block_rate": (
            len(blocked) / len(harmful_candidates) if harmful_candidates else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = Path(config["source_batch"])
    source_pack = Path(config["source_pack"])
    if source_pack.is_file():
        runs = read_pack(source_pack, config)
        source_mode = "tracked_score_pack"
    else:
        run_dirs = sorted(
            path for path in (source / "runs").iterdir() if (path / "metrics.json").is_file()
        )
        runs = [evaluate_run(path, config) for path in run_dirs]
        source_mode = "original_run_directories"
    if len(runs) != 75:
        raise RuntimeError(f"expected 75 frozen runs, found {len(runs)}")
    commit, dirty = git_state(Path.cwd())
    report = {
        "schema_version": 1,
        "status": "completed_real_mvtec_feedback_replay",
        "created_at": utc_now(),
        "scope": config["scope"],
        "source_batch": str(source),
        "source_pack": str(source_pack),
        "source_pack_sha256": file_sha256(source_pack) if source_pack.is_file() else None,
        "source_mode": source_mode,
        "source_runs": len(runs),
        "categories": len({run["category"] for run in runs}),
        "seeds": sorted({run["seed"] for run in runs}),
        "config_path": str(args.config),
        "config_sha256": file_sha256(args.config),
        "git_commit": commit,
        "dirty": dirty,
        "hardware": platform.platform(),
        "summary": {strategy: aggregate(runs, strategy) for strategy in config["strategies"]},
        "safety_effect": safety_effect(runs),
        "runs": runs,
        "claim_limits": config["claims"],
        "warning": (
            "This is an offline replay over frozen real MVTec scores and simulated revealed "
            "operator labels; it is not a production user study or a claim of accuracy gain."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    write_json(args.output, report)
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
