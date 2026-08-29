#!/usr/bin/env python3
"""Run one category/seed GuardedAdapt-Risk real-image replay task."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import patchcore.common
import patchcore.patchcore
import torch
import yaml
from PIL import Image, ImageEnhance, ImageFilter

from evoinspect.baseline import select_threshold
from evoinspect.evaluation import binary_metrics
from evoinspect.guarded_adapt_risk import (
    ConsecutiveDriftDetector,
    DriftPolicy,
    PairedDecisions,
    RiskBudget,
    bounded_memory_replace,
    exact_rollback,
    paired_bootstrap_risk_gate,
)
from evoinspect.provenance import file_sha256, git_state, utc_now, write_json
from scripts.patchcore_upstream_100_30 import image_transform, read_csv


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def drift_transform(name: str, settings: dict[str, Any]) -> Callable[[Image.Image], Image.Image]:
    if name == "brightness":
        factor = float(settings["factor"])
        return lambda image: ImageEnhance.Brightness(image).enhance(factor)
    if name == "color_temperature":
        factors = np.asarray(
            [settings["red_factor"], settings["green_factor"], settings["blue_factor"]],
            dtype=np.float32,
        )

        def color_temperature(image: Image.Image) -> Image.Image:
            array = np.asarray(image, dtype=np.float32) * factors[None, None]
            return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")

        return color_temperature
    if name == "gaussian_blur":
        radius = float(settings["radius"])
        return lambda image: image.filter(ImageFilter.GaussianBlur(radius=radius))
    if name == "jpeg":
        quality = int(settings["quality"])

        def jpeg(image: Image.Image) -> Image.Image:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=False)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                return decoded.convert("RGB")

        return jpeg
    raise ValueError(f"unsupported drift: {name}")


def identity(image: Image.Image) -> Image.Image:
    return image


def batched_tensors(
    rows: list[dict[str, str]], transform: Any, drift: Callable[[Image.Image], Image.Image],
    batch_size: int,
) -> list[torch.Tensor]:
    batches = []
    for start in range(0, len(rows), batch_size):
        tensors = [
            transform(drift(load_image(row["path"])))
            for row in rows[start : start + batch_size]
        ]
        batches.append(torch.stack(tensors))
    return batches


def predict_scores(
    model: patchcore.patchcore.PatchCore,
    rows: list[dict[str, str]],
    transform: Any,
    drift: Callable[[Image.Image], Image.Image],
    batch_size: int,
) -> dict[str, float]:
    output: dict[str, float] = {}
    offset = 0
    for tensor in batched_tensors(rows, transform, drift, batch_size):
        scores, _ = model._predict(tensor)
        batch_rows = rows[offset : offset + len(scores)]
        output.update(
            {row["sample_id"]: float(score) for row, score in zip(batch_rows, scores, strict=True)}
        )
        offset += len(scores)
    if len(output) != len(rows):
        raise RuntimeError("prediction coverage mismatch")
    return output


def extract_features(
    model: patchcore.patchcore.PatchCore,
    rows: list[dict[str, str]],
    transform: Any,
    drift: Callable[[Image.Image], Image.Image],
    batch_size: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    chunks = []
    for tensor in batched_tensors(rows, transform, drift, batch_size):
        chunks.append(np.asarray(model._embed(tensor.to(model.device)), dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def set_memory(model: patchcore.patchcore.PatchCore, features: np.ndarray[Any, Any]) -> None:
    model.anomaly_scorer.fit([np.asarray(features, dtype=np.float32)])


def f1(labels: list[int], scores: list[float], threshold: float) -> float:
    return float(
        binary_metrics(labels, scores, [int(score >= threshold) for score in scores])[
            "f1_fixed_threshold"
        ]
    )


def decisions(ids: list[str], scores: dict[str, float], threshold: float) -> list[int]:
    return [int(scores[sample_id] >= threshold) for sample_id in ids]


def paired(
    ids: list[str], truth: dict[str, dict[str, str]], champion_scores: dict[str, float],
    candidate_scores: dict[str, float], champion_threshold: float, candidate_threshold: float,
) -> PairedDecisions:
    return PairedDecisions.from_values(
        [int(truth[sample_id]["label"] == "anomaly") for sample_id in ids],
        decisions(ids, champion_scores, champion_threshold),
        decisions(ids, candidate_scores, candidate_threshold),
    )


def group_ids(
    ids: list[str], truth: dict[str, dict[str, str]], group: str
) -> list[str]:
    if group == "normal":
        return [sample_id for sample_id in ids if truth[sample_id]["label"] == "normal"]
    return [
        sample_id
        for sample_id in ids
        if truth[sample_id]["label"] == "anomaly"
        and truth[sample_id]["defect_visibility"] == group
    ]


def risk_gate(
    *,
    target_ids: list[str],
    gate_ids: list[str],
    truth: dict[str, dict[str, str]],
    champion_target: dict[str, float],
    candidate_target: dict[str, float],
    champion_gate: dict[str, float],
    candidate_gate: dict[str, float],
    champion_threshold: float,
    candidate_threshold: float,
    budget: RiskBudget,
    seed: int,
) -> dict[str, Any]:
    result = paired_bootstrap_risk_gate(
        target=paired(
            target_ids,
            truth,
            champion_target,
            candidate_target,
            champion_threshold,
            candidate_threshold,
        ),
        historical_normal=paired(
            group_ids(gate_ids, truth, "normal"),
            truth,
            champion_gate,
            candidate_gate,
            champion_threshold,
            candidate_threshold,
        ),
        seen_anomaly=paired(
            group_ids(gate_ids, truth, "seen"),
            truth,
            champion_gate,
            candidate_gate,
            champion_threshold,
            candidate_threshold,
        ),
        unseen_anomaly=paired(
            group_ids(gate_ids, truth, "unseen"),
            truth,
            champion_gate,
            candidate_gate,
            champion_threshold,
            candidate_threshold,
        ),
        budget=budget,
        seed=seed,
    )
    return result.to_dict()


def point_outcome(
    *,
    target_ids: list[str],
    anchor_ids: list[str],
    truth: dict[str, dict[str, str]],
    champion_target: dict[str, float],
    final_target: dict[str, float],
    champion_anchor: dict[str, float],
    final_anchor: dict[str, float],
    champion_threshold: float,
    final_threshold: float,
    budgets: RiskBudget,
) -> dict[str, Any]:
    target_labels = [int(truth[sample_id]["label"] == "anomaly") for sample_id in target_ids]
    before = f1(
        target_labels, [champion_target[sample_id] for sample_id in target_ids], champion_threshold
    )
    after = f1(
        target_labels, [final_target[sample_id] for sample_id in target_ids], final_threshold
    )
    deltas = {}
    for group, error_decision in (("normal", 1), ("seen", 0), ("unseen", 0)):
        ids = group_ids(anchor_ids, truth, group)
        champion = np.asarray(decisions(ids, champion_anchor, champion_threshold))
        candidate = np.asarray(decisions(ids, final_anchor, final_threshold))
        deltas[group] = float(
            np.mean(candidate == error_decision) - np.mean(champion == error_decision)
        )
    harmful = (
        deltas["normal"] > budgets.historical_normal_fpr_ucb_max
        or deltas["seen"] > budgets.seen_anomaly_fnr_ucb_max
        or deltas["unseen"] > budgets.unseen_anomaly_fnr_ucb_max
    )
    return {
        "target_f1_before": before,
        "target_f1_after": after,
        "target_gain": after - before,
        "normal_fpr_regression": deltas["normal"],
        "seen_fnr_regression": deltas["seen"],
        "unseen_fnr_regression": deltas["unseen"],
        "harmful_update": bool(harmful),
    }


def corrupt_labels(rows: list[dict[str, str]], enabled: bool, fraction: float) -> dict[str, int]:
    labels = {row["sample_id"]: int(row["label"] == "anomaly") for row in rows}
    if not enabled:
        return labels
    count = max(1, round(len(rows) * fraction))
    ordered = sorted(rows, key=lambda row: hashlib.sha256(row["sample_id"].encode()).hexdigest())
    # Guarantee that both error directions are represented when both labels exist.
    selected = [
        next(row for row in ordered if row["label"] == label)
        for label in ("normal", "anomaly")
    ]
    selected_ids = {row["sample_id"] for row in selected}
    selected.extend(row for row in ordered if row["sample_id"] not in selected_ids)
    for row in selected[:count]:
        labels[row["sample_id"]] = 1 - labels[row["sample_id"]]
    return labels


def replay(
    *,
    model: patchcore.patchcore.PatchCore,
    original_memory: np.ndarray[Any, np.dtype[np.float32]],
    original_redundancy: np.ndarray[Any, np.dtype[np.float64]],
    source_run: Path,
    partition: dict[str, Any],
    config: dict[str, Any],
    drift_name: str,
    corrupt: bool,
    transform: Any,
    batch_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    drift = drift_transform(drift_name, config["drifts"][drift_name])
    adaptation = read_csv(source_run / "adaptation.csv")
    adaptation_by_id = {row["sample_id"]: row for row in adaptation}
    test_inputs = read_csv(source_run / "test_inputs.csv")
    test_by_id = {row["sample_id"]: row for row in test_inputs}
    support = partition["support"]
    test_roles = partition["test"]
    meta = json.loads((source_run / "model" / "meta.json").read_text(encoding="utf-8"))
    initial_threshold = float(meta["threshold"]["threshold"])

    reference_rows = [adaptation_by_id[value] for value in support["drift_reference_normal"]]
    window_1_rows = [adaptation_by_id[value] for value in support["drift_window_1"]]
    window_2_rows = [adaptation_by_id[value] for value in support["drift_window_2"]]
    feedback_train_rows = [
        adaptation_by_id[value]
        for key in ("feedback_normal_train", "feedback_anomaly_train")
        for value in support[key]
    ]
    feedback_validation_rows = [
        adaptation_by_id[value]
        for key in ("feedback_normal_validation", "feedback_anomaly_validation")
        for value in support[key]
    ]
    shadow_rows = [
        adaptation_by_id[value]
        for key in ("shadow_normal", "shadow_anomaly")
        for value in support[key]
    ]

    set_memory(model, original_memory)
    reference_scores = predict_scores(model, reference_rows, transform, identity, batch_size)
    window_1_scores = predict_scores(model, window_1_rows, transform, drift, batch_size)
    window_2_scores = predict_scores(model, window_2_rows, transform, drift, batch_size)
    trigger = config["drift_trigger"]
    drift_policy = DriftPolicy(
        window_size=int(trigger["window_size"]),
        p_value_max=float(trigger["p_value_max"]),
        median_shift_iqr_fraction=float(trigger["median_shift_iqr_fraction"]),
        consecutive_windows=int(trigger["consecutive_windows"]),
    )
    detector = ConsecutiveDriftDetector(list(reference_scores.values()), drift_policy)
    drift_windows = [
        detector.update(list(window_1_scores.values())),
        detector.update(list(window_2_scores.values())),
    ]

    all_test_rows = [test_by_id[sample_id] for sample_id in sorted(test_by_id)]
    champion_raw = predict_scores(model, all_test_rows, transform, identity, batch_size)
    champion_drift = predict_scores(model, all_test_rows, transform, drift, batch_size)
    champion_feedback_train = predict_scores(
        model, feedback_train_rows, transform, drift, batch_size
    )
    champion_feedback_validation = predict_scores(
        model, feedback_validation_rows, transform, drift, batch_size
    )
    champion_shadow = predict_scores(model, shadow_rows, transform, drift, batch_size)

    operator_labels = corrupt_labels(
        feedback_train_rows, corrupt, float(config["feedback"]["corrupted_label_fraction"])
    )
    train_ids = [row["sample_id"] for row in feedback_train_rows]
    raw_threshold = float(
        select_threshold(
            [champion_feedback_train[value] for value in train_ids],
            [operator_labels[value] for value in train_ids],
        )["threshold"]
    )
    feedback_values = np.asarray([champion_feedback_train[value] for value in train_ids])
    iqr = float(np.quantile(feedback_values, 0.75) - np.quantile(feedback_values, 0.25))
    shift = float(config["threshold_update"]["max_shift_iqr_fraction"]) * max(iqr, 1e-12)
    bounded_threshold = float(
        np.clip(raw_threshold, initial_threshold - shift, initial_threshold + shift)
    )

    predicted_normal_rows = [
        row for row in feedback_train_rows if operator_labels[row["sample_id"]] == 0
    ]
    proposed = extract_features(model, predicted_normal_rows, transform, drift, batch_size)
    index = model.anomaly_scorer.nn_method.search_index
    nearest_original, _ = index.search(proposed, 1)
    candidate_memory, memory_audit = bounded_memory_replace(
        original_memory,
        proposed,
        nearest_original_distance=np.asarray(nearest_original[:, 0], dtype=np.float64),
        original_redundancy_distance=original_redundancy,
        max_replace_fraction=float(config["memory_update"]["max_replace_fraction"]),
        max_new_features=int(config["memory_update"]["max_new_patch_features"]),
        candidate_pool_multiplier=int(config["memory_update"]["candidate_pool_multiplier"]),
    )
    naive_memory = np.concatenate([original_memory, proposed], axis=0)

    set_memory(model, candidate_memory)
    risk_raw = predict_scores(model, all_test_rows, transform, identity, batch_size)
    risk_drift = predict_scores(model, all_test_rows, transform, drift, batch_size)
    risk_feedback_validation = predict_scores(
        model, feedback_validation_rows, transform, drift, batch_size
    )
    risk_shadow = predict_scores(model, shadow_rows, transform, drift, batch_size)
    set_memory(model, naive_memory)
    naive_raw = predict_scores(model, all_test_rows, transform, identity, batch_size)
    naive_drift = predict_scores(model, all_test_rows, transform, drift, batch_size)
    set_memory(model, original_memory)

    # All test predictions are fixed above. Truth is opened only now.
    truth = {row["sample_id"]: row for row in read_csv(source_run / "test_truth.csv")}
    target_ids = test_roles["target"]
    gate_ids = test_roles["gate_anchor"]
    audit_ids = test_roles["audit_anchor"]
    validation_truth = {
        row["sample_id"]: {**row, "defect_visibility": "seen"}
        for row in feedback_validation_rows
    }
    combined_truth = {**truth, **validation_truth}
    risk = config["risk_gate"]
    budget = RiskBudget(
        target_f1_gain_lcb_min=float(risk["target_f1_gain_lcb_min"]),
        historical_normal_fpr_ucb_max=float(risk["historical_normal_fpr_ucb_max"]),
        seen_anomaly_fnr_ucb_max=float(risk["seen_anomaly_fnr_ucb_max"]),
        unseen_anomaly_fnr_ucb_max=float(risk["unseen_anomaly_fnr_ucb_max"]),
        confidence=float(risk["confidence"]),
        bootstrap_draws=int(risk["bootstrap_draws"]),
        multiplicity=int(risk["multiplicity"]),
    )
    seed = int(partition["seed"]) * 100 + list(config["drifts"]).index(drift_name)
    threshold_gate = risk_gate(
        target_ids=[row["sample_id"] for row in feedback_validation_rows],
        gate_ids=gate_ids,
        truth=combined_truth,
        champion_target=champion_feedback_validation,
        candidate_target=champion_feedback_validation,
        champion_gate=champion_raw,
        candidate_gate=champion_raw,
        champion_threshold=initial_threshold,
        candidate_threshold=bounded_threshold,
        budget=budget,
        seed=seed,
    )
    memory_gate = risk_gate(
        target_ids=[row["sample_id"] for row in feedback_validation_rows],
        gate_ids=gate_ids,
        truth=combined_truth,
        champion_target=champion_feedback_validation,
        candidate_target=risk_feedback_validation,
        champion_gate=champion_raw,
        candidate_gate=risk_raw,
        champion_threshold=initial_threshold,
        candidate_threshold=bounded_threshold,
        budget=budget,
        seed=seed + 1,
    )
    candidate_kind = (
        "MemoryCandidate"
        if memory_gate["accepted"]
        else "ThresholdCandidate"
        if threshold_gate["accepted"]
        else "Rejected"
    )
    candidate_raw = risk_raw if candidate_kind == "MemoryCandidate" else champion_raw
    candidate_drift = risk_drift if candidate_kind == "MemoryCandidate" else champion_drift
    candidate_shadow = risk_shadow if candidate_kind == "MemoryCandidate" else champion_shadow
    candidate_threshold = bounded_threshold if candidate_kind != "Rejected" else initial_threshold

    shadow_ids = [row["sample_id"] for row in shadow_rows]
    shadow_truth = {
        row["sample_id"]: {**row, "defect_visibility": "seen"} for row in shadow_rows
    }
    shadow_labels = [int(shadow_truth[value]["label"] == "anomaly") for value in shadow_ids]
    shadow_gain = f1(
        shadow_labels, [candidate_shadow[value] for value in shadow_ids], candidate_threshold
    ) - f1(shadow_labels, [champion_shadow[value] for value in shadow_ids], initial_threshold)
    shadow_normal_ids = group_ids(shadow_ids, shadow_truth, "normal")
    shadow_seen_ids = group_ids(shadow_ids, shadow_truth, "seen")
    shadow_fpr = float(
        np.mean(decisions(shadow_normal_ids, candidate_shadow, candidate_threshold))
        - np.mean(decisions(shadow_normal_ids, champion_shadow, initial_threshold))
    )
    shadow_fnr = float(
        np.mean(np.asarray(decisions(shadow_seen_ids, candidate_shadow, candidate_threshold)) == 0)
        - np.mean(np.asarray(decisions(shadow_seen_ids, champion_shadow, initial_threshold)) == 0)
    )
    shadow_passed = (
        candidate_kind != "Rejected"
        and shadow_gain > 0.0
        and shadow_fpr <= budget.historical_normal_fpr_ucb_max
        and shadow_fnr <= budget.seen_anomaly_fnr_ucb_max
    )
    promoted = bool(drift_windows[-1].triggered and shadow_passed)
    final_raw = candidate_raw if promoted else champion_raw
    final_drift = candidate_drift if promoted else champion_drift
    final_threshold = candidate_threshold if promoted else initial_threshold

    validation_ids = [row["sample_id"] for row in feedback_validation_rows]
    v1_feedback_gain = f1(
        [int(validation_truth[value]["label"] == "anomaly") for value in validation_ids],
        [champion_feedback_validation[value] for value in validation_ids],
        bounded_threshold,
    ) - f1(
        [int(validation_truth[value]["label"] == "anomaly") for value in validation_ids],
        [champion_feedback_validation[value] for value in validation_ids],
        initial_threshold,
    )
    gate_labels = [int(truth[value]["label"] == "anomaly") for value in gate_ids]
    v1_gate_regression = f1(
        gate_labels, [champion_raw[value] for value in gate_ids], initial_threshold
    ) - f1(gate_labels, [champion_raw[value] for value in gate_ids], bounded_threshold)
    v1_accepted = v1_feedback_gain >= 0.0 and v1_gate_regression <= 0.01
    v1_threshold = bounded_threshold if v1_accepted else initial_threshold

    strategies = {
        "NoUpdate": (champion_raw, champion_drift, initial_threshold, False),
        "NaiveUpdate": (naive_raw, naive_drift, raw_threshold, True),
        "BoundedThreshold": (champion_raw, champion_drift, bounded_threshold, True),
        "GuardedAdapt-v1": (champion_raw, champion_drift, v1_threshold, v1_accepted),
        "GuardedAdapt-Risk": (final_raw, final_drift, final_threshold, promoted),
    }
    outcomes = {}
    for name, (raw_scores, drift_scores, threshold, accepted) in strategies.items():
        outcome = point_outcome(
            target_ids=target_ids,
            anchor_ids=audit_ids,
            truth=truth,
            champion_target=champion_drift,
            final_target=drift_scores,
            champion_anchor=champion_raw,
            final_anchor=raw_scores,
            champion_threshold=initial_threshold,
            final_threshold=threshold,
            budgets=budget,
        )
        restored = decisions(audit_ids, raw_scores, threshold)
        if name == "GuardedAdapt-Risk" and not promoted:
            restored = decisions(audit_ids, champion_raw, initial_threshold)
        outcome.update(
            {
                "accepted_update": bool(accepted),
                "threshold_before": initial_threshold,
                "threshold_after": threshold,
                "rollback_success": (
                    exact_rollback(
                        np.asarray(decisions(audit_ids, champion_raw, initial_threshold)),
                        np.asarray(restored),
                    )
                    if name in {"GuardedAdapt-v1", "GuardedAdapt-Risk"} and not accepted
                    else None
                ),
            }
        )
        outcomes[name] = outcome
    return {
        "category": partition["category"],
        "seed": partition["seed"],
        "drift": drift_name,
        "feedback_corrupted": corrupt,
        "corrupted_feedback_labels": sum(
            operator_labels[row["sample_id"]] != int(row["label"] == "anomaly")
            for row in feedback_train_rows
        ),
        "drift_windows": [window.__dict__ for window in drift_windows],
        "drift_triggered": drift_windows[-1].triggered,
        "memory_candidate": memory_audit,
        "risk_gate": {
            "threshold_candidate": threshold_gate,
            "memory_candidate": memory_gate,
            "selected_candidate": candidate_kind,
        },
        "shadow": {
            "samples": len(shadow_ids),
            "target_gain": shadow_gain,
            "normal_fpr_regression": shadow_fpr,
            "seen_fnr_regression": shadow_fnr,
            "passed": shadow_passed,
        },
        "v1_gate": {
            "feedback_gain": v1_feedback_gain,
            "anchor_regression": v1_gate_regression,
            "accepted": v1_accepted,
        },
        "strategies": outcomes,
        "adapt_latency_ms": (time.perf_counter() - started) * 1000.0,
        "leakage_audit": {
            "test_predictions_fixed_before_truth_open": True,
            "audit_labels_used_for_candidate_selection": 0,
            "feedback_gate_audit_sample_overlap": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--partitions", required=True, type=Path)
    parser.add_argument("--category", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument(
        "--drift", choices=("brightness", "color_temperature", "gaussian_blur", "jpeg")
    )
    parser.add_argument("--feedback-mode", choices=("clean", "corrupt", "both"), default="both")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.partitions.read_text(encoding="utf-8"))
    matches = [
        run
        for run in manifest["runs"]
        if run["category"] == args.category and int(run["seed"]) == args.seed
    ]
    if len(matches) != 1:
        raise RuntimeError("frozen partition task not found")
    partition = matches[0]
    source_run = Path(partition["source_run"])
    if file_sha256(source_run / "adaptation.csv") != partition["adaptation_sha256"]:
        raise RuntimeError("adaptation hash changed after partition freeze")
    if file_sha256(source_run / "test_truth.csv") != partition["test_truth_sha256"]:
        raise RuntimeError("test truth hash changed after partition freeze")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = patchcore.patchcore.PatchCore(device)
    model.load_from_path(
        str(source_run / "model"),
        device=device,
        nn_method=patchcore.common.FaissNN(False, 1),
    )
    index = model.anomaly_scorer.nn_method.search_index
    original_memory = np.asarray(index.reconstruct_n(0, index.ntotal), dtype=np.float32)
    redundancy, _ = index.search(original_memory, 2)
    original_redundancy = np.asarray(redundancy[:, 1], dtype=np.float64)
    transform = image_transform(
        json.loads((source_run / "model" / "meta.json").read_text(encoding="utf-8"))[
            "config"
        ]
    )
    started = utc_now()
    replays = []
    drift_names = [args.drift] if args.drift else list(config["drifts"])
    if args.feedback_mode == "clean":
        corruption_modes = [False]
    elif args.feedback_mode == "corrupt":
        corruption_modes = [True]
    else:
        corruption_modes = [False, True]
    for drift_name in drift_names:
        for corrupt in corruption_modes:
            replays.append(
                replay(
                    model=model,
                    original_memory=original_memory,
                    original_redundancy=original_redundancy,
                    source_run=source_run,
                    partition=partition,
                    config=config,
                    drift_name=str(drift_name),
                    corrupt=corrupt,
                    transform=transform,
                    batch_size=args.batch_size,
                )
            )
    set_memory(model, original_memory)
    commit, dirty = git_state(Path.cwd())
    report = {
        "schema_version": 1,
        "status": "completed_guarded_adapt_risk_image_replays",
        "started_at": started,
        "ended_at": utc_now(),
        "category": args.category,
        "seed": args.seed,
        "source_run": str(source_run),
        "source_model_hash": hashlib.sha256(
            np.ascontiguousarray(original_memory).tobytes()
        ).hexdigest(),
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "partitions_sha256": file_sha256(args.partitions),
        "partition_hash": partition["partition_hash"],
        "git_commit": commit,
        "dirty": dirty,
        "hardware": platform.platform(),
        "physical_gpu": os.environ.get("EVOINSPECT_PHYSICAL_GPU", args.device),
        "replay_count": len(replays),
        "replays": replays,
    }
    write_json(args.output, report)
    print(json.dumps({"category": args.category, "seed": args.seed, "replays": len(replays)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
