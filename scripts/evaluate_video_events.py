#!/usr/bin/env python3
"""Evaluate a video-demo report against the frozen manual event GT."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evoinspect.video_evaluation import aggregate_metrics, evaluate_clip


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    tolerance = float(ground_truth["dataset_metadata"]["matching_tolerance_seconds"])
    predicted_by_name = {Path(item["source"]).name: item for item in predictions["videos"]}
    results: list[dict[str, Any]] = []
    for clip in ground_truth["clips"]:
        file_name = str(clip["file_name"])
        predicted = predicted_by_name.get(file_name)
        if predicted is None:
            raise RuntimeError(f"prediction missing for {file_name}")
        if predicted["source_sha256"] != clip["source_sha256"]:
            raise RuntimeError(f"source hash mismatch for {file_name}")
        duration = float(predicted["duration_seconds"])
        if abs(duration - float(clip["duration_sec"])) > 0.01:
            raise RuntimeError(f"duration mismatch for {file_name}")
        result = evaluate_clip(
            predicted["events"],
            clip["ground_truth_events"],
            duration_seconds=duration,
            tolerance_seconds=tolerance,
        )
        result.update(
            {
                "clip_id": clip["clip_id"],
                "file_name": file_name,
                "title": clip["title"],
                "source_sha256": clip["source_sha256"],
            }
        )
        results.append(result)
    aggregate = aggregate_metrics(results)
    output = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "scope": "fixed_camera_desktop_functional_demo_only",
        "benchmark_claim_allowed": False,
        "predictions_path": str(args.predictions),
        "ground_truth_path": str(args.ground_truth),
        "protocol": {
            "matching": "maximum-cardinality bipartite",
            "event_identity": ["event_type", "component"],
            "instantaneous_tolerance_seconds": tolerance,
            "missing_window": "[gt_start, decoded_video_duration + tolerance]",
            "one_to_one": True,
        },
        "clips": results,
        "aggregate": aggregate,
        "limitations": [
            "five supplied fixed-camera desktop clips only",
            "manual GT was finalized after system review and is not a blinded industrial benchmark",
            "REMOVE actions are outside the current visual-front-end event vocabulary",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(aggregate["micro"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
