#!/usr/bin/env python3
"""Export a compact, image-free score pack for clean GuardedAdapt replay."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

from evoinspect.provenance import file_sha256


def truth(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {
            row["sample_id"]: int(row["label"] == "anomaly")
            for row in csv.DictReader(stream)
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-batch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_dirs = sorted(
        path
        for path in (args.source_batch / "runs").iterdir()
        if (path / "metrics.json").is_file()
    )
    if len(run_dirs) != 75:
        raise RuntimeError(f"expected 75 runs, found {len(run_dirs)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for run_dir in run_dirs:
                labels = truth(run_dir / "test_truth.csv")
                records = []
                with (run_dir / "predictions.jsonl").open(encoding="utf-8") as stream:
                    for line in stream:
                        item = json.loads(line)
                        records.append(
                            {
                                "sample_id": item["sample_id"],
                                "score": float(item["upstream_patchcore_score"]),
                                "label": labels[item["sample_id"]],
                            }
                        )
                split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
                meta = json.loads((run_dir / "model" / "meta.json").read_text(encoding="utf-8"))
                item = {
                    "split": {
                        "category": split["category"],
                        "seed": split["seed"],
                        "split_hash": split["split_hash"],
                    },
                    "initial_threshold": float(meta["threshold"]["threshold"]),
                    "records": records,
                    "evidence": {
                        "predictions_sha256": file_sha256(run_dir / "predictions.jsonl"),
                        "truth_sha256": file_sha256(run_dir / "test_truth.csv"),
                    },
                }
                compressed.write((json.dumps(item, sort_keys=True) + "\n").encode())
    print(f"{args.output} {args.output.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
