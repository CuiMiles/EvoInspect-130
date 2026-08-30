#!/usr/bin/env python3
"""Build a frozen deployment bundle or an explicit failed-gate diagnostic bundle."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import yaml

from evoinspect.provenance import file_sha256, utc_now, write_json


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-gate", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--test-inputs", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-failed-gate-diagnostic",
        action="store_true",
        help="package a frozen failed-gate model for hardware diagnostics only",
    )
    parser.add_argument("--representative-category")
    parser.add_argument("--representative-seed", type=int)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    upstream = repo / "third_party/anomalib-2.3.0"
    image = args.image
    if args.test_inputs is not None:
        with args.test_inputs.open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        image = Path(row["path"])
    assert image is not None
    for path in (
        args.quality_gate,
        args.checkpoint,
        args.metrics,
        image,
        args.config,
        upstream / "pyproject.toml",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    gate = load_json(args.quality_gate)
    metrics = load_json(args.metrics)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gate_passed = gate.get("passed") is True
    if not gate_passed and not args.allow_failed_gate_diagnostic:
        raise RuntimeError("refusing deployment bundle: frozen quality gate did not pass")
    if metrics.get("model_id") != config.get("model_id"):
        raise RuntimeError("metrics model_id does not match deployment config")
    deployment = config.get("deployment_benchmark", {})
    representative_category = args.representative_category or deployment.get(
        "representative_category"
    )
    representative_seed = args.representative_seed
    if representative_seed is None:
        representative_seed = deployment.get("representative_seed")
    if representative_category is None or representative_seed is None:
        raise RuntimeError(
            "deployment representative is missing; pass --representative-category and "
            "--representative-seed for an explicit diagnostic selection"
        )
    category = str(metrics.get("category", "")).removeprefix("mvtec_ad_")
    if category != str(representative_category):
        raise RuntimeError("metrics is not the predeclared deployment benchmark category")
    if int(metrics.get("seed", -1)) != int(representative_seed):
        raise RuntimeError("metrics is not the predeclared deployment benchmark seed")
    threshold = metrics.get("calibration", {}).get("threshold_development_only", {}).get(
        "threshold"
    )
    if not isinstance(threshold, int | float):
        raise RuntimeError("metrics lacks a development-only frozen threshold")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if dirty:
        raise RuntimeError("refusing deployment bundle from a modified tracked worktree")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")

    with tempfile.TemporaryDirectory(prefix="evoinspect-2060-") as temporary:
        root = Path(temporary) / "evoinspect-2060-bundle"
        root.mkdir()
        archive = subprocess.Popen(["git", "archive", "HEAD"], cwd=repo, stdout=subprocess.PIPE)
        assert archive.stdout is not None
        with tarfile.open(fileobj=archive.stdout, mode="r|") as source:
            source.extractall(root, filter="data")
        if archive.wait() != 0:
            raise RuntimeError("git archive failed")
        shutil.copytree(
            upstream,
            root / "third_party/anomalib-2.3.0",
            ignore=shutil.ignore_patterns(".git"),
        )
        payload = root / "deployment_payload"
        payload.mkdir()
        copies = {
            "model.ckpt": args.checkpoint,
            "metrics.json": args.metrics,
            "quality-gate.json": args.quality_gate,
            "benchmark_input.png": image,
            "config.yaml": args.config,
        }
        for name, source in copies.items():
            shutil.copy2(source, payload / name)
        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "git_commit": commit,
            "model_id": config["model_id"],
            "model_size": config["model_size"],
            "quality_gate_passed": gate_passed,
            "diagnostic_only": not gate_passed,
            "claim_eligible": gate_passed,
            "representative_category": category,
            "representative_seed": int(representative_seed),
            "files": {
                name: {
                    "sha256": file_sha256(payload / name),
                    "bytes": (payload / name).stat().st_size,
                }
                for name in sorted(copies)
            },
            "scope": (
                "frozen EfficientAD 2500x2500 latency measurement on actual target hardware"
                if gate_passed
                else "failed-quality-gate EfficientAD hardware diagnostic; no deployment claim"
            ),
        }
        write_json(payload / "bundle_manifest.json", manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.output, "w:gz") as output:
            output.add(root, arcname=root.name)
    print(f"{args.output} sha256={file_sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
