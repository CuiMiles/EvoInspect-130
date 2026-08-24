from __future__ import annotations

import csv
import platform
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from .provenance import append_csv, canonical_hash, file_sha256, git_state

REGISTRY_COLUMNS = [
    "run_id",
    "status",
    "start_time",
    "end_time",
    "git_commit",
    "dirty",
    "config_hash",
    "data_hash",
    "split_hash",
    "seed",
    "hardware",
    "model",
    "protocol",
    "metrics_path",
    "artifact_path",
    "failure_reason",
    "notes",
]

METRIC_NAMES = ("instance_auroc", "full_pixel_auroc", "anomaly_pixel_auroc")


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        if not separator or not key:
            raise RuntimeError(f"invalid key-value line in {path}: {line!r}")
        values[key] = value
    return values


def read_upstream_result(path: Path) -> tuple[str, dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    category_rows = [row for row in rows if row.get("Row Names") != "Mean"]
    if len(category_rows) != 1:
        raise RuntimeError(f"expected one category row in {path}, got {len(category_rows)}")
    row = category_rows[0]
    dataset_name = row.get("Row Names", "")
    if not dataset_name.startswith("mvtec_"):
        raise RuntimeError(f"unexpected dataset row in {path}: {dataset_name!r}")
    metrics = {name: float(row[name]) for name in METRIC_NAMES}
    if any(not 0.0 <= value <= 1.0 for value in metrics.values()):
        raise RuntimeError(f"metric outside [0, 1] in {path}: {metrics}")
    return dataset_name.removeprefix("mvtec_"), metrics


def _one_match(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} under {root}, got {len(matches)}")
    return matches[0]


def collect_batch(batch_root: Path) -> dict[str, Any]:
    static = read_key_values(batch_root / "static-provenance.txt")
    runs: list[dict[str, Any]] = []
    for run_dir in sorted((batch_root / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        metadata = read_key_values(run_dir / "run-meta.txt")
        if metadata.get("status") != "completed":
            continue
        result_path = _one_match(run_dir, "results.csv")
        category, metrics = read_upstream_result(result_path)
        if category != metadata["category"]:
            raise RuntimeError(
                f"category mismatch for {run_dir}: {category} != {metadata['category']}"
            )
        model_dir = _one_match(run_dir, "patchcore_params.pkl").parent
        model_files = sorted(path for path in model_dir.iterdir() if path.is_file())
        file_hashes = [(path.name, file_sha256(path)) for path in model_files]
        run = {
            "run_id": metadata["run_id"],
            "category": category,
            "seed": int(metadata["seed"]),
            "physical_gpu": int(metadata["physical_gpu"]),
            "started_at": metadata["started_at"],
            "completed_at": metadata["completed_at"],
            "metrics": metrics,
            "results_path": str(result_path),
            "results_sha256": file_sha256(result_path),
            "model_path": str(model_dir),
            "model_bytes": sum(path.stat().st_size for path in model_files),
            "model_files": [
                {"name": name, "sha256": sha256} for name, sha256 in file_hashes
            ],
            "model_hash": canonical_hash(file_hashes),
            "split_hash": canonical_hash(
                {
                    "protocol": "mvtec_ad_standard_public_benchmark",
                    "category": category,
                    "manifest_sha256": static["manifest_sha256"],
                }
            ),
        }
        runs.append(run)
    categories = sorted({str(run["category"]) for run in runs})
    macro = {
        metric: mean(float(run["metrics"][metric]) for run in runs)
        for metric in METRIC_NAMES
    }
    return {
        "schema_version": 1,
        "status": (
            "mvtec_15_category_upstream_patchcore_standard_baseline"
            if len(categories) == 15
            else "partial_upstream_patchcore_standard_baseline"
        ),
        "protocol": "mvtec_ad_standard_public_benchmark",
        "dataset": "MVTec_AD_direct_archive",
        "upstream_repository": "https://github.com/amazon-science/patchcore-inspection.git",
        "upstream_commit": static["upstream_commit"],
        "upstream_license": static["upstream_license"],
        "manifest_sha256": static["manifest_sha256"],
        "weights_sha256": static["weights_sha256"],
        "config_sha256": static["config_sha256"],
        "categories": categories,
        "seeds": sorted({int(run["seed"]) for run in runs}),
        "run_count": len(runs),
        "macro_category_mean": macro,
        "runs": runs,
        "warning": (
            "Formal pinned upstream PatchCore implementation under the standard MVTec AD "
            "protocol; not directly comparable with the official-style 100+30 protocol."
        ),
    }


def register_batch(summary: dict[str, Any], registry_path: Path, repo_root: Path) -> None:
    commit, dirty = git_state(repo_root)
    for run in summary["runs"]:
        append_csv(
            registry_path,
            REGISTRY_COLUMNS,
            {
                "run_id": run["run_id"],
                "status": "completed_standard_baseline",
                "start_time": run["started_at"],
                "end_time": run["completed_at"],
                "git_commit": commit,
                "dirty": str(dirty).lower(),
                "config_hash": summary["config_sha256"],
                "data_hash": summary["manifest_sha256"],
                "split_hash": run["split_hash"],
                "seed": run["seed"],
                "hardware": (
                    f"RTX 3090 physical GPU {run['physical_gpu']}; {platform.platform()}; "
                    f"torch {torch.__version__}; FAISS CPU nearest-neighbour index"
                ),
                "model": "upstream-patchcore-wrn50-l2-l3-p01",
                "protocol": summary["protocol"],
                "metrics_path": run["results_path"],
                "artifact_path": run["model_path"],
                "failure_reason": "",
                "notes": (
                    f"Pinned upstream {summary['upstream_commit']}; category={run['category']}; "
                    "standard MVTec protocol; not official-style 100+30"
                ),
            },
        )
