from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests  # type: ignore[import-untyped]

REPOSITORY = "Voxel51/mvtec-ad"
COMMIT = "30a183a3b96e3aef953f230784b123b719b09d97"
BASE_URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{COMMIT}"
API_URL = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{COMMIT}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(url: str) -> Any:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def expected_hashes(data_directories: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for directory in sorted(data_directories):
        items = get_json(f"{API_URL}/data/{directory}?recursive=true&expand=false")
        for item in items:
            if item.get("type") != "file":
                continue
            lfs = item.get("lfs") or {}
            digest = lfs.get("oid")
            if digest:
                result[str(item["path"])] = str(digest)
    return result


def download_one(remote_path: str, expected: str, destination: Path) -> tuple[str, str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == expected:
        return remote_path, expected, destination.stat().st_size
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    with requests.get(f"{BASE_URL}/{remote_path}", stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                if not chunk:
                    continue
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"hash mismatch for {remote_path}: {actual} != {expected}")
    temporary.replace(destination)
    destination.chmod(0o444)
    return remote_path, actual, size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    samples_payload = get_json(f"{BASE_URL}/samples.json")
    samples = [
        sample for sample in samples_payload["samples"] if sample["category"]["label"] == "bottle"
    ]
    if len(samples) != 292:
        raise RuntimeError(f"expected 292 bottle samples, found {len(samples)}")
    data_directories = {sample["filepath"].split("/")[1] for sample in samples}
    hashes = expected_hashes(data_directories)
    jobs: list[tuple[dict[str, Any], str, Path]] = []
    for sample in samples:
        remote_path = str(sample["filepath"])
        expected = hashes.get(remote_path)
        if not expected:
            raise RuntimeError(f"missing LFS SHA-256 for {remote_path}")
        defect = str(sample["defect"]["label"])
        split = str(sample["split"])
        sample_id = str(sample["_id"]["$oid"])
        destination = output_root / "bottle" / split / defect / f"{sample_id}.png"
        jobs.append((sample, expected, destination))

    receipts: dict[str, tuple[str, int]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, str(sample["filepath"]), expected, destination): (
                sample,
                destination,
            )
            for sample, expected, destination in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            sample, destination = futures[future]
            remote_path, digest, size = future.result()
            receipts[remote_path] = (digest, size)
            if completed % 25 == 0 or completed == len(futures):
                print(json.dumps({"downloaded_or_verified": completed, "total": len(futures)}))

    manifest_path = output_root / "bottle_manifest.csv"
    fields = [
        "sample_id",
        "path",
        "label",
        "defect_type",
        "product_id",
        "source",
        "license_id",
        "official_split",
        "content_sha256",
        "remote_path",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample, _, destination in sorted(jobs, key=lambda item: str(item[0]["_id"]["$oid"])):
            remote_path = str(sample["filepath"])
            defect = str(sample["defect"]["label"])
            writer.writerow(
                {
                    "sample_id": sample["_id"]["$oid"],
                    "path": str(destination),
                    "label": "normal" if defect == "good" else "anomaly",
                    "defect_type": "" if defect == "good" else defect,
                    "product_id": "mvtec_ad_bottle",
                    "source": f"huggingface:{REPOSITORY}@{COMMIT}",
                    "license_id": "CC-BY-NC-SA-4.0",
                    "official_split": sample["split"],
                    "content_sha256": receipts[remote_path][0],
                    "remote_path": remote_path,
                }
            )
    manifest_hash = sha256(manifest_path)
    receipt = {
        "schema_version": 1,
        "dataset": "MVTec_AD",
        "category": "bottle",
        "mirror": REPOSITORY,
        "mirror_commit": COMMIT,
        "license": "CC-BY-NC-SA-4.0",
        "sample_count": len(samples),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "downloaded_bytes": sum(size for _, size in receipts.values()),
        "warning": "Community mirror of MVTec AD; not an official archive checksum reproduction.",
    }
    (output_root / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(manifest_path, 0o444)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
