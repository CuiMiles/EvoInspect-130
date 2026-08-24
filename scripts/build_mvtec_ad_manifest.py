from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MANIFEST_COLUMNS = [
    "sample_id",
    "path",
    "label",
    "defect_type",
    "product_id",
    "source",
    "license_id",
    "official_split",
    "content_sha256",
    "mask_path",
    "mask_sha256",
    "width",
    "height",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_manifest(
    dataset_root: Path,
    output: Path,
    archive_sha256: str,
    category_filter: str | None = None,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    categories = sorted(
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()
    )
    if not categories:
        raise RuntimeError(f"no MVTec AD categories found under {dataset_root}")
    if category_filter is not None:
        if category_filter not in categories:
            raise RuntimeError(f"category not found: {category_filter}")
        categories = [category_filter]

    rows: list[dict[str, Any]] = []
    content_owners: dict[str, str] = {}
    for category in categories:
        category_root = dataset_root / category
        for split in ("train", "test"):
            split_root = category_root / split
            for path in image_paths(split_root):
                relative = path.relative_to(split_root)
                if len(relative.parts) != 2:
                    raise RuntimeError(f"unexpected image path layout: {path}")
                defect_type = relative.parts[0]
                normal = defect_type == "good"
                digest = sha256(path)
                owner = content_owners.get(digest)
                if owner is not None:
                    raise RuntimeError(f"content duplicate: {path} and {owner} share {digest}")
                content_owners[digest] = str(path)
                mask_path = ""
                mask_digest = ""
                if split == "test" and not normal:
                    candidate = (
                        category_root
                        / "ground_truth"
                        / defect_type
                        / f"{path.stem}_mask.png"
                    )
                    if not candidate.is_file():
                        raise RuntimeError(f"missing ground-truth mask: {candidate}")
                    mask_path = str(candidate.resolve())
                    mask_digest = sha256(candidate)
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                rows.append(
                    {
                        "sample_id": f"mvtec-{category}-{split}-{defect_type}-{path.stem}",
                        "path": str(path.resolve()),
                        "label": "normal" if normal else "anomaly",
                        "defect_type": "" if normal else defect_type,
                        "product_id": f"mvtec_ad_{category}",
                        "source": "MVTec_AD_official_mydrive_archive",
                        "license_id": "CC-BY-NC-SA-4.0",
                        "official_split": split,
                        "content_sha256": digest,
                        "mask_path": mask_path,
                        "mask_sha256": mask_digest,
                        "width": width,
                        "height": height,
                    }
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    output.chmod(0o444)

    by_category = Counter(str(row["product_id"]) for row in rows)
    by_label = Counter(str(row["label"]) for row in rows)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "MVTec_AD",
        "source": "official_mydrive_direct_archive",
        "archive_sha256": archive_sha256,
        "dataset_root": str(dataset_root),
        "manifest_path": str(output.resolve()),
        "manifest_sha256": sha256(output),
        "categories": categories,
        "category_count": len(categories),
        "sample_count": len(rows),
        "counts_by_category": dict(sorted(by_category.items())),
        "counts_by_label": dict(sorted(by_label.items())),
        "content_duplicate_count": 0,
        "warning": "Archive identity still depends on the recorded URL and human source review.",
    }
    receipt_path = output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path.chmod(0o444)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--category")
    args = parser.parse_args()
    receipt = build_manifest(args.dataset_root, args.output, args.archive_sha256, args.category)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
