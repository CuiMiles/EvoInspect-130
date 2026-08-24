from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from scripts.build_mvtec_ad_manifest import build_manifest


def save_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (4, 4), color=value).save(path)


def test_build_manifest_separates_masks_and_verifies_images(tmp_path: Path) -> None:
    dataset = tmp_path / "mvtec"
    save_image(dataset / "widget" / "train" / "good" / "000.png", 10)
    save_image(dataset / "widget" / "test" / "good" / "001.png", 20)
    save_image(dataset / "widget" / "test" / "crack" / "002.png", 30)
    save_image(dataset / "widget" / "ground_truth" / "crack" / "002_mask.png", 255)
    output = tmp_path / "manifest.csv"

    receipt = build_manifest(dataset, output, "a" * 64)

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert receipt["sample_count"] == 3
    assert receipt["content_duplicate_count"] == 0
    anomaly = next(row for row in rows if row["label"] == "anomaly")
    assert anomaly["defect_type"] == "crack"
    assert anomaly["mask_path"].endswith("002_mask.png")
    assert all("ground_truth" not in row["path"] for row in rows)
    assert output.stat().st_mode & 0o222 == 0
