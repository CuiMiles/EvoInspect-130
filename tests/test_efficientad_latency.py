from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from scripts.benchmark_efficientad_latency import source_png


def test_source_png_accepts_self_contained_image(tmp_path: Path) -> None:
    image = tmp_path / "benchmark.png"
    Image.new("RGB", (10, 8), (12, 34, 56)).save(image)

    encoded, sample_id, source = source_png(32, image_path=image)

    assert encoded.startswith(b"\x89PNG")
    assert sample_id == "benchmark"
    assert source == image


def test_source_png_accepts_manifest(tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    Image.new("RGB", (10, 8), (12, 34, 56)).save(image)
    manifest = tmp_path / "test_inputs.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "path"])
        writer.writeheader()
        writer.writerow({"sample_id": "fixed-id", "path": str(image)})

    _, sample_id, source = source_png(32, inputs=manifest)

    assert sample_id == "fixed-id"
    assert source == image


def test_source_png_requires_one_source(tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    with pytest.raises(ValueError, match="exactly one"):
        source_png(32)
    with pytest.raises(ValueError, match="exactly one"):
        source_png(32, inputs=tmp_path / "inputs.csv", image_path=image)
