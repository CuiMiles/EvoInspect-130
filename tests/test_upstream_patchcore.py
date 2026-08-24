from __future__ import annotations

import csv
from pathlib import Path

import pytest

from evoinspect.upstream_patchcore import read_key_values, read_upstream_result


def test_read_key_values(tmp_path: Path) -> None:
    path = tmp_path / "meta.txt"
    path.write_text("seed=0\ncategory=bottle\n", encoding="utf-8")
    assert read_key_values(path) == {"seed": "0", "category": "bottle"}


def test_read_upstream_result(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["Row Names", "instance_auroc", "full_pixel_auroc", "anomaly_pixel_auroc"]
        )
        writer.writerow(["mvtec_bottle", "1.0", "0.98", "0.97"])
        writer.writerow(["Mean", "1.0", "0.98", "0.97"])
    category, metrics = read_upstream_result(path)
    assert category == "bottle"
    assert metrics["instance_auroc"] == pytest.approx(1.0)
    assert metrics["full_pixel_auroc"] == pytest.approx(0.98)
