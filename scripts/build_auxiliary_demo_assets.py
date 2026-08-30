#!/usr/bin/env python3
"""Generate the curated auxiliary package's representative input and heatmap."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    with (args.run_dir / "test_inputs.csv").open(encoding="utf-8", newline="") as stream:
        inputs = list(csv.DictReader(stream))
    sample = inputs[0]
    maps = np.load(args.run_dir / "strict_result_v2/prediction_maps.npz")["predictions"]
    normalized = cv2.normalize(maps[0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample["path"], args.output_dir / "sample_input.png")
    if not cv2.imwrite(str(args.output_dir / "expected_heatmap.png"), heatmap):
        raise RuntimeError("failed to write expected heatmap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
