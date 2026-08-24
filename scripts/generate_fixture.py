from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path


def write_pgm(path: Path, pixels: Iterable[int], width: int = 16, height: int = 16) -> None:
    values = list(pixels)
    if len(values) != width * height:
        raise ValueError("wrong fixture pixel count")
    path.write_text(
        f"P2\n{width} {height}\n255\n" + " ".join(str(value) for value in values) + "\n",
        encoding="ascii",
    )


def generate(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for index in range(10):
        pixels = [40 + index + ((x + y + index) % 3) for y in range(16) for x in range(16)]
        filename = f"normal_{index:02d}.pgm"
        write_pgm(output_dir / filename, pixels)
        rows.append(
            {
                "sample_id": f"normal-{index:02d}",
                "path": filename,
                "label": "normal",
                "defect_type": "",
                "product_id": "fixture-product",
                "source": "generated_smoke_fixture",
                "license_id": "repository-test-code",
            }
        )
    for defect_type, count in (("scratch", 6), ("dent", 4)):
        for index in range(count):
            base = 44 + index
            pixels = [base + ((x + y) % 2) for y in range(16) for x in range(16)]
            if defect_type == "scratch":
                column = 3 + index
                for y in range(16):
                    pixels[y * 16 + column] = 220 - index
            else:
                start = 5 + (index % 2)
                for y in range(start, start + 4):
                    for x in range(start, start + 4):
                        pixels[y * 16 + x] = 2 + index
            filename = f"{defect_type}_{index:02d}.pgm"
            write_pgm(output_dir / filename, pixels)
            rows.append(
                {
                    "sample_id": f"{defect_type}-{index:02d}",
                    "path": filename,
                    "label": "anomaly",
                    "defect_type": defect_type,
                    "product_id": "fixture-product",
                    "source": "generated_smoke_fixture",
                    "license_id": "repository-test-code",
                }
            )
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(generate(args.output_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
