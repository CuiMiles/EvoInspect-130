#!/usr/bin/env python3
"""Build submission figures directly from frozen machine-readable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1600, 900
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size=size)
    return ImageFont.load_default()


def save(image: Image.Image, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def pareto(report: dict[str, Any], output: Path, report_sha: str) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "Target gain vs. harmful-update rate", font=font(48), fill="#0f172a")
    draw.text((72, 95), "Frozen 219-replay GuardedAdapt comparison", font=font(27), fill="#475569")
    left, top, right, bottom = 150, 170, 1510, 760
    draw.line((left, bottom, right, bottom), fill="#334155", width=4)
    draw.line((left, bottom, left, top), fill="#334155", width=4)
    for index in range(6):
        x = left + (right - left) * index / 5
        value = -0.01 + 0.04 * index / 5
        draw.line((x, bottom, x, bottom + 10), fill="#334155", width=2)
        draw.text((x - 30, bottom + 18), f"{value:.3f}", font=font(21), fill="#475569")
    for index in range(6):
        y = bottom - (bottom - top) * index / 5
        value = 0.5 * index / 5
        draw.line((left - 10, y, left, y), fill="#334155", width=2)
        draw.text((55, y - 14), f"{value:.2f}", font=font(21), fill="#475569")
    draw.text((650, 825), "Mean target F1 gain", font=font(27), fill="#334155")
    vertical_label = Image.new("RGBA", (280, 50), (0, 0, 0, 0))
    ImageDraw.Draw(vertical_label).text(
        (0, 5), "Harmful-update rate", font=font(25), fill="#334155"
    )
    vertical_label = vertical_label.rotate(90, expand=True)
    image.paste(vertical_label, (10, 315), vertical_label)
    colors = {
        "NaiveUpdate": "#dc2626",
        "BoundedThreshold": "#f59e0b",
        "GuardedAdapt-v1": "#2563eb",
        "GuardedAdapt-Risk": "#059669",
        "NoUpdate": "#64748b",
    }
    for name, values in report["strategies"].items():
        gain = float(values["target_gain_mean"])
        harmful = float(values["harmful_update_rate"])
        x = left + (gain + 0.01) / 0.04 * (right - left)
        y = bottom - harmful / 0.5 * (bottom - top)
        color = colors.get(name, "#7c3aed")
        draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill=color, outline="white", width=3)
        label = name
        label_x, label_y = x + 18, y - 17
        if name == "GuardedAdapt-Risk":
            label += " (0% accepted; FAIL)"
            label_y = y - 82
        elif name == "NoUpdate":
            label_y = y - 47
        elif name == "BoundedThreshold":
            label_x = x - 260
        draw.text((label_x, label_y), label, font=font(23), fill=color)
    draw.text((72, 865), f"source_sha256={report_sha}", font=font(17), fill="#64748b")
    save(image, output)


def flow(output: Path, report_sha: str) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "GuardedAdapt-Risk frozen release protocol", font=font(46), fill="#0f172a")
    draw.text(
        (72, 95),
        "Implemented chain and observed preregistered outcome",
        font=font(27),
        fill="#475569",
    )
    boxes = [
        ("Drift trigger", "KS p<0.01 + median shift\n2 consecutive windows"),
        ("Candidate", "Bounded threshold\nMemory replace <=5%"),
        ("Risk gate", "Target gain LCB > 0\nFPR/FNR group budgets"),
        ("Shadow", "32 samples\nChampion stays live"),
        ("Release / rollback", "Promote if safe\notherwise exact restore"),
    ]
    box_w, box_h, gap = 260, 250, 42
    start_x, y = 50, 230
    for index, (title, detail) in enumerate(boxes):
        x = start_x + index * (box_w + gap)
        fill = "#e0f2fe" if index < 2 else "#ecfdf5"
        draw.rounded_rectangle(
            (x, y, x + box_w, y + box_h),
            radius=24,
            fill=fill,
            outline="#0f766e",
            width=4,
        )
        draw.text((x + 18, y + 24), title, font=font(29), fill="#0f172a")
        draw.multiline_text((x + 18, y + 82), detail, font=font(21), fill="#334155", spacing=10)
        if index < len(boxes) - 1:
            x1, x2, cy = x + box_w + 8, x + box_w + gap - 8, y + box_h / 2
            draw.line((x1, cy, x2, cy), fill="#0f766e", width=6)
            draw.polygon(((x2, cy), (x2 - 18, cy - 12), (x2 - 18, cy + 12)), fill="#0f766e")
    draw.rounded_rectangle(
        (235, 585, 1365, 760), radius=25, fill="#fff7ed", outline="#c2410c", width=5
    )
    draw.text(
        (270, 615),
        "Observed: 50/50 harmful v1 candidates blocked; 219/219 rollback exact",
        font=font(29),
        fill="#9a3412",
    )
    draw.text(
        (270, 675),
        "Failure: 0/219 updates accepted (required >=40%); not a positive innovation claim",
        font=font(26),
        fill="#b91c1c",
    )
    draw.text((72, 865), f"source_sha256={report_sha}", font=font(17), fill="#64748b")
    save(image, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    if report.get("total_replays") != 219 or report.get("passed") is not False:
        raise RuntimeError("expected frozen failed 219-replay report")
    report_sha = hashlib.sha256(raw).hexdigest()
    pareto(report, args.output_dir / "guarded_risk_pareto.png", report_sha)
    flow(args.output_dir / "guarded_risk_release_flow.png", report_sha)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
