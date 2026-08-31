#!/usr/bin/env python3
# ruff: noqa: RUF001, UP035
"""Build the one-page submission introduction with a clean PDF text layer."""

import argparse
import re
from pathlib import Path
from typing import Sequence

import yaml  # type: ignore[import-untyped]
from reportlab.lib.colors import HexColor  # type: ignore[import-untyped]
from reportlab.lib.pagesizes import A4  # type: ignore[import-untyped]
from reportlab.pdfbase import pdfmetrics  # type: ignore[import-untyped]
from reportlab.pdfbase.cidfonts import UnicodeCIDFont  # type: ignore[import-untyped]
from reportlab.pdfgen.canvas import Canvas  # type: ignore[import-untyped]

FONT_NAME = "STSong-Light"


def wrapped_lines(text: str, *, font_size: float, max_width: float) -> Sequence[str]:
    """Wrap mixed Chinese/Latin text by rendered width."""
    lines = []
    current = ""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.+×%/:-]*|.", text, flags=re.DOTALL)
    for token in tokens:
        candidate = current + token
        if current and pdfmetrics.stringWidth(candidate, FONT_NAME, font_size) > max_width:
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intro", type=Path, default=Path("submission/works_intro.txt"))
    parser.add_argument("--metadata", type=Path, default=Path("submission/metadata.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = args.intro.read_text(encoding="utf-8").strip()
    metadata = yaml.safe_load(args.metadata.read_text(encoding="utf-8"))
    chinese_characters = len(re.findall(r"[\u3400-\u9fff]", text))
    non_whitespace_characters = len(re.sub(r"\s", "", text))
    if chinese_characters > 300 or non_whitespace_characters > 300:
        raise ValueError(
            "introduction exceeds limit: "
            f"Chinese={chinese_characters}, non-whitespace={non_whitespace_characters}"
        )

    pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = A4
    margin = 42.0
    content_width = page_width - 2 * margin
    canvas = Canvas(str(args.output), pagesize=A4, pageCompression=1, invariant=1)
    canvas.setTitle("智检演化130（EvoInspect-130）参赛作品简介")
    canvas.setAuthor(str(metadata["leader"]))

    y = page_height - 62
    canvas.setFillColor(HexColor("#0B5671"))
    canvas.setFont(FONT_NAME, 20)
    canvas.drawString(margin, y, "智检演化130（EvoInspect-130）")
    y -= 30
    canvas.setFillColor(HexColor("#6B7280"))
    canvas.setFont(FONT_NAME, 11)
    canvas.drawString(
        margin,
        y,
        f"参赛作品简介 / {metadata['team_name']} / {metadata['school']}",
    )

    body_lines = wrapped_lines(text, font_size=10.0, max_width=content_width - 28)
    line_height = 20
    box_height = len(body_lines) * line_height + 30
    y -= 34
    canvas.setFillColor(HexColor("#F2FAFB"))
    canvas.rect(margin, y - box_height, content_width, box_height, fill=1, stroke=0)
    canvas.setFillColor(HexColor("#18A0AE"))
    canvas.rect(margin, y - box_height, 5, box_height, fill=1, stroke=0)
    canvas.setFillColor(HexColor("#172033"))
    canvas.setFont(FONT_NAME, 10.0)
    line_y = y - 21
    for line in body_lines:
        canvas.drawString(margin + 17, line_y, line)
        line_y -= line_height

    y -= box_height + 26
    canvas.setFillColor(HexColor("#0F766E"))
    canvas.setFont(FONT_NAME, 8.5)
    canvas.drawString(
        margin,
        y,
        "　".join(
            [
                f"团队：{metadata['team_name']}",
                f"队长：{metadata['leader']}（单人团队）",
                f"学校：{metadata['school']}",
                str(metadata["competition_group"]),
            ]
        ),
    )
    canvas.showPage()
    canvas.save()
    print(
        f"generated {args.output} Chinese={chinese_characters} "
        f"non_whitespace={non_whitespace_characters}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
