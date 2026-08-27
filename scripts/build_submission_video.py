#!/usr/bin/env python3
"""Build a <=5 minute evidence-bounded MP4 from the annotated video demo."""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH = 1280
HEIGHT = 720
FPS = 30.0
FONT_PATH = "/usr/share/fonts/todesk/NotoSansCJK-Regular.ttc"
BOLD_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD_PATH if bold else FONT_PATH, size)


def slide(title: str, lines: list[str]) -> np.ndarray:
    image = Image.new("RGB", (WIDTH, HEIGHT), (12, 25, 43))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 24, HEIGHT), fill=(24, 160, 174))
    draw.text((80, 75), title, font=font(48, bold=True), fill=(232, 250, 252))
    y = 175
    for line in lines:
        draw.text((84, y), line, font=font(27), fill=(218, 229, 239))
        y += 58
    draw.text(
        (84, HEIGHT - 64),
        "功能验证，不作为工业 benchmark；GTX 2060 性能待实机验证",
        font=font(20),
        fill=(251, 146, 60),
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def repeat_frame(writer: Any, frame: np.ndarray, seconds: float) -> None:
    for _ in range(round(seconds * FPS)):
        writer.write(frame)


def video_panel(frame: np.ndarray, label: str, events: list[dict[str, Any]]) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), (43, 25, 12), dtype=np.uint8)
    scale = HEIGHT / frame.shape[0]
    resized = cv2.resize(frame, (round(frame.shape[1] * scale), HEIGHT))
    x0 = 54
    canvas[:, x0 : x0 + resized.shape[1]] = resized
    right = x0 + resized.shape[1] + 40
    cv2.putText(canvas, label, (right, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (174, 230, 241), 2)
    cv2.putText(
        canvas,
        "Expected: bottle -> cup -> mouse",
        (right, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.63,
        (220, 220, 220),
        1,
    )
    y = 180
    for event in events[:8]:
        kind = event["public_kind"] or "completed"
        text = f"{kind}: {event['step']} @ {event['start_time_s']:.1f}s"
        color = (80, 210, 255) if event["public_kind"] else (130, 230, 160)
        cv2.putText(canvas, text, (right, y), cv2.FONT_HERSHEY_SIMPLEX, 0.61, color, 1)
        y += 42
    cv2.putText(
        canvas,
        "OpenCV decode + component detector + FSM",
        (right, HEIGHT - 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (190, 190, 190),
        1,
    )
    cv2.putText(
        canvas,
        "FUNCTIONAL DEMO ONLY",
        (right, HEIGHT - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (50, 120, 255),
        2,
    )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.video_report.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {args.output}")
    try:
        repeat_frame(
            writer,
            slide(
                "智检演化130 · EvoInspect-130",
                [
                    "Accuracy Engine：冻结 PatchCore 强基线",
                    "Edge Engine：EfficientAD-M 复现中，S 为速度 fallback",
                    "GuardedAdapt：有害更新门禁与可回滚反馈闭环",
                    "视频：真实解码、组件检测、装配顺序 FSM",
                ],
            ),
            6.0,
        )
        for index, item in enumerate(report["videos"], start=1):
            anomaly_names = [event["public_kind"] for event in item["public_anomalies"]]
            summary = "normal sequence" if not anomaly_names else ", ".join(anomaly_names)
            repeat_frame(
                writer,
                slide(
                    f"视频 {index}/5",
                    [
                        f"时长：{item['duration_seconds']:.2f} 秒",
                        f"FSM 输出：{summary}",
                        "输入为固定机位桌面装配素材",
                    ],
                ),
                2.0,
            )
            capture = cv2.VideoCapture(item["annotated_video"])
            if not capture.isOpened():
                raise RuntimeError(f"cannot open {item['annotated_video']}")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(video_panel(frame, f"Video {index}/5", item["events"]))
            capture.release()
        repeat_frame(
            writer,
            slide(
                "证据边界与待完成项",
                [
                    "MVTec AD PatchCore：Overall F1 0.9224，Image AUROC 0.9817",
                    "GuardedAdapt：75 个冻结真实分数流反馈回放已完成",
                    "RCBR 未通过预注册门禁，仅保留为研究负结果",
                    "EfficientAD-M 质量门与真实 GTX 2060 时延仍待完成",
                ],
            ),
            8.0,
        )
    finally:
        writer.release()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
