#!/usr/bin/env python3
"""Build the competition-facing EvoInspect-130 project video."""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS = 1280, 720, 30.0
FONT_PATH = "/usr/share/fonts/todesk/NotoSansCJK-Regular.ttc"
BOLD_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD_PATH if bold else FONT_PATH, size)


def slide(title: str, lines: list[str], *, accent: str = "EVIDENCE") -> np.ndarray:
    image = Image.new("RGB", (WIDTH, HEIGHT), (12, 25, 43))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 24, HEIGHT), fill=(24, 160, 174))
    draw.text((80, 62), title, font=font(46, bold=True), fill=(232, 250, 252))
    y = 160
    for line in lines:
        draw.text((84, y), line, font=font(27), fill=(218, 229, 239))
        y += 62
    draw.rounded_rectangle((82, HEIGHT - 75, 300, HEIGHT - 34), 12, fill=(19, 94, 108))
    draw.text((102, HEIGHT - 68), accent, font=font(20, bold=True), fill=(226, 250, 252))
    draw.text(
        (330, HEIGHT - 67),
        "公开基准与固定机位功能验证；所有数字来自机器可读证据",
        font=font(18),
        fill=(152, 176, 198),
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def repeat_frame(writer: Any, frame: np.ndarray, seconds: float) -> None:
    for _ in range(round(seconds * FPS)):
        writer.write(frame)


def architecture_frame() -> np.ndarray:
    image = Image.new("RGB", (WIDTH, HEIGHT), (242, 249, 250))
    draw = ImageDraw.Draw(image)
    draw.text((55, 38), "一个接口 · 四条能力支路", font=font(43, bold=True), fill=(7, 86, 107))
    boxes = [
        ("Accuracy", "PatchCore\n高精度外观定位", (45, 140, 315, 330)),
        ("Realtime", "EfficientAD-M\n2060 ONNX FP16", (350, 140, 620, 330)),
        ("Video", "组件检测 + FSM\n缺件/重复/换序", (655, 140, 925, 330)),
        ("Feedback", "GuardedAdapt\n门禁/版本/回滚", (960, 140, 1230, 330)),
    ]
    for title, detail, rectangle in boxes:
        draw.rounded_rectangle(rectangle, 22, fill=(226, 244, 247), outline=(23, 143, 157), width=4)
        draw.text(
            (rectangle[0] + 20, rectangle[1] + 20),
            title,
            font=font(28, bold=True),
            fill=(7, 86, 107),
        )
        draw.multiline_text(
            (rectangle[0] + 20, rectangle[1] + 75),
            detail,
            font=font(22),
            fill=(38, 59, 80),
            spacing=10,
        )
    draw.rounded_rectangle((180, 430, 1100, 590), 24, fill=(12, 25, 43))
    draw.text((225, 462), "Unified InferenceEngine", font=font(34, bold=True), fill=(150, 230, 236))
    draw.text(
        (225, 520),
        "score · region · confidence · event interval · version · latency",
        font=font(24),
        fill=(220, 230, 240),
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def image_detection_frame(run_dir: Path) -> np.ndarray:
    with (run_dir / "test_inputs.csv").open(encoding="utf-8", newline="") as stream:
        inputs = list(csv.DictReader(stream))
    with (run_dir / "test_truth.csv").open(encoding="utf-8", newline="") as stream:
        truth = {row["sample_id"]: row for row in csv.DictReader(stream)}
    index = next(i for i, row in enumerate(inputs) if truth[row["sample_id"]]["label"] == "anomaly")
    source = cv2.imread(inputs[index]["path"])
    if source is None:
        raise RuntimeError("cannot load frozen image example")
    maps = np.load(run_dir / "strict_result_v2" / "prediction_maps.npz")["predictions"]
    anomaly = maps[index]
    normalized = cv2.normalize(anomaly, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    source = cv2.resize(source, (470, 470))
    heatmap = cv2.resize(heatmap, (470, 470))
    overlay = cv2.addWeighted(source, 0.55, heatmap, 0.45, 0)
    canvas = np.full((HEIGHT, WIDTH, 3), (250, 247, 242), dtype=np.uint8)
    canvas[145:615, 55:525] = source
    canvas[145:615, 755:1225] = overlay
    cv2.putText(canvas, "INPUT", (55, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (107, 86, 7), 2)
    cv2.putText(
        canvas, "ANOMALY HEATMAP", (755, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (107, 86, 7), 2
    )
    cv2.putText(
        canvas,
        "Accuracy / Realtime mode",
        (500, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (107, 86, 7),
        2,
    )
    cv2.arrowedLine(canvas, (555, 380), (720, 380), (174, 160, 24), 5, tipLength=0.12)
    cv2.putText(
        canvas, "score + region", (550, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (107, 86, 7), 2
    )
    cv2.putText(
        canvas,
        "MVTec AD representative example",
        (410, 680),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (90, 105, 120),
        1,
    )
    return canvas


def hardware_frame() -> np.ndarray:
    image = Image.new("RGB", (WIDTH, HEIGHT), (9, 18, 30))
    draw = ImageDraw.Draw(image)
    draw.text(
        (55, 38), "真实 GTX2060 · 2500×2500 端到端", font=font(42, bold=True), fill=(226, 250, 252)
    )
    draw.rounded_rectangle(
        (55, 120, 1225, 560), 18, fill=(15, 31, 48), outline=(52, 184, 196), width=3
    )
    terminal = [
        "$ nvidia-smi --query-gpu=name,memory.total",
        "NVIDIA GeForce RTX 2060, 6144 MiB",
        "$ benchmark --batch 1 --warmup 100 --repeats 1000 --input 2500 2500",
        "EfficientAD-M ONNX FP16  model-only p95     19.355 ms",
        "EfficientAD-M ONNX FP16  end-to-end p95   166.165 ms   PASS < 200 ms",
        "model size                                 41.5 MB",
    ]
    y = 155
    for line in terminal:
        color = (121, 226, 164) if "PASS" in line else (211, 225, 236)
        draw.text((85, y), line, font=font(23), fill=color)
        y += 58
    draw.text(
        (58, 620),
        "decode + preprocess/transfer + model + postprocess + serialization",
        font=font(22),
        fill=(144, 170, 192),
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def video_panel(frame: np.ndarray, label: str, events: list[dict[str, Any]]) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), (43, 25, 12), dtype=np.uint8)
    resized = cv2.resize(frame, (720, 720))
    canvas[:, :720] = resized
    cv2.putText(canvas, label, (755, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (174, 230, 241), 2)
    cv2.putText(
        canvas,
        "cup -> bottle -> mouse",
        (755, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (220, 220, 220),
        1,
    )
    y = 180
    for event in events[:7]:
        kind = event["public_kind"] or "step_completed"
        text = f"{kind}: {event['step']} @ {event['start_time_s']:.1f}s"
        color = (80, 210, 255) if event["public_kind"] else (130, 230, 160)
        cv2.putText(canvas, text, (755, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1)
        y += 43
    cv2.putText(
        canvas,
        "2x playback · event timeline",
        (755, 665),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (190, 190, 190),
        1,
    )
    return canvas


def play_accelerated(writer: Any, item: dict[str, Any], stride: int = 2) -> None:
    capture = cv2.VideoCapture(item["annotated_video"])
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {item['annotated_video']}")
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            writer.write(video_panel(frame, item["title"], item["events"]))
        index += 1
    capture.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-report", required=True, type=Path)
    parser.add_argument("--image-run", required=True, type=Path)
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
                    "100正常+30缺陷快速适配",
                    "精度模式 + 实时模式 + 视频逻辑 + 安全反馈",
                    "面向新产品、新工况和人工反馈持续演化",
                ],
                accent="SYSTEM",
            ),
            10,
        )
        repeat_frame(
            writer,
            slide(
                "工业痛点",
                [
                    "缺陷样本少：新产品难以快速上线",
                    "图像之外：缺件、重复和工序换序同样关键",
                    "直接在线更新：误标可能破坏历史能力",
                    "目标：GTX2060处理2500×2500输入低于200ms",
                ],
                accent="CHALLENGE",
            ),
            10,
        )
        repeat_frame(writer, architecture_frame(), 15)
        repeat_frame(writer, image_detection_frame(args.image_run), 20)
        repeat_frame(writer, hardware_frame(), 18)
        repeat_frame(
            writer,
            slide(
                "视频装配逻辑",
                [
                    "冻结标准工序：cup → bottle → mouse",
                    "FSM输出缺件、重复、换序及事件时间区间",
                    "19个GT事件匹配18个 · Micro F1 0.9474",
                    "下面以2倍速展示正常、缺件和重复三个代表片段",
                ],
                accent="VIDEO GT",
            ),
            6,
        )
        for item in (report["videos"][0], report["videos"][2], report["videos"][3]):
            play_accelerated(writer, item)
        repeat_frame(
            writer,
            slide(
                "GuardedAdapt · 候选发布协议",
                [
                    "操作员反馈 → 有界候选更新",
                    "反馈收益检查 + 历史锚点回归检查",
                    "安全候选发布；危险候选拒绝并恢复Champion",
                    "所有版本、原因和回滚点可审计",
                ],
                accent="SAFE UPDATE",
            ),
            8,
        )
        repeat_frame(
            writer,
            slide(
                "一次有益反馈被接受",
                [
                    "候选通过反馈收益和历史锚点门禁",
                    "发布新阈值版本，Champion版本号前移",
                    "75-run回放中GuardedAdapt接受率：85.33%",
                    "更新不是默认发生，而是有证据地发生",
                ],
                accent="ACCEPT",
            ),
            10,
        )
        repeat_frame(
            writer,
            slide(
                "一次有害反馈被拒绝并回滚",
                [
                    "候选触发历史能力回退门禁",
                    "线上Champion保持不变，候选被拒绝",
                    "11/11次拒绝更新均精确恢复",
                    "有害更新率：Naive 9.33% → GuardedAdapt 2.67%",
                ],
                accent="ROLLBACK",
            ),
            10,
        )
        repeat_frame(
            writer,
            slide(
                "四个可复核结果",
                [
                    "PatchCore Overall F1：0.9224",
                    "EfficientAD-M Overall F1：0.9036",
                    "真实GTX2060端到端p95：166.2ms",
                    "实拍视频事件F1：0.9474",
                ],
                accent="RESULTS",
            ),
            14,
        )
        repeat_frame(
            writer,
            slide(
                "不只是检测缺陷",
                [
                    "精度模式保证公开基准能力",
                    "实时模式支撑低端GPU部署",
                    "视频FSM覆盖装配逻辑",
                    "GuardedAdapt让人工反馈在安全门内演化",
                ],
                accent="EVOLVE SAFELY",
            ),
            10,
        )
    finally:
        writer.release()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
