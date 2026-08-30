#!/usr/bin/env python3
"""Decode real videos, detect desktop components, and emit FSM event intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import yaml

from evoinspect.sequence import AssemblySequenceFSM, FrameObservation, SequenceEvent, SequenceRule
from evoinspect.video import DesktopComponentDetector, VideoDetectorConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_kind(event: SequenceEvent) -> str | None:
    if event.kind == "missing_step":
        return "missing" if event.reason.startswith("stream ended") else "skip"
    return {
        "reordered_step": "reorder",
        "repeated_step": "repeat",
        "unexpected_step": "unknown",
    }.get(event.kind)


def load_config(path: Path) -> tuple[VideoDetectorConfig, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise RuntimeError(f"invalid video config: {path}")
    simple = raw["simple_detector"]
    config = VideoDetectorConfig(
        expected_steps=tuple(str(value) for value in raw["expected_steps"]),
        marker_to_component=tuple(
            (int(marker), str(component))
            for marker, component in raw["marker_to_component"].items()
        ),
        processing_width=int(raw["processing_width"]),
        sample_every_frames=int(raw["sample_every_frames"]),
        stable_samples=int(raw["stable_samples"]),
        blue_pixel_min=int(simple["blue_pixel_min"]),
        cup_without_bottle_min=int(simple["cup_without_bottle_min"]),
        cup_with_bottle_min=int(simple["cup_with_bottle_min"]),
        mouse_bright_pixel_min=int(simple["mouse_bright_pixel_min"]),
    )
    return config, raw


def annotate(
    frame: Any, present: tuple[str, ...], events: list[dict[str, Any]], *, title: str
) -> Any:
    output = frame.copy()
    lines = [title, f"Present: {', '.join(present) if present else 'none'}"]
    lines.extend(
        f"{event['public_kind'] or event['kind']}: {event['step']} @ {event['start_time_s']:.2f}s"
        for event in events[-4:]
    )
    for index, line in enumerate(lines):
        y = 38 + 34 * index
        cv2.putText(output, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
        cv2.putText(output, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(
        output,
        "FUNCTIONAL DEMO - NOT AN INDUSTRIAL BENCHMARK",
        (18, output.shape[0] - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 0, 255),
        2,
    )
    return output


def evaluate_video(
    source: Path,
    destination: Path,
    config: VideoDetectorConfig,
    *,
    write_video: bool,
    clip_id: str,
    title: str,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video metadata: {source}")
    output_height = round(height * config.processing_width / width)
    writer = None
    if write_video:
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (config.processing_width, output_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV could not create {destination}")

    detector = DesktopComponentDetector(config)
    fsm = AssemblySequenceFSM(SequenceRule(config.expected_steps, max_gap_s=60.0))
    events: list[dict[str, Any]] = []
    mode_counts = {"aruco": 0, "simple": 0}
    last_present: tuple[str, ...] = ()
    decoded = 0
    sampled = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = decoded
            decoded += 1
            timestamp = frame_index / fps
            if frame_index % config.sample_every_frames == 0:
                observation = detector.observe(frame, frame_index, timestamp)
                sampled += 1
                mode_counts[observation.mode] += 1
                last_present = observation.present
                new_events = fsm.process(
                    FrameObservation(
                        frame_index,
                        timestamp,
                        step=observation.component,
                        component=observation.component,
                    )
                )
                for event in new_events:
                    item = event.to_dict()
                    item["public_kind"] = public_kind(event)
                    events.append(item)
            if writer is not None:
                resized = cv2.resize(frame, (config.processing_width, output_height))
                writer.write(annotate(resized, last_present, events, title=title))
        for event in fsm.finalize():
            item = event.to_dict()
            item["public_kind"] = public_kind(event)
            events.append(item)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    elapsed = time.perf_counter() - started
    return {
        "clip_id": clip_id,
        "title": title,
        "source": str(source),
        "source_sha256": sha256(source),
        "source_size_bytes": source.stat().st_size,
        "decoder": "OpenCV VideoCapture",
        "codec_output": "mp4v" if write_video else None,
        "width": width,
        "height": height,
        "fps": fps,
        "declared_frames": declared_frames,
        "decoded_frames": decoded,
        "sampled_frames": sampled,
        "duration_seconds": decoded / fps,
        "processing_seconds": elapsed,
        "processing_fps": decoded / elapsed if elapsed else 0.0,
        "detector_mode_counts": mode_counts,
        "expected_steps": list(config.expected_steps),
        "events": events,
        "public_anomalies": [event for event in events if event["public_kind"] is not None],
        "annotated_video": str(destination) if write_video else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/video/video_5"))
    parser.add_argument("--config", type=Path, default=Path("configs/video/desktop_assembly.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    config, raw_config = load_config(args.config)
    sources = sorted(args.input_dir.glob("*.mp4"))
    if not sources:
        raise RuntimeError(f"no mp4 videos found under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for source in sources:
        clip = raw_config.get("clips", {}).get(source.name)
        if not isinstance(clip, dict):
            raise RuntimeError(f"missing clip metadata for {source.name}")
        destination = args.output_dir / "annotated" / str(clip["output_name"])
        results.append(
            evaluate_video(
                source,
                destination,
                config,
                write_video=not args.no_video,
                clip_id=str(clip["clip_id"]),
                title=str(clip["title"]),
            )
        )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "scope": raw_config["scope"],
        "benchmark_claim_allowed": bool(raw_config["benchmark_claim_allowed"]),
        "config": asdict(config),
        "config_path": str(args.config),
        "videos": results,
        "summary": {
            "videos": len(results),
            "decoded_frames": sum(int(result["decoded_frames"]) for result in results),
            "duration_seconds": sum(float(result["duration_seconds"]) for result in results),
            "public_anomaly_events": sum(len(result["public_anomalies"]) for result in results),
        },
        "limitations": [
            "fixed-camera desktop component detector",
            "no industrial accuracy claim",
            "no claim of generalization beyond these supplied recordings",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
