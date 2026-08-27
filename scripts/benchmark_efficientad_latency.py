#!/usr/bin/env python3
"""Frozen EfficientAD 2500x2500 model-only and end-to-end latency benchmark."""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import subprocess
import time
from pathlib import Path, PosixPath
from typing import Any

import numpy as np
import yaml
from PIL import Image

from evoinspect.provenance import file_sha256, utc_now, write_json
from scripts.efficientad_rcbr_100_30 import infer_array


def percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def source_png(inputs: Path, resolution: int) -> tuple[bytes, str]:
    with inputs.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    with Image.open(row["path"]) as image:
        resized = image.convert("RGB").resize((resolution, resolution), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue(), row["sample_id"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--test-inputs", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--resolution", type=int, default=2500)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()

    import torch
    from anomalib.models import EfficientAd

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    threshold = float(metrics["calibration"]["threshold_development_only"]["threshold"])
    input_shape = tuple(int(value) for value in config["input_resolution"])
    torch.serialization.add_safe_globals([PosixPath])
    model = EfficientAd.load_from_checkpoint(args.checkpoint, map_location="cuda")
    model.cuda().eval()
    encoded, sample_id = source_png(args.test_inputs, args.resolution)
    sections: dict[str, list[float]] = {
        name: []
        for name in (
            "decode",
            "preprocess_and_transfer",
            "model_only",
            "postprocess",
            "serialization",
            "end_to_end",
        )
    }

    def iteration(record: bool) -> None:
        total = time.perf_counter()
        started = time.perf_counter()
        with Image.open(io.BytesIO(encoded)) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        decode = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        anomaly_map, model_ms = infer_array(model, array, input_shape)
        inference_wall = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        score = float(np.quantile(anomaly_map, float(config["inference"]["score_quantile"])))
        decision = score >= threshold
        postprocess = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        json.dumps({"sample_id": sample_id, "score": score, "decision": decision})
        serialization = (time.perf_counter() - started) * 1000
        end_to_end = (time.perf_counter() - total) * 1000
        if record:
            sections["decode"].append(decode)
            sections["preprocess_and_transfer"].append(max(0.0, inference_wall - model_ms))
            sections["model_only"].append(model_ms)
            sections["postprocess"].append(postprocess)
            sections["serialization"].append(serialization)
            sections["end_to_end"].append(end_to_end)

    for _ in range(args.warmup):
        iteration(False)
    for index in range(args.repeats):
        iteration(True)
        if (index + 1) % 100 == 0:
            print(f"benchmark {index + 1}/{args.repeats}", flush=True)
    output = {
        "schema_version": 1,
        "status": "completed_frozen_efficientad_2500_latency",
        "created_at": utc_now(),
        "model_id": config["model_id"],
        "model_size": config["model_size"],
        "sample_id": sample_id,
        "input_resolution": [args.resolution, args.resolution],
        "model_input_resolution": list(input_shape),
        "batch_size": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "precision": config["training"]["precision"],
        "latency_ms": {name: percentiles(values) for name, values in sections.items()},
        "throughput_images_per_second": 1000.0 / float(np.mean(sections["end_to_end"])),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "physical_gpu": args.physical_gpu,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "metrics_sha256": file_sha256(args.metrics),
        "test_inputs_sha256": file_sha256(args.test_inputs),
        "warning": (
            "Source image resized to 2500x2500 before measurement; accuracy at native 2500 "
            "resolution is not implied. Hardware claims are limited to the recorded device."
        ),
    }
    write_json(args.output, output)
    query = subprocess.run(
        ["nvidia-smi", "-i", args.physical_gpu, "-q"],
        check=False,
        capture_output=True,
        text=True,
    )
    args.output.with_name("nvidia-smi-q.txt").write_text(query.stdout, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
