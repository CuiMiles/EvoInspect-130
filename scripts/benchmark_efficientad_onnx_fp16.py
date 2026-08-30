#!/usr/bin/env python3
"""One-shot ONNX FP16 EfficientAD latency diagnostic on frozen checkpoints."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path, PosixPath
from typing import Any

import numpy as np
import yaml

from evoinspect.provenance import file_sha256, utc_now, write_json
from scripts.benchmark_efficientad_latency import percentiles, source_png


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--onnx-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--resolution", type=int, default=2500)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()

    import cv2
    import onnx
    import onnxruntime as ort
    import torch
    from anomalib.models import EfficientAd
    from onnxconverter_common.float16 import convert_float_to_float16

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    threshold = float(metrics["calibration"]["threshold_development_only"]["threshold"])
    input_shape = tuple(int(value) for value in config["input_resolution"])
    torch.serialization.add_safe_globals([PosixPath])
    lightning_model = EfficientAd.load_from_checkpoint(args.checkpoint, map_location="cpu")
    core_model = lightning_model.model.eval()

    class AnomalyMapWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return self.model(batch).anomaly_map

    wrapper = AnomalyMapWrapper(core_model).eval()
    sample = torch.zeros((1, 3, *input_shape), dtype=torch.float32)
    args.onnx_output.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = args.onnx_output.with_suffix(".fp32.onnx")
    build_started = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            sample,
            fp32_path,
            input_names=["input"],
            output_names=["anomaly_map"],
            opset_version=17,
            do_constant_folding=True,
        )
    converted = convert_float_to_float16(onnx.load(fp32_path), keep_io_types=True)
    onnx.save(converted, args.onnx_output)
    fp32_path.unlink()
    build_seconds = time.perf_counter() - build_started

    session = ort.InferenceSession(
        str(args.onnx_output),
        providers=[
            ("CUDAExecutionProvider", {"device_id": int(args.physical_gpu)}),
            "CPUExecutionProvider",
        ],
    )
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"CUDAExecutionProvider unavailable: {session.get_providers()}")
    encoded, sample_id, source_path = source_png(args.resolution, image_path=args.image)

    def decode_and_preprocess() -> tuple[np.ndarray, np.ndarray]:
        bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("OpenCV failed to decode benchmark input")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return rgb, tensor

    _, fidelity_input = decode_and_preprocess()
    core_model.cuda().eval()
    with torch.inference_mode():
        reference = (
            core_model(torch.from_numpy(fidelity_input).cuda())
            .anomaly_map.float()
            .cpu()
            .numpy()
        )
    candidate = session.run(["anomaly_map"], {"input": fidelity_input})[0].astype(np.float32)
    difference = np.abs(reference - candidate)
    reference_score = float(np.quantile(reference, float(config["inference"]["score_quantile"])))
    candidate_score = float(np.quantile(candidate, float(config["inference"]["score_quantile"])))
    fidelity = {
        "max_abs_anomaly_map_difference": float(difference.max()),
        "mean_abs_anomaly_map_difference": float(difference.mean()),
        "reference_score": reference_score,
        "candidate_score": candidate_score,
        "score_abs_difference": abs(reference_score - candidate_score),
        "fixed_threshold": threshold,
        "decision_match": (reference_score >= threshold) == (candidate_score >= threshold),
    }
    del core_model, lightning_model, wrapper, reference, candidate
    torch.cuda.empty_cache()

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
        bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("OpenCV failed to decode benchmark input")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        decode = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        resized = cv2.resize(rgb, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        preprocess = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        anomaly_map = session.run(["anomaly_map"], {"input": tensor})[0]
        model_ms = (time.perf_counter() - started) * 1000
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
            sections["preprocess_and_transfer"].append(preprocess)
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
        "status": "completed_frozen_efficientad_onnx_fp16_2500_latency",
        "created_at": utc_now(),
        "model_id": config["model_id"],
        "model_size": config["model_size"],
        "sample_id": sample_id,
        "input_resolution": [args.resolution, args.resolution],
        "model_input_resolution": list(input_shape),
        "batch_size": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "precision": "ONNX internal FP16 with FP32 input/output",
        "preprocessing": "OpenCV imdecode and CPU bilinear resize before transfer",
        "latency_ms": {name: percentiles(values) for name, values in sections.items()},
        "throughput_images_per_second": 1000.0 / float(np.mean(sections["end_to_end"])),
        "fidelity": fidelity,
        "onnx_build_seconds": build_seconds,
        "onnx_sha256": file_sha256(args.onnx_output),
        "onnx_bytes": args.onnx_output.stat().st_size,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "physical_gpu": args.physical_gpu,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "providers": session.get_providers(),
        },
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "metrics_sha256": file_sha256(args.metrics),
        "benchmark_source": str(source_path),
        "benchmark_source_sha256": file_sha256(source_path),
        "warning": (
            "Failed-quality-gate model hardware diagnostic only. Source image is resized to "
            "2500x2500; native-resolution accuracy and deployment quality are not implied."
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
