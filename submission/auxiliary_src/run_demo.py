#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = yaml.safe_load((root / "configs/final_realtime.yaml").read_text(encoding="utf-8"))
    model_path = root / config["model_file"]
    providers = [name for name in config["provider_order"] if name in ort.get_available_providers()]
    if not providers:
        raise RuntimeError(f"no requested ONNX provider available: {ort.get_available_providers()}")
    session = ort.InferenceSession(str(model_path), providers=providers)
    encoded = np.fromfile(args.image, dtype=np.uint8)
    started = time.perf_counter()
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot decode {args.image}")
    decode_ms = (time.perf_counter() - started) * 1000
    original_shape = [int(bgr.shape[1]), int(bgr.shape[0])]
    started = time.perf_counter()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = [int(value) for value in config["model_input_resolution"]]
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    preprocess_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    anomaly_map = session.run(["anomaly_map"], {"input": tensor})[0].astype(np.float32)
    model_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    score = float(np.quantile(anomaly_map, float(config["score_quantile"])))
    decision = "anomaly" if score >= float(config["threshold"]) else "normal"
    normalized = cv2.normalize(anomaly_map.squeeze(), None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    postprocess_ms = (time.perf_counter() - started) * 1000
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "heatmap.png"), heatmap)
    payload = {
        "score": score,
        "threshold": float(config["threshold"]),
        "decision": decision,
        "input_resolution": original_shape,
        "model_input_resolution": [width, height],
        "provider": session.get_providers(),
        "latency_ms": {
            "decode": decode_ms,
            "preprocess": preprocess_ms,
            "model": model_ms,
            "postprocess": postprocess_ms,
            "end_to_end": decode_ms + preprocess_ms + model_ms + postprocess_ms,
        },
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "warning": config["claim_boundary"],
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
