from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evoinspect.provenance import file_sha256, utc_now, write_json
from evoinspect.rcbr import (
    NormalRiskCalibrator,
    RouterLimits,
    UtilityModel,
    attach_costs_and_utility,
    fuse_refinements,
    generate_candidates,
    high_frequency_map,
    multiscale_disagreement,
    select_under_budget,
)
from scripts.efficientad_rcbr_100_30 import infer_array, resize_map


def percentile(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def load_first_image(test_inputs: Path, target_size: int) -> tuple[bytes, str]:
    with test_inputs.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    with Image.open(row["path"]) as source:
        image = source.convert("RGB").resize((target_size, target_size), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue(), row["sample_id"]


def benchmark(args: argparse.Namespace) -> None:
    import torch
    import yaml  # type: ignore[import-untyped]
    from anomalib.models import EfficientAd

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    baseline: dict[str, Any] = yaml.safe_load(args.baseline_config.read_text(encoding="utf-8"))
    input_shape = tuple(int(value) for value in baseline["input_resolution"])
    model = EfficientAd.load_from_checkpoint(args.checkpoint, map_location="cuda")
    model.cuda().eval()
    with np.load(args.router_state) as state:
        normal_maps = np.asarray(state["normal_maps"], dtype=np.float32)
        utility = UtilityModel(
            np.asarray(state["utility_feature_mean"], dtype=np.float64),
            np.asarray(state["utility_feature_scale"], dtype=np.float64),
            np.asarray(state["utility_coefficients"], dtype=np.float64),
            float(state["utility_intercept"][0]),
        )
        roi_cost_ms = float(state["roi_cost_ms"][0])
    calibrator = NormalRiskCalibrator.fit(normal_maps)
    encoded, sample_id = load_first_image(args.test_inputs, args.resolution)
    router = config["router"]
    limits = RouterLimits(
        latency_budget_ms=float(router["latency_budget_ms"]),
        max_rois=int(router["max_rois"]),
        max_total_area_fraction=float(router["max_single_image_roi_area_fraction"]),
        nms_iou=float(router["nms_iou"]),
    )
    sections: dict[str, list[float]] = {
        key: []
        for key in (
            "decode",
            "preprocess_and_transfer",
            "global_model",
            "routing",
            "local_preprocess_and_transfer",
            "local_models",
            "postprocess",
            "serialization",
            "end_to_end",
        )
    }
    roi_counts: list[int] = []
    roi_areas: list[float] = []

    def iteration(record: bool) -> None:
        total_started = time.perf_counter()
        started = time.perf_counter()
        with Image.open(io.BytesIO(encoded)) as decoded:
            image = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
        decode_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        global_map, global_model_ms = infer_array(model, image, input_shape)
        global_call_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        calibrated = calibrator.transform(global_map)
        image_small = np.asarray(
            Image.fromarray(image).resize((input_shape[1], input_shape[0])), dtype=np.float32
        )
        candidates = generate_candidates(
            calibrated,
            multiscale_disagreement(global_map),
            high_frequency_map(image_small),
            window_fractions=tuple(float(value) for value in router["candidate_window_fractions"]),
            stride_fraction=float(router["candidate_stride_fraction"]),
            per_scale=int(router["candidates_per_scale"]),
        )
        latency_table = {candidate.area: roi_cost_ms for candidate in candidates}
        candidates = attach_costs_and_utility(
            candidates,
            utility,
            latency_table,
            false_positive_penalty=float(router["utility_false_positive_penalty"]),
        )
        selected = select_under_budget(candidates, input_shape, limits)
        routing_ms = (time.perf_counter() - started) * 1000
        local_model_ms = 0.0
        local_wall_ms = 0.0
        refinements = []
        height, width = image.shape[:2]
        for roi in selected:
            y0, y1 = (
                round(roi.y0 * height / input_shape[0]),
                round(roi.y1 * height / input_shape[0]),
            )
            x0, x1 = round(roi.x0 * width / input_shape[1]), round(roi.x1 * width / input_shape[1])
            started = time.perf_counter()
            local, model_ms = infer_array(model, image[y0:y1, x0:x1], input_shape)
            local_wall_ms += (time.perf_counter() - started) * 1000
            local_model_ms += model_ms
            refinements.append(
                (
                    roi,
                    # Keep local and global maps in the same raw EfficientAD score space.
                    # Risk calibration is used only for routing, matching the revised RCBR
                    # fusion implemented by the training/evaluation path.
                    resize_map(local, (roi.height, roi.width)),
                    roi.predicted_benefit,
                )
            )
        started = time.perf_counter()
        fused, audit = fuse_refinements(
            calibrated,
            refinements,
            minimum_evidence=float(router["minimum_refinement_evidence"]),
        )
        score = float(np.quantile(fused, float(baseline["inference"]["score_quantile"])))
        postprocess_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        json.dumps(
            {"sample_id": sample_id, "score": score, "rois": audit},
            ensure_ascii=False,
            sort_keys=True,
        )
        serialization_ms = (time.perf_counter() - started) * 1000
        end_to_end_ms = (time.perf_counter() - total_started) * 1000
        if record:
            sections["decode"].append(decode_ms)
            sections["preprocess_and_transfer"].append(max(0.0, global_call_ms - global_model_ms))
            sections["global_model"].append(global_model_ms)
            sections["routing"].append(routing_ms)
            sections["local_preprocess_and_transfer"].append(
                max(0.0, local_wall_ms - local_model_ms)
            )
            sections["local_models"].append(local_model_ms)
            sections["postprocess"].append(postprocess_ms)
            sections["serialization"].append(serialization_ms)
            sections["end_to_end"].append(end_to_end_ms)
            roi_counts.append(len(selected))
            union = np.zeros(input_shape, dtype=np.bool_)
            for roi in selected:
                union[roi.y0 : roi.y1, roi.x0 : roi.x1] = True
            roi_areas.append(float(union.mean()))

    for _ in range(args.warmup):
        iteration(False)
    for index in range(args.repeats):
        iteration(True)
        if (index + 1) % 100 == 0:
            print(f"benchmark {index + 1}/{args.repeats}", flush=True)
    output = {
        "schema_version": 1,
        "status": "completed_synthetic_resolution_latency_benchmark",
        "created_at": utc_now(),
        "sample_id": sample_id,
        "input_resolution": [args.resolution, args.resolution],
        "batch_size": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "precision": baseline["training"]["precision"],
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "physical_gpu": args.physical_gpu,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "latency_ms": {key: percentile(values) for key, values in sections.items()},
        "throughput_images_per_second_from_mean_e2e": 1000.0 / np.mean(sections["end_to_end"]),
        "roi_count": percentile([float(value) for value in roi_counts]),
        "roi_area_fraction": percentile(roi_areas),
        "model_sha256": file_sha256(args.checkpoint),
        "router_state_sha256": file_sha256(args.router_state),
        "warning": (
            "The source image was resized to 2500x2500 before benchmarking. This is an RTX 3090 "
            "synthetic-resolution end-to-end measurement, not native high-resolution accuracy or "
            "RTX 2060 evidence."
        ),
    }
    write_json(args.output, output)
    query = subprocess.run(
        ["nvidia-smi", "-i", args.physical_gpu, "-q"], check=False, capture_output=True, text=True
    )
    args.output.with_name("nvidia-smi-q.txt").write_text(query.stdout, encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--checkpoint", required=True, type=Path)
    value.add_argument("--router-state", required=True, type=Path)
    value.add_argument("--test-inputs", required=True, type=Path)
    value.add_argument("--config", required=True, type=Path)
    value.add_argument("--baseline-config", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--physical-gpu", required=True)
    value.add_argument("--resolution", type=int, default=2500)
    value.add_argument("--warmup", type=int, default=100)
    value.add_argument("--repeats", type=int, default=1000)
    return value


if __name__ == "__main__":
    benchmark(parser().parse_args())
