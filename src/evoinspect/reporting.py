from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_report(metrics_path: Path, model_path: Path, output_path: Path) -> None:
    metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    model: dict[str, Any] = json.loads(model_path.read_text(encoding="utf-8"))
    overall = metrics["overall"]
    lines = [
        "# Fixture vertical-slice report",
        "",
        "> Engineering smoke evidence only. These synthetic metrics are forbidden in scientific,",
        "> competition, abstract, slide, or deployment-performance claims.",
        "",
        f"- Model: `{model['model_id']}` (`{model['model_hash']}`)",
        f"- Protocol: `{metrics['protocol']}`",
        f"- Split hash: `{metrics['split_hash']}`",
        f"- Samples: {int(overall['samples'])}",
        f"- Accuracy: {overall['accuracy']:.6f}",
        f"- Fixed-threshold F1: {overall['f1_fixed_threshold']:.6f}",
        f"- AUROC: {overall['auroc']:.6f}",
        f"- Average precision: {overall['average_precision']:.6f}",
        "",
        "The threshold was selected using only the development role. Adaptation had no final-test",
        "rows; inference had no test labels; evaluation joined sealed truth only after prediction.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
