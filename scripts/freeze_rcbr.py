from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evoinspect.provenance import file_sha256, utc_now, write_json

TRACKED = (
    "configs/baselines/efficientad_s_100_30.yaml",
    "configs/innovations/rcbr_v1_dev.yaml",
    "src/evoinspect/localization.py",
    "src/evoinspect/rcbr.py",
    "scripts/efficientad_rcbr_100_30.py",
    "scripts/aggregate_rcbr.py",
    "scripts/benchmark_rcbr_latency.py",
    "scripts/freeze_rcbr.py",
    "scripts/run_rcbr_experiment_suite.sh",
)


def snapshot(root: Path) -> dict[str, str]:
    return {relative: file_sha256(root / relative) for relative in TRACKED}


def main() -> int:
    args = parser().parse_args()
    if args.command == "create":
        gate = json.loads(args.gate.read_text(encoding="utf-8"))
        if not gate.get("gate", {}).get("passed"):
            raise RuntimeError("cannot freeze a method that did not pass the full development gate")
        write_json(
            args.manifest,
            {
                "schema_version": 1,
                "created_at": utc_now(),
                "gate_path": str(args.gate),
                "gate_sha256": file_sha256(args.gate),
                "tracked_sha256": snapshot(args.root),
                "confirmation_seeds": [138, 139, 140, 141, 142],
                "warning": "Confirmation is one-shot. Any tracked change invalidates this freeze.",
            },
        )
        return 0
    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    current = snapshot(args.root)
    if current != manifest.get("tracked_sha256"):
        changed = sorted(
            key
            for key in set(current) | set(manifest.get("tracked_sha256", {}))
            if current.get(key) != manifest.get("tracked_sha256", {}).get(key)
        )
        raise RuntimeError(f"frozen RCBR files changed: {changed}")
    if file_sha256(Path(manifest["gate_path"])) != manifest["gate_sha256"]:
        raise RuntimeError("frozen development gate file changed")
    print("RCBR freeze verified")
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("command", choices=("create", "verify"))
    value.add_argument("--root", required=True, type=Path)
    value.add_argument("--manifest", required=True, type=Path)
    value.add_argument("--gate", type=Path)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
