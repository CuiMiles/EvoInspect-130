from __future__ import annotations

import argparse
import json
import platform
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .archive import inspect_archive
from .baseline import adapt_fixture_baseline, infer_manifest, load_model
from .data import read_manifest, split_manifest, validate_manifest, write_protocol_views
from .errors import EvoInspectError
from .evaluation import evaluate_predictions
from .provenance import (
    append_csv,
    canonical_hash,
    file_sha256,
    git_state,
    read_config,
    utc_now,
    write_json,
)
from .reporting import generate_report

REGISTRY_COLUMNS = [
    "run_id",
    "status",
    "start_time",
    "end_time",
    "git_commit",
    "dirty",
    "config_hash",
    "data_hash",
    "split_hash",
    "seed",
    "hardware",
    "model",
    "protocol",
    "metrics_path",
    "artifact_path",
    "failure_reason",
    "notes",
]


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _emit(event: str, run_id: str, **values: Any) -> None:
    payload = {"event": event, "run_id": run_id, "time": utc_now(), **values}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evoinspect")
    groups = parser.add_subparsers(dest="group", required=True)

    data = groups.add_parser("data")
    data_commands = data.add_subparsers(dest="command", required=True)
    validate = data_commands.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=_path)
    validate.add_argument("--output", required=True, type=_path)
    validate.add_argument("--summary", required=True, type=_path)
    split = data_commands.add_parser("split")
    split.add_argument("--manifest", required=True, type=_path)
    split.add_argument("--config", required=True, type=_path)
    split.add_argument("--output", required=True, type=_path)
    split.add_argument("--adaptation-output", required=True, type=_path)
    split.add_argument("--test-inputs-output", required=True, type=_path)
    split.add_argument("--test-truth-output", required=True, type=_path)
    split.add_argument("--summary", required=True, type=_path)
    inspect = data_commands.add_parser("inspect-archive")
    inspect.add_argument("--archive", required=True, type=_path)
    inspect.add_argument("--expected-sha256", required=True)
    inspect.add_argument("--dataset-id", required=True)
    inspect.add_argument("--license-id", required=True)
    inspect.add_argument("--output", required=True, type=_path)

    adapt = groups.add_parser("adapt")
    adapt_commands = adapt.add_subparsers(dest="command", required=True)
    product = adapt_commands.add_parser("product")
    product.add_argument("--manifest", required=True, type=_path)
    product.add_argument("--config", required=True, type=_path)
    product.add_argument("--output", required=True, type=_path)
    product.add_argument("--summary", required=True, type=_path)

    infer = groups.add_parser("infer")
    infer_commands = infer.add_subparsers(dest="command", required=True)
    image = infer_commands.add_parser("image")
    image.add_argument("--manifest", required=True, type=_path)
    image.add_argument("--model", required=True, type=_path)
    image.add_argument("--output", required=True, type=_path)
    image.add_argument("--summary", required=True, type=_path)
    image.add_argument("--roles", default="final_test")

    evaluate = groups.add_parser("evaluate")
    evaluate.add_argument("--manifest", required=True, type=_path)
    evaluate.add_argument("--predictions", required=True, type=_path)
    evaluate.add_argument("--model", required=True, type=_path)
    evaluate.add_argument("--config", required=True, type=_path)
    evaluate.add_argument("--output", required=True, type=_path)
    evaluate.add_argument("--registry", required=True, type=_path)
    evaluate.add_argument("--run-id")

    report = groups.add_parser("report")
    report_commands = report.add_subparsers(dest="command", required=True)
    generate = report_commands.add_parser("generate")
    generate.add_argument("--metrics", required=True, type=_path)
    generate.add_argument("--model", required=True, type=_path)
    generate.add_argument("--output", required=True, type=_path)
    return parser


def _data_hash(manifest: Path) -> str:
    rows = read_manifest(manifest)
    return canonical_hash(sorted(row["content_sha256"] for row in rows))


def run(args: argparse.Namespace, run_id: str) -> None:
    if args.group == "data" and args.command == "validate":
        summary = validate_manifest(args.manifest, args.output)
        write_json(args.summary, summary)
        _emit("data_validated", run_id, **summary)
        return
    if args.group == "data" and args.command == "split":
        config = read_config(args.config)
        summary = split_manifest(args.manifest, args.output, config["split"])
        summary.update(
            write_protocol_views(
                args.output,
                args.adaptation_output,
                args.test_inputs_output,
                args.test_truth_output,
            )
        )
        summary["config_hash"] = canonical_hash(config)
        write_json(args.summary, summary)
        _emit("data_split", run_id, **summary)
        return
    if args.group == "data" and args.command == "inspect-archive":
        receipt = inspect_archive(
            args.archive,
            args.expected_sha256,
            args.dataset_id,
            args.license_id,
            args.output,
        )
        _emit("archive_inspected", run_id, **receipt)
        return
    if args.group == "adapt" and args.command == "product":
        config = read_config(args.config)
        model = adapt_fixture_baseline(args.manifest, config, args.output)
        summary = {
            "model_id": model["model_id"],
            "model_hash": model["model_hash"],
            "split_hash": model["split_hash"],
            "config_hash": model["config_hash"],
            "calibration": model["calibration"],
            "status": model["status"],
        }
        write_json(args.summary, summary)
        _emit("product_adapted", run_id, **summary)
        return
    if args.group == "infer" and args.command == "image":
        roles = {value.strip() for value in args.roles.split(",") if value.strip()}
        summary = infer_manifest(args.manifest, args.model, args.output, roles)
        write_json(args.summary, summary)
        _emit("images_inferred", run_id, **summary)
        return
    if args.group == "evaluate":
        started = utc_now()
        model = load_model(args.model)
        metrics = evaluate_predictions(
            args.manifest, args.predictions, args.output, str(model["model_hash"])
        )
        config = read_config(args.config)
        commit, dirty = git_state(Path.cwd())
        experiment_id = args.run_id or run_id
        append_csv(
            args.registry,
            REGISTRY_COLUMNS,
            {
                "run_id": experiment_id,
                "status": "completed_fixture_only",
                "start_time": started,
                "end_time": utc_now(),
                "git_commit": commit,
                "dirty": str(dirty).lower(),
                "config_hash": canonical_hash(config),
                "data_hash": _data_hash(args.manifest),
                "split_hash": file_sha256(args.manifest),
                "seed": config["split"]["seed"],
                "hardware": (
                    f"{platform.platform()}; Python {platform.python_version()}; CPU fixture"
                ),
                "model": model["model_id"],
                "protocol": "fixture_vertical_slice",
                "metrics_path": str(args.output),
                "artifact_path": str(args.model),
                "failure_reason": "",
                "notes": (
                    "Synthetic engineering fixture; forbidden as research/competition evidence"
                ),
            },
        )
        _emit("evaluation_completed", experiment_id, metrics=metrics["overall"])
        return
    if args.group == "report" and args.command == "generate":
        generate_report(args.metrics, args.model, args.output)
        _emit("report_generated", run_id, output=str(args.output))
        return
    raise EvoInspectError("unsupported command")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_id = f"cli-{uuid.uuid4().hex[:12]}"
    try:
        run(args, run_id)
    except (EvoInspectError, KeyError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "event": "command_failed",
                    "run_id": run_id,
                    "time": utc_now(),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
