from __future__ import annotations

import csv
import random
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .errors import EvoInspectError
from .images import load_grayscale
from .provenance import canonical_hash, file_sha256

REQUIRED_COLUMNS = ("sample_id", "path", "label", "defect_type", "product_id", "source")
VALID_LABELS = {"normal", "anomaly"}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise EvoInspectError(f"manifest has no header: {path}")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise EvoInspectError(f"manifest missing columns {missing}: {path}")
        return [dict(row) for row in reader]


def write_manifest(path: Path, records: Iterable[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in columns})
    temporary.replace(path)


def sample_path(record: dict[str, str], manifest_path: Path) -> Path:
    resolved = record.get("resolved_path", "")
    if resolved:
        candidate = Path(resolved)
        if candidate.exists():
            return candidate
    candidate = Path(record["path"])
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def validate_manifest(input_path: Path, output_path: Path) -> dict[str, Any]:
    records = read_manifest(input_path)
    if not records:
        raise EvoInspectError("manifest is empty")
    seen_ids: set[str] = set()
    hashes: dict[str, str] = {}
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=2):
        sample_id = record["sample_id"].strip()
        if not sample_id or sample_id in seen_ids:
            raise EvoInspectError(f"blank or duplicate sample_id at CSV row {index}: {sample_id!r}")
        seen_ids.add(sample_id)
        label = record["label"].strip().lower()
        if label not in VALID_LABELS:
            raise EvoInspectError(f"invalid label at CSV row {index}: {label!r}")
        defect_type = record["defect_type"].strip()
        if label == "anomaly" and not defect_type:
            raise EvoInspectError(f"anomaly lacks defect_type at CSV row {index}")
        path = sample_path(record, input_path)
        if not path.is_file():
            raise EvoInspectError(f"sample does not exist at CSV row {index}: {path}")
        digest = file_sha256(path)
        if digest in hashes:
            raise EvoInspectError(
                f"content duplicate: {sample_id!r} and {hashes[digest]!r} share {digest}"
            )
        hashes[digest] = sample_id
        width, height, _ = load_grayscale(path)
        enriched: dict[str, Any] = dict(record)
        enriched.update(
            {
                "sample_id": sample_id,
                "label": label,
                "defect_type": defect_type,
                "resolved_path": str(path),
                "content_sha256": digest,
                "width": width,
                "height": height,
            }
        )
        validated.append(enriched)
    validated.sort(key=lambda item: str(item["sample_id"]))
    columns = [
        *REQUIRED_COLUMNS,
        "license_id",
        "resolved_path",
        "content_sha256",
        "width",
        "height",
    ]
    write_manifest(output_path, validated, columns)
    return {
        "samples": len(validated),
        "labels": dict(sorted(Counter(str(row["label"]) for row in validated).items())),
        "defect_types": dict(
            sorted(
                Counter(
                    str(row["defect_type"]) for row in validated if row["label"] == "anomaly"
                ).items()
            )
        ),
        "data_hash": canonical_hash(sorted(row["content_sha256"] for row in validated)),
    }


def _take(items: list[dict[str, str]], count: int, role: str) -> list[dict[str, str]]:
    if len(items) < count:
        raise EvoInspectError(f"role {role} needs {count} samples but only {len(items)} remain")
    selected = items[:count]
    del items[:count]
    for record in selected:
        record["role"] = role
    return selected


def split_manifest(
    input_path: Path, output_path: Path, split_config: dict[str, Any]
) -> dict[str, Any]:
    records = read_manifest(input_path)
    if any(not row.get("content_sha256") for row in records):
        raise EvoInspectError("split input must be produced by data validate")
    if len({row["content_sha256"] for row in records}) != len(records):
        raise EvoInspectError("validated manifest contains repeated content hashes")
    seed = int(split_config["seed"])
    randomizer = random.Random(seed)
    normals = [dict(row) for row in records if row["label"] == "normal"]
    anomalies = [dict(row) for row in records if row["label"] == "anomaly"]
    randomizer.shuffle(normals)
    randomizer.shuffle(anomalies)
    available_types = sorted({row["defect_type"] for row in anomalies})
    unseen_types = {str(value) for value in split_config.get("unseen_defect_types", [])}
    if not unseen_types:
        raise EvoInspectError("unseen_defect_types must be predeclared before splitting")
    if not unseen_types < set(available_types):
        raise EvoInspectError(
            "unseen_defect_types must be present and leave at least one seen type; "
            f"available={available_types}, unseen={sorted(unseen_types)}"
        )
    seen = [row for row in anomalies if row["defect_type"] not in unseen_types]
    unseen = [row for row in anomalies if row["defect_type"] in unseen_types]
    selected: list[dict[str, str]] = []
    selected.extend(_take(normals, int(split_config["normal_support"]), "support_normal"))
    selected.extend(_take(seen, int(split_config["anomaly_support"]), "support_anomaly"))
    selected.extend(_take(normals, int(split_config["development_normal"]), "development"))
    selected.extend(_take(seen, int(split_config["development_anomaly"]), "development"))
    for row in normals + seen + unseen:
        row["role"] = "final_test"
        selected.append(row)
    for row in selected:
        if row["label"] == "normal":
            row["defect_visibility"] = "normal"
        elif row["defect_type"] in unseen_types:
            row["defect_visibility"] = "unseen"
        else:
            row["defect_visibility"] = "seen"
    final_test = [row for row in selected if row["role"] == "final_test"]
    if {row["label"] for row in final_test} != VALID_LABELS:
        raise EvoInspectError("final_test must contain both normal and anomaly samples")
    test_visibility = {
        "unseen" if row["defect_type"] in unseen_types else "seen"
        for row in final_test
        if row["label"] == "anomaly"
    }
    if test_visibility != {"seen", "unseen"}:
        raise EvoInspectError(
            "final_test must retain both seen and unseen anomaly types; "
            f"found={sorted(test_visibility)}"
        )
    selected.sort(key=lambda row: (row["role"], row["sample_id"]))
    columns = [*list(records[0]), "role", "defect_visibility"]
    write_manifest(output_path, selected, columns)
    role_counts = dict(sorted(Counter(row["role"] for row in selected).items()))
    return {
        "seed": seed,
        "roles": role_counts,
        "seen_defect_types": sorted(set(available_types) - unseen_types),
        "unseen_defect_types": sorted(unseen_types),
        "data_hash": canonical_hash(sorted(row["content_sha256"] for row in selected)),
        "split_hash": file_sha256(output_path),
    }


def write_protocol_views(
    combined_manifest: Path,
    adaptation_output: Path,
    test_inputs_output: Path,
    test_truth_output: Path,
) -> dict[str, str]:
    records = read_manifest(combined_manifest)
    columns = list(records[0])
    adaptation = [dict(row) for row in records if row.get("role") != "final_test"]
    final_test = [dict(row) for row in records if row.get("role") == "final_test"]
    if not adaptation or not final_test:
        raise EvoInspectError("combined manifest lacks adaptation or final_test records")
    test_inputs: list[dict[str, str]] = []
    test_truth: list[dict[str, str]] = []
    for row in final_test:
        input_row = dict(row)
        input_row["label"] = ""
        input_row["defect_type"] = ""
        input_row["defect_visibility"] = ""
        test_inputs.append(input_row)
        truth_row = dict(row)
        truth_row["path"] = ""
        truth_row["resolved_path"] = ""
        test_truth.append(truth_row)
    write_manifest(adaptation_output, adaptation, columns)
    write_manifest(test_inputs_output, test_inputs, columns)
    write_manifest(test_truth_output, test_truth, columns)
    return {
        "adaptation_manifest_hash": file_sha256(adaptation_output),
        "test_inputs_manifest_hash": file_sha256(test_inputs_output),
        "test_truth_manifest_hash": file_sha256(test_truth_output),
    }
