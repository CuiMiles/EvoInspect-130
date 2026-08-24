from __future__ import annotations

import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import EvoInspectError
from .provenance import file_sha256, utc_now, write_json


def _unsafe_name(name: str) -> bool:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    return path.is_absolute() or ".." in path.parts or normalized.startswith("/")


def _check_names(names: Iterable[str]) -> list[str]:
    unsafe = sorted(name for name in names if _unsafe_name(name))
    if unsafe:
        raise EvoInspectError(f"archive contains unsafe member paths: {unsafe[:5]}")
    return unsafe


def inspect_archive(
    archive: Path,
    expected_sha256: str,
    dataset_id: str,
    license_id: str,
    output_path: Path,
) -> dict[str, Any]:
    if not archive.is_file():
        raise EvoInspectError(f"archive does not exist: {archive}")
    actual_sha256 = file_sha256(archive)
    if actual_sha256.lower() != expected_sha256.lower():
        raise EvoInspectError(
            f"archive SHA-256 mismatch: expected={expected_sha256}, actual={actual_sha256}"
        )
    member_count = 0
    uncompressed_bytes = 0
    archive_format: str
    if zipfile.is_zipfile(archive):
        archive_format = "zip"
        with zipfile.ZipFile(archive) as stream:
            zip_members = stream.infolist()
            _check_names(member.filename for member in zip_members)
            if any((member.external_attr >> 16) & 0o170000 == 0o120000 for member in zip_members):
                raise EvoInspectError("archive contains symbolic links")
            member_count = len(zip_members)
            uncompressed_bytes = sum(member.file_size for member in zip_members)
    elif tarfile.is_tarfile(archive):
        archive_format = "tar"
        with tarfile.open(archive, mode="r:*") as stream:
            tar_members = stream.getmembers()
            _check_names(member.name for member in tar_members)
            if any(member.issym() or member.islnk() for member in tar_members):
                raise EvoInspectError("archive contains symbolic or hard links")
            member_count = len(tar_members)
            uncompressed_bytes = sum(member.size for member in tar_members if member.isfile())
    else:
        raise EvoInspectError(f"unsupported archive format: {archive}")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "verified_not_extracted",
        "inspected_at": utc_now(),
        "dataset_id": dataset_id,
        "license_id": license_id,
        "archive_path": str(archive.resolve()),
        "archive_format": archive_format,
        "archive_sha256": actual_sha256,
        "archive_bytes": archive.stat().st_size,
        "member_count": member_count,
        "uncompressed_bytes": uncompressed_bytes,
        "warning": (
            "Receipt verifies integrity and archive paths, not dataset identity "
            "or license eligibility."
        ),
    }
    write_json(output_path, receipt)
    return receipt
