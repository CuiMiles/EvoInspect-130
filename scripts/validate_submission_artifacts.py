#!/usr/bin/env python3
"""Machine-readable validation for the four preliminary-submission artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import cv2

LIMIT = 209_715_200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf(path: Path) -> dict[str, object]:
    info = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE).group(1))  # type: ignore[union-attr]
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "pages": pages,
        "valid_pdf": path.read_bytes()[:5] == b"%PDF-",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-dir", type=Path, default=Path("submission/drafts"))
    parser.add_argument("--intro", type=Path, default=Path("submission/works_intro.txt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    intro_text = args.intro.read_text(encoding="utf-8").strip()
    chinese = len(re.findall(r"[\u3400-\u9fff]", intro_text))
    non_whitespace = len(re.sub(r"\s", "", intro_text))
    intro = pdf(args.draft_dir / "works_intro.pdf")
    intro.update(
        {
            "source_path": str(args.intro),
            "chinese_characters": chinese,
            "non_whitespace_characters": non_whitespace,
            "within_300_chinese_characters": chinese <= 300,
            "within_300_non_whitespace_characters": non_whitespace <= 300,
        }
    )
    document = pdf(args.draft_dir / "project_document.pdf")
    video_path = args.draft_dir / "project_video.mp4"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("project video cannot be opened")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    video = {
        "path": str(video_path),
        "size_bytes": video_path.stat().st_size,
        "sha256": sha256(video_path),
        "fps": fps,
        "frames": frames,
        "duration_seconds": frames / fps,
        "width": width,
        "height": height,
        "within_300_seconds": frames / fps <= 300,
        "within_200_mib": video_path.stat().st_size <= LIMIT,
    }
    zip_path = args.draft_dir / "auxiliary_material.zip"
    with zipfile.ZipFile(zip_path) as archive:
        corrupt = archive.testzip()
        members = len(archive.infolist())
    auxiliary = {
        "path": str(zip_path),
        "size_bytes": zip_path.stat().st_size,
        "sha256": sha256(zip_path),
        "members": members,
        "corrupt_member": corrupt,
        "valid_zip": corrupt is None,
        "within_200_mib": zip_path.stat().st_size <= LIMIT,
    }
    constraints_passed = bool(
        intro["valid_pdf"]
        and intro["within_300_chinese_characters"]
        and intro["within_300_non_whitespace_characters"]
        and document["valid_pdf"]
        and video["within_300_seconds"]
        and video["within_200_mib"]
        and auxiliary["valid_zip"]
        and auxiliary["within_200_mib"]
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "constraints_passed": constraints_passed,
        "final_upload_ready": False,
        "final_upload_blockers": [
            "team name, group, author/member, school and division of work are not provided",
            "official filename placeholders have not been replaced",
            "EfficientAD-M gate and actual GTX 2060 benchmark are incomplete",
        ],
        "artifacts": {
            "entry_summary": intro,
            "project_document": document,
            "project_video": video,
            "auxiliary_material": auxiliary,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"constraints_passed": constraints_passed, "duration": video["duration_seconds"]}
        )
    )
    return 0 if constraints_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
