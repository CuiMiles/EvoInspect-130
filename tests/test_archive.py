from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from evoinspect.archive import inspect_archive
from evoinspect.errors import EvoInspectError
from evoinspect.provenance import file_sha256


class ArchiveInspectionTest(unittest.TestCase):
    def test_safe_zip_produces_receipt_without_extracting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("category/train/good/image.txt", "fixture")
            output = root / "receipt.json"
            receipt = inspect_archive(
                archive,
                file_sha256(archive),
                "fixture",
                "repository-test-code",
                output,
            )
            self.assertEqual(receipt["status"], "verified_not_extracted")
            self.assertEqual(receipt["member_count"], 1)
            self.assertTrue(output.is_file())
            self.assertFalse((root / "category").exists())

    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("../escape.txt", "unsafe")
            with self.assertRaisesRegex(EvoInspectError, "unsafe member"):
                inspect_archive(
                    archive,
                    file_sha256(archive),
                    "fixture",
                    "repository-test-code",
                    root / "receipt.json",
                )

    def test_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("data.txt", "fixture")
            with self.assertRaisesRegex(EvoInspectError, "SHA-256 mismatch"):
                inspect_archive(
                    archive,
                    "0" * 64,
                    "fixture",
                    "repository-test-code",
                    root / "receipt.json",
                )


if __name__ == "__main__":
    unittest.main()
