from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from evoinspect.data import read_manifest, split_manifest, validate_manifest
from evoinspect.errors import EvoInspectError
from scripts.generate_fixture import generate


class DataProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = generate(self.root / "source")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_validation_rejects_content_duplicates(self) -> None:
        rows = list(csv.DictReader(self.source.read_text(encoding="utf-8").splitlines()))
        duplicate = dict(rows[0])
        duplicate["sample_id"] = "duplicate-content"
        rows.append(duplicate)
        duplicate_manifest = self.source.parent / "duplicate.csv"
        with duplicate_manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(EvoInspectError, "content duplicate"):
            validate_manifest(duplicate_manifest, self.root / "should-not-exist.csv")

    def test_split_is_hash_disjoint_and_holds_out_declared_defect(self) -> None:
        validated = self.root / "validated.csv"
        validate_manifest(self.source, validated)
        split = self.root / "split.csv"
        config = {
            "seed": 130,
            "normal_support": 4,
            "anomaly_support": 2,
            "development_normal": 2,
            "development_anomaly": 2,
            "unseen_defect_types": ["dent"],
        }
        summary = split_manifest(validated, split, config)
        rows = read_manifest(split)
        self.assertEqual(len(rows), len({row["content_sha256"] for row in rows}))
        self.assertEqual(summary["unseen_defect_types"], ["dent"])
        non_test_types = {
            row["defect_type"]
            for row in rows
            if row["role"] != "final_test" and row["label"] == "anomaly"
        }
        self.assertNotIn("dent", non_test_types)
        test_dents = [
            row for row in rows if row["role"] == "final_test" and row["defect_type"] == "dent"
        ]
        self.assertTrue(test_dents)
        self.assertTrue(all(row["defect_visibility"] == "unseen" for row in test_dents))
        test_visibilities = {
            row["defect_visibility"]
            for row in rows
            if row["role"] == "final_test" and row["label"] == "anomaly"
        }
        self.assertEqual(test_visibilities, {"seen", "unseen"})

    def test_split_rejects_unavailable_holdout(self) -> None:
        validated = self.root / "validated.csv"
        validate_manifest(self.source, validated)
        with self.assertRaisesRegex(EvoInspectError, "must be present"):
            split_manifest(
                validated,
                self.root / "split.csv",
                {
                    "seed": 1,
                    "normal_support": 4,
                    "anomaly_support": 2,
                    "development_normal": 2,
                    "development_anomaly": 2,
                    "unseen_defect_types": ["nonexistent"],
                },
            )


if __name__ == "__main__":
    unittest.main()
