from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evoinspect.baseline import adapt_fixture_baseline, infer_manifest
from evoinspect.data import (
    read_manifest,
    split_manifest,
    validate_manifest,
    write_protocol_views,
)
from evoinspect.evaluation import evaluate_predictions
from scripts.generate_fixture import generate

CONFIG = {
    "schema_version": 1,
    "feature_grid": 4,
    "split": {
        "seed": 130,
        "normal_support": 4,
        "anomaly_support": 2,
        "development_normal": 2,
        "development_anomaly": 2,
        "unseen_defect_types": ["dent"],
    },
}


class VerticalSliceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source = generate(self.root / "source")
        self.validated = self.root / "validated.csv"
        validate_manifest(source, self.validated)
        self.split = self.root / "split.csv"
        split_manifest(self.validated, self.split, CONFIG["split"])
        self.adaptation = self.root / "adaptation.csv"
        self.test_inputs = self.root / "test_inputs.csv"
        self.test_truth = self.root / "test_truth.csv"
        write_protocol_views(
            self.split,
            self.adaptation,
            self.test_inputs,
            self.test_truth,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_adaptation_manifest_contains_no_final_test_metadata(self) -> None:
        adaptation_rows = read_manifest(self.adaptation)
        test_input_rows = read_manifest(self.test_inputs)
        test_truth_rows = read_manifest(self.test_truth)
        self.assertNotIn("final_test", {row["role"] for row in adaptation_rows})
        self.assertTrue(all(not row["label"] for row in test_input_rows))
        self.assertTrue(all(not row["defect_type"] for row in test_input_rows))
        self.assertTrue(all(not row["path"] for row in test_truth_rows))
        adaptation_ids = {row["sample_id"] for row in adaptation_rows}
        test_ids = {row["sample_id"] for row in test_truth_rows}
        self.assertFalse(adaptation_ids & test_ids)
        model_path = self.root / "model.json"
        model = adapt_fixture_baseline(self.adaptation, CONFIG, model_path)
        self.assertEqual(model["calibration_samples"], 4)
        self.assertTrue(model_path.is_file())

    def test_complete_vertical_slice(self) -> None:
        model_path = self.root / "model.json"
        model = adapt_fixture_baseline(self.adaptation, CONFIG, model_path)
        predictions = self.root / "predictions.jsonl"
        infer_summary = infer_manifest(self.test_inputs, model_path, predictions, {"final_test"})
        metrics_path = self.root / "metrics.json"
        metrics = evaluate_predictions(
            self.test_truth, predictions, metrics_path, str(model["model_hash"])
        )
        final_test_count = len(read_manifest(self.test_truth))
        self.assertEqual(infer_summary["predictions"], final_test_count)
        self.assertEqual(int(metrics["overall"]["samples"]), final_test_count)
        self.assertEqual(metrics["status"], "engineering_test_only")
        stored = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertIn("forbidden", stored["warning"])


if __name__ == "__main__":
    unittest.main()
