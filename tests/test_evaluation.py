from __future__ import annotations

import unittest

from evoinspect.evaluation import average_precision, binary_metrics, roc_auc


class EvaluationTest(unittest.TestCase):
    def test_perfect_binary_metrics(self) -> None:
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        predictions = [0, 0, 1, 1]
        metrics = binary_metrics(labels, scores, predictions)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1_fixed_threshold"], 1.0)
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["average_precision"], 1.0)

    def test_auc_ties_count_half(self) -> None:
        self.assertEqual(roc_auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertEqual(average_precision([0, 1], [0.1, 0.9]), 1.0)


if __name__ == "__main__":
    unittest.main()
