from __future__ import annotations

import unittest

from scripts.patchcore_lite_bottle import protocol_counts, resolve_category


class PatchCoreLiteMvtecProtocolTest(unittest.TestCase):
    def test_full_support_counts_preserve_development_and_final_test(self) -> None:
        self.assertEqual(
            protocol_counts(normal_count=209, seen_anomaly_count=42),
            {
                "normal_support": 100,
                "development_normal": 20,
                "anomaly_support": 30,
                "development_anomaly": 6,
                "final_seen_anomaly": 6,
            },
        )

    def test_insufficient_category_reduces_support_without_reuse(self) -> None:
        counts = protocol_counts(normal_count=60, seen_anomaly_count=30)
        self.assertEqual(counts["normal_support"], 48)
        self.assertEqual(counts["development_normal"], 12)
        self.assertEqual(counts["anomaly_support"], 22)
        self.assertEqual(counts["development_anomaly"], 4)
        self.assertEqual(counts["final_seen_anomaly"], 4)
        self.assertEqual(counts["normal_support"] + counts["development_normal"], 60)
        self.assertEqual(
            counts["anomaly_support"]
            + counts["development_anomaly"]
            + counts["final_seen_anomaly"],
            30,
        )

    def test_category_accepts_short_name_but_rejects_unknown(self) -> None:
        rows = [
            {"product_id": "mvtec_ad_bottle"},
            {"product_id": "mvtec_ad_cable"},
        ]
        self.assertEqual(resolve_category(rows, "bottle"), "mvtec_ad_bottle")
        with self.assertRaisesRegex(RuntimeError, "unknown or ambiguous category"):
            resolve_category(rows, "pill")


if __name__ == "__main__":
    unittest.main()

