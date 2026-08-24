from __future__ import annotations

import unittest

from evoinspect.fusion import (
    apply_guarded_fusion,
    apply_selective_rescue,
    calibrate_guarded_fusion,
    calibrate_selective_rescue,
)


class GuardedFusionTest(unittest.TestCase):
    def test_or_is_selected_only_with_required_development_gain(self) -> None:
        calibration = calibrate_guarded_fusion(
            patch_scores=[0.1, 0.2, 0.4, 0.9],
            supervised_scores=[0.1, 0.2, 0.9, 0.8],
            labels=[0, 0, 1, 1],
            patch_threshold=0.5,
            supervised_threshold=0.5,
            min_development_f1_gain=0.1,
        )
        self.assertEqual(calibration["strategy"], "guarded_or")

    def test_patchcore_is_conservative_tie_break(self) -> None:
        calibration = calibrate_guarded_fusion(
            patch_scores=[0.1, 0.2, 0.8, 0.9],
            supervised_scores=[0.1, 0.2, 0.8, 0.9],
            labels=[0, 0, 1, 1],
            patch_threshold=0.5,
            supervised_threshold=0.5,
            min_development_f1_gain=0.0,
        )
        self.assertEqual(calibration["strategy"], "patchcore_only")

    def test_guarded_or_never_removes_patchcore_positive(self) -> None:
        _, decision = apply_guarded_fusion(0.8, 0.1, 0.5, 0.5, "guarded_or")
        self.assertEqual(decision, 1)

    def test_small_anomaly_development_slice_forces_fallback(self) -> None:
        calibration = calibrate_guarded_fusion(
            patch_scores=[0.1, 0.2, 0.4, 0.9],
            supervised_scores=[0.1, 0.2, 0.9, 0.8],
            labels=[0, 0, 1, 1],
            patch_threshold=0.5,
            supervised_threshold=0.5,
            min_development_f1_gain=0.0,
            min_development_anomalies=3,
        )
        self.assertEqual(calibration["strategy"], "patchcore_only")
        self.assertEqual(calibration["fallback_reason"], "insufficient_development_anomalies")

    def test_selective_rescue_requires_both_evidence_sources(self) -> None:
        _, far_patch_decision = apply_selective_rescue(
            0.1, 0.99, 0.5, 0.5, 0.5, "selective_rescue"
        )
        _, weak_supervised_decision = apply_selective_rescue(
            0.4, 0.1, 0.5, 0.5, 0.5, "selective_rescue"
        )
        _, rescued_decision = apply_selective_rescue(
            0.4, 0.9, 0.5, 0.5, 0.5, "selective_rescue"
        )
        self.assertEqual(
            (far_patch_decision, weak_supervised_decision, rescued_decision), (0, 0, 1)
        )

    def test_selective_rescue_selects_conservative_eligible_band(self) -> None:
        calibration = calibrate_selective_rescue(
            patch_scores=[0.1, 0.2, 0.3, 0.8],
            supervised_scores=[0.1, 0.2, 0.9, 0.8],
            labels=[0, 0, 1, 1],
            patch_threshold=0.5,
            supervised_threshold=0.5,
            min_patch_ratio_candidates=[0.25, 0.5, 0.75],
            min_development_f1_gain=0.1,
            max_development_precision_drop=0.0,
            min_development_anomalies=2,
        )
        self.assertEqual(calibration["strategy"], "selective_rescue")
        self.assertEqual(calibration["selected_min_patch_ratio"], 0.5)

    def test_selective_rescue_falls_back_when_precision_drops(self) -> None:
        calibration = calibrate_selective_rescue(
            patch_scores=[0.1, 0.4, 0.3, 0.8],
            supervised_scores=[0.1, 0.9, 0.9, 0.8],
            labels=[0, 0, 1, 1],
            patch_threshold=0.5,
            supervised_threshold=0.5,
            min_patch_ratio_candidates=[0.5],
            min_development_f1_gain=0.0,
            max_development_precision_drop=0.0,
            min_development_anomalies=2,
        )
        self.assertEqual(calibration["strategy"], "patchcore_only")
        self.assertEqual(calibration["fallback_reason"], "development_constraints_not_met")


if __name__ == "__main__":
    unittest.main()
