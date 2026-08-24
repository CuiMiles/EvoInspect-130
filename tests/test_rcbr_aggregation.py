from __future__ import annotations

import pytest

from scripts.aggregate_rcbr import reference_row
from scripts.evaluate_saved_localization import mean_optional_metric


def test_optional_metric_skips_unavailable_category_rows() -> None:
    rows = [
        {"source": {"unseen": {"f1": 0.8}}},
        {"source": {"unseen": {"available": False}}},
        {"source": {"unseen": {"f1": 1.0}}},
    ]
    assert mean_optional_metric(rows, ("source", "unseen", "f1")) == pytest.approx(0.9)
    assert mean_optional_metric(rows[1:2], ("source", "unseen", "f1")) is None


def test_patchcore_reference_preserves_unavailable_unseen_f1() -> None:
    reference = {
        "per_category": {
            "mvtec_ad_toothbrush": {
                "aupro_at_0_05": 0.5,
                "aupro_at_0_30": 0.8,
                "pro_at_fpr_0_01": 0.4,
                "fixed_small_aupro_at_0_05": None,
                "overall_f1": 0.9,
                "unseen_f1": None,
                "image_auroc": 1.0,
            }
        }
    }
    row = reference_row(reference, "mvtec_ad_toothbrush")
    assert row["unseen_f1"] is None
