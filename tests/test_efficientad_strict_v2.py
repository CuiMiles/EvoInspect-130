from __future__ import annotations

from scripts.evaluate_efficientad_strict_100_30 import strict_calibration_rows


def test_strict_calibration_uses_only_held_out_normals_and_support_anomalies() -> None:
    rows = [
        *(
            {"sample_id": f"n-{index}", "role": "support_normal", "label": "normal"}
            for index in range(100)
        ),
        *(
            {"sample_id": f"a-{index}", "role": "support_anomaly", "label": "anomaly"}
            for index in range(30)
        ),
        {"sample_id": "dev", "role": "development", "label": "anomaly"},
    ]
    normals, anomalies = strict_calibration_rows(rows, 80)
    assert [row["sample_id"] for row in normals] == [f"n-{index}" for index in range(80, 100)]
    assert len(anomalies) == 30
    assert all(row["role"] != "development" for row in [*normals, *anomalies])
