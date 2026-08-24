# Deterministic sequence-FSM fixture report

## Scope

This is a CPU-only engineering fixture for the sequence FSM. It is not a real video benchmark and
does not support an official accuracy, latency, or 200 ms claim.

## Protocol and result

- Expected process: `screen -> battery -> cover`.
- Scenarios: normal, missing battery, reordered battery, repeated screen, unexpected robot step,
  and an anomaly burst.
- Command:

  ```bash
  PYTHONPATH=src python scripts/evaluate_sequence_fixture.py \
    --output reports/experiments/system-closure-sequence-fixture-20260825T054200Z/report.json
  ```

- Actual output: `scenarios=6 correct=6 accuracy=1.000000`.
- Event intervals, explanations, and observed/required event kinds are stored in `report.json`.

The fixture verifies the deterministic event vocabulary and interval serialization only. Real LOCO or
licensed video data, temporal IoU, event precision/recall, frame-rate stress, and deployment latency
remain open experiments.
