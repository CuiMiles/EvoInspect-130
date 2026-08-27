from __future__ import annotations

from evoinspect.inference import CallableInferenceEngine, SwitchableInferenceEngine


def engine(name: str, score: float) -> CallableInferenceEngine:
    return CallableInferenceEngine(
        name=name,  # type: ignore[arg-type]
        model_version=f"{name}-v1",
        threshold=0.5,
        preprocess=lambda value: value + 1,
        backend=lambda value: {"anomaly_score": score, "metadata": {"input": value}},
    )


def test_switchable_engine_keeps_contract_and_version() -> None:
    service = SwitchableInferenceEngine(engine("accuracy", 0.8), engine("edge", 0.2))
    accuracy = service.infer(4, engine="accuracy")
    edge = service.infer(4, engine="edge")
    assert accuracy.is_anomaly is True
    assert accuracy.defect_tag == "unknown"
    assert accuracy.model_version == "accuracy-v1"
    assert accuracy.metadata == {"input": 5}
    assert edge.is_anomaly is False
    assert edge.defect_tag == "normal"
    assert service.model_versions() == {"accuracy": "accuracy-v1", "edge": "edge-v1"}
    assert accuracy.latency.end_to_end_ms >= 0
