"""Unified, switchable inference contract for accuracy and edge engines."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

EngineName = Literal["accuracy", "edge"]


@dataclass(frozen=True)
class LatencyBreakdown:
    """Milliseconds spent in each online stage for one batch-size-one request."""

    preprocess_ms: float
    model_ms: float
    postprocess_ms: float
    serialization_ms: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def end_to_end_ms(self) -> float:
        return sum(self.to_dict().values())

    def to_dict(self) -> dict[str, float]:
        return {
            "preprocess_ms": float(self.preprocess_ms),
            "model_ms": float(self.model_ms),
            "postprocess_ms": float(self.postprocess_ms),
            "serialization_ms": float(self.serialization_ms),
        }


@dataclass(frozen=True)
class InferenceResult:
    """Model-independent AOI output required by the frozen online graph."""

    anomaly_score: float
    is_anomaly: bool
    confidence: float
    defect_tag: str
    model_version: str
    engine: EngineName
    latency: LatencyBreakdown
    regions: tuple[tuple[int, int, int, int], ...] = ()
    mask: Any | None = field(default=None, repr=False, compare=False)
    nearest_normal_evidence: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.anomaly_score):
            raise ValueError("anomaly_score must be finite")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite in [0, 1]")
        if not self.defect_tag.strip() or not self.model_version.strip():
            raise ValueError("defect_tag and model_version must be non-empty")

    def to_dict(self, *, include_mask: bool = False) -> dict[str, Any]:
        output: dict[str, Any] = {
            "anomaly_score": float(self.anomaly_score),
            "is_anomaly": self.is_anomaly,
            "confidence": float(self.confidence),
            "defect_tag": self.defect_tag,
            "model_version": self.model_version,
            "engine": self.engine,
            "regions": [list(region) for region in self.regions],
            "nearest_normal_evidence": self.nearest_normal_evidence,
            "latency_ms": {
                **self.latency.to_dict(),
                "end_to_end_ms": self.latency.end_to_end_ms,
            },
            "metadata": self.metadata,
        }
        if include_mask:
            output["mask"] = self.mask
        return output


class InferenceEngine(Protocol):
    """Minimal interface implemented by PatchCore and EfficientAD adapters."""

    name: EngineName
    model_version: str

    def infer(self, image: Any) -> InferenceResult:
        """Run one batch-size-one request."""


Backend = Callable[[Any], dict[str, Any]]


def _regions(values: Any) -> tuple[tuple[int, int, int, int], ...]:
    result: list[tuple[int, int, int, int]] = []
    for value in values:
        if len(value) != 4:
            raise ValueError("each region must contain exactly four coordinates")
        result.append((int(value[0]), int(value[1]), int(value[2]), int(value[3])))
    return tuple(result)


class CallableInferenceEngine:
    """Adapter around a frozen callable backend with auditable stage timings."""

    def __init__(
        self,
        *,
        name: EngineName,
        model_version: str,
        threshold: float,
        preprocess: Callable[[Any], Any],
        backend: Backend,
        postprocess: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if not model_version.strip() or not math.isfinite(threshold):
            raise ValueError("model_version must be non-empty and threshold finite")
        self.name = name
        self.model_version = model_version
        self.threshold = float(threshold)
        self._preprocess = preprocess
        self._backend = backend
        self._postprocess = postprocess or (lambda value: value)

    def infer(self, image: Any) -> InferenceResult:
        started = time.perf_counter_ns()
        prepared = self._preprocess(image)
        after_preprocess = time.perf_counter_ns()
        raw = self._backend(prepared)
        after_model = time.perf_counter_ns()
        output = self._postprocess(raw)
        after_postprocess = time.perf_counter_ns()
        score = float(output["anomaly_score"])
        confidence = float(output.get("confidence", min(1.0, abs(score - self.threshold))))
        is_anomaly = bool(output.get("is_anomaly", score >= self.threshold))
        latency = LatencyBreakdown(
            preprocess_ms=(after_preprocess - started) / 1e6,
            model_ms=(after_model - after_preprocess) / 1e6,
            postprocess_ms=(after_postprocess - after_model) / 1e6,
        )
        return InferenceResult(
            anomaly_score=score,
            is_anomaly=is_anomaly,
            confidence=confidence,
            defect_tag=str(output.get("defect_tag", "unknown" if is_anomaly else "normal")),
            model_version=self.model_version,
            engine=self.name,
            latency=latency,
            regions=_regions(output.get("regions", ())),
            mask=output.get("mask"),
            nearest_normal_evidence=output.get("nearest_normal_evidence"),
            metadata=dict(output.get("metadata", {})),
        )


class SwitchableInferenceEngine:
    """One stable API with explicit Accuracy/Edge selection and no silent fallback."""

    def __init__(self, accuracy: InferenceEngine, edge: InferenceEngine) -> None:
        if accuracy.name != "accuracy" or edge.name != "edge":
            raise ValueError("engines must be registered under accuracy and edge respectively")
        self._engines: dict[EngineName, InferenceEngine] = {
            "accuracy": accuracy,
            "edge": edge,
        }

    def infer(self, image: Any, *, engine: EngineName) -> InferenceResult:
        return self._engines[engine].infer(image)

    def model_versions(self) -> dict[EngineName, str]:
        return {
            name: implementation.model_version for name, implementation in self._engines.items()
        }
