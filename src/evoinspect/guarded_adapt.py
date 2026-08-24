"""Reversible feedback updates with candidate-model gatekeeping.

This module is deliberately model-agnostic.  It records the evidence required
to publish a candidate model, but it does not train one or treat feedback as
permission to mutate the active model directly.  Threshold/memory updates are
reversible immediate changes; model artifacts require a replay/anchor/shadow
gate and remain rollbackable after publication.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

ImmediateKind = Literal["threshold", "memory", "rule"]


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class GatePolicy:
    """Predeclared candidate publication limits."""

    min_feedback_gain: float = 0.0
    max_anchor_regression: float = 0.01
    require_shadow_pass: bool = True
    max_artifact_bytes: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_feedback_gain):
            raise ValueError("min_feedback_gain must be finite")
        if not math.isfinite(self.max_anchor_regression) or self.max_anchor_regression < 0:
            raise ValueError("max_anchor_regression must be finite and non-negative")
        if self.max_artifact_bytes is not None and self.max_artifact_bytes <= 0:
            raise ValueError("max_artifact_bytes must be positive when provided")


@dataclass(frozen=True)
class ModelVersion:
    """Immutable model-version record retained for rollback."""

    version_id: str
    artifact_path: str
    artifact_sha256: str
    training_data_hash: str
    parent_version: str | None
    status: Literal["active", "superseded", "rejected"]
    created_at: str


@dataclass(frozen=True)
class CandidateUpdate:
    """Evidence package for a candidate model update."""

    version_id: str
    parent_version: str
    artifact_path: str
    training_data_hash: str
    feedback_f1_before: float
    feedback_f1_after: float
    anchor_f1_before: float
    anchor_f1_after: float
    shadow_passed: bool
    created_at: str = field(default_factory=_utc_now)

    @property
    def feedback_gain(self) -> float:
        return self.feedback_f1_after - self.feedback_f1_before

    @property
    def anchor_regression(self) -> float:
        return self.anchor_f1_before - self.anchor_f1_after

    def __post_init__(self) -> None:
        if not self.version_id.strip() or not self.parent_version.strip():
            raise ValueError("version_id and parent_version must be non-empty")
        if not self.training_data_hash.strip():
            raise ValueError("training_data_hash must be recorded")
        for name in (
            "feedback_f1_before",
            "feedback_f1_after",
            "anchor_f1_before",
            "anchor_f1_after",
        ):
            _finite(float(getattr(self, name)), name)
        if not self.artifact_path.strip():
            raise ValueError("artifact_path must be non-empty")


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    version_id: str
    feedback_gain: float
    anchor_regression: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "version_id": self.version_id,
            "feedback_gain": self.feedback_gain,
            "anchor_regression": self.anchor_regression,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ImmediateUpdate:
    update_id: str
    kind: ImmediateKind
    key: str
    value: float | str
    previous_value: float | str | None
    feedback_data_hash: str
    reason: str
    created_at: str
    reverted: bool = False


class GuardedAdaptController:
    """Two-level, auditable adaptation controller with rollback."""

    def __init__(
        self,
        initial_version: ModelVersion,
        *,
        policy: GatePolicy | None = None,
        state_path: Path | None = None,
    ) -> None:
        if initial_version.status != "active":
            raise ValueError("initial_version must have active status")
        self.policy = policy or GatePolicy()
        self.state_path = state_path
        self._versions: list[ModelVersion] = [initial_version]
        self._active_version_id = initial_version.version_id
        self._thresholds: dict[str, float] = {}
        self._memory: list[str] = []
        self._immediate_updates: list[ImmediateUpdate] = []
        self._decisions: list[GateDecision] = []

    @property
    def active_version(self) -> ModelVersion:
        for version in reversed(self._versions):
            if version.version_id == self._active_version_id:
                return version
        raise RuntimeError("active version record is missing")

    @property
    def thresholds(self) -> dict[str, float]:
        return dict(self._thresholds)

    @property
    def memory(self) -> tuple[str, ...]:
        return tuple(self._memory)

    @property
    def versions(self) -> tuple[ModelVersion, ...]:
        return tuple(self._versions)

    @property
    def decisions(self) -> tuple[GateDecision, ...]:
        return tuple(self._decisions)

    def _record_immediate(
        self,
        kind: ImmediateKind,
        key: str,
        value: float | str,
        previous_value: float | str | None,
        feedback_data_hash: str,
        reason: str,
    ) -> ImmediateUpdate:
        if not feedback_data_hash.strip() or not reason.strip():
            raise ValueError("feedback_data_hash and reason are required")
        update = ImmediateUpdate(
            update_id=f"imm-{len(self._immediate_updates) + 1:04d}",
            kind=kind,
            key=key,
            value=value,
            previous_value=previous_value,
            feedback_data_hash=feedback_data_hash,
            reason=reason,
            created_at=_utc_now(),
        )
        self._immediate_updates.append(update)
        self._save_if_configured()
        return update

    def update_threshold(
        self, key: str, value: float, *, feedback_data_hash: str, reason: str
    ) -> ImmediateUpdate:
        if not key.strip():
            raise ValueError("threshold key must be non-empty")
        new_value = _finite(value, "threshold")
        previous = self._thresholds.get(key)
        self._thresholds[key] = new_value
        return self._record_immediate(
            "threshold", key, new_value, previous, feedback_data_hash, reason
        )

    def add_memory(
        self, memory_id: str, *, feedback_data_hash: str, reason: str
    ) -> ImmediateUpdate:
        if not memory_id.strip():
            raise ValueError("memory_id must be non-empty")
        if memory_id in self._memory:
            raise ValueError("memory_id is already active")
        self._memory.append(memory_id)
        return self._record_immediate(
            "memory", memory_id, memory_id, None, feedback_data_hash, reason
        )

    def revert_immediate(self, update_id: str) -> ImmediateUpdate:
        """Revert one immediate update without deleting its audit record."""

        for index, update in enumerate(self._immediate_updates):
            if update.update_id != update_id:
                continue
            if update.reverted:
                return update
            if update.kind == "threshold":
                if update.previous_value is None:
                    self._thresholds.pop(update.key, None)
                else:
                    self._thresholds[update.key] = float(update.previous_value)
            elif update.kind == "memory":
                self._memory = [item for item in self._memory if item != update.key]
            reverted = ImmediateUpdate(**{**asdict(update), "reverted": True})
            self._immediate_updates[index] = reverted
            self._save_if_configured()
            return reverted
        raise KeyError(f"unknown immediate update: {update_id}")

    def evaluate_candidate(self, candidate: CandidateUpdate) -> GateDecision:
        """Evaluate a candidate without publishing it."""

        reasons: list[str] = []
        artifact = Path(candidate.artifact_path)
        if not artifact.is_file():
            reasons.append("candidate artifact does not exist")
        elif self.policy.max_artifact_bytes is not None:
            if artifact.stat().st_size > self.policy.max_artifact_bytes:
                reasons.append("candidate artifact exceeds the declared size limit")
        if candidate.parent_version != self._active_version_id:
            reasons.append("candidate parent_version is not the active model version")
        if candidate.feedback_gain < self.policy.min_feedback_gain:
            reasons.append("feedback slice did not meet the minimum gain")
        if candidate.anchor_regression > self.policy.max_anchor_regression:
            reasons.append("anchor regression exceeds the allowed limit")
        if self.policy.require_shadow_pass and not candidate.shadow_passed:
            reasons.append("shadow validation did not pass")
        decision = GateDecision(
            accepted=not reasons,
            version_id=candidate.version_id,
            feedback_gain=float(candidate.feedback_gain),
            anchor_regression=float(candidate.anchor_regression),
            reasons=tuple(reasons),
        )
        self._decisions.append(decision)
        self._save_if_configured()
        return decision

    def publish_candidate(self, candidate: CandidateUpdate) -> GateDecision:
        """Publish only a candidate that passes the full gate."""

        decision = self.evaluate_candidate(candidate)
        if not decision.accepted:
            return decision
        artifact = Path(candidate.artifact_path)
        artifact_hash = _sha256(artifact)
        for version in self._versions:
            if version.version_id == self._active_version_id:
                self._versions[self._versions.index(version)] = ModelVersion(
                    version_id=version.version_id,
                    artifact_path=version.artifact_path,
                    artifact_sha256=version.artifact_sha256,
                    training_data_hash=version.training_data_hash,
                    parent_version=version.parent_version,
                    status="superseded",
                    created_at=version.created_at,
                )
                break
        self._versions.append(
            ModelVersion(
                version_id=candidate.version_id,
                artifact_path=str(artifact),
                artifact_sha256=artifact_hash,
                training_data_hash=candidate.training_data_hash,
                parent_version=candidate.parent_version,
                status="active",
                created_at=candidate.created_at,
            )
        )
        self._active_version_id = candidate.version_id
        self._save_if_configured()
        return decision

    def rollback_model(self, version_id: str) -> ModelVersion:
        """Activate a previous immutable version; retain all version records."""

        target = next((item for item in self._versions if item.version_id == version_id), None)
        if target is None:
            raise KeyError(f"unknown model version: {version_id}")
        if not Path(target.artifact_path).is_file():
            raise FileNotFoundError(target.artifact_path)
        self._versions = [
            ModelVersion(
                version_id=item.version_id,
                artifact_path=item.artifact_path,
                artifact_sha256=item.artifact_sha256,
                training_data_hash=item.training_data_hash,
                parent_version=item.parent_version,
                status="active" if item.version_id == version_id else "superseded",
                created_at=item.created_at,
            )
            for item in self._versions
        ]
        self._active_version_id = version_id
        self._save_if_configured()
        return self.active_version

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": asdict(self.policy),
            "active_version_id": self._active_version_id,
            "versions": [asdict(item) for item in self._versions],
            "thresholds": dict(self._thresholds),
            "memory": list(self._memory),
            "immediate_updates": [asdict(item) for item in self._immediate_updates],
            "decisions": [item.to_dict() for item in self._decisions],
        }

    def save(self, path: Path | None = None) -> None:
        """Atomically persist controller state for restart and audit."""

        destination = path or self.state_path
        if destination is None:
            raise ValueError("a state path is required")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as stream:
            stream.write(payload)
            temporary = Path(stream.name)
        temporary.replace(destination)

    def _save_if_configured(self) -> None:
        if self.state_path is not None:
            self.save()
