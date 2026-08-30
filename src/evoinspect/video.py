"""OpenCV video front end for the desktop assembly functional demonstration.

The color/ROI detector is intentionally a deterministic demo adapter, not an
industrial benchmark model.  ArUco markers are preferred when present; the
simple detector covers the fixed-camera bottle/cup/mouse recordings supplied
with this repository.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

DetectorMode = Literal["aruco", "simple"]


@dataclass(frozen=True)
class VideoDetectorConfig:
    expected_steps: tuple[str, ...] = ("cup", "bottle", "mouse")
    marker_to_component: tuple[tuple[int, str], ...] = ((0, "bottle"), (1, "cup"), (2, "mouse"))
    processing_width: int = 540
    sample_every_frames: int = 3
    stable_samples: int = 5
    blue_pixel_min: int = 5_000
    cup_without_bottle_min: int = 6_000
    cup_with_bottle_min: int = 17_000
    mouse_bright_pixel_min: int = 8_000

    def __post_init__(self) -> None:
        if not self.expected_steps or len(set(self.expected_steps)) != len(self.expected_steps):
            raise ValueError("expected_steps must be non-empty and unique")
        if self.processing_width < 64 or self.sample_every_frames < 1 or self.stable_samples < 1:
            raise ValueError("invalid video sampling configuration")


@dataclass(frozen=True)
class ComponentObservation:
    frame_index: int
    timestamp_s: float
    component: str | None
    mode: DetectorMode
    present: tuple[str, ...]
    confidence: float
    diagnostics: dict[str, float]


class StablePresenceTracker:
    """Emit a component only on a stable absent-to-present transition."""

    def __init__(self, components: tuple[str, ...], stable_samples: int) -> None:
        self._stable_samples = stable_samples
        self._stable = {component: False for component in components}
        self._candidate = dict(self._stable)
        self._counts = {component: 0 for component in components}

    def update(self, raw: dict[str, bool]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        appeared: list[str] = []
        for component in self._stable:
            value = bool(raw.get(component, False))
            if value == self._candidate[component]:
                self._counts[component] += 1
            else:
                self._candidate[component] = value
                self._counts[component] = 1
            if self._counts[component] >= self._stable_samples and value != self._stable[component]:
                self._stable[component] = value
                if value:
                    appeared.append(component)
        present = tuple(component for component, value in self._stable.items() if value)
        return tuple(appeared), present


class DesktopComponentDetector:
    """ArUco-first detector with a fixed-camera color/ROI fallback."""

    def __init__(self, config: VideoDetectorConfig | None = None) -> None:
        self.config = config or VideoDetectorConfig()
        self._tracker = StablePresenceTracker(
            self.config.expected_steps, self.config.stable_samples
        )

    def _aruco(self, frame: Any) -> tuple[dict[str, bool], dict[str, float]] | None:
        import cv2

        if not hasattr(cv2, "aruco"):
            return None
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary)
            corners, ids, _ = detector.detectMarkers(frame)
        elif hasattr(cv2.aruco, "detectMarkers"):
            corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary)
        else:
            return None
        if ids is None or len(corners) == 0:
            return None
        mapping = dict(self.config.marker_to_component)
        raw = {component: False for component in self.config.expected_steps}
        unknown = 0
        for marker in ids.flatten().tolist():
            component = mapping.get(int(marker))
            if component is None:
                unknown += 1
            else:
                raw[component] = True
        return raw, {"aruco_count": float(len(ids)), "unknown_marker_count": float(unknown)}

    def _simple(self, frame: Any) -> tuple[dict[str, bool], dict[str, float]]:
        import cv2
        import numpy as np

        height, width = frame.shape[:2]
        scale = self.config.processing_width / width
        resized = cv2.resize(frame, (self.config.processing_width, round(height * scale)))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        scaled_height, scaled_width = hsv.shape[:2]
        x0, x1 = round(0.09 * scaled_width), round(0.91 * scaled_width)
        y0, y1 = round(0.23 * scaled_height), round(0.75 * scaled_height)
        mat = hsv[y0:y1, x0:x1]
        blue = (
            (mat[:, :, 0] >= 90) & (mat[:, :, 0] <= 130) & (mat[:, :, 1] > 70) & (mat[:, :, 2] > 50)
        )
        bright = (mat[:, :, 1] < 70) & (mat[:, :, 2] > 145)
        split = round(0.36 * bright.shape[1])
        blue_pixels = int(np.count_nonzero(blue))
        left_bright = int(np.count_nonzero(bright[:, :split]))
        right_bright = int(np.count_nonzero(bright[:, split:]))
        bottle = blue_pixels >= self.config.blue_pixel_min
        cup_limit = (
            self.config.cup_with_bottle_min if bottle else self.config.cup_without_bottle_min
        )
        cup = right_bright >= cup_limit
        mouse = left_bright >= self.config.mouse_bright_pixel_min
        raw = {"bottle": bottle, "cup": cup, "mouse": mouse}
        return raw, {
            "blue_pixels": float(blue_pixels),
            "left_bright_pixels": float(left_bright),
            "right_bright_pixels": float(right_bright),
        }

    def observe(self, frame: Any, frame_index: int, timestamp_s: float) -> ComponentObservation:
        if frame_index < 0 or not math.isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError("invalid frame index or timestamp")
        detected = self._aruco(frame)
        mode: DetectorMode
        if detected is None:
            raw, diagnostics = self._simple(frame)
            mode = "simple"
        else:
            raw, diagnostics = detected
            mode = "aruco"
        appeared, present = self._tracker.update(raw)
        component = appeared[0] if appeared else None
        confidence = 1.0 if component is not None else 0.0
        return ComponentObservation(
            frame_index,
            timestamp_s,
            component,
            mode,
            present,
            confidence,
            diagnostics,
        )
