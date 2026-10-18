from __future__ import annotations
from typing import Any, Callable
from .types import Array
from .tracker import Track
from .sort_tracker import SortTrack
from time import perf_counter
import cv2
from .tracker import LKTracker
from .sort_tracker import SortTracker


class Pipeline:
    """Synchronous LK tracking with configurable detector scheduling."""

    def __init__(
        self,
        detector: Callable[[Array], Array],
        mode: str = "adaptive",
        max_interval: int = 10,
        min_confidence: float = 0.6,
        max_displacement: float = 0.15,
    ) -> None:
        """Initialize validated configuration and owned runtime state."""
        if mode not in ("baseline", "adaptive", "fixed") or max_interval < 1:
            raise ValueError("Invalid mode or detection interval")
        self.detector, self.mode = detector, mode
        self.max_interval = max_interval
        self.min_confidence, self.max_displacement = min_confidence, max_displacement
        self.tracker = LKTracker()
        self.frame_id, self.last_detection = 0, 0

    def step(self, frame: Array) -> tuple[list[Track], dict[str, Any]]:
        """Process one source frame and return tracks with timing telemetry."""
        start = perf_counter()
        self.frame_id += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.tracker.predict(gray)
        predicted = perf_counter()
        tracks = self.tracker.visible()
        confidence = min((t.confidence for t in tracks), default=0.0)
        displacement = max((t.displacement for t in tracks), default=0.0)
        reason = None
        if self.mode == "baseline":
            reason = "baseline"
        elif self.mode == "fixed":
            reason = "fixed_interval" if (self.frame_id - 1) % self.max_interval == 0 else None
        elif not tracks:
            reason = "no_tracks"
        elif self.frame_id - self.last_detection >= self.max_interval:
            reason = "max_interval"
        elif confidence < self.min_confidence:
            reason = "confidence"
        elif displacement > self.max_displacement:
            reason = "displacement"
        detector_ms = update_ms = 0.0
        if reason:
            before = perf_counter()
            detections = self.detector(frame)
            after = perf_counter()
            self.tracker.update(gray, detections)
            detector_ms = (after - before) * 1000
            update_ms = (perf_counter() - after) * 1000
            self.last_detection = self.frame_id
        return self.tracker.visible(), {
            "frame": self.frame_id,
            "detected": reason is not None,
            "trigger": reason,
            "confidence_before_detection": confidence,
            "displacement_fraction": displacement,
            "flow_ms": (predicted - start) * 1000,
            "detector_ms": detector_ms,
            "association_ms": update_ms,
            "pipeline_ms": (perf_counter() - start) * 1000,
        }


class SortPipeline:
    """Controlled baseline/fixed scheduling using eight-state Kalman prediction."""

    def __init__(
        self,
        detector: Callable[[Array], Array],
        mode: str = "baseline",
        max_interval: int = 1,
        max_age: int = 30,
        min_hits: int = 1,
        match_iou: float = 0.3,
    ) -> None:
        """Initialize validated configuration and owned runtime state."""
        if mode not in ("baseline", "fixed") or max_interval < 1:
            raise ValueError("SORT supports baseline or fixed mode with positive interval")
        self.detector, self.mode, self.max_interval = detector, mode, max_interval
        self.tracker = SortTracker(max_age, min_hits, match_iou)
        self.frame_id = 0

    def step(self, frame: Array) -> tuple[list[SortTrack], dict[str, Any]]:
        """Process one source frame and return tracks with timing telemetry."""
        start = perf_counter()
        self.frame_id += 1
        detected = self.mode == "baseline" or (self.frame_id - 1) % self.max_interval == 0
        detections = self.detector(frame) if detected else None
        after_detection = perf_counter()
        tracks = self.tracker.step(detections)
        end = perf_counter()
        return tracks, {
            "frame": self.frame_id,
            "detected": detected,
            "trigger": ("baseline" if self.mode == "baseline" else "fixed_interval") if detected else None,
            "confidence_before_detection": None,
            "displacement_fraction": None,
            "flow_ms": 0.0,
            "detector_ms": (after_detection - start) * 1000 if detected else 0.0,
            "tracker_ms": (end - after_detection) * 1000,
            "pipeline_ms": (end - start) * 1000,
        }
