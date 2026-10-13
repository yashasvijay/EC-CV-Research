"""Isolated opt-in pipeline while the grid search continues using src/pipeline.py."""
from time import perf_counter
import cv2

from .pipeline import Pipeline
from .native_gating import make_engine, native_decision


class NativeGatingPipeline(Pipeline):
    def __init__(self, detector, mode="adaptive", max_interval=10,
                 min_confidence=.6, max_displacement=.15, *, use_native_gating=False,
                 native_module_dir=None):
        super().__init__(detector, mode, max_interval, min_confidence, max_displacement)
        self.use_native_gating = use_native_gating
        self._native_engine = (make_engine(max_interval, min_confidence, max_displacement,
                                           native_module_dir) if use_native_gating else None)

    def step(self, frame):
        if not self.use_native_gating:
            return super().step(frame)  # unchanged original implementation
        start = perf_counter()
        self.frame_id += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.tracker.predict(gray)      # LK remains unchanged
        predicted = perf_counter()
        tracks = self.tracker.visible()
        reason, confidence, displacement = native_decision(
            self._native_engine, tracks, self.frame_id, self.last_detection, self.mode)
        gated = perf_counter()
        detector_ms = update_ms = 0.
        if reason is not None:
            before = perf_counter()
            detections = self.detector(frame)
            after = perf_counter()
            self.tracker.update(gray, detections)
            detector_ms = (after - before) * 1000
            update_ms = (perf_counter() - after) * 1000
            self.last_detection = self.frame_id
        return self.tracker.visible(), {
            "frame": self.frame_id, "detected": reason is not None, "trigger": reason,
            "confidence_before_detection": confidence, "displacement_fraction": displacement,
            "flow_ms": (predicted - start) * 1000, "gating_ms": (gated - predicted) * 1000,
            "detector_ms": detector_ms, "association_ms": update_ms,
            "pipeline_ms": (perf_counter() - start) * 1000, "gating_backend": "native",
        }
