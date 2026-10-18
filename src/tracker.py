"""Translation-only sparse LK tracker; Hungarian IoU association, not SORT."""

from __future__ import annotations
from .types import Array

from dataclasses import dataclass, field
from .motion import advance_batch
import numpy as np
from scipy.optimize import linear_sum_assignment
from .features import select_roi_features


def box_iou(a: Array, b: Array) -> float:
    """Return intersection over union for two xyxy boxes."""
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    intersection = np.maximum(hi - lo, 0).prod()
    return float(intersection / max(1e-9, np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection))


@dataclass
class Track:
    """Mutable LK track state anchored by detector observations."""

    id: int
    box: Array
    score: float
    cls: int
    points: Array = field(default_factory=lambda: np.empty((0, 1, 2), np.float32))
    confidence: float = 1.0
    displacement: float = 0.0
    fb_error: float | None = None
    missed: int = 0


class LKTracker:
    """Propagate sparse ROI features and associate detector boxes by IoU."""

    def __init__(
        self,
        min_points: int = 6,
        max_points: int = 60,
        fb_threshold: float = 1.5,
        match_iou: float = 0.3,
        max_missed: int = 2,
    ) -> None:
        """Initialize validated configuration and owned runtime state."""
        self.min_points, self.max_points = min_points, max_points
        self.fb_threshold, self.match_iou = fb_threshold, match_iou
        self.max_missed = max_missed  # detector updates, not video frames
        self.tracks: list[Track] = []
        self.previous: Array | None = None
        self.next_id = 1

    def predict(self, gray: Array) -> None:
        """Advance track state to the next source frame."""
        if self.previous is None:
            self.previous = gray.copy()
            return
        if gray.shape != self.previous.shape:
            raise ValueError("Frame dimensions changed within sequence")
        updates = advance_batch(
            self.previous,
            gray,
            [t.box for t in self.tracks],
            [t.points for t in self.tracks],
            min_points=self.min_points,
            fb_threshold=self.fb_threshold,
        )
        for track, update in zip(self.tracks, updates):
            track.box, track.points = update.box, update.points
            track.confidence *= update.quality
            track.displacement, track.fb_error = update.displacement, update.fb_error
        self.previous = gray.copy()

    def update(self, gray: Array, detections: Array) -> None:
        """Fuse the current detector observations into track state."""
        matched_tracks, matched_detections = set(), set()
        if self.tracks and len(detections):
            costs = np.ones((len(self.tracks), len(detections))) * 1e6
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(detections):
                    overlap = box_iou(t.box, d[:4])
                    if t.cls == int(d[5]) and overlap >= self.match_iou:
                        costs[i, j] = 1 - overlap
            rows, cols = linear_sum_assignment(costs)
            for row_index, column_index in zip(rows, cols):
                i, j = int(row_index), int(column_index)
                if costs[i, j] >= 1e6:
                    continue
                t, d = self.tracks[i], detections[j]
                t.box, t.score, t.missed = d[:4].copy(), float(d[4]), 0
                matched_tracks.add(i)
                matched_detections.add(j)
        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.missed += 1
                t.confidence = 0.0
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        for j, d in enumerate(detections):
            if j not in matched_detections:
                self.tracks.append(Track(self.next_id, d[:4].copy(), float(d[4]), int(d[5])))
                self.next_id += 1
        for t in self.tracks:
            if t.missed:
                continue
            t.points = select_roi_features(gray, t.box, self.max_points)
            t.confidence = 1.0 if len(t.points) >= self.min_points else 0.0
            t.displacement, t.fb_error = 0.0, None
        self.previous = gray.copy()

    def visible(self) -> list[Track]:
        """Return tracks eligible for publication on the current frame."""
        return [t for t in self.tracks if t.missed == 0]
