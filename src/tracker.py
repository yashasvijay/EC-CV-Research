"""Translation-only sparse LK tracker; Hungarian IoU association, not SORT."""
from dataclasses import dataclass, field
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou(a, b):
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    intersection = np.maximum(hi - lo, 0).prod()
    return float(intersection / max(1e-9, np.prod(a[2:]-a[:2]) +
                                    np.prod(b[2:]-b[:2]) - intersection))


@dataclass
class Track:
    id: int
    box: np.ndarray
    score: float
    cls: int
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 1, 2), np.float32))
    confidence: float = 1.0
    displacement: float = 0.0
    fb_error: float | None = None
    missed: int = 0


class LKTracker:
    def __init__(self, min_points=6, max_points=60, fb_threshold=1.5,
                 match_iou=0.3, max_missed=2):
        self.min_points, self.max_points = min_points, max_points
        self.fb_threshold, self.match_iou = fb_threshold, match_iou
        self.max_missed = max_missed  # detector updates, not video frames
        self.tracks, self.previous, self.next_id = [], None, 1

    def predict(self, gray):
        if self.previous is None:
            self.previous = gray.copy()
            return
        if gray.shape != self.previous.shape:
            raise ValueError("Frame dimensions changed within sequence")
        for t in self.tracks:
            t.displacement, t.fb_error = 0.0, None
            if len(t.points) < self.min_points:
                t.confidence = 0.0
                continue
            old = t.points
            new, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, old, None,
                                                   winSize=(21, 21), maxLevel=3)
            if new is None or not np.isfinite(new).all():
                t.confidence = 0.0
                t.points = np.empty((0, 1, 2), np.float32)
                continue
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, new, None,
                                                         winSize=(21, 21), maxLevel=3)
            if back is None:
                t.confidence = 0.0
                continue
            errors = np.linalg.norm((back - old).reshape(-1, 2), axis=1)
            xy = new.reshape(-1, 2)
            good = ((status.ravel() == 1) & (back_status.ravel() == 1) &
                    np.isfinite(errors) & (errors <= self.fb_threshold) &
                    (xy[:, 0] >= 0) & (xy[:, 0] < gray.shape[1]) &
                    (xy[:, 1] >= 0) & (xy[:, 1] < gray.shape[0]))
            t.points = new[good].reshape(-1, 1, 2)
            if good.sum() < self.min_points:
                t.confidence = 0.0
                continue
            shifts = (new - old).reshape(-1, 2)[good]
            delta = np.median(shifts, axis=0)
            spread = float(np.median(np.linalg.norm(shifts - delta, axis=1)))
            t.fb_error = float(np.median(errors[good]))
            quality = float(good.mean() * np.exp(-t.fb_error / self.fb_threshold) *
                            np.exp(-spread / 3.0))
            t.confidence *= quality  # cumulative drift since last detector refresh
            diagonal = max(1.0, float(np.linalg.norm(t.box[2:] - t.box[:2])))
            t.displacement = float(np.linalg.norm(delta) / diagonal)
            t.box += np.tile(delta, 2)
            t.box[[0, 2]] = t.box[[0, 2]].clip(0, gray.shape[1])
            t.box[[1, 3]] = t.box[[1, 3]].clip(0, gray.shape[0])
            if np.any(t.box[2:] <= t.box[:2]):
                t.confidence = 0.0
        self.previous = gray.copy()

    def update(self, gray, detections):
        matched_tracks, matched_detections = set(), set()
        if self.tracks and len(detections):
            costs = np.ones((len(self.tracks), len(detections))) * 1e6
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(detections):
                    overlap = box_iou(t.box, d[:4])
                    if t.cls == int(d[5]) and overlap >= self.match_iou:
                        costs[i, j] = 1 - overlap
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols):
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
            mask = np.zeros_like(gray)
            x1, y1, x2, y2 = t.box.astype(int)
            mask[y1:y2, x1:x2] = 255
            points = cv2.goodFeaturesToTrack(gray, self.max_points, 0.01, 5, mask=mask)
            t.points = points if points is not None else np.empty((0, 1, 2), np.float32)
            t.confidence = 1.0 if len(t.points) >= self.min_points else 0.0
            t.displacement, t.fb_error = 0.0, None
        self.previous = gray.copy()

    def visible(self):
        return [t for t in self.tracks if t.missed == 0]
