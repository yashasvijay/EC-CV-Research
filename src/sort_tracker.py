"""Eight-state SORT variant: [cx, cy, area, ratio, vx, vy, varea, vratio].

Original SORT has seven states (no ratio velocity). This implementation adds
the requested eighth state and explicit skipped-inference semantics. Units are
pixels and frames, not seconds. No appearance features or re-identification.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def measurement(box):
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("Expected finite, positive-area xyxy box")
    width, height = box[2:] - box[:2]
    return np.array([*(box[:2] + box[2:]) / 2, width * height, width / height])


class KalmanBox:
    def __init__(self, box):
        self.x = np.zeros(8, dtype=np.float64)
        self.x[:4] = measurement(box)
        self.P = np.diag([10., 10., 100., .1, 1000., 1000., 10000., 1.])
        self.H = np.eye(4, 8)
        self.R = np.diag([1., 1., 10., .01])

    def _positive_shape(self):
        for index in (2, 3):
            if self.x[index] <= 1e-6:
                self.x[index] = 1e-6
                self.x[index + 4] = 0.

    @property
    def box(self):
        width = np.sqrt(self.x[2] * self.x[3])
        height = self.x[2] / width
        half_size = np.array([width, height]) / 2
        return np.concatenate((self.x[:2] - half_size, self.x[:2] + half_size))

    def predict(self, dt=1.0):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        transition = np.eye(8)
        transition[:4, 4:] = np.eye(4) * dt
        # White acceleration model, with separate units for area and ratio.
        acceleration = np.diag([1., 1., 10., .001])
        noise = np.block([[dt**4 / 4 * acceleration, dt**3 / 2 * acceleration],
                          [dt**3 / 2 * acceleration, dt**2 * acceleration]])
        for index in (2, 3):
            if self.x[index] + dt * self.x[index + 4] <= 1e-6:
                self.x[index + 4] = 0.
        self.x = transition @ self.x
        self.P = transition @ self.P @ transition.T + noise
        self._positive_shape()
        return self.box

    def update(self, box):
        innovation = measurement(box) - self.H @ self.x
        covariance = self.H @ self.P @ self.H.T + self.R
        gain = np.linalg.solve(covariance, self.H @ self.P).T
        self.x += gain @ innovation
        residual = np.eye(8) - gain @ self.H
        # Joseph form protects covariance symmetry/positive semidefiniteness.
        self.P = residual @ self.P @ residual.T + gain @ self.R @ gain.T
        self.P = (self.P + self.P.T) / 2
        self._positive_shape()
        return self.box


class SortTrack:
    def __init__(self, track_id, detection):
        self.id = track_id
        self.filter = KalmanBox(detection[:4])
        self.score, self.cls = float(detection[4]), int(detection[5])
        self.hits, self.time_since_update = 1, 0
        self.matched_at_last_detection = True

    @property
    def box(self):
        return self.filter.box


def pairwise_iou(boxes, detections):
    lower = np.maximum(boxes[:, None, :2], detections[None, :, :2])
    upper = np.minimum(boxes[:, None, 2:], detections[None, :, 2:])
    intersection = np.maximum(upper - lower, 0).prod(axis=2)
    union = ((boxes[:, 2:] - boxes[:, :2]).prod(axis=1)[:, None]
             + (detections[:, 2:] - detections[:, :2]).prod(axis=1)[None, :] - intersection)
    return intersection / np.maximum(union, 1e-12)


class SortTracker:
    """Call step once per video frame.

    detections=None: intentionally skipped detector; emit confirmed predictions.
    detections=empty (0,6): detector ran and found nothing; suppress unmatched
    outputs until reacquired. max_age counts video frames since last match.
    min_hits counts total matched detector observations (default 1).
    """
    def __init__(self, max_age=30, min_hits=1, iou_threshold=0.3):
        if max_age < 0 or min_hits < 1 or not 0 < iou_threshold <= 1:
            raise ValueError("Invalid SORT age, hits, or IoU threshold")
        self.max_age, self.min_hits, self.iou_threshold = max_age, min_hits, iou_threshold
        self.tracks, self.next_id = [], 1

    def step(self, detections=None):
        if detections is not None:
            detections = np.asarray(detections, dtype=np.float64)
            if detections.ndim != 2 or detections.shape[1] != 6:
                raise ValueError("detections must have shape (N,6): xyxy,score,class")
            if (not np.isfinite(detections).all()
                    or np.any(detections[:, 2:4] <= detections[:, :2])
                    or np.any(detections[:, 5] != np.floor(detections[:, 5]))):
                raise ValueError("Invalid detection values")
        # A track stale at the preceding frame cannot be resurrected later.
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        for track in self.tracks:
            track.filter.predict()
            track.time_since_update += 1
        if detections is not None:
            matched = set()
            for track in self.tracks:
                track.matched_at_last_detection = False
            if self.tracks and len(detections):
                overlaps = pairwise_iou(np.array([t.box for t in self.tracks]), detections[:, :4])
                classes = np.array([t.cls for t in self.tracks])[:, None]
                valid = (overlaps >= self.iou_threshold) & (classes == detections[None, :, 5])
                cost = np.where(valid, 1 - overlaps, 1e6)
                rows, cols = linear_sum_assignment(cost)
                for i, j in zip(rows, cols):
                    if not valid[i, j]:
                        continue
                    track = self.tracks[i]
                    track.filter.update(detections[j, :4])
                    track.score = float(detections[j, 4])
                    track.hits += 1
                    track.time_since_update = 0
                    track.matched_at_last_detection = True
                    matched.add(j)
            for j, detection in enumerate(detections):
                if j not in matched:
                    self.tracks.append(SortTrack(self.next_id, detection))
                    self.next_id += 1
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.hits >= self.min_hits and t.matched_at_last_detection]
