import cv2
import numpy as np


def select_roi_features(gray: np.ndarray, box: np.ndarray, max_points: int = 60,
                        quality_level: float = 0.01, min_distance: float = 5.0) -> np.ndarray:
    """Select corners inside a clipped box crop and return global pixel coordinates."""
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Expected a uint8 grayscale image")
    coordinates = np.asarray(box, dtype=np.float64)
    if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
        raise ValueError("Expected four finite xyxy coordinates")
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)) or max_points < 1:
        raise ValueError("max_points must be a positive integer")
    if not np.isfinite(quality_level) or not 0 < quality_level <= 1:
        raise ValueError("quality_level must be in (0,1]")
    if not np.isfinite(min_distance) or min_distance < 0:
        raise ValueError("min_distance must be finite and nonnegative")
    empty = np.empty((0, 1, 2), dtype=np.float32)
    x1, y1, x2, y2 = coordinates
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(gray.shape[1]), x2), min(float(gray.shape[0]), y2)
    if x2 <= x1 or y2 <= y1:
        return empty
    left, top = int(np.floor(x1)), int(np.floor(y1))
    right, bottom = int(np.ceil(x2)), int(np.ceil(y2))
    if right-left < 3 or bottom-top < 3:
        return empty
    roi = gray[top:bottom, left:right]
    mask = np.zeros(roi.shape, dtype=np.uint8)
    mask[int(np.ceil(y1))-top:, int(np.ceil(x1))-left:] = 255
    if not mask.any():
        return empty
    points = cv2.goodFeaturesToTrack(roi, int(max_points), float(quality_level),
                                     float(min_distance), mask=mask)
    if points is None:
        return empty
    points = points.astype(np.float32, copy=False) + np.array([left, top], dtype=np.float32)
    xy = points.reshape(-1, 2)
    inside = ((xy[:, 0] >= x1) & (xy[:, 0] < x2) &
              (xy[:, 1] >= y1) & (xy[:, 1] < y2))
    return np.ascontiguousarray(points[inside].reshape(-1, 1, 2), dtype=np.float32)
