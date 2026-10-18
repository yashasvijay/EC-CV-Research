from ..opencv_compat import sparse_lk
import numpy as np
from .packets import RuntimeConfig
from ..features import select_roi_features


def _owned(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True, order="C")
    result.flags.writeable = False
    return result


def _advance_points(
    previous: np.ndarray, current: np.ndarray, points: np.ndarray, box: np.ndarray, config: RuntimeConfig
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(points) < config.min_points:
        return box.copy(), np.empty((0, 1, 2), np.float32), 0.0
    forward, status, _ = sparse_lk(previous, current, points)
    if forward is None or status is None or not np.isfinite(forward).all():
        return box.copy(), np.empty((0, 1, 2), np.float32), 0.0
    backward, bs, _ = sparse_lk(current, previous, forward)
    if backward is None or bs is None:
        return box.copy(), np.empty((0, 1, 2), np.float32), 0.0
    errors = np.linalg.norm((backward - points).reshape(-1, 2), axis=1)
    xy = forward.reshape(-1, 2)
    valid = (
        (status.ravel() == 1)
        & (bs.ravel() == 1)
        & np.isfinite(errors)
        & (errors <= config.fb_threshold)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < current.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < current.shape[0])
    )
    if valid.sum() < config.min_points:
        return box.copy(), np.empty((0, 1, 2), np.float32), 0.0
    shifts = (forward - points).reshape(-1, 2)[valid]
    delta = np.median(shifts, axis=0)
    spread = np.median(np.linalg.norm(shifts - delta, axis=1))
    quality = float(
        valid.mean() * np.exp(-np.median(errors[valid]) / config.fb_threshold) * np.exp(-spread / 3)
    )
    moved = box + np.tile(delta, 2)
    moved[[0, 2]] = moved[[0, 2]].clip(0, current.shape[1])
    moved[[1, 3]] = moved[[1, 3]].clip(0, current.shape[0])
    return moved, forward[valid].reshape(-1, 1, 2), quality


def _seed(gray: np.ndarray, box: np.ndarray, config: RuntimeConfig) -> np.ndarray:
    return select_roi_features(gray, box, config.max_points)
