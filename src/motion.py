"""Batched sparse LK: one forward/backward pyramid computation per image pair."""

from dataclasses import dataclass
import numpy as np
from .opencv_compat import sparse_lk
from .types import Array


@dataclass(frozen=True)
class MotionUpdate:
    """Owned propagated box/features and per-track reliability measurements."""

    box: Array
    points: Array
    quality: float = 0.0
    displacement: float = 0.0
    fb_error: float | None = None


def advance_batch(
    previous: Array,
    current: Array,
    boxes: list[Array],
    point_sets: list[Array],
    *,
    min_points: int = 6,
    fb_threshold: float = 1.5,
) -> list[MotionUpdate]:
    """Share LK image pyramids across tracks, then reduce each track independently.

    Python OpenCV rejects prebuilt pyramid tuples; batching lets each native LK
    call construct its pyramids once, rather than rebuilding them per track.
    """
    if len(boxes) != len(point_sets):
        raise ValueError("Boxes and point sets must have equal lengths")
    updates = [MotionUpdate(box.copy(), np.empty((0, 1, 2), np.float32)) for box in boxes]
    indices = [i for i, points in enumerate(point_sets) if len(points) >= min_points]
    if not indices:
        return updates
    old = np.ascontiguousarray(np.concatenate([point_sets[i] for i in indices]), dtype=np.float32)
    forward, status, _ = sparse_lk(previous, current, old)
    if forward is None or status is None:
        return updates
    finite = np.isfinite(forward).all(axis=(1, 2))
    safe = forward.copy()
    safe[~finite] = old[~finite]
    backward, bs, _ = sparse_lk(current, previous, safe)
    if backward is None or bs is None:
        return updates
    errors = np.linalg.norm((backward - old).reshape(-1, 2), axis=1)
    xy = forward.reshape(-1, 2)
    good = (
        (status.ravel() == 1)
        & (bs.ravel() == 1)
        & finite
        & np.isfinite(errors)
        & (errors <= fb_threshold)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < current.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < current.shape[0])
    )
    offset = 0
    for i in indices:
        end = offset + len(point_sets[i])
        valid = good[offset:end]
        new = forward[offset:end]
        original = old[offset:end]
        error = errors[offset:end]
        offset = end
        if np.count_nonzero(valid) < min_points:
            continue
        shifts = (new - original).reshape(-1, 2)[valid]
        delta = np.median(shifts, axis=0)
        spread = float(np.median(np.linalg.norm(shifts - delta, axis=1)))
        fb = float(np.median(error[valid]))
        quality = float(valid.mean() * np.exp(-fb / fb_threshold) * np.exp(-spread / 3))
        box = boxes[i] + np.tile(delta, 2)
        box[[0, 2]] = box[[0, 2]].clip(0, current.shape[1])
        box[[1, 3]] = box[[1, 3]].clip(0, current.shape[0])
        if np.any(box[2:] <= box[:2]):
            quality = 0.0
        displacement = float(
            np.linalg.norm(delta) / max(1.0, float(np.linalg.norm(boxes[i][2:] - boxes[i][:2])))
        )
        updates[i] = MotionUpdate(
            box.astype(np.float32), new[valid].reshape(-1, 1, 2).copy(), quality, displacement, fb
        )
    return updates
