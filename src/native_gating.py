"""Explicit native decision helpers; never loaded by the default pipeline."""

from __future__ import annotations
from typing import Any
from .types import Array
from .tracker import Track
from types import ModuleType

from importlib import import_module
from pathlib import Path
import sys

import numpy as np


def load_native(module_dir: str | Path | None = None) -> ModuleType:
    """Load the explicitly requested optional native extension."""
    if module_dir is not None:
        directory = str(Path(module_dir).resolve())
        if directory not in sys.path:
            sys.path.insert(0, directory)
    try:
        return import_module("edge_gating")
    except ImportError as exc:
        raise RuntimeError(
            "Native gating requested but edge_gating is unavailable. Build csrc and "
            "set PYTHONPATH=csrc/build/python (or --native-module-dir)."
        ) from exc


def make_engine(
    max_interval: int = 10,
    min_confidence: float = 0.6,
    max_displacement: float = 0.15,
    module_dir: str | Path | None = None,
) -> Any:
    """Construct the native engine with the supplied gating thresholds."""
    native = load_native(module_dir)
    return native.GatingEngine(
        native.Config(
            max_interval=max_interval, min_confidence=min_confidence, max_displacement=max_displacement
        )
    )


def native_decision(
    engine: Any,
    tracks: list[Track],
    frame_id: int,
    last_detection: int,
    mode: str = "adaptive",
    release_gil: bool = True,
) -> tuple[str | None, float, float]:
    # Include packing in pipeline/benchmark latency; it is not a zero-cost bridge.
    """Pack track signals and return the native scheduling decision."""
    confidences = np.array([track.confidence for track in tracks], dtype=np.float64)
    displacements = np.array([track.displacement for track in tracks], dtype=np.float64)
    decision = engine.decide(
        frame_id, last_detection, confidences, displacements, mode=mode, release_gil=release_gil
    )
    return decision.reason, decision.min_confidence, decision.max_displacement


def python_decision(
    tracks: list[Track],
    frame_id: int,
    last_detection: int,
    mode: str = "adaptive",
    max_interval: int = 10,
    min_confidence: float = 0.6,
    max_displacement: float = 0.15,
) -> tuple[str | None, float, float]:
    """Reference block matching the existing valid-input Python schedule."""
    confidence = min((track.confidence for track in tracks), default=0.0)
    displacement = max((track.displacement for track in tracks), default=0.0)
    reason = None
    if mode == "baseline":
        reason = "baseline"
    elif mode == "fixed":
        reason = "fixed_interval" if (frame_id - 1) % max_interval == 0 else None
    elif not tracks:
        reason = "no_tracks"
    elif frame_id - last_detection >= max_interval:
        reason = "max_interval"
    elif confidence < min_confidence:
        reason = "confidence"
    elif displacement > max_displacement:
        reason = "displacement"
    return reason, confidence, displacement


def python_motion(
    previous: Array,
    forward: Array,
    backward: Array,
    fs: Array,
    bs: Array,
    box: Array,
    prior: float,
    width: int,
    height: int,
    min_points: int = 6,
    fb_threshold: float = 1.5,
    spread_scale: float = 3.0,
) -> tuple[float, float, Array, Array, Array, float | None]:
    """NumPy reference for the current LK reductions; excludes the LK solver.

    Return confidence, displacement, delta, box, valid indices and median error.
    This reference is used only by isolated validation and microbenchmarks.
    """
    count = len(previous)
    empty = np.empty(0, np.int64)
    zero = np.zeros(2, np.float32)
    if count < min_points or not np.isfinite(forward).all():
        return 0.0, 0.0, zero, box.copy(), empty, None
    error = np.linalg.norm((backward - previous).reshape(-1, 2), axis=1)
    xy = forward.reshape(-1, 2)
    good = (
        (fs.ravel() == 1)
        & (bs.ravel() == 1)
        & np.isfinite(error)
        & (error <= fb_threshold)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    indices = np.flatnonzero(good)
    if len(indices) < min_points:
        return 0.0, 0.0, zero, box.copy(), indices, None
    shifts = (forward - previous).reshape(-1, 2)[good]
    delta = np.median(shifts, axis=0)
    spread = float(np.median(np.linalg.norm(shifts - delta, axis=1)))
    median_error = float(np.median(error[good]))
    confidence = prior * float(
        good.mean() * np.exp(-median_error / fb_threshold) * np.exp(-spread / spread_scale)
    )
    diagonal = max(1.0, float(np.linalg.norm(box[2:] - box[:2])))
    displacement = float(np.linalg.norm(delta) / diagonal)
    translated = box + np.tile(delta, 2)
    translated[[0, 2]] = translated[[0, 2]].clip(0, width)
    translated[[1, 3]] = translated[[1, 3]].clip(0, height)
    if np.any(translated[2:] <= translated[:2]):
        confidence = 0.0
    return confidence, displacement, delta, translated, indices, median_error
