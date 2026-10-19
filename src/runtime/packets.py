from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol
import numpy as np
from ..schedulers.base import integer, number

Detector = Callable[[np.ndarray], np.ndarray]
Clock = Callable[[], int]


class FrameLease(Protocol):
    """Lease contract retaining contiguous history during replay."""

    frames: tuple[_Frame, ...]

    def release(self) -> None:
        """Release retained history after replay or result disposal."""
        ...


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated queue, history, replay and latency-budget limits."""

    deadline_ms: float = 33.0
    frame_rate: float = 30.0
    queue_capacity: int = 1
    history_frames: int = 16
    history_bytes: int = 64 * 1024 * 1024
    max_frame_bytes: int = 32 * 1024 * 1024
    max_detection_age_ms: float = 500.0
    max_replay_frames: int = 12
    max_commit_frames: int = 2
    fusion_headroom_ms: float = 3.0
    replay_budget_ms: float = 100.0
    max_tracks: int = 128
    max_points: int = 60
    min_points: int = 6
    max_frame_gap: int = 2
    match_iou: float = 0.3
    fb_threshold: float = 1.5
    max_track_age_ms: float = 1500.0
    telemetry_frames: int = 4096
    covariance: bool = True
    synchronous_bootstrap: bool = True
    bootstrap_min_interval_frames: int = 15
    offline: bool = False
    epoch: str = "default"

    def __post_init__(self) -> None:
        for name in (
            "deadline_ms",
            "frame_rate",
            "max_detection_age_ms",
            "replay_budget_ms",
            "fb_threshold",
            "max_track_age_ms",
        ):
            value = number(getattr(self, name), name)
            if value == 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "fusion_headroom_ms", number(self.fusion_headroom_ms, "fusion_headroom_ms"))
        object.__setattr__(self, "match_iou", number(self.match_iou, "match_iou", maximum=1))
        if self.match_iou == 0:
            raise ValueError("match_iou must be positive")
        for name in (
            "queue_capacity",
            "history_frames",
            "history_bytes",
            "max_frame_bytes",
            "max_replay_frames",
            "max_tracks",
            "max_points",
            "min_points",
            "max_frame_gap",
            "telemetry_frames",
            "bootstrap_min_interval_frames",
        ):
            object.__setattr__(self, name, integer(getattr(self, name), name, 1))
        object.__setattr__(self, "max_commit_frames", integer(self.max_commit_frames, "max_commit_frames"))
        if self.max_points < self.min_points:
            raise ValueError("max_points must be >= min_points")
        if not all(isinstance(v, bool) for v in (self.covariance, self.offline, self.synchronous_bootstrap)):
            raise TypeError("covariance/offline must be bool")
        if not isinstance(self.epoch, str) or not self.epoch:
            raise ValueError("epoch must be nonempty")

    @property
    def effective_replay_budget_ms(self) -> float:
        """Reserve a pickup interval inside the main thread's target-lag window."""
        window = self.max_commit_frames * 1000 / self.frame_rate
        reserve = min(1000 / self.frame_rate, window / 2)
        return min(self.replay_budget_ms, max(0.0, window - reserve))


@dataclass(frozen=True)
class _Frame:
    frame_idx: int
    capture_ns: int
    source_ns: int
    gray: np.ndarray


@dataclass(frozen=True)
class _Anchor:
    track_id: int
    revision: int
    box: tuple[float, ...]
    cls: int
    x: np.ndarray | None
    covariance: np.ndarray | None


@dataclass(frozen=True)
class DetectionRequest:
    """Owned source image and anchors admitted to detector work."""

    request_id: int
    epoch: str
    source: _Frame
    image: np.ndarray
    anchors: tuple[_Anchor, ...]
    admitted_ns: int
    expiry_ns: int
    lease: FrameLease

    def release(self) -> None:
        """Release the retained frame-history lease."""
        self.lease.release()


@dataclass(frozen=True)
class _Correction:
    """Births carry target-frame points; matches carry source residual/filter state."""

    detection: np.ndarray
    points: np.ndarray
    confidence: float
    track_id: int | None
    revision: int | None
    residual: np.ndarray | None
    filter_x: np.ndarray | None
    filter_p: np.ndarray | None


@dataclass(frozen=True)
class DetectionResult:
    """Source observation with matched residuals and births aligned to target_frame."""

    request_id: int
    epoch: str
    source_frame: int
    source_capture_ns: int
    source_time_ns: int
    expiry_ns: int
    detections: np.ndarray
    started_ns: int
    completed_ns: int
    published_ns: int
    target_frame: int
    corrections: tuple[_Correction, ...]
    target_lease: FrameLease | None
    rejection: str | None = None

    def release(self) -> None:
        """Release the retained frame-history lease."""
        if self.target_lease is not None:
            self.target_lease.release()


@dataclass(frozen=True)
class OutputTrack:
    """Immutable track state published for a specific source frame."""

    id: int
    box: tuple[float, ...]
    score: float
    cls: int
    confidence: float
    anchor_source_frame: int


@dataclass(frozen=True)
class FrameOutput:
    """Immutable causal output and per-frame timing telemetry."""

    frame_idx: int
    capture_ns: int
    published_ns: int
    tracks: tuple[OutputTrack, ...]
    telemetry: Mapping[str, object]


@dataclass(frozen=True)
class ShutdownReport:
    """Explicit worker shutdown outcome and failure state."""

    stopped: bool
    worker_alive: bool
    worker_error: str | None
