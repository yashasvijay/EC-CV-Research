from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping

import numpy as np

Box = tuple[float, float, float, float]
Covariance2D = tuple[tuple[float, float], tuple[float, float]]


class MissingSignalError(ValueError):
    """The selected policy requires a signal the tracker does not provide."""


def integer(value: int, name: str, minimum: int = 0) -> int:
    """Validate an integer configuration value and its lower bound."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def number(value: float, name: str, minimum: float = 0.0, maximum: float | None = None) -> float:
    """Validate a finite numeric configuration value and bounds."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its finite valid range")
    return result


def text(value: str, name: str) -> str:
    """Validate nonempty text used in typed scheduler records."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class TrackState:
    """Immutable per-track signals supplied to a scheduling decision."""

    confidence: float | None = None
    displacement: float | None = None
    flow_valid: bool = True
    box: Box | None = None
    position_covariance: Covariance2D | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.flow_valid, bool):
            raise TypeError("flow_valid must be bool")
        for name, maximum in (("confidence", 1.0), ("displacement", None)):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, number(value, name, maximum=maximum))
        if self.box is not None:
            try:
                box = tuple(float(v) for v in self.box)
            except (TypeError, ValueError) as exc:
                raise ValueError("box must be finite xyxy coordinates") from exc
            if (
                len(box) != 4
                or not all(math.isfinite(v) for v in box)
                or box[2] <= box[0]
                or box[3] <= box[1]
                or not math.isfinite(
                    (box[2] - box[0]) * (box[2] - box[0]) + (box[3] - box[1]) * (box[3] - box[1])
                )
            ):
                raise ValueError("box must have finite positive width and height")
            object.__setattr__(self, "box", box)
        if self.position_covariance is not None:
            try:
                covariance = np.asarray(self.position_covariance, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("position_covariance must be a finite 2x2 matrix") from exc
            if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
                raise ValueError("position_covariance must be a finite 2x2 matrix")
            if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-10):
                raise ValueError("position_covariance must be symmetric")
            if np.linalg.eigvalsh(covariance).min() < -1e-10:
                raise ValueError("position_covariance must be positive semidefinite")
            covariance = covariance / 2 + covariance.T / 2
            object.__setattr__(
                self, "position_covariance", tuple(tuple(float(v) for v in row) for row in covariance)
            )


@dataclass(frozen=True)
class TrackerState:
    """Immutable visible tracks and latest fused source observation."""

    tracks: tuple[TrackState, ...] = ()
    last_fused_frame: int = 0
    last_fused_time_ns: int | None = None

    def __post_init__(self) -> None:
        try:
            tracks = tuple(self.tracks)
        except TypeError as exc:
            raise TypeError("tracks must contain TrackState records") from exc
        if any(not isinstance(t, TrackState) for t in tracks):
            raise TypeError("tracks must contain TrackState records")
        object.__setattr__(self, "tracks", tracks)
        object.__setattr__(self, "last_fused_frame", integer(self.last_fused_frame, "last_fused_frame"))
        if self.last_fused_time_ns is not None:
            if self.last_fused_frame == 0:
                raise ValueError("A fused timestamp requires a fused source frame")
            object.__setattr__(
                self, "last_fused_time_ns", integer(self.last_fused_time_ns, "last_fused_time_ns")
            )


@dataclass(frozen=True)
class StreamInfo:
    """Source cadence, offline mode and stream epoch identity."""

    epoch: str = "default"
    offline: bool = False
    frame_rate: float | None = None

    def __post_init__(self) -> None:
        text(self.epoch, "epoch")
        if not isinstance(self.offline, bool):
            raise TypeError("offline must be bool")
        if self.frame_rate is not None:
            rate = number(self.frame_rate, "frame_rate")
            if rate == 0:
                raise ValueError("frame_rate must be positive")
            object.__setattr__(self, "frame_rate", rate)


@dataclass(frozen=True)
class ScheduleContext:
    """Frame, timing and detector-admission inputs for one decision."""

    frame_idx: int
    tracker_state: TrackerState
    source_time_ns: int | None = None
    epoch: str = "default"
    offline: bool = False
    worker_available: bool = True
    pending_age_ms: float | None = None
    detector_service_ms: float | None = None
    remaining_budget_ms: float | None = None
    last_requested_frame: int | None = None
    last_admitted_frame: int | None = None
    last_started_frame: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_idx", integer(self.frame_idx, "frame_idx", 1))
        if not isinstance(self.tracker_state, TrackerState):
            raise TypeError("tracker_state must be TrackerState")
        text(self.epoch, "epoch")
        if not isinstance(self.offline, bool) or not isinstance(self.worker_available, bool):
            raise TypeError("offline and worker_available must be bool")
        if self.tracker_state.last_fused_frame > self.frame_idx:
            raise ValueError("A fused observation cannot be from a future frame")
        if self.source_time_ns is not None:
            object.__setattr__(self, "source_time_ns", integer(self.source_time_ns, "source_time_ns"))
            last = self.tracker_state.last_fused_time_ns
            if last is not None and last > self.source_time_ns:
                raise ValueError("A fused source timestamp cannot be in the future")
        for name in ("pending_age_ms", "detector_service_ms", "remaining_budget_ms"):
            value = getattr(self, name)
            if value is not None:
                # Negative headroom is valid after a deadline overrun.
                minimum = -math.inf if name == "remaining_budget_ms" else 0.0
                object.__setattr__(self, name, number(value, name, minimum=minimum))
        for name in ("last_requested_frame", "last_admitted_frame", "last_started_frame"):
            value = getattr(self, name)
            if value is not None:
                value = integer(value, name)
                if value > self.frame_idx:
                    raise ValueError(f"{name} cannot be from a future frame")
                object.__setattr__(self, name, value)


@dataclass(frozen=True)
class TriggerDecision:
    """Detector intent with reason, urgency and interpretable signal metadata."""

    should_detect: bool
    reason: str
    confidence: float | None = None
    urgency: str = "normal"
    max_source_age_ms: float | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.should_detect, bool):
            raise TypeError("should_detect must be bool")
        text(self.reason, "reason")
        if self.confidence is not None:
            object.__setattr__(self, "confidence", number(self.confidence, "confidence", maximum=1))
        if self.urgency not in ("normal", "urgent"):
            raise ValueError("urgency must be normal or urgent")
        if self.max_source_age_ms is not None:
            age = number(self.max_source_age_ms, "max_source_age_ms")
            if age == 0:
                raise ValueError("max_source_age_ms must be positive")
            object.__setattr__(self, "max_source_age_ms", age)
        values = dict(self.metrics)
        for key, value in values.items():
            text(key, "metric name")
            values[key] = number(value, key, minimum=-math.inf)
        object.__setattr__(self, "metrics", MappingProxyType(values))

    @property
    def request_detection(self) -> bool:
        """Return the detector request flag for blueprint compatibility."""
        return self.should_detect

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of decision metadata."""
        return {
            "should_detect": self.should_detect,
            "reason": self.reason,
            "confidence": self.confidence,
            "urgency": self.urgency,
            "max_source_age_ms": self.max_source_age_ms,
            "metrics": dict(self.metrics),
        }


ScheduleDecision = TriggerDecision


class EventKind(str, Enum):
    """Lifecycle events separating request admission from observation fusion."""

    ADMITTED = "admitted"
    REPLACED = "replaced"
    REJECTED = "rejected"
    STARTED = "started"
    COMPLETED = "completed"
    FUSED = "fused"
    STALE_DISCARDED = "stale_discarded"
    FAILED = "failed"
    SHUTDOWN_DISCARDED = "shutdown_discarded"


@dataclass(frozen=True)
class SchedulerEvent:
    """Source-identified event updating scheduler history."""

    kind: EventKind
    frame_idx: int
    epoch: str = "default"
    source_time_ns: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EventKind(self.kind))
        object.__setattr__(self, "frame_idx", integer(self.frame_idx, "frame_idx", 1))
        text(self.epoch, "epoch")
        if self.source_time_ns is not None:
            object.__setattr__(self, "source_time_ns", integer(self.source_time_ns, "source_time_ns"))


class BaseScheduler(ABC):
    """Single-owner policy. Decisions express intent; runtime admission stays external."""

    def __init__(self) -> None:
        """Initialize validated configuration and owned runtime state."""
        self._stream = StreamInfo()
        self._last_fused_frame = 0
        self._last_fused_time_ns: int | None = None
        self._last_admitted_frame = 0

    @property
    def last_fused_frame(self) -> int:
        """Return the newest fused source-frame identifier."""
        return self._last_fused_frame

    def reset(self, stream_info: StreamInfo | None = None) -> None:
        """Reset policy history for a new source stream."""
        info = StreamInfo() if stream_info is None else stream_info
        if not isinstance(info, StreamInfo):
            raise TypeError("stream_info must be StreamInfo")
        self._stream = info
        self._last_fused_frame = self._last_admitted_frame = 0
        self._last_fused_time_ns = None

    def should_detect(self, frame_idx: int, tracker_state: TrackerState) -> TriggerDecision:
        """Evaluate detector triggering for a source frame and tracker snapshot."""
        return self.decide(
            ScheduleContext(frame_idx, tracker_state, epoch=self._stream.epoch, offline=self._stream.offline)
        )

    def validate_context(self, context: ScheduleContext) -> None:
        """Reject invalid context types, epochs and source histories."""
        if not isinstance(context, ScheduleContext):
            raise TypeError("context must be ScheduleContext")
        if context.epoch != self._stream.epoch:
            raise ValueError("Context belongs to another stream epoch; reset the scheduler")
        if self.fused_frame(context) > context.frame_idx:
            raise ValueError("Scheduler fusion history is ahead of the source frame")
        if self._last_admitted_frame > context.frame_idx:
            raise ValueError("Scheduler admission history is ahead of the source frame")
        if (
            self._last_fused_time_ns is not None
            and context.source_time_ns is not None
            and self._last_fused_time_ns > context.source_time_ns
        ):
            raise ValueError("Scheduler fused source timestamp cannot be in the future")

    def fused_frame(self, context: ScheduleContext) -> int:
        """Return the latest fused source frame from policy and runtime history."""
        return max(self._last_fused_frame, context.tracker_state.last_fused_frame)

    def admitted_frame(self, context: ScheduleContext) -> int:
        """Return the latest admitted detector request frame."""
        return max(self._last_admitted_frame, context.last_admitted_frame or 0)

    def source_age_ms(self, context: ScheduleContext) -> float:
        """Compute fused-observation age in the source clock domain."""
        state = context.tracker_state
        if self._last_fused_frame > state.last_fused_frame:
            last = self._last_fused_time_ns
        elif state.last_fused_frame > self._last_fused_frame:
            last = state.last_fused_time_ns
        else:
            known = (self._last_fused_time_ns, state.last_fused_time_ns)
            if all(value is not None for value in known) and known[0] != known[1]:
                raise ValueError("Conflicting source timestamps for the same fused frame")
            last = known[0] if known[0] is not None else known[1]
        if last is None or context.source_time_ns is None:
            raise MissingSignalError("A source-age guard requires current and fused source timestamps")
        if last > context.source_time_ns:
            raise ValueError("Fused source timestamp cannot be in the future")
        return (context.source_time_ns - last) / 1_000_000

    def on_event(self, event: SchedulerEvent) -> None:
        """Update scheduler history from an admitted or fused observation."""
        if not isinstance(event, SchedulerEvent):
            raise TypeError("event must be SchedulerEvent")
        if event.epoch != self._stream.epoch:
            raise ValueError("Event belongs to another stream epoch")
        if event.kind is EventKind.ADMITTED:
            self._last_admitted_frame = max(self._last_admitted_frame, event.frame_idx)
        elif event.kind is EventKind.FUSED and event.frame_idx > self._last_fused_frame:
            self._last_fused_frame = event.frame_idx
            self._last_fused_time_ns = event.source_time_ns

    @abstractmethod
    def decide(self, context: ScheduleContext) -> TriggerDecision:
        """Return a trigger decision from the validated source-frame context."""
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict[str, object]:
        """Return serializable policy configuration and reference semantics."""
        raise NotImplementedError
