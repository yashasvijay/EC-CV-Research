from typing import cast

from .base import (
    BaseScheduler,
    MissingSignalError,
    ScheduleContext,
    StreamInfo,
    TriggerDecision,
    integer,
    number,
)


class AdaptiveHeuristicPolicy(BaseScheduler):
    """Trigger detection from flow failure, confidence drift and displacement."""

    def __init__(
        self,
        max_interval: int = 10,
        min_confidence: float = 0.6,
        max_displacement: float = 0.15,
        cooldown_frames: int = 0,
        recovery_confidence: float | None = None,
        recovery_displacement: float | None = None,
        max_source_age_ms: float | None = None,
    ) -> None:
        """Initialize validated configuration and owned runtime state."""
        super().__init__()
        self.max_interval = integer(max_interval, "max_interval", 1)
        self.min_confidence = number(min_confidence, "min_confidence", maximum=1)
        self.max_displacement = number(max_displacement, "max_displacement")
        if self.max_displacement == 0:
            raise ValueError("max_displacement must be positive")
        self.cooldown_frames = integer(cooldown_frames, "cooldown_frames")
        self.recovery_confidence = (
            self.min_confidence
            if recovery_confidence is None
            else number(recovery_confidence, "recovery_confidence", maximum=1)
        )
        self.recovery_displacement = (
            self.max_displacement
            if recovery_displacement is None
            else number(recovery_displacement, "recovery_displacement")
        )
        if (
            self.recovery_confidence < self.min_confidence
            or self.recovery_displacement > self.max_displacement
        ):
            raise ValueError("Recovery thresholds must lie on the stable side of trigger thresholds")
        self.max_source_age_ms = (
            None if max_source_age_ms is None else number(max_source_age_ms, "max_source_age_ms")
        )
        if self.max_source_age_ms == 0:
            raise ValueError("max_source_age_ms must be positive")
        self._confidence_latched = self._displacement_latched = False

    def reset(self, stream_info: StreamInfo | None = None) -> None:
        """Reset policy history for a new source stream."""
        super().reset(stream_info)
        self._confidence_latched = self._displacement_latched = False

    def decide(self, context: ScheduleContext) -> TriggerDecision:
        """Return a trigger decision from the validated source-frame context."""
        self.validate_context(context)
        tracks = context.tracker_state.tracks
        if not tracks:
            return TriggerDecision(
                True, "no_tracks", confidence=0.0, urgency="urgent", max_source_age_ms=self.max_source_age_ms
            )
        if any(t.confidence is None or t.displacement is None for t in tracks):
            raise MissingSignalError(
                "Heuristic policy requires confidence and normalized displacement for every track"
            )
        confidence = min(cast(float, t.confidence) for t in tracks)
        displacement = max(cast(float, t.displacement) for t in tracks)
        metrics = {"min_confidence": confidence, "max_displacement": displacement}
        reason = "stable"
        if context.frame_idx - self.fused_frame(context) >= self.max_interval:
            reason = "max_interval"
        elif any(not t.flow_valid for t in tracks):
            reason = "flow_failure"
        if self.max_source_age_ms is not None:
            age = self.source_age_ms(context)
            metrics["source_age_ms"] = age
            if age >= self.max_source_age_ms:
                reason = "source_age"
        if confidence < self.min_confidence:
            self._confidence_latched = True
        elif confidence >= self.recovery_confidence:
            self._confidence_latched = False
        if displacement > self.max_displacement:
            self._displacement_latched = True
        elif displacement <= self.recovery_displacement:
            self._displacement_latched = False
        if reason == "stable":
            if self._confidence_latched:
                reason = "confidence"
            elif self._displacement_latched:
                reason = "displacement"
            admitted = self.admitted_frame(context)
            if reason != "stable" and admitted > 0:
                if context.frame_idx - admitted < self.cooldown_frames:
                    reason = "cooldown"
        due = reason not in ("stable", "cooldown")
        urgent = reason in ("no_tracks", "flow_failure", "max_interval", "source_age")
        return TriggerDecision(
            due,
            reason,
            confidence=confidence,
            urgency="urgent" if urgent else "normal",
            max_source_age_ms=self.max_source_age_ms,
            metrics=metrics,
        )

    def describe(self) -> dict[str, object]:
        """Return serializable policy configuration and reference semantics."""
        return {
            "policy": "heuristic",
            "max_interval": self.max_interval,
            "min_confidence": self.min_confidence,
            "max_displacement": self.max_displacement,
            "cooldown_frames": self.cooldown_frames,
            "recovery_confidence": self.recovery_confidence,
            "recovery_displacement": self.recovery_displacement,
            "max_source_age_ms": self.max_source_age_ms,
            "aggregation": "worst_visible_track",
        }
