import numpy as np

from .base import BaseScheduler, MissingSignalError, ScheduleContext, TriggerDecision, integer, number


class KalmanUncertaintyPolicy(BaseScheduler):
    """Trigger detection when normalized box-center covariance exceeds its limit."""

    def __init__(
        self, threshold: float, max_interval: int | None = 30, max_source_age_ms: float | None = None
    ) -> None:
        """Initialize validated configuration and owned runtime state."""
        super().__init__()
        self.threshold = number(threshold, "threshold")
        if self.threshold == 0:
            raise ValueError("threshold must be positive")
        self.max_interval = None if max_interval is None else integer(max_interval, "max_interval", 1)
        self.max_source_age_ms = (
            None if max_source_age_ms is None else number(max_source_age_ms, "max_source_age_ms")
        )
        if self.max_source_age_ms == 0:
            raise ValueError("max_source_age_ms must be positive")

    def decide(self, context: ScheduleContext) -> TriggerDecision:
        """Return a trigger decision from the validated source-frame context."""
        self.validate_context(context)
        if not context.tracker_state.tracks:
            return TriggerDecision(
                True, "no_tracks", urgency="urgent", max_source_age_ms=self.max_source_age_ms
            )
        uncertainties = []
        for track in context.tracker_state.tracks:
            if track.box is None or track.position_covariance is None:
                raise MissingSignalError(
                    "Kalman policy requires xyxy boxes and 2x2 position covariance for every track"
                )
            x1, y1, x2, y2 = track.box
            diagonal_squared = max(1.0, (x2 - x1) ** 2 + (y2 - y1) ** 2)
            variance = max(0.0, float(np.linalg.eigvalsh(track.position_covariance)[-1]))
            uncertainties.append(variance / diagonal_squared)
        uncertainty = max(uncertainties)
        metrics = {"normalized_uncertainty": uncertainty}
        reason = "uncertainty" if uncertainty > self.threshold else "stable"
        if (
            self.max_interval is not None
            and context.frame_idx - self.fused_frame(context) >= self.max_interval
        ):
            reason = "max_interval"
        if self.max_source_age_ms is not None:
            age = self.source_age_ms(context)
            metrics["source_age_ms"] = age
            if age >= self.max_source_age_ms:
                reason = "source_age"
        due = reason != "stable"
        return TriggerDecision(
            due,
            reason,
            urgency="urgent" if due else "normal",
            max_source_age_ms=self.max_source_age_ms,
            metrics=metrics,
        )

    def describe(self) -> dict[str, object]:
        """Return serializable policy configuration and reference semantics."""
        return {
            "policy": "kalman",
            "threshold": self.threshold,
            "max_interval": self.max_interval,
            "max_source_age_ms": self.max_source_age_ms,
            "uncertainty": "max(lambda_max(P_xy) / max(1, box_diagonal_squared))",
        }
