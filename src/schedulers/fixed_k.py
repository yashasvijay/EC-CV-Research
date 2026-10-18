from .base import BaseScheduler, ScheduleContext, TriggerDecision, integer


class FixedKPolicy(BaseScheduler):
    """Request detection on a fixed cadence anchored to source frame one."""

    def __init__(self, k: int) -> None:
        """Initialize validated configuration and owned runtime state."""
        super().__init__()
        self.k = integer(k, "k", 1)

    def decide(self, context: ScheduleContext) -> TriggerDecision:
        """Return a trigger decision from the validated source-frame context."""
        self.validate_context(context)
        due = (context.frame_idx - 1) % self.k == 0
        return TriggerDecision(due, "fixed_interval" if due else "interval_not_due")

    def describe(self) -> dict[str, object]:
        """Return serializable policy configuration and reference semantics."""
        return {"policy": "fixed_k", "k": self.k, "frame_indexing": "one_based_source"}
