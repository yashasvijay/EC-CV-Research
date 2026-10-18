from typing import Iterable

from .base import BaseScheduler, ScheduleContext, StreamInfo, TriggerDecision, integer


class OraclePolicy(BaseScheduler):
    """Replay a supplied hindsight schedule; no global optimality is inferred."""

    def __init__(
        self,
        schedule: Iterable[int],
        *,
        offline: bool = False,
        provenance: str = "precomputed_schedule",
        inference_budget: int | None = None,
    ) -> None:
        """Initialize validated configuration and owned runtime state."""
        super().__init__()
        if offline is not True:
            raise ValueError("OraclePolicy requires explicit offline=True")
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("Schedule provenance must be a nonempty string")
        frames = tuple(integer(f, "schedule frame", 1) for f in schedule)
        if len(set(frames)) != len(frames):
            raise ValueError("Oracle schedule contains duplicate source frames")
        self.schedule = tuple(sorted(frames))
        self._scheduled = frozenset(frames)
        self.provenance = provenance
        self.inference_budget = (
            None if inference_budget is None else integer(inference_budget, "inference_budget")
        )
        if self.inference_budget is not None and len(frames) > self.inference_budget:
            raise ValueError("Oracle schedule exceeds the declared inference budget")
        self.reset(StreamInfo(offline=True))

    def reset(self, stream_info: StreamInfo | None = None) -> None:
        """Reset policy history for a new source stream."""
        info = StreamInfo(offline=True) if stream_info is None else stream_info
        if not isinstance(info, StreamInfo):
            raise TypeError("stream_info must be StreamInfo")
        if not info.offline:
            raise ValueError("OraclePolicy cannot be configured for a live stream")
        super().reset(info)

    def decide(self, context: ScheduleContext) -> TriggerDecision:
        """Return a trigger decision from the validated source-frame context."""
        self.validate_context(context)
        if not context.offline:
            raise ValueError("OraclePolicy requires an offline evaluation context")
        due = context.frame_idx in self._scheduled
        return TriggerDecision(due, "oracle_schedule" if due else "oracle_not_scheduled")

    def describe(self) -> dict[str, object]:
        """Return serializable policy configuration and reference semantics."""
        return {
            "policy": "oracle",
            "offline_only": True,
            "noncausal_reference": True,
            "schedule": list(self.schedule),
            "provenance": self.provenance,
            "inference_budget": self.inference_budget,
            "optimality": "supplied schedule; no global upper-bound proof",
        }
