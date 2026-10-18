from types import MappingProxyType
from typing import Mapping

from .base import (
    BaseScheduler,
    EventKind,
    MissingSignalError,
    ScheduleContext,
    ScheduleDecision,
    SchedulerEvent,
    StreamInfo,
    TrackerState,
    TrackState,
    TriggerDecision,
)
from .fixed_k import FixedKPolicy
from .heuristic import AdaptiveHeuristicPolicy
from .kalman import KalmanUncertaintyPolicy
from .oracle import OraclePolicy

POLICIES: Mapping[str, type[BaseScheduler]] = MappingProxyType(
    {
        "fixed_k": FixedKPolicy,
        "kalman": KalmanUncertaintyPolicy,
        "heuristic": AdaptiveHeuristicPolicy,
        "oracle": OraclePolicy,
    }
)

__all__ = [
    "BaseScheduler",
    "EventKind",
    "MissingSignalError",
    "ScheduleContext",
    "ScheduleDecision",
    "SchedulerEvent",
    "StreamInfo",
    "TrackerState",
    "TrackState",
    "TriggerDecision",
    "FixedKPolicy",
    "AdaptiveHeuristicPolicy",
    "KalmanUncertaintyPolicy",
    "OraclePolicy",
    "POLICIES",
]
