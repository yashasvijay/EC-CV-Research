from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from src.schedulers import (
    AdaptiveHeuristicPolicy,
    BaseScheduler,
    EventKind,
    FixedKPolicy,
    KalmanUncertaintyPolicy,
    MissingSignalError,
    OraclePolicy,
    POLICIES,
    ScheduleContext,
    SchedulerEvent,
    StreamInfo,
    TrackerState,
    TrackState,
    TriggerDecision,
)
from src.sort_tracker import KalmanBox


def state(confidence=0.9, displacement=0.01, last=1, **kwargs):
    return TrackerState((TrackState(confidence, displacement, **kwargs),), last_fused_frame=last)


@pytest.mark.parametrize("k,expected", [(1, list(range(1, 10))), (2, [1, 3, 5, 7, 9]), (3, [1, 4, 7])])
def test_fixed_source_cadence(k, expected):
    policy = FixedKPolicy(k)
    assert [i for i in range(1, 10) if policy.should_detect(i, TrackerState()).should_detect] == expected
    assert policy.should_detect(2, TrackerState()).should_detect == (k == 1)


def test_fixed_ignores_health_worker_and_fusion_events():
    policy = FixedKPolicy(3)
    policy.on_event(SchedulerEvent(EventKind.FUSED, 2))
    decisions = [
        policy.decide(ScheduleContext(i, TrackerState(), worker_available=False)) for i in [3, 4, 6, 7]
    ]
    assert [d.request_detection for d in decisions] == [False, True, False, True]


def test_heuristic_matches_existing_trigger_precedence():
    policy = AdaptiveHeuristicPolicy(max_interval=3)
    last = 0
    sequence = [
        (),
        (TrackState(0.6, 0.15),),
        (TrackState(0.59, 0.2),),
        (TrackState(0.8, 0.16),),
        (TrackState(0.9, 0.01),),
        (TrackState(0.9, 0.01),),
        (TrackState(0.2, 0.5),),
    ]
    for i, tracks in enumerate(sequence, 1):
        confidence = min((t.confidence for t in tracks), default=0)
        displacement = max((t.displacement for t in tracks), default=0)
        expected = (
            "no_tracks"
            if not tracks
            else "max_interval"
            if i - last >= 3
            else "confidence"
            if confidence < 0.6
            else "displacement"
            if displacement > 0.15
            else "stable"
        )
        decision = policy.should_detect(i, TrackerState(tracks, last))
        assert decision.reason == expected
        assert decision.should_detect == (expected != "stable")
        if decision.should_detect:
            policy.on_event(SchedulerEvent(EventKind.FUSED, i))
            last = i


def test_heuristic_worst_track_and_flow_failure():
    policy = AdaptiveHeuristicPolicy()
    tracks = (TrackState(0.99, 0.01), TrackState(0.5, 0.2))
    decision = policy.should_detect(2, TrackerState(tracks, 1))
    assert decision.reason == "confidence" and decision.confidence == 0.5
    assert decision.metrics["max_displacement"] == 0.2
    assert policy.should_detect(3, state(flow_valid=False)).reason == "flow_failure"
    assert policy.should_detect(11, state(flow_valid=False)).reason == "max_interval"


def test_hysteresis_and_reset():
    policy = AdaptiveHeuristicPolicy(max_interval=100, recovery_confidence=0.8, recovery_displacement=0.1)
    values = [(0.5, 0.01), (0.65, 0.01), (0.8, 0.01), (0.9, 0.2), (0.9, 0.12), (0.9, 0.1)]
    reasons = [policy.should_detect(i, state(c, d)).reason for i, (c, d) in enumerate(values, 2)]
    assert reasons == ["confidence", "confidence", "stable", "displacement", "displacement", "stable"]
    policy.should_detect(9, state(0.5))
    policy.reset(StreamInfo(epoch="new"))
    assert policy.should_detect(1, state(0.65)).reason == "stable"
    assert policy.last_fused_frame == 0


def test_cooldown_only_after_admission_and_urgent_bypass():
    policy = AdaptiveHeuristicPolicy(cooldown_frames=3)
    assert policy.should_detect(2, state(0.5)).should_detect
    policy.on_event(SchedulerEvent(EventKind.REJECTED, 2))
    assert policy.should_detect(3, state(0.5)).should_detect
    policy.on_event(SchedulerEvent(EventKind.ADMITTED, 3))
    assert policy.should_detect(4, state(0.5)).reason == "cooldown"
    assert policy.should_detect(5, state(0.5, flow_valid=False)).reason == "flow_failure"
    assert policy.should_detect(6, state(0.5)).reason == "confidence"
    assert policy.should_detect(11, state(0.5)).reason == "max_interval"


def test_only_fusion_advances_observation_history():
    policy = AdaptiveHeuristicPolicy(max_interval=3)
    policy.should_detect(3, state(last=0))
    for kind in EventKind:
        if kind is EventKind.FUSED:
            continue
        policy.on_event(SchedulerEvent(kind, 3))
        assert policy.last_fused_frame == 0
    assert policy.should_detect(4, state(last=0)).reason == "max_interval"
    policy.on_event(SchedulerEvent(EventKind.FUSED, 3))
    assert policy.should_detect(4, state(last=0)).reason == "stable"
    policy.on_event(SchedulerEvent(EventKind.FUSED, 2))
    assert policy.last_fused_frame == 3
    assert policy.should_detect(6, state(last=0)).reason == "max_interval"


def test_kalman_prediction_growth_and_update_recovery():
    policy = KalmanUncertaintyPolicy(0.1, max_interval=None)
    filter_ = KalmanBox([0, 0, 20, 10])

    def snapshot():
        return TrackerState((TrackState(box=tuple(filter_.box), position_covariance=filter_.P[:2, :2]),), 1)

    assert policy.should_detect(1, snapshot()).reason == "stable"
    filter_.predict()
    decision = policy.should_detect(2, snapshot())
    assert decision.reason == "uncertainty" and decision.metrics["normalized_uncertainty"] > 0.1
    filter_.update([0, 0, 20, 10])
    assert policy.should_detect(3, snapshot()).reason == "stable"


def test_uncertainty_eigenvalue_scale_invariance_and_boundary():
    policy = KalmanUncertaintyPolicy(0.1, max_interval=None)
    for scale in [1, 2, 10]:
        covariance = np.array([[30.0, 20.0], [20.0, 30.0]]) * scale**2
        track = TrackState(box=(0, 0, 20 * scale, 10 * scale), position_covariance=covariance)
        decision = policy.should_detect(2, TrackerState((track,), 1))
        assert decision.metrics["normalized_uncertainty"] == pytest.approx(0.1)
        assert not decision.should_detect
    worst = TrackState(box=(0, 0, 10, 10), position_covariance=((40, 0), (0, 40)))
    assert policy.should_detect(3, TrackerState((track, worst), 1)).should_detect


def test_covariance_capability_bootstrap_and_interval_guard():
    policy = KalmanUncertaintyPolicy(0.1, max_interval=3)
    assert policy.should_detect(1, TrackerState()).reason == "no_tracks"
    with pytest.raises(MissingSignalError):
        policy.should_detect(4, state())
    track = TrackState(box=(0, 0, 20, 10), position_covariance=((1, 0), (0, 1)))
    assert policy.should_detect(4, TrackerState((track,), 1)).reason == "max_interval"


@pytest.mark.parametrize(
    "policy",
    [AdaptiveHeuristicPolicy(max_source_age_ms=50), KalmanUncertaintyPolicy(0.1, max_source_age_ms=50)],
)
def test_source_age_guard_uses_source_observation_time(policy):
    track = TrackState(0.9, 0.01, box=(0, 0, 20, 10), position_covariance=((1, 0), (0, 1)))
    tracker = TrackerState((track,), 1, 1_000_000_000)
    assert policy.decide(ScheduleContext(2, tracker, source_time_ns=1_049_000_000)).reason == "stable"
    assert policy.decide(ScheduleContext(3, tracker, source_time_ns=1_050_000_000)).reason == "source_age"
    policy.on_event(SchedulerEvent(EventKind.FUSED, 3, source_time_ns=1_050_000_000))
    assert (
        policy.decide(ScheduleContext(4, TrackerState((track,), 3), source_time_ns=1_051_000_000)).reason
        == "stable"
    )
    with pytest.raises(MissingSignalError):
        policy.should_detect(4, TrackerState((track,), 3))


def test_context_admission_history_and_conflicting_source_times():
    policy = AdaptiveHeuristicPolicy(cooldown_frames=3, max_source_age_ms=50)
    tracker = TrackerState((TrackState(0.5, 0.01),), 1, 1_000_000_000)
    context = ScheduleContext(
        3,
        tracker,
        source_time_ns=1_001_000_000,
        last_requested_frame=2,
        last_admitted_frame=2,
        last_started_frame=2,
    )
    assert policy.decide(context).reason == "cooldown"
    policy.on_event(SchedulerEvent(EventKind.FUSED, 1, source_time_ns=1_002_000_000))
    with pytest.raises(ValueError, match="Conflicting"):
        policy.decide(ScheduleContext(3, tracker, source_time_ns=1_003_000_000))
    with pytest.raises(ValueError):
        ScheduleContext(3, tracker, last_admitted_frame=4)
    with pytest.raises(ValueError):
        policy.decide(ScheduleContext(3, TrackerState(tracker.tracks, 1), source_time_ns=999_000_000))


def test_oracle_schedule_is_offline_budgeted_and_not_opportunistic():
    policy = OraclePolicy([7, 1, 4], offline=True, inference_budget=3, provenance="GT hindsight replay")
    assert [i for i in range(1, 9) if policy.should_detect(i, TrackerState()).should_detect] == [1, 4, 7]
    assert not policy.should_detect(2, TrackerState()).should_detect
    assert policy.describe()["schedule"] == [1, 4, 7]
    assert "no global" in policy.describe()["optimality"]
    with pytest.raises(ValueError):
        policy.reset(StreamInfo(offline=False))
    with pytest.raises(ValueError):
        policy.decide(ScheduleContext(1, TrackerState(), offline=False))


def test_empty_oracle_and_source_gaps_preserve_schedule():
    assert not OraclePolicy([], offline=True).should_detect(1, TrackerState()).should_detect
    policy = OraclePolicy([1, 4, 7], offline=True)
    assert [i for i in [1, 3, 5, 7] if policy.should_detect(i, TrackerState()).should_detect] == [1, 7]
    fixed = FixedKPolicy(3)
    assert [i for i in [1, 3, 5, 7] if fixed.should_detect(i, TrackerState()).should_detect] == [1, 7]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FixedKPolicy(0),
        lambda: FixedKPolicy(True),
        lambda: FixedKPolicy(1.5),
        lambda: KalmanUncertaintyPolicy(0),
        lambda: KalmanUncertaintyPolicy(float("nan")),
        lambda: KalmanUncertaintyPolicy(0.1, max_interval=0),
        lambda: AdaptiveHeuristicPolicy(max_interval=0),
        lambda: AdaptiveHeuristicPolicy(min_confidence=1.1),
        lambda: AdaptiveHeuristicPolicy(max_displacement=0),
        lambda: AdaptiveHeuristicPolicy(cooldown_frames=-1),
        lambda: AdaptiveHeuristicPolicy(recovery_confidence=0.5),
        lambda: AdaptiveHeuristicPolicy(recovery_displacement=0.2),
        lambda: AdaptiveHeuristicPolicy(max_source_age_ms=0),
        lambda: OraclePolicy([1]),
        lambda: OraclePolicy([1, 1], offline=True),
        lambda: OraclePolicy([0], offline=True),
        lambda: OraclePolicy([1, 2], offline=True, inference_budget=1),
    ],
)
def test_invalid_policy_configuration(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("covariance", [np.eye(8), [[1, 2], [0, 1]], [[1, 2], [2, 1]], [[np.nan, 0], [0, 1]]])
def test_invalid_covariance(covariance):
    with pytest.raises(ValueError):
        TrackState(position_covariance=covariance)


def test_snapshot_and_decision_metadata_are_immutable():
    covariance = np.eye(2)
    box = np.array([0.0, 0.0, 20.0, 10.0])
    track = TrackState(box=box, position_covariance=covariance)
    covariance[0, 0] = 100
    box[2] = 100
    assert track.position_covariance[0][0] == 1 and track.box[2] == 20
    metrics = {"normalized_uncertainty": 0.1}
    decision = TriggerDecision(True, "uncertainty", metrics=metrics)
    metrics["normalized_uncertainty"] = 99
    assert decision.metrics["normalized_uncertainty"] == 0.1
    with pytest.raises(TypeError):
        decision.metrics["x"] = 1
    with pytest.raises(FrozenInstanceError):
        track.confidence = 0.1
    json.dumps(decision.as_dict(), allow_nan=False)


def test_contract_errors_registry_and_provenance():
    with pytest.raises(TypeError):
        BaseScheduler()
    policy = FixedKPolicy(2)
    for frame in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            policy.should_detect(frame, TrackerState())
    with pytest.raises(TypeError):
        policy.should_detect(1, {})
    with pytest.raises(ValueError):
        policy.should_detect(2, TrackerState(last_fused_frame=3))
    with pytest.raises(ValueError):
        policy.decide(ScheduleContext(1, TrackerState(), epoch="old"))
    with pytest.raises(ValueError):
        policy.on_event(SchedulerEvent(EventKind.FUSED, 1, epoch="old"))
    with pytest.raises(ValueError):
        ScheduleContext(2, TrackerState(last_fused_frame=1, last_fused_time_ns=20), source_time_ns=10)
    with pytest.raises(MissingSignalError):
        AdaptiveHeuristicPolicy().should_detect(2, TrackerState((TrackState(),), 1))
    assert ScheduleContext(1, TrackerState(), remaining_budget_ms=-1).remaining_budget_ms == -1
    for instance in [
        FixedKPolicy(3),
        KalmanUncertaintyPolicy(0.1),
        AdaptiveHeuristicPolicy(),
        OraclePolicy([], offline=True),
    ]:
        assert isinstance(instance, BaseScheduler)
        assert POLICIES[instance.describe()["policy"]] is type(instance)
        json.dumps(instance.describe(), allow_nan=False)
