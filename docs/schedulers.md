# Pluggable detection schedulers

> Current scope (2026-10-18): All four policies are implemented. The current oracle sweep default is an every-frame OraclePolicy reference with no optimality claim. A single FixedKPolicy(3) timing run exists, while the real four-policy accuracy sweep remains pending. See [verified performance](verified-performance.md).

`src/schedulers/` implements policy decisions independently of detector execution, queues, tracker mutation, or clock advancement. These modules do not activate the async runtime or change `src/pipeline.py`. The policy registry is an immutable mapping with `fixed_k`, `kalman`, `heuristic`, and `oracle` entries.

## Shared contract

```python
from src.schedulers import (
    AdaptiveHeuristicPolicy, EventKind, FixedKPolicy, SchedulerEvent,
    TrackerState, TrackState,
)

state = TrackerState(
    tracks=(TrackState(confidence=0.9, displacement=0.02),),
    last_fused_frame=1,
)
policy = AdaptiveHeuristicPolicy()
decision = policy.should_detect(2, state)
assert not decision.should_detect
assert decision.reason == "stable"
metadata = decision.as_dict()

fixed = FixedKPolicy(k=3)
assert fixed.should_detect(4, state).should_detect
fixed.on_event(SchedulerEvent(EventKind.ADMITTED, frame_idx=4))
assert fixed.last_fused_frame == 0
fixed.on_event(SchedulerEvent(EventKind.FUSED, frame_idx=4))
assert fixed.last_fused_frame == 4
```

`BaseScheduler` exposes `should_detect(frame_idx, tracker_state) -> TriggerDecision`, abstract `decide(context)` and `describe()`, plus `reset(stream_info)` and `on_event(event)`. `ScheduleDecision` aliases `TriggerDecision` for blueprint compatibility. Source frames are one-based; frame zero is only the sentinel for no prior observation. Frame IDs must retain the original source numbering when input frames are dropped.

`TriggerDecision` contains `should_detect` (also exposed as `request_detection`), reason, optional tracker confidence, urgency, optional maximum useful source age, and immutable numeric metrics. Confidence is the worst visible LK track reliability, when applicable; it is not a probability that detection is needed. Fixed-K, Kalman and oracle decisions leave it `None` rather than fabricate a confidence value. Use `as_dict()` for JSON export and `describe()` to serialize policy configuration.

`TrackerState` contains immutable per-track snapshots and the last **fused source observation**, not detector arrival time or request time. `TrackState` carries optional confidence, normalized displacement, a flow-valid flag, xyxy box, and 2×2 covariance for box-center position in pixel-squared units. Input boxes and covariance arrays are converted to owned immutable tuples. A full eight-state covariance must be reduced explicitly to its center-position block; passing an 8×8 matrix is rejected. Covariance must be finite, symmetric, and positive semidefinite within numerical tolerance.

`ScheduleContext` additionally carries source time, stream epoch, offline mode, worker availability, pending age, detector service estimate, budget headroom, and latest requested/admitted/started source IDs. These are admission and telemetry inputs; a busy worker does not silently change Fixed-K intent. Negative budget headroom represents an overrun. Configuring a time-based age guard requires `decide(context)` with current and fused source timestamps from the same clock domain; `should_detect()` deliberately supplies no fabricated time.

Schedulers have one runtime owner. Call `reset(StreamInfo(epoch=...))` when a sequence restarts. Events and contexts from another epoch raise `ValueError`. Only `fused` events advance the internal fused frame; older events cannot roll it back. Admissions update cooldown history separately. The effective fused observation is the newer of runtime `TrackerState` and recorded fusion events. Conflicting source timestamps for the same observation are errors. `replaced` marks the superseded request; submit a separate `admitted` event for its accepted replacement.

## Policies and exact thresholds

| Policy | Behavior |
| --- | --- |
| `FixedKPolicy(k)` | Requests on `(frame_idx - 1) % k == 0`; the first source frame is included. Empty tracks and fusion events do not shift cadence. |
| `KalmanUncertaintyPolicy(threshold, ...)` | Requests with no tracks, or when the maximum normalized position variance strictly exceeds `threshold`. Optional frame/time age guards bound observation staleness. |
| `AdaptiveHeuristicPolicy(...)` | Defaults to no tracks → interval expiry → flow failure → confidence → displacement. Confidence uses strict `<`; displacement uses strict `>`; interval expiry uses `>=`. Optional source-age expiry takes precedence. |
| `OraclePolicy(schedule, offline=True, ...)` | Replays an explicit finite source-frame schedule, with no opportunistic triggers. Live-mode reset/context is rejected. |

Kalman uncertainty is:

\[
U = \max_i \frac{\lambda_{\max}(P_i^{xy})}
{\max(1,(x_{2,i}-x_{1,i})^2+(y_{2,i}-y_{1,i})^2)}.
\]

The scheduler reads covariance; it does not propagate filters, choose `dt`, calibrate noise, or apply delayed measurements. Those remain tracker/fusion responsibilities. The existing `KalmanBox` uses frame units for velocity; adopting elapsed seconds requires an explicit tracker change. Missing boxes/covariance on visible tracks raise `MissingSignalError`, even if an age guard is due. Empty-track bootstrap does not establish that a subsequently supplied tracker has covariance capability.

The heuristic aggregates minimum confidence and maximum displacement across supplied visible tracks. Default recovery thresholds equal trigger thresholds and cooldown is zero, preserving legacy behavior for valid LK signals. To enable hysteresis, choose `recovery_confidence >= min_confidence` and `recovery_displacement <= max_displacement`. Trigger conditions remain latched until recovery thresholds are reached. A cooldown begins only after admission, not a rejected request or a decision alone; bootstrap, flow failure, and age/interval expiry bypass cooldown. Tracker adapters must provide finite signals and explicitly mark flow failure rather than pass NaNs as confidence.

Example covariance and offline reference:

```python
from src.schedulers import KalmanUncertaintyPolicy, OraclePolicy, TrackerState, TrackState

state = TrackerState(
    tracks=(TrackState(box=(0, 0, 20, 10),
                       position_covariance=((60, 0), (0, 60))),),
    last_fused_frame=1,
)
assert KalmanUncertaintyPolicy(0.1).should_detect(2, state).reason == "uncertainty"

oracle = OraclePolicy([1, 4, 7], offline=True, inference_budget=3,
                      provenance="ground-truth hindsight schedule from offline replay")
assert oracle.should_detect(4, TrackerState()).should_detect
```

The oracle accepts a schedule produced by external ground-truth/hindsight analysis. It does not compute an optimal schedule, consume ground-truth boxes, or prove a global upper bound. Duplicate/invalid frames and schedules exceeding the declared budget are rejected. Report its provenance, candidate-search scope and future/GT access. The runtime must apply the same queue/service/fusion delays to this schedule for a latency-comparable reference; scheduled requests can still be dropped by runtime admission.

## Validation and research instrumentation

Invalid configuration/state values raise `ValueError`; wrong record types raise `TypeError`; required absent tracker signals raise `MissingSignalError` (a `ValueError` subclass). No policy silently switches to another strategy. Frozen records and immutable decision metrics prevent policy code from changing caller-owned state.

Unit tests cover source cadence and gaps, legacy trigger precedence and boundaries, delayed/rejected observations, cooldown/hysteresis, stream reset, covariance growth/recovery and scale normalization, oracle restrictions, and malformed inputs. Run:

```bash
.venv/bin/python -m pytest -q tests/test_schedulers.py
```

At runtime, time `decide()` externally with `perf_counter`, log the decision metadata alongside admission/fusion events, and retain per-frame latency, CPU/RSS and offline HOTA/MOTA instrumentation. Policies perform no resource sampling or I/O inside decisions. Unit tests establish decision correctness, not FPS, deadline performance, inference savings, or tracking accuracy.
