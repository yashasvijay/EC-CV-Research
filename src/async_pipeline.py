"""Single-owner LK runtime with bounded asynchronous detection and delayed fusion."""

from __future__ import annotations
from collections import Counter, deque
from dataclasses import asdict
import json
from pathlib import Path
import queue
import threading
from time import perf_counter_ns
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping
import cv2
import numpy as np
import psutil
from .schedulers import (
    BaseScheduler,
    EventKind,
    KalmanUncertaintyPolicy,
    ScheduleContext,
    SchedulerEvent,
    StreamInfo,
    TrackerState,
    TrackState,
)
from .schedulers.base import integer, number
from .sort_tracker import KalmanBox, pairwise_iou
from .tracker import LKTracker, Track
from .runtime.packets import (
    Clock,
    Detector,
    DetectionRequest,
    DetectionResult,
    FrameOutput,
    OutputTrack,
    RuntimeConfig,
    ShutdownReport,
    _Correction,
    _Anchor,
    _Frame,
)
from .runtime.history import _History
from .runtime.motion import _owned, _advance_points
from .runtime.mailbox import _offer_latest
from .runtime.detector_worker import DetectionWorker, DetectionWorkerError


def _stats(values: Iterable[float]) -> dict[str, float] | None:
    values = list(values)
    if not values:
        return None
    return {
        "mean": float(np.mean(values)),
        **{f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)},
    }


class AsyncTrackerRuntime:
    """One caller owns live LK tracks; inference and historical replay run off-thread."""

    def __init__(
        self,
        detector_factory: Callable[[], Detector],
        scheduler: BaseScheduler,
        config: RuntimeConfig | None = None,
        *,
        clock: Clock = perf_counter_ns,
        metadata: Mapping[str, object] | None = None,
    ):
        """Initialize validated configuration and owned runtime state."""
        if not callable(detector_factory) or not isinstance(scheduler, BaseScheduler):
            raise TypeError("Provide a detector factory and a BaseScheduler")
        self.config = RuntimeConfig() if config is None else config
        if not isinstance(self.config, RuntimeConfig):
            raise TypeError("config must be RuntimeConfig")
        if isinstance(scheduler, KalmanUncertaintyPolicy) and not self.config.covariance:
            raise ValueError("Kalman policy requires covariance=True")
        self.scheduler, self.clock = scheduler, clock
        self.metadata = json.loads(json.dumps(dict(metadata or {}), allow_nan=False))
        self.requests: queue.Queue[DetectionRequest] = queue.Queue(maxsize=self.config.queue_capacity)
        self.results: queue.Queue[DetectionResult] = queue.Queue(maxsize=self.config.queue_capacity)
        self.history = _History(self.config)
        self.worker = DetectionWorker(
            detector_factory, self.requests, self.results, self.history, self.config, clock
        )
        self.tracker = LKTracker(
            min_points=self.config.min_points,
            max_points=self.config.max_points,
            fb_threshold=self.config.fb_threshold,
            match_iou=self.config.match_iou,
        )
        self.filters: dict[int, KalmanBox] = {}
        self.revisions: dict[int, int] = {}
        self.anchors: dict[int, tuple[int, int]] = {}
        self.state = "created"
        self.owner: int | None = None
        self.last_frame = self.first_frame = 0
        self.last_source_ns: int | None = None
        self.last_fused_frame = 0
        self.last_fused_source_ns: int | None = None
        self.last_fused_capture_ns: int | None = None
        self.last_requested = self.last_admitted = 0
        self.request_id = 0
        self.shape: tuple[int, ...] | None = None
        self.counts: Counter[str] = Counter()
        self.rows: deque[dict[str, Any]] = deque(maxlen=self.config.telemetry_frames)
        self.outputs: deque[FrameOutput] = deque(maxlen=self.config.telemetry_frames)
        self.events: deque[dict[str, Any]] = deque(maxlen=self.config.telemetry_frames)
        self.started_ns = self.ended_ns = 0
        self.measurement_end_ns = 0
        self.main_error: str | None = None
        self.fusion_costs: deque[float] = deque(maxlen=128)
        self.fusion_estimate_ms = self.config.fusion_headroom_ms
        self.process = psutil.Process()
        self.cpu_started = self.sample_ns = 0.0
        self.resource: dict[str, float | None] = {"process_cpu_percent": None, "rss_mib": None}
        self.sampled_peak_rss = 0.0

    def _assert_owner(self) -> None:
        if self.owner is not None and threading.get_ident() != self.owner:
            raise RuntimeError("Live state must be accessed by the runtime's owning thread")

    def start(self, timeout_s: float = 5.0) -> "AsyncTrackerRuntime":
        """Initialize the worker and wait for readiness outside the frame loop."""
        number(timeout_s, "timeout_s")
        if self.state != "created":
            raise RuntimeError("Runtime instances are single-use; create a new instance for another epoch")
        self.owner = threading.get_ident()
        self.scheduler.reset(
            StreamInfo(
                epoch=self.config.epoch, offline=self.config.offline, frame_rate=self.config.frame_rate
            )
        )
        self.state = "starting"
        self.worker.start()
        if not self.worker.ready.wait(timeout_s):
            self.state = "stopping"
            self.main_error = "TimeoutError: detector initialization timed out"
            self.worker.stop_requested.set()
            raise TimeoutError("Detector initialization timed out; close() must verify worker exit")
        if self.worker.error:
            self.state = "failed"
            self.worker.join(timeout_s)
            raise DetectionWorkerError(self.worker.error)
        self.started_ns = self.clock()
        self.cpu_started = sum(self.process.cpu_times()[:2])
        self.process.cpu_percent(None)
        self.sample_ns = self.started_ns
        self.resource["rss_mib"] = self.process.memory_info().rss / 1024**2
        self.sampled_peak_rss = float(self.process.memory_info().rss / 1024**2)
        self.state = "running"
        return self

    def _event(self, kind: EventKind, frame: int, source_ns: int | None = None) -> None:
        event = SchedulerEvent(kind, frame, epoch=self.config.epoch, source_time_ns=source_ns)
        self.scheduler.on_event(event)
        if len(self.events) == self.events.maxlen:
            self.counts["telemetry_events_evicted"] += 1
        self.events.append({"kind": kind.value, "source_frame": frame, "source_time_ns": source_ns})

    def _pump_events(self) -> None:
        for _ in range(256):
            try:
                event = self.worker.events.get_nowait()
            except queue.Empty:
                break
            self.worker.events.task_done()
            self._event(event.kind, event.frame_idx, event.source_time_ns)

    def health(self) -> dict[str, object]:
        """Return worker lifecycle, queue depth and failure state."""
        return {
            "state": self.state,
            "worker_alive": self.worker.is_alive(),
            "worker_ready": self.worker.ready.is_set(),
            "worker_busy": self.worker.busy.is_set(),
            "worker_error": self.worker.error,
            "request_queue_depth": self.requests.qsize(),
            "main_error": self.main_error,
            "result_queue_depth": self.results.qsize(),
            "history": self.history.usage(),
        }

    def _reject_result(self, result: DetectionResult, reason: str) -> str:
        self.counts[f"result_rejected_{reason}"] += 1
        self._event(EventKind.STALE_DISCARDED, result.source_frame, result.source_time_ns)
        return reason

    def _fuse(self, result: DetectionResult, current: _Frame, deadline: int) -> str:
        if result.epoch != self.config.epoch:
            return self._reject_result(result, "epoch")
        if result.source_frame <= self.last_fused_frame:
            return self._reject_result(result, "older_observation")
        if self.clock() > result.expiry_ns:
            return self._reject_result(result, "expired")
        if result.rejection:
            return self._reject_result(result, result.rejection)
        if current.frame_idx - result.target_frame > self.config.max_commit_frames:
            return self._reject_result(result, "target_lag")
        if (deadline - self.clock()) / 1e6 < self.fusion_estimate_ms:
            return self._reject_result(result, "deadline_headroom")
        path = self.history.lease(result.target_frame, current.frame_idx)
        if path is None:
            return self._reject_result(result, "history_gap")
        live = {t.id: t for t in self.tracker.tracks}
        staged: list[tuple[_Correction, np.ndarray, np.ndarray, float, KalmanBox | None]] = []
        try:
            for correction in result.corrections:
                if self.clock() >= deadline:
                    return self._reject_result(result, "commit_budget")
                track = live.get(correction.track_id) if correction.track_id is not None else None
                if correction.track_id is not None:
                    if track is None or self.revisions[track.id] != correction.revision:
                        self.counts["corrections_rejected_revision"] += 1
                        continue
                box, points, confidence = (
                    correction.detection[:4].copy(),
                    correction.points.copy(),
                    correction.confidence,
                )
                filter_ = None
                if correction.filter_x is not None:
                    filter_ = KalmanBox(box)
                    assert correction.filter_p is not None
                    filter_.x, filter_.P = correction.filter_x.copy(), correction.filter_p.copy()
                for previous, following in zip(path.frames, path.frames[1:]):
                    if self.clock() >= deadline:
                        return self._reject_result(result, "commit_budget")
                    box, points, quality = _advance_points(
                        previous.gray, following.gray, points, box, self.config
                    )
                    confidence *= quality
                    if filter_ is not None:
                        filter_.predict(
                            (following.source_ns - previous.source_ns) / 1e9 * self.config.frame_rate
                        )
                if track is not None:
                    box = track.box + correction.residual
                elif len(points) < self.config.min_points or confidence <= 0:
                    self.counts["unsupported_births"] += 1
                    continue
                box[[0, 2]] = box[[0, 2]].clip(0, current.gray.shape[1])
                box[[1, 3]] = box[[1, 3]].clip(0, current.gray.shape[0])
                if np.any(box[2:] <= box[:2]):
                    self.counts["invalid_corrected_boxes"] += 1
                    continue
                staged.append((correction, box.astype(np.float32), points, confidence, filter_))
            if self.clock() >= deadline:
                return self._reject_result(result, "commit_budget")
            # No live mutation occurs until all preparation/budget checks above finish.
            accepted = 0
            for correction, box, points, confidence, filter_ in staged:
                track = live.get(correction.track_id) if correction.track_id is not None else None
                if track is None:
                    if len(self.tracker.tracks) >= self.config.max_tracks:
                        self.counts["births_capped"] += 1
                        continue
                    existing = self.tracker.tracks
                    if existing:
                        overlaps = pairwise_iou(np.array([t.box for t in existing]), box[None])[:, 0]
                        if any(
                            t.cls == int(correction.detection[5]) and overlaps[i] >= self.config.match_iou
                            for i, t in enumerate(existing)
                        ):
                            self.counts["duplicate_births"] += 1
                            continue
                    track = Track(
                        self.tracker.next_id,
                        box.copy(),
                        float(correction.detection[4]),
                        int(correction.detection[5]),
                    )
                    self.tracker.next_id += 1
                    self.tracker.tracks.append(track)
                    live[track.id] = track
                    self.revisions[track.id] = 0
                track.box, track.score = box.copy(), float(correction.detection[4])
                track.missed = 0
                if len(points) >= self.config.min_points:
                    track.points, track.confidence = points.copy(), confidence
                self.revisions[track.id] += 1
                self.anchors[track.id] = (result.source_frame, result.source_time_ns)
                if filter_ is not None:
                    self.filters[track.id] = filter_
                accepted += 1
            if result.detections.size and not accepted:
                return self._reject_result(result, "unsupported_corrections")
            self.last_fused_frame = result.source_frame
            self.last_fused_source_ns = result.source_time_ns
            self.last_fused_capture_ns = result.source_capture_ns
            self.counts["fused"] += 1
            self._event(EventKind.FUSED, result.source_frame, result.source_time_ns)
            return "fused"
        finally:
            path.release()

    def _retire_old_tracks(self, source_ns: int) -> None:
        retained = []
        for track in self.tracker.tracks:
            anchor = self.anchors.get(track.id)
            if anchor is not None and (source_ns - anchor[1]) / 1e6 <= self.config.max_track_age_ms:
                retained.append(track)
            else:
                self.filters.pop(track.id, None)
                self.revisions.pop(track.id, None)
                self.anchors.pop(track.id, None)
                self.counts["tracks_retired"] += 1
        self.tracker.tracks = retained

    def _tracker_state(self) -> TrackerState:
        tracks = tuple(
            TrackState(
                confidence=float(t.confidence),
                displacement=float(t.displacement),
                flow_valid=len(t.points) >= self.config.min_points and t.confidence > 0,
                box=(float(t.box[0]), float(t.box[1]), float(t.box[2]), float(t.box[3])),
                position_covariance=(
                    (
                        (float(self.filters[t.id].P[0, 0]), float(self.filters[t.id].P[0, 1])),
                        (float(self.filters[t.id].P[1, 0]), float(self.filters[t.id].P[1, 1])),
                    )
                    if t.id in self.filters
                    else None
                ),
            )
            for t in self.tracker.visible()
        )
        return TrackerState(tracks, self.last_fused_frame, self.last_fused_source_ns)

    def _process_frame(
        self,
        frame: np.ndarray,
        *,
        frame_idx: int | None = None,
        capture_time_ns: int | None = None,
        source_time_ns: int | None = None,
    ) -> FrameOutput:
        self._assert_owner()
        if self.state != "running":
            raise RuntimeError("Runtime must be running before frames are submitted")
        if self.worker.error:
            self.state = "failed"
            raise DetectionWorkerError(self.worker.error)
        started = self.clock()
        capture = started if capture_time_ns is None else integer(capture_time_ns, "capture_time_ns")
        source = capture if source_time_ns is None else integer(source_time_ns, "source_time_ns")
        index = self.last_frame + 1 if frame_idx is None else integer(frame_idx, "frame_idx", 1)
        if capture > started or index <= self.last_frame:
            raise ValueError("Capture time cannot be in the future and source IDs must increase")
        if self.last_source_ns is not None and source <= self.last_source_ns:
            raise ValueError("Source timestamps must increase within a stream")
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or min(frame.shape[:2]) < 1
            or frame.nbytes > self.config.max_frame_bytes
        ):
            raise ValueError("Expected a bounded uint8 HxWx3 BGR frame")
        if self.shape is not None and frame.shape != self.shape:
            raise ValueError("Frame dimensions changed; create a new runtime/epoch")
        self.shape = frame.shape
        self._pump_events()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray.flags.writeable = False
        current = _Frame(index, capture, source, gray)
        deadline = capture + int(self.config.deadline_ms * 1e6)
        gap = index - self.last_frame if self.last_frame else 1
        if gap > 1:
            self.counts["source_frames_missing"] += gap - 1
        if gap > self.config.max_frame_gap:
            self.tracker.previous = None
            for track in self.tracker.tracks:
                track.points, track.confidence = np.empty((0, 1, 2), np.float32), 0.0
        self._retire_old_tracks(source)
        before_flow = self.clock()
        self.tracker.predict(gray)
        if self.last_source_ns is not None:
            dt = (source - self.last_source_ns) / 1e9 * self.config.frame_rate
            for filter_ in self.filters.values():
                filter_.predict(dt)
        after_flow = self.clock()
        admitted_history = self.history.admit(current)
        if not admitted_history:
            self.counts["history_capacity_misses"] += 1
        fusion = "none"
        result_age = None
        before_fusion = self.clock()
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            result = None
        if result is not None:
            try:
                result_age = (self.clock() - result.source_capture_ns) / 1e6
                fusion = self._fuse(result, current, deadline)
            finally:
                result.release()
                self.results.task_done()
        after_fusion = self.clock()
        if fusion in ("fused", "commit_budget"):
            self.fusion_costs.append((after_fusion - before_fusion) / 1e6)
            self.fusion_estimate_ms = max(
                self.config.fusion_headroom_ms, float(np.percentile(self.fusion_costs, 95))
            )
        latest_started, latest_service = self.worker.latest()
        context = ScheduleContext(
            index,
            self._tracker_state(),
            source_time_ns=source,
            epoch=self.config.epoch,
            offline=self.config.offline,
            worker_available=not self.worker.busy.is_set(),
            remaining_budget_ms=(deadline - self.clock()) / 1e6,
            last_requested_frame=self.last_requested,
            last_admitted_frame=self.last_admitted,
            last_started_frame=latest_started,
            detector_service_ms=latest_service,
        )
        decision = self.scheduler.decide(context)
        admission = "not_requested"
        if decision.should_detect:
            self.counts["policy_requests"] += 1
            self.last_requested = index
            lease = self.history.lease(index, index) if admitted_history else None
            if lease is None:
                admission = "history_unavailable"
                self._event(EventKind.REJECTED, index, source)
                self.counts["requests_rejected"] += 1
            else:
                try:
                    anchors = tuple(
                        _Anchor(
                            t.id,
                            self.revisions[t.id],
                            tuple(float(v) for v in t.box),
                            t.cls,
                            _owned(self.filters[t.id].x) if t.id in self.filters else None,
                            _owned(self.filters[t.id].P) if t.id in self.filters else None,
                        )
                        for t in self.tracker.tracks
                    )
                    age_limit = min(
                        self.config.max_detection_age_ms,
                        decision.max_source_age_ms or self.config.max_detection_age_ms,
                    )
                    self.request_id += 1
                    request = DetectionRequest(
                        self.request_id,
                        self.config.epoch,
                        current,
                        _owned(frame),
                        anchors,
                        self.clock(),
                        capture + int(age_limit * 1e6),
                        lease,
                    )
                    evicted = _offer_latest(self.requests, request)
                except BaseException:
                    lease.release()
                    raise
                if evicted is not None:
                    evicted.release()
                    self.counts["requests_superseded"] += 1
                    self._event(EventKind.REPLACED, evicted.source.frame_idx, evicted.source.source_ns)
                self.last_admitted = index
                self.counts["requests_admitted"] += 1
                admission = "replaced_pending" if evicted else "admitted"
                self._event(EventKind.ADMITTED, index, source)
        after_schedule = self.clock()
        sample_updated = after_schedule - self.sample_ns >= 250_000_000
        if sample_updated:
            self.resource = {
                "process_cpu_percent": self.process.cpu_percent(None),
                "rss_mib": self.process.memory_info().rss / 1024**2,
            }
            self.sampled_peak_rss = max(
                self.sampled_peak_rss, float(self.process.memory_info().rss / 1024**2)
            )
            self.sample_ns = after_schedule
        output_tracks = tuple(
            OutputTrack(
                t.id, tuple(float(v) for v in t.box), t.score, t.cls, t.confidence, self.anchors[t.id][0]
            )
            for t in self.tracker.visible()
        )
        published = self.clock()
        age = None if self.last_fused_capture_ns is None else (published - self.last_fused_capture_ns) / 1e6
        row: dict[str, object] = {
            "frame": index,
            "source_time_ns": source,
            "capture_ns": capture,
            "published_ns": published,
            "capture_timestamp_provenance": "caller_monotonic"
            if capture_time_ns is not None
            else "host_receipt",
            "flow_ms": (after_flow - before_flow) / 1e6,
            "fusion_ms": (after_fusion - before_fusion) / 1e6,
            "scheduler_admission_ms": (after_schedule - after_fusion) / 1e6,
            "frame_work_ms": (published - started) / 1e6,
            "capture_to_output_ms": (published - capture) / 1e6,
            "deadline_missed": published > deadline,
            "deadline_ms": self.config.deadline_ms,
            "request_queue_depth": self.requests.qsize(),
            "result_queue_depth": self.results.qsize(),
            "admission": admission,
            "trigger": decision.reason,
            "detection_requested": decision.should_detect,
            "confidence": decision.confidence,
            "trigger_urgency": decision.urgency,
            **{f"policy_{key}": value for key, value in decision.metrics.items()},
            "fusion": fusion,
            "result_source_age_ms": result_age,
            "detector_freshness_ms": age,
            "detector_service_ms": (result.completed_ns - result.started_ns) / 1e6 if result else None,
            "detector_prepare_ms": (result.published_ns - result.completed_ns) / 1e6 if result else None,
            "estimated_fusion_ms": self.fusion_estimate_ms,
            "last_fused_source_frame": self.last_fused_frame,
            "tracks": len(output_tracks),
            "source_gap_frames": gap - 1,
            "cpu_sample_updated": sample_updated,
            **self.resource,
        }
        self.counts["frames"] += 1
        self.counts["deadline_misses"] += int(published > deadline)
        if len(self.rows) == self.rows.maxlen:
            self.counts["telemetry_frames_evicted"] += 1
        self.rows.append(row)
        self.last_frame, self.last_source_ns = index, source
        self.first_frame = self.first_frame or index
        self.measurement_end_ns = published
        output = FrameOutput(index, capture, published, output_tracks, MappingProxyType(dict(row)))
        self.outputs.append(output)
        return output

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        frame_idx: int | None = None,
        capture_time_ns: int | None = None,
        source_time_ns: int | None = None,
    ) -> FrameOutput:
        """Propagate one frame and admit or fuse detector work without waiting."""
        self._assert_owner()
        if self.state != "running":
            raise RuntimeError("Runtime must be running before frames are submitted")
        try:
            return self._process_frame(
                frame, frame_idx=frame_idx, capture_time_ns=capture_time_ns, source_time_ns=source_time_ns
            )
        except BaseException as exc:
            self.main_error = f"{type(exc).__name__}: {exc}"
            self.state = "failed"
            self.worker.stop_requested.set()
            raise

    step = process_frame

    def run(self, frames: Iterable[np.ndarray], *, paced: bool = True) -> Iterator[FrameOutput]:
        """Yield causal outputs from the input iterable at optional source cadence."""
        self._assert_owner()
        release = self.clock()
        first = self.last_frame + 1
        source_release = (
            self.last_source_ns + int(1e9 / self.config.frame_rate) if self.last_source_ns is not None else 0
        )
        for offset, frame in enumerate(frames):
            capture = release + int(offset / self.config.frame_rate * 1e9) if paced else self.clock()
            if paced:
                self.worker.stop_requested.wait(max(0, (capture - self.clock()) / 1e9))
                if self.worker.stop_requested.is_set():
                    break
            yield self.process_frame(
                frame,
                frame_idx=first + offset,
                capture_time_ns=capture,
                source_time_ns=source_release + int(offset / self.config.frame_rate * 1e9),
            )

    def close(self, timeout_s: float = 2.0) -> ShutdownReport:
        """Stop admission and verify worker shutdown within the timeout."""
        self._assert_owner()
        number(timeout_s, "timeout_s")
        if self.state == "created":
            self.state = "stopped"
            return ShutdownReport(True, False, None)
        if self.state == "stopped":
            return ShutdownReport(True, False, self.worker.error)
        self.state = "stopping"
        self.worker.stop_requested.set()
        while True:
            try:
                request = self.requests.get_nowait()
            except queue.Empty:
                break
            request.release()
            self.requests.task_done()
            self.counts["requests_shutdown_discarded"] += 1
            self._event(EventKind.SHUTDOWN_DISCARDED, request.source.frame_idx, request.source.source_ns)
        self.worker.join(timeout_s)
        if self.worker.is_alive():
            return ShutdownReport(False, True, self.worker.error)
        while True:
            try:
                result = self.results.get_nowait()
            except queue.Empty:
                break
            result.release()
            self.results.task_done()
            self.counts["results_shutdown_discarded"] += 1
        self._pump_events()
        self.history.clear()
        self.ended_ns = self.clock()
        self.state = "failed" if self.worker.error or self.main_error else "stopped"
        return ShutdownReport(True, False, self.worker.error)

    stop = close

    def summary(self) -> dict[str, object]:
        """Return aggregate telemetry with explicit measurement scope."""
        self._assert_owner()
        rows = list(self.rows)
        elapsed = (self.measurement_end_ns - self.started_ns) / 1e9 if self.measurement_end_ns else 0
        lifecycle_seconds = (
            ((self.ended_ns or self.clock()) - self.started_ns) / 1e9 if self.started_ns else 0
        )
        frames = self.counts["frames"]
        source_frames = frames + self.counts["source_frames_missing"]
        worker = self.worker.snapshot()
        cpu = sum(self.process.cpu_times()[:2]) - self.cpu_started if self.started_ns else 0
        return {
            "status": "complete" if self.state == "stopped" and not self.worker.error else "partial",
            "runtime_state": self.state,
            "config": asdict(self.config),
            "scheduler": self.scheduler.describe(),
            "metadata": dict(self.metadata),
            "cpu_convention": "100% = one logical core; can exceed 100%",
            "throughput_window": "worker ready to last output; shutdown excluded",
            "source_frame_range": [self.first_frame, self.last_frame] if frames else None,
            "frames": frames,
            "source_frames": source_frames,
            "published_frames": frames,
            "coverage": frames / source_frames if source_frames else None,
            "deadline_miss_rate_published": self.counts["deadline_misses"] / frames if frames else None,
            "deadline_miss_rate_source": (
                self.counts["deadline_misses"] + self.counts["source_frames_missing"]
            )
            / source_frames
            if source_frames
            else None,
            "frame_latency_ms": _stats(r["frame_work_ms"] for r in rows),
            "capture_to_output_latency_ms": _stats(r["capture_to_output_ms"] for r in rows),
            "detector_freshness_ms": _stats(
                r["detector_freshness_ms"] for r in rows if r["detector_freshness_ms"] is not None
            ),
            "quantile_scope": "retained_frame_window",
            "quantile_frames": len(rows),
            "ledger_complete": self.counts["telemetry_frames_evicted"] == 0,
            "event_ledger_complete": (
                self.counts["telemetry_events_evicted"] == 0
                and not worker["counts"].get("lost_worker_events", 0)
            ),
            "throughput_fps": frames / elapsed if elapsed > 0 else None,
            "wall_seconds": elapsed,
            "process_cpu_seconds": cpu,
            "cpu_window_seconds": lifecycle_seconds,
            "process_cpu_percent_mean": 100 * cpu / lifecycle_seconds if lifecycle_seconds > 0 else None,
            "sampled_peak_rss_mib": self.sampled_peak_rss or None,
            "counts": dict(self.counts),
            "worker_counts": worker["counts"],
            "detector_service_ms": _stats(worker["service_ms"]),
            "detector_queue_wait_ms": _stats(worker["queue_wait_ms"]),
            "worker_error": self.worker.error,
            "main_error": self.main_error,
            "HOTA": None,
            "MOTA": None,
            "IDF1": None,
            "IDSW": None,
            "power_watts": None,
            "accuracy_status": "not_evaluated; export causal outputs against the full source-frame ledger",
        }

    def export(self, output: str | Path) -> None:
        """Write immutable causal predictions and telemetry after shutdown."""
        self._assert_owner()
        if self.worker.is_alive() or self.state not in ("stopped", "failed"):
            raise RuntimeError("Export outside the frame loop after verified worker shutdown")
        path = Path(output)
        path.mkdir(parents=True, exist_ok=False)
        (path / "summary.json").write_text(json.dumps(self.summary(), indent=2, allow_nan=False))
        for name, records in (("frames.jsonl", self.rows), ("events.jsonl", self.events)):
            with (path / name).open("w") as stream:
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
        for name, deadline_valid in (("tracks.txt", False), ("deadline_tracks.txt", True)):
            with (path / name).open("w") as stream:
                for output_frame in self.outputs:
                    if deadline_valid and output_frame.telemetry["deadline_missed"]:
                        continue
                    for track in output_frame.tracks:
                        x1, y1, x2, y2 = track.box
                        stream.write(
                            f"{output_frame.frame_idx},{track.id},{x1 + 1:.3f},{y1 + 1:.3f},"
                            f"{x2 - x1:.3f},{y2 - y1:.3f},{track.score:.5f},-1,-1,-1\n"
                        )

    def __enter__(self) -> "AsyncTrackerRuntime":
        """Enter the resource lifecycle and return its active owner."""
        return self.start()

    def __exit__(self, *_args: object) -> None:
        """Exit the resource lifecycle and verify cleanup."""
        report = self.close()
        if not report.stopped:
            raise TimeoutError("Detector worker is still alive; shutdown is incomplete")
