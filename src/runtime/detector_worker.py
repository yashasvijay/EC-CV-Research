from __future__ import annotations
from collections import Counter, deque
import queue
import threading
from time import perf_counter_ns
from typing import Any, Callable
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..schedulers import EventKind, SchedulerEvent
from ..sort_tracker import KalmanBox, pairwise_iou
from .packets import Clock, Detector, DetectionRequest, DetectionResult, RuntimeConfig, _Anchor, _Correction
from .history import _History
from .motion import _owned, _seed
from ..motion import advance_batch
from .mailbox import _offer_latest


class DetectionWorkerError(RuntimeError):
    """Detector initialization, inference or replay failure."""

    pass


class DetectionWorker(threading.Thread):
    """Owns inference and bounded historical proposal preparation; never live tracks."""

    def __init__(
        self,
        factory: Callable[[], Detector],
        requests: queue.Queue[DetectionRequest],
        results: queue.Queue[DetectionResult],
        history: _History,
        config: RuntimeConfig,
        clock: Clock = perf_counter_ns,
    ):
        """Initialize validated configuration and owned runtime state."""
        super().__init__(name="detection-worker", daemon=False)
        self.factory, self.requests, self.results = factory, requests, results
        self.history, self.config, self.clock = history, config, clock
        self.ready, self.stop_requested, self.busy = threading.Event(), threading.Event(), threading.Event()
        self.error: str | None = None
        self.lock = threading.Lock()
        self.counts: Counter[str] = Counter()
        self.events: queue.Queue[SchedulerEvent] = queue.Queue(maxsize=256)
        self.inference_lock = threading.Lock()
        self.detector: Detector | None = None
        self.latest_started = 0
        self.service_ms: deque[float] = deque(maxlen=config.telemetry_frames)
        self.queue_wait_ms: deque[float] = deque(maxlen=config.telemetry_frames)

    def count(self, reason: str, amount: int = 1) -> None:
        """Increment a worker counter under the telemetry lock."""
        with self.lock:
            self.counts[reason] += amount

    def snapshot(self) -> dict[str, Any]:
        """Copy worker counters and bounded service-time samples."""
        with self.lock:
            return {
                "counts": dict(self.counts),
                "latest_started": self.latest_started,
                "service_ms": list(self.service_ms),
                "queue_wait_ms": list(self.queue_wait_ms),
            }

    def latest(self) -> tuple[int, float | None]:
        """Publish the newest source frame to bounded worker history."""
        with self.lock:
            return self.latest_started, self.service_ms[-1] if self.service_ms else None

    def emit(self, kind: EventKind, request: DetectionRequest) -> None:
        """Publish a worker lifecycle event with bounded accounting."""
        try:
            self.events.put_nowait(
                SchedulerEvent(
                    kind,
                    request.source.frame_idx,
                    epoch=request.epoch,
                    source_time_ns=request.source.source_ns,
                )
            )
        except queue.Full:
            self.count("lost_worker_events")

    def _detections(self, raw: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        detections = np.array(raw, dtype=np.float32, copy=True)
        if (
            detections.ndim != 2
            or detections.shape[1] != 6
            or not np.isfinite(detections).all()
            or np.any(detections[:, 2:4] <= detections[:, :2])
            or np.any((detections[:, 4] < 0) | (detections[:, 4] > 1))
            or np.any(detections[:, 5] != np.floor(detections[:, 5]))
            or np.any(detections[:, 5] < 0)
        ):
            raise ValueError("Detector must return finite (N,6) xyxy/score/integer-class rows")
        detections[:, [0, 2]] = detections[:, [0, 2]].clip(0, shape[1])
        detections[:, [1, 3]] = detections[:, [1, 3]].clip(0, shape[0])
        detections = detections[(detections[:, 2] > detections[:, 0]) & (detections[:, 3] > detections[:, 1])]
        order = np.argsort(-detections[:, 4], kind="stable")
        self.count("detections_capped", max(0, len(detections) - self.config.max_tracks))
        return _owned(detections[order[: self.config.max_tracks]])

    def detect_sync(self, image: np.ndarray) -> np.ndarray:
        """Serialize an explicit cold-start inference with background detector use."""
        if self.detector is None or not self.ready.is_set():
            raise RuntimeError("Detector is not ready for bootstrap")
        self.count("bootstrap_started")
        with self.inference_lock:
            if self.stop_requested.is_set():
                raise RuntimeError("Detector is stopping")
            detections = self._detections(self.detector(image), image.shape)
        self.count("bootstrap_completed")
        return detections

    def _prepare(
        self, request: DetectionRequest, detections: np.ndarray, started: int, completed: int
    ) -> DetectionResult:
        preparation_start = self.clock()
        lease = self.history.lease(request.source.frame_idx)
        reason = None
        corrections: list[_Correction] = []
        target = request.source.frame_idx
        target_lease = None
        try:
            if lease is None:
                reason = "history_gap"
            elif len(lease.frames) - 1 > self.config.max_replay_frames:
                reason = "replay_depth"
            else:
                target = lease.frames[-1].frame_idx
                matched: dict[int, _Anchor] = {}
                if request.anchors and len(detections):
                    overlaps = pairwise_iou(np.array([a.box for a in request.anchors]), detections[:, :4])
                    classes = np.array([a.cls for a in request.anchors])[:, None]
                    valid = (overlaps >= self.config.match_iou) & (classes == detections[None, :, 5])
                    rows, cols = linear_sum_assignment(np.where(valid, 1 - overlaps, 1e6))
                    matched = {int(j): request.anchors[i] for i, j in zip(rows, cols) if valid[i, j]}
                budget = self.config.effective_replay_budget_ms
                boxes = [d[:4].copy() for d in detections]
                points = [_seed(lease.frames[0].gray, box, self.config) for box in boxes]
                confidence = [1.0 if len(p) >= self.config.min_points else 0.0 for p in points]
                filters: list[KalmanBox | None] = []
                for j, detection in enumerate(detections):
                    filter_ = KalmanBox(detection[:4]) if self.config.covariance else None
                    anchor = matched.get(j)
                    if filter_ is not None and anchor is not None and anchor.x is not None:
                        assert anchor.covariance is not None
                        filter_.x, filter_.P = anchor.x.copy(), anchor.covariance.copy()
                        filter_.update(detection[:4])
                    filters.append(filter_)
                for previous, current in zip(lease.frames, lease.frames[1:]):
                    if self.stop_requested.is_set() or self.clock() > request.expiry_ns:
                        reason = "expired"
                        break
                    if (self.clock() - preparation_start) / 1e6 >= budget:
                        reason = "replay_budget"
                        break
                    updates = advance_batch(
                        previous.gray,
                        current.gray,
                        boxes,
                        points,
                        min_points=self.config.min_points,
                        fb_threshold=self.config.fb_threshold,
                    )
                    for j, update in enumerate(updates):
                        boxes[j], points[j] = update.box, update.points
                        confidence[j] *= update.quality
                        filter_ = filters[j]
                        if filter_ is not None:
                            filter_.predict(
                                (current.source_ns - previous.source_ns) / 1e9 * self.config.frame_rate
                            )
                if self.clock() > request.expiry_ns:
                    reason = "expired"
                elif (self.clock() - preparation_start) / 1e6 >= budget:
                    reason = "replay_budget"
                if reason is None:
                    for j, detection in enumerate(detections):
                        anchor, filter_ = matched.get(j), filters[j]
                        if anchor is None and len(points[j]) < self.config.min_points:
                            self.count("unsupported_births")
                            continue
                        corrections.append(
                            _Correction(
                                _owned(np.r_[boxes[j], detection[4:]].astype(np.float32)),
                                _owned(points[j]),
                                confidence[j],
                                anchor.track_id if anchor else None,
                                anchor.revision if anchor else None,
                                _owned(detection[:4] - np.asarray(anchor.box)) if anchor else None,
                                _owned(filter_.x) if filter_ else None,
                                _owned(filter_.P) if filter_ else None,
                            )
                        )
                if reason is None:
                    target_lease = self.history.lease(target, target)
                    if target_lease is None:
                        reason = "history_gap"
        finally:
            if lease is not None:
                lease.release()
        if reason is not None:
            corrections.clear()
        return DetectionResult(
            request.request_id,
            request.epoch,
            request.source.frame_idx,
            request.source.capture_ns,
            request.source.source_ns,
            request.expiry_ns,
            detections,
            started,
            completed,
            self.clock(),
            target,
            tuple(corrections),
            target_lease,
            reason,
        )

    def run(self) -> None:
        """Yield causal outputs from the input iterable at optional source cadence."""
        request = None
        try:
            detector = self.factory()
            if not callable(detector):
                raise TypeError("detector_factory must return a callable detector")
            self.detector = detector
            self.ready.set()
            while not self.stop_requested.is_set():
                try:
                    request = self.requests.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    if self.stop_requested.is_set():
                        self.count("shutdown_discarded")
                        self.emit(EventKind.SHUTDOWN_DISCARDED, request)
                        continue
                    if self.clock() > request.expiry_ns:
                        self.count("request_expired")
                        self.emit(EventKind.STALE_DISCARDED, request)
                        continue
                    self.busy.set()
                    started = self.clock()
                    with self.lock:
                        self.latest_started = request.source.frame_idx
                        self.queue_wait_ms.append((started - request.admitted_ns) / 1e6)
                    self.count("started")
                    self.emit(EventKind.STARTED, request)
                    with self.inference_lock:
                        detections = self._detections(detector(request.image), request.source.gray.shape)
                    completed = self.clock()
                    with self.lock:
                        self.service_ms.append((completed - started) / 1e6)
                    self.count("completed")
                    self.emit(EventKind.COMPLETED, request)
                    if self.stop_requested.is_set():
                        self.count("shutdown_discarded")
                        continue
                    result = self._prepare(request, detections, started, completed)
                    if self.stop_requested.is_set():
                        result.release()
                        self.count("shutdown_discarded")
                        continue
                    old = _offer_latest(self.results, result)
                    if old is not None:
                        old.release()
                        self.count("result_superseded")
                    self.count("published")
                except BaseException:
                    self.emit(EventKind.FAILED, request)
                    raise
                finally:
                    self.busy.clear()
                    request.release()
                    self.requests.task_done()
                    request = None
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.count("failed")
            if request is not None:
                self.emit(EventKind.FAILED, request)
        finally:
            self.ready.set()
