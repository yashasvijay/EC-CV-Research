from dataclasses import FrozenInstanceError, replace
import json
import threading

import cv2
import numpy as np
import pytest

from src.async_pipeline import AsyncTrackerRuntime, DetectionWorkerError, RuntimeConfig
from src.schedulers import FixedKPolicy, KalmanUncertaintyPolicy, OraclePolicy


def image(shift=0):
    canvas = np.zeros((120, 180, 3), np.uint8)
    canvas[30:90, 40:100] = np.random.default_rng(17).integers(0, 256, (60, 60, 3), dtype=np.uint8)
    return cv2.warpAffine(canvas, np.float32([[1, 0, shift * 2], [0, 1, shift]]), (180, 120))


def detection(shift=0):
    return np.array([[40 + shift * 2, 30 + shift, 100 + shift * 2, 90 + shift, 0.9, 0]], np.float32)


def wait_result(runtime):
    with runtime.results.not_empty:
        assert runtime.results.not_empty.wait_for(lambda: bool(runtime.results.queue), timeout=3)


class ControlledDetector:
    def __init__(self, shift=0):
        self.entered, self.release = threading.Event(), threading.Event()
        self.shift = shift
        self.thread_id = None
        self.received = None

    def __call__(self, frame):
        self.thread_id = threading.get_ident()
        assert not frame.flags.writeable
        self.entered.set()
        assert self.release.wait(3), "Test failed to release detector"
        self.received = frame.copy()
        return detection(self.shift)


def test_lifecycle_nonblocking_frames_latest_pending_and_shutdown():
    detector = ControlledDetector()
    runtime = AsyncTrackerRuntime(lambda: detector, FixedKPolicy(1), RuntimeConfig(deadline_ms=500))
    factory_thread = []
    runtime.worker.factory = lambda: factory_thread.append(threading.get_ident()) or detector
    try:
        runtime.start()
        assert factory_thread[0] != threading.get_ident()
        first = runtime.process_frame(image(), source_time_ns=0)
        assert detector.entered.wait(2)
        for index in range(2, 6):
            output = runtime.process_frame(image(), source_time_ns=index * 33_333_333)
            assert output.frame_idx == index
        assert not detector.release.is_set()  # All five outputs arrived during blocked inference.
        assert runtime.requests.qsize() == 1
        assert runtime.counts["requests_superseded"] == 3
        assert runtime.requests.queue[0].source.frame_idx == 5
        assert len(first.tracks) == 0
        assert runtime.close(timeout_s=0).worker_alive
        detector.release.set()
        assert runtime.close(timeout_s=2).stopped
        assert runtime.close().stopped
        assert not runtime.worker.is_alive()
        assert runtime.requests.unfinished_tasks == runtime.results.unfinished_tasks == 0
        assert runtime.summary()["frames"] == 5
        assert runtime.history.usage()["pinned_frames"] == 0
    finally:
        detector.release.set()
        runtime.close()


def test_delayed_birth_is_propagated_to_current_frame_and_outputs_are_immutable(tmp_path):
    detector = ControlledDetector()
    runtime = AsyncTrackerRuntime(lambda: detector, FixedKPolicy(100), RuntimeConfig(deadline_ms=500))
    try:
        runtime.start()
        previous = runtime.process_frame(image(), source_time_ns=0)
        assert detector.entered.wait(2)
        runtime.process_frame(image(1), source_time_ns=33_333_333)
        runtime.process_frame(image(2), source_time_ns=66_666_666)
        detector.release.set()
        wait_result(runtime)
        output = runtime.process_frame(image(3), source_time_ns=99_999_999)
        assert output.telemetry["fusion"] == "fused"
        assert len(output.tracks) == 1 and output.tracks[0].id == 1
        np.testing.assert_allclose(output.tracks[0].box, [46, 33, 106, 93], atol=0.4)
        assert output.tracks[0].anchor_source_frame == 1
        assert runtime.scheduler.last_fused_frame == 1
        assert not previous.tracks
        with pytest.raises(FrozenInstanceError):
            output.tracks[0].score = 0.1
        with pytest.raises(TypeError):
            output.telemetry["frame"] = 42
        assert runtime.close().stopped
        runtime.export(tmp_path / "export")
        summary = json.loads((tmp_path / "export/summary.json").read_text())
        assert summary["ledger_complete"] and summary["HOTA"] is None
        assert (tmp_path / "export/tracks.txt").read_text().startswith("4,1,")
        assert summary["frame_latency_ms"]["p99"] >= summary["frame_latency_ms"]["p95"]
    finally:
        detector.release.set()
        runtime.close()


def test_camera_buffer_ownership():
    detector = ControlledDetector()
    runtime = AsyncTrackerRuntime(lambda: detector, FixedKPolicy(100), RuntimeConfig(deadline_ms=500))
    original = image()
    try:
        runtime.start()
        runtime.process_frame(original, source_time_ns=0)
        assert detector.entered.wait(2)
        original[:] = 0
        detector.release.set()
        wait_result(runtime)
        np.testing.assert_array_equal(detector.received, image())
        output = runtime.process_frame(image(), source_time_ns=33_333_333)
        assert len(output.tracks) == 1
    finally:
        detector.release.set()
        runtime.close()


@pytest.mark.parametrize("kind", ["expired", "target_lag", "epoch", "duplicate"])
def test_reject_stale_results(kind):
    runtime = AsyncTrackerRuntime(
        lambda: lambda frame: detection(),
        FixedKPolicy(100),
        RuntimeConfig(deadline_ms=500, max_commit_frames=1),
    )
    try:
        runtime.start()
        runtime.process_frame(image(), source_time_ns=0)
        wait_result(runtime)
        result = runtime.results.get_nowait()
        runtime.results.task_done()
        if kind == "expired":
            result = replace(result, expiry_ns=0)
        elif kind == "epoch":
            result = replace(result, epoch="old")
        elif kind == "target_lag":
            runtime.process_frame(image(), source_time_ns=33_333_333)
        runtime.results.put_nowait(result)
        output = runtime.process_frame(image(), source_time_ns=66_666_666)
        if kind == "duplicate":
            assert output.telemetry["fusion"] == "fused"
            runtime.results.put_nowait(result)
            output = runtime.process_frame(image(), source_time_ns=99_999_999)
            assert output.telemetry["fusion"] == "older_observation"
            assert len(output.tracks) == 1
        else:
            assert output.telemetry["fusion"] == kind
            assert not output.tracks and runtime.last_fused_frame == 0
    finally:
        runtime.close()


def test_history_memory_bound_does_not_drop_tracking_frames():
    detector = ControlledDetector()
    config = RuntimeConfig(deadline_ms=500, history_frames=1, history_bytes=120 * 180)
    runtime = AsyncTrackerRuntime(lambda: detector, FixedKPolicy(1), config)
    try:
        runtime.start()
        runtime.process_frame(image(), source_time_ns=0)
        assert detector.entered.wait(2)
        for i in range(2, 5):
            assert runtime.process_frame(image(), source_time_ns=i * 33_333_333).frame_idx == i
        assert runtime.history.usage()["bytes"] <= config.history_bytes
        assert runtime.counts["history_capacity_misses"] == 3
        assert runtime.counts["requests_rejected"] == 3
    finally:
        detector.release.set()
        runtime.close()


class FakeClock:
    def __init__(self):
        self.now = 1_000_000_000
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.now

    def advance(self, ns):
        with self.lock:
            self.now += ns


def test_deadline_misses_gaps_and_quantiles_use_fake_clock():
    clock = FakeClock()
    runtime = AsyncTrackerRuntime(
        lambda: lambda frame: np.empty((0, 6), np.float32), FixedKPolicy(100), clock=clock
    )
    try:
        runtime.start()
        original = runtime.tracker.predict

        def slow_predict(gray):
            clock.advance(40_000_000)
            original(gray)

        runtime.tracker.predict = slow_predict
        outputs = [runtime.process_frame(image(), frame_idx=i) for i in [1, 3]]
        assert all(o.telemetry["deadline_missed"] for o in outputs)
        runtime.close()
        summary = runtime.summary()
        assert summary["frames"] == 2 and summary["source_frames"] == 3
        assert summary["deadline_miss_rate_published"] == 1
        assert summary["deadline_miss_rate_source"] == 1
        assert summary["coverage"] == pytest.approx(2 / 3)
        assert summary["frame_latency_ms"]["p95"] == 40
        assert summary["capture_to_output_latency_ms"]["p99"] == 40
    finally:
        runtime.close()


def test_factory_and_inference_failures_are_visible_and_workers_exit():
    def bad_factory():
        raise ValueError("factory fixture")

    runtime = AsyncTrackerRuntime(bad_factory, FixedKPolicy(1))
    with pytest.raises(DetectionWorkerError, match="factory fixture"):
        runtime.start()
    assert runtime.close().stopped and not runtime.worker.is_alive()
    failed = threading.Event()

    def bad_detector(frame):
        failed.set()
        raise ValueError("inference fixture")

    runtime = AsyncTrackerRuntime(lambda: bad_detector, FixedKPolicy(1))
    try:
        runtime.start()
        runtime.process_frame(image())
        assert failed.wait(2)
        runtime.worker.join(2)
        with pytest.raises(DetectionWorkerError, match="inference fixture"):
            runtime.process_frame(image())
        assert runtime.close().worker_error
        assert runtime.summary()["status"] == "partial"
    finally:
        runtime.close()


def test_schedulers_and_bounded_telemetry():
    runtime = AsyncTrackerRuntime(
        lambda: lambda frame: np.empty((0, 6), np.float32),
        OraclePolicy([], offline=True),
        RuntimeConfig(offline=True, telemetry_frames=2),
    )
    with runtime:
        outputs = list(runtime.run([image() for _ in range(5)], paced=False))
        assert len(outputs) == 5 and all(o.telemetry["admission"] == "not_requested" for o in outputs)
    assert not runtime.summary()["ledger_complete"]
    assert runtime.summary()["quantile_frames"] == 2 and runtime.summary()["frames"] == 5
    with pytest.raises(ValueError):
        AsyncTrackerRuntime(
            lambda: lambda frame: detection(), KalmanUncertaintyPolicy(0.1), RuntimeConfig(covariance=False)
        )


def test_input_errors_owner_and_single_use_lifecycle():
    runtime = AsyncTrackerRuntime(lambda: lambda frame: detection(), FixedKPolicy(100))
    assert runtime.close().stopped
    with pytest.raises(RuntimeError):
        runtime.start()
    runtime = AsyncTrackerRuntime(lambda: lambda frame: detection(), FixedKPolicy(100)).start()
    try:
        failures = []
        thread = threading.Thread(target=lambda: _wrong_owner(runtime, failures))
        thread.start()
        thread.join(2)
        assert failures and runtime.state == "running"
        with pytest.raises(ValueError):
            runtime.process_frame(np.zeros((20, 20), np.uint8))
        assert runtime.state == "failed"
    finally:
        runtime.close()
    assert runtime.summary()["status"] == "partial"


def _wrong_owner(runtime, failures):
    try:
        runtime.process_frame(image())
    except RuntimeError as exc:
        failures.append(str(exc))


class TwoStageDetector:
    def __init__(self, move=False):
        self.calls = 0
        self.second_entered = threading.Event()
        self.second_release = threading.Event()
        self.move = move

    def __call__(self, frame):
        self.calls += 1
        if self.calls == 1:
            return detection()
        self.second_entered.set()
        assert self.second_release.wait(3)
        result = detection(3) if self.move else detection()
        if self.move:
            result[:, [0, 2]] -= 2
        return result


def test_matched_delayed_residual_preserves_current_motion_and_id():
    detector = TwoStageDetector(move=True)
    runtime = AsyncTrackerRuntime(lambda: detector, FixedKPolicy(3), RuntimeConfig(deadline_ms=500))
    try:
        runtime.start()
        runtime.process_frame(image(), source_time_ns=0)
        wait_result(runtime)
        assert runtime.process_frame(image(1), source_time_ns=33_333_333).tracks[0].id == 1
        runtime.process_frame(image(2), source_time_ns=66_666_666)
        runtime.process_frame(image(3), source_time_ns=99_999_999)
        assert detector.second_entered.wait(2)
        runtime.process_frame(image(4), source_time_ns=133_333_332)
        runtime.process_frame(image(5), source_time_ns=166_666_665)
        detector.second_release.set()
        wait_result(runtime)
        output = runtime.process_frame(image(6), source_time_ns=199_999_998)
        assert output.telemetry["fusion"] == "fused" and len(output.tracks) == 1
        assert output.tracks[0].id == 1 and output.tracks[0].anchor_source_frame == 4
        np.testing.assert_allclose(output.tracks[0].box, [50, 36, 110, 96], atol=0.5)
        assert np.linalg.eigvalsh(runtime.filters[1].P).min() >= -1e-8
        np.testing.assert_allclose(runtime.filters[1].P, runtime.filters[1].P.T, atol=1e-10)
    finally:
        detector.second_release.set()
        runtime.close()


def test_delayed_matched_result_cannot_resurrect_retired_track():
    detector = TwoStageDetector()
    config = RuntimeConfig(deadline_ms=500, max_track_age_ms=120)
    runtime = AsyncTrackerRuntime(lambda: detector, FixedKPolicy(3), config)
    try:
        runtime.start()
        runtime.process_frame(image(), source_time_ns=0)
        wait_result(runtime)
        runtime.process_frame(image(), source_time_ns=33_333_333)
        runtime.process_frame(image(), source_time_ns=66_666_666)
        runtime.process_frame(image(), source_time_ns=99_999_999)
        assert detector.second_entered.wait(2)
        assert not runtime.process_frame(image(), source_time_ns=200_000_000).tracks
        detector.second_release.set()
        wait_result(runtime)
        output = runtime.process_frame(image(), source_time_ns=233_333_333)
        assert not output.tracks
        assert runtime.counts["corrections_rejected_revision"] == 1
        assert output.telemetry["fusion"] == "unsupported_corrections"
    finally:
        detector.second_release.set()
        runtime.close()


def test_start_timeout_reports_incomplete_then_verified_shutdown():
    entered, release = threading.Event(), threading.Event()

    def factory():
        entered.set()
        assert release.wait(3)
        return lambda frame: detection()

    runtime = AsyncTrackerRuntime(factory, FixedKPolicy(1))
    try:
        with pytest.raises(TimeoutError):
            runtime.start(timeout_s=0.01)
        assert entered.is_set()
        assert not runtime.close(timeout_s=0).stopped
        release.set()
        assert runtime.close(timeout_s=2).stopped
        assert runtime.summary()["status"] == "partial"
    finally:
        release.set()
        runtime.close()


def test_zero_texture_rejects_unsupported_birth():
    runtime = AsyncTrackerRuntime(
        lambda: lambda frame: detection(), FixedKPolicy(100), RuntimeConfig(deadline_ms=500)
    )
    try:
        runtime.start()
        blank = np.zeros_like(image())
        runtime.process_frame(blank, source_time_ns=0)
        wait_result(runtime)
        output = runtime.process_frame(blank, source_time_ns=33_333_333)
        assert not output.tracks and output.telemetry["fusion"] == "unsupported_corrections"
    finally:
        runtime.close()


def test_actual_onnx_wrapper_runs_in_worker(tmp_path):
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    from src.detector import YOLODetector

    output = np.zeros((1, 84, 1), np.float32)
    output[0, :5, 0] = [25, 32, 20, 25, 0.9]
    node = helper.make_node("Constant", [], ["output"], value=numpy_helper.from_array(output))
    graph = helper.make_graph(
        [node],
        "fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 84, 1])],
    )
    model_path = tmp_path / "fixture.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), model_path)
    runtime = AsyncTrackerRuntime(
        lambda: YOLODetector(model_path, threads=1),
        FixedKPolicy(100),
        RuntimeConfig(deadline_ms=1000),
        metadata={"model": "synthetic constant ONNX fixture"},
    )
    with runtime:
        runtime.process_frame(image(), source_time_ns=0)
        wait_result(runtime)
        output = runtime.process_frame(image(), source_time_ns=33_333_333)
        assert output.tracks and output.telemetry["fusion"] == "fused"
        assert output.telemetry["detector_service_ms"] >= 0
    assert runtime.summary()["metadata"]["model"].startswith("synthetic")
