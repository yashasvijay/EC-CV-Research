import threading
import cv2
import numpy as np
import pytest
from src.async_pipeline import AsyncTrackerRuntime, RuntimeConfig
from src.schedulers import FixedKPolicy
from src.motion import advance_batch
from src.features import select_roi_features


def scene(shift=0):
    image = np.random.default_rng(42).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    return cv2.warpAffine(image, np.float32([[1, 0, shift], [0, 1, 0]]), (128, 96))


def boxes():
    return np.array([[15, 15, 60, 65, 0.9, 0], [65, 20, 110, 75, 0.9, 0]], dtype=np.float32)


def test_default_synchronous_bootstrap_and_background_refresh():
    calls = []

    def detect(frame):
        calls.append(threading.get_ident())
        return boxes()

    runtime = AsyncTrackerRuntime(lambda: detect, FixedKPolicy(2), RuntimeConfig(deadline_ms=500))
    owner = threading.get_ident()
    with runtime:
        first = runtime.process_frame(scene(), source_time_ns=0)
        assert len(first.tracks) == 2
        assert calls == [owner]
        assert first.telemetry["bootstrap_attempted"] is True
        assert first.telemetry["bootstrap_ms"] > 0
        assert first.telemetry["frame_work_ms"] >= first.telemetry["bootstrap_ms"]
        original_ids = [t.id for t in first.tracks]
        runtime.process_frame(scene(1), source_time_ns=33_333_333)
        third = runtime.process_frame(scene(), source_time_ns=66_666_666)
        with runtime.results.not_empty:
            assert runtime.results.not_empty.wait_for(lambda: bool(runtime.results.queue), timeout=3)
        fourth = runtime.process_frame(scene(1), source_time_ns=99_999_999)
        assert len(fourth.tracks) == 2
        assert [t.id for t in fourth.tracks] == original_ids
        assert fourth.telemetry["accepted_detection_update"] is True
        assert any(thread != owner for thread in calls)
        assert runtime.counts["bootstrap_successes"] == 1
        assert runtime.counts["fused_with_tracks"] == 1
        assert len(third.tracks) == 2
    assert runtime.summary()["fused_detection_ratio"] > 0
    assert not runtime.worker.is_alive()


def test_rebootstrap_after_all_tracks_retire():
    runtime = AsyncTrackerRuntime(
        lambda: lambda image: boxes(), FixedKPolicy(100), RuntimeConfig(deadline_ms=500, max_track_age_ms=10)
    )
    with runtime:
        first = runtime.process_frame(scene(), source_time_ns=0)
        second = runtime.process_frame(scene(), source_time_ns=33_333_333)
        assert first.tracks and second.tracks
        assert first.tracks[0].id != second.tracks[0].id
        assert runtime.counts["bootstrap_successes"] == 2


def test_bootstrap_empty_detector_does_not_invent_tracks():
    with AsyncTrackerRuntime(lambda: lambda image: np.empty((0, 6), np.float32), FixedKPolicy(3)) as runtime:
        assert not runtime.process_frame(scene()).tracks
        assert runtime.counts["bootstrap_empty"] == 1
        assert runtime.last_fused_frame == 0


def test_batch_uses_one_lk_pair_for_multiple_tracks(monkeypatch):
    from src import motion

    previous = cv2.cvtColor(scene(), cv2.COLOR_BGR2GRAY)
    current = cv2.cvtColor(scene(2), cv2.COLOR_BGR2GRAY)
    bb = [row[:4] for row in boxes()]
    points = [select_roi_features(previous, b) for b in bb]
    original = motion.sparse_lk
    calls = []

    def spy(a, b, p):
        calls.append(len(p))
        return original(a, b, p)

    monkeypatch.setattr(motion, "sparse_lk", spy)
    result = advance_batch(previous, current, bb, points)
    assert len(calls) == 2
    assert calls == [sum(map(len, points))] * 2
    for before, after in zip(bb, result):
        assert after.quality > 0
        np.testing.assert_allclose(after.box - before, [2, 0, 2, 0], atol=0.2)


def test_replay_budget_reserves_pickup_time():
    config = RuntimeConfig(replay_budget_ms=100, max_commit_frames=2, frame_rate=30)
    assert config.effective_replay_budget_ms == pytest.approx(1000 / 30)
    assert config.effective_replay_budget_ms < config.max_commit_frames * 1000 / config.frame_rate


def test_bootstrap_cost_and_deadline_miss_are_not_hidden():
    class Clock:
        now = 1_000_000_000

        def __call__(self):
            return self.now

    clock = Clock()

    def detect(frame):
        clock.now += 80_000_000
        return boxes()

    with AsyncTrackerRuntime(lambda: detect, FixedKPolicy(3), clock=clock) as runtime:
        output = runtime.process_frame(scene(), source_time_ns=0)
        assert output.tracks
        assert output.telemetry["bootstrap_ms"] == 80.0
        assert output.telemetry["frame_work_ms"] >= 80.0
        assert output.telemetry["deadline_missed"] is True
