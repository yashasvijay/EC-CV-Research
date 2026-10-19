"""Concurrency and packet ownership invariants under synthetic detector delays."""

from dataclasses import replace
import threading
import time

import numpy as np
import pytest

from src.async_pipeline import AsyncTrackerRuntime, RuntimeConfig
from src.runtime.history import _History
from src.runtime.packets import _Frame
from src.schedulers import FixedKPolicy
from tests.test_async_bootstrap import scene, boxes


def wait_result(runtime):
    with runtime.results.not_empty:
        assert runtime.results.not_empty.wait_for(lambda: bool(runtime.results.queue), timeout=3)


def assert_conserved(runtime):
    summary = runtime.summary()
    accounting = summary["packet_accounting"]
    assert accounting["quiescent"]
    assert accounting["admitted"] == accounting["started"] + accounting["superseded"]
    assert accounting["published"] == accounting["fused"] + accounting["rejected"]
    assert runtime.requests.unfinished_tasks == runtime.results.unfinished_tasks == 0
    assert runtime.history.usage() == {"frames": 0, "bytes": 0, "pinned_frames": 0}
    assert not runtime.worker.is_alive() and not runtime.worker.busy.is_set()
    assert (
        runtime.counts["corrections_received"]
        == runtime.counts["corrections_accepted"] + runtime.counts["corrections_rejected"]
    )


def test_concurrent_lease_release_is_exactly_once(monkeypatch):
    history = _History(RuntimeConfig())
    history.admit(_Frame(1, 1, 1, np.zeros((8, 8), np.uint8)))
    lease = history.lease(1)
    barrier = threading.Barrier(12)
    calls = []
    original = history.release

    def slow_release(frames):
        calls.append(1)
        time.sleep(0.005)  # Exposes a check-then-set race without the lease lock.
        original(frames)

    monkeypatch.setattr(history, "release", slow_release)
    threads = [threading.Thread(target=lambda: (barrier.wait(), lease.release())) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    assert len(calls) == 1
    assert lease.released and lease.frames == ()
    assert history.usage()["pinned_frames"] == 0


def test_bootstrap_adopts_inflight_result_without_secondary_inference():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def detector(frame):
        calls.append(1)
        entered.set()
        assert release.wait(3)
        return boxes()

    runtime = AsyncTrackerRuntime(
        lambda: detector, FixedKPolicy(1), RuntimeConfig(synchronous_bootstrap=False, deadline_ms=500)
    )
    try:
        runtime.start()
        runtime.process_frame(scene(), source_time_ns=0)
        assert entered.wait(2)
        # Enable recovery while inference is already in flight.
        runtime.config = replace(runtime.config, synchronous_bootstrap=True)
        output = runtime.process_frame(scene(), source_time_ns=33_333_333)
        assert output.telemetry["bootstrap_deferred_busy"]
        assert not output.telemetry["bootstrap_attempted"]
        assert calls == [1] and not release.is_set()
        release.set()
        wait_result(runtime)
        output = runtime.process_frame(scene(), source_time_ns=66_666_666)
        assert output.tracks
        assert runtime.counts["bootstrap_adopted_results"] == 1
        assert runtime.counts["bootstrap_calls"] == 0
    finally:
        release.set()
        assert runtime.close().stopped
    assert_conserved(runtime)


def test_matched_corrections_ignore_birth_lag_and_reseed_current_frame(monkeypatch):
    import src.async_pipeline as pipeline

    runtime = AsyncTrackerRuntime(
        lambda: lambda frame: boxes(), FixedKPolicy(1), RuntimeConfig(deadline_ms=500, max_commit_frames=0)
    )
    try:
        runtime.start()
        first = runtime.process_frame(scene(), source_time_ns=0)
        runtime.process_frame(scene(), source_time_ns=33_333_333)
        wait_result(runtime)
        proposal = runtime.results.queue[0]
        assert all(c.track_id is not None and len(c.points) == 0 for c in proposal.corrections)
        assert proposal.target_lease is None
        seeds = []
        original = pipeline._seed

        def seed(gray, box, config):
            seeds.append(gray)
            return original(gray, box, config)

        def forbidden(*args, **kwargs):
            raise AssertionError("Matched corrections must not replay LK")

        monkeypatch.setattr(pipeline, "_seed", seed)
        monkeypatch.setattr(pipeline, "advance_batch", forbidden)
        output = runtime.process_frame(scene(), source_time_ns=66_666_666)
        assert output.telemetry["fusion"] == "fused"
        assert [t.id for t in output.tracks] == [t.id for t in first.tracks]
        assert len(seeds) == 2 and all(g is runtime.history.frames[3].gray for g in seeds)
        assert runtime.counts["corrections_accepted"] == 2
    finally:
        assert runtime.close().stopped
    assert_conserved(runtime)


@pytest.mark.parametrize("seed", range(8))
def test_random_latency_packet_conservation_and_zero_leaks(seed):
    rng = np.random.default_rng(seed)
    delays = iter(rng.uniform(0, 0.012, 100))
    call_count = 0

    def detector(frame):
        nonlocal call_count
        call_count += 1
        time.sleep(next(delays))
        return np.empty((0, 6), np.float32) if call_count % 5 == 0 else boxes()

    config = RuntimeConfig(
        synchronous_bootstrap=False,
        deadline_ms=500,
        max_detection_age_ms=50,
        max_commit_frames=3,
        history_frames=12,
    )
    runtime = AsyncTrackerRuntime(lambda: detector, FixedKPolicy(1), config)
    try:
        runtime.start()
        for i in range(45):
            runtime.process_frame(scene(), source_time_ns=i * 33_333_333)
            time.sleep(float(rng.uniform(0, 0.003)))
    finally:
        assert runtime.close().stopped
    assert runtime.counts["requests_admitted"] > 1
    assert runtime.counts["requests_superseded"] == 0
    assert_conserved(runtime)
    summary = runtime.summary()
    assert summary["IDF1"] is None and summary["IDSW"] is None
    assert summary["identification_metrics"]["status"].startswith("not_evaluated")


def test_failed_birth_replay_preserves_independent_matched_corrections():
    entered, release = threading.Event(), threading.Event()
    calls = 0

    def detector(frame):
        nonlocal calls
        calls += 1
        if calls == 1:
            return boxes()
        entered.set()
        assert release.wait(3)
        return np.vstack([boxes(), np.array([[1, 1, 12, 12, 0.8, 0]], np.float32)])

    config = RuntimeConfig(deadline_ms=500, max_replay_frames=1, max_commit_frames=0)
    runtime = AsyncTrackerRuntime(lambda: detector, FixedKPolicy(1), config)
    try:
        runtime.start()
        runtime.process_frame(scene(), source_time_ns=0)
        runtime.process_frame(scene(), source_time_ns=33_333_333)
        assert entered.wait(2)
        runtime.process_frame(scene(), source_time_ns=66_666_666)
        runtime.process_frame(scene(), source_time_ns=99_999_999)
        release.set()
        wait_result(runtime)
        output = runtime.process_frame(scene(), source_time_ns=133_333_332)
        assert output.telemetry["fusion"] == "fused"
        assert len(output.tracks) == 2
        assert runtime.worker.snapshot()["counts"]["birth_replay_discarded"] == 1
        assert runtime.counts["corrections_accepted"] == 2
    finally:
        release.set()
        assert runtime.close().stopped
    assert_conserved(runtime)


def test_expired_before_start_is_conserved_as_superseded():
    checked = threading.Event()
    owner = threading.get_ident()

    def clock():
        if threading.get_ident() != owner:
            checked.set()
            return 10_000_000_000
        return 1_000_000_000

    def forbidden(frame):
        raise AssertionError("Expired request must not start inference")

    runtime = AsyncTrackerRuntime(
        lambda: forbidden, FixedKPolicy(1), RuntimeConfig(synchronous_bootstrap=False), clock=clock
    )
    try:
        runtime.start()
        runtime.process_frame(scene(), source_time_ns=0)
        assert checked.wait(2)
    finally:
        assert runtime.close().stopped
    assert runtime.worker.snapshot()["counts"]["prestart_cancelled"] == 1
    assert runtime.summary()["packet_accounting"]["superseded"] == 1
    assert_conserved(runtime)


def test_pending_shutdown_request_is_conserved(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    runtime = AsyncTrackerRuntime(
        lambda: lambda frame: boxes(), FixedKPolicy(1), RuntimeConfig(synchronous_bootstrap=False)
    )
    original_get = runtime.requests.get

    def gated_get(*args, **kwargs):
        if kwargs.get("block") is False:
            return original_get(*args, **kwargs)
        entered.set()
        assert release.wait(3)
        return original_get(*args, **kwargs)

    monkeypatch.setattr(runtime.requests, "get", gated_get)
    try:
        runtime.start()
        assert entered.wait(2)
        runtime.process_frame(scene(), source_time_ns=0)
        assert runtime.requests.qsize() == 1
        assert not runtime.close(timeout_s=0).stopped
        assert runtime.counts["requests_shutdown_discarded"] == 1
    finally:
        release.set()
        assert runtime.close().stopped
    assert_conserved(runtime)
