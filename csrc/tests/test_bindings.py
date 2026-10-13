"""Run with PYTHONPATH=csrc/build/python; no pipeline/backend activation."""
from concurrent.futures import ThreadPoolExecutor
import gc

import numpy as np
import pytest
native = pytest.importorskip("edge_gating", reason="Build the optional native module and set PYTHONPATH=csrc/build/python")


def points(count=6):
    previous = np.array([[10 + i * 3, 10 + i] for i in range(count)], np.float32).reshape(count,2)
    forward = previous + np.array([3, 4], np.float32)
    status = np.ones((count, 1), np.uint8)
    box = np.array([0, 0, 30, 40], np.float32)
    return previous, forward, previous.copy(), status, status.copy(), box


def assess(engine, buffers, prior=.8, **kwargs):
    return engine.assess_motion(*buffers, prior, 200, 200, **kwargs)


def test_translation_and_normalization():
    buffers = points()
    result = assess(native.GatingEngine(), buffers)
    assert result.failure == "none" and result.valid_points == 6
    assert result.delta == [3, 4]
    assert result.displacement == pytest.approx(.1)
    assert result.confidence == pytest.approx(.8)
    assert result.fb_error == result.spread == 0
    np.testing.assert_array_equal(result.box, [3, 4, 33, 44])
    np.testing.assert_array_equal(result.valid_indices, np.arange(6))


def test_even_median_and_consistency_filter():
    buffers = list(points(8))
    buffers[1] = buffers[0] + np.array([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[99,99],[99,99]], np.float32)
    buffers[3][-2:] = 0
    result = assess(native.GatingEngine(), buffers)
    assert result.delta == [3.5, 4.5]
    assert result.survival_ratio == .75
    np.testing.assert_array_equal(result.valid_indices, np.arange(6))
    buffers[2][:3, 0] += 2  # backward error exceeds threshold for these points
    failed = assess(native.GatingEngine(), buffers)
    assert failed.valid_points == 3 and failed.failure == "insufficient_points"
    assert failed.confidence == 0 and failed.fb_error is None


def test_nonfinite_empty_and_clipping():
    buffers = list(points())
    buffers[1][0, 0] = np.nan
    assert assess(native.GatingEngine(), buffers).failure == "nonfinite_forward"
    empty = points(0)
    empty = (np.empty((0,2),np.float32),) * 3 + (np.empty(0,np.uint8),) * 2 + (empty[-1],)
    assert assess(native.GatingEngine(), empty).failure == "insufficient_points"
    buffers = list(points())
    buffers[-1] = np.array([180,180,199,199], np.float32)
    result = assess(native.GatingEngine(), buffers)
    assert result.box == [183,184,200,200]
    buffers[-1] = np.array([0,0,0,1], np.float32)
    assert assess(native.GatingEngine(), buffers).failure == "invalid_box"


def test_numpy_parity_random_motion():
    rng = np.random.default_rng(42)
    engine = native.GatingEngine()
    for count in (6, 7, 20, 60):
        previous = rng.uniform(10, 100, (count,1,2)).astype(np.float32)
        forward = previous + rng.normal(0, .5, previous.shape).astype(np.float32)
        backward = previous + rng.normal(0, .2, previous.shape).astype(np.float32)
        status = np.ones((count,1), np.uint8)
        box = np.array([0,0,120,120], np.float32)
        result = engine.assess_motion(previous, forward, backward, status, status, box, .9, 200, 200)
        shifts = (forward - previous).reshape(-1,2)
        errors = np.linalg.norm((backward - previous).reshape(-1,2), axis=1)
        delta = np.median(shifts, axis=0)
        spread = np.median(np.linalg.norm(shifts - delta, axis=1))
        expected_confidence = .9 * np.exp(-float(np.median(errors)) / 1.5) * np.exp(-float(spread) / 3)
        np.testing.assert_allclose(result.delta, delta, atol=1e-6, rtol=1e-6)
        assert result.confidence == pytest.approx(expected_confidence, rel=1e-6)
        assert result.displacement == pytest.approx(float(np.linalg.norm(delta)) / float(np.linalg.norm(box[2:]-box[:2])), rel=1e-6)


@pytest.mark.parametrize("mode,frame,last,conf,disp,expected", [
    ("adaptive",2,1,.6,.15,None),
    ("adaptive",2,1,.59,.15,"confidence"),
    ("adaptive",2,1,.6,.151,"displacement"),
    ("adaptive",11,1,.1,.3,"max_interval"),
    ("adaptive",2,1,float("nan"),0,"invalid_motion"),
    ("baseline",2,1,.9,0,"baseline"),
    ("fixed",11,1,.9,0,"fixed_interval"),
    ("fixed",2,1,.1,1,None),
])
def test_thresholds_and_precedence(mode, frame, last, conf, disp, expected):
    decision = native.GatingEngine().decide(frame,last,np.array([conf]),np.array([disp],dtype=np.float64),mode)
    assert decision.reason == expected
    assert decision.should_detect == (expected is not None)


def test_no_tracks_frame_validation_and_invalid_config():
    engine = native.GatingEngine()
    empty = np.empty(0, np.float64)
    assert engine.decide(1,0,empty,empty).reason == "no_tracks"
    with pytest.raises(ValueError): engine.decide(0,0,empty,empty)
    with pytest.raises(ValueError): engine.decide(2,3,empty,empty)
    with pytest.raises(ValueError): native.Config(fb_threshold=0)
    with pytest.raises(ValueError): native.Config(min_confidence=float("nan"))
    with pytest.raises(ValueError): native.Config(max_interval=0)
    with pytest.raises(ValueError): engine.decide(1,0,empty,empty,mode="unknown")


def test_buffer_dtype_shape_layout_alignment_and_readonly():
    engine = native.GatingEngine()
    buffers = list(points())
    with pytest.raises(TypeError): assess(engine, [buffers[0].astype(np.float64)] + buffers[1:])
    with pytest.raises(TypeError): assess(engine, [buffers[0].tolist()] + buffers[1:])
    with pytest.raises(ValueError): assess(engine, [buffers[0][::2]] + buffers[1:])
    with pytest.raises(ValueError): assess(engine, [buffers[0][:,:1]] + buffers[1:])
    misaligned = np.ndarray((6,2), np.float32, buffer=bytearray(6*2*4+1), offset=1)
    with pytest.raises(ValueError): assess(engine, [misaligned] + buffers[1:])
    originals = [a.copy() for a in buffers]
    for a in buffers: a.setflags(write=False)
    assert assess(engine, buffers, release_gil=False).confidence == pytest.approx(.8)
    result = assess(engine, buffers)
    for a,b in zip(buffers, originals): np.testing.assert_array_equal(a,b)
    indices = result.valid_indices
    del result, buffers
    gc.collect()
    np.testing.assert_array_equal(indices, np.arange(6))


def test_batch_parity_offsets_and_thread_safety():
    engine = native.GatingEngine()
    buffers = points()
    previous, forward, backward, fs, bs, box = buffers
    arguments = [np.concatenate([a,a]) for a in (previous,forward,backward,fs,bs)]
    offsets = np.array([0,6,12], np.int64)
    boxes = np.stack([box,box])
    priors = np.array([.8,.5], np.float64)
    results = engine.assess_batch(*arguments, offsets, boxes, priors, 200, 200)
    assert [result.confidence for result in results] == pytest.approx([.8,.5])
    np.testing.assert_array_equal(results[1].valid_indices, np.arange(6))
    with pytest.raises(ValueError): engine.assess_batch(*arguments,np.array([0,13,12],np.int64),boxes,priors,200,200)
    with pytest.raises(TypeError): engine.assess_batch(*arguments,offsets.astype(np.int32),boxes,priors,200,200)
    for array in buffers: array.setflags(write=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda _: assess(engine, buffers).confidence, range(8)))
    assert outputs == pytest.approx([.8]*8)
    empty_points = np.empty((0,2),np.float32)
    empty_status = np.empty(0,np.uint8)
    assert engine.assess_batch(empty_points,empty_points,empty_points,empty_status,empty_status,
                              np.array([0],np.int64),np.empty((0,4),np.float32),np.empty(0,np.float64),200,200) == []
