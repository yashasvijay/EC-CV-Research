"""Opt-in bridge for a later Python integration; importing this changes no pipeline.

Call apply_lk_result after OpenCV's forward/backward LK calls. Call decide after
tracking, at the point where Pipeline.step currently chooses its trigger.
Do not switch a running experiment to this backend.
"""
import numpy as np
from edge_gating import Config, GatingEngine


def make_engine(pipeline):
    tracker = pipeline.tracker
    return GatingEngine(Config(
        min_points=tracker.min_points, fb_threshold=tracker.fb_threshold,
        max_interval=pipeline.max_interval, min_confidence=pipeline.min_confidence,
        max_displacement=pipeline.max_displacement,
    ))


def apply_lk_result(engine, track, previous, forward, backward,
                    forward_status, backward_status, width, height):
    # Inputs already have the tracker's native float32/uint8 layout. No implicit
    # forcecast or ascontiguousarray hides an allocation. Own them until return.
    result = engine.assess_motion(previous, forward, backward, forward_status,
                                 backward_status, track.box, track.confidence, width, height)
    track.points = forward[result.valid_indices].reshape(-1, 1, 2)
    track.confidence = result.confidence
    track.displacement = result.displacement
    track.fb_error = result.fb_error
    track.box = np.asarray(result.box, dtype=np.float32)
    return result


def decide(engine, pipeline):
    tracks = pipeline.tracker.visible()
    return engine.decide(
        pipeline.frame_id, pipeline.last_detection,
        np.array([track.confidence for track in tracks], dtype=np.float64),
        np.array([track.displacement for track in tracks], dtype=np.float64),
        mode=pipeline.mode,
    )
