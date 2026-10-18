import cv2
import numpy as np
import pytest

from src.features import select_roi_features
from src.tracker import LKTracker
from src.runtime.motion import _seed
from src.runtime.packets import RuntimeConfig


def test_active_track_features_use_cropped_images_and_local_masks(monkeypatch):
    gray = np.random.default_rng(7).integers(0, 256, (1080, 1920), dtype=np.uint8)
    detections = np.array(
        [[30.4, 40.2, 130.6, 150.7, 0.9, 0], [600.2, 400.3, 720.8, 540.9, 0.8, 0]], np.float32
    )
    calls = []
    original = cv2.goodFeaturesToTrack

    def observed(image, *args, mask=None, **kwargs):
        calls.append((image.shape, mask.copy()))
        return original(image, *args, mask=mask, **kwargs)

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", observed)
    tracker = LKTracker()
    tracker.update(gray, detections)
    assert len(calls) == 2
    for track, (shape, mask) in zip(tracker.visible(), calls):
        assert len(track.points) >= tracker.min_points
        x1, y1, x2, y2 = track.box
        points = track.points.reshape(-1, 2)
        assert np.all((points[:, 0] >= x1) & (points[:, 0] < x2))
        assert np.all((points[:, 1] >= y1) & (points[:, 1] < y2))
        assert shape == (int(np.ceil(y2)) - int(np.floor(y1)), int(np.ceil(x2)) - int(np.floor(x1)))
        assert mask.shape == shape and mask.dtype == np.uint8
        assert not mask[0].any() and not mask[:, 0].any()
        assert shape[0] * shape[1] < gray.size / 50
    tracker.update(gray, np.empty((0, 6), np.float32))
    assert len(calls) == 2 and not tracker.visible()


@pytest.mark.parametrize("box", [(-10.5, -9.2, 30.7, 40.8), (70.3, 60.2, 120, 100), (10.1, 15.2, 60.3, 65.4)])
def test_clipped_fractional_boxes_and_async_seed(box):
    gray = np.random.default_rng(11).integers(0, 256, (80, 100), dtype=np.uint8)
    before = gray.copy()
    points = select_roi_features(gray, box)
    assert len(points) > 0 and points.dtype == np.float32 and points.flags.c_contiguous
    xy = points.reshape(-1, 2)
    assert np.all((xy[:, 0] >= max(0, box[0])) & (xy[:, 0] < min(100, box[2])))
    assert np.all((xy[:, 1] >= max(0, box[1])) & (xy[:, 1] < min(80, box[3])))
    np.testing.assert_array_equal(points, _seed(gray, np.array(box), RuntimeConfig()))
    np.testing.assert_array_equal(before, gray)


@pytest.mark.parametrize(
    "box", [(200, 200, 220, 220), (-20, -20, -1, -1), (40, 30, 20, 10), (20, 20, 20, 40), (1, 1, 2, 2)]
)
def test_empty_clipped_or_degenerate_roi_skips_opencv(box, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("OpenCV must not receive an invalid crop")

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", forbidden)
    points = select_roi_features(np.zeros((80, 100), np.uint8), box)
    assert points.shape == (0, 1, 2) and points.dtype == np.float32


def test_no_texture_and_invalid_inputs():
    assert not len(select_roi_features(np.zeros((80, 100), np.uint8), [10, 10, 50, 50]))
    for box in ([0, 1, 2], [0, 1, np.nan, 4]):
        with pytest.raises(ValueError):
            select_roi_features(np.zeros((80, 100), np.uint8), box)


def test_paired_benchmark_records_latency_without_speed_assertions(tmp_path, monkeypatch):
    from scripts import benchmark_roi_features as benchmark

    source = tmp_path / "MOT17-02-FRCNN"
    (source / "img1").mkdir(parents=True)
    (source / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-02-FRCNN\nimDir=img1\nframeRate=30\n"
        "seqLength=3\nimWidth=100\nimHeight=80\nimExt=.png\n"
    )
    image = np.random.default_rng(19).integers(0, 256, (80, 100, 3), dtype=np.uint8)
    for i in range(1, 4):
        cv2.imwrite(str(source / "img1" / f"{i:06d}.png"), image)
    tracks = tmp_path / "tracks.txt"
    tracks.write_text("".join(f"{i},1,16,16,30,30,.9,-1,-1,-1\n" for i in range(1, 4)))
    monkeypatch.setattr(benchmark, "running_searches", lambda root: [])
    output = tmp_path / "comparison"
    benchmark.main(
        [
            "--sequence",
            str(source),
            "--tracks",
            str(tracks),
            "--frames",
            "3",
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--output",
            str(output),
        ]
    )
    import json

    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["HOTA"] is None
    assert len((output / "frames.jsonl").read_text().splitlines()) == 6
    assert summary["aggregates"]["full_frame_mask"]["calls"] == 3
    assert summary["aggregates"]["cropped_roi_mask"]["mean_ms_per_frame"] > 0
