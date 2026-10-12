"""Strict, lazy MOT17 sequence loading with original one-based frame IDs."""
from configparser import ConfigParser
from dataclasses import asdict, dataclass
from pathlib import Path
import re

import cv2
import numpy as np


@dataclass(frozen=True)
class SequenceInfo:
    name: str
    frame_rate: float
    length: int
    width: int
    height: int
    image_dir: str
    image_ext: str


@dataclass(frozen=True)
class MOTFrame:
    frame_id: int
    timestamp_seconds: float
    path: Path
    image: np.ndarray


class MOT17Sequence:
    def __init__(self, path):
        self.path = Path(path).resolve()
        parser = ConfigParser(interpolation=None)
        with (self.path / "seqinfo.ini").open() as stream:
            parser.read_file(stream)
        try:
            section = parser["Sequence"]
            self.info = SequenceInfo(
                section["name"], float(section["frameRate"]),
                int(section["seqLength"]), int(section["imWidth"]),
                int(section["imHeight"]), section["imDir"], section["imExt"],
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid seqinfo.ini: {self.path}") from exc
        info = self.info
        if (info.name != self.path.name or not np.isfinite(info.frame_rate)
                or info.frame_rate <= 0 or min(info.length, info.width, info.height) <= 0
                or not info.image_ext.startswith(".")):
            raise ValueError(f"Invalid sequence metadata: {info}")
        self.image_dir = (self.path / info.image_dir).resolve()
        if not self.image_dir.is_relative_to(self.path):
            raise ValueError("imDir must stay inside the sequence directory")
        indexed = {}
        for file in self.image_dir.iterdir():
            if file.is_file() and file.suffix.lower() == info.image_ext.lower():
                if not file.stem.isdigit():
                    raise ValueError(f"Non-numeric frame filename: {file}")
                frame_id = int(file.stem)
                if frame_id in indexed:
                    raise ValueError(f"Duplicate frame ID: {frame_id}")
                indexed[frame_id] = file
        expected = set(range(1, info.length + 1))
        if set(indexed) != expected:
            missing = sorted(expected - set(indexed))[:10]
            extra = sorted(set(indexed) - expected)[:10]
            raise ValueError(f"Frame sequence mismatch: missing={missing}, extra={extra}")
        self.frame_paths = tuple(indexed[i] for i in range(1, info.length + 1))

    def __len__(self):
        return self.info.length

    @property
    def metadata(self):
        return asdict(self.info) | {"path": str(self.path)}

    @property
    def ground_truth_path(self):
        return self.path / "gt" / "gt.txt"

    def read_frame(self, frame_id):
        if not 1 <= frame_id <= len(self):
            raise IndexError(f"Frame {frame_id} outside 1..{len(self)}")
        path = self.frame_paths[frame_id - 1]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode {path}")
        if image.shape[:2] != (self.info.height, self.info.width):
            raise ValueError(f"Frame dimensions disagree with seqinfo.ini: {path}")
        return MOTFrame(frame_id, (frame_id - 1) / self.info.frame_rate, path, image)

    def __iter__(self):
        for frame_id in range(1, len(self) + 1):
            yield self.read_frame(frame_id)

    def images(self):
        for frame in self:
            yield frame.image


class MOT17Dataset:
    """root/train/<sequence>. Default FRCNN suffix avoids triplicate videos.

    The suffix selects images only; detections still come from our ONNX model.
    seqinfo.ini supplies dimensions/FPS, not camera intrinsics or extrinsics.
    """
    def __init__(self, root, split="train", detector_suffix="FRCNN"):
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        if detector_suffix not in ("FRCNN", "DPM", "SDP"):
            raise ValueError("Choose exactly one MOT17 detector suffix")
        self.path = Path(root).resolve() / split
        pattern = re.compile(rf"MOT17-\d{{2}}-{detector_suffix}")
        self.sequence_paths = tuple(sorted(
            p for p in self.path.iterdir() if p.is_dir() and pattern.fullmatch(p.name)
        ))
        if not self.sequence_paths:
            raise ValueError(f"No {detector_suffix} sequences in {self.path}")

    def __len__(self):
        return len(self.sequence_paths)

    def __iter__(self):
        for path in self.sequence_paths:
            yield MOT17Sequence(path)
