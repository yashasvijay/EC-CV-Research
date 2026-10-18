from __future__ import annotations
from collections import Counter, OrderedDict
import threading
from .packets import RuntimeConfig, _Frame


class _Lease:
    def __init__(self, history: "_History", frames: tuple[_Frame, ...]):
        """Initialize validated configuration and owned runtime state."""
        self.history, self.frames = history, frames
        self.released = False

    def release(self) -> None:
        """Release the retained frame-history lease."""
        if not self.released:
            self.history.release(self.frames)
            self.released = True
            self.frames = ()


class _History:
    def __init__(self, config: RuntimeConfig):
        """Initialize validated configuration and owned runtime state."""
        self.config = config
        self.lock = threading.Lock()
        self.frames: OrderedDict[int, _Frame] = OrderedDict()
        self.pins: Counter[int] = Counter()
        self.bytes = 0

    def admit(self, frame: _Frame) -> bool:
        """Admit a frame if bounded history can retain it safely."""
        with self.lock:
            if frame.gray.nbytes > self.config.history_bytes:
                return False
            while (
                len(self.frames) >= self.config.history_frames
                or self.bytes + frame.gray.nbytes > self.config.history_bytes
            ):
                removable = next((i for i in self.frames if not self.pins[i]), None)
                if removable is None:
                    return False
                self.bytes -= self.frames.pop(removable).gray.nbytes
                self.pins.pop(removable, None)
            self.frames[frame.frame_idx] = frame
            self.bytes += frame.gray.nbytes
            return True

    def lease(self, source: int, target: int | None = None) -> _Lease | None:
        """Pin a contiguous source-frame path for replay."""
        with self.lock:
            if target is None:
                target = next(reversed(self.frames), source) if self.frames else source
            if source not in self.frames or target not in self.frames or target < source:
                return None
            frames = tuple(v for i, v in self.frames.items() if source <= i <= target)
            if len(frames) != target - source + 1:
                return None
            for frame in frames:
                self.pins[frame.frame_idx] += 1
            return _Lease(self, frames)

    def release(self, frames: tuple[_Frame, ...]) -> None:
        """Release the retained frame-history lease."""
        with self.lock:
            for frame in frames:
                self.pins[frame.frame_idx] -= 1
                if self.pins[frame.frame_idx] == 0:
                    self.pins.pop(frame.frame_idx)

    def usage(self) -> dict[str, int]:
        """Return current history frame and byte accounting."""
        with self.lock:
            return {"frames": len(self.frames), "bytes": self.bytes, "pinned_frames": len(self.pins)}

    def clear(self) -> None:
        """Release all unpinned history storage."""
        with self.lock:
            if self.pins:
                raise RuntimeError("Cannot clear history while workers retain leases")
            self.frames.clear()
            self.bytes = 0
