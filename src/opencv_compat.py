"""Typed adapter for OpenCV's supported optional LK output buffer."""

from typing import Protocol, cast
import cv2
from .types import Array


class _LKFunction(Protocol):
    def __call__(
        self,
        previous: Array,
        current: Array,
        points: Array,
        output: None,
        *,
        winSize: tuple[int, int],
        maxLevel: int,
    ) -> tuple[Array | None, Array | None, Array | None]: ...


def sparse_lk(
    previous: Array, current: Array, points: Array
) -> tuple[Array | None, Array | None, Array | None]:
    """Run LK with an allocated output; upstream stubs incorrectly reject None."""
    return cast(_LKFunction, cv2.calcOpticalFlowPyrLK)(
        previous, current, points, None, winSize=(21, 21), maxLevel=3
    )
