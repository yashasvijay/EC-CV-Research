"""Dataset adapters. Ground truth is kept separate from inference frames."""

from .mot17 import MOT17Dataset, MOT17Sequence

__all__ = ["MOT17Dataset", "MOT17Sequence"]
