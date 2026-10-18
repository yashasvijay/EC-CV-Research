"""Array boundary type for OpenCV and variable-dtype NumPy buffers."""

from typing import Any, TypeAlias
import numpy as np

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]
