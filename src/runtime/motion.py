from ..motion import advance_batch
import numpy as np
from .packets import RuntimeConfig
from ..features import select_roi_features


def _owned(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True, order="C")
    result.flags.writeable = False
    return result


def _advance_points(
    previous: np.ndarray, current: np.ndarray, points: np.ndarray, box: np.ndarray, config: RuntimeConfig
) -> tuple[np.ndarray, np.ndarray, float]:
    update = advance_batch(
        previous, current, [box], [points], min_points=config.min_points, fb_threshold=config.fb_threshold
    )[0]
    return update.box, update.points, update.quality


def _seed(gray: np.ndarray, box: np.ndarray, config: RuntimeConfig) -> np.ndarray:
    return select_roi_features(gray, box, config.max_points)
