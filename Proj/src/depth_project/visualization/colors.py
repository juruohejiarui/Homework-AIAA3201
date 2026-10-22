"""Fixed-range depth color mapping."""
import cv2
import numpy as np


def colorize(values, lower, upper, reverse=False):
    valid = np.isfinite(values)
    normalized = np.clip((np.nan_to_num(values, nan=lower) - lower) / max(upper - lower, 1e-6), 0, 1)
    if reverse:
        normalized = 1 - normalized
    colors = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colors[~valid] = 0
    return colors
