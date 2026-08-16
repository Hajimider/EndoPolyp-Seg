"""Small, explainable post-processing operations for binary masks."""

from __future__ import annotations

import cv2
import numpy as np


def clean_mask(mask: np.ndarray, min_area: int = 0, kernel_size: int = 0) -> np.ndarray:
    """Remove tiny components and optionally close small holes."""
    result = (np.asarray(mask) > 0).astype(np.uint8)
    if min_area > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
        keep = np.zeros_like(result)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == label] = 1
        result = keep
    if kernel_size > 0:
        size = int(kernel_size)
        if size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result
