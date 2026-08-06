"""Windows-safe image helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read through bytes so non-ASCII Windows paths work reliably with OpenCV."""
    path = Path(path)
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Unable to encode image: {path}")
    encoded.tofile(path)


def binary_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a grayscale or color annotation to a 0/1 mask."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 0).astype(np.uint8)


def resize_pair(image: np.ndarray, mask: np.ndarray, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    resized_image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(binary_mask(mask), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return resized_image, binary_mask(resized_mask)
