"""Binary segmentation metrics used by training, evaluation, and baselines."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


METRIC_NAMES = ("dice", "iou", "precision", "recall")


def binary_metrics(target: np.ndarray, prediction: np.ndarray, eps: float = 1e-7) -> dict[str, float]:
    target_bool = np.asarray(target, dtype=bool)
    prediction_bool = np.asarray(prediction, dtype=bool)
    if target_bool.shape != prediction_bool.shape:
        raise ValueError(f"Mask shapes differ: {target_bool.shape} vs {prediction_bool.shape}")

    true_positive = float(np.logical_and(target_bool, prediction_bool).sum())
    false_positive = float(np.logical_and(~target_bool, prediction_bool).sum())
    false_negative = float(np.logical_and(target_bool, ~prediction_bool).sum())
    return {
        "dice": (2.0 * true_positive + eps) / (2.0 * true_positive + false_positive + false_negative + eps),
        "iou": (true_positive + eps) / (true_positive + false_positive + false_negative + eps),
        "precision": (true_positive + eps) / (true_positive + false_positive + eps),
        "recall": (true_positive + eps) / (true_positive + false_negative + eps),
    }


def mean_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    values = list(rows)
    if not values:
        return {name: 0.0 for name in METRIC_NAMES}
    return {name: float(np.mean([row[name] for row in values])) for name in METRIC_NAMES}
