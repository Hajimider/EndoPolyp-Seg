import numpy as np
import pytest

from polypseg.metrics import binary_metrics, mean_metrics


def test_binary_metrics_for_partial_overlap() -> None:
    target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    prediction = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    result = binary_metrics(target, prediction)
    assert result["dice"] == pytest.approx(0.5)
    assert result["iou"] == pytest.approx(1.0 / 3.0)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)


def test_binary_metrics_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="Mask shapes differ"):
        binary_metrics(np.zeros((2, 2)), np.zeros((3, 3)))


def test_mean_metrics_handles_empty_rows() -> None:
    assert mean_metrics([]) == {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
