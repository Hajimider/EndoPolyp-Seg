import numpy as np
import pytest

from polypseg.postprocess import clean_mask


def test_clean_mask_removes_small_component_and_closes_hole() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:12, 2:12] = 1
    mask[6:8, 6:8] = 0
    mask[17, 17] = 1
    result = clean_mask(mask, min_area=10, kernel_size=5)
    assert result[17, 17] == 0
    assert result[6:8, 6:8].all()


def test_clean_mask_requires_odd_kernel() -> None:
    with pytest.raises(ValueError):
        clean_mask(np.ones((5, 5), dtype=np.uint8), kernel_size=4)
