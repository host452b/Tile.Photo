import numpy as np
import pytest

from mosaic_core import lab_mean


def test_lab_mean_pure_red():
    """Pure sRGB red has LAB ~= (53.24, 80.09, 67.20)."""
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    result = lab_mean(rgb)
    expected = np.array([53.24, 80.09, 67.20], dtype=np.float32)
    assert result.shape == (3,)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, atol=0.5)
