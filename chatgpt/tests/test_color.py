import numpy as np
import pytest

from mosaic_core import lab_mean, ciede2000


def test_lab_mean_pure_red():
    """Pure sRGB red has LAB ~= (53.24, 80.09, 67.20)."""
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    result = lab_mean(rgb)
    expected = np.array([53.24, 80.09, 67.20], dtype=np.float32)
    assert result.shape == (3,)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, atol=0.5)


def test_ciede2000_identity_is_zero():
    """ΔE between a LAB triple and itself must be (numerically) zero."""
    lab = np.array([50.0, 10.0, -5.0], dtype=np.float32)
    assert ciede2000(lab, lab) < 1e-6


def test_ciede2000_nonzero_for_different_colors():
    """Two clearly different LAB points must have positive ΔE."""
    a = np.array([50.0, 80.0, 0.0], dtype=np.float32)
    b = np.array([50.0, -80.0, 0.0], dtype=np.float32)
    assert ciede2000(a, b) > 10.0
