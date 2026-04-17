import numpy as np
import pytest

from mosaic_core import lab_mean, ciede2000, reinhard_transfer


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


def test_reinhard_tau_zero_returns_original():
    """τ=0 must short-circuit and return the input tile unchanged (byte-exact)."""
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    target_lab = np.array([60.0, 20.0, -10.0], dtype=np.float32)
    out = reinhard_transfer(rgb, target_lab, tau=0.0)
    np.testing.assert_array_equal(out, rgb)


def test_reinhard_tau_one_matches_target_mean():
    """τ=1 must drag the tile's LAB mean onto target_lab_mean (within round-trip error)."""
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    target_lab = np.array([60.0, 10.0, -5.0], dtype=np.float32)
    out = reinhard_transfer(rgb, target_lab, tau=1.0)
    got = lab_mean(out)
    # sRGB clipping + round-trip introduces ~1-3 unit drift; 4.0 is safe.
    np.testing.assert_allclose(got, target_lab, atol=4.0)
