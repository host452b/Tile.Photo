import numpy as np
from PIL import Image
from pathlib import Path
from src.renderer import reinhard_tone_transfer


def _solid_rgb(color: tuple[int, int, int], size=(16, 16)) -> np.ndarray:
    return np.full((size[1], size[0], 3), color, dtype=np.uint8)


def test_tau_zero_returns_source_unchanged():
    src = _solid_rgb((150, 40, 40))
    target = _solid_rgb((40, 40, 150))
    out = reinhard_tone_transfer(src, target, tau=0.0)
    np.testing.assert_array_equal(out, src)


def test_tau_one_matches_target_mean_closely():
    src = _solid_rgb((150, 40, 40))
    target = _solid_rgb((40, 40, 150))
    out = reinhard_tone_transfer(src, target, tau=1.0)
    # After full transfer, mean color should be very close to target's
    src_mean = src.reshape(-1, 3).mean(axis=0)
    out_mean = out.reshape(-1, 3).mean(axis=0)
    target_mean = target.reshape(-1, 3).mean(axis=0)
    # Closer to target than to source
    assert np.linalg.norm(out_mean - target_mean) < np.linalg.norm(out_mean - src_mean)


def test_tau_interpolates_linearly_in_lab():
    src = _solid_rgb((150, 40, 40))
    target = _solid_rgb((40, 40, 150))
    half = reinhard_tone_transfer(src, target, tau=0.5)
    full = reinhard_tone_transfer(src, target, tau=1.0)
    # Half-strength output should lie between src and full-transfer
    src_mean = src.reshape(-1, 3).mean(axis=0)
    half_mean = half.reshape(-1, 3).mean(axis=0)
    full_mean = full.reshape(-1, 3).mean(axis=0)
    assert np.linalg.norm(half_mean - src_mean) < np.linalg.norm(full_mean - src_mean)
    assert np.linalg.norm(half_mean - full_mean) < np.linalg.norm(src_mean - full_mean)
