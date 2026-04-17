import numpy as np

from mosaic.transfer import reinhard_transfer


def test_tau_zero_returns_identity():
    tile = np.random.RandomState(0).randint(0, 255, (64, 64, 3), dtype=np.uint8)
    target_patch = np.full((64, 64, 3), 128, dtype=np.uint8)
    out = reinhard_transfer(tile, target_patch, tau=0.0)
    assert np.array_equal(out, tile)


def test_tau_one_matches_target_mean_in_lab():
    from skimage.color import rgb2lab
    tile = np.full((64, 64, 3), 50, dtype=np.uint8)
    target_patch = np.full((64, 64, 3), 200, dtype=np.uint8)
    out = reinhard_transfer(tile, target_patch, tau=1.0)
    out_lab = rgb2lab(out / 255.0).reshape(-1, 3).mean(axis=0)
    target_lab = rgb2lab(target_patch / 255.0).reshape(-1, 3).mean(axis=0)
    # L channel should be close (a,b can stay at 0 since both inputs are gray)
    assert abs(out_lab[0] - target_lab[0]) < 2.0


def test_tau_half_between_endpoints():
    tile = np.full((64, 64, 3), 50, dtype=np.uint8)
    target_patch = np.full((64, 64, 3), 200, dtype=np.uint8)
    out_0 = reinhard_transfer(tile, target_patch, tau=0.0)
    out_1 = reinhard_transfer(tile, target_patch, tau=1.0)
    out_h = reinhard_transfer(tile, target_patch, tau=0.5)
    assert out_0.mean() < out_h.mean() < out_1.mean()


def test_zero_std_channel_does_not_divide_by_zero():
    """纯色 tile σ=0，不该崩溃。"""
    tile = np.full((64, 64, 3), 100, dtype=np.uint8)
    target_patch = np.random.RandomState(0).randint(0, 255, (64, 64, 3), dtype=np.uint8)
    out = reinhard_transfer(tile, target_patch, tau=1.0)
    assert out.shape == tile.shape
    assert np.isfinite(out).all()
