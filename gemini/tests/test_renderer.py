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


def test_render_mosaic_writes_expected_shape(tmp_path: Path):
    # 3 tiles (solid R, G, B), 2x3 grid
    tiles = [tmp_path / f"t{i}.jpg" for i in range(3)]
    colors = [(200, 20, 20), (20, 200, 20), (20, 20, 200)]
    for p, c in zip(tiles, colors):
        Image.fromarray(_solid_rgb(c, size=(32, 32))).save(p)
    assignment = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
    from src.renderer import render_mosaic
    # Passing tau=0 means source tile colors are preserved
    cell_rgb = np.zeros((2, 3, 2, 2, 3), dtype=np.uint8)  # placeholder target, irrelevant at tau=0
    img, usage = render_mosaic(assignment, [str(p) for p in tiles], cell_rgb, tile_px=16, tau=0.0)
    assert img.size == (3 * 16, 2 * 16)
    assert usage == {0: 2, 1: 2, 2: 2}


def test_render_mosaic_uses_tone_transfer_when_tau_nonzero(tmp_path: Path):
    # One vivid red tile; target cell is blue. At tau=1, the rendered pixel should be bluish.
    tile = tmp_path / "red.jpg"
    Image.fromarray(_solid_rgb((200, 20, 20), size=(32, 32))).save(tile)
    assignment = np.array([[0]], dtype=np.int64)
    cell_rgb = np.full((1, 1, 2, 2, 3), (20, 20, 200), dtype=np.uint8)
    from src.renderer import render_mosaic
    img, _ = render_mosaic(assignment, [str(tile)], cell_rgb, tile_px=16, tau=1.0)
    arr = np.asarray(img)
    # Mean B channel should now exceed mean R channel (original was opposite)
    assert arr[..., 2].mean() > arr[..., 0].mean()
