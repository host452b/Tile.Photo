import numpy as np
import pytest
from PIL import Image

from mosaic_core import split_target


def test_split_target_shape_and_dtype():
    """split_target(img, grid_w=10, grid_h=5) must return ndarray shape (5, 10, 3) float32."""
    img = Image.new("RGB", (100, 50), (128, 128, 128))  # 100 px wide, 50 px tall
    lab_grid = split_target(img, grid_w=10, grid_h=5)
    assert lab_grid.shape == (5, 10, 3)
    assert lab_grid.dtype == np.float32


def test_split_target_uniform_gray_constant_lab():
    """Uniform gray input -> all cells share the same LAB value (L~54, a~0, b~0)."""
    img = Image.new("RGB", (80, 40), (128, 128, 128))
    lab_grid = split_target(img, grid_w=8, grid_h=4)
    L_plane = lab_grid[..., 0]
    # All cells equal within float noise.
    assert L_plane.max() - L_plane.min() < 0.01
    # L ~= 54 for sRGB 128 gray.
    assert 52 < L_plane.mean() < 56


def test_render_mosaic_output_size():
    """Output PIL image size = (grid_w * tile_px, grid_h * tile_px)."""
    from mosaic_core import TileRecord, render_mosaic
    # One deterministic tile: solid red 16×16.
    tile_rgb = np.full((16, 16, 3), fill_value=0, dtype=np.uint8)
    tile_rgb[..., 0] = 255
    tile_lab = np.array([53.24, 80.09, 67.20], dtype=np.float32)
    records = [TileRecord(path=None, lab_mean=tile_lab, rgb_thumb=tile_rgb)]
    # 4×3 grid, all cells point to tile 0.
    assignment = np.zeros((3, 4), dtype=np.int64)
    target_lab = np.broadcast_to(tile_lab, (3, 4, 3)).copy()
    img = render_mosaic(assignment, records, tile_px=16, tau=0.0, target_lab=target_lab)
    assert img.size == (4 * 16, 3 * 16)


def test_render_mosaic_tau_zero_preserves_tile_bytes():
    """τ=0 means every cell is the tile's raw rgb_thumb (byte-exact)."""
    from mosaic_core import TileRecord, render_mosaic
    rng = np.random.default_rng(7)
    tile_rgb = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    tile_lab = np.array([50.0, 0.0, 0.0], dtype=np.float32)
    records = [TileRecord(path=None, lab_mean=tile_lab, rgb_thumb=tile_rgb)]
    assignment = np.zeros((2, 2), dtype=np.int64)
    target_lab = np.full((2, 2, 3), [30.0, 20.0, 5.0], dtype=np.float32)  # different from tile
    img = render_mosaic(assignment, records, tile_px=16, tau=0.0, target_lab=target_lab)
    out = np.asarray(img)
    # Top-left 16x16 block must equal tile_rgb.
    np.testing.assert_array_equal(out[:16, :16], tile_rgb)


def test_ensure_seed_tiles_creates_files(tmp_path):
    """ensure_seed_tiles on an empty path creates n JPEG files."""
    from mosaic_core import ensure_seed_tiles
    target_dir = tmp_path / "seeds"
    ensure_seed_tiles(target_dir, n=10)
    jpgs = list(target_dir.glob("*.jpg"))
    assert len(jpgs) == 10
    # Each file should be a readable 64×64 image.
    for p in jpgs:
        img = Image.open(p)
        assert img.size == (64, 64)


def test_ensure_seed_tiles_noop_when_nonempty(tmp_path):
    """If the dir already has images, ensure_seed_tiles must not add more."""
    from mosaic_core import ensure_seed_tiles
    target_dir = tmp_path / "existing"
    target_dir.mkdir()
    (target_dir / "user.jpg").write_bytes(b"fake")
    ensure_seed_tiles(target_dir, n=5)
    assert len(list(target_dir.glob("*"))) == 1
