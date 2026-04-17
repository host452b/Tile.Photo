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


def test_scan_tile_pool_happy_path(tmp_path):
    """Scan returns TileRecord list and empty bad_files on clean JPGs."""
    from mosaic_core import scan_tile_pool, ensure_seed_tiles
    ensure_seed_tiles(tmp_path / "tiles", n=5)
    cache = tmp_path / "cache.pkl"
    tiles, bad = scan_tile_pool(tmp_path / "tiles", cache)
    assert len(tiles) == 5
    assert bad == []
    for t in tiles:
        assert t.lab_mean.shape == (3,)
        assert t.rgb_thumb.shape == (64, 64, 3)


def test_scan_tile_pool_skips_corrupt_files(tmp_path):
    """Corrupt JPG goes into bad_files and does not crash the scan."""
    from mosaic_core import scan_tile_pool, ensure_seed_tiles
    d = tmp_path / "tiles"
    ensure_seed_tiles(d, n=3)
    (d / "broken.jpg").write_bytes(b"not a real jpeg")
    cache = tmp_path / "cache.pkl"
    tiles, bad = scan_tile_pool(d, cache)
    assert len(tiles) == 3
    assert len(bad) == 1
    assert bad[0].name == "broken.jpg"


def test_scan_tile_pool_uses_cache(tmp_path):
    """Second call with same cache is trivial — cache file exists after first run."""
    from mosaic_core import scan_tile_pool, ensure_seed_tiles
    d = tmp_path / "tiles"
    ensure_seed_tiles(d, n=4)
    cache = tmp_path / "cache.pkl"
    tiles1, _ = scan_tile_pool(d, cache)
    assert cache.exists()
    tiles2, _ = scan_tile_pool(d, cache)
    assert len(tiles1) == len(tiles2)
    assert (tiles1[0].lab_mean == tiles2[0].lab_mean).all()


def test_build_report_structural_fields(tmp_path):
    """build_report returns a ReportBundle whose text contains key headlines."""
    import numpy as np
    from mosaic_core import TileRecord, build_report, ensure_seed_tiles, scan_tile_pool
    ensure_seed_tiles(tmp_path / "tiles", n=20)
    records, bad = scan_tile_pool(tmp_path / "tiles", tmp_path / "cache.pkl")
    assignment = np.zeros((4, 5), dtype=np.int64)  # all cells use tile 0
    bundle = build_report(assignment, records, elapsed_seconds=3.14, bad_files=bad)
    assert "扫到" in bundle.text or "tiles" in bundle.text
    assert "冷宫" in bundle.text or "cold" in bundle.text
    assert "3.14" in bundle.text or "3.1" in bundle.text
    # Figures are matplotlib Figures.
    import matplotlib
    assert isinstance(bundle.usage_bar_fig, matplotlib.figure.Figure)
    assert isinstance(bundle.cold_wall_fig, matplotlib.figure.Figure)
