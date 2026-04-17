from pathlib import Path

import numpy as np
from PIL import Image

from src import render
from src.scan import TilePool


def _solid_thumb(path: Path, rgb: tuple[int, int, int], size: int) -> None:
    arr = np.full((size, size, 3), rgb, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_render_assembles_canvas(tmp_path):
    _solid_thumb(tmp_path / "red.png", (255, 0, 0), 8)
    _solid_thumb(tmp_path / "blue.png", (0, 0, 255), 8)

    pool = TilePool(
        lab=np.zeros((2, 3), dtype=np.float32),
        thumbs_paths=[str(tmp_path / "red.png"), str(tmp_path / "blue.png")],
        source_paths=["red", "blue"],
    )
    index_grid = np.array([[0, 1], [1, 0]], dtype=np.int32)

    out_path = tmp_path / "mosaic.png"
    img = render.render_mosaic(index_grid, pool, tile_px=8, output_path=out_path)

    assert out_path.exists()
    assert img.size == (16, 16)
    arr = np.asarray(img)
    np.testing.assert_array_equal(
        arr[0:8, 0:8], np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        arr[0:8, 8:16], np.full((8, 8, 3), (0, 0, 255), dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        arr[8:16, 0:8], np.full((8, 8, 3), (0, 0, 255), dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        arr[8:16, 8:16], np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8)
    )


def test_render_returns_tile_usage(tmp_path):
    _solid_thumb(tmp_path / "a.png", (10, 10, 10), 4)
    _solid_thumb(tmp_path / "b.png", (200, 200, 200), 4)

    pool = TilePool(
        lab=np.zeros((2, 3), dtype=np.float32),
        thumbs_paths=[str(tmp_path / "a.png"), str(tmp_path / "b.png")],
        source_paths=["a", "b"],
    )
    index_grid = np.array([[0, 0, 1], [1, 1, 0]], dtype=np.int32)

    _, usage = render.render_mosaic_with_usage(
        index_grid, pool, tile_px=4, output_path=tmp_path / "m.png"
    )
    assert usage == {0: 3, 1: 3}


def test_tone_strength_zero_matches_phase1_bitwise(tmp_path):
    _solid_thumb(tmp_path / "r.png", (255, 0, 0), 8)
    _solid_thumb(tmp_path / "g.png", (0, 255, 0), 8)
    pool = TilePool(
        lab=np.zeros((2, 3), dtype=np.float32),
        thumbs_paths=[str(tmp_path / "r.png"), str(tmp_path / "g.png")],
        source_paths=["r", "g"],
    )
    idx = np.array([[0, 1], [1, 0]], dtype=np.int32)

    phase1 = render.render_mosaic(idx, pool, tile_px=8, output_path=tmp_path / "a.png")
    phase2 = render.render_mosaic_with_usage(
        idx, pool, tile_px=8, output_path=tmp_path / "b.png",
        target_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        tone_strength=0.0,
    )[0]

    np.testing.assert_array_equal(np.asarray(phase1), np.asarray(phase2))


def test_tone_strength_one_shifts_tiles_toward_target(tmp_path):
    from skimage.color import rgb2lab

    _solid_thumb(tmp_path / "gray.png", (128, 128, 128), 8)
    pool = TilePool(
        lab=np.zeros((1, 3), dtype=np.float32),
        thumbs_paths=[str(tmp_path / "gray.png")],
        source_paths=["gray"],
    )
    idx = np.array([[0, 0], [0, 0]], dtype=np.int32)

    target_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    target_rgb[0:8, 0:8] = (220, 40, 40)
    target_rgb[0:8, 8:16] = (40, 220, 40)
    target_rgb[8:16, 0:8] = (40, 40, 220)
    target_rgb[8:16, 8:16] = (220, 220, 40)

    img, _ = render.render_mosaic_with_usage(
        idx, pool, tile_px=8, output_path=tmp_path / "m.png",
        target_rgb=target_rgb, tone_strength=1.0,
    )
    out = np.asarray(img)
    for (r, c), expected_rgb in [
        ((0, 0), (220, 40, 40)),
        ((0, 8), (40, 220, 40)),
        ((8, 0), (40, 40, 220)),
        ((8, 8), (220, 220, 40)),
    ]:
        tile = out[r : r + 8, c : c + 8]
        tile_mean_lab = rgb2lab(tile.astype(np.float32) / 255.0).mean(axis=(0, 1))
        expected_patch = np.full((8, 8, 3), expected_rgb, dtype=np.uint8)
        expected_mean_lab = rgb2lab(expected_patch.astype(np.float32) / 255.0).mean(axis=(0, 1))
        diff = np.linalg.norm(tile_mean_lab - expected_mean_lab)
        assert diff < 5.0, f"patch at ({r},{c}) LAB ΔE = {diff}"
