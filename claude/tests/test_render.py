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
