from pathlib import Path

import numpy as np
from PIL import Image

from src import match, render, scan


PALETTE = [
    (220, 20, 20),
    (20, 220, 20),
    (20, 20, 220),
    (220, 220, 20),
    (220, 20, 220),
    (20, 220, 220),
    (240, 240, 240),
    (20, 20, 20),
    (200, 120, 60),
    (60, 120, 200),
]


def _solid_jpg(path: Path, rgb, size: int = 96) -> None:
    arr = np.full((size, size, 3), rgb, dtype=np.uint8)
    Image.fromarray(arr).save(path, quality=95)


def test_end_to_end_pure_colors_match_exactly(tmp_path):
    base = tmp_path / "photos"
    base.mkdir()
    for i, rgb in enumerate(PALETTE):
        _solid_jpg(base / f"c{i}.jpg", rgb)

    pool = scan.build_pool(base, tmp_path / "cache", tile_px=24)

    grid_h, grid_w = 4, 10
    tile_px = 24
    target_rgb = np.zeros((grid_h * tile_px, grid_w * tile_px, 3), dtype=np.uint8)
    expected_color = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for col in range(grid_w):
        rgb = PALETTE[col]
        target_rgb[:, col * tile_px : (col + 1) * tile_px] = rgb
        expected_color[:, col] = rgb

    from skimage.color import rgb2lab

    target_lab = rgb2lab(target_rgb.astype(np.float32) / 255.0)
    target_lab_grid = target_lab.reshape(
        grid_h, tile_px, grid_w, tile_px, 3
    ).mean(axis=(1, 3)).astype(np.float32)

    idx = match.match_grid(target_lab_grid, pool.lab)
    img = render.render_mosaic(idx, pool, tile_px, tmp_path / "out.png")

    out = np.asarray(img)
    for row in range(grid_h):
        for col in range(grid_w):
            tile_rgb = out[
                row * tile_px : (row + 1) * tile_px,
                col * tile_px : (col + 1) * tile_px,
            ]
            mean = tile_rgb.reshape(-1, 3).mean(axis=0)
            diff = np.abs(mean - expected_color[row, col]).max()
            assert diff < 8, (
                f"cell ({row},{col}) off: got {mean}, expected {expected_color[row, col]}"
            )


def test_end_to_end_with_knobs(tmp_path):
    base = tmp_path / "photos"
    base.mkdir()
    for i, rgb in enumerate(PALETTE):
        _solid_jpg(base / f"c{i}.jpg", rgb)

    pool = scan.build_pool(base, tmp_path / "cache", tile_px=24)

    grid_h, grid_w = 4, 10
    tile_px = 24
    target_rgb = np.zeros((grid_h * tile_px, grid_w * tile_px, 3), dtype=np.uint8)
    for col in range(grid_w):
        target_rgb[:, col * tile_px : (col + 1) * tile_px] = PALETTE[col]

    from skimage.color import rgb2lab

    target_lab = rgb2lab(target_rgb.astype(np.float32) / 255.0)
    target_lab_grid = target_lab.reshape(
        grid_h, tile_px, grid_w, tile_px, 3
    ).mean(axis=(1, 3)).astype(np.float32)

    idx = match.match_grid(target_lab_grid, pool.lab, lambda_=2.0, mu=10.0)
    img, usage = render.render_mosaic_with_usage(
        idx, pool, tile_px, tmp_path / "out.png",
        target_rgb=target_rgb, tone_strength=0.3,
    )
    assert img.size == (grid_w * tile_px, grid_h * tile_px)
    assert len(usage) >= len(PALETTE) // 2
