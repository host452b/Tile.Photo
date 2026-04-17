"""Tests for src/mosaic/match.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def two_tone_target(tmp_path: Path) -> Path:
    """100×100 image: left half red (255,0,0), right half blue (0,0,255)."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :50] = [255, 0, 0]
    arr[:, 50:] = [0, 0, 255]
    p = tmp_path / "target.png"
    Image.fromarray(arr).save(p)
    return p


def test_split_target_shape_and_lab(two_tone_target: Path) -> None:
    from mosaic.match import split_target

    cells = split_target(two_tone_target, grid=(4, 2))  # 4 cols × 2 rows
    assert cells.shape == (2, 4, 3)  # (rows, cols, LAB)
    # Left two columns are red, right two columns are blue.
    # In CIE LAB: red has +b (~67), blue has strongly -b (~-108); both have +a,
    # so b is the clean separator.
    left_b = cells[:, :2, 2].mean()
    right_b = cells[:, 2:, 2].mean()
    assert left_b > 20  # red has positive b
    assert right_b < -20  # blue has strongly negative b


from mosaic.tiles import TilePool


def _synthetic_pool(tile_px: int = 8) -> TilePool:
    """5 solid-color tiles — gray, warm-light, cool-dark, red, green."""
    from skimage.color import rgb2lab

    colors = [
        (128, 128, 128),   # gray
        (230, 200, 180),   # warm light
        (40, 50, 70),      # cool dark
        (220, 40, 40),     # red
        (40, 200, 60),     # green
    ]
    thumbs = np.stack([np.full((tile_px, tile_px, 3), c, dtype=np.uint8) for c in colors])
    lab_means = np.stack(
        [rgb2lab(t.astype(np.float32) / 255.0).reshape(-1, 3).mean(0) for t in thumbs]
    ).astype(np.float32)
    return TilePool(
        paths=[f"tile_{i}.png" for i in range(len(colors))],
        lab_means=lab_means,
        thumbnails=thumbs,
    )


def test_match_all_tiles_picks_nearest_color_when_no_penalty() -> None:
    """With λ=μ=0, each cell should pick the color-closest tile."""
    from mosaic.config import MosaicConfig
    from mosaic.match import match_all_tiles

    pool = _synthetic_pool()
    # Build target cells that each match one tile exactly
    target = pool.lab_means.reshape(1, 5, 3).copy()  # 1×5 grid, each cell = one tile's LAB

    cfg = MosaicConfig(
        tile_source_dir=Path("/tmp"),
        target_image=Path("/tmp"),
        grid=(5, 1),
        tile_px=8,
        candidate_k=5,
        lambda_reuse=0.0,
        mu_neighbor=0.0,
        verbose=False,
    )
    assignment, use_count = match_all_tiles(target, pool, cfg)
    assert assignment.shape == (1, 5)
    # Each cell must pick its own tile (identity mapping)
    np.testing.assert_array_equal(assignment[0], np.arange(5))
    assert sum(use_count.values()) == 5
