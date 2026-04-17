"""Tests for src/mosaic/render.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _single_tile_pool():
    from skimage.color import rgb2lab

    from mosaic.tiles import TilePool

    # One red tile, 8×8
    thumb = np.full((8, 8, 3), [200, 40, 40], dtype=np.uint8)
    lab_mean = (
        rgb2lab(thumb.astype(np.float32) / 255.0)
        .reshape(-1, 3)
        .mean(0)
        .astype(np.float32)
    )
    return TilePool(
        paths=["red.png"],
        lab_means=lab_mean.reshape(1, 3),
        thumbnails=thumb[None, ...],
    )


def test_render_tau_zero_preserves_tile() -> None:
    """τ=0 means no tone transfer — output block equals tile's original pixels."""
    from mosaic.config import MosaicConfig
    from mosaic.render import render_mosaic

    pool = _single_tile_pool()
    target_cells = np.array([[[50.0, 0.0, 0.0]]], dtype=np.float32)  # 1×1 gray target
    assignment = np.array([[0]], dtype=np.int32)
    cfg = MosaicConfig(
        tile_source_dir=Path("/tmp"), target_image=Path("/tmp"),
        grid=(1, 1), tile_px=8, tau_tone=0.0, verbose=False,
    )
    img = render_mosaic(assignment, pool, target_cells, cfg)
    arr = np.asarray(img)
    assert arr.shape == (8, 8, 3)
    np.testing.assert_array_equal(arr, pool.thumbnails[0])


def test_render_tau_one_matches_target_lab() -> None:
    """τ=1 means full tone transfer — output LAB mean ≈ target LAB."""
    from skimage.color import rgb2lab

    from mosaic.config import MosaicConfig
    from mosaic.render import render_mosaic

    pool = _single_tile_pool()
    target_lab = np.array([50.0, 0.0, 0.0], dtype=np.float32)  # neutral gray
    target_cells = target_lab.reshape(1, 1, 3)
    assignment = np.array([[0]], dtype=np.int32)
    cfg = MosaicConfig(
        tile_source_dir=Path("/tmp"), target_image=Path("/tmp"),
        grid=(1, 1), tile_px=8, tau_tone=1.0, verbose=False,
    )
    img = render_mosaic(assignment, pool, target_cells, cfg)
    arr = np.asarray(img).astype(np.float32) / 255.0
    out_lab_mean = rgb2lab(arr).reshape(-1, 3).mean(0)
    np.testing.assert_allclose(out_lab_mean, target_lab, atol=2.0)
