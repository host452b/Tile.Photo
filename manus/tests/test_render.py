from pathlib import Path

import numpy as np

from mosaic.pool import Tile, lab_mean
from mosaic.render import compose, reinhard_transfer


def test_reinhard_tau_zero_is_identity():
    rng = np.random.default_rng(0)
    tile_lab = rng.uniform(-50, 50, size=(8, 8, 3)).astype(np.float32)
    out = reinhard_transfer(
        tile_lab,
        target_mean=np.array([90.0, 10.0, 10.0], dtype=np.float32),
        target_std=np.array([5.0, 5.0, 5.0], dtype=np.float32),
        tau=0.0,
    )
    assert np.allclose(out, tile_lab, atol=1e-5)


def test_reinhard_tau_one_matches_target_mean():
    rng = np.random.default_rng(0)
    tile_lab = rng.uniform(-50, 50, size=(8, 8, 3)).astype(np.float32)
    target_mean = np.array([90.0, 10.0, 10.0], dtype=np.float32)
    target_std = tile_lab.reshape(-1, 3).std(axis=0).astype(np.float32)
    out = reinhard_transfer(tile_lab, target_mean=target_mean, target_std=target_std, tau=1.0)
    assert np.allclose(out.reshape(-1, 3).mean(axis=0), target_mean, atol=1e-3)


def _solid_tile(name: str, rgb: tuple[int, int, int], px: int = 4) -> Tile:
    thumb = np.full((px, px, 3), rgb, dtype=np.uint8)
    return Tile(path=Path(name), lab=lab_mean(thumb), thumb=thumb)


def test_compose_output_shape():
    tiles = [_solid_tile("a.png", (255, 0, 0)), _solid_tile("b.png", (0, 0, 255))]
    grid = np.array([[0, 1], [1, 0]], dtype=np.int64)
    cell_lab_means = np.zeros((2, 2, 3), dtype=np.float32)
    out = compose(tiles, grid, cell_lab_means, tile_px=4, tau=0.0)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8


def test_compose_places_correct_color_at_tau_zero():
    tiles = [_solid_tile("a.png", (255, 0, 0)), _solid_tile("b.png", (0, 0, 255))]
    grid = np.array([[0, 1]], dtype=np.int64)
    cell_lab_means = np.zeros((1, 2, 3), dtype=np.float32)
    out = compose(tiles, grid, cell_lab_means, tile_px=4, tau=0.0)
    assert out[0, 0, 0] > 200 and out[0, 0, 2] < 50
    assert out[0, 7, 2] > 200 and out[0, 7, 0] < 50
