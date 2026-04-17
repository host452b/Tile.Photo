from pathlib import Path

import numpy as np

from mosaic.pool import Tile, lab_mean
from mosaic.report import cold_wall, describe_lab, text_report, usage_hist_figure


def _tile(name, rgb, px=4):
    thumb = np.full((px, px, 3), rgb, dtype=np.uint8)
    return Tile(path=Path(name), lab=lab_mean(thumb), thumb=thumb)


def test_text_report_contains_totals():
    tiles = [_tile(f"{i}.png", (i * 10 % 255, 0, 0)) for i in range(5)]
    uses = np.array([10, 0, 3, 0, 7], dtype=np.int64)
    report = text_report(
        tiles, uses, grid_shape=(4, 5), params={"lambda_": 2.0, "mu": 0.5, "tau": 0.5}
    )
    assert "Tiles placed: 20" in report
    assert "Unique tiles used: 3" in report
    assert "0.png" in report
    assert "λ=2.0" in report


def test_cold_wall_returns_image_with_correct_count():
    tiles = [_tile(f"{i}.png", (i * 20 % 255, 0, 0), px=4) for i in range(6)]
    uses = np.array([5, 0, 0, 0, 2, 0], dtype=np.int64)
    wall = cold_wall(tiles, uses, n=4, thumb_px=4, cols=2)
    assert wall.shape == (8, 8, 3)
    assert wall.dtype == np.uint8


def test_usage_hist_figure_runs_without_error():
    tiles = [_tile(f"{i}.png", (0, 0, 0)) for i in range(3)]
    uses = np.array([1, 2, 3], dtype=np.int64)
    fig = usage_hist_figure(tiles, uses)
    assert fig is not None


def test_describe_lab_covers_known_colors():
    assert describe_lab(np.array([70.0, -5.0, -30.0])) == "sky blue"
    assert describe_lab(np.array([65.0, 15.0, 25.0])) == "skin tone"
