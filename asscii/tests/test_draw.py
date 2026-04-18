from __future__ import annotations

import numpy as np
from collections import Counter

from src.draw import draw_ascii
from src.render import AsciiResult


def _make(grid: list[list[str]], colors: list[list[tuple[int, int, int]]]) -> AsciiResult:
    usage: Counter[str] = Counter()
    for row in grid:
        usage.update(row)
    return AsciiResult(
        grid=grid,
        colors=colors,
        char_usage=usage,
        cols=len(grid[0]),
        rows=len(grid),
    )


def test_output_size_matches_grid_times_cell_px():
    grid = [[" ", " "], [" ", " "]]
    colors = [[(0, 0, 0)] * 2, [(0, 0, 0)] * 2]
    img = draw_ascii(_make(grid, colors), bg=(30, 30, 30), cell_px=16)
    assert img.size == (32, 32)


def test_background_color_fills_spaces():
    grid = [[" ", " "]]
    colors = [[(0, 0, 0), (0, 0, 0)]]
    img = draw_ascii(_make(grid, colors), bg=(200, 50, 50), cell_px=16)
    arr = np.asarray(img)
    assert arr[:, :, 0].min() == 200
    assert arr[:, :, 1].min() == 50
    assert arr[:, :, 2].min() == 50
    assert arr[0, 0].tolist() == [200, 50, 50]


def test_non_space_cell_draws_pixels_distinct_from_background():
    grid = [["@"]]
    colors = [[(255, 255, 255)]]
    img = draw_ascii(_make(grid, colors), bg=(0, 0, 0), cell_px=20)
    arr = np.asarray(img)
    assert (arr != 0).any()


def test_cell_color_honored_for_drawn_glyph():
    grid = [["@"]]
    colors = [[(255, 0, 0)]]
    img = draw_ascii(_make(grid, colors), bg=(0, 0, 0), cell_px=20)
    arr = np.asarray(img)
    non_bg = arr[(arr != [0, 0, 0]).any(axis=-1)]
    assert (non_bg[:, 0] >= non_bg[:, 1]).all()
    assert (non_bg[:, 0] >= non_bg[:, 2]).all()
