from __future__ import annotations

from collections import Counter

from src.background import hall_of_oblivion_color
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


def test_single_rare_char_defines_background():
    grid = [
        [" ", " ", " ", "@"],
        [" ", " ", " ", " "],
    ]
    colors = [
        [(10, 10, 10), (10, 10, 10), (10, 10, 10), (200, 50, 50)],
        [(10, 10, 10), (10, 10, 10), (10, 10, 10), (10, 10, 10)],
    ]
    bg = hall_of_oblivion_color(_make(grid, colors), pct=0.25)
    assert bg == (200, 50, 50)


def test_uniform_grid_falls_back_to_mean_color():
    grid = [["#", "#"], ["#", "#"]]
    colors = [[(0, 0, 0), (100, 100, 100)], [(50, 50, 50), (150, 150, 150)]]
    bg = hall_of_oblivion_color(_make(grid, colors), pct=0.2)
    assert bg == (75, 75, 75)


def test_pct_one_includes_all_chars():
    grid = [
        ["a", "b"],
        ["c", "d"],
    ]
    colors = [
        [(0, 0, 0), (80, 0, 0)],
        [(0, 80, 0), (0, 0, 80)],
    ]
    bg = hall_of_oblivion_color(_make(grid, colors), pct=1.0)
    assert bg == (20, 20, 20)


def test_pct_is_clamped():
    grid = [["x", "y"]]
    colors = [[(0, 0, 0), (200, 200, 200)]]
    assert hall_of_oblivion_color(_make(grid, colors), pct=-1.0) == hall_of_oblivion_color(
        _make(grid, colors), pct=0.0
    )
    assert hall_of_oblivion_color(_make(grid, colors), pct=5.0) == hall_of_oblivion_color(
        _make(grid, colors), pct=1.0
    )


def test_picks_bottom_pct_of_used_chars_by_frequency():
    grid = [
        ["a", "a", "a", "a"],
        ["a", "a", "b", "c"],
    ]
    colors = [
        [(10, 10, 10), (10, 10, 10), (10, 10, 10), (10, 10, 10)],
        [(10, 10, 10), (10, 10, 10), (200, 0, 0), (0, 0, 200)],
    ]
    bg = hall_of_oblivion_color(_make(grid, colors), pct=0.5)
    assert bg == (100, 0, 100)
