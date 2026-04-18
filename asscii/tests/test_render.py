from __future__ import annotations

from PIL import Image

from src.render import render_ascii


def _solid(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color)


def test_grid_dimensions_match_cols_and_rows():
    img = _solid((40, 20), (128, 128, 128))
    res = render_ascii(img, cols=8, rows=4, ramp=" .:-=+*#")
    assert res.rows == 4 and res.cols == 8
    assert len(res.grid) == 4
    assert all(len(row) == 8 for row in res.grid)
    assert len(res.colors) == 4
    assert all(len(row) == 8 for row in res.colors)


def test_white_image_uses_lightest_char():
    img = _solid((16, 16), (255, 255, 255))
    ramp = " .:-=+*#"
    res = render_ascii(img, cols=4, rows=4, ramp=ramp)
    for row in res.grid:
        for ch in row:
            assert ch == ramp[0]


def test_black_image_uses_densest_char():
    img = _solid((16, 16), (0, 0, 0))
    ramp = " .:-=+*#"
    res = render_ascii(img, cols=4, rows=4, ramp=ramp)
    for row in res.grid:
        for ch in row:
            assert ch == ramp[-1]


def test_every_emitted_char_is_drawn_from_ramp():
    img = Image.effect_mandelbrot((64, 64), (-2, -1.5, 1, 1.5), 80).convert("RGB")
    ramp = " .:-=+*#"
    res = render_ascii(img, cols=16, rows=16, ramp=ramp)
    flat = {ch for row in res.grid for ch in row}
    assert flat.issubset(set(ramp))


def test_char_usage_counts_every_cell_once():
    img = _solid((32, 32), (200, 50, 50))
    res = render_ascii(img, cols=8, rows=8, ramp=" .:-=+*#")
    assert sum(res.char_usage.values()) == 64


def test_per_cell_color_matches_source_block():
    img = _solid((32, 32), (200, 50, 50))
    res = render_ascii(img, cols=4, rows=4, ramp=" .:-=+*#")
    for row in res.colors:
        for r, g, b in row:
            assert abs(r - 200) <= 1 and abs(g - 50) <= 1 and abs(b - 50) <= 1


def test_invert_maps_bright_to_dense():
    img = _solid((16, 16), (255, 255, 255))
    ramp = " .:-=+*#"
    res = render_ascii(img, cols=2, rows=2, ramp=ramp, invert=True)
    for row in res.grid:
        for ch in row:
            assert ch == ramp[-1]
