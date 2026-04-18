from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.pipeline import build, parse_bg


def _gradient(size: tuple[int, int]) -> Image.Image:
    w, h = size
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return Image.fromarray(np.stack([arr, arr, arr], axis=-1))


def test_end_to_end_returns_image_and_result_with_matching_dims():
    img = _gradient((64, 64))
    out_img, result = build(img, cols=16, rows=16, density=0.6, bg=(0, 0, 0), cell_px=10)
    assert out_img.size == (160, 160)
    assert result.cols == 16 and result.rows == 16


def test_auto_bg_equals_hall_of_oblivion_color():
    from src.background import hall_of_oblivion_color

    img = _gradient((32, 32))
    _, result = build(img, cols=8, rows=8, density=0.6, bg="auto", cell_px=8)
    expected = hall_of_oblivion_color(result, pct=0.2)
    img2, _ = build(img, cols=8, rows=8, density=0.6, bg=expected, cell_px=8)
    img3, _ = build(img, cols=8, rows=8, density=0.6, bg="auto", cell_px=8)
    assert np.array_equal(np.asarray(img2), np.asarray(img3))


def test_parse_bg_hex_and_tuple_and_auto():
    assert parse_bg("auto") == "auto"
    assert parse_bg("#ff8040") == (255, 128, 64)
    assert parse_bg("FF8040") == (255, 128, 64)
    assert parse_bg("10,20,30") == (10, 20, 30)


def test_build_writes_png_when_output_given(tmp_path: Path):
    img = _gradient((48, 48))
    out = tmp_path / "ascii.png"
    build(img, cols=12, rows=12, density=0.4, bg="auto", cell_px=10, output=out)
    assert out.exists()
    with Image.open(out) as written:
        assert written.size == (120, 120)
