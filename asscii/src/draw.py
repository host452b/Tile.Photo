from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.render import AsciiResult

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/Library/Fonts/Courier New.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


def _load_font(font_path: str | None, cell_px: int) -> ImageFont.ImageFont:
    size = max(6, int(cell_px * 0.9))
    paths = [font_path] if font_path else _FONT_CANDIDATES
    for p in paths:
        if p and Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_ascii(
    result: AsciiResult,
    bg: tuple[int, int, int],
    cell_px: int = 12,
    font_path: str | None = None,
) -> Image.Image:
    if cell_px < 4:
        raise ValueError("cell_px must be >= 4 for legibility")
    w = result.cols * cell_px
    h = result.rows * cell_px
    canvas = Image.new("RGB", (w, h), bg)
    drawer = ImageDraw.Draw(canvas)
    font = _load_font(font_path, cell_px)

    try:
        bbox = font.getbbox("M")
        glyph_w = bbox[2] - bbox[0]
        glyph_h = bbox[3] - bbox[1]
        pad_x = (cell_px - glyph_w) // 2 - bbox[0]
        pad_y = (cell_px - glyph_h) // 2 - bbox[1]
    except AttributeError:
        pad_x = pad_y = 0

    for r, (chars, colors) in enumerate(zip(result.grid, result.colors)):
        for c, (ch, color) in enumerate(zip(chars, colors)):
            if ch == " ":
                continue
            drawer.text(
                (c * cell_px + pad_x, r * cell_px + pad_y),
                ch,
                fill=color,
                font=font,
            )

    return canvas
