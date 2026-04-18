from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image

from src.background import hall_of_oblivion_color
from src.charset import get_ramp
from src.draw import draw_ascii
from src.render import AsciiResult, render_ascii

BgSpec = Union[tuple[int, int, int], str]


def parse_bg(value: str) -> BgSpec:
    s = value.strip()
    if s.lower() == "auto":
        return "auto"
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 6 and all(c in "0123456789abcdefABCDEF" for c in s):
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    if "," in s:
        parts = [int(p.strip()) for p in s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"expected r,g,b got {value!r}")
        for p in parts:
            if not 0 <= p <= 255:
                raise ValueError(f"channel out of range in {value!r}")
        return (parts[0], parts[1], parts[2])
    raise ValueError(f"unrecognized bg spec {value!r}")


def build(
    image: Image.Image,
    cols: int,
    rows: int,
    density: float,
    bg: BgSpec,
    cell_px: int = 12,
    invert: bool = False,
    cold_pct: float = 0.2,
    output: Path | str | None = None,
) -> tuple[Image.Image, AsciiResult]:
    ramp = get_ramp(density)
    result = render_ascii(image, cols=cols, rows=rows, ramp=ramp, invert=invert)

    if bg == "auto":
        bg_rgb = hall_of_oblivion_color(result, pct=cold_pct)
    else:
        bg_rgb = bg

    out_img = draw_ascii(result, bg=bg_rgb, cell_px=cell_px)

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(output)

    return out_img, result
