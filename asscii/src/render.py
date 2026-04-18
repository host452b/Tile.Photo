from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class AsciiResult:
    grid: list[list[str]]
    colors: list[list[tuple[int, int, int]]]
    char_usage: Counter[str] = field(default_factory=Counter)
    cols: int = 0
    rows: int = 0


def render_ascii(
    image: Image.Image,
    cols: int,
    rows: int,
    ramp: str,
    invert: bool = False,
) -> AsciiResult:
    if cols <= 0 or rows <= 0:
        raise ValueError("cols and rows must be positive")
    if len(ramp) < 2:
        raise ValueError("ramp must have at least 2 characters")

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape

    col_edges = np.linspace(0, w, cols + 1, dtype=int)
    row_edges = np.linspace(0, h, rows + 1, dtype=int)

    grid: list[list[str]] = []
    colors: list[list[tuple[int, int, int]]] = []
    usage: Counter[str] = Counter()
    L = len(ramp)

    for r in range(rows):
        y0, y1 = row_edges[r], row_edges[r + 1]
        grid_row: list[str] = []
        color_row: list[tuple[int, int, int]] = []
        for c in range(cols):
            x0, x1 = col_edges[c], col_edges[c + 1]
            block = rgb[y0:y1, x0:x1]
            mean = block.reshape(-1, 3).mean(axis=0)
            luma = 0.299 * mean[0] + 0.587 * mean[1] + 0.114 * mean[2]
            t = luma / 255.0
            idx = round((t if invert else 1.0 - t) * (L - 1))
            idx = max(0, min(L - 1, idx))
            ch = ramp[idx]
            grid_row.append(ch)
            color_row.append((int(round(mean[0])), int(round(mean[1])), int(round(mean[2]))))
            usage[ch] += 1
        grid.append(grid_row)
        colors.append(color_row)

    return AsciiResult(grid=grid, colors=colors, char_usage=usage, cols=cols, rows=rows)
