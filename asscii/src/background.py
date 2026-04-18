from __future__ import annotations

import math

from src.render import AsciiResult


def hall_of_oblivion_color(result: AsciiResult, pct: float = 0.2) -> tuple[int, int, int]:
    p = max(0.0, min(1.0, float(pct)))
    used = [(cnt, ch) for ch, cnt in result.char_usage.items() if cnt > 0]
    if not used:
        return (128, 128, 128)

    used.sort()
    cutoff = max(1, math.ceil(p * len(used)))
    cold_chars = {ch for _, ch in used[:cutoff]}

    r_sum = g_sum = b_sum = 0
    n = 0
    for row_chars, row_colors in zip(result.grid, result.colors):
        for ch, (r, g, b) in zip(row_chars, row_colors):
            if ch in cold_chars:
                r_sum += r
                g_sum += g
                b_sum += b
                n += 1

    if n == 0:
        return (128, 128, 128)
    return (round(r_sum / n), round(g_sum / n), round(b_sum / n))
