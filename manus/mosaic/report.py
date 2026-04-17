from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def describe_lab(lab: np.ndarray) -> str:
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    if b < -20 and -15 < a < 15:
        return "sky blue"
    if a > 10 and b > 15 and L > 40:
        return "skin tone"
    if a < -15 and b > 10:
        return "foliage green"
    if 10 < a and -5 < b < 20 and L < 55:
        return "warm brown"
    if L < 25:
        return "shadow"
    if L > 85:
        return "highlight"
    return "neutral gray"


def text_report(
    tiles: list,
    uses: np.ndarray,
    grid_shape: tuple[int, int],
    params: dict,
) -> str:
    total_cells = grid_shape[0] * grid_shape[1]
    unique_used = int((uses > 0).sum())
    order = np.argsort(-uses)
    lines: list[str] = []
    lines.append("Tile.Photo — mosaic report")
    lines.append("")
    lines.append(f"Pool size: {len(tiles)}")
    lines.append(f"Tiles placed: {total_cells} ({grid_shape[1]} × {grid_shape[0]})")
    denom = max(len(tiles), 1)
    lines.append(f"Unique tiles used: {unique_used} ({unique_used / denom:.0%} of pool)")
    lines.append("")
    lines.append("Most used (top 5):")
    for i in order[:5]:
        if uses[i] == 0:
            break
        reason = describe_lab(tiles[i].lab)
        lines.append(f"  {tiles[i].path.name} — {int(uses[i])} uses  (mostly: {reason})")
    lines.append("")
    cold_idx = [i for i in order[::-1] if uses[i] == 0][:20]
    cold = [tiles[i].path.name for i in cold_idx]
    lines.append(
        f"Never used ({int((uses == 0).sum())} tiles). Top 20 cold: "
        f"{', '.join(cold) if cold else '(none)'}"
    )
    lines.append("")
    lines.append(
        "Parameters: "
        f"λ={params.get('lambda_', '?')}, μ={params.get('mu', '?')}, τ={params.get('tau', '?')}"
    )
    return "\n".join(lines)


def cold_wall(
    tiles: list,
    uses: np.ndarray,
    n: int = 20,
    thumb_px: int = 32,
    cols: int = 5,
) -> np.ndarray:
    cold_idx = [int(i) for i in np.argsort(uses) if uses[i] == 0][:n]
    if not cold_idx:
        return np.zeros((thumb_px, thumb_px, 3), dtype=np.uint8)
    rows = (len(cold_idx) + cols - 1) // cols
    wall = np.zeros((rows * thumb_px, cols * thumb_px, 3), dtype=np.uint8)
    for k, ti in enumerate(cold_idx):
        r, c = divmod(k, cols)
        thumb = tiles[ti].thumb
        if thumb.shape[0] != thumb_px:
            thumb = np.array(
                Image.fromarray(thumb).resize((thumb_px, thumb_px), Image.LANCZOS),
                dtype=np.uint8,
            )
        wall[r * thumb_px : (r + 1) * thumb_px, c * thumb_px : (c + 1) * thumb_px] = thumb
    return wall


def usage_hist_figure(tiles: list, uses: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 3))
    bins = min(30, max(int(uses.max()) + 1, 2))
    ax.hist(uses, bins=bins)
    ax.set_xlabel("times used")
    ax.set_ylabel("tile count")
    ax.set_title(f"Usage distribution across {len(tiles)} tiles")
    fig.tight_layout()
    return fig
