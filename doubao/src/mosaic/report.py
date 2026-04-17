"""Self-deprecating report + usage histogram + cold-photo wall."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .tiles import TilePool


def generate_text_report(
    use_count: dict[int, int],
    pool: TilePool,
    total_cells: int,
) -> str:
    """Self-deprecating ASCII report."""
    n_used = sum(1 for v in use_count.values() if v > 0)
    n_pool = len(pool)

    top5 = sorted(use_count.items(), key=lambda kv: kv[1], reverse=True)[:5]
    used_idx = {i for i, v in use_count.items() if v > 0}
    cold_idx = [i for i in range(n_pool) if i not in used_idx]

    lines = [
        f"本次使用了你 {n_pool:,} 张照片里的 {n_used:,} 张。",
        f"总格数 {total_cells:,}，平均每张底图被用 {total_cells / max(n_used, 1):.1f} 次。",
        "",
        "用得最多的 TOP 5：",
    ]
    for idx, cnt in top5:
        name = Path(str(pool.paths[idx])).name
        lines.append(f"  - {name} ({cnt} 次)")
    lines.append("")
    cold_show = cold_idx[:5]
    if cold_show:
        lines.append(f"冷宫照片 TOP 5（共 {len(cold_idx)} 张一次都没被用）：")
        for idx in cold_show:
            name = Path(str(pool.paths[idx])).name
            lines.append(f"  - {name}")
    else:
        lines.append("无冷宫照片 —— 每张都至少被用了一次。")
    return "\n".join(lines)


def plot_usage_histogram(use_count: dict[int, int]):
    """Return a matplotlib Figure of usage count distribution."""
    import matplotlib.pyplot as plt

    counts = sorted(use_count.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(counts)), counts)
    ax.set_xlabel("底图（按使用次数降序）")
    ax.set_ylabel("使用次数")
    ax.set_title(f"底图使用分布（{len(counts)} 张被用）")
    fig.tight_layout()
    return fig


def build_cold_wall(
    pool: TilePool,
    use_count: dict[int, int],
    max_shown: int = 64,
) -> Image.Image:
    """Grid image of tiles that were never used."""
    used = {i for i, v in use_count.items() if v > 0}
    cold = [i for i in range(len(pool)) if i not in used]
    if not cold:
        # Return a 1×1 blank so caller doesn't blow up
        return Image.new("RGB", (1, 1), (0, 0, 0))

    cold = cold[:max_shown]
    n = len(cold)
    side = int(math.ceil(math.sqrt(n)))
    tile_px = pool.thumbnails.shape[1]
    canvas = np.full((side * tile_px, side * tile_px, 3), 255, dtype=np.uint8)
    for k, idx in enumerate(cold):
        r, c = k // side, k % side
        canvas[
            r * tile_px : (r + 1) * tile_px,
            c * tile_px : (c + 1) * tile_px,
        ] = pool.thumbnails[idx]
    return Image.fromarray(canvas)
