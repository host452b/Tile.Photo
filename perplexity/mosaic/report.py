"""生成自嘲式报告：文字 + 使用次数柱状图 + 冷宫照片墙。"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _region_label(positions: list, rows: int, cols: int) -> str:
    if not positions:
        return "未知"
    ys = [p[0] / max(rows - 1, 1) for p in positions]
    xs = [p[1] / max(cols - 1, 1) for p in positions]
    y_mean = sum(ys) / len(ys)
    x_mean = sum(xs) / len(xs)
    if y_mean < 0.33:
        return "天空"
    if y_mean > 0.67:
        return "地面"
    if abs(x_mean - 0.5) < 0.2:
        return "主体"
    return "填充"


def _save_usage_chart(usage: Counter, output: Path) -> None:
    top = usage.most_common(30)
    if not top:
        Image.new("RGB", (600, 200), (240, 240, 240)).save(output)
        return
    labels = [Path(name).name[:20] for name, _ in top]
    values = [v for _, v in top]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(top)), values, color="#4c72b0")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("使用次数")
    ax.set_title("Top 30 被使用最多的 tile")
    fig.tight_layout()
    fig.savefig(output, dpi=100)
    plt.close(fig)


def _save_cold_wall(pool: dict, usage: Counter, output: Path, n_cols: int = 5, n_rows: int = 4) -> None:
    cold = [path for path in pool if usage.get(path, 0) == 0]
    if not cold:
        Image.new("RGB", (320, 256), (240, 240, 240)).save(output)
        return
    picks = random.Random(42).sample(cold, min(n_cols * n_rows, len(cold)))
    tile_h, tile_w = 64, 64
    canvas = np.full((tile_h * n_rows, tile_w * n_cols, 3), 230, dtype=np.uint8)
    for i, path in enumerate(picks):
        r, c = divmod(i, n_cols)
        thumb = pool[path]["thumbnail"]
        thumb_img = Image.fromarray(thumb).resize((tile_w, tile_h), Image.LANCZOS)
        canvas[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = np.asarray(thumb_img)
    Image.fromarray(canvas).save(output)


def generate_report(
    pool: dict,
    usage: Counter,
    positions: dict,
    grid_shape: tuple,
    output_md: Path,
    output_chart: Path,
    output_cold: Path,
) -> str:
    rows, cols = grid_shape
    pool_total = len(pool)
    used_count = sum(1 for v in usage.values() if v > 0)
    cold_count = pool_total - used_count

    if usage:
        top_tile_path, top_count = usage.most_common(1)[0]
        top_tile_name = Path(top_tile_path).name
        top_region_guess = _region_label(positions.get(top_tile_path, []), rows, cols)
    else:
        top_tile_name, top_count, top_region_guess = "(none)", 0, "无"

    cold_picks = [p for p in pool if usage.get(p, 0) == 0][:5]
    cold_names = ", ".join(Path(p).name for p in cold_picks) if cold_picks else "(none)"

    _save_usage_chart(usage, output_chart)
    _save_cold_wall(pool, usage, output_cold, n_cols=5, n_rows=4)

    text = f"""# Photomosaic 报告

本次使用了你 **{pool_total}** 张照片里的 **{used_count}** 张（冷宫 {cold_count} 张）。

其中 `{top_tile_name}` 被用了 **{top_count}** 次（主要用于填充**{top_region_guess}**）。

冷宫照片 TOP 5 是：{cold_names}（都是你的废片）。

![使用次数柱状图]({output_chart.name})

![冷宫照片墙]({output_cold.name})
"""
    output_md.write_text(text, encoding="utf-8")
    return text
