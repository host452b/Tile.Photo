from __future__ import annotations
from typing import Mapping
from pathlib import Path


def build_text_report(tile_paths: list[str], usage: Mapping[int, int],
                      tags: Mapping[str, str], total_cells: int) -> str:
    pool_size = len(tile_paths)
    used = [(i, n) for i, n in usage.items() if n > 0]
    used_count = len(used)
    cold = [p for i, p in enumerate(tile_paths) if usage.get(i, 0) == 0]
    # Top 5 most-used
    top = sorted(used, key=lambda x: -x[1])[:5]
    # Tag shares (weighted by usage count)
    tag_counts: dict[str, int] = {}
    for i, n in usage.items():
        tag = tags.get(tile_paths[i], "untagged")
        tag_counts[tag] = tag_counts.get(tag, 0) + n
    tag_share = sorted(tag_counts.items(), key=lambda x: -x[1])

    lines: list[str] = []
    lines.append(f"本次使用了你 {pool_size} 张照片里的 {used_count} 张 ({used_count / max(pool_size, 1):.0%}).")
    lines.append("")
    lines.append("最勤奋的拼豆 (用得最多的 5 张):")
    for i, n in top:
        name = Path(tile_paths[i]).name
        lines.append(f"  - {tile_paths[i]} ({name}): 被用了 {n} 次 ({n / total_cells:.1%} 的格子)")
    lines.append("")
    lines.append(f"冷宫照片 ({len(cold)} 张一次都没被用上):")
    for p in cold[:5]:
        lines.append(f"  - {p}")
    if len(cold) > 5:
        lines.append(f"  ... 还有 {len(cold) - 5} 张")
    lines.append("")
    lines.append("构成配方 (按标签):")
    for tag, n in tag_share:
        lines.append(f"  - {tag}: {n / total_cells:.1%} ({n} / {total_cells} 格)")
    return "\n".join(lines)


def save_usage_plot(usage: Mapping[int, int], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = sorted(usage.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(counts)), counts, width=1.0)
    ax.set_xlabel("tile rank (most-used → least-used)")
    ax.set_ylabel("use count")
    ax.set_title("Tile usage distribution (long tail = healthy diversity)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_cold_wall(tile_paths: list[str], usage: Mapping[int, int],
                   out_path: str, cols: int = 10, tile_px: int = 64) -> None:
    from PIL import Image as _I
    cold_paths = [p for i, p in enumerate(tile_paths) if usage.get(i, 0) == 0]
    if not cold_paths:
        # Still write a placeholder so caller can rely on file presence
        _I.new("RGB", (tile_px, tile_px), (40, 40, 40)).save(out_path)
        return
    rows = (len(cold_paths) + cols - 1) // cols
    canvas = _I.new("RGB", (cols * tile_px, rows * tile_px), (0, 0, 0))
    for i, p in enumerate(cold_paths):
        with _I.open(p) as im:
            im = im.convert("RGB")
            s = min(im.size)
            left = (im.size[0] - s) // 2
            top = (im.size[1] - s) // 2
            im = im.crop((left, top, left + s, top + s)).resize((tile_px, tile_px), _I.LANCZOS)
        canvas.paste(im, ((i % cols) * tile_px, (i // cols) * tile_px))
    canvas.save(out_path)
