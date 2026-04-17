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
