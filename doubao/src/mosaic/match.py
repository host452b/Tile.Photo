"""Split target + match tiles to cells."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

from .config import MosaicConfig
from .tiles import TilePool


def split_target(target_path: Path, grid: tuple[int, int]) -> np.ndarray:
    """Load target, resize to (cols, rows) via block averaging, return LAB cells.

    Returns array of shape (rows, cols, 3) with LAB mean per cell.
    """
    cols, rows = grid
    with Image.open(target_path) as im:
        im = im.convert("RGB")
        # Resize so each pixel is one cell's mean; PIL LANCZOS approximates mean
        im = im.resize((cols, rows), Image.BOX)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    lab = rgb2lab(arr)  # (rows, cols, 3)
    return lab.astype(np.float32)


def _top_k_candidates(target_lab: np.ndarray, pool_lab: np.ndarray, k: int) -> np.ndarray:
    """Return indices of k nearest pool entries to target_lab by L2 in LAB.

    target_lab shape (3,), pool_lab shape (N, 3). Returns (min(k, N),) int array,
    sorted by distance ascending.
    """
    d2 = ((pool_lab - target_lab) ** 2).sum(axis=1)
    k = min(k, len(pool_lab))
    idx = np.argpartition(d2, k - 1)[:k]
    return idx[np.argsort(d2[idx])]


def match_all_tiles(
    target_cells: np.ndarray,
    pool: TilePool,
    cfg: MosaicConfig,
) -> tuple[np.ndarray, dict[int, int]]:
    """Row-major match each target cell to a pool tile.

    Returns (assignment (rows, cols) int32, use_count dict[tile_idx -> count]).
    """
    rows, cols, _ = target_cells.shape
    assignment = np.full((rows, cols), -1, dtype=np.int32)
    use_count: dict[int, int] = defaultdict(int)

    for r in range(rows):
        for c in range(cols):
            t_lab = target_cells[r, c]
            cand = _top_k_candidates(t_lab, pool.lab_means, cfg.candidate_k)
            color_dist = np.linalg.norm(pool.lab_means[cand] - t_lab, axis=1)
            scores = color_dist.copy()
            # (penalties added in Task 6)
            best_local = int(np.argmin(scores))
            best = int(cand[best_local])
            assignment[r, c] = best
            use_count[best] += 1
            if cfg.verbose:
                print(
                    f"[{r:3d},{c:3d}] -> tile#{best} "
                    f"dist={color_dist[best_local]:.1f} reuse={use_count[best]}"
                )

    return assignment, dict(use_count)
