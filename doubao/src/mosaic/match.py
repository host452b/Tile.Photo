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

    Score = color_dist
          + λ · log(1 + use_count[cand])
          + μ · similarity(cand, already-placed left + top neighbor)
      where similarity = 1 / (1 + mean LAB dist), so identical neighbors
      contribute μ, different neighbors contribute ~0.

    Returns (assignment (rows, cols) int32, use_count dict[tile_idx -> count]).
    """
    rows, cols, _ = target_cells.shape
    assignment = np.full((rows, cols), -1, dtype=np.int32)
    use_count: dict[int, int] = defaultdict(int)
    lam = float(cfg.lambda_reuse)
    mu = float(cfg.mu_neighbor)

    for r in range(rows):
        for c in range(cols):
            t_lab = target_cells[r, c]
            cand = _top_k_candidates(t_lab, pool.lab_means, cfg.candidate_k)
            cand_lab = pool.lab_means[cand]
            color_dist = np.linalg.norm(cand_lab - t_lab, axis=1)

            reuse_term = np.zeros_like(color_dist)
            if lam > 0:
                uses = np.array(
                    [use_count.get(int(i), 0) for i in cand], dtype=np.float32
                )
                reuse_term = lam * np.log1p(uses)

            neigh_term = np.zeros_like(color_dist)
            if mu > 0:
                neigh_labs: list[np.ndarray] = []
                if c > 0 and assignment[r, c - 1] >= 0:
                    neigh_labs.append(pool.lab_means[assignment[r, c - 1]])
                if r > 0 and assignment[r - 1, c] >= 0:
                    neigh_labs.append(pool.lab_means[assignment[r - 1, c]])
                if neigh_labs:
                    neigh_arr = np.stack(neigh_labs)  # (M, 3)
                    mean_dist = np.linalg.norm(
                        cand_lab[:, None, :] - neigh_arr[None, :, :], axis=2
                    ).mean(axis=1)
                    similarity = 1.0 / (1.0 + mean_dist)
                    neigh_term = mu * similarity

            scores = color_dist + reuse_term + neigh_term
            best_local = int(np.argmin(scores))
            best = int(cand[best_local])
            assignment[r, c] = best
            use_count[best] += 1
            if cfg.verbose:
                pen = float(reuse_term[best_local] + neigh_term[best_local])
                print(
                    f"[{r:3d},{c:3d}] -> tile#{best} "
                    f"dist={color_dist[best_local]:.1f} "
                    f"reuse={use_count[best]} pen={pen:.2f}"
                )

    return assignment, dict(use_count)
