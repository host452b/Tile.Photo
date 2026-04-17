from __future__ import annotations
import numpy as np
import faiss


def color_topk(patches_lab: np.ndarray, tile_lab: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    patches_lab: (rows, cols, 3) — target cells in LAB
    tile_lab:    (N, 3)          — tile pool LAB means
    Returns (topk_idx, topk_dist) each (rows, cols, k_effective) where k_effective = min(k, N).
    """
    rows, cols, _ = patches_lab.shape
    n_tiles = tile_lab.shape[0]
    k_eff = min(k, n_tiles)
    index = faiss.IndexFlatL2(3)
    index.add(tile_lab.astype(np.float32))
    q = patches_lab.reshape(-1, 3).astype(np.float32)
    dist, idx = index.search(q, k_eff)
    return idx.reshape(rows, cols, k_eff), dist.reshape(rows, cols, k_eff)
