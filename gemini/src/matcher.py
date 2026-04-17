from __future__ import annotations
import numpy as np
import faiss
from typing import Callable, Optional

CellCallback = Callable[[int, int, int, list[tuple[int, float]]], None]


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


def assign_with_penalties(topk_idx: np.ndarray, topk_dist: np.ndarray,
                          lambda_repeat: float, mu_neighbor: float,
                          on_cell: Optional[CellCallback] = None) -> np.ndarray:
    """
    Greedy left-to-right, top-to-bottom assignment minimizing:
      score(tile) = sqrt(dist) + lambda_repeat * log1p(usage[tile])
                    + mu_neighbor * 1_{tile in {left_neighbor, top_neighbor}}
    Returns assignment array of shape (rows, cols).
    """
    rows, cols, k = topk_idx.shape
    assignment = np.full((rows, cols), -1, dtype=np.int64)
    usage: dict[int, int] = {}
    for r in range(rows):
        for c in range(cols):
            best_tile = -1
            best_score = float("inf")
            candidate_log: list[tuple[int, float]] = []
            left = int(assignment[r, c - 1]) if c > 0 else -1
            up = int(assignment[r - 1, c]) if r > 0 else -1
            for cand_rank in range(k):
                t = int(topk_idx[r, c, cand_rank])
                d = float(topk_dist[r, c, cand_rank])
                score = np.sqrt(max(d, 0.0))
                score += lambda_repeat * np.log1p(usage.get(t, 0))
                if t == left or t == up:
                    score += mu_neighbor
                candidate_log.append((t, float(score)))
                if score < best_score:
                    best_score = score
                    best_tile = t
            assignment[r, c] = best_tile
            usage[best_tile] = usage.get(best_tile, 0) + 1
            if on_cell is not None:
                on_cell(r, c, best_tile, candidate_log)
    return assignment


def assign_with_clip(topk_idx: np.ndarray, topk_dist: np.ndarray,
                     patches_lab: np.ndarray, tile_lab: np.ndarray,
                     tile_clip: np.ndarray, patch_clip: np.ndarray,
                     lambda_repeat: float, mu_neighbor: float, clip_weight: float,
                     on_cell: Optional[CellCallback] = None) -> np.ndarray:
    """
    Same greedy scan as assign_with_penalties, plus a cosine-similarity bonus:
      score -= clip_weight * cosine(tile_clip[t], patch_clip[r,c])
    Assumes both tile_clip and patch_clip are L2-normalized (cosine = dot).
    """
    rows, cols, k = topk_idx.shape
    assignment = np.full((rows, cols), -1, dtype=np.int64)
    usage: dict[int, int] = {}
    for r in range(rows):
        for c in range(cols):
            best_tile = -1
            best_score = float("inf")
            candidate_log: list[tuple[int, float]] = []
            left = int(assignment[r, c - 1]) if c > 0 else -1
            up = int(assignment[r - 1, c]) if r > 0 else -1
            p_emb = patch_clip[r, c]
            for cand_rank in range(k):
                t = int(topk_idx[r, c, cand_rank])
                d = float(topk_dist[r, c, cand_rank])
                score = np.sqrt(max(d, 0.0))
                score += lambda_repeat * np.log1p(usage.get(t, 0))
                if t == left or t == up:
                    score += mu_neighbor
                score -= clip_weight * float(np.dot(tile_clip[t], p_emb))
                candidate_log.append((t, float(score)))
                if score < best_score:
                    best_score = score
                    best_tile = t
            assignment[r, c] = best_tile
            usage[best_tile] = usage.get(best_tile, 0) + 1
            if on_cell is not None:
                on_cell(r, c, best_tile, candidate_log)
    return assignment
