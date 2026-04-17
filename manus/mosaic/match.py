from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def build_index(tile_labs: np.ndarray) -> cKDTree:
    return cKDTree(tile_labs.astype(np.float32))


def topk_candidates(index: cKDTree, query_lab: np.ndarray, k: int) -> np.ndarray:
    k = min(k, index.n)
    _, idx = index.query(query_lab, k=k)
    if np.ndim(idx) == 0:
        idx = np.array([idx])
    return np.asarray(idx, dtype=np.int64)


def _neighbor_sim(tile_lab: np.ndarray, neighbor_labs: list[np.ndarray]) -> float:
    if not neighbor_labs:
        return 0.0
    t = tile_lab / (np.linalg.norm(tile_lab) + 1e-9)
    acc = 0.0
    for n in neighbor_labs:
        nn = n / (np.linalg.norm(n) + 1e-9)
        acc += 1.0 - float(np.dot(t, nn))
    return acc / len(neighbor_labs)


def match_grid(
    index: cKDTree,
    tile_labs: np.ndarray,
    cell_labs: np.ndarray,
    k: int = 20,
    lambda_: float = 2.0,
    mu: float = 0.5,
    log_every: int = 0,
) -> np.ndarray:
    """Row-major greedy match. Returns (grid_h, grid_w) int64 array of tile indices."""
    grid_h, grid_w, _ = cell_labs.shape
    choices = np.full((grid_h, grid_w), -1, dtype=np.int64)
    uses = np.zeros(tile_labs.shape[0], dtype=np.int64)

    total = grid_h * grid_w
    step = 0
    for r in range(grid_h):
        for c in range(grid_w):
            step += 1
            cell = cell_labs[r, c]
            cand = topk_candidates(index, cell, k=k)
            neighbor_labs: list[np.ndarray] = []
            if r > 0 and choices[r - 1, c] >= 0:
                neighbor_labs.append(tile_labs[choices[r - 1, c]])
            if c > 0 and choices[r, c - 1] >= 0:
                neighbor_labs.append(tile_labs[choices[r, c - 1]])
            if r > 0 and c > 0 and choices[r - 1, c - 1] >= 0:
                neighbor_labs.append(tile_labs[choices[r - 1, c - 1]])
            if r > 0 and c + 1 < grid_w and choices[r - 1, c + 1] >= 0:
                neighbor_labs.append(tile_labs[choices[r - 1, c + 1]])

            best_i = int(cand[0])
            best_score = float("inf")
            for ti in cand:
                dist = float(np.linalg.norm(tile_labs[ti] - cell))
                diversity = lambda_ * float(np.log1p(uses[ti]))
                neighbor = mu * _neighbor_sim(tile_labs[ti], neighbor_labs)
                score = dist + diversity + neighbor
                if score < best_score:
                    best_score = score
                    best_i = int(ti)
            choices[r, c] = best_i
            uses[best_i] += 1
            if log_every and step % log_every == 0:
                print(f"[{step}/{total}] cell ({r},{c}) -> tile #{best_i} (uses={uses[best_i]})")
    return choices
