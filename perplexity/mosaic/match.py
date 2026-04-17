"""贪心匹配求解器：faiss top-K 颜色候选 + λ 重复惩罚 + μ 邻居相似惩罚。

每格的 cost：
    cost(g, t) = ||LAB(g) - LAB(t)||^2
               + λ * log(1 + usage[t])
               + μ * Σ_{n in decided N4(g)} exp(-||LAB(t) - LAB(t_n)||^2 / σ^2)
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import faiss
import numpy as np
from tqdm import tqdm


def _neighbor_coords(r: int, c: int, rows: int, cols: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def solve_assignment(
    pool: dict,
    cells: list,
    grid_shape: tuple,
    lambda_reuse: float,
    mu_neighbor: float,
    topk: int = 64,
    neighbor_sigma: float = 20.0,
    semantic_reranker: Optional[Callable] = None,
    log_every: int = 100,
) -> dict:
    """返回 {(row, col): tile_path}。"""
    rows, cols = grid_shape
    tile_paths = list(pool.keys())
    if not tile_paths:
        raise ValueError("Empty pool")

    lab_matrix = np.stack([pool[p]["lab_mean"] for p in tile_paths]).astype(np.float32)
    path_to_idx = {p: i for i, p in enumerate(tile_paths)}
    index = faiss.IndexFlatL2(3)
    index.add(lab_matrix)

    actual_topk = min(topk, len(tile_paths))
    usage = np.zeros(len(tile_paths), dtype=np.int64)
    assignment: dict = {}

    # 方差降序扫描
    order = sorted(range(len(cells)), key=lambda i: -cells[i]["variance"])

    for step, cell_idx in enumerate(tqdm(order, desc="匹配中")):
        cell = cells[cell_idx]
        query = cell["lab_mean"].astype(np.float32).reshape(1, 3)
        color_dists, cand_indices = index.search(query, actual_topk)
        color_dists = color_dists[0]
        cand_indices = cand_indices[0]

        best_cost = math.inf
        best_tile_idx = cand_indices[0]

        for rank, ti in enumerate(cand_indices):
            if ti < 0:
                continue
            cost = float(color_dists[rank])
            cost += lambda_reuse * math.log(1.0 + int(usage[ti]))

            if mu_neighbor > 0.0:
                neigh_pen = 0.0
                for nr, nc in _neighbor_coords(cell["row"], cell["col"], rows, cols):
                    nkey = (nr, nc)
                    if nkey not in assignment:
                        continue
                    n_ti = path_to_idx[assignment[nkey]]
                    diff = lab_matrix[ti] - lab_matrix[n_ti]
                    neigh_pen += math.exp(-float(np.dot(diff, diff)) / (neighbor_sigma ** 2))
                cost += mu_neighbor * neigh_pen

            if semantic_reranker is not None:
                cost += float(semantic_reranker(cell, tile_paths[ti]))

            if cost < best_cost:
                best_cost = cost
                best_tile_idx = ti

        chosen_path = tile_paths[best_tile_idx]
        assignment[(cell["row"], cell["col"])] = chosen_path
        usage[best_tile_idx] += 1

        if step % log_every == 0:
            print(
                f"  ({cell['row']:3d},{cell['col']:3d}) → "
                f"{chosen_path.split('/')[-1]} | "
                f"color={color_dists[np.where(cand_indices == best_tile_idx)[0][0]]:.2f}, "
                f"used={int(usage[best_tile_idx])}x"
            )

    return assignment
