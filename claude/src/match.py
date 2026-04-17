from __future__ import annotations

import numpy as np


def match_grid(
    target_lab: np.ndarray,
    pool_lab: np.ndarray,
    *,
    lambda_: float = 0.0,
    mu: float = 0.0,
) -> np.ndarray:
    """For each cell of target_lab, pick a pool index.

    Args:
        target_lab: (H, W, 3) float32 — LAB mean per cell.
        pool_lab:   (N, 3)    float32 — LAB mean per pool tile.
        lambda_:    repetition penalty coefficient (>=0).
        mu:         neighbor-match penalty coefficient (>=0).

    Returns:
        (H, W) int32 of indices into pool_lab.
    """
    if pool_lab.shape[0] == 0:
        raise ValueError("pool_lab is empty; cannot match")

    if lambda_ == 0.0 and mu == 0.0:
        target_f = target_lab.astype(np.float32, copy=False)
        pool_f = pool_lab.astype(np.float32, copy=False)
        diff = target_f[:, :, None, :] - pool_f[None, None, :, :]
        dist2 = (diff * diff).sum(axis=-1)
        return dist2.argmin(axis=-1).astype(np.int32)

    return _greedy_match(target_lab, pool_lab, lambda_, mu)


def _greedy_match(
    target_lab: np.ndarray,
    pool_lab: np.ndarray,
    lambda_: float,
    mu: float,
) -> np.ndarray:
    h, w, _ = target_lab.shape
    n = pool_lab.shape[0]
    placed = np.full((h, w), -1, dtype=np.int32)
    uses = np.zeros(n, dtype=np.int64)
    tgt = target_lab.astype(np.float32, copy=False)
    pool = pool_lab.astype(np.float32, copy=False)

    for r in range(h):
        for c in range(w):
            diff = tgt[r, c] - pool
            dist2 = (diff * diff).sum(axis=-1)
            score = dist2 + lambda_ * np.log1p(uses)
            if mu > 0.0:
                if r > 0:
                    score[placed[r - 1, c]] += mu
                if c > 0:
                    score[placed[r, c - 1]] += mu
            idx = int(score.argmin())
            placed[r, c] = idx
            uses[idx] += 1
    return placed
