from __future__ import annotations

import numpy as np


def match_grid(target_lab: np.ndarray, pool_lab: np.ndarray) -> np.ndarray:
    """For each (h, w) cell in target_lab, return the index into pool_lab
    whose LAB triplet has smallest Euclidean (ΔE76) distance.

    Args:
        target_lab: (H, W, 3) float32
        pool_lab:   (N, 3)    float32

    Returns:
        (H, W) int32 of indices into pool_lab.
    """
    if pool_lab.shape[0] == 0:
        raise ValueError("pool_lab is empty; cannot match")
    target_f = target_lab.astype(np.float32, copy=False)
    pool_f = pool_lab.astype(np.float32, copy=False)
    diff = target_f[:, :, None, :] - pool_f[None, None, :, :]
    dist2 = (diff * diff).sum(axis=-1)
    return dist2.argmin(axis=-1).astype(np.int32)
