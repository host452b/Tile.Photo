from __future__ import annotations

import numpy as np
from skimage.color import lab2rgb, rgb2lab


def reinhard_transfer(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Shift source's LAB mean toward target's LAB mean by `strength`.

    Args:
        source_rgb: (H, W, 3) uint8 — the tile being placed.
        target_rgb: (H, W, 3) uint8 — the target patch at that grid cell.
        strength: [0, 1]. 0 = return source unchanged. 1 = adopt target mean.

    Returns:
        (H, W, 3) uint8, tone-shifted source.
    """
    if strength == 0.0:
        return source_rgb
    source_lab = rgb2lab(source_rgb.astype(np.float32) / 255.0)
    target_lab_mean = rgb2lab(target_rgb.astype(np.float32) / 255.0).mean(axis=(0, 1))
    source_lab_mean = source_lab.mean(axis=(0, 1))
    delta = (target_lab_mean - source_lab_mean) * strength
    adjusted_lab = source_lab + delta
    adjusted_rgb = lab2rgb(adjusted_lab)
    return np.clip(adjusted_rgb * 255.0, 0, 255).astype(np.uint8)
