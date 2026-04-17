from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab


EPS = 1e-6


def reinhard_transfer(
    tile_lab: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Shift tile_lab toward (target_mean, target_std) by strength tau in [0,1].

    tau=0: identity. tau=1: result mean == target_mean, per-channel std == target_std.
    """
    flat = tile_lab.reshape(-1, 3)
    src_mean = flat.mean(axis=0).astype(np.float32)
    src_std = flat.std(axis=0).astype(np.float32)
    scale = target_std / (src_std + EPS)
    eff_scale = (1.0 - tau) + tau * scale
    shifted = (flat - src_mean) * eff_scale + src_mean + tau * (target_mean - src_mean)
    return shifted.reshape(tile_lab.shape).astype(np.float32)


def compose(
    tiles: list,
    grid: np.ndarray,
    cell_lab_means: np.ndarray,
    tile_px: int,
    tau: float,
) -> np.ndarray:
    """Compose the final mosaic as a (grid_h*tile_px, grid_w*tile_px, 3) uint8 ndarray."""
    grid_h, grid_w = grid.shape
    out = np.zeros((grid_h * tile_px, grid_w * tile_px, 3), dtype=np.uint8)

    for r in range(grid_h):
        for c in range(grid_w):
            t = tiles[int(grid[r, c])]
            thumb = t.thumb
            if thumb.shape[0] != tile_px or thumb.shape[1] != tile_px:
                thumb = np.array(
                    Image.fromarray(thumb).resize((tile_px, tile_px), Image.LANCZOS),
                    dtype=np.uint8,
                )
            if tau > 0:
                thumb_lab = rgb2lab(thumb.astype(np.float32) / 255.0).astype(np.float32)
                target_mean = cell_lab_means[r, c]
                target_std = thumb_lab.reshape(-1, 3).std(axis=0).astype(np.float32)
                shifted = reinhard_transfer(thumb_lab, target_mean, target_std, tau)
                rgb = np.clip(lab2rgb(shifted) * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb = thumb
            out[r * tile_px : (r + 1) * tile_px, c * tile_px : (c + 1) * tile_px] = rgb
    return out
