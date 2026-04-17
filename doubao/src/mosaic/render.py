"""Render the mosaic: optional Reinhard tone transfer + paste tiles."""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab

from .config import MosaicConfig
from .tiles import TilePool


def reinhard_transfer(
    tile_rgb: np.ndarray,
    target_lab: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Shift tile's LAB mean toward target_lab by factor tau.

    tile_rgb: (H, W, 3) uint8
    target_lab: (3,) float — target cell's LAB mean
    tau: 0..1 — 0 means no change, 1 means fully match target LAB mean
    Returns (H, W, 3) uint8.
    """
    if tau <= 0.0:
        return tile_rgb
    lab = rgb2lab(tile_rgb.astype(np.float32) / 255.0)
    cur_mean = lab.reshape(-1, 3).mean(0)
    shifted = lab + tau * (target_lab - cur_mean)
    shifted[..., 0] = np.clip(shifted[..., 0], 0, 100)
    shifted[..., 1] = np.clip(shifted[..., 1], -128, 127)
    shifted[..., 2] = np.clip(shifted[..., 2], -128, 127)
    rgb = lab2rgb(shifted)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def render_mosaic(
    assignment: np.ndarray,
    pool: TilePool,
    target_cells: np.ndarray,
    cfg: MosaicConfig,
) -> Image.Image:
    """Compose the final mosaic.

    assignment: (rows, cols) int — tile index per cell
    target_cells: (rows, cols, 3) — LAB mean per cell (used for tone transfer)
    Returns PIL.Image of size (cols * tile_px, rows * tile_px).
    """
    rows, cols = assignment.shape
    tile_px = cfg.tile_px
    canvas = np.zeros((rows * tile_px, cols * tile_px, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            idx = int(assignment[r, c])
            thumb = pool.thumbnails[idx]
            if cfg.tau_tone > 0.0:
                thumb = reinhard_transfer(thumb, target_cells[r, c], cfg.tau_tone)
            canvas[
                r * tile_px : (r + 1) * tile_px,
                c * tile_px : (c + 1) * tile_px,
            ] = thumb

    return Image.fromarray(canvas)
