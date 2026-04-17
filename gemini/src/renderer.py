from __future__ import annotations
import numpy as np
from skimage.color import rgb2lab, lab2rgb


def reinhard_tone_transfer(src_rgb: np.ndarray, target_rgb: np.ndarray, tau: float) -> np.ndarray:
    """
    Reinhard color transfer in LAB space, interpolated by tau ∈ [0, 1].
    tau=0 returns src unchanged; tau=1 shifts src's per-channel mean/std toward target's.
    Inputs are uint8 HxWx3; output uint8 HxWx3.
    """
    if tau <= 0.0:
        return src_rgb.copy()
    src_lab = rgb2lab(src_rgb.astype(np.float32) / 255.0)
    tgt_lab = rgb2lab(target_rgb.astype(np.float32) / 255.0)
    src_mean = src_lab.reshape(-1, 3).mean(axis=0)
    src_std = src_lab.reshape(-1, 3).std(axis=0) + 1e-6
    tgt_mean = tgt_lab.reshape(-1, 3).mean(axis=0)
    tgt_std = tgt_lab.reshape(-1, 3).std(axis=0) + 1e-6
    # Full Reinhard: shifted = (src - src_mean) * (tgt_std/src_std) + tgt_mean
    shifted = (src_lab - src_mean) * (tgt_std / src_std) + tgt_mean
    blended = (1.0 - tau) * src_lab + tau * shifted
    out_rgb = np.clip(lab2rgb(blended), 0.0, 1.0)
    return (out_rgb * 255.0 + 0.5).astype(np.uint8)


from PIL import Image as _Image
from pathlib import Path as _Path
from collections import Counter


def _load_and_fit(path: str, size: int) -> np.ndarray:
    with _Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        # center-crop to square, then resize to size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        im = im.crop((left, top, left + s, top + s)).resize((size, size), _Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def render_mosaic(assignment: np.ndarray, tile_paths: list[str],
                  cell_rgb: np.ndarray, tile_px: int, tau: float
                  ) -> tuple[_Image.Image, dict[int, int]]:
    """
    assignment: (rows, cols) tile indices
    tile_paths: list of file paths, indexed by assignment values
    cell_rgb:   (rows, cols, patch_px, patch_px, 3) uint8 — target color per cell (for tone transfer)
    Returns (PIL image, usage_counter).
    """
    rows, cols = assignment.shape
    out = np.zeros((rows * tile_px, cols * tile_px, 3), dtype=np.uint8)
    tile_cache: dict[int, np.ndarray] = {}
    usage: Counter[int] = Counter()
    for r in range(rows):
        for c in range(cols):
            t = int(assignment[r, c])
            usage[t] += 1
            if t not in tile_cache:
                tile_cache[t] = _load_and_fit(tile_paths[t], tile_px)
            tile_img = tile_cache[t]
            if tau > 0.0:
                # Use the target cell's mean color as a (1x1) reference, tiled
                tgt_mean = cell_rgb[r, c].reshape(-1, 3).mean(axis=0).astype(np.uint8)
                tgt_patch = np.full((tile_px, tile_px, 3), tgt_mean, dtype=np.uint8)
                tile_img = reinhard_tone_transfer(tile_img, tgt_patch, tau=tau)
            out[r*tile_px:(r+1)*tile_px, c*tile_px:(c+1)*tile_px] = tile_img
    return _Image.fromarray(out), dict(usage)
