from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from mosaic.pool import lab_mean


@dataclass
class TargetGrid:
    canvas: np.ndarray       # (H, W, 3) uint8, center-cropped to grid aspect
    lab_means: np.ndarray    # (grid_h, grid_w, 3) float32


def _center_crop_to_aspect(img: Image.Image, aspect: float) -> Image.Image:
    w, h = img.size
    cur = w / h
    if cur > aspect:
        new_w = int(h * aspect)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    new_h = int(w / aspect)
    top = (h - new_h) // 2
    return img.crop((0, top, w, top + new_h))


def load_and_slice(path: Path, grid_w: int, grid_h: int) -> TargetGrid:
    """Load target → center-crop to grid_w:grid_h aspect → compute per-cell LAB means."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = _center_crop_to_aspect(im, grid_w / grid_h)
        scale_w = grid_w * 8
        scale_h = grid_h * 8
        im = im.resize((scale_w, scale_h), Image.LANCZOS)
        canvas = np.array(im, dtype=np.uint8)

    cell_w = scale_w // grid_w
    cell_h = scale_h // grid_h
    lab_means = np.zeros((grid_h, grid_w, 3), dtype=np.float32)
    for r in range(grid_h):
        for c in range(grid_w):
            patch = canvas[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            lab_means[r, c] = lab_mean(patch)
    return TargetGrid(canvas=canvas, lab_means=lab_means)
