"""读目标图、按网格宽高比 center-crop、分网格计算每格 LAB 均值与方差。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def _center_crop_to_aspect(img: Image.Image, target_cols: int, target_rows: int) -> Image.Image:
    w, h = img.size
    target_ratio = target_cols / target_rows
    img_ratio = w / h
    if abs(img_ratio - target_ratio) < 1e-6:
        return img
    if img_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    new_h = int(w / target_ratio)
    top = (h - new_h) // 2
    return img.crop((0, top, w, top + new_h))


def load_and_grid(target_path: Path, grid_cols: int, grid_rows: int) -> dict:
    """返回 {'shape': (rows, cols), 'cells': [{'row', 'col', 'lab_mean', 'variance'}, ...], 'image': PIL.Image}."""
    img = Image.open(target_path).convert("RGB")
    img = _center_crop_to_aspect(img, grid_cols, grid_rows)

    arr = np.asarray(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    usable = arr[: cell_h * grid_rows, : cell_w * grid_cols]
    lab = rgb2lab(usable / 255.0).astype(np.float32)

    cells = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            patch = lab[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            cells.append({
                "row": r,
                "col": c,
                "lab_mean": patch.reshape(-1, 3).mean(axis=0),
                "variance": float(patch.reshape(-1, 3).var(axis=0).sum()),
            })

    return {
        "shape": (grid_rows, grid_cols),
        "cells": cells,
        "image": img,
        "cell_size": (cell_h, cell_w),
    }
