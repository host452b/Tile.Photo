"""Reinhard LAB 色调迁移：把 tile 的 (μ, σ) 搬到 target_patch 的 (μ, σ)，按 τ 线性混合。"""
from __future__ import annotations

import numpy as np
from skimage.color import lab2rgb, rgb2lab

EPS = 1e-6


def reinhard_transfer(tile_rgb: np.ndarray, target_patch_rgb: np.ndarray, tau: float) -> np.ndarray:
    """
    tile_rgb, target_patch_rgb: HxWx3 uint8
    tau ∈ [0, 1]
    返回 HxWx3 uint8
    """
    if tau <= 0.0:
        return tile_rgb.copy()

    tile_lab = rgb2lab(tile_rgb / 255.0)
    target_lab = rgb2lab(target_patch_rgb / 255.0)

    tile_mean = tile_lab.reshape(-1, 3).mean(axis=0)
    tile_std = tile_lab.reshape(-1, 3).std(axis=0)
    target_mean = target_lab.reshape(-1, 3).mean(axis=0)
    target_std = target_lab.reshape(-1, 3).std(axis=0)

    scale = target_std / np.maximum(tile_std, EPS)
    transferred_lab = (tile_lab - tile_mean) * scale + target_mean

    mixed_lab = (1.0 - tau) * tile_lab + tau * transferred_lab
    mixed_rgb = np.clip(lab2rgb(mixed_lab) * 255.0, 0, 255).astype(np.uint8)
    return mixed_rgb
