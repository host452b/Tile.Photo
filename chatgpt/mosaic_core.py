"""Pure-function helpers for the photomosaic toy.

Every non-IO function in this module is deterministic and unit-tested.
IO helpers (`scan_tile_pool`, `ensure_seed_tiles`, `export_deepzoom`) are
thin wrappers that do one thing each.
"""

from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab


# ---------- dataclasses ----------

@dataclass
class TileRecord:
    path: Path
    lab_mean: np.ndarray  # float32[3]
    rgb_thumb: np.ndarray  # uint8[64, 64, 3]


@dataclass
class MosaicConfig:
    target_path: Optional[Path] = Path("target.jpg")
    tile_dir: Path = Path("my_tiles")
    grid_w: int = 120
    grid_h: int = 68
    tile_px: int = 16
    k_candidates: int = 32
    lambda_repeat: float = 0.5
    mu_neighbor: float = 0.3
    tau_transfer: float = 0.4
    cache_path: Path = Path(".cache/tiles.pkl")
    out_dir: Path = Path("out")


@dataclass
class ReportBundle:
    text: str
    usage_bar_fig: object  # matplotlib.figure.Figure — kept as object to avoid import at module top
    cold_wall_fig: object


# ---------- color ----------

def lab_mean(rgb: np.ndarray) -> np.ndarray:
    """Return LAB mean of an H×W×3 uint8 RGB array as float32[3]."""
    lab = rgb2lab(rgb.astype(np.float64) / 255.0)
    return lab.reshape(-1, 3).mean(axis=0).astype(np.float32)


def ciede2000(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    """Scalar ΔE*_CIEDE2000 between two LAB triples (shape (3,))."""
    a = np.asarray(lab_a, dtype=np.float64).reshape(1, 1, 3)
    b = np.asarray(lab_b, dtype=np.float64).reshape(1, 1, 3)
    return float(deltaE_ciede2000(a, b).squeeze())


def reinhard_transfer(
    tile_rgb: np.ndarray,
    target_lab_mean: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Shift tile's LAB mean onto target_lab_mean, then mix by τ∈[0,1]."""
    if tau == 0.0:
        return tile_rgb
    tile_lab = rgb2lab(tile_rgb.astype(np.float64) / 255.0)
    tile_lab_mean = tile_lab.reshape(-1, 3).mean(axis=0)
    offset = np.asarray(target_lab_mean, dtype=np.float64) - tile_lab_mean
    transferred_lab = tile_lab + offset
    transferred_rgb = np.clip(lab2rgb(transferred_lab) * 255.0, 0, 255).astype(np.uint8)
    if tau == 1.0:
        return transferred_rgb
    blend = tile_rgb.astype(np.float32) * (1.0 - tau) + transferred_rgb.astype(np.float32) * tau
    return np.clip(blend, 0, 255).astype(np.uint8)
