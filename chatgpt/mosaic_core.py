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
