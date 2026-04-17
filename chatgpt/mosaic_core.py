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


# ---------- grid / render ----------

def split_target(img: Image.Image, grid_w: int, grid_h: int) -> np.ndarray:
    """Split img into grid_h × grid_w patches and return each patch's LAB mean.

    Returns: ndarray[grid_h, grid_w, 3] float32.
    """
    patch_w = img.width // grid_w
    patch_h = img.height // grid_h
    resized = img.resize((grid_w * patch_w, grid_h * patch_h), Image.BILINEAR)
    rgb = np.asarray(resized, dtype=np.uint8)
    lab = rgb2lab(rgb.astype(np.float64) / 255.0)
    # Reshape to (grid_h, patch_h, grid_w, patch_w, 3) then mean over patch_h, patch_w.
    reshaped = lab.reshape(grid_h, patch_h, grid_w, patch_w, 3)
    return reshaped.mean(axis=(1, 3)).astype(np.float32)


def render_mosaic(
    assignment: np.ndarray,
    tile_records: list[TileRecord],
    tile_px: int,
    tau: float,
    target_lab: np.ndarray,
) -> Image.Image:
    """Paste tiles onto an (grid_h*tile_px, grid_w*tile_px) canvas.

    assignment: int64[grid_h, grid_w] — tile record index per cell.
    target_lab: float32[grid_h, grid_w, 3] — per-cell LAB target for τ transfer.
    """
    grid_h, grid_w = assignment.shape
    canvas = np.zeros((grid_h * tile_px, grid_w * tile_px, 3), dtype=np.uint8)
    for r in range(grid_h):
        for c in range(grid_w):
            rec = tile_records[int(assignment[r, c])]
            thumb = rec.rgb_thumb
            if thumb.shape[0] != tile_px or thumb.shape[1] != tile_px:
                thumb = np.asarray(
                    Image.fromarray(thumb).resize((tile_px, tile_px), Image.BILINEAR)
                )
            if tau > 0.0:
                thumb = reinhard_transfer(thumb, target_lab[r, c], tau)
            y0, x0 = r * tile_px, c * tile_px
            canvas[y0:y0 + tile_px, x0:x0 + tile_px] = thumb
    return Image.fromarray(canvas)


# ---------- matching ----------

def build_faiss_index(tile_labs: np.ndarray):
    """Build a flat L2 faiss index over an N×3 LAB matrix."""
    import faiss  # lazy import to avoid paying at module load
    arr = np.ascontiguousarray(tile_labs, dtype=np.float32)
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    return index


def knn_candidates(target_lab: np.ndarray, faiss_index, k: int = 32) -> np.ndarray:
    """Return top-k tile indices per target cell: int64[H*W, k]."""
    h, w, _ = target_lab.shape
    query = np.ascontiguousarray(target_lab.reshape(h * w, 3), dtype=np.float32)
    effective_k = min(k, faiss_index.ntotal)
    _dists, idxs = faiss_index.search(query, effective_k)
    return idxs.astype(np.int64)
