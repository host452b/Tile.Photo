from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src import tone
from src.scan import TilePool


def _load_thumb_cached(path: str, tile_px: int, cache: dict[str, np.ndarray]) -> np.ndarray:
    if path in cache:
        return cache[path]
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (tile_px, tile_px):
            img = img.resize((tile_px, tile_px), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.uint8)
    cache[path] = arr
    return arr


def render_mosaic(
    index_grid: np.ndarray,
    pool: TilePool,
    tile_px: int,
    output_path: Path,
) -> Image.Image:
    img, _ = render_mosaic_with_usage(index_grid, pool, tile_px, output_path)
    return img


def render_mosaic_with_usage(
    index_grid: np.ndarray,
    pool: TilePool,
    tile_px: int,
    output_path: Path,
    *,
    target_rgb: np.ndarray | None = None,
    tone_strength: float = 0.0,
) -> tuple[Image.Image, dict[int, int]]:
    h, w = index_grid.shape
    canvas = np.zeros((h * tile_px, w * tile_px, 3), dtype=np.uint8)
    cache: dict[str, np.ndarray] = {}
    usage: Counter[int] = Counter()
    apply_tone = tone_strength > 0.0 and target_rgb is not None

    for row in range(h):
        for col in range(w):
            idx = int(index_grid[row, col])
            thumb_path = pool.thumbs_paths[idx]
            raw_tile = _load_thumb_cached(thumb_path, tile_px, cache)
            if apply_tone:
                patch = target_rgb[
                    row * tile_px : (row + 1) * tile_px,
                    col * tile_px : (col + 1) * tile_px,
                ]
                tile = tone.reinhard_transfer(raw_tile, patch, tone_strength)
            else:
                tile = raw_tile
            canvas[
                row * tile_px : (row + 1) * tile_px,
                col * tile_px : (col + 1) * tile_px,
            ] = tile
            usage[idx] += 1

    img = Image.fromarray(canvas)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return img, dict(usage)
