from __future__ import annotations

import colorsys
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab


@dataclass
class TilePool:
    lab: np.ndarray
    thumbs_paths: list[str]
    source_paths: list[str]


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _synthesize_demo_tiles(
    cache_dir: Path,
    tile_px: int,
    count: int = 500,
) -> TilePool:
    thumbs_dir = cache_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    labs = np.zeros((count, 3), dtype=np.float32)
    thumbs_paths: list[str] = []
    source_paths: list[str] = []

    rng = np.random.default_rng(seed=42)
    for i in range(count):
        h = i / count
        s = 0.4 + 0.6 * rng.random()
        v = 0.3 + 0.7 * rng.random()
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        rgb = np.array([r, g, b], dtype=np.float32)
        tile_rgb = np.broadcast_to(
            (rgb * 255).astype(np.uint8), (tile_px, tile_px, 3)
        ).copy()
        thumb_path = thumbs_dir / f"demo_{i:04d}.jpg"
        Image.fromarray(tile_rgb).save(thumb_path, quality=92)
        lab = rgb2lab(tile_rgb.astype(np.float32) / 255.0).mean(axis=(0, 1))
        labs[i] = lab.astype(np.float32)
        thumbs_paths.append(str(thumb_path))
        source_paths.append(f"demo://tile_{i:04d}")
    return TilePool(lab=labs, thumbs_paths=thumbs_paths, source_paths=source_paths)


def build_pool(
    base_dir: Path,
    cache_dir: Path,
    tile_px: int,
    demo_mode: bool = False,
) -> TilePool:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(base_dir)

    use_demo = demo_mode or (not base_dir.exists()) or (
        not any(base_dir.iterdir()) if base_dir.exists() else True
    )
    if use_demo:
        return _synthesize_demo_tiles(cache_dir, tile_px)

    raise NotImplementedError("Real-file scanning lands in Task 3")
