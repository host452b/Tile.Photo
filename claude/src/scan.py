from __future__ import annotations

import colorsys
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


@dataclass
class TilePool:
    lab: np.ndarray
    thumbs_paths: list[str]
    source_paths: list[str]


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _iter_candidate_files(base_dir: Path):
    for path in base_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _load_and_thumbnail(path: Path, tile_px: int) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((tile_px, tile_px), Image.Resampling.LANCZOS)
        return np.asarray(img, dtype=np.uint8)


def _lab_mean(tile_rgb: np.ndarray) -> np.ndarray:
    return rgb2lab(tile_rgb.astype(np.float32) / 255.0).mean(axis=(0, 1)).astype(np.float32)


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
        labs[i] = _lab_mean(tile_rgb)
        thumbs_paths.append(str(thumb_path))
        source_paths.append(f"demo://tile_{i:04d}")
    return TilePool(lab=labs, thumbs_paths=thumbs_paths, source_paths=source_paths)


def _scan_directory(
    base_dir: Path,
    cache_dir: Path,
    tile_px: int,
) -> TilePool:
    thumbs_dir = cache_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    labs: list[np.ndarray] = []
    thumbs_paths: list[str] = []
    source_paths: list[str] = []

    files = list(_iter_candidate_files(base_dir))
    for path in tqdm(files, desc="scan", unit="img"):
        try:
            tile_rgb = _load_and_thumbnail(path, tile_px)
        except Exception as e:
            logger.warning("skip %s: %s", path, e)
            continue
        thumb_path = thumbs_dir / f"{_sha1(str(path))}.jpg"
        Image.fromarray(tile_rgb).save(thumb_path, quality=92)
        labs.append(_lab_mean(tile_rgb))
        thumbs_paths.append(str(thumb_path))
        source_paths.append(str(path))

    if not labs:
        return TilePool(
            lab=np.zeros((0, 3), dtype=np.float32),
            thumbs_paths=[],
            source_paths=[],
        )
    return TilePool(
        lab=np.stack(labs).astype(np.float32),
        thumbs_paths=thumbs_paths,
        source_paths=source_paths,
    )


def build_pool(
    base_dir: Path,
    cache_dir: Path,
    tile_px: int,
    demo_mode: bool = False,
) -> TilePool:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(base_dir)

    use_demo = demo_mode or not base_dir.exists()
    if not use_demo and base_dir.exists():
        has_any = any(True for _ in _iter_candidate_files(base_dir))
        if not has_any:
            use_demo = True

    if use_demo:
        return _synthesize_demo_tiles(cache_dir, tile_px)
    return _scan_directory(base_dir, cache_dir, tile_px)
