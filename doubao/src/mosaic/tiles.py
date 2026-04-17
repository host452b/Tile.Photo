"""Tile pool: scan images, compute LAB mean + cached thumbnails."""
from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.color import rgb2lab
from tqdm import tqdm

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SKIP_DIR_PARTS = {".git", "__pycache__", ".cache", "@eaDir", ".ipynb_checkpoints"}


@dataclass
class TilePool:
    paths: list[Path]
    lab_means: np.ndarray          # (N, 3) float32
    thumbnails: np.ndarray         # (N, tile_px, tile_px, 3) uint8

    def __len__(self) -> int:
        return len(self.paths)


def scan_tile_pool(root: Path) -> list[Path]:
    """Recursively collect image paths, skip junk."""
    root = root.resolve()
    out: list[Path] = []
    for p in root.rglob("*"):
        if any(part in SKIP_DIR_PARTS or part.endswith(".app") for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            out.append(p)
    out.sort()
    return out


def _prepare_tile(path: Path, tile_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Load -> center-crop to square -> resize -> (thumb uint8, lab_mean float32)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((tile_px, tile_px), Image.LANCZOS)
        thumb = np.asarray(im, dtype=np.uint8)
    lab = rgb2lab(thumb.astype(np.float32) / 255.0)
    return thumb, lab.reshape(-1, 3).mean(axis=0).astype(np.float32)


def load_or_build(
    source_dir: Path,
    tile_px: int,
    cache_dir: Path,
) -> TilePool:
    """Return a TilePool, using the pickle cache where mtime matches."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"tiles_{tile_px}.pkl"

    prev: dict[str, dict] = {}
    if cache_file.exists():
        try:
            with cache_file.open("rb") as f:
                prev = pickle.load(f)
        except Exception as e:
            warnings.warn(f"cache corrupted ({e}), rebuilding")
            prev = {}

    paths = scan_tile_pool(source_dir)
    entries: dict[str, dict] = {}
    thumbs = []
    lab_means = []

    iterator = tqdm(paths, desc="tiles") if len(paths) > 50 else paths
    for p in iterator:
        key = str(p)
        mtime = p.stat().st_mtime
        cached = prev.get(key)
        if cached and cached.get("mtime") == mtime:
            thumb = cached["thumb"]
            lab_mean = cached["lab_mean"]
        else:
            try:
                thumb, lab_mean = _prepare_tile(p, tile_px)
            except (UnidentifiedImageError, OSError) as e:
                warnings.warn(f"skip corrupted tile {p.name}: {e}")
                continue
        entries[key] = {"mtime": mtime, "thumb": thumb, "lab_mean": lab_mean}
        thumbs.append(thumb)
        lab_means.append(lab_mean)

    with cache_file.open("wb") as f:
        pickle.dump(entries, f)

    if not thumbs:
        raise ValueError(f"no valid tiles found in {source_dir}")

    return TilePool(
        paths=[Path(k) for k in entries.keys()],
        lab_means=np.stack(lab_means).astype(np.float32),
        thumbnails=np.stack(thumbs).astype(np.uint8),
    )
