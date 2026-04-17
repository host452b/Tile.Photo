from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def lab_mean(rgb_uint8: np.ndarray) -> np.ndarray:
    """Return the mean LAB color of an (H,W,3) uint8 RGB image as (3,) float32."""
    lab = rgb2lab(rgb_uint8.astype(np.float32) / 255.0)
    return lab.reshape(-1, 3).mean(axis=0).astype(np.float32)


@dataclass
class Tile:
    path: Path
    lab: np.ndarray     # shape (3,), float32
    thumb: np.ndarray   # shape (thumb_px, thumb_px, 3), uint8


def _pool_hash(pool_dir: Path, thumb_px: int) -> str:
    files = sorted(
        p for p in pool_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    h = hashlib.sha256()
    h.update(str(thumb_px).encode())
    for p in files:
        st = p.stat()
        h.update(f"{p.relative_to(pool_dir)}|{st.st_size}|{int(st.st_mtime)}\n".encode())
    return h.hexdigest()[:16]


def _load_and_thumb(path: Path, thumb_px: int) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        im = im.crop((left, top, left + s, top + s))
        im = im.resize((thumb_px, thumb_px), Image.LANCZOS)
        thumb = np.array(im, dtype=np.uint8)
    return thumb, lab_mean(thumb)


def scan_pool(pool_dir: Path, cache_dir: Path, thumb_px: int = 32) -> list[Tile]:
    """Scan pool_dir recursively; return list[Tile]. Caches to cache_dir/pool_<hash>.pkl."""
    pool_dir = Path(pool_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _pool_hash(pool_dir, thumb_px)
    cache_path = cache_dir / f"pool_{key}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)["tiles"]

    files = sorted(
        p for p in pool_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    tiles: list[Tile] = []
    for p in tqdm(files, desc="scanning pool"):
        try:
            thumb, lab = _load_and_thumb(p, thumb_px)
        except Exception as e:
            print(f"skip {p.name}: {e}")
            continue
        tiles.append(Tile(path=p, lab=lab, thumb=thumb))

    with cache_path.open("wb") as f:
        pickle.dump({"tiles": tiles, "thumb_px": thumb_px}, f)
    return tiles
