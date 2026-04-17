"""底图池扫描 + LAB 平均色 + pickle 缓存（按 mtime 增量）。"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THUMBNAIL_SIZE = 16


def load_cache(cache_path: Path) -> dict:
    if not Path(cache_path).exists():
        return {}
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def save_cache(cache_path: Path, features: dict) -> None:
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(features, f)


def _compute_features(img_path: Path) -> dict:
    img = Image.open(img_path).convert("RGB")
    thumb = img.resize((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.LANCZOS)
    thumb_arr = np.asarray(thumb, dtype=np.uint8)
    lab = rgb2lab(thumb_arr / 255.0)
    return {
        "mtime": img_path.stat().st_mtime,
        "lab_mean": lab.reshape(-1, 3).mean(axis=0).astype(np.float32),
        "thumbnail": thumb_arr,
    }


def scan_pool(pool_dir: Path, cache_path: Path) -> dict:
    """扫描 pool_dir，增量更新 cache_path，返回 {path_str: feature_entry}。"""
    pool_dir = Path(pool_dir)
    cache_path = Path(cache_path)
    cache = load_cache(cache_path)

    current_paths = set()
    skipped = []

    for entry in pool_dir.rglob("*"):
        if not entry.is_file() or entry.suffix.lower() not in IMG_EXTENSIONS:
            continue
        path_str = str(entry)
        current_paths.add(path_str)

        mtime = entry.stat().st_mtime
        if path_str in cache and cache[path_str]["mtime"] == mtime:
            continue
        try:
            cache[path_str] = _compute_features(entry)
        except Exception as exc:
            skipped.append((path_str, str(exc)))
            cache.pop(path_str, None)

    for stale in set(cache.keys()) - current_paths:
        cache.pop(stale)

    save_cache(cache_path, cache)

    if skipped:
        print(f"[pool] skipped {len(skipped)} files (corrupt or unreadable)")
    print(f"[pool] scanned {len(cache)} tiles")
    return cache
