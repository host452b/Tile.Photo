from __future__ import annotations
from pathlib import Path
from typing import List
from PIL import Image, UnidentifiedImageError

_EXTS = {".jpg", ".jpeg", ".png"}


def scan_tile_dir(tile_dir: str, min_side: int = 32) -> List[str]:
    """Recursively find valid image files, skipping broken and tiny ones."""
    root = Path(tile_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"tile_dir does not exist: {tile_dir}")
    out: List[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in _EXTS:
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                if min(im.size) < min_side:
                    continue
        except (UnidentifiedImageError, OSError):
            continue
        out.append(str(p))
    return out
