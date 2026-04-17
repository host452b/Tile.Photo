from __future__ import annotations
from pathlib import Path
from typing import Dict, List
from PIL import Image, UnidentifiedImageError
import fnmatch
import hashlib
import json as _json
import re
import pickle
from dataclasses import dataclass
import numpy as np
from skimage.color import rgb2lab

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


@dataclass
class TileIndex:
    paths: List[str]
    lab_mean: np.ndarray  # shape (N, 3), dtype float32
    clip_emb: np.ndarray | None = None  # shape (N, D) if computed


def _lab_mean_of(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((64, 64), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    lab = rgb2lab(arr)
    return lab.reshape(-1, 3).mean(axis=0).astype(np.float32)


def build_tile_index(tile_dir: str, min_side: int = 32) -> TileIndex:
    paths = scan_tile_dir(tile_dir, min_side=min_side)
    if not paths:
        raise ValueError(f"no valid tiles in {tile_dir}")
    lab = np.stack([_lab_mean_of(p) for p in paths]).astype(np.float32)
    return TileIndex(paths=paths, lab_mean=lab)


def _cache_key(tile_dir: str, min_side: int) -> str:
    h = hashlib.sha1(f"{Path(tile_dir).resolve()}::{min_side}".encode()).hexdigest()[:16]
    return f"tileindex_{h}.pkl"


def load_or_build_index(tile_dir: str, cache_dir: str = ".cache", min_side: int = 32) -> TileIndex:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / _cache_key(tile_dir, min_side)
    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)
    idx = build_tile_index(tile_dir, min_side=min_side)
    with cache_path.open("wb") as f:
        pickle.dump(idx, f)
    return idx


def add_clip_embeddings(idx: TileIndex, model_name: str = "ViT-B-32",
                        pretrained: str = "openai", batch_size: int = 32) -> TileIndex:
    """Attach L2-normalized CLIP image embeddings. Requires open_clip + torch."""
    import open_clip
    import torch
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    embs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(idx.paths), batch_size):
            batch_paths = idx.paths[start:start + batch_size]
            batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
            feat = model.encode_image(batch)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            embs.append(feat.cpu().numpy().astype(np.float32))
    clip_emb = np.concatenate(embs, axis=0)
    return TileIndex(paths=idx.paths, lab_mean=idx.lab_mean, clip_emb=clip_emb)


def _glob_match(path: str, pattern: str) -> bool:
    """Match *path* against *pattern*, supporting ** for zero-or-more path segments."""
    if "**" not in pattern:
        return fnmatch.fnmatch(path, pattern)
    # Translate glob pattern to regex: **/ → (.+/)? (zero or more dir segments)
    result = ""
    i = 0
    while i < len(pattern):
        if pattern[i:i+3] == "**/":
            result += "(.+/)?"
            i += 3
        elif pattern[i:i+2] == "**":
            result += ".*"
            i += 2
        elif pattern[i] == "*":
            result += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            result += "[^/]"
            i += 1
        else:
            result += re.escape(pattern[i])
            i += 1
    return bool(re.match("^" + result + "$", path))


def load_tags(tile_dir: str, paths: List[str]) -> Dict[str, str]:
    """Return {abs_path: tag_name}. First matching pattern wins; unmatched → 'untagged'."""
    tag_file = Path(tile_dir) / "tags.json"
    if not tag_file.exists():
        return {p: "untagged" for p in paths}
    patterns = _json.loads(tag_file.read_text())
    root = Path(tile_dir).resolve()
    out: Dict[str, str] = {}
    for p in paths:
        rel = str(Path(p).resolve().relative_to(root))
        match = "untagged"
        for pat, name in patterns.items():
            if _glob_match(rel, pat):
                match = name
                break
        out[p] = match
    return out
