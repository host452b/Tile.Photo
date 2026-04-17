# Bead Mosaic Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Jupyter notebook that turns a target image + a folder of photos into a photomosaic PNG, with a zero-config demo mode so the pipeline runs top-to-bottom on a fresh clone.

**Architecture:** Three focused modules (`scan` / `match` / `render`) behind a TilePool dataclass, orchestrated by a 7-cell notebook. LAB color-mean nearest-neighbor matching via numpy broadcast. Demo mode synthesizes 500 HSV-sampled color tiles so the smoke test doesn't need user photos.

**Tech Stack:** Python 3.12, Pillow, numpy, scikit-image, tqdm, pillow-heif, pytest, jupyter/nbformat.

**Reference spec:** `docs/superpowers/specs/2026-04-17-bead-mosaic-phase1-design.md`

**Working directory:** `/Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/claude`

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `pytest.ini`

- [ ] **Step 1: Create `requirements.txt`**

```
pillow>=10
numpy>=1.26
scikit-image>=0.22
tqdm>=4.66
pillow-heif>=0.15
pytest>=8
jupyter>=1
nbformat>=5
```

- [ ] **Step 2: Create `.gitignore`**

```
.cache/
__pycache__/
*.pyc
.ipynb_checkpoints/
.pytest_cache/
output.png
.DS_Store
.venv/
```

- [ ] **Step 3: Create `src/__init__.py`**

```python
from pillow_heif import register_heif_opener

register_heif_opener()
```

This registers HEIC as a first-class format on import — Pillow then opens iPhone photos transparently.

- [ ] **Step 4: Create `tests/__init__.py`**

Empty file.

- [ ] **Step 5: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v
```

- [ ] **Step 6: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: successful install, no compile errors on Apple Silicon.

- [ ] **Step 7: Sanity-check empty test run**

Run: `pytest`
Expected: `no tests ran` (exit code 5 is OK — no tests yet).

- [ ] **Step 8: Commit**

```bash
git add requirements.txt .gitignore src/__init__.py tests/__init__.py pytest.ini
git commit -m "Scaffold Phase 1 project structure"
```

---

## Task 2: TilePool + demo mode synthesis

**Files:**
- Create: `src/scan.py`
- Create: `tests/test_scan.py`

- [ ] **Step 1: Write failing test for demo mode**

Create `tests/test_scan.py`:

```python
from pathlib import Path

import numpy as np
from PIL import Image

from src import scan


def test_demo_mode_returns_500_tiles(tmp_path):
    pool = scan.build_pool(
        base_dir=tmp_path / "nonexistent",
        cache_dir=tmp_path / "cache",
        tile_px=24,
        demo_mode=True,
    )
    assert pool.lab.shape == (500, 3)
    assert pool.lab.dtype == np.float32
    assert len(pool.thumbs_paths) == 500
    assert len(pool.source_paths) == 500
    for p in pool.thumbs_paths[:5]:
        img = Image.open(p)
        assert img.size == (24, 24)


def test_demo_mode_covers_hue_spectrum(tmp_path):
    pool = scan.build_pool(
        base_dir=tmp_path / "nonexistent",
        cache_dir=tmp_path / "cache",
        tile_px=24,
        demo_mode=True,
    )
    a_channel = pool.lab[:, 1]
    b_channel = pool.lab[:, 2]
    assert a_channel.max() - a_channel.min() > 50
    assert b_channel.max() - b_channel.min() > 50
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_scan.py -v`
Expected: FAIL with `ImportError: cannot import name 'scan'` or `AttributeError: module has no attribute 'build_pool'`.

- [ ] **Step 3: Implement `src/scan.py` demo mode**

```python
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

    use_demo = demo_mode or (not base_dir.exists()) or (not any(base_dir.iterdir()) if base_dir.exists() else True)
    if use_demo:
        return _synthesize_demo_tiles(cache_dir, tile_px)

    raise NotImplementedError("Real-file scanning lands in Task 3")
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_scan.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scan.py tests/test_scan.py
git commit -m "Add TilePool + demo mode tile synthesis"
```

---

## Task 3: Real-file scanning (EXIF + HEIC + thumb cache)

**Files:**
- Modify: `src/scan.py`
- Modify: `tests/test_scan.py`

- [ ] **Step 1: Write failing test with real-file fixture**

Append to `tests/test_scan.py`:

```python
def _write_solid_jpg(path: Path, rgb: tuple[int, int, int], size: int = 128) -> None:
    arr = np.full((size, size, 3), rgb, dtype=np.uint8)
    Image.fromarray(arr).save(path, quality=95)


def test_scans_real_files(tmp_path):
    base = tmp_path / "photos"
    base.mkdir()
    _write_solid_jpg(base / "red.jpg", (230, 20, 20))
    _write_solid_jpg(base / "green.jpg", (20, 200, 20))
    _write_solid_jpg(base / "blue.jpg", (20, 20, 220))

    pool = scan.build_pool(
        base_dir=base,
        cache_dir=tmp_path / "cache",
        tile_px=24,
        demo_mode=False,
    )
    assert pool.lab.shape == (3, 3)
    assert len(pool.source_paths) == 3
    assert all(Path(p).exists() for p in pool.thumbs_paths)
    for p in pool.thumbs_paths:
        assert Image.open(p).size == (24, 24)


def test_skips_corrupt_files(tmp_path, caplog):
    base = tmp_path / "photos"
    base.mkdir()
    _write_solid_jpg(base / "good.jpg", (100, 100, 100))
    (base / "bad.jpg").write_bytes(b"not a jpeg")

    pool = scan.build_pool(
        base_dir=base,
        cache_dir=tmp_path / "cache",
        tile_px=24,
        demo_mode=False,
    )
    assert pool.lab.shape == (1, 3)
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_scan.py::test_scans_real_files -v`
Expected: FAIL — `NotImplementedError: Real-file scanning lands in Task 3`.

- [ ] **Step 3: Implement real-file scanning**

Replace the `NotImplementedError` branch in `src/scan.py` and add helpers. Full replacement for `src/scan.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_scan.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scan.py tests/test_scan.py
git commit -m "Add real-file scan with EXIF + crop + thumb cache"
```

---

## Task 4: Incremental scan (cache invalidation)

**Files:**
- Modify: `src/scan.py`
- Modify: `tests/test_scan.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_scan.py`:

```python
def test_second_scan_reuses_cache(tmp_path, monkeypatch):
    base = tmp_path / "photos"
    base.mkdir()
    _write_solid_jpg(base / "a.jpg", (100, 150, 200))
    _write_solid_jpg(base / "b.jpg", (200, 50, 100))

    cache = tmp_path / "cache"
    pool1 = scan.build_pool(base, cache, tile_px=24)

    calls: list[Path] = []
    original = scan._load_and_thumbnail

    def spy(path, tile_px):
        calls.append(path)
        return original(path, tile_px)

    monkeypatch.setattr(scan, "_load_and_thumbnail", spy)
    pool2 = scan.build_pool(base, cache, tile_px=24)

    assert calls == []
    np.testing.assert_allclose(pool1.lab, pool2.lab)


def test_changed_tile_px_invalidates_cache(tmp_path, monkeypatch):
    base = tmp_path / "photos"
    base.mkdir()
    _write_solid_jpg(base / "a.jpg", (100, 150, 200))

    cache = tmp_path / "cache"
    scan.build_pool(base, cache, tile_px=24)

    calls: list[Path] = []
    original = scan._load_and_thumbnail

    def spy(path, tile_px):
        calls.append(path)
        return original(path, tile_px)

    monkeypatch.setattr(scan, "_load_and_thumbnail", spy)
    scan.build_pool(base, cache, tile_px=32)

    assert len(calls) == 1
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_scan.py::test_second_scan_reuses_cache -v`
Expected: FAIL — the second scan currently reprocesses every file.

- [ ] **Step 3: Add cache persistence**

Replace the `_scan_directory` and `build_pool` functions in `src/scan.py` with caching-aware versions. Add these imports to the top:

```python
import json
```

Then replace both functions:

```python
CACHE_FILE = "pool.json"


def _load_cache(cache_dir: Path, tile_px: int) -> dict:
    p = cache_dir / CACHE_FILE
    if not p.exists():
        return {"tile_px": tile_px, "entries": {}}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {"tile_px": tile_px, "entries": {}}
    if data.get("tile_px") != tile_px:
        return {"tile_px": tile_px, "entries": {}}
    return data


def _save_cache(cache_dir: Path, data: dict) -> None:
    (cache_dir / CACHE_FILE).write_text(json.dumps(data))


def _scan_directory(
    base_dir: Path,
    cache_dir: Path,
    tile_px: int,
) -> TilePool:
    thumbs_dir = cache_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_dir, tile_px)
    entries: dict[str, dict] = cache["entries"]

    labs: list[np.ndarray] = []
    thumbs_paths: list[str] = []
    source_paths: list[str] = []

    files = list(_iter_candidate_files(base_dir))
    for path in tqdm(files, desc="scan", unit="img"):
        key = str(path)
        mtime = path.stat().st_mtime
        cached = entries.get(key)
        thumb_path = thumbs_dir / f"{_sha1(key)}.jpg"

        if cached and cached["mtime"] == mtime and Path(cached["thumb"]).exists():
            labs.append(np.asarray(cached["lab"], dtype=np.float32))
            thumbs_paths.append(cached["thumb"])
            source_paths.append(key)
            continue

        try:
            tile_rgb = _load_and_thumbnail(path, tile_px)
        except Exception as e:
            logger.warning("skip %s: %s", path, e)
            continue
        Image.fromarray(tile_rgb).save(thumb_path, quality=92)
        lab = _lab_mean(tile_rgb)
        entries[key] = {
            "mtime": mtime,
            "lab": lab.tolist(),
            "thumb": str(thumb_path),
        }
        labs.append(lab)
        thumbs_paths.append(str(thumb_path))
        source_paths.append(key)

    alive = set(str(p) for p in files)
    for dead in list(entries.keys()):
        if dead not in alive:
            entries.pop(dead)

    _save_cache(cache_dir, {"tile_px": tile_px, "entries": entries})

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
```

`build_pool` does not need changes — it already delegates to `_scan_directory`.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_scan.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scan.py tests/test_scan.py
git commit -m "Add incremental scan caching on (path, mtime, tile_px)"
```

---

## Task 5: Match

**Files:**
- Create: `src/match.py`
- Create: `tests/test_match.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np

from src import match


def test_match_picks_nearest_lab():
    pool_lab = np.array(
        [
            [50.0, 0.0, 0.0],      # gray
            [50.0, 60.0, 40.0],    # reddish
            [50.0, -40.0, 50.0],   # greenish
            [50.0, 20.0, -60.0],   # bluish
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [[50.0, 0.0, 0.0], [50.0, 60.0, 40.0]],
            [[50.0, -40.0, 50.0], [50.0, 20.0, -60.0]],
        ],
        dtype=np.float32,
    )
    idx = match.match_grid(target, pool_lab)
    assert idx.shape == (2, 2)
    assert idx.dtype == np.int32
    np.testing.assert_array_equal(idx, np.array([[0, 1], [2, 3]], dtype=np.int32))


def test_match_picks_closest_when_no_exact():
    pool_lab = np.array(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    target = np.array([[[30.0, 0.0, 0.0]]], dtype=np.float32)
    idx = match.match_grid(target, pool_lab)
    assert idx[0, 0] == 0


def test_match_raises_on_empty_pool():
    import pytest

    pool_lab = np.zeros((0, 3), dtype=np.float32)
    target = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
    with pytest.raises(ValueError):
        match.match_grid(target, pool_lab)
```

Save as `tests/test_match.py`.

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_match.py -v`
Expected: FAIL — `ModuleNotFoundError` or `AttributeError`.

- [ ] **Step 3: Implement**

Create `src/match.py`:

```python
from __future__ import annotations

import numpy as np


def match_grid(target_lab: np.ndarray, pool_lab: np.ndarray) -> np.ndarray:
    """For each (h, w) cell in target_lab, return the index into pool_lab
    whose LAB triplet has smallest Euclidean (ΔE76) distance.

    Args:
        target_lab: (H, W, 3) float32
        pool_lab:   (N, 3)    float32

    Returns:
        (H, W) int32 of indices into pool_lab.
    """
    if pool_lab.shape[0] == 0:
        raise ValueError("pool_lab is empty; cannot match")
    target_f = target_lab.astype(np.float32, copy=False)
    pool_f = pool_lab.astype(np.float32, copy=False)
    diff = target_f[:, :, None, :] - pool_f[None, None, :, :]
    dist2 = (diff * diff).sum(axis=-1)
    return dist2.argmin(axis=-1).astype(np.int32)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_match.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/match.py tests/test_match.py
git commit -m "Add LAB nearest-neighbor match_grid"
```

---

## Task 6: Render

**Files:**
- Create: `src/render.py`
- Create: `tests/test_render.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_render.py`:

```python
from pathlib import Path

import numpy as np
from PIL import Image

from src import render
from src.scan import TilePool


def _solid_thumb(path: Path, rgb: tuple[int, int, int], size: int) -> None:
    arr = np.full((size, size, 3), rgb, dtype=np.uint8)
    Image.fromarray(arr).save(path, quality=92)


def test_render_assembles_canvas(tmp_path):
    _solid_thumb(tmp_path / "red.jpg", (255, 0, 0), 8)
    _solid_thumb(tmp_path / "blue.jpg", (0, 0, 255), 8)

    pool = TilePool(
        lab=np.zeros((2, 3), dtype=np.float32),
        thumbs_paths=[str(tmp_path / "red.jpg"), str(tmp_path / "blue.jpg")],
        source_paths=["red", "blue"],
    )
    index_grid = np.array([[0, 1], [1, 0]], dtype=np.int32)

    out_path = tmp_path / "mosaic.png"
    img = render.render_mosaic(index_grid, pool, tile_px=8, output_path=out_path)

    assert out_path.exists()
    assert img.size == (16, 16)
    arr = np.asarray(img)
    # top-left quadrant red
    np.testing.assert_array_equal(arr[0:8, 0:8], np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8))
    # top-right quadrant blue
    np.testing.assert_array_equal(arr[0:8, 8:16], np.full((8, 8, 3), (0, 0, 255), dtype=np.uint8))
    # bottom-left blue
    np.testing.assert_array_equal(arr[8:16, 0:8], np.full((8, 8, 3), (0, 0, 255), dtype=np.uint8))
    # bottom-right red
    np.testing.assert_array_equal(arr[8:16, 8:16], np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8))


def test_render_returns_tile_usage(tmp_path):
    _solid_thumb(tmp_path / "a.jpg", (10, 10, 10), 4)
    _solid_thumb(tmp_path / "b.jpg", (200, 200, 200), 4)

    pool = TilePool(
        lab=np.zeros((2, 3), dtype=np.float32),
        thumbs_paths=[str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")],
        source_paths=["a", "b"],
    )
    index_grid = np.array([[0, 0, 1], [1, 1, 0]], dtype=np.int32)

    _, usage = render.render_mosaic_with_usage(
        index_grid, pool, tile_px=4, output_path=tmp_path / "m.png"
    )
    assert usage == {0: 3, 1: 3}
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_render.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/render.py`:

```python
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

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
) -> tuple[Image.Image, dict[int, int]]:
    h, w = index_grid.shape
    canvas = np.zeros((h * tile_px, w * tile_px, 3), dtype=np.uint8)
    cache: dict[str, np.ndarray] = {}
    usage: Counter[int] = Counter()

    for row in range(h):
        for col in range(w):
            idx = int(index_grid[row, col])
            thumb_path = pool.thumbs_paths[idx]
            tile = _load_thumb_cached(thumb_path, tile_px, cache)
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_render.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/render.py tests/test_render.py
git commit -m "Add render_mosaic + tile usage counter"
```

---

## Task 7: End-to-end smoke test

**Files:**
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write the canonical smoke test**

Create `tests/test_pipeline.py`:

```python
from pathlib import Path

import numpy as np
from PIL import Image

from src import match, render, scan


PALETTE = [
    (220, 20, 20),
    (20, 220, 20),
    (20, 20, 220),
    (220, 220, 20),
    (220, 20, 220),
    (20, 220, 220),
    (240, 240, 240),
    (20, 20, 20),
    (200, 120, 60),
    (60, 120, 200),
]


def _solid_jpg(path: Path, rgb, size: int = 96) -> None:
    arr = np.full((size, size, 3), rgb, dtype=np.uint8)
    Image.fromarray(arr).save(path, quality=95)


def test_end_to_end_pure_colors_match_exactly(tmp_path):
    base = tmp_path / "photos"
    base.mkdir()
    for i, rgb in enumerate(PALETTE):
        _solid_jpg(base / f"c{i}.jpg", rgb)

    pool = scan.build_pool(base, tmp_path / "cache", tile_px=24)

    grid_h, grid_w = 4, 10
    tile_px = 24
    target_rgb = np.zeros((grid_h * tile_px, grid_w * tile_px, 3), dtype=np.uint8)
    expected_color = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for col in range(grid_w):
        rgb = PALETTE[col]
        target_rgb[:, col * tile_px : (col + 1) * tile_px] = rgb
        expected_color[:, col] = rgb

    from skimage.color import rgb2lab

    target_lab = rgb2lab(target_rgb.astype(np.float32) / 255.0)
    target_lab_grid = target_lab.reshape(
        grid_h, tile_px, grid_w, tile_px, 3
    ).mean(axis=(1, 3)).astype(np.float32)

    idx = match.match_grid(target_lab_grid, pool.lab)
    img = render.render_mosaic(idx, pool, tile_px, tmp_path / "out.png")

    out = np.asarray(img)
    for row in range(grid_h):
        for col in range(grid_w):
            tile_rgb = out[
                row * tile_px : (row + 1) * tile_px,
                col * tile_px : (col + 1) * tile_px,
            ]
            mean = tile_rgb.reshape(-1, 3).mean(axis=0)
            diff = np.abs(mean - expected_color[row, col]).max()
            assert diff < 8, (
                f"cell ({row},{col}) off: got {mean}, expected {expected_color[row, col]}"
            )
```

- [ ] **Step 2: Run smoke test**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run: `pytest -v`
Expected: all tests (scan + match + render + pipeline) PASS, ~13 total.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "Add end-to-end smoke test (pure-color palette round-trip)"
```

---

## Task 8: Notebook

**Files:**
- Create: `scripts/build_notebook.py`
- Create: `bead_mosaic.ipynb` (generated)

- [ ] **Step 1: Create `scripts/build_notebook.py`**

```python
"""Build bead_mosaic.ipynb via nbformat. Run: python scripts/build_notebook.py"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bead_mosaic.ipynb"

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Bead Mosaic — Phase 1\n"
    "Zero-config demo: run all cells top-to-bottom.\n"
    "To use your own photos: set `CONFIG['BASE_DIR']` + `CONFIG['TARGET_PATH']` in Cell 2 and flip `DEMO_MODE` to `False`."
))

cells.append(nbf.v4.new_code_cell(
    "%pip install -q -r requirements.txt\n"
    "import sys\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, str(Path.cwd()))\n"
    "\n"
    "import numpy as np\n"
    "from PIL import Image, ImageOps\n"
    "from skimage import data\n"
    "from skimage.color import rgb2lab\n"
    "from IPython.display import display\n"
    "\n"
    "from src import match, render, scan"
))

cells.append(nbf.v4.new_code_cell(
    "# --- edit these two for real usage ---\n"
    "CONFIG = {\n"
    "    'BASE_DIR': Path.cwd() / 'my_photos',\n"
    "    'TARGET_PATH': None,  # None -> use skimage astronaut\n"
    "    'GRID_W': 120,\n"
    "    'GRID_H': 68,\n"
    "    'TILE_PX': 24,\n"
    "    'CACHE_DIR': Path.cwd() / '.cache',\n"
    "    'OUTPUT_PATH': Path.cwd() / 'output.png',\n"
    "    'DEMO_MODE': True,\n"
    "}\n"
    "CONFIG"
))

cells.append(nbf.v4.new_code_cell(
    "pool = scan.build_pool(\n"
    "    base_dir=CONFIG['BASE_DIR'],\n"
    "    cache_dir=CONFIG['CACHE_DIR'],\n"
    "    tile_px=CONFIG['TILE_PX'],\n"
    "    demo_mode=CONFIG['DEMO_MODE'],\n"
    ")\n"
    "print(f\"pool size: {pool.lab.shape[0]} tiles\")"
))

cells.append(nbf.v4.new_code_cell(
    "def load_target(path, grid_w, grid_h, tile_px):\n"
    "    if path is None:\n"
    "        arr = data.astronaut()\n"
    "        img = Image.fromarray(arr)\n"
    "    else:\n"
    "        img = Image.open(path)\n"
    "        img = ImageOps.exif_transpose(img).convert('RGB')\n"
    "    out_w, out_h = grid_w * tile_px, grid_h * tile_px\n"
    "    # letterbox\n"
    "    src_w, src_h = img.size\n"
    "    scale = min(out_w / src_w, out_h / src_h)\n"
    "    new_w, new_h = int(src_w * scale), int(src_h * scale)\n"
    "    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)\n"
    "    canvas = Image.new('RGB', (out_w, out_h), (0, 0, 0))\n"
    "    canvas.paste(resized, ((out_w - new_w) // 2, (out_h - new_h) // 2))\n"
    "    return np.asarray(canvas, dtype=np.uint8)\n"
    "\n"
    "target_rgb = load_target(\n"
    "    CONFIG['TARGET_PATH'], CONFIG['GRID_W'], CONFIG['GRID_H'], CONFIG['TILE_PX']\n"
    ")\n"
    "target_lab = rgb2lab(target_rgb.astype(np.float32) / 255.0)\n"
    "target_lab_grid = target_lab.reshape(\n"
    "    CONFIG['GRID_H'], CONFIG['TILE_PX'],\n"
    "    CONFIG['GRID_W'], CONFIG['TILE_PX'], 3\n"
    ").mean(axis=(1, 3)).astype(np.float32)\n"
    "target_lab_grid.shape"
))

cells.append(nbf.v4.new_code_cell(
    "idx = match.match_grid(target_lab_grid, pool.lab)\n"
    "idx.shape"
))

cells.append(nbf.v4.new_code_cell(
    "img, usage = render.render_mosaic_with_usage(\n"
    "    idx, pool, CONFIG['TILE_PX'], CONFIG['OUTPUT_PATH']\n"
    ")\n"
    "display(img)\n"
    "print(f\"\\noutput written to {CONFIG['OUTPUT_PATH']}\")\n"
    "print(f\"tiles used: {len(usage)} distinct / {sum(usage.values())} total placements\")"
))

nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
OUT.write_text(nbf.writes(nb))
print(f"wrote {OUT}")
```

- [ ] **Step 2: Build the notebook**

Run: `python scripts/build_notebook.py`
Expected: prints `wrote .../bead_mosaic.ipynb`; file exists.

- [ ] **Step 3: Execute the notebook end-to-end in demo mode**

Run: `jupyter nbconvert --to notebook --execute bead_mosaic.ipynb --output bead_mosaic.ipynb`
Expected: all cells run without error. `output.png` now exists in project root.

- [ ] **Step 4: Clear cell outputs for clean commit**

Run: `jupyter nbconvert --clear-output --inplace bead_mosaic.ipynb`

- [ ] **Step 5: Open the generated `output.png` and eyeball it**

Run: `open output.png`
Expected: visibly recognizable astronaut image composed of colored square tiles.

If it doesn't look recognizable, the pipeline has a bug — stop and debug before committing.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_notebook.py bead_mosaic.ipynb
git commit -m "Add notebook orchestrator + nbformat build script"
```

---

## Task 9: CHANGELOG (first entry)

**Files:**
- Create: `CHANGELOG.md`

- [ ] **Step 1: Create `CHANGELOG.md`**

```markdown
# CHANGELOG

> Format: YAML entries, written for future agents (see `memory/feedback_changelog_for_agents.md`).
> Compression trigger: 50 entries OR 6 months.

## 活跃条目

- date: 2026-04-17
  type: feat
  target: Phase 1 core pipeline (src/scan.py, src/match.py, src/render.py, bead_mosaic.ipynb)
  change: 从零搭建 photomosaic 最小闭环：扫描底图池 (LAB 均值 + mtime/tile_px 增量缓存) → numpy 暴力 ΔE76 最近邻匹配 → 贴图渲染到 2880×1632 PNG。Demo 模式用 skimage.data.astronaut + 合成 500 HSV 色块，零配置跑通。
  rationale: 按 2026-04-17 brainstorm 结论，Phase 1 只做核心闭环，不引入 λ/μ 惩罚、色调迁移、CLIP、DeepZoom、Gradio（依次分到 Phase 2-6 独立 spec/plan）。可迭代性优先于一次性完成。
  action: 9 个 TDD 任务完成；7 cell notebook 通过 nbformat 脚本生成；src/ 拆成 scan/match/render 三模块为 Phase 2-6 预埋扩展点。
  result: pytest 全绿（smoke test 对纯色 palette 做端到端 round-trip 验证每格 ΔRGB < 8）；demo mode notebook 产出 output.png 肉眼可辨 astronaut。
  validation: tests/test_scan.py (6) + test_match.py (3) + test_render.py (2) + test_pipeline.py (1) 全过；手动跑通 bead_mosaic.ipynb demo mode 并查看 output.png。
  status: stable
  spec: docs/superpowers/specs/2026-04-17-bead-mosaic-phase1-design.md
  plan: docs/superpowers/plans/2026-04-17-bead-mosaic-phase1.md
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "Add CHANGELOG with Phase 1 completion entry"
```

---

## Self-Review Checklist

**Spec coverage:**
- DoD item 1 (zero-config runs on fresh clone) → Task 8 Step 3
- DoD item 2 (demo mode with astronaut + 500 synth tiles) → Task 2, Task 8
- DoD item 3 (user edits 2 CONFIG entries to go real) → Task 8 Cell 2
- DoD item 4 (incremental scan) → Task 4
- DoD item 5 (smoke test passes) → Task 7
- All 9 key technical decisions from spec §2 → covered in code
- File structure from spec §4 → Tasks 1, 2, 5, 6, 7, 8 cover every file
- Module interfaces from spec §5 → exact signatures used
- 7-cell notebook from spec §5.4 → Task 8 generates all 7 cells
- Risks from spec §8 (EXIF, HEIC, long aspect ratio, cache invalidation, bad files) → handled in scan.py + Task 8 Cell 5 letterbox
- Phase 2-6 extension hooks from spec §9 → `render_mosaic_with_usage` returns usage dict (Phase 3), `TilePool` is dataclass (easy to extend with clip_emb for Phase 5), module split done

**Placeholder scan:** No TBDs, no "handle edge cases", no "similar to Task N". All code blocks complete.

**Type consistency:** `TilePool.lab` is `np.ndarray` of shape `(N, 3)` dtype `float32` throughout. `match_grid` returns `(H, W) int32`. `render_mosaic_with_usage` returns `(Image.Image, dict[int, int])`. All call sites match.

Plan is ready.
