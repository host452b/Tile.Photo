# Tile.Photo Mosaic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a local-only photomosaic "toy" ipynb that turns a target photo into a grid of tiles drawn from a user's photo folder, with interactive sliders, a self-deprecating stats report, and a zoomable DeepZoom HTML viewer.

**Architecture:** Single notebook `mosaic.ipynb` (8 cells) orchestrates a small pure-python helper package `mosaic/` (pool scan → kd-tree match with diversity/neighbor penalties → Reinhard tone transfer → report → DeepZoom export). All math runs in LAB color space on CPU. Zero deep-learning dependencies.

**Tech Stack:** Python 3.10+, numpy, pillow, scikit-image, scipy (cKDTree), tqdm, ipywidgets, matplotlib, deepzoom. No torch, no CLIP, no faiss, no Gradio.

**Spec reference:** `docs/superpowers/specs/2026-04-17-tile-photo-mosaic-design.md`

**CWD:** All paths below are relative to `Tile.Photo/manus/` unless otherwise noted.

---

## File Structure

Files produced by this plan (in order created):

| Path | Purpose |
|---|---|
| `.gitignore` | Exclude `cache/`, `out/`, pickles, notebook checkpoints |
| `requirements.txt` | 8 pip deps |
| `CHANGELOG.md` | Initial entry per user convention |
| `mosaic/__init__.py` | Package marker + version |
| `mosaic/pool.py` | `lab_mean()`, `scan_pool()`, `Tile` dataclass, cache hash |
| `mosaic/target.py` | `load_and_slice()` — fit-crop + slice target into grid cells |
| `mosaic/match.py` | `build_index()`, `match_grid()` — cKDTree + score rerank |
| `mosaic/render.py` | `reinhard_transfer()`, `compose()` — tone transfer + paste |
| `mosaic/report.py` | `text_report()`, `cold_wall()`, `usage_hist()` |
| `mosaic/dzi.py` | `export_dzi()` — pyramid + OpenSeadragon `index.html` |
| `fixtures/` | 20 tiny solid-color PNGs + 1 target for smoke test |
| `tests/test_pool.py` | LAB mean + pool scan tests |
| `tests/test_target.py` | Slice + center-crop tests |
| `tests/test_match.py` | Score math + grid match tests |
| `tests/test_render.py` | Reinhard math + compose tests |
| `tests/test_report.py` | Report format tests |
| `tests/test_dzi.py` | DeepZoom export smoke test |
| `tests/test_smoke.py` | Full pipeline integration test |
| `tests/conftest.py` | Fixture helpers |
| `mosaic.ipynb` | 8-cell orchestration notebook |

---

## Task 1: Scaffold repo

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `CHANGELOG.md`
- Create: `mosaic/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write `.gitignore`**

```
__pycache__/
*.pyc
.ipynb_checkpoints/
.pytest_cache/
cache/
out/
*.pickle
*.pkl
.DS_Store
.venv/
venv/
```

- [ ] **Step 2: Write `requirements.txt`**

```
numpy>=1.26
pillow>=10.0
scikit-image>=0.22
scipy>=1.11
tqdm>=4.66
ipywidgets>=8.1
matplotlib>=3.8
deepzoom>=0.5
pytest>=8.0
```

- [ ] **Step 3: Write `CHANGELOG.md`**

```markdown
# CHANGELOG

> Maintained per agent-oriented convention: verbose, preserve try-failed chains, ISO dates, compress on 50 entries or 6 months.

## 活跃条目

- date: 2026-04-17
  type: feat
  target: entire project
  change: Scaffold Tier B photomosaic toy (mosaic/ package + mosaic.ipynb) per spec 2026-04-17-tile-photo-mosaic-design.md
  rationale: User wants a local ipynb that builds a photo-of-photos mosaic, with interactive sliders, self-deprecating stats, and DeepZoom HTML export. Vetoed torch/CLIP/Gradio.
  action: Create mosaic/ helper package (pool/target/match/render/report/dzi), mosaic.ipynb with 8 cells, pytest suite with fixtures
  result: Scaffolding in place; awaiting real pool + target to produce first real mosaic
  validation: pytest tests/ passes; smoke test produces a 20×20 px mosaic from 20-fixture pool
  status: experimental
```

- [ ] **Step 4: Write `mosaic/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 5: Write `tests/__init__.py`**

Empty file.

- [ ] **Step 6: Commit**

```bash
git add .gitignore requirements.txt CHANGELOG.md mosaic/__init__.py tests/__init__.py
git commit -m "feat: scaffold tile.photo mosaic project"
```

---

## Task 2: `mosaic/pool.py` — LAB mean helper

**Files:**
- Create: `mosaic/pool.py`
- Create: `tests/test_pool.py`
- Create: `tests/conftest.py`
- Create: `fixtures/red.png` (8×8 pure red PNG)
- Create: `fixtures/blue.png` (8×8 pure blue PNG)

- [ ] **Step 1: Write `tests/conftest.py`**

```python
from pathlib import Path
import numpy as np
from PIL import Image
import pytest


FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES


def _write_solid(path: Path, rgb: tuple[int, int, int], size: int = 8):
    arr = np.full((size, size, 3), rgb, dtype=np.uint8)
    Image.fromarray(arr).save(path)


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures():
    FIXTURES.mkdir(exist_ok=True)
    if not (FIXTURES / "red.png").exists():
        _write_solid(FIXTURES / "red.png", (255, 0, 0))
    if not (FIXTURES / "blue.png").exists():
        _write_solid(FIXTURES / "blue.png", (0, 0, 255))
```

- [ ] **Step 2: Write failing test `tests/test_pool.py`**

```python
import numpy as np
from PIL import Image

from mosaic.pool import lab_mean


def test_lab_mean_pure_red(fixtures_dir):
    img = np.array(Image.open(fixtures_dir / "red.png").convert("RGB"))
    lab = lab_mean(img)
    # Pure red in LAB: L ≈ 53, a ≈ 80, b ≈ 67
    assert lab.shape == (3,)
    assert 50 < lab[0] < 56
    assert 75 < lab[1] < 85
    assert 60 < lab[2] < 72


def test_lab_mean_pure_blue(fixtures_dir):
    img = np.array(Image.open(fixtures_dir / "blue.png").convert("RGB"))
    lab = lab_mean(img)
    # Pure blue in LAB: L ≈ 32, a ≈ 79, b ≈ -108
    assert lab.shape == (3,)
    assert 28 < lab[0] < 36
    assert -115 < lab[2] < -100
```

- [ ] **Step 3: Run to verify failure**

```bash
cd Tile.Photo/manus
pytest tests/test_pool.py::test_lab_mean_pure_red -v
```

Expected: `ImportError: cannot import name 'lab_mean'` (module doesn't exist yet).

- [ ] **Step 4: Implement `mosaic/pool.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.color import rgb2lab


def lab_mean(rgb_uint8: np.ndarray) -> np.ndarray:
    """Return the mean LAB color of an (H,W,3) uint8 RGB image as (3,) float32."""
    lab = rgb2lab(rgb_uint8.astype(np.float32) / 255.0)
    return lab.reshape(-1, 3).mean(axis=0).astype(np.float32)


@dataclass
class Tile:
    path: Path
    lab: np.ndarray    # shape (3,), float32
    thumb: np.ndarray  # shape (thumb_px, thumb_px, 3), uint8
```

- [ ] **Step 5: Run to verify pass**

```bash
pytest tests/test_pool.py -v
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

```bash
git add mosaic/pool.py tests/test_pool.py tests/conftest.py
git commit -m "feat(pool): add lab_mean helper and Tile dataclass"
```

---

## Task 3: `mosaic/pool.py` — pool scan with cache

**Files:**
- Modify: `mosaic/pool.py`
- Modify: `tests/test_pool.py`

- [ ] **Step 1: Extend `tests/test_pool.py`**

Append:

```python
import pickle
from mosaic.pool import scan_pool


def test_scan_pool_reads_images(tmp_path, fixtures_dir):
    # copy two fixtures into a tmp pool dir
    pool = tmp_path / "pool"
    pool.mkdir()
    for name in ("red.png", "blue.png"):
        (pool / name).write_bytes((fixtures_dir / name).read_bytes())

    cache_dir = tmp_path / "cache"
    tiles = scan_pool(pool, cache_dir, thumb_px=8)

    assert len(tiles) == 2
    paths = {t.path.name for t in tiles}
    assert paths == {"red.png", "blue.png"}
    for t in tiles:
        assert t.thumb.shape == (8, 8, 3)
        assert t.lab.shape == (3,)


def test_scan_pool_uses_cache(tmp_path, fixtures_dir):
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "red.png").write_bytes((fixtures_dir / "red.png").read_bytes())
    cache_dir = tmp_path / "cache"

    scan_pool(pool, cache_dir, thumb_px=8)
    cache_files = list(cache_dir.glob("*.pkl"))
    assert len(cache_files) == 1

    # second call should be fast (cache hit); easiest check: modify cache and see it re-emerges
    with cache_files[0].open("rb") as f:
        data = pickle.load(f)
    assert len(data["tiles"]) == 1


def test_scan_pool_invalidates_on_new_file(tmp_path, fixtures_dir):
    pool = tmp_path / "pool"
    pool.mkdir()
    cache_dir = tmp_path / "cache"
    (pool / "red.png").write_bytes((fixtures_dir / "red.png").read_bytes())
    first = scan_pool(pool, cache_dir, thumb_px=8)
    assert len(first) == 1

    (pool / "blue.png").write_bytes((fixtures_dir / "blue.png").read_bytes())
    second = scan_pool(pool, cache_dir, thumb_px=8)
    assert len(second) == 2
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_pool.py -v
```

Expected: `ImportError: cannot import name 'scan_pool'`.

- [ ] **Step 3: Implement `scan_pool` in `mosaic/pool.py`**

Append to `mosaic/pool.py`:

```python
import hashlib
import pickle

from PIL import Image
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


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
    """Return (thumb_rgb uint8, lab_mean float32)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        # center-crop to square then resize
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        im = im.resize((thumb_px, thumb_px), Image.LANCZOS)
        thumb = np.array(im, dtype=np.uint8)
    return thumb, lab_mean(thumb)


def scan_pool(pool_dir: Path, cache_dir: Path, thumb_px: int = 32) -> list[Tile]:
    """Scan pool_dir recursively, return list of Tile. Caches to cache_dir/pool_<hash>.pkl."""
    pool_dir = Path(pool_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _pool_hash(pool_dir, thumb_px)
    cache_path = cache_dir / f"pool_{key}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            data = pickle.load(f)
        return data["tiles"]

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
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_pool.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add mosaic/pool.py tests/test_pool.py
git commit -m "feat(pool): add scan_pool with content-hash cache"
```

---

## Task 4: `mosaic/target.py` — load and slice target

**Files:**
- Create: `mosaic/target.py`
- Create: `tests/test_target.py`
- Modify: `tests/conftest.py` (add target fixture)

- [ ] **Step 1: Extend `tests/conftest.py`**

Append inside `ensure_fixtures()` (before the `return`):

```python
    if not (FIXTURES / "target.png").exists():
        # 160×90 gradient: left half red, right half blue
        arr = np.zeros((90, 160, 3), dtype=np.uint8)
        arr[:, :80] = (255, 0, 0)
        arr[:, 80:] = (0, 0, 255)
        Image.fromarray(arr).save(FIXTURES / "target.png")
```

- [ ] **Step 2: Write failing test `tests/test_target.py`**

```python
import numpy as np
from mosaic.target import load_and_slice


def test_load_and_slice_shape(fixtures_dir):
    grid = load_and_slice(fixtures_dir / "target.png", grid_w=10, grid_h=5)
    assert grid.lab_means.shape == (5, 10, 3)
    assert grid.canvas.shape[2] == 3


def test_load_and_slice_left_is_red_right_is_blue(fixtures_dir):
    grid = load_and_slice(fixtures_dir / "target.png", grid_w=10, grid_h=5)
    left = grid.lab_means[:, :5].mean(axis=(0, 1))
    right = grid.lab_means[:, 5:].mean(axis=(0, 1))
    # red has positive a*, blue has very negative b*
    assert left[1] > 50        # red a* ≈ +80
    assert right[2] < -50      # blue b* ≈ -108
```

- [ ] **Step 3: Run to verify failure**

```bash
pytest tests/test_target.py -v
```

Expected: `ImportError: cannot import name 'load_and_slice'`.

- [ ] **Step 4: Implement `mosaic/target.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from mosaic.pool import lab_mean


@dataclass
class TargetGrid:
    canvas: np.ndarray      # (H, W, 3) uint8, center-cropped to grid aspect
    lab_means: np.ndarray   # (grid_h, grid_w, 3) float32


def _center_crop_to_aspect(img: Image.Image, aspect: float) -> Image.Image:
    w, h = img.size
    cur = w / h
    if cur > aspect:
        new_w = int(h * aspect)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    new_h = int(w / aspect)
    top = (h - new_h) // 2
    return img.crop((0, top, w, top + new_h))


def load_and_slice(path: Path, grid_w: int, grid_h: int) -> TargetGrid:
    """Load target, center-crop to grid_w:grid_h aspect, compute per-cell LAB means."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = _center_crop_to_aspect(im, grid_w / grid_h)
        # resize so each cell is at least 1px but not absurdly large
        scale_w = grid_w * 8
        scale_h = grid_h * 8
        im = im.resize((scale_w, scale_h), Image.LANCZOS)
        canvas = np.array(im, dtype=np.uint8)

    cell_w = scale_w // grid_w
    cell_h = scale_h // grid_h
    lab_means = np.zeros((grid_h, grid_w, 3), dtype=np.float32)
    for r in range(grid_h):
        for c in range(grid_w):
            patch = canvas[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            lab_means[r, c] = lab_mean(patch)
    return TargetGrid(canvas=canvas, lab_means=lab_means)
```

- [ ] **Step 5: Run to verify pass**

```bash
pytest tests/test_target.py -v
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

```bash
git add mosaic/target.py tests/test_target.py tests/conftest.py fixtures/target.png
git commit -m "feat(target): load + center-crop + slice target into grid cells"
```

---

## Task 5: `mosaic/match.py` — cKDTree + top-K candidates

**Files:**
- Create: `mosaic/match.py`
- Create: `tests/test_match.py`

- [ ] **Step 1: Write failing test `tests/test_match.py`**

```python
import numpy as np
from mosaic.match import build_index, topk_candidates


def _fake_tiles_lab():
    # 5 fake tiles at 5 distinct LAB points
    return np.array(
        [
            [50.0, 0.0, 0.0],     # gray
            [53.0, 80.0, 67.0],   # red
            [32.0, 79.0, -108.0], # blue
            [88.0, -79.0, 81.0],  # green
            [97.0, 0.0, 0.0],     # white
        ],
        dtype=np.float32,
    )


def test_topk_returns_nearest_first():
    labs = _fake_tiles_lab()
    idx = build_index(labs)
    query = np.array([53.0, 80.0, 67.0], dtype=np.float32)  # red
    cand = topk_candidates(idx, query, k=3)
    assert cand[0] == 1  # red tile is index 1
    assert len(cand) == 3


def test_topk_k_clamps_to_pool_size():
    labs = _fake_tiles_lab()
    idx = build_index(labs)
    cand = topk_candidates(idx, labs[0], k=100)
    assert len(cand) == 5
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_match.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement index helpers in `mosaic/match.py`**

```python
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def build_index(tile_labs: np.ndarray) -> cKDTree:
    return cKDTree(tile_labs.astype(np.float32))


def topk_candidates(index: cKDTree, query_lab: np.ndarray, k: int) -> np.ndarray:
    k = min(k, index.n)
    _, idx = index.query(query_lab, k=k)
    if np.ndim(idx) == 0:
        idx = np.array([idx])
    return np.asarray(idx, dtype=np.int64)
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_match.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add mosaic/match.py tests/test_match.py
git commit -m "feat(match): cKDTree index + topk_candidates"
```

---

## Task 6: `mosaic/match.py` — scored grid matching

**Files:**
- Modify: `mosaic/match.py`
- Modify: `tests/test_match.py`

- [ ] **Step 1: Extend `tests/test_match.py`**

Append:

```python
from mosaic.match import match_grid


def test_match_grid_without_penalties_picks_nearest():
    # 4 tiles, 2x2 target grid: each cell should pick its exact LAB twin
    labs = np.array(
        [
            [50.0, 0.0, 0.0],
            [53.0, 80.0, 67.0],
            [32.0, 79.0, -108.0],
            [88.0, -79.0, 81.0],
        ],
        dtype=np.float32,
    )
    cells = labs.reshape(2, 2, 3)  # cell (r,c) matches tile index r*2+c
    idx = build_index(labs)
    grid = match_grid(idx, labs, cells, k=4, lambda_=0.0, mu=0.0)
    assert grid.shape == (2, 2)
    assert grid[0, 0] == 0
    assert grid[0, 1] == 1
    assert grid[1, 0] == 2
    assert grid[1, 1] == 3


def test_match_grid_diversity_forces_spread():
    # Two tiles, identical to all target cells. With lambda > 0 the second
    # pick should be the other tile even though both have distance 0.
    labs = np.array(
        [[50.0, 0.0, 0.0], [50.01, 0.0, 0.0]],
        dtype=np.float32,
    )
    cells = np.array([[[50.0, 0.0, 0.0], [50.0, 0.0, 0.0]]], dtype=np.float32)  # 1x2 grid
    idx = build_index(labs)
    grid = match_grid(idx, labs, cells, k=2, lambda_=10.0, mu=0.0)
    assert set(grid.ravel()) == {0, 1}


def test_match_grid_shape_matches_input():
    labs = np.random.default_rng(0).uniform(-50, 50, size=(30, 3)).astype(np.float32)
    cells = np.random.default_rng(1).uniform(-50, 50, size=(4, 6, 3)).astype(np.float32)
    idx = build_index(labs)
    grid = match_grid(idx, labs, cells, k=5, lambda_=1.0, mu=0.5)
    assert grid.shape == (4, 6)
    assert grid.dtype == np.int64
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_match.py::test_match_grid_without_penalties_picks_nearest -v
```

Expected: `ImportError: cannot import name 'match_grid'`.

- [ ] **Step 3: Implement `match_grid`**

Append to `mosaic/match.py`:

```python
def _neighbor_sim(tile_lab: np.ndarray, neighbor_labs: list[np.ndarray]) -> float:
    """Mean (1 - cosine similarity) between tile_lab and a list of neighbor tile LABs."""
    if not neighbor_labs:
        return 0.0
    t = tile_lab / (np.linalg.norm(tile_lab) + 1e-9)
    acc = 0.0
    for n in neighbor_labs:
        nn = n / (np.linalg.norm(n) + 1e-9)
        acc += 1.0 - float(np.dot(t, nn))
    return acc / len(neighbor_labs)


def match_grid(
    index: cKDTree,
    tile_labs: np.ndarray,
    cell_labs: np.ndarray,
    k: int = 20,
    lambda_: float = 2.0,
    mu: float = 0.5,
    log_every: int = 0,
) -> np.ndarray:
    """Row-major greedy match. Returns (grid_h, grid_w) int64 array of tile indices."""
    grid_h, grid_w, _ = cell_labs.shape
    choices = np.full((grid_h, grid_w), -1, dtype=np.int64)
    uses = np.zeros(tile_labs.shape[0], dtype=np.int64)

    total = grid_h * grid_w
    step = 0
    for r in range(grid_h):
        for c in range(grid_w):
            step += 1
            cell = cell_labs[r, c]
            cand = topk_candidates(index, cell, k=k)
            neighbor_labs: list[np.ndarray] = []
            if r > 0 and choices[r - 1, c] >= 0:
                neighbor_labs.append(tile_labs[choices[r - 1, c]])
            if c > 0 and choices[r, c - 1] >= 0:
                neighbor_labs.append(tile_labs[choices[r, c - 1]])
            if r > 0 and c > 0 and choices[r - 1, c - 1] >= 0:
                neighbor_labs.append(tile_labs[choices[r - 1, c - 1]])
            if r > 0 and c + 1 < grid_w and choices[r - 1, c + 1] >= 0:
                neighbor_labs.append(tile_labs[choices[r - 1, c + 1]])

            best_i = int(cand[0])
            best_score = float("inf")
            for ti in cand:
                dist = float(np.linalg.norm(tile_labs[ti] - cell))
                diversity = lambda_ * np.log1p(uses[ti])
                neighbor = mu * _neighbor_sim(tile_labs[ti], neighbor_labs)
                score = dist + diversity + neighbor
                if score < best_score:
                    best_score = score
                    best_i = int(ti)
            choices[r, c] = best_i
            uses[best_i] += 1
            if log_every and step % log_every == 0:
                print(f"[{step}/{total}] cell ({r},{c}) -> tile #{best_i} (uses={uses[best_i]})")
    return choices
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_match.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add mosaic/match.py tests/test_match.py
git commit -m "feat(match): scored grid match with diversity + neighbor penalties"
```

---

## Task 7: `mosaic/render.py` — Reinhard tone transfer

**Files:**
- Create: `mosaic/render.py`
- Create: `tests/test_render.py`

- [ ] **Step 1: Write failing test `tests/test_render.py`**

```python
import numpy as np
from mosaic.render import reinhard_transfer


def test_reinhard_tau_zero_is_identity():
    rng = np.random.default_rng(0)
    tile_lab = rng.uniform(-50, 50, size=(8, 8, 3)).astype(np.float32)
    out = reinhard_transfer(
        tile_lab,
        target_mean=np.array([90.0, 10.0, 10.0], dtype=np.float32),
        target_std=np.array([5.0, 5.0, 5.0], dtype=np.float32),
        tau=0.0,
    )
    assert np.allclose(out, tile_lab, atol=1e-5)


def test_reinhard_tau_one_matches_target_mean():
    rng = np.random.default_rng(0)
    tile_lab = rng.uniform(-50, 50, size=(8, 8, 3)).astype(np.float32)
    target_mean = np.array([90.0, 10.0, 10.0], dtype=np.float32)
    target_std = tile_lab.reshape(-1, 3).std(axis=0).astype(np.float32)
    out = reinhard_transfer(tile_lab, target_mean=target_mean, target_std=target_std, tau=1.0)
    assert np.allclose(out.reshape(-1, 3).mean(axis=0), target_mean, atol=1e-3)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_render.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `reinhard_transfer`**

```python
from __future__ import annotations

import numpy as np
from skimage.color import lab2rgb, rgb2lab


EPS = 1e-6


def reinhard_transfer(
    tile_lab: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Shift tile_lab toward target distribution by strength tau in [0,1].

    tau=0: returns tile_lab unchanged.
    tau=1: result has mean == target_mean (and std == target_std in each channel).
    """
    flat = tile_lab.reshape(-1, 3)
    src_mean = flat.mean(axis=0).astype(np.float32)
    src_std = flat.std(axis=0).astype(np.float32)
    scale = target_std / (src_std + EPS)
    # tau-interpolated scale and offset
    eff_scale = (1.0 - tau) + tau * scale
    shifted = (flat - src_mean) * eff_scale + src_mean + tau * (target_mean - src_mean)
    return shifted.reshape(tile_lab.shape).astype(np.float32)
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_render.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add mosaic/render.py tests/test_render.py
git commit -m "feat(render): Reinhard LAB tone transfer with tau blend"
```

---

## Task 8: `mosaic/render.py` — compose mosaic

**Files:**
- Modify: `mosaic/render.py`
- Modify: `tests/test_render.py`

- [ ] **Step 1: Extend `tests/test_render.py`**

Append:

```python
from mosaic.pool import Tile
from mosaic.render import compose


def _solid_tile(path_name: str, rgb: tuple[int, int, int], thumb_px: int = 4) -> Tile:
    from pathlib import Path
    from mosaic.pool import lab_mean

    thumb = np.full((thumb_px, thumb_px, 3), rgb, dtype=np.uint8)
    return Tile(path=Path(path_name), lab=lab_mean(thumb), thumb=thumb)


def test_compose_output_shape():
    tiles = [_solid_tile("a.png", (255, 0, 0)), _solid_tile("b.png", (0, 0, 255))]
    grid = np.array([[0, 1], [1, 0]], dtype=np.int64)
    cell_lab_means = np.zeros((2, 2, 3), dtype=np.float32)
    out = compose(tiles, grid, cell_lab_means, tile_px=4, tau=0.0)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8


def test_compose_places_correct_color_at_tau_zero():
    tiles = [_solid_tile("a.png", (255, 0, 0)), _solid_tile("b.png", (0, 0, 255))]
    grid = np.array([[0, 1]], dtype=np.int64)
    cell_lab_means = np.zeros((1, 2, 3), dtype=np.float32)
    out = compose(tiles, grid, cell_lab_means, tile_px=4, tau=0.0)
    # left 4 px should be red; right 4 px should be blue
    assert out[0, 0, 0] > 200 and out[0, 0, 2] < 50
    assert out[0, 7, 2] > 200 and out[0, 7, 0] < 50
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_render.py -v
```

Expected: `ImportError: cannot import name 'compose'`.

- [ ] **Step 3: Implement `compose`**

Append to `mosaic/render.py`:

```python
from PIL import Image


def compose(
    tiles: list,
    grid: np.ndarray,
    cell_lab_means: np.ndarray,
    tile_px: int,
    tau: float,
) -> np.ndarray:
    """Compose the final mosaic as a (grid_h*tile_px, grid_w*tile_px, 3) uint8 ndarray."""
    grid_h, grid_w = grid.shape
    out = np.zeros((grid_h * tile_px, grid_w * tile_px, 3), dtype=np.uint8)

    for r in range(grid_h):
        for c in range(grid_w):
            t = tiles[int(grid[r, c])]
            thumb = t.thumb
            if thumb.shape[0] != tile_px or thumb.shape[1] != tile_px:
                thumb = np.array(
                    Image.fromarray(thumb).resize((tile_px, tile_px), Image.LANCZOS),
                    dtype=np.uint8,
                )
            if tau > 0:
                thumb_lab = rgb2lab(thumb.astype(np.float32) / 255.0).astype(np.float32)
                target_mean = cell_lab_means[r, c]
                target_std = thumb_lab.reshape(-1, 3).std(axis=0).astype(np.float32)
                shifted = reinhard_transfer(thumb_lab, target_mean, target_std, tau)
                rgb = np.clip(lab2rgb(shifted) * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb = thumb
            out[r * tile_px : (r + 1) * tile_px, c * tile_px : (c + 1) * tile_px] = rgb
    return out
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_render.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add mosaic/render.py tests/test_render.py
git commit -m "feat(render): compose mosaic with per-tile Reinhard tone transfer"
```

---

## Task 9: `mosaic/report.py` — stats, histogram, cold wall

**Files:**
- Create: `mosaic/report.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Write failing test `tests/test_report.py`**

```python
from pathlib import Path

import numpy as np
from PIL import Image

from mosaic.pool import Tile, lab_mean
from mosaic.report import cold_wall, describe_lab, text_report, usage_hist_figure


def _tile(name, rgb, px=4):
    thumb = np.full((px, px, 3), rgb, dtype=np.uint8)
    return Tile(path=Path(name), lab=lab_mean(thumb), thumb=thumb)


def test_text_report_contains_totals():
    tiles = [_tile(f"{i}.png", (i * 10 % 255, 0, 0)) for i in range(5)]
    uses = np.array([10, 0, 3, 0, 7], dtype=np.int64)
    report = text_report(tiles, uses, grid_shape=(4, 5), params={"lambda_": 2.0, "mu": 0.5, "tau": 0.5})
    assert "Tiles placed: 20" in report
    assert "Unique tiles used: 3" in report
    assert "0.png" in report  # top used
    assert "λ=2.0" in report


def test_cold_wall_returns_image_with_correct_count(tmp_path):
    tiles = [_tile(f"{i}.png", (i * 20 % 255, 0, 0), px=4) for i in range(6)]
    uses = np.array([5, 0, 0, 0, 2, 0], dtype=np.int64)
    wall = cold_wall(tiles, uses, n=4, thumb_px=4, cols=2)
    assert wall.shape == (8, 8, 3)
    assert wall.dtype == np.uint8


def test_usage_hist_figure_runs_without_error():
    tiles = [_tile(f"{i}.png", (0, 0, 0)) for i in range(3)]
    uses = np.array([1, 2, 3], dtype=np.int64)
    fig = usage_hist_figure(tiles, uses)
    assert fig is not None


def test_describe_lab_covers_known_colors():
    # sky blue ≈ L~70, a~-5, b~-30
    assert describe_lab(np.array([70.0, -5.0, -30.0])) == "sky blue"
    # skin tone ≈ L~65, a~15, b~25
    assert describe_lab(np.array([65.0, 15.0, 25.0])) == "skin tone"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_report.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `mosaic/report.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def describe_lab(lab: np.ndarray) -> str:
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    if b < -20 and -15 < a < 15:
        return "sky blue"
    if a > 10 and b > 15 and L > 40:
        return "skin tone"
    if a < -15 and b > 10:
        return "foliage green"
    if a > 10 and b > -5 and b < 20 and L < 55:
        return "warm brown"
    if L < 25:
        return "shadow"
    if L > 85:
        return "highlight"
    return "neutral gray"


def text_report(
    tiles: list,
    uses: np.ndarray,
    grid_shape: tuple[int, int],
    params: dict,
) -> str:
    total_cells = grid_shape[0] * grid_shape[1]
    unique_used = int((uses > 0).sum())
    order = np.argsort(-uses)
    lines: list[str] = []
    lines.append("Tile.Photo — mosaic report")
    lines.append("")
    lines.append(f"Pool size: {len(tiles)}")
    lines.append(f"Tiles placed: {total_cells} ({grid_shape[1]} × {grid_shape[0]})")
    lines.append(f"Unique tiles used: {unique_used} ({unique_used / max(len(tiles), 1):.0%} of pool)")
    lines.append("")
    lines.append("Most used (top 5):")
    for i in order[:5]:
        if uses[i] == 0:
            break
        reason = describe_lab(tiles[i].lab)
        lines.append(f"  {tiles[i].path.name} — {int(uses[i])} uses  (mostly: {reason})")
    lines.append("")
    cold = [tiles[i].path.name for i in order[::-1] if uses[i] == 0][:20]
    lines.append(f"Never used ({(uses == 0).sum()} tiles). Top 20 cold: {', '.join(cold)}")
    lines.append("")
    lines.append(
        "Parameters: "
        f"λ={params.get('lambda_', '?')}, μ={params.get('mu', '?')}, τ={params.get('tau', '?')}"
    )
    return "\n".join(lines)


def cold_wall(
    tiles: list,
    uses: np.ndarray,
    n: int = 20,
    thumb_px: int = 32,
    cols: int = 5,
) -> np.ndarray:
    cold_idx = [i for i in np.argsort(uses) if uses[i] == 0][:n]
    if not cold_idx:
        return np.zeros((thumb_px, thumb_px, 3), dtype=np.uint8)
    rows = (len(cold_idx) + cols - 1) // cols
    wall = np.zeros((rows * thumb_px, cols * thumb_px, 3), dtype=np.uint8)
    for k, ti in enumerate(cold_idx):
        r, c = divmod(k, cols)
        thumb = tiles[ti].thumb
        if thumb.shape[0] != thumb_px:
            thumb = np.array(
                Image.fromarray(thumb).resize((thumb_px, thumb_px), Image.LANCZOS),
                dtype=np.uint8,
            )
        wall[r * thumb_px : (r + 1) * thumb_px, c * thumb_px : (c + 1) * thumb_px] = thumb
    return wall


def usage_hist_figure(tiles: list, uses: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(uses, bins=min(30, max(int(uses.max()) + 1, 2)))
    ax.set_xlabel("times used")
    ax.set_ylabel("tile count")
    ax.set_title(f"Usage distribution across {len(tiles)} tiles")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_report.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add mosaic/report.py tests/test_report.py
git commit -m "feat(report): text report, cold wall, usage histogram"
```

---

## Task 10: `mosaic/dzi.py` — DeepZoom export + OpenSeadragon html

**Files:**
- Create: `mosaic/dzi.py`
- Create: `tests/test_dzi.py`

- [ ] **Step 1: Write failing test `tests/test_dzi.py`**

```python
import numpy as np
from PIL import Image

from mosaic.dzi import export_dzi


def test_export_dzi_produces_html_and_dzi(tmp_path):
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))
    out_dir = tmp_path / "dzi"
    html = export_dzi(img, out_dir)
    assert html.exists()
    assert (out_dir / "image.dzi").exists()
    assert (out_dir / "image_files").is_dir()
    # index.html references the dzi file
    assert "image.dzi" in html.read_text()
    assert "openseadragon" in html.read_text().lower()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_dzi.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `mosaic/dzi.py`**

```python
from __future__ import annotations

from pathlib import Path

import deepzoom
from PIL import Image


_HTML = """<!doctype html>
<meta charset="utf-8">
<title>Tile.Photo mosaic</title>
<style>html,body{{margin:0;padding:0;height:100%;background:#111;color:#ccc;font:13px sans-serif}}#v{{width:100%;height:100vh}}</style>
<div id="v"></div>
<script src="https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/openseadragon.min.js"></script>
<script>
OpenSeadragon({{
  id: "v",
  prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/images/",
  tileSources: "{dzi}"
}});
</script>
"""


def export_dzi(image: Image.Image, out_dir: Path) -> Path:
    """Write a DeepZoom pyramid (image.dzi + image_files/) + index.html into out_dir.

    Returns the path to index.html.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_png = out_dir / "_src.png"
    image.save(src_png)
    creator = deepzoom.ImageCreator(
        tile_size=254,
        tile_overlap=1,
        tile_format="jpg",
        image_quality=0.9,
        resize_filter="lanczos",
    )
    creator.create(str(src_png), str(out_dir / "image.dzi"))
    src_png.unlink()

    html = out_dir / "index.html"
    html.write_text(_HTML.format(dzi="image.dzi"), encoding="utf-8")
    return html
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_dzi.py -v
```

Expected: test passes.

- [ ] **Step 5: Commit**

```bash
git add mosaic/dzi.py tests/test_dzi.py
git commit -m "feat(dzi): DeepZoom export + OpenSeadragon html"
```

---

## Task 11: End-to-end smoke test

**Files:**
- Create: `tests/test_smoke.py`
- Modify: `tests/conftest.py` (add smoke fixture pool)

- [ ] **Step 1: Extend `tests/conftest.py`**

Append inside `ensure_fixtures()`:

```python
    smoke = FIXTURES / "smoke_pool"
    smoke.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(20):
        rgb = tuple(int(x) for x in rng.integers(0, 255, size=3))
        _write_solid(smoke / f"tile_{i:02d}.png", rgb, size=8)
```

- [ ] **Step 2: Write `tests/test_smoke.py`**

```python
from pathlib import Path

import numpy as np
from PIL import Image

from mosaic.dzi import export_dzi
from mosaic.match import build_index, match_grid
from mosaic.pool import scan_pool
from mosaic.render import compose
from mosaic.report import cold_wall, text_report
from mosaic.target import load_and_slice


def test_full_pipeline(tmp_path, fixtures_dir):
    pool_dir = fixtures_dir / "smoke_pool"
    target = fixtures_dir / "target.png"

    tiles = scan_pool(pool_dir, tmp_path / "cache", thumb_px=8)
    assert len(tiles) == 20

    grid = load_and_slice(target, grid_w=5, grid_h=3)
    assert grid.lab_means.shape == (3, 5, 3)

    tile_labs = np.stack([t.lab for t in tiles])
    idx = build_index(tile_labs)
    choices = match_grid(idx, tile_labs, grid.lab_means, k=5, lambda_=1.0, mu=0.3)
    assert choices.shape == (3, 5)

    mosaic = compose(tiles, choices, grid.lab_means, tile_px=4, tau=0.5)
    assert mosaic.shape == (12, 20, 3)
    assert mosaic.dtype == np.uint8
    assert mosaic.sum() > 0

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    Image.fromarray(mosaic).save(out_dir / "mosaic.png")

    uses = np.bincount(choices.ravel(), minlength=len(tiles))
    report = text_report(tiles, uses, grid_shape=choices.shape, params={"lambda_": 1.0, "mu": 0.3, "tau": 0.5})
    (out_dir / "report.txt").write_text(report)
    assert "Tiles placed: 15" in report

    wall = cold_wall(tiles, uses, n=5, thumb_px=8, cols=5)
    Image.fromarray(wall).save(out_dir / "cold_wall.png")

    html = export_dzi(Image.fromarray(mosaic), out_dir / "dzi")
    assert html.exists()
```

- [ ] **Step 3: Run to verify pass**

```bash
pytest tests/test_smoke.py -v
```

Expected: test passes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke.py tests/conftest.py
git commit -m "test: end-to-end smoke covering pool → match → render → report → dzi"
```

---

## Task 12: `mosaic.ipynb` — 8-cell orchestration notebook

**Files:**
- Create: `mosaic.ipynb`

Because the notebook is JSON, the content below is the *source* of each cell; write a notebook via `jupytext` or by constructing the nbformat JSON in Python. Simplest path: write a tiny generator script and run it once.

- [ ] **Step 1: Write `scripts/build_notebook.py`**

```python
"""One-shot helper: generate mosaic.ipynb from inline cell sources."""
import json
from pathlib import Path

CELLS = [
    # Cell 1 — imports
    ("code", """\
%pip install -q -r requirements.txt
from pathlib import Path
import numpy as np
from PIL import Image
import ipywidgets as W
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

from mosaic.pool import scan_pool
from mosaic.target import load_and_slice
from mosaic.match import build_index, match_grid
from mosaic.render import compose
from mosaic.report import text_report, cold_wall, usage_hist_figure
from mosaic.dzi import export_dzi
"""),
    # Cell 2 — config + widgets
    ("code", """\
target_path = W.Text(value="", description="target:", layout=W.Layout(width="90%"))
pool_path = W.Text(value="", description="pool dir:", layout=W.Layout(width="90%"))
grid_w = W.IntSlider(value=120, min=40, max=240, step=4, description="grid_w")
tile_px = W.IntSlider(value=16, min=8, max=48, step=1, description="tile_px")
lam = W.FloatSlider(value=2.0, min=0.0, max=10.0, step=0.1, description="λ diversity")
mu = W.FloatSlider(value=0.5, min=0.0, max=5.0, step=0.1, description="μ neighbor")
tau = W.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05, description="τ tone")
regen = W.Button(description="Regenerate", button_style="primary")
dzi_btn = W.Button(description="Export DeepZoom")
log = W.Output(layout=W.Layout(height="200px", border="1px solid #888", overflow="auto"))
preview = W.Image(layout=W.Layout(width="100%"))

display(W.VBox([target_path, pool_path, W.HBox([grid_w, tile_px]), W.HBox([lam, mu, tau]),
                W.HBox([regen, dzi_btn]), log, preview]))

state = {"tiles": None, "grid": None, "mosaic": None, "out_dir": None}
"""),
    # Cell 3 — scan pool
    ("code", """\
def _scan():
    with log:
        clear_output()
        if not pool_path.value:
            print("fill pool dir path above, then run this cell")
            return
        print(f"scanning {pool_path.value} ...")
        tiles = scan_pool(Path(pool_path.value), Path("cache"), thumb_px=32)
        print(f"got {len(tiles)} tiles")
        state["tiles"] = tiles

_scan()
"""),
    # Cell 4 — load and slice target
    ("code", """\
def _slice():
    with log:
        clear_output()
        if not target_path.value:
            print("fill target path above, then run this cell")
            return
        tg = load_and_slice(Path(target_path.value), grid_w=grid_w.value, grid_h=int(grid_w.value * 9 / 16))
        state["grid"] = tg
        print(f"target sliced into {tg.lab_means.shape[1]} x {tg.lab_means.shape[0]} cells")

_slice()
"""),
    # Cell 5 — match
    ("code", """\
def _match():
    with log:
        clear_output()
        tiles = state["tiles"]
        tg = state["grid"]
        if not tiles or tg is None:
            print("run cells 3 and 4 first")
            return
        tile_labs = np.stack([t.lab for t in tiles])
        idx = build_index(tile_labs)
        choices = match_grid(idx, tile_labs, tg.lab_means, k=20,
                             lambda_=lam.value, mu=mu.value,
                             log_every=max(50, tg.lab_means.size // (3 * 20)))
        state["choices"] = choices
        print(f"match complete: {choices.shape}")

_match()
"""),
    # Cell 6 — render
    ("code", """\
def _render():
    with log:
        clear_output()
        tiles = state["tiles"]
        tg = state["grid"]
        choices = state.get("choices")
        if not tiles or tg is None or choices is None:
            print("run cells 3–5 first")
            return
        out_dir = Path("out") / Path(target_path.value).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        state["out_dir"] = out_dir
        mosaic = compose(tiles, choices, tg.lab_means, tile_px=tile_px.value, tau=tau.value)
        state["mosaic"] = mosaic
        Image.fromarray(mosaic).save(out_dir / "mosaic.png")
        with open(out_dir / "mosaic.png", "rb") as f:
            preview.value = f.read()
        print(f"wrote {out_dir / 'mosaic.png'}")

_render()
"""),
    # Cell 7 — report
    ("code", """\
def _report():
    with log:
        clear_output()
        tiles = state["tiles"]
        choices = state.get("choices")
        out_dir = state.get("out_dir")
        if not tiles or choices is None or out_dir is None:
            print("run cells 3–6 first")
            return
        uses = np.bincount(choices.ravel(), minlength=len(tiles))
        report = text_report(tiles, uses, grid_shape=choices.shape,
                             params={"lambda_": lam.value, "mu": mu.value, "tau": tau.value})
        (out_dir / "report.txt").write_text(report)
        print(report)

        wall = cold_wall(tiles, uses, n=20, thumb_px=32, cols=5)
        Image.fromarray(wall).save(out_dir / "cold_wall.png")

        fig = usage_hist_figure(tiles, uses)
        fig.savefig(out_dir / "usage_hist.png", dpi=110)
        plt.close(fig)

_report()
"""),
    # Cell 8 — DeepZoom export
    ("code", """\
def _dzi():
    with log:
        clear_output()
        mosaic = state.get("mosaic")
        out_dir = state.get("out_dir")
        if mosaic is None or out_dir is None:
            print("run cell 6 first")
            return
        html = export_dzi(Image.fromarray(mosaic), out_dir / "dzi")
        print(f"wrote {html}. Open it in a browser; share the folder {out_dir / 'dzi'} to give someone a zoomable version.")

_dzi()
"""),
]

def main():
    nb = {
        "cells": [
            {"cell_type": kind, "metadata": {}, "source": src, "outputs": [], "execution_count": None}
            if kind == "code"
            else {"cell_type": kind, "metadata": {}, "source": src}
            for kind, src in CELLS
        ],
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # link cells 5, 6, 7 to regen button and cell 8 to dzi_btn is skipped in v1 —
    # user reruns cells manually after moving sliders; good enough for a toy.
    Path("mosaic.ipynb").write_text(json.dumps(nb, indent=1))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it once to produce the notebook**

```bash
cd Tile.Photo/manus
mkdir -p scripts
# (the file above is written to scripts/build_notebook.py)
python scripts/build_notebook.py
ls -la mosaic.ipynb
```

Expected: `mosaic.ipynb` exists.

- [ ] **Step 3: Validate notebook JSON**

```bash
python -c "import json; json.loads(open('mosaic.ipynb').read()); print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_notebook.py mosaic.ipynb
git commit -m "feat(nb): 8-cell orchestration notebook with ipywidgets"
```

---

## Task 13: Wire Regenerate button + final polish

**Files:**
- Modify: `mosaic.ipynb` (via `scripts/build_notebook.py`)

- [ ] **Step 1: Edit `scripts/build_notebook.py`**

Append to Cell 2's source (just before the final blank line):

```python
def _on_regen(_):
    _match(); _render(); _report()

def _on_dzi(_):
    _dzi()

regen.on_click(_on_regen)
dzi_btn.on_click(_on_dzi)
```

Actually — since those functions are defined in later cells, wiring must happen in Cell 8 or in a new Cell 2 cleanup. Instead: move the `on_click` assignments into **Cell 8**, after `_dzi` is defined. Update `scripts/build_notebook.py` so Cell 8 ends with:

```python
regen.on_click(lambda _: (_match(), _render(), _report()))
dzi_btn.on_click(lambda _: _dzi())
```

- [ ] **Step 2: Regenerate notebook + re-validate**

```bash
python scripts/build_notebook.py
python -c "import json; nb = json.loads(open('mosaic.ipynb').read()); print('cells:', len(nb['cells']))"
```

Expected: `cells: 8`.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_notebook.py mosaic.ipynb
git commit -m "feat(nb): wire Regenerate + Export DeepZoom buttons"
```

---

## Task 14: Final verification + CHANGELOG update

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Run full test suite**

```bash
cd Tile.Photo/manus
pytest -v
```

Expected: all tests pass. Print the full output and paste into commit body below.

- [ ] **Step 2: Update `CHANGELOG.md`**

Change the status of the initial entry from `experimental` to `stable` **only if** you manually ran the notebook end-to-end on a real pool and target and got a reasonable-looking mosaic. If you only ran the smoke test, keep status as `experimental` and add a second entry:

```yaml
- date: 2026-04-17
  type: validation
  target: entire project
  change: Smoke pipeline passes; module interfaces match spec
  rationale: confirm scaffolding works before first real run
  action: pytest -v → all green (N tests)
  result: all tests pass
  validation: see test output in commit message
  status: stable (modules); experimental (whole notebook — needs user's real pool)
```

- [ ] **Step 3: Final commit**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog — smoke pipeline validated"
```

---

## Self-Review Log

### Spec coverage
- §2 user stories → Tasks 11 (smoke) + 12 (notebook)
- §3.1 file layout → Task 1 (scaffold) + 2–10 (module files)
- §3.2 8 cells → Task 12
- §4.1 LAB space → Task 2
- §4.2 cKDTree → Task 5
- §4.3 score rerank → Task 6
- §4.4 Reinhard → Task 7
- §4.5 default params → Tasks 6/7 defaults + Task 12 widgets
- §5 outputs → Task 12 cells 6–8
- §6 report format → Task 9 (including `describe_lab`)
- §7 DeepZoom → Task 10
- §8 widgets → Task 12 cell 2
- §9 error handling → Task 3 (pool skip on error); no other defense added (per spec)
- §10 smoke test → Task 11
- §11 CHANGELOG → Task 1 (initial) + Task 14 (validation)
- §12 out-of-scope → not implemented, as required
- §13 deps → Task 1
- §14 assumptions → documented in spec, not violated by any task

No gaps.

### Placeholder scan
- No TBD / TODO / "implement later" / "handle edge cases" without concrete code
- Every code step has actual code
- Every test step has actual assertions

### Type consistency
- `Tile` dataclass defined in Task 2, used by Tasks 3, 8, 9, 11 — fields `path`, `lab`, `thumb` consistent everywhere
- `TargetGrid` defined in Task 4 — fields `canvas`, `lab_means` consistent in Tasks 11, 12
- `scan_pool(pool_dir, cache_dir, thumb_px)` signature consistent across Tasks 3, 11, 12
- `match_grid(index, tile_labs, cell_labs, k, lambda_, mu, log_every)` signature consistent across Tasks 6, 11, 12
- `compose(tiles, grid, cell_lab_means, tile_px, tau)` signature consistent across Tasks 8, 11, 12
- `text_report(tiles, uses, grid_shape, params)` consistent across Tasks 9, 11, 12
- `cold_wall(tiles, uses, n, thumb_px, cols)` consistent across Tasks 9, 11, 12
- `export_dzi(image, out_dir) -> Path` consistent across Tasks 10, 11, 12
