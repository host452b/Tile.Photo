# Bead Mosaic Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Layer three interactive knobs (λ repetition penalty, μ neighbor penalty, τ Reinhard tone transfer) on top of the Phase 1 core pipeline, with ipywidgets sliders in the notebook.

**Architecture:** Extend `match_grid` with a greedy row-major path guarded by a fast-path special-case so λ=μ=0 stays bitwise-equal to Phase 1. Extract tone transfer to a new `src/tone.py` module so Phase 5 (CLIP) can swap it out. Render consumes `target_rgb` + `tone_strength` to apply Reinhard per tile when τ>0.

**Tech Stack:** Same as Phase 1 plus `ipywidgets>=8`.

**Reference spec:** `docs/superpowers/specs/2026-04-17-bead-mosaic-phase2-design.md`

**Working directory:** `/Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/claude`

---

## Task 1: Add ipywidgets dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Append ipywidgets to `requirements.txt`**

New full contents of `requirements.txt`:

```
pillow>=10
numpy>=1.26
scikit-image>=0.22
tqdm>=4.66
pillow-heif>=0.15
pytest>=8
jupyter>=1
nbformat>=5
ipywidgets>=8
```

- [ ] **Step 2: Install**

Run: `pip install -q -r requirements.txt 2>&1 | tail -5`
Expected: either a short install log for `ipywidgets` or a "Requirement already satisfied" line. No compile errors.

- [ ] **Step 3: Verify import works**

Run: `python -c "import ipywidgets; print(ipywidgets.__version__)"`
Expected: prints a version string (e.g., `8.1.x`).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "Add ipywidgets dep for Phase 2 sliders"
```

---

## Task 2: `src/tone.py` — Reinhard mean-only transfer

**Files:**
- Create: `src/tone.py`
- Create: `tests/test_tone.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tone.py`:

```python
import numpy as np

from src import tone


def _solid(rgb: tuple[int, int, int], size: int = 8) -> np.ndarray:
    return np.full((size, size, 3), rgb, dtype=np.uint8)


def test_strength_zero_returns_source_unchanged():
    src = _solid((120, 80, 40))
    tgt = _solid((20, 200, 200))
    out = tone.reinhard_transfer(src, tgt, strength=0.0)
    np.testing.assert_array_equal(out, src)


def test_strength_one_moves_source_mean_toward_target_mean():
    from skimage.color import rgb2lab

    src = _solid((120, 80, 40))
    tgt = _solid((20, 200, 200))
    out = tone.reinhard_transfer(src, tgt, strength=1.0)

    out_lab = rgb2lab(out.astype(np.float32) / 255.0).mean(axis=(0, 1))
    tgt_lab = rgb2lab(tgt.astype(np.float32) / 255.0).mean(axis=(0, 1))
    diff = np.linalg.norm(out_lab - tgt_lab)
    assert diff < 5.0, f"expected ΔE < 5, got {diff}"


def test_strength_half_is_between_source_and_target():
    from skimage.color import rgb2lab

    src = _solid((120, 80, 40))
    tgt = _solid((20, 200, 200))
    src_lab = rgb2lab(src.astype(np.float32) / 255.0).mean(axis=(0, 1))
    tgt_lab = rgb2lab(tgt.astype(np.float32) / 255.0).mean(axis=(0, 1))

    out = tone.reinhard_transfer(src, tgt, strength=0.5)
    out_lab = rgb2lab(out.astype(np.float32) / 255.0).mean(axis=(0, 1))

    to_src = np.linalg.norm(out_lab - src_lab)
    to_tgt = np.linalg.norm(out_lab - tgt_lab)
    full = np.linalg.norm(tgt_lab - src_lab)
    assert 0.2 * full < to_src < 0.8 * full
    assert 0.2 * full < to_tgt < 0.8 * full


def test_output_shape_and_dtype_preserved():
    src = np.zeros((12, 16, 3), dtype=np.uint8)
    tgt = np.full((12, 16, 3), 200, dtype=np.uint8)
    out = tone.reinhard_transfer(src, tgt, strength=0.5)
    assert out.shape == (12, 16, 3)
    assert out.dtype == np.uint8
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_tone.py -v 2>&1 | tail -5`
Expected: FAIL — `ImportError: cannot import name 'tone'`.

- [ ] **Step 3: Implement `src/tone.py`**

```python
from __future__ import annotations

import numpy as np
from skimage.color import lab2rgb, rgb2lab


def reinhard_transfer(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Shift source's LAB mean toward target's LAB mean by `strength`.

    Args:
        source_rgb: (H, W, 3) uint8 — the tile being placed.
        target_rgb: (H, W, 3) uint8 — the target patch at that grid cell.
        strength: [0, 1]. 0 = return source unchanged. 1 = adopt target mean.

    Returns:
        (H, W, 3) uint8, tone-shifted source.
    """
    if strength == 0.0:
        return source_rgb
    source_lab = rgb2lab(source_rgb.astype(np.float32) / 255.0)
    target_lab_mean = rgb2lab(target_rgb.astype(np.float32) / 255.0).mean(axis=(0, 1))
    source_lab_mean = source_lab.mean(axis=(0, 1))
    delta = (target_lab_mean - source_lab_mean) * strength
    adjusted_lab = source_lab + delta
    adjusted_rgb = lab2rgb(adjusted_lab)
    return np.clip(adjusted_rgb * 255.0, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_tone.py -v 2>&1 | tail -7`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tone.py tests/test_tone.py
git commit -m "Add tone.reinhard_transfer (LAB mean-only)"
```

---

## Task 3: `match_grid` — λ repetition penalty

**Files:**
- Modify: `src/match.py`
- Modify: `tests/test_match.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_match.py`:

```python
def test_lambda_zero_matches_phase1_broadcast():
    rng = np.random.default_rng(0)
    pool = rng.uniform(0, 100, size=(30, 3)).astype(np.float32)
    target = rng.uniform(0, 100, size=(5, 7, 3)).astype(np.float32)

    old = match.match_grid(target, pool)
    new = match.match_grid(target, pool, lambda_=0.0, mu=0.0)
    np.testing.assert_array_equal(old, new)


def test_lambda_reduces_max_usage():
    pool = np.array(
        [
            [50.0, 0.0, 0.0],
            [50.0, 10.0, 10.0],
            [50.0, -10.0, -10.0],
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(1)
    target = rng.uniform(40, 60, size=(10, 10, 3)).astype(np.float32)

    idx0 = match.match_grid(target, pool, lambda_=0.0)
    idx_penalty = match.match_grid(target, pool, lambda_=20.0)

    max0 = np.bincount(idx0.ravel(), minlength=3).max()
    maxp = np.bincount(idx_penalty.ravel(), minlength=3).max()
    assert maxp < max0, f"expected λ=20 to flatten usage; max0={max0}, maxp={maxp}"
    usage_p = np.bincount(idx_penalty.ravel(), minlength=3)
    assert (usage_p >= 1).all(), f"expected all 3 tiles used; got {usage_p}"
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_match.py::test_lambda_zero_matches_phase1_broadcast tests/test_match.py::test_lambda_reduces_max_usage -v 2>&1 | tail -5`
Expected: FAIL — `TypeError: match_grid() got an unexpected keyword argument 'lambda_'`.

- [ ] **Step 3: Replace `src/match.py` with the extended version**

Full replacement for `src/match.py`:

```python
from __future__ import annotations

import numpy as np


def match_grid(
    target_lab: np.ndarray,
    pool_lab: np.ndarray,
    *,
    lambda_: float = 0.0,
    mu: float = 0.0,
) -> np.ndarray:
    """For each cell of target_lab, pick a pool index.

    Args:
        target_lab: (H, W, 3) float32 — LAB mean per cell.
        pool_lab:   (N, 3)    float32 — LAB mean per pool tile.
        lambda_:    repetition penalty coefficient (>=0).
        mu:         neighbor-match penalty coefficient (>=0).

    Returns:
        (H, W) int32 of indices into pool_lab.
    """
    if pool_lab.shape[0] == 0:
        raise ValueError("pool_lab is empty; cannot match")

    if lambda_ == 0.0 and mu == 0.0:
        target_f = target_lab.astype(np.float32, copy=False)
        pool_f = pool_lab.astype(np.float32, copy=False)
        diff = target_f[:, :, None, :] - pool_f[None, None, :, :]
        dist2 = (diff * diff).sum(axis=-1)
        return dist2.argmin(axis=-1).astype(np.int32)

    return _greedy_match(target_lab, pool_lab, lambda_, mu)


def _greedy_match(
    target_lab: np.ndarray,
    pool_lab: np.ndarray,
    lambda_: float,
    mu: float,
) -> np.ndarray:
    h, w, _ = target_lab.shape
    n = pool_lab.shape[0]
    placed = np.full((h, w), -1, dtype=np.int32)
    uses = np.zeros(n, dtype=np.int64)
    tgt = target_lab.astype(np.float32, copy=False)
    pool = pool_lab.astype(np.float32, copy=False)

    for r in range(h):
        for c in range(w):
            diff = tgt[r, c] - pool
            dist2 = (diff * diff).sum(axis=-1)
            score = dist2 + lambda_ * np.log1p(uses)
            if mu > 0.0:
                if r > 0:
                    score[placed[r - 1, c]] += mu
                if c > 0:
                    score[placed[r, c - 1]] += mu
            idx = int(score.argmin())
            placed[r, c] = idx
            uses[idx] += 1
    return placed
```

- [ ] **Step 4: Run λ tests to verify pass**

Run: `pytest tests/test_match.py -v 2>&1 | tail -8`
Expected: all 5 tests PASS (3 Phase 1 + 2 new λ tests).

- [ ] **Step 5: Commit**

```bash
git add src/match.py tests/test_match.py
git commit -m "Add λ repetition penalty to match_grid (greedy path)"
```

---

## Task 4: `match_grid` — μ neighbor penalty

**Files:**
- Modify: `tests/test_match.py`

Note: the greedy `_greedy_match` from Task 3 already implements μ. This task just adds behavioral tests to verify it works.

- [ ] **Step 1: Write failing test**

Append to `tests/test_match.py`:

```python
def test_mu_penalizes_adjacent_duplicates():
    pool = np.array(
        [
            [50.0, 0.0, 0.0],
            [50.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )
    target = np.full((6, 6, 3), [50.0, 2.5, 2.5], dtype=np.float32)

    idx = match.match_grid(target, pool, lambda_=0.0, mu=500.0)
    for r in range(6):
        for c in range(6):
            if r > 0:
                assert idx[r, c] != idx[r - 1, c], f"top neighbor match at ({r},{c})"
            if c > 0:
                assert idx[r, c] != idx[r, c - 1], f"left neighbor match at ({r},{c})"


def test_mu_zero_does_not_affect_result():
    rng = np.random.default_rng(2)
    pool = rng.uniform(0, 100, size=(10, 3)).astype(np.float32)
    target = rng.uniform(0, 100, size=(4, 5, 3)).astype(np.float32)

    no_mu = match.match_grid(target, pool, lambda_=3.0, mu=0.0)
    with_zero = match.match_grid(target, pool, lambda_=3.0, mu=0.0)
    np.testing.assert_array_equal(no_mu, with_zero)
```

- [ ] **Step 2: Run to verify pass (μ impl already in place from Task 3)**

Run: `pytest tests/test_match.py -v 2>&1 | tail -10`
Expected: all 7 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_match.py
git commit -m "Add μ neighbor penalty behavior tests"
```

---

## Task 5: `render_mosaic_with_usage` — τ integration

**Files:**
- Modify: `src/render.py`
- Modify: `tests/test_render.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_render.py`:

```python
def test_tone_strength_zero_matches_phase1_bitwise(tmp_path):
    _solid_thumb(tmp_path / "r.png", (255, 0, 0), 8)
    _solid_thumb(tmp_path / "g.png", (0, 255, 0), 8)
    pool = TilePool(
        lab=np.zeros((2, 3), dtype=np.float32),
        thumbs_paths=[str(tmp_path / "r.png"), str(tmp_path / "g.png")],
        source_paths=["r", "g"],
    )
    idx = np.array([[0, 1], [1, 0]], dtype=np.int32)

    phase1 = render.render_mosaic(idx, pool, tile_px=8, output_path=tmp_path / "a.png")
    phase2 = render.render_mosaic_with_usage(
        idx, pool, tile_px=8, output_path=tmp_path / "b.png",
        target_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        tone_strength=0.0,
    )[0]

    np.testing.assert_array_equal(np.asarray(phase1), np.asarray(phase2))


def test_tone_strength_one_shifts_tiles_toward_target(tmp_path):
    from skimage.color import rgb2lab

    _solid_thumb(tmp_path / "gray.png", (128, 128, 128), 8)
    pool = TilePool(
        lab=np.zeros((1, 3), dtype=np.float32),
        thumbs_paths=[str(tmp_path / "gray.png")],
        source_paths=["gray"],
    )
    idx = np.array([[0, 0], [0, 0]], dtype=np.int32)

    target_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    target_rgb[0:8, 0:8] = (220, 40, 40)
    target_rgb[0:8, 8:16] = (40, 220, 40)
    target_rgb[8:16, 0:8] = (40, 40, 220)
    target_rgb[8:16, 8:16] = (220, 220, 40)

    img, _ = render.render_mosaic_with_usage(
        idx, pool, tile_px=8, output_path=tmp_path / "m.png",
        target_rgb=target_rgb, tone_strength=1.0,
    )
    out = np.asarray(img)
    for (r, c), expected_rgb in [
        ((0, 0), (220, 40, 40)),
        ((0, 8), (40, 220, 40)),
        ((8, 0), (40, 40, 220)),
        ((8, 8), (220, 220, 40)),
    ]:
        tile = out[r : r + 8, c : c + 8]
        tile_mean_lab = rgb2lab(tile.astype(np.float32) / 255.0).mean(axis=(0, 1))
        expected_patch = np.full((8, 8, 3), expected_rgb, dtype=np.uint8)
        expected_mean_lab = rgb2lab(expected_patch.astype(np.float32) / 255.0).mean(axis=(0, 1))
        diff = np.linalg.norm(tile_mean_lab - expected_mean_lab)
        assert diff < 5.0, f"patch at ({r},{c}) LAB ΔE = {diff}"
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_render.py::test_tone_strength_zero_matches_phase1_bitwise tests/test_render.py::test_tone_strength_one_shifts_tiles_toward_target -v 2>&1 | tail -5`
Expected: FAIL — `TypeError: ... got an unexpected keyword argument 'target_rgb'`.

- [ ] **Step 3: Replace `src/render.py` with the extended version**

Full replacement for `src/render.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_render.py -v 2>&1 | tail -8`
Expected: all 4 tests PASS (2 Phase 1 + 2 new τ tests).

- [ ] **Step 5: Commit**

```bash
git add src/render.py tests/test_render.py
git commit -m "Add τ tone-transfer path to render_mosaic_with_usage"
```

---

## Task 6: End-to-end smoke test with knobs

**Files:**
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_pipeline.py`:

```python
def test_end_to_end_with_knobs(tmp_path):
    base = tmp_path / "photos"
    base.mkdir()
    for i, rgb in enumerate(PALETTE):
        _solid_jpg(base / f"c{i}.jpg", rgb)

    pool = scan.build_pool(base, tmp_path / "cache", tile_px=24)

    grid_h, grid_w = 4, 10
    tile_px = 24
    target_rgb = np.zeros((grid_h * tile_px, grid_w * tile_px, 3), dtype=np.uint8)
    for col in range(grid_w):
        target_rgb[:, col * tile_px : (col + 1) * tile_px] = PALETTE[col]

    from skimage.color import rgb2lab

    target_lab = rgb2lab(target_rgb.astype(np.float32) / 255.0)
    target_lab_grid = target_lab.reshape(
        grid_h, tile_px, grid_w, tile_px, 3
    ).mean(axis=(1, 3)).astype(np.float32)

    idx = match.match_grid(target_lab_grid, pool.lab, lambda_=2.0, mu=10.0)
    img, usage = render.render_mosaic_with_usage(
        idx, pool, tile_px, tmp_path / "out.png",
        target_rgb=target_rgb, tone_strength=0.3,
    )
    assert img.size == (grid_w * tile_px, grid_h * tile_px)
    assert len(usage) >= len(PALETTE) // 2
```

- [ ] **Step 2: Run full suite**

Run: `pytest -v 2>&1 | tail -22`
Expected: 18 tests PASS (12 from Phase 1 + 4 tone + 2 λ/μ... wait recount: scan 6 + match 5 + render 4 + tone 4 + pipeline 2 = 21 tests PASS).

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "Add end-to-end smoke test with λ/μ/τ knobs"
```

---

## Task 7: Regenerate notebook with CONFIG + slider cell

**Files:**
- Modify: `scripts/build_notebook.py`
- Modify: `bead_mosaic.ipynb` (generated)

- [ ] **Step 1: Replace `scripts/build_notebook.py`**

Full replacement:

```python
"""Build bead_mosaic.ipynb via nbformat. Run: python scripts/build_notebook.py"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bead_mosaic.ipynb"

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Bead Mosaic — Phase 2\n"
    "Zero-config demo: run all cells top-to-bottom.\n"
    "To use your own photos: set `CONFIG['BASE_DIR']` + `CONFIG['TARGET_PATH']` in Cell 2 and flip `DEMO_MODE` to `False`.\n"
    "\n"
    "**Phase 2 adds three knobs:**\n"
    "- `LAMBDA` — repetition penalty (0 = no penalty, large = force tile diversity)\n"
    "- `MU` — neighbor penalty (0 = ignore, large = no adjacent duplicates)\n"
    "- `TAU` — tone transfer strength [0, 1] (0 = original tile colors, 1 = tiles adopt target patch's LAB mean)\n"
))

cells.append(nbf.v4.new_code_cell(
    "%pip install -q -r requirements.txt\n"
    "import sys\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, str(Path.cwd()))\n"
    "\n"
    "import numpy as np\n"
    "import ipywidgets as widgets\n"
    "from PIL import Image, ImageOps\n"
    "from skimage import data\n"
    "from skimage.color import rgb2lab\n"
    "from IPython.display import display\n"
    "\n"
    "from src import match, render, scan"
))

cells.append(nbf.v4.new_code_cell(
    "# --- edit these for real usage ---\n"
    "CONFIG = {\n"
    "    'BASE_DIR': Path.cwd() / 'my_photos',\n"
    "    'TARGET_PATH': None,  # None -> use skimage astronaut\n"
    "    'GRID_W': 120,\n"
    "    'GRID_H': 68,\n"
    "    'TILE_PX': 24,\n"
    "    'CACHE_DIR': Path.cwd() / '.cache',\n"
    "    'OUTPUT_PATH': Path.cwd() / 'output.png',\n"
    "    'DEMO_MODE': True,\n"
    "    'LAMBDA': 0.0,  # repetition penalty\n"
    "    'MU': 0.0,      # neighbor penalty\n"
    "    'TAU': 0.0,     # tone transfer strength [0, 1]\n"
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
    "idx = match.match_grid(\n"
    "    target_lab_grid, pool.lab,\n"
    "    lambda_=CONFIG['LAMBDA'], mu=CONFIG['MU'],\n"
    ")\n"
    "idx.shape"
))

cells.append(nbf.v4.new_code_cell(
    "img, usage = render.render_mosaic_with_usage(\n"
    "    idx, pool, CONFIG['TILE_PX'], CONFIG['OUTPUT_PATH'],\n"
    "    target_rgb=target_rgb, tone_strength=CONFIG['TAU'],\n"
    ")\n"
    "display(img)\n"
    "print(f\"\\noutput written to {CONFIG['OUTPUT_PATH']}\")\n"
    "print(f\"tiles used: {len(usage)} distinct / {sum(usage.values())} total placements\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 交互滑条\n"
    "拖动滑条不会自动重跑——点 **重跑** 按钮触发一次 match+render。"
))

cells.append(nbf.v4.new_code_cell(
    "lambda_slider = widgets.FloatSlider(value=CONFIG['LAMBDA'], min=0, max=50, step=0.5, description='λ (重复)', continuous_update=False)\n"
    "mu_slider = widgets.FloatSlider(value=CONFIG['MU'], min=0, max=200, step=5, description='μ (邻居)', continuous_update=False)\n"
    "tau_slider = widgets.FloatSlider(value=CONFIG['TAU'], min=0, max=1, step=0.05, description='τ (色调)', continuous_update=False)\n"
    "rerun_btn = widgets.Button(description='重跑', button_style='primary')\n"
    "out = widgets.Output()\n"
    "\n"
    "def _rerun(_):\n"
    "    with out:\n"
    "        out.clear_output()\n"
    "        idx = match.match_grid(\n"
    "            target_lab_grid, pool.lab,\n"
    "            lambda_=lambda_slider.value, mu=mu_slider.value,\n"
    "        )\n"
    "        img, usage = render.render_mosaic_with_usage(\n"
    "            idx, pool, CONFIG['TILE_PX'], CONFIG['OUTPUT_PATH'],\n"
    "            target_rgb=target_rgb, tone_strength=tau_slider.value,\n"
    "        )\n"
    "        display(img)\n"
    "        print(f\"tiles used: {len(usage)} distinct / {sum(usage.values())} total\")\n"
    "\n"
    "rerun_btn.on_click(_rerun)\n"
    "display(widgets.VBox([lambda_slider, mu_slider, tau_slider, rerun_btn, out]))"
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

- [ ] **Step 2: Regenerate the notebook**

Run: `python scripts/build_notebook.py`
Expected: prints `wrote .../bead_mosaic.ipynb`.

- [ ] **Step 3: Execute notebook end-to-end in demo mode**

Run: `jupyter nbconvert --to notebook --execute bead_mosaic.ipynb --output bead_mosaic.ipynb`
Expected: all cells execute with no error. `output.png` exists in project root.

- [ ] **Step 4: Eyeball output.png**

Read `output.png` visually.
Expected: recognizable astronaut (LAMBDA/MU/TAU all default 0, so output should be ≈ identical to Phase 1). If the image looks wrong or different from Phase 1, stop and debug.

- [ ] **Step 5: Clear cell outputs for clean commit**

Run: `jupyter nbconvert --clear-output --inplace bead_mosaic.ipynb`

- [ ] **Step 6: Commit**

```bash
git add scripts/build_notebook.py bead_mosaic.ipynb
git commit -m "Regen notebook with λ/μ/τ CONFIG + ipywidgets slider cell"
```

---

## Task 8: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Append Phase 2 entry**

Insert this entry **above** the existing Phase 1 entry in the "活跃条目" section of `CHANGELOG.md`:

```markdown
- date: 2026-04-17
  type: feat
  target: Phase 2 knobs (src/match.py, src/render.py, src/tone.py, bead_mosaic.ipynb)
  change: 三个交互旋钮落地。match_grid 加 kwonly `lambda_` (重复惩罚，score += λ·log1p(uses)) 与 `mu` (邻居惩罚，score += μ·(top_match + left_match))；λ=μ=0 保留 Phase 1 broadcast fast-path 确保 bitwise-equal。render_mosaic_with_usage 加 kwonly `target_rgb` + `tone_strength`，τ>0 时每 tile 贴图前跑 Reinhard LAB 均值迁移 (新模块 src/tone.py)。notebook 加 ipywidgets 三滑条 + "重跑" 按钮 cell。
  rationale: Phase 1 brainstorm 里锁定的"甜区三滑条"。λ 治"万能照被狂贴"，μ 治"大面积同色扎堆"，τ 治"远看色斑"。全部 kwonly 默认 0 让 Phase 1 的 12 个测试和 demo 行为不动。
  action: 7 个 TDD 任务；新增 src/tone.py 独立模块（为 Phase 5 CLIP 可换色调算法预埋）；match_grid 引入 _greedy_match 私有路径（Python 外循环 + numpy 内向量化，8160 格 <1s）；render fast-path 在 τ=0 或 target_rgb=None 时完全跳过 LAB 转换。
  result: 12 Phase 1 测试全绿未改；新增 9 测试全过（tone=4, match λ/μ=4, render τ=2, pipeline knobs=1）；demo notebook 默认参数下 output.png 与 Phase 1 结果肉眼一致（λ=μ=τ=0）；手动将 LAMBDA=5, TAU=0.4 重跑可见 tile 分布更散、贴图色调更贴合 target（视觉确认）。
  validation: pytest 21/21；`jupyter nbconvert --execute bead_mosaic.ipynb` 无报错；Read output.png 检查可辨识度；手动交互测试滑条 rerun 回调。
  status: stable
  spec: docs/superpowers/specs/2026-04-17-bead-mosaic-phase2-design.md
  plan: docs/superpowers/plans/2026-04-17-bead-mosaic-phase2.md
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "Log Phase 2 completion in CHANGELOG"
```

---

## Self-Review Checklist

**Spec coverage:**
- §2.1 λ behavior → Task 3 (impl) + Task 3 tests (λ=0 regression, λ>0 flattens usage)
- §2.2 μ behavior → Task 3 (impl inside _greedy_match) + Task 4 tests (μ>0 no neighbors match, μ=0 no-op)
- §2.3 τ Reinhard mean-only → Task 2 (tone module) + Task 5 (render integration)
- §3.1 match_grid signature → Task 3
- §3.2 render signature → Task 5
- §3.3 tone.reinhard_transfer signature → Task 2
- §3.4 notebook changes + slider cell → Task 7
- §4 ipywidgets dep → Task 1
- §6.1 greedy loop + fast-path for λ=μ=0 → Task 3 implementation
- §6.2 τ=0 bitwise-equal Phase 1 → Task 5 test `test_tone_strength_zero_matches_phase1_bitwise`
- §7.1 Phase 1 regression → implicitly via every test run (Phase 1 tests are preserved)
- §7.2/7.3/7.4 behavior tests → Tasks 3, 4, 5
- §7.5 integration smoke → Task 6

**Placeholder scan:** No TBDs, no "handle error", no "similar to Task N". Full code in every step.

**Type consistency:**
- `match_grid(target_lab, pool_lab, *, lambda_=0.0, mu=0.0)` — used consistently across Tasks 3, 4, 6, 7
- `render_mosaic_with_usage(index_grid, pool, tile_px, output_path, *, target_rgb=None, tone_strength=0.0)` — consistent Tasks 5, 6, 7
- `reinhard_transfer(source_rgb, target_rgb, strength)` — consistent Tasks 2, 5
- `TilePool` import path unchanged

Plan ready.
