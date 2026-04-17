# Photomosaic ipynb 生成器 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个本地 Mac 上跑的 Jupyter notebook photomosaic 生成器，把硬盘里一堆照片重组成另一张照片，MVP 含颜色匹配 + 重复/邻居惩罚 + Reinhard 色调迁移 + ipywidgets 交互 + 自嘲式报告 + DeepZoom 导出。

**Architecture:** 一个 `mosaic.ipynb` 作入口（8 cell），业务逻辑全拆到 `mosaic/` package 下 8 个 <300 行模块文件。贪心匹配 + faiss 加速 + ipywidgets 交互。详见 `docs/superpowers/specs/2026-04-17-photomosaic-ipynb-design.md`。

**Tech Stack:** Python 3.11+, pillow, numpy, scikit-image, faiss-cpu, tqdm, ipywidgets, matplotlib, deepzoom, pytest。纯 pip，无 brew 依赖。

**Spec reference:** `docs/superpowers/specs/2026-04-17-photomosaic-ipynb-design.md`

---

## Task 0: 项目脚手架

**Files:**
- Create: `perplexity/requirements.txt`
- Create: `perplexity/mosaic/__init__.py`
- Create: `perplexity/tests/__init__.py`
- Create: `perplexity/tests/conftest.py`

- [ ] **Step 1: 写 requirements.txt**

Create `perplexity/requirements.txt`:
```
pillow>=10.0
numpy>=1.26
scikit-image>=0.22
faiss-cpu>=1.8
tqdm>=4.66
ipywidgets>=8.1
matplotlib>=3.8
deepzoom>=0.5
pytest>=8.0
jupyter>=1.0
notebook>=7.0
```

- [ ] **Step 2: 建 package 空壳**

Create `perplexity/mosaic/__init__.py`:
```python
"""Photomosaic generator — toy project, Mac local, ipynb entry."""
__version__ = "0.1.0"
```

Create `perplexity/tests/__init__.py`: (empty file)

- [ ] **Step 3: 建 pytest fixtures**

Create `perplexity/tests/conftest.py`:
```python
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tmp_pool_dir(tmp_path):
    """16 张纯色 tile（4x4 颜色网格）写到临时目录。"""
    pool = tmp_path / "pool"
    pool.mkdir()
    for i in range(4):
        for j in range(4):
            color = (i * 64, j * 64, ((i + j) * 32) % 256)
            img = Image.new("RGB", (64, 64), color)
            img.save(pool / f"tile_{i}_{j}.jpg")
    return pool


@pytest.fixture
def tmp_target_img(tmp_path):
    """4x4 四象限纯色目标图（每象限 64x64）。"""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:128, :128] = (255, 0, 0)
    arr[:128, 128:] = (0, 255, 0)
    arr[128:, :128] = (0, 0, 255)
    arr[128:, 128:] = (255, 255, 0)
    path = tmp_path / "target.png"
    Image.fromarray(arr).save(path)
    return path
```

- [ ] **Step 4: 安装依赖并 smoke test**

Run:
```bash
cd perplexity && pip install -r requirements.txt
cd perplexity && python -c "import mosaic; print(mosaic.__version__)"
cd perplexity && python -m pytest tests/ -v
```

Expected:
- `pip install` 成功无错
- `import mosaic` 打印 `0.1.0`
- pytest 跑 0 tests（没写 test 但 conftest 不该报错）

- [ ] **Step 5: Commit**

```bash
git add perplexity/requirements.txt perplexity/mosaic/ perplexity/tests/
git commit -m "scaffold: photomosaic package skeleton + pytest fixtures"
```

---

## Task 1: pool.py — 底图扫描与 LAB 缓存

**Files:**
- Create: `perplexity/mosaic/pool.py`
- Create: `perplexity/tests/test_pool.py`

- [ ] **Step 1: 写 test_pool.py 失败测试**

Create `perplexity/tests/test_pool.py`:
```python
import pickle
import time

import numpy as np
import pytest
from PIL import Image

from mosaic.pool import scan_pool, load_cache


def test_scan_returns_entry_per_image(tmp_pool_dir, tmp_path):
    cache_path = tmp_path / "pool.pkl"
    features = scan_pool(tmp_pool_dir, cache_path)
    assert len(features) == 16
    for path, entry in features.items():
        assert "lab_mean" in entry
        assert "thumbnail" in entry
        assert "mtime" in entry
        assert entry["lab_mean"].shape == (3,)
        assert entry["thumbnail"].shape == (16, 16, 3)


def test_scan_caches_to_pickle(tmp_pool_dir, tmp_path):
    cache_path = tmp_path / "pool.pkl"
    scan_pool(tmp_pool_dir, cache_path)
    assert cache_path.exists()
    with open(cache_path, "rb") as f:
        loaded = pickle.load(f)
    assert len(loaded) == 16


def test_scan_is_incremental_on_mtime(tmp_pool_dir, tmp_path):
    cache_path = tmp_path / "pool.pkl"
    first = scan_pool(tmp_pool_dir, cache_path)
    first_lab = next(iter(first.values()))["lab_mean"].copy()

    target = next(iter(tmp_pool_dir.glob("*.jpg")))
    time.sleep(0.01)
    Image.new("RGB", (64, 64), (10, 20, 30)).save(target)

    second = scan_pool(tmp_pool_dir, cache_path)
    changed = second[str(target)]["lab_mean"]
    assert not np.allclose(changed, first_lab)


def test_scan_drops_deleted_files(tmp_pool_dir, tmp_path):
    cache_path = tmp_path / "pool.pkl"
    scan_pool(tmp_pool_dir, cache_path)

    victim = next(iter(tmp_pool_dir.glob("*.jpg")))
    victim.unlink()

    second = scan_pool(tmp_pool_dir, cache_path)
    assert str(victim) not in second
    assert len(second) == 15


def test_scan_skips_non_images(tmp_pool_dir, tmp_path):
    (tmp_pool_dir / "notes.txt").write_text("not an image")
    (tmp_pool_dir / "broken.jpg").write_bytes(b"not really a jpg")
    cache_path = tmp_path / "pool.pkl"
    features = scan_pool(tmp_pool_dir, cache_path)
    assert len(features) == 16  # 16 real, 2 skipped


def test_load_cache_returns_empty_when_missing(tmp_path):
    features = load_cache(tmp_path / "nonexistent.pkl")
    assert features == {}
```

- [ ] **Step 2: 跑测试确认全失败**

Run: `cd perplexity && python -m pytest tests/test_pool.py -v`

Expected: 6 tests error with `ImportError: cannot import name 'scan_pool' from 'mosaic.pool'`

- [ ] **Step 3: 实现 pool.py**

Create `perplexity/mosaic/pool.py`:
```python
"""底图池扫描 + LAB 平均色 + pickle 缓存（按 mtime 增量）。"""
from __future__ import annotations

import os
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
```

- [ ] **Step 4: 跑测试确认全过**

Run: `cd perplexity && python -m pytest tests/test_pool.py -v`

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add perplexity/mosaic/pool.py perplexity/tests/test_pool.py
git commit -m "feat(pool): scan tile pool with LAB mean + mtime pickle cache"
```

---

## Task 2: target.py — 读目标图 + 分网格

**Files:**
- Create: `perplexity/mosaic/target.py`
- Create: `perplexity/tests/test_target.py`

- [ ] **Step 1: 写 test_target.py 失败测试**

Create `perplexity/tests/test_target.py`:
```python
import numpy as np
import pytest

from mosaic.target import load_and_grid


def test_grid_produces_cols_x_rows_cells(tmp_target_img):
    grid = load_and_grid(tmp_target_img, grid_cols=4, grid_rows=4)
    assert grid["shape"] == (4, 4)
    assert len(grid["cells"]) == 16


def test_each_cell_has_lab_mean_and_variance(tmp_target_img):
    grid = load_and_grid(tmp_target_img, grid_cols=4, grid_rows=4)
    for cell in grid["cells"]:
        assert cell["lab_mean"].shape == (3,)
        assert np.isscalar(cell["variance"]) or cell["variance"].shape == ()
        assert "row" in cell and "col" in cell


def test_quadrants_have_distinct_colors(tmp_target_img):
    grid = load_and_grid(tmp_target_img, grid_cols=2, grid_rows=2)
    means = [c["lab_mean"] for c in grid["cells"]]
    # four cells, all pairs should differ
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.allclose(means[i], means[j], atol=1.0)


def test_center_crops_non_matching_aspect(tmp_path):
    # 320x256 target, grid 2x2 (asks for 1:1) → should center-crop to 256x256
    from PIL import Image
    arr = np.zeros((256, 320, 3), dtype=np.uint8)
    arr[:, :160] = (255, 0, 0)
    arr[:, 160:] = (0, 255, 0)
    path = tmp_path / "wide.png"
    Image.fromarray(arr).save(path)

    grid = load_and_grid(path, grid_cols=2, grid_rows=2)
    # After center crop to 256x256, split into 2x2, left and right halves different
    cells = {(c["row"], c["col"]): c["lab_mean"] for c in grid["cells"]}
    assert not np.allclose(cells[(0, 0)], cells[(0, 1)], atol=1.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd perplexity && python -m pytest tests/test_target.py -v`

Expected: 4 tests error with `ImportError`

- [ ] **Step 3: 实现 target.py**

Create `perplexity/mosaic/target.py`:
```python
"""读目标图、按网格宽高比 center-crop、分网格计算每格 LAB 均值与方差。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def _center_crop_to_aspect(img: Image.Image, target_cols: int, target_rows: int) -> Image.Image:
    w, h = img.size
    target_ratio = target_cols / target_rows
    img_ratio = w / h
    if abs(img_ratio - target_ratio) < 1e-6:
        return img
    if img_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    new_h = int(w / target_ratio)
    top = (h - new_h) // 2
    return img.crop((0, top, w, top + new_h))


def load_and_grid(target_path: Path, grid_cols: int, grid_rows: int) -> dict:
    """返回 {'shape': (rows, cols), 'cells': [{'row', 'col', 'lab_mean', 'variance'}, ...], 'image': PIL.Image}."""
    img = Image.open(target_path).convert("RGB")
    img = _center_crop_to_aspect(img, grid_cols, grid_rows)

    arr = np.asarray(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    usable = arr[: cell_h * grid_rows, : cell_w * grid_cols]
    lab = rgb2lab(usable / 255.0).astype(np.float32)

    cells = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            patch = lab[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            cells.append({
                "row": r,
                "col": c,
                "lab_mean": patch.reshape(-1, 3).mean(axis=0),
                "variance": float(patch.reshape(-1, 3).var(axis=0).sum()),
            })

    return {
        "shape": (grid_rows, grid_cols),
        "cells": cells,
        "image": img,
        "cell_size": (cell_h, cell_w),
    }
```

- [ ] **Step 4: 跑测试确认过**

Run: `cd perplexity && python -m pytest tests/test_target.py -v`

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add perplexity/mosaic/target.py perplexity/tests/test_target.py
git commit -m "feat(target): load image, center-crop to aspect, split into grid with LAB mean + variance"
```

---

## Task 3: match.py — faiss top-K + λ/μ 贪心求解器

**Files:**
- Create: `perplexity/mosaic/match.py`
- Create: `perplexity/tests/test_match.py`

- [ ] **Step 1: 写 test_match.py 失败测试**

Create `perplexity/tests/test_match.py`:
```python
import numpy as np
import pytest

from mosaic.match import solve_assignment


def _synthetic_setup():
    """造 4 种纯色 tile + 4 格目标（每格颜色对应一种 tile）。"""
    pool = {}
    for i, color_lab in enumerate([
        np.array([50, 80, 60], dtype=np.float32),   # red-ish
        np.array([90, -80, 80], dtype=np.float32),  # green-ish
        np.array([30, 70, -100], dtype=np.float32), # blue-ish
        np.array([95, -20, 90], dtype=np.float32),  # yellow-ish
    ]):
        pool[f"tile_{i}.jpg"] = {
            "mtime": 0.0,
            "lab_mean": color_lab,
            "thumbnail": np.full((16, 16, 3), i * 50, dtype=np.uint8),
        }

    cells = [
        {"row": 0, "col": 0, "lab_mean": np.array([50, 80, 60], dtype=np.float32), "variance": 0.0},
        {"row": 0, "col": 1, "lab_mean": np.array([90, -80, 80], dtype=np.float32), "variance": 0.0},
        {"row": 1, "col": 0, "lab_mean": np.array([30, 70, -100], dtype=np.float32), "variance": 0.0},
        {"row": 1, "col": 1, "lab_mean": np.array([95, -20, 90], dtype=np.float32), "variance": 0.0},
    ]
    return pool, cells


def test_no_penalty_picks_nearest_color():
    pool, cells = _synthetic_setup()
    assignment = solve_assignment(
        pool, cells, grid_shape=(2, 2),
        lambda_reuse=0.0, mu_neighbor=0.0, topk=4,
    )
    assert assignment[(0, 0)] == "tile_0.jpg"
    assert assignment[(0, 1)] == "tile_1.jpg"
    assert assignment[(1, 0)] == "tile_2.jpg"
    assert assignment[(1, 1)] == "tile_3.jpg"


def test_high_lambda_forces_unique_tiles():
    pool, _ = _synthetic_setup()
    # 4 cells all want tile_0 color, but λ should spread
    same_color = np.array([50, 80, 60], dtype=np.float32)
    cells = [
        {"row": 0, "col": 0, "lab_mean": same_color, "variance": 0.0},
        {"row": 0, "col": 1, "lab_mean": same_color, "variance": 0.0},
        {"row": 1, "col": 0, "lab_mean": same_color, "variance": 0.0},
        {"row": 1, "col": 1, "lab_mean": same_color, "variance": 0.0},
    ]
    assignment = solve_assignment(
        pool, cells, grid_shape=(2, 2),
        lambda_reuse=1e6, mu_neighbor=0.0, topk=4,
    )
    used = list(assignment.values())
    assert len(set(used)) == 4


def test_zero_lambda_allows_repetition():
    pool, _ = _synthetic_setup()
    same_color = np.array([50, 80, 60], dtype=np.float32)
    cells = [
        {"row": 0, "col": 0, "lab_mean": same_color, "variance": 0.0},
        {"row": 0, "col": 1, "lab_mean": same_color, "variance": 0.0},
    ]
    assignment = solve_assignment(
        pool, cells, grid_shape=(1, 2),
        lambda_reuse=0.0, mu_neighbor=0.0, topk=4,
    )
    # both should pick tile_0 (nearest color), no diversity pressure
    assert assignment[(0, 0)] == "tile_0.jpg"
    assert assignment[(0, 1)] == "tile_0.jpg"


def test_variance_ordering_picks_rare_color_first():
    """高方差格子先选，rare color 不会被平凡格抢走。"""
    # 2 tiles only: rare_color (1 copy) + common_color (easy match for both)
    rare = np.array([50, 100, 0], dtype=np.float32)
    common = np.array([50, 0, 0], dtype=np.float32)
    pool = {
        "rare.jpg": {"mtime": 0.0, "lab_mean": rare, "thumbnail": np.zeros((16,16,3), np.uint8)},
        "common.jpg": {"mtime": 0.0, "lab_mean": common, "thumbnail": np.zeros((16,16,3), np.uint8)},
    }
    # two cells: one exactly wants rare (variance high), one wants common (variance low)
    cells = [
        {"row": 0, "col": 0, "lab_mean": common, "variance": 0.1},
        {"row": 0, "col": 1, "lab_mean": rare, "variance": 99.0},
    ]
    assignment = solve_assignment(
        pool, cells, grid_shape=(1, 2),
        lambda_reuse=1e6, mu_neighbor=0.0, topk=2,
    )
    # high λ + variance-first means (0,1) claims rare.jpg first, (0,0) left with common
    assert assignment[(0, 1)] == "rare.jpg"
    assert assignment[(0, 0)] == "common.jpg"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd perplexity && python -m pytest tests/test_match.py -v`

Expected: 4 tests error with `ImportError`

- [ ] **Step 3: 实现 match.py**

Create `perplexity/mosaic/match.py`:
```python
"""贪心匹配求解器：faiss top-K 颜色候选 + λ 重复惩罚 + μ 邻居相似惩罚。

每格的 cost：
    cost(g, t) = ||LAB(g) - LAB(t)||^2
               + λ * log(1 + usage[t])
               + μ * Σ_{n in decided N4(g)} exp(-||LAB(t) - LAB(t_n)||^2 / σ^2)
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import faiss
import numpy as np
from tqdm import tqdm


def _neighbor_coords(r: int, c: int, rows: int, cols: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def solve_assignment(
    pool: dict,
    cells: list,
    grid_shape: tuple,
    lambda_reuse: float,
    mu_neighbor: float,
    topk: int = 64,
    neighbor_sigma: float = 20.0,
    semantic_reranker: Optional[Callable] = None,
    log_every: int = 100,
) -> dict:
    """返回 {(row, col): tile_path}。"""
    rows, cols = grid_shape
    tile_paths = list(pool.keys())
    if not tile_paths:
        raise ValueError("Empty pool")

    lab_matrix = np.stack([pool[p]["lab_mean"] for p in tile_paths]).astype(np.float32)
    index = faiss.IndexFlatL2(3)
    index.add(lab_matrix)

    actual_topk = min(topk, len(tile_paths))
    usage = np.zeros(len(tile_paths), dtype=np.int64)
    assignment: dict = {}

    # 方差降序扫描
    order = sorted(range(len(cells)), key=lambda i: -cells[i]["variance"])

    for step, cell_idx in enumerate(tqdm(order, desc="匹配中")):
        cell = cells[cell_idx]
        query = cell["lab_mean"].astype(np.float32).reshape(1, 3)
        color_dists, cand_indices = index.search(query, actual_topk)
        color_dists = color_dists[0]
        cand_indices = cand_indices[0]

        best_cost = math.inf
        best_tile_idx = cand_indices[0]

        for rank, ti in enumerate(cand_indices):
            if ti < 0:
                continue
            cost = float(color_dists[rank])
            cost += lambda_reuse * math.log(1.0 + int(usage[ti]))

            if mu_neighbor > 0.0:
                neigh_pen = 0.0
                for nr, nc in _neighbor_coords(cell["row"], cell["col"], rows, cols):
                    nkey = (nr, nc)
                    if nkey not in assignment:
                        continue
                    n_ti = tile_paths.index(assignment[nkey])
                    diff = lab_matrix[ti] - lab_matrix[n_ti]
                    neigh_pen += math.exp(-float(np.dot(diff, diff)) / (neighbor_sigma ** 2))
                cost += mu_neighbor * neigh_pen

            if semantic_reranker is not None:
                cost += float(semantic_reranker(cell, tile_paths[ti]))

            if cost < best_cost:
                best_cost = cost
                best_tile_idx = ti

        chosen_path = tile_paths[best_tile_idx]
        assignment[(cell["row"], cell["col"])] = chosen_path
        usage[best_tile_idx] += 1

        if step % log_every == 0:
            print(
                f"  ({cell['row']:3d},{cell['col']:3d}) → "
                f"{chosen_path.split('/')[-1]} | "
                f"color={color_dists[np.where(cand_indices == best_tile_idx)[0][0]]:.2f}, "
                f"used={int(usage[best_tile_idx])}x"
            )

    return assignment
```

- [ ] **Step 4: 跑测试确认过**

Run: `cd perplexity && python -m pytest tests/test_match.py -v`

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add perplexity/mosaic/match.py perplexity/tests/test_match.py
git commit -m "feat(match): faiss top-K + lambda reuse + mu neighbor greedy solver"
```

---

## Task 4: transfer.py — Reinhard LAB 色调迁移

**Files:**
- Create: `perplexity/mosaic/transfer.py`
- Create: `perplexity/tests/test_transfer.py`

- [ ] **Step 1: 写 test_transfer.py 失败测试**

Create `perplexity/tests/test_transfer.py`:
```python
import numpy as np

from mosaic.transfer import reinhard_transfer


def test_tau_zero_returns_identity():
    tile = np.random.RandomState(0).randint(0, 255, (64, 64, 3), dtype=np.uint8)
    target_patch = np.full((64, 64, 3), 128, dtype=np.uint8)
    out = reinhard_transfer(tile, target_patch, tau=0.0)
    assert np.array_equal(out, tile)


def test_tau_one_matches_target_mean_in_lab():
    from skimage.color import rgb2lab
    tile = np.full((64, 64, 3), 50, dtype=np.uint8)
    target_patch = np.full((64, 64, 3), 200, dtype=np.uint8)
    out = reinhard_transfer(tile, target_patch, tau=1.0)
    out_lab = rgb2lab(out / 255.0).reshape(-1, 3).mean(axis=0)
    target_lab = rgb2lab(target_patch / 255.0).reshape(-1, 3).mean(axis=0)
    # L channel should be close (a,b can stay at 0 since both inputs are gray)
    assert abs(out_lab[0] - target_lab[0]) < 2.0


def test_tau_half_between_endpoints():
    tile = np.full((64, 64, 3), 50, dtype=np.uint8)
    target_patch = np.full((64, 64, 3), 200, dtype=np.uint8)
    out_0 = reinhard_transfer(tile, target_patch, tau=0.0)
    out_1 = reinhard_transfer(tile, target_patch, tau=1.0)
    out_h = reinhard_transfer(tile, target_patch, tau=0.5)
    assert out_0.mean() < out_h.mean() < out_1.mean()


def test_zero_std_channel_does_not_divide_by_zero():
    """纯色 tile σ=0，不该崩溃。"""
    tile = np.full((64, 64, 3), 100, dtype=np.uint8)
    target_patch = np.random.RandomState(0).randint(0, 255, (64, 64, 3), dtype=np.uint8)
    out = reinhard_transfer(tile, target_patch, tau=1.0)
    assert out.shape == tile.shape
    assert np.isfinite(out).all()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd perplexity && python -m pytest tests/test_transfer.py -v`

Expected: 4 tests error with `ImportError`

- [ ] **Step 3: 实现 transfer.py**

Create `perplexity/mosaic/transfer.py`:
```python
"""Reinhard LAB 色调迁移：把 tile 的 (μ, σ) 搬到 target_patch 的 (μ, σ)，按 τ 线性混合。"""
from __future__ import annotations

import numpy as np
from skimage.color import lab2rgb, rgb2lab

EPS = 1e-6


def reinhard_transfer(tile_rgb: np.ndarray, target_patch_rgb: np.ndarray, tau: float) -> np.ndarray:
    """
    tile_rgb, target_patch_rgb: HxWx3 uint8
    tau ∈ [0, 1]
    返回 HxWx3 uint8
    """
    if tau <= 0.0:
        return tile_rgb.copy()

    tile_lab = rgb2lab(tile_rgb / 255.0)
    target_lab = rgb2lab(target_patch_rgb / 255.0)

    tile_mean = tile_lab.reshape(-1, 3).mean(axis=0)
    tile_std = tile_lab.reshape(-1, 3).std(axis=0)
    target_mean = target_lab.reshape(-1, 3).mean(axis=0)
    target_std = target_lab.reshape(-1, 3).std(axis=0)

    scale = target_std / np.maximum(tile_std, EPS)
    transferred_lab = (tile_lab - tile_mean) * scale + target_mean

    mixed_lab = (1.0 - tau) * tile_lab + tau * transferred_lab
    mixed_rgb = np.clip(lab2rgb(mixed_lab) * 255.0, 0, 255).astype(np.uint8)
    return mixed_rgb
```

- [ ] **Step 4: 跑测试确认过**

Run: `cd perplexity && python -m pytest tests/test_transfer.py -v`

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add perplexity/mosaic/transfer.py perplexity/tests/test_transfer.py
git commit -m "feat(transfer): Reinhard LAB tonal transfer with tau interpolation"
```

---

## Task 5: render.py — 贴图 + 使用统计

**Files:**
- Create: `perplexity/mosaic/render.py`
- Create: `perplexity/tests/test_render.py`

- [ ] **Step 1: 写 test_render.py 失败测试**

Create `perplexity/tests/test_render.py`:
```python
import numpy as np
from PIL import Image

from mosaic.render import render_mosaic


def test_render_outputs_correct_dimensions(tmp_pool_dir, tmp_target_img):
    from mosaic.pool import scan_pool
    from mosaic.target import load_and_grid

    pool = scan_pool(tmp_pool_dir, tmp_target_img.parent / "cache.pkl")
    grid = load_and_grid(tmp_target_img, grid_cols=4, grid_rows=4)

    first_tile = next(iter(pool.keys()))
    assignment = {(c["row"], c["col"]): first_tile for c in grid["cells"]}

    result = render_mosaic(assignment, pool, grid, tile_px=32, tau=0.0)

    assert result["image"].size == (128, 128)  # 4 cols * 32 px
    assert result["usage"][first_tile] == 16
    for path in pool:
        if path != first_tile:
            assert result["usage"][path] == 0


def test_render_applies_tau_transfer(tmp_pool_dir, tmp_target_img, monkeypatch):
    from mosaic.pool import scan_pool
    from mosaic.target import load_and_grid

    calls = []

    import mosaic.render
    original = mosaic.render.reinhard_transfer

    def spy(tile, patch, tau):
        calls.append(tau)
        return original(tile, patch, tau)

    monkeypatch.setattr(mosaic.render, "reinhard_transfer", spy)

    pool = scan_pool(tmp_pool_dir, tmp_target_img.parent / "cache.pkl")
    grid = load_and_grid(tmp_target_img, grid_cols=2, grid_rows=2)
    first_tile = next(iter(pool.keys()))
    assignment = {(c["row"], c["col"]): first_tile for c in grid["cells"]}

    render_mosaic(assignment, pool, grid, tile_px=32, tau=0.75)

    assert len(calls) == 4
    assert all(c == 0.75 for c in calls)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd perplexity && python -m pytest tests/test_render.py -v`

Expected: 2 tests error with `ImportError`

- [ ] **Step 3: 实现 render.py**

Create `perplexity/mosaic/render.py`:
```python
"""组装最终 mosaic：对每格 tile 做色调迁移后贴到画布，累计 usage。"""
from __future__ import annotations

from collections import Counter

import numpy as np
from PIL import Image

from mosaic.transfer import reinhard_transfer


def render_mosaic(assignment: dict, pool: dict, grid: dict, tile_px: int, tau: float) -> dict:
    """
    assignment: {(row, col): tile_path}
    pool: {tile_path: {lab_mean, thumbnail, mtime}}
    grid: {'shape': (rows, cols), 'cells': [...], 'image': PIL.Image, 'cell_size': (h, w)}
    返回 {'image': PIL.Image (RGB), 'usage': Counter[tile_path]}
    """
    rows, cols = grid["shape"]
    target_image = grid["image"]
    target_h, target_w = target_image.size[1], target_image.size[0]
    cell_h = target_h // rows
    cell_w = target_w // cols

    out_h = tile_px * rows
    out_w = tile_px * cols
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    target_arr = np.asarray(target_image, dtype=np.uint8)
    usage: Counter = Counter()

    tile_cache = {}
    for r in range(rows):
        for c in range(cols):
            tile_path = assignment[(r, c)]
            if tile_path not in tile_cache:
                img = Image.open(tile_path).convert("RGB")
                tile_cache[tile_path] = np.asarray(
                    img.resize((tile_px, tile_px), Image.LANCZOS), dtype=np.uint8
                )
            tile_arr = tile_cache[tile_path]

            target_patch = target_arr[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            if target_patch.shape[0] != tile_px or target_patch.shape[1] != tile_px:
                patch_img = Image.fromarray(target_patch).resize((tile_px, tile_px), Image.LANCZOS)
                target_patch = np.asarray(patch_img, dtype=np.uint8)

            transferred = reinhard_transfer(tile_arr, target_patch, tau=tau)
            canvas[r * tile_px : (r + 1) * tile_px, c * tile_px : (c + 1) * tile_px] = transferred
            usage[tile_path] += 1

    return {"image": Image.fromarray(canvas), "usage": usage}
```

- [ ] **Step 4: 跑测试确认过**

Run: `cd perplexity && python -m pytest tests/test_render.py -v`

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add perplexity/mosaic/render.py perplexity/tests/test_render.py
git commit -m "feat(render): paste tiles with tau transfer, return image + usage counter"
```

---

## Task 6: report.py — 自嘲式文字 + 柱状图 + 冷宫墙

**Files:**
- Create: `perplexity/mosaic/report.py`
- Create: `perplexity/tests/test_report.py`

- [ ] **Step 1: 写 test_report.py 失败测试**

Create `perplexity/tests/test_report.py`:
```python
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from mosaic.report import generate_report, _region_label


def test_region_label_top_means_sky():
    pos = [(0, 5), (0, 10), (1, 7)]
    assert _region_label(pos, rows=10, cols=20) == "天空"


def test_region_label_bottom_means_ground():
    pos = [(9, 5), (8, 10)]
    assert _region_label(pos, rows=10, cols=20) == "地面"


def test_region_label_center_means_subject():
    pos = [(5, 10), (5, 11), (4, 10)]
    assert _region_label(pos, rows=10, cols=20) == "主体"


def test_report_writes_markdown_with_stats(tmp_path):
    pool = {
        f"p{i}.jpg": {
            "mtime": 0.0,
            "lab_mean": np.array([50.0, 0.0, 0.0], dtype=np.float32),
            "thumbnail": np.full((16, 16, 3), i * 20, dtype=np.uint8),
        }
        for i in range(10)
    }
    usage = Counter({"p0.jpg": 50, "p1.jpg": 30, "p2.jpg": 5})
    positions = {"p0.jpg": [(0, 0), (0, 1)], "p1.jpg": [(5, 5)], "p2.jpg": [(9, 9)]}

    out_path = tmp_path / "report.md"
    chart_path = tmp_path / "chart.png"
    cold_path = tmp_path / "cold.png"

    generate_report(
        pool=pool, usage=usage, positions=positions, grid_shape=(10, 10),
        output_md=out_path, output_chart=chart_path, output_cold=cold_path,
    )

    md = out_path.read_text()
    assert "10" in md  # pool total
    assert "3" in md   # used count
    assert "p0.jpg" in md  # top tile
    assert chart_path.exists()
    assert cold_path.exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd perplexity && python -m pytest tests/test_report.py -v`

Expected: 4 tests error with `ImportError`

- [ ] **Step 3: 实现 report.py**

Create `perplexity/mosaic/report.py`:
```python
"""生成自嘲式报告：文字 + 使用次数柱状图 + 冷宫照片墙。"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _region_label(positions: list, rows: int, cols: int) -> str:
    if not positions:
        return "未知"
    ys = [p[0] / max(rows - 1, 1) for p in positions]
    xs = [p[1] / max(cols - 1, 1) for p in positions]
    y_mean = sum(ys) / len(ys)
    x_mean = sum(xs) / len(xs)
    if y_mean < 0.33:
        return "天空"
    if y_mean > 0.67:
        return "地面"
    if abs(x_mean - 0.5) < 0.2:
        return "主体"
    return "填充"


def _save_usage_chart(usage: Counter, output: Path) -> None:
    top = usage.most_common(30)
    if not top:
        Image.new("RGB", (600, 200), (240, 240, 240)).save(output)
        return
    labels = [Path(name).name[:20] for name, _ in top]
    values = [v for _, v in top]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(top)), values, color="#4c72b0")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("使用次数")
    ax.set_title("Top 30 被使用最多的 tile")
    fig.tight_layout()
    fig.savefig(output, dpi=100)
    plt.close(fig)


def _save_cold_wall(pool: dict, usage: Counter, output: Path, n_cols: int = 5, n_rows: int = 4) -> None:
    cold = [path for path in pool if usage.get(path, 0) == 0]
    if not cold:
        Image.new("RGB", (320, 256), (240, 240, 240)).save(output)
        return
    picks = random.Random(42).sample(cold, min(n_cols * n_rows, len(cold)))
    tile_h, tile_w = 64, 64
    canvas = np.full((tile_h * n_rows, tile_w * n_cols, 3), 230, dtype=np.uint8)
    for i, path in enumerate(picks):
        r, c = divmod(i, n_cols)
        thumb = pool[path]["thumbnail"]
        thumb_img = Image.fromarray(thumb).resize((tile_w, tile_h), Image.LANCZOS)
        canvas[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = np.asarray(thumb_img)
    Image.fromarray(canvas).save(output)


def generate_report(
    pool: dict,
    usage: Counter,
    positions: dict,
    grid_shape: tuple,
    output_md: Path,
    output_chart: Path,
    output_cold: Path,
) -> str:
    rows, cols = grid_shape
    pool_total = len(pool)
    used_count = sum(1 for v in usage.values() if v > 0)
    cold_count = pool_total - used_count

    if usage:
        top_tile_path, top_count = usage.most_common(1)[0]
        top_tile_name = Path(top_tile_path).name
        top_region_guess = _region_label(positions.get(top_tile_path, []), rows, cols)
    else:
        top_tile_name, top_count, top_region_guess = "(none)", 0, "无"

    cold_picks = [p for p in pool if usage.get(p, 0) == 0][:5]
    cold_names = ", ".join(Path(p).name for p in cold_picks) if cold_picks else "(none)"

    _save_usage_chart(usage, output_chart)
    _save_cold_wall(pool, usage, output_cold, n_cols=5, n_rows=4)

    text = f"""# Photomosaic 报告

本次使用了你 **{pool_total}** 张照片里的 **{used_count}** 张（冷宫 {cold_count} 张）。

其中 `{top_tile_name}` 被用了 **{top_count}** 次（主要用于填充**{top_region_guess}**）。

冷宫照片 TOP 5 是：{cold_names}（都是你的废片）。

![使用次数柱状图]({output_chart.name})

![冷宫照片墙]({output_cold.name})
"""
    output_md.write_text(text, encoding="utf-8")
    return text
```

- [ ] **Step 4: 跑测试确认过**

Run: `cd perplexity && python -m pytest tests/test_report.py -v`

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add perplexity/mosaic/report.py perplexity/tests/test_report.py
git commit -m "feat(report): self-deprecating markdown + usage bar chart + cold-storage wall"
```

---

## Task 7: zoom.py — DeepZoom + OpenSeadragon HTML

**Files:**
- Create: `perplexity/mosaic/zoom.py`
- Create: `perplexity/tests/test_zoom.py`

- [ ] **Step 1: 写 test_zoom.py 失败测试**

Create `perplexity/tests/test_zoom.py`:
```python
from pathlib import Path

import numpy as np
from PIL import Image

from mosaic.zoom import export_deepzoom


def test_deepzoom_produces_html_and_dzi(tmp_path):
    img = Image.fromarray(np.random.RandomState(0).randint(0, 255, (512, 512, 3), dtype=np.uint8))
    png_path = tmp_path / "mosaic.png"
    img.save(png_path)

    out_dir = tmp_path / "zoom"
    result = export_deepzoom(png_path, out_dir)

    assert result["html"].exists()
    assert result["dzi"].exists()
    html = result["html"].read_text()
    assert "openseadragon" in html.lower()
    assert result["dzi"].name in html


def test_deepzoom_overwrites_existing(tmp_path):
    img = Image.fromarray(np.random.RandomState(0).randint(0, 255, (512, 512, 3), dtype=np.uint8))
    png_path = tmp_path / "mosaic.png"
    img.save(png_path)
    out_dir = tmp_path / "zoom"

    export_deepzoom(png_path, out_dir)
    result = export_deepzoom(png_path, out_dir)
    assert result["html"].exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd perplexity && python -m pytest tests/test_zoom.py -v`

Expected: 2 tests error with `ImportError`

- [ ] **Step 3: 实现 zoom.py**

Create `perplexity/mosaic/zoom.py`:
```python
"""DeepZoom 金字塔切片 + 自包含 OpenSeadragon HTML 查看器。"""
from __future__ import annotations

import shutil
from pathlib import Path

import deepzoom

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Photomosaic</title>
<style>html,body{{margin:0;padding:0;background:#111;}}#viewer{{width:100vw;height:100vh;}}</style>
<script src="https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/openseadragon.min.js"></script>
</head>
<body>
<div id="viewer"></div>
<script>
OpenSeadragon({{
  id: "viewer",
  tileSources: "{dzi_name}",
  prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/images/",
  showNavigator: true,
  animationTime: 0.5,
  maxZoomPixelRatio: 8
}});
</script>
</body>
</html>
"""


def export_deepzoom(png_path: Path, output_dir: Path) -> dict:
    png_path = Path(png_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    dzi_path = output_dir / "mosaic.dzi"
    creator = deepzoom.ImageCreator(tile_size=254, tile_overlap=1, tile_format="jpg", image_quality=0.9)
    creator.create(str(png_path), str(dzi_path))

    html_path = output_dir / "index.html"
    html_path.write_text(HTML_TEMPLATE.format(dzi_name=dzi_path.name), encoding="utf-8")

    return {"html": html_path, "dzi": dzi_path}
```

- [ ] **Step 4: 跑测试确认过**

Run: `cd perplexity && python -m pytest tests/test_zoom.py -v`

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add perplexity/mosaic/zoom.py perplexity/tests/test_zoom.py
git commit -m "feat(zoom): DeepZoom tiles + OpenSeadragon HTML viewer"
```

---

## Task 8: config.py — 默认参数 + ipywidgets UI 工厂

**Files:**
- Create: `perplexity/mosaic/config.py`
- Create: `perplexity/tests/test_config.py`

- [ ] **Step 1: 写 test_config.py 失败测试**

Create `perplexity/tests/test_config.py`:
```python
from mosaic.config import DEFAULT_CONFIG, build_widgets


def test_default_config_has_expected_keys():
    required = {
        "target_path", "pool_dir", "grid_cols", "grid_rows", "tile_px",
        "preview_grid_cols", "preview_grid_rows", "preview_tile_px",
        "lambda_reuse", "mu_neighbor", "tau_transfer",
        "topk_candidates", "neighbor_sigma", "cache_dir", "output_dir", "seed",
    }
    assert required.issubset(DEFAULT_CONFIG.keys())


def test_default_grid_is_16_9_ish():
    assert DEFAULT_CONFIG["grid_cols"] == 120
    assert DEFAULT_CONFIG["grid_rows"] == 68


def test_build_widgets_returns_container_and_values():
    result = build_widgets()
    assert "container" in result
    assert "get_params" in result
    params = result["get_params"]()
    assert set(params.keys()) == {"lambda_reuse", "mu_neighbor", "tau_transfer"}
    assert params["lambda_reuse"] == DEFAULT_CONFIG["lambda_reuse"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd perplexity && python -m pytest tests/test_config.py -v`

Expected: 3 tests error with `ImportError`

- [ ] **Step 3: 实现 config.py**

Create `perplexity/mosaic/config.py`:
```python
"""默认参数 + ipywidgets 交互面板工厂（三滑条 λ/μ/τ）。"""
from __future__ import annotations

import ipywidgets as widgets

DEFAULT_CONFIG = {
    "target_path": "target.jpg",
    "pool_dir": "pool/",
    "grid_cols": 120,
    "grid_rows": 68,
    "tile_px": 16,
    "preview_grid_cols": 48,
    "preview_grid_rows": 27,
    "preview_tile_px": 6,
    "lambda_reuse": 1.0,
    "mu_neighbor": 0.5,
    "tau_transfer": 0.5,
    "topk_candidates": 64,
    "neighbor_sigma": 20.0,
    "cache_dir": ".cache",
    "output_dir": "output",
    "seed": 42,
}


def build_widgets() -> dict:
    lam = widgets.FloatSlider(
        value=DEFAULT_CONFIG["lambda_reuse"], min=0.0, max=5.0, step=0.1,
        description="λ reuse:", style={"description_width": "initial"},
    )
    mu = widgets.FloatSlider(
        value=DEFAULT_CONFIG["mu_neighbor"], min=0.0, max=5.0, step=0.1,
        description="μ neighbor:", style={"description_width": "initial"},
    )
    tau = widgets.FloatSlider(
        value=DEFAULT_CONFIG["tau_transfer"], min=0.0, max=1.0, step=0.05,
        description="τ transfer:", style={"description_width": "initial"},
    )
    preview_btn = widgets.Button(description="预览 48×27", button_style="info")
    render_btn = widgets.Button(description="正式渲染 120×68", button_style="primary")

    container = widgets.VBox([
        widgets.HTML("<h4>参数</h4>"), lam, mu, tau,
        widgets.HBox([preview_btn, render_btn]),
    ])

    def get_params() -> dict:
        return {"lambda_reuse": lam.value, "mu_neighbor": mu.value, "tau_transfer": tau.value}

    return {
        "container": container,
        "get_params": get_params,
        "preview_button": preview_btn,
        "render_button": render_btn,
    }
```

- [ ] **Step 4: 跑测试确认过**

Run: `cd perplexity && python -m pytest tests/test_config.py -v`

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add perplexity/mosaic/config.py perplexity/tests/test_config.py
git commit -m "feat(config): DEFAULT_CONFIG + ipywidgets three-slider factory"
```

---

## Task 9: pipeline.py — 把所有模块串成可复用的 run() 函数

**Files:**
- Create: `perplexity/mosaic/pipeline.py`
- Create: `perplexity/tests/test_pipeline.py`

**理由**: 如果把整条 pipeline 写在 notebook cell 里，没法跑端到端测试。单独抽一个 `run_full()` / `run_preview()` 函数，notebook 和 test 都调它。

- [ ] **Step 1: 写 test_pipeline.py 失败测试**

Create `perplexity/tests/test_pipeline.py`:
```python
from pathlib import Path

from mosaic.pipeline import run_pipeline


def test_preview_pipeline_produces_image(tmp_pool_dir, tmp_target_img, tmp_path):
    result = run_pipeline(
        target_path=tmp_target_img,
        pool_dir=tmp_pool_dir,
        grid_cols=4, grid_rows=4,
        tile_px=16,
        lambda_reuse=1.0, mu_neighbor=0.5, tau_transfer=0.5,
        cache_path=tmp_path / "cache.pkl",
        output_dir=tmp_path / "out",
        do_deepzoom=False,
        do_report=False,
    )
    assert result["image"].size == (64, 64)
    assert sum(result["usage"].values()) == 16


def test_full_pipeline_produces_all_artifacts(tmp_pool_dir, tmp_target_img, tmp_path):
    out_dir = tmp_path / "out"
    result = run_pipeline(
        target_path=tmp_target_img,
        pool_dir=tmp_pool_dir,
        grid_cols=4, grid_rows=4,
        tile_px=16,
        lambda_reuse=1.0, mu_neighbor=0.5, tau_transfer=0.5,
        cache_path=tmp_path / "cache.pkl",
        output_dir=out_dir,
        do_deepzoom=True,
        do_report=True,
    )
    assert result["png_path"].exists()
    assert result["report_path"].exists()
    assert result["chart_path"].exists()
    assert result["cold_path"].exists()
    assert result["deepzoom_html"].exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd perplexity && python -m pytest tests/test_pipeline.py -v`

Expected: 2 tests error with `ImportError`

- [ ] **Step 3: 实现 pipeline.py**

Create `perplexity/mosaic/pipeline.py`:
```python
"""端到端 pipeline：scan pool → grid target → solve → render → (report + deepzoom)。"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

from mosaic.match import solve_assignment
from mosaic.pool import scan_pool
from mosaic.render import render_mosaic
from mosaic.report import generate_report
from mosaic.target import load_and_grid
from mosaic.zoom import export_deepzoom


def run_pipeline(
    target_path: Path,
    pool_dir: Path,
    grid_cols: int,
    grid_rows: int,
    tile_px: int,
    lambda_reuse: float,
    mu_neighbor: float,
    tau_transfer: float,
    cache_path: Path,
    output_dir: Path,
    topk_candidates: int = 64,
    neighbor_sigma: float = 20.0,
    do_deepzoom: bool = True,
    do_report: bool = True,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    pool = scan_pool(Path(pool_dir), Path(cache_path))
    grid = load_and_grid(Path(target_path), grid_cols=grid_cols, grid_rows=grid_rows)

    assignment = solve_assignment(
        pool=pool, cells=grid["cells"], grid_shape=grid["shape"],
        lambda_reuse=lambda_reuse, mu_neighbor=mu_neighbor,
        topk=topk_candidates, neighbor_sigma=neighbor_sigma,
    )
    rendered = render_mosaic(assignment, pool, grid, tile_px=tile_px, tau=tau_transfer)

    png_path = output_dir / f"mosaic_{ts}.png"
    rendered["image"].save(png_path)
    result = {"image": rendered["image"], "usage": rendered["usage"], "png_path": png_path}

    if do_report:
        positions = defaultdict(list)
        for (r, c), tile_path in assignment.items():
            positions[tile_path].append((r, c))
        report_path = output_dir / f"report_{ts}.md"
        chart_path = output_dir / f"chart_{ts}.png"
        cold_path = output_dir / f"cold_{ts}.png"
        generate_report(
            pool=pool, usage=rendered["usage"], positions=dict(positions),
            grid_shape=grid["shape"],
            output_md=report_path, output_chart=chart_path, output_cold=cold_path,
        )
        result.update({"report_path": report_path, "chart_path": chart_path, "cold_path": cold_path})

    if do_deepzoom:
        dzi_dir = output_dir / f"deepzoom_{ts}"
        dzi = export_deepzoom(png_path, dzi_dir)
        result["deepzoom_html"] = dzi["html"]
        result["deepzoom_dir"] = dzi_dir

    return result
```

> 注意：参数名用 `do_report` / `do_deepzoom`，避免和同模块 import 的 `generate_report` / `export_deepzoom` 函数名 shadow。

- [ ] **Step 4: 跑测试确认过**

Run: `cd perplexity && python -m pytest tests/test_pipeline.py -v`

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add perplexity/mosaic/pipeline.py perplexity/tests/test_pipeline.py
git commit -m "feat(pipeline): end-to-end run_pipeline orchestrating all modules"
```

---

## Task 10: mosaic.ipynb — 8-cell 入口

**Files:**
- Create: `perplexity/mosaic.ipynb`

**Goal**: 一个能直接 Run All 跑起来的 notebook。

- [ ] **Step 1: 创建 notebook 结构**

Create `perplexity/mosaic.ipynb` as JSON. Each cell's content is listed below. Use Python to generate the `.ipynb` file:

Run this in terminal:
```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/perplexity
python <<'PY'
import json
from pathlib import Path

cells = [
    # Cell 1: markdown intro
    {"cell_type": "markdown", "metadata": {}, "source": [
        "# 📷 Photomosaic 生成器\n",
        "\n",
        "本地 Mac 玩具。把 `pool/` 里的一堆照片重组成 `target.jpg`。\n",
        "\n",
        "**用法**: Run All → 看 Cell 2 里的滑条 → 点预览 / 正式渲染。"
    ]},
    # Cell 2: imports
    {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [
        "from pathlib import Path\n",
        "from IPython.display import display, Image as IPImage, Markdown\n",
        "import ipywidgets as widgets\n",
        "\n",
        "from mosaic.config import DEFAULT_CONFIG, build_widgets\n",
        "from mosaic.pipeline import run_pipeline"
    ]},
    # Cell 3: config
    {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [
        "CFG = dict(DEFAULT_CONFIG)\n",
        "# 改下面三项指向你的实际路径\n",
        "CFG['target_path'] = 'target.jpg'\n",
        "CFG['pool_dir'] = 'pool/'\n",
        "CFG['cache_dir'] = '.cache'\n",
        "CFG['output_dir'] = 'output'\n",
        "\n",
        "Path(CFG['cache_dir']).mkdir(exist_ok=True)\n",
        "Path(CFG['output_dir']).mkdir(exist_ok=True)\n",
        "print('config:', CFG)"
    ]},
    # Cell 4: widgets + button handlers
    {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [
        "ui = build_widgets()\n",
        "output = widgets.Output()\n",
        "\n",
        "def _run(is_preview):\n",
        "    params = ui['get_params']()\n",
        "    with output:\n",
        "        output.clear_output()\n",
        "        grid_cols = CFG['preview_grid_cols'] if is_preview else CFG['grid_cols']\n",
        "        grid_rows = CFG['preview_grid_rows'] if is_preview else CFG['grid_rows']\n",
        "        tile_px = CFG['preview_tile_px'] if is_preview else CFG['tile_px']\n",
        "        print(f\"{'预览' if is_preview else '正式'}: {grid_cols}x{grid_rows} @ {tile_px}px\")\n",
        "        result = run_pipeline(\n",
        "            target_path=CFG['target_path'], pool_dir=CFG['pool_dir'],\n",
        "            grid_cols=grid_cols, grid_rows=grid_rows, tile_px=tile_px,\n",
        "            lambda_reuse=params['lambda_reuse'], mu_neighbor=params['mu_neighbor'],\n",
        "            tau_transfer=params['tau_transfer'],\n",
        "            cache_path=Path(CFG['cache_dir']) / 'pool_features.pkl',\n",
        "            output_dir=CFG['output_dir'],\n",
        "            topk_candidates=CFG['topk_candidates'],\n",
        "            neighbor_sigma=CFG['neighbor_sigma'],\n",
        "            do_deepzoom=not is_preview,\n",
        "            do_report=not is_preview,\n",
        "        )\n",
        "        display(IPImage(str(result['png_path'])))\n",
        "        if not is_preview:\n",
        "            display(Markdown(result['report_path'].read_text()))\n",
        "            print(f\"DeepZoom: {result['deepzoom_html']}\")\n",
        "\n",
        "ui['preview_button'].on_click(lambda _: _run(is_preview=True))\n",
        "ui['render_button'].on_click(lambda _: _run(is_preview=False))\n",
        "display(ui['container'], output)"
    ]},
    # Cell 5: markdown footer with tips
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## 使用提示\n",
        "\n",
        "- `λ` 大 → tile 不重复；`λ=0` → 谁近用谁\n",
        "- `μ` 大 → 相邻 tile 会被逼得不一样\n",
        "- `τ=0` 近看照片清晰，`τ=1` 远看完美。甜区 `[0.4, 0.6]`\n",
        "- 正式渲染会生成 PNG + Markdown 报告 + DeepZoom HTML，在 `output/`"
    ]},
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

Path("mosaic.ipynb").write_text(json.dumps(nb, indent=1))
print("Wrote mosaic.ipynb")
PY
```

- [ ] **Step 2: smoke test notebook**

准备最小 fixtures 来端到端试一次：

Run:
```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/perplexity
python <<'PY'
import numpy as np
from PIL import Image
from pathlib import Path

Path('pool').mkdir(exist_ok=True)
for i in range(16):
    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    Image.new('RGB', (64, 64), color).save(f'pool/tile_{i}.jpg')

arr = np.random.RandomState(0).randint(0, 255, (256, 256, 3), dtype=np.uint8)
Image.fromarray(arr).save('target.jpg')

from mosaic.pipeline import run_pipeline
r = run_pipeline(
    target_path='target.jpg', pool_dir='pool/',
    grid_cols=8, grid_rows=8, tile_px=16,
    lambda_reuse=1.0, mu_neighbor=0.5, tau_transfer=0.5,
    cache_path=Path('.cache/test.pkl'),
    output_dir='output',
    do_deepzoom=True, do_report=True,
)
print('OK:', r['png_path'], r['report_path'], r['deepzoom_html'])
PY
```

Expected: 打印 `OK: output/mosaic_*.png output/report_*.md output/deepzoom_*/index.html`，所有文件实际存在。

- [ ] **Step 3: 清掉 smoke test 产物（它们应被 gitignore）**

Run:
```bash
cd perplexity
rm -rf pool/ target.jpg .cache/ output/
git status  # 应只看见 mosaic.ipynb 待提交
```

- [ ] **Step 4: jupyter 启动检查**

Run:
```bash
cd perplexity
jupyter nbconvert --to notebook --execute mosaic.ipynb --output mosaic_smoke.ipynb 2>&1 | head -30 || true
```

**注意**: 这一步会失败（因为 `pool/` 和 `target.jpg` 被删了）。我们**只需验证 notebook JSON 格式合法** — 能打开、nbconvert 不会因语法崩溃。如果是因为 FileNotFoundError 失败，算过。其它错误（SyntaxError/KernelError）需要修。

- [ ] **Step 5: Commit**

```bash
cd perplexity
rm -f mosaic_smoke.ipynb
git add mosaic.ipynb
git commit -m "feat(notebook): 5-cell ipynb entry with widgets wiring preview/render"
```

---

## Task 11: README 使用说明

**Files:**
- Create: `perplexity/README.md`

- [ ] **Step 1: 写 README.md**

Create `perplexity/README.md`:
```markdown
# Photomosaic 生成器

本地 Mac 玩具：把硬盘里一堆照片重组成另一张照片。

## Setup

```bash
cd perplexity
pip install -r requirements.txt
jupyter notebook mosaic.ipynb
```

## 使用

1. 把一堆照片放到 `perplexity/pool/` 目录（递归扫描，支持 jpg/png/webp/bmp）
2. 把目标图放到 `perplexity/target.jpg`
3. 打开 `mosaic.ipynb`，Run All
4. 看 Cell 4 的滑条：
   - **λ reuse**: tile 重复惩罚。大 → 每张 tile 最多用一次；0 → 谁近用谁
   - **μ neighbor**: 邻居相似惩罚。大 → 相邻 tile 颜色必须差异大
   - **τ transfer**: 色调迁移强度。0 → 照片原色（近看清晰），1 → 贴合目标（远看完美）
5. 点"预览 48×27"看效果，满意后点"正式渲染 120×68"

## 产物

- `output/mosaic_<ts>.png` — 成品图
- `output/report_<ts>.md` — 自嘲式报告（哪张 tile 被用最多、哪些冷宫）
- `output/deepzoom_<ts>/index.html` — 用浏览器双击打开，能无限放大

## 性能

第一次扫底图池要算 LAB 均值，`.cache/pool_features.pkl` 保存。之后按 mtime 增量。
3K 张池 + 120×68 网格 ≈ 10-30 秒完整渲染。

## 砍掉的东西（后续 V2/V3）

- CLIP 语义匹配（有 `semantic_reranker` 钩子预留）
- 底图打标签 + 叙事报告（"23% 来自 2019 日本旅行"）
- Cursed Mode 预设
- 实时"看它思考"动画
- 主体 saliency mask

## 开发

```bash
pytest tests/ -v
```
```

- [ ] **Step 2: Commit**

```bash
git add perplexity/README.md
git commit -m "docs: README with setup, usage, outputs, V2/V3 roadmap"
```

---

## Task 12: CHANGELOG 初始化

**Files:**
- Create: `perplexity/CHANGELOG.md`

> **注意**：本项目的 CHANGELOG 按 memory 中的 `feedback_changelog_for_agents.md` 规范维护——刻意啰嗦、保留 try-failed 链条、达到 50 条/6 个月才压缩。

- [ ] **Step 1: 写初始 CHANGELOG.md**

Create `perplexity/CHANGELOG.md`:
```markdown
# CHANGELOG

> 本 changelog 的主要读者是 agent，不是人。规则见 `~/.claude/...` memory。
> 要求刻意啰嗦、保留 try-failed 链条、ISO 日期。

## 活跃条目

- date: 2026-04-17
  type: feat
  target: perplexity/mosaic/ (whole package)
  change: 初始实现 photomosaic 生成器 MVP
  rationale: 按 spec docs/superpowers/specs/2026-04-17-photomosaic-ipynb-design.md 的 MVP 范围实现；定位是本地 Mac 玩具，速度/普适性/稳定性都不追求，可解释性优先
  action: |
    - mosaic/pool.py: 扫描底图 + LAB 均值 + pickle 缓存（mtime 增量）
    - mosaic/target.py: 读目标图 + center-crop + 网格分割
    - mosaic/match.py: faiss top-K + λ 重复惩罚 + μ 邻居惩罚 + 贪心求解（方差降序）
    - mosaic/transfer.py: Reinhard LAB 色调迁移（τ 线性混合）
    - mosaic/render.py: 贴图 + usage 计数
    - mosaic/report.py: 自嘲式文字 + top-30 柱状图 + 冷宫墙
    - mosaic/zoom.py: DeepZoom + OpenSeadragon HTML
    - mosaic/config.py: DEFAULT_CONFIG + ipywidgets 三滑条工厂
    - mosaic/pipeline.py: run_pipeline 端到端（便于 notebook 和 test 复用）
    - mosaic.ipynb: 5-cell 入口（Run All 即用）
  result: 本地 smoke test 通过：16 张 tile + 8×8 网格 → PNG + report + deepzoom html 都生成
  validation: tests/test_pool.py (6), test_target.py (4), test_match.py (4), test_transfer.py (4), test_render.py (2), test_report.py (4), test_zoom.py (2), test_config.py (3), test_pipeline.py (2) 共 31 tests 全过
  status: stable
```

- [ ] **Step 2: Commit**

```bash
git add perplexity/CHANGELOG.md
git commit -m "docs: initial CHANGELOG following agent-oriented format"
```

---

## 全局 smoke check（最后一步）

- [ ] **Run: `cd perplexity && python -m pytest tests/ -v`**

Expected: **31 passed**（0 skipped, 0 failed）

- [ ] **Run: `cd perplexity && python -c "import mosaic.pipeline; print('imports ok')"`**

Expected: `imports ok`

如果全部通过，实现完成。
