# Photomosaic Toy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local ipynb photomosaic toy that rearranges a pool of photos into a target image, with λ/μ/τ tunable sliders, self-deprecating report, and DeepZoom HTML output.

**Architecture:** Approach II — thin 8-cell `mosaic.ipynb` driving a testable pure-function module `mosaic_core.py`. TDD for every pure function; notebook is exercised via `nbconvert --execute` as smoke test.

**Tech Stack:** Python 3.11+, pillow, numpy, scikit-image (rgb2lab / deltaE_ciede2000), faiss-cpu, tqdm, ipywidgets, matplotlib, deepzoom, pytest, nbformat.

**Spec:** `chatgpt/docs/superpowers/specs/2026-04-17-photomosaic-toy-design.md`

**Working directory for all tasks:** `/Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/chatgpt/` (abbreviated as `chatgpt/` below).

**Important — CHANGELOG convention:** Every task ends with appending a YAML entry to `CHANGELOG.md` per the agent-friendly format (see memory `feedback_changelog_for_agents`). Template included in Task 1.

**Important — commit hygiene:** Each task produces exactly one git commit. Commit message format: `<type>: <short imperative>` where type is `feat|fix|test|chore|docs`. `-m` via HEREDOC.

---

## Spec-level refinement locked in this plan

Spec §4 listed `scan_tile_pool` as returning `list[TileRecord]`. The MVP must surface the bad-file list to the report (spec §7 says "加入 bad_files 列表" and spec §8 tests "坏图 ... 归入 bad_files"). We therefore implement:

```python
def scan_tile_pool(dir: Path, cache_path: Path) -> tuple[list[TileRecord], list[Path]]
```

The tuple's second element is `bad_files`. This ripples into `build_report(..., bad_files)` per spec §4.

---

## File structure produced by this plan

```
chatgpt/
├── plan.md                             # existing — untouched
├── docs/
│   └── superpowers/
│       ├── specs/2026-04-17-photomosaic-toy-design.md   # existing
│       └── plans/2026-04-17-photomosaic-toy-implementation.md  # this file
├── mosaic_core.py                      # NEW — all pure functions + dataclasses
├── mosaic.ipynb                        # NEW — 8-cell notebook (generated)
├── _build_notebook.py                  # NEW — source of truth for mosaic.ipynb
├── requirements.txt                    # NEW
├── .gitignore                          # NEW
├── CHANGELOG.md                        # NEW
├── CHANGELOG.archive.md                # NEW (header only)
└── tests/
    ├── test_color.py                   # NEW
    ├── test_matching.py                # NEW
    └── test_transfer.py                # NEW
```

Runtime artifacts (`.cache/`, `out/`, `my_tiles/`, `target.jpg`) are `.gitignore`d.

---

## Task 1: Project scaffolding

**Files:**
- Create: `chatgpt/requirements.txt`
- Create: `chatgpt/.gitignore`
- Create: `chatgpt/CHANGELOG.md`
- Create: `chatgpt/CHANGELOG.archive.md`
- Create: `chatgpt/tests/` (empty directory, no `__init__.py` needed since pytest rootdir-style works)

- [ ] **Step 1: Create `requirements.txt`**

Exact content:

```
pillow>=10
numpy>=1.26
scipy>=1.11
scikit-image>=0.22
faiss-cpu>=1.7
tqdm>=4.66
jupyter>=1.0
ipywidgets>=8.1
matplotlib>=3.8
deepzoom>=0.2
pytest>=7.4
nbformat>=5.9
```

Path: `chatgpt/requirements.txt`.

- [ ] **Step 2: Create `.gitignore`**

Exact content:

```
__pycache__/
*.py[cod]
*.egg-info/
.cache/
.pytest_cache/
.ipynb_checkpoints/
out/
my_tiles/
target.jpg
.DS_Store
```

Path: `chatgpt/.gitignore`.

- [ ] **Step 3: Create `CHANGELOG.md` with initial entry**

Exact content:

```markdown
# CHANGELOG

> Format: agent-friendly YAML per feedback_changelog_for_agents memory.
> Compression triggers at 50 entries or 6 months (archived in CHANGELOG.archive.md).

## 活跃条目

- date: 2026-04-17
  type: feat
  target: chatgpt/
  change: 初始化 photomosaic toy 项目骨架,产出 requirements.txt / .gitignore / CHANGELOG.md / tests/ 空目录
  rationale: 按已批准的 spec (docs/superpowers/specs/2026-04-17-photomosaic-toy-design.md) 开始实现;MVP 定位为本地 ipynb 玩具,λ/μ/τ 三滑条 + 自嘲式报告 + DeepZoom。
  action: 创建 requirements.txt (11 个 pip-only 依赖,零 native/brew)、.gitignore (忽略 .cache/out/my_tiles/target.jpg)、CHANGELOG 头,建 tests/ 目录。
  result: 骨架就绪,进入 TDD 模块实现阶段。
  validation: 文件存在;pip install -r requirements.txt 能装;git status 干净。
  status: stable
```

Path: `chatgpt/CHANGELOG.md`.

- [ ] **Step 4: Create `CHANGELOG.archive.md` (header only)**

Exact content:

```markdown
# CHANGELOG Archive

> Compressed entries land here when CHANGELOG.md hits 50 entries or 6 months.
> Preserves full try-failed chains and experimental/reverted history.

(empty — nothing compressed yet)
```

Path: `chatgpt/CHANGELOG.archive.md`.

- [ ] **Step 5: Create empty `tests/` directory**

Run: `mkdir -p chatgpt/tests`

- [ ] **Step 6: Install dependencies and verify**

Run from `chatgpt/`:
```bash
pip install -r requirements.txt
```

Expected: all packages install without native-build errors. If `faiss-cpu` fails on Apple Silicon, retry with `conda install -c conda-forge faiss-cpu` OR `pip install faiss-cpu --no-cache-dir`. If still blocked, halt and escalate — faiss is load-bearing for matching.

- [ ] **Step 7: Verify pytest is wired**

Run from `chatgpt/`:
```bash
pytest --version
```

Expected: prints pytest 7.4+ version.

- [ ] **Step 8: Commit**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo
git add chatgpt/requirements.txt chatgpt/.gitignore chatgpt/CHANGELOG.md chatgpt/CHANGELOG.archive.md
git commit -m "$(cat <<'EOF'
chore: scaffold photomosaic toy project

Add requirements.txt, .gitignore, CHANGELOG.md/archive per
the agent-friendly convention. Sets the stage for TDD modules.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Color — `lab_mean`

**Files:**
- Create: `chatgpt/mosaic_core.py` (skeleton with imports + dataclasses)
- Create: `chatgpt/tests/test_color.py`

- [ ] **Step 1: Write `mosaic_core.py` skeleton**

Path: `chatgpt/mosaic_core.py`. Exact content:

```python
"""Pure-function helpers for the photomosaic toy.

Every non-IO function in this module is deterministic and unit-tested.
IO helpers (`scan_tile_pool`, `ensure_seed_tiles`, `export_deepzoom`) are
thin wrappers that do one thing each.
"""

from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab


# ---------- dataclasses ----------

@dataclass
class TileRecord:
    path: Path
    lab_mean: np.ndarray  # float32[3]
    rgb_thumb: np.ndarray  # uint8[64, 64, 3]


@dataclass
class MosaicConfig:
    target_path: Optional[Path] = Path("target.jpg")
    tile_dir: Path = Path("my_tiles")
    grid_w: int = 120
    grid_h: int = 68
    tile_px: int = 16
    k_candidates: int = 32
    lambda_repeat: float = 0.5
    mu_neighbor: float = 0.3
    tau_transfer: float = 0.4
    cache_path: Path = Path(".cache/tiles.pkl")
    out_dir: Path = Path("out")


@dataclass
class ReportBundle:
    text: str
    usage_bar_fig: object  # matplotlib.figure.Figure — kept as object to avoid import at module top
    cold_wall_fig: object
```

- [ ] **Step 2: Write failing test for `lab_mean`**

Path: `chatgpt/tests/test_color.py`. Exact content:

```python
import numpy as np
import pytest

from mosaic_core import lab_mean


def test_lab_mean_pure_red():
    """Pure sRGB red has LAB ~= (53.24, 80.09, 67.20)."""
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    result = lab_mean(rgb)
    expected = np.array([53.24, 80.09, 67.20], dtype=np.float32)
    assert result.shape == (3,)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, atol=0.5)
```

- [ ] **Step 3: Run test — verify it fails**

Run from `chatgpt/`:
```bash
pytest tests/test_color.py::test_lab_mean_pure_red -v
```

Expected: FAIL with `ImportError: cannot import name 'lab_mean' from 'mosaic_core'`.

- [ ] **Step 4: Implement `lab_mean` in `mosaic_core.py`**

Append to `mosaic_core.py`:

```python
# ---------- color ----------

def lab_mean(rgb: np.ndarray) -> np.ndarray:
    """Return LAB mean of an H×W×3 uint8 RGB array as float32[3]."""
    lab = rgb2lab(rgb.astype(np.float64) / 255.0)
    return lab.reshape(-1, 3).mean(axis=0).astype(np.float32)
```

- [ ] **Step 5: Run test — verify it passes**

Run from `chatgpt/`:
```bash
pytest tests/test_color.py::test_lab_mean_pure_red -v
```

Expected: PASS.

- [ ] **Step 6: Commit with CHANGELOG entry**

Append to `CHANGELOG.md` under "## 活跃条目":

```yaml
- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 lab_mean(rgb) -> float32[3],基于 skimage.color.rgb2lab 对 uint8 RGB 求 LAB 均值
  rationale: 色差匹配的基础;后续 reinhard_transfer、ciede2000、scan_tile_pool 都依赖它
  action: 新建 mosaic_core.py 含 TileRecord/MosaicConfig/ReportBundle 三个 dataclass 骨架 + lab_mean
  result: 纯红图像测试通过,返回值与 sRGB → CIELAB 理论值 ± 0.5 内吻合
  validation: tests/test_color.py::test_lab_mean_pure_red 绿
  status: stable
```

Then commit:
```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo
git add chatgpt/mosaic_core.py chatgpt/tests/test_color.py chatgpt/CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat: add lab_mean for LAB-space color averaging

First pure function in mosaic_core. TDD: pure-red sRGB asserts to
(53.24, 80.09, 67.20) CIELAB within 0.5 unit tolerance.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Color — `ciede2000`

**Files:**
- Modify: `chatgpt/mosaic_core.py` (add `ciede2000`)
- Modify: `chatgpt/tests/test_color.py` (add test)

- [ ] **Step 1: Add failing test**

Append to `tests/test_color.py`:

```python
def test_ciede2000_identity_is_zero():
    """ΔE between a LAB triple and itself must be (numerically) zero."""
    from mosaic_core import ciede2000
    lab = np.array([50.0, 10.0, -5.0], dtype=np.float32)
    assert ciede2000(lab, lab) < 1e-6


def test_ciede2000_nonzero_for_different_colors():
    """Two clearly different LAB points must have positive ΔE."""
    from mosaic_core import ciede2000
    a = np.array([50.0, 80.0, 0.0], dtype=np.float32)
    b = np.array([50.0, -80.0, 0.0], dtype=np.float32)
    assert ciede2000(a, b) > 10.0
```

- [ ] **Step 2: Run — fail**

Run: `pytest tests/test_color.py -v`

Expected: 2 new FAILs with `ImportError: cannot import name 'ciede2000'`.

- [ ] **Step 3: Implement `ciede2000`**

Append to `mosaic_core.py` under the `# ---------- color ----------` section:

```python
def ciede2000(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    """Scalar ΔE*_CIEDE2000 between two LAB triples (shape (3,))."""
    a = np.asarray(lab_a, dtype=np.float64).reshape(1, 1, 3)
    b = np.asarray(lab_b, dtype=np.float64).reshape(1, 1, 3)
    return float(deltaE_ciede2000(a, b).squeeze())
```

- [ ] **Step 4: Run — pass**

Run: `pytest tests/test_color.py -v`

Expected: 3/3 PASS.

- [ ] **Step 5: Commit with CHANGELOG entry**

Append to `CHANGELOG.md`:

```yaml
- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 ciede2000(lab_a, lab_b) -> float,包装 skimage.color.deltaE_ciede2000 返回标量
  rationale: ΔE_CIEDE2000 是 rerank 里的主色差项;纯 LAB 欧氏会在深色/饱和色区域失真
  action: 新增函数,内部 reshape 到 (1,1,3) 后 squeeze 出标量
  result: 同点 ΔE < 1e-6,红-绿对比 > 10 两条测试均通过
  validation: tests/test_color.py::test_ciede2000_* 绿
  status: stable
```

Commit:
```bash
git add chatgpt/mosaic_core.py chatgpt/tests/test_color.py chatgpt/CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat: add ciede2000 scalar color-difference wrapper

Thin wrapper over skimage deltaE_ciede2000 that returns a scalar
for rerank's primary cost term.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Color — `reinhard_transfer`

**Files:**
- Modify: `chatgpt/mosaic_core.py` (add `reinhard_transfer`)
- Modify: `chatgpt/tests/test_color.py` (add 2 tests)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_color.py`:

```python
def test_reinhard_tau_zero_returns_original():
    """τ=0 must short-circuit and return the input tile unchanged (byte-exact)."""
    from mosaic_core import reinhard_transfer
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    target_lab = np.array([60.0, 20.0, -10.0], dtype=np.float32)
    out = reinhard_transfer(rgb, target_lab, tau=0.0)
    np.testing.assert_array_equal(out, rgb)


def test_reinhard_tau_one_matches_target_mean():
    """τ=1 must drag the tile's LAB mean onto target_lab_mean (within round-trip error)."""
    from mosaic_core import lab_mean, reinhard_transfer
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    target_lab = np.array([60.0, 10.0, -5.0], dtype=np.float32)
    out = reinhard_transfer(rgb, target_lab, tau=1.0)
    got = lab_mean(out)
    # sRGB clipping + round-trip introduces ~1-3 unit drift; 4.0 is safe.
    np.testing.assert_allclose(got, target_lab, atol=4.0)
```

- [ ] **Step 2: Run — fail**

Run: `pytest tests/test_color.py -v`

Expected: 2 new FAILs with `ImportError`.

- [ ] **Step 3: Implement `reinhard_transfer`**

Append to `mosaic_core.py`:

```python
def reinhard_transfer(
    tile_rgb: np.ndarray,
    target_lab_mean: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Shift tile's LAB mean onto target_lab_mean, then mix by τ∈[0,1]."""
    if tau == 0.0:
        return tile_rgb
    tile_lab = rgb2lab(tile_rgb.astype(np.float64) / 255.0)
    tile_lab_mean = tile_lab.reshape(-1, 3).mean(axis=0)
    offset = np.asarray(target_lab_mean, dtype=np.float64) - tile_lab_mean
    transferred_lab = tile_lab + offset
    transferred_rgb = np.clip(lab2rgb(transferred_lab) * 255.0, 0, 255).astype(np.uint8)
    if tau == 1.0:
        return transferred_rgb
    blend = tile_rgb.astype(np.float32) * (1.0 - tau) + transferred_rgb.astype(np.float32) * tau
    return np.clip(blend, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run — pass**

Run: `pytest tests/test_color.py -v`

Expected: all 5 color tests PASS.

- [ ] **Step 5: Commit + CHANGELOG**

Append YAML to CHANGELOG.md:

```yaml
- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 reinhard_transfer(tile_rgb, target_lab_mean, τ) -> rgb,按 τ 线性混合 LAB 均值迁移后的 RGB 与原图
  rationale: τ 是"有质感层"的核心滑条;原 Reinhard 含 std 缩放,此处只做 mean-shift 够用且稳定
  action: τ=0 短路返回原 tile;τ=1 返回纯迁移结果;中间做 (1-τ)*orig + τ*transferred 线性混
  result: τ=0 byte-exact;τ=1 后新均值与目标误差 < 4 LAB 单位(sRGB 裁剪导致的正常漂移)
  validation: tests/test_color.py::test_reinhard_tau_{zero,one}_* 绿
  status: stable
```

Commit:
```bash
git add chatgpt/mosaic_core.py chatgpt/tests/test_color.py chatgpt/CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat: add reinhard_transfer for LAB mean color shift

τ-blended mean-shift in LAB space. τ=0 short-circuits; τ=1 gives
full transfer. Drives the "质感" slider.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Grid — `split_target`

**Files:**
- Modify: `chatgpt/mosaic_core.py`
- Create: `chatgpt/tests/test_transfer.py`

- [ ] **Step 1: Write failing test**

Path: `chatgpt/tests/test_transfer.py`. Exact content:

```python
import numpy as np
import pytest
from PIL import Image

from mosaic_core import split_target


def test_split_target_shape_and_dtype():
    """split_target(img, grid_w=10, grid_h=5) must return ndarray shape (5, 10, 3) float32."""
    img = Image.new("RGB", (100, 50), (128, 128, 128))  # 100 px wide, 50 px tall
    lab_grid = split_target(img, grid_w=10, grid_h=5)
    assert lab_grid.shape == (5, 10, 3)
    assert lab_grid.dtype == np.float32


def test_split_target_uniform_gray_constant_lab():
    """Uniform gray input -> all cells share the same LAB value (L~54, a~0, b~0)."""
    img = Image.new("RGB", (80, 40), (128, 128, 128))
    lab_grid = split_target(img, grid_w=8, grid_h=4)
    L_plane = lab_grid[..., 0]
    # All cells equal within float noise.
    assert L_plane.max() - L_plane.min() < 0.01
    # L ~= 54 for sRGB 128 gray.
    assert 52 < L_plane.mean() < 56
```

- [ ] **Step 2: Run — fail**

Run: `pytest tests/test_transfer.py -v`

Expected: FAIL with `ImportError: cannot import name 'split_target'`.

- [ ] **Step 3: Implement `split_target`**

Append to `mosaic_core.py`:

```python
# ---------- grid / render ----------

def split_target(img: Image.Image, grid_w: int, grid_h: int) -> np.ndarray:
    """Split img into grid_h × grid_w patches and return each patch's LAB mean.

    Returns: ndarray[grid_h, grid_w, 3] float32.
    """
    patch_w = img.width // grid_w
    patch_h = img.height // grid_h
    resized = img.resize((grid_w * patch_w, grid_h * patch_h), Image.BILINEAR)
    rgb = np.asarray(resized, dtype=np.uint8)
    lab = rgb2lab(rgb.astype(np.float64) / 255.0)
    # Reshape to (grid_h, patch_h, grid_w, patch_w, 3) then mean over patch_h, patch_w.
    reshaped = lab.reshape(grid_h, patch_h, grid_w, patch_w, 3)
    return reshaped.mean(axis=(1, 3)).astype(np.float32)
```

- [ ] **Step 4: Run — pass**

Run: `pytest tests/test_transfer.py -v`

Expected: 2/2 PASS.

- [ ] **Step 5: Commit + CHANGELOG**

Append to CHANGELOG.md:

```yaml
- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 split_target(img, grid_w, grid_h) -> float32[H, W, 3],输出每 patch LAB 均值
  rationale: 匹配阶段需要"每格目标颜色"作为 KNN 查询向量
  action: resize 到 grid_w*patch_w × grid_h*patch_h 后 reshape+mean 向量化计算,无 Python 循环
  result: 100×50 图切 10×5 返回正确 shape;灰度图各 cell 的 LAB 一致且 L~54
  validation: tests/test_transfer.py::test_split_target_* 绿
  status: stable
```

Commit:
```bash
git add chatgpt/mosaic_core.py chatgpt/tests/test_transfer.py chatgpt/CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat: add split_target for per-cell LAB averaging

Vectorized grid splitter: resize → reshape → mean. No Python loops.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Grid — `render_mosaic`

**Files:**
- Modify: `chatgpt/mosaic_core.py`
- Modify: `chatgpt/tests/test_transfer.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_transfer.py`:

```python
def test_render_mosaic_output_size():
    """Output PIL image size = (grid_w * tile_px, grid_h * tile_px)."""
    from mosaic_core import TileRecord, render_mosaic
    # One deterministic tile: solid red 16×16.
    tile_rgb = np.full((16, 16, 3), fill_value=0, dtype=np.uint8)
    tile_rgb[..., 0] = 255
    tile_lab = np.array([53.24, 80.09, 67.20], dtype=np.float32)
    records = [TileRecord(path=None, lab_mean=tile_lab, rgb_thumb=tile_rgb)]
    # 4×3 grid, all cells point to tile 0.
    assignment = np.zeros((3, 4), dtype=np.int64)
    target_lab = np.broadcast_to(tile_lab, (3, 4, 3)).copy()
    img = render_mosaic(assignment, records, tile_px=16, tau=0.0, target_lab=target_lab)
    assert img.size == (4 * 16, 3 * 16)


def test_render_mosaic_tau_zero_preserves_tile_bytes():
    """τ=0 means every cell is the tile's raw rgb_thumb (byte-exact)."""
    from mosaic_core import TileRecord, render_mosaic
    rng = np.random.default_rng(7)
    tile_rgb = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    tile_lab = np.array([50.0, 0.0, 0.0], dtype=np.float32)
    records = [TileRecord(path=None, lab_mean=tile_lab, rgb_thumb=tile_rgb)]
    assignment = np.zeros((2, 2), dtype=np.int64)
    target_lab = np.full((2, 2, 3), [30.0, 20.0, 5.0], dtype=np.float32)  # different from tile
    img = render_mosaic(assignment, records, tile_px=16, tau=0.0, target_lab=target_lab)
    out = np.asarray(img)
    # Top-left 16x16 block must equal tile_rgb.
    np.testing.assert_array_equal(out[:16, :16], tile_rgb)
```

- [ ] **Step 2: Run — fail**

Run: `pytest tests/test_transfer.py -v`

Expected: 2 new FAILs with `ImportError`.

- [ ] **Step 3: Implement `render_mosaic`**

Append to `mosaic_core.py`:

```python
def render_mosaic(
    assignment: np.ndarray,
    tile_records: list[TileRecord],
    tile_px: int,
    tau: float,
    target_lab: np.ndarray,
) -> Image.Image:
    """Paste tiles onto an (grid_h*tile_px, grid_w*tile_px) canvas.

    assignment: int64[grid_h, grid_w] — tile record index per cell.
    target_lab: float32[grid_h, grid_w, 3] — per-cell LAB target for τ transfer.
    """
    grid_h, grid_w = assignment.shape
    canvas = np.zeros((grid_h * tile_px, grid_w * tile_px, 3), dtype=np.uint8)
    for r in range(grid_h):
        for c in range(grid_w):
            rec = tile_records[int(assignment[r, c])]
            thumb = rec.rgb_thumb
            if thumb.shape[0] != tile_px or thumb.shape[1] != tile_px:
                thumb = np.asarray(
                    Image.fromarray(thumb).resize((tile_px, tile_px), Image.BILINEAR)
                )
            if tau > 0.0:
                thumb = reinhard_transfer(thumb, target_lab[r, c], tau)
            y0, x0 = r * tile_px, c * tile_px
            canvas[y0:y0 + tile_px, x0:x0 + tile_px] = thumb
    return Image.fromarray(canvas)
```

- [ ] **Step 4: Run — pass**

Run: `pytest tests/test_transfer.py -v`

Expected: 4/4 PASS.

- [ ] **Step 5: Commit + CHANGELOG**

Append:

```yaml
- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 render_mosaic(assignment, tile_records, tile_px, τ, target_lab) -> PIL.Image
  rationale: 把 assignment 表变成可视化像素矩阵;τ>0 时每 tile 贴前做 reinhard_transfer
  action: 双层 for 循环按格贴图,尺寸不匹配时 resize;短路 τ=0 省去 LAB 往返开销
  result: 4×3 grid 输出 64×48 PIL Image;τ=0 块内字节完全等于 tile 原图
  validation: tests/test_transfer.py::test_render_mosaic_* 绿
  status: stable
```

Commit:
```bash
git add chatgpt/mosaic_core.py chatgpt/tests/test_transfer.py chatgpt/CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat: add render_mosaic canvas compositor

Two-pass composition: resize if needed, τ>0 routes through
reinhard_transfer per cell. Simple and testable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Matching — `build_faiss_index` + `knn_candidates`

**Files:**
- Modify: `chatgpt/mosaic_core.py`
- Create: `chatgpt/tests/test_matching.py`

- [ ] **Step 1: Write failing tests**

Path: `chatgpt/tests/test_matching.py`. Exact content:

```python
import numpy as np
import pytest

from mosaic_core import build_faiss_index, knn_candidates


def test_build_faiss_index_returns_top_k_deterministic():
    """Same query must give identical top-5 on repeated calls."""
    rng = np.random.default_rng(42)
    tile_labs = rng.uniform(0, 100, size=(100, 3)).astype(np.float32)
    index = build_faiss_index(tile_labs)
    query = np.array([[50.0, 0.0, 0.0]], dtype=np.float32)
    d1, i1 = index.search(query, 5)
    d2, i2 = index.search(query, 5)
    np.testing.assert_array_equal(i1, i2)
    # Top hit must be the closest tile by L2.
    dists = np.linalg.norm(tile_labs - query, axis=1)
    assert i1[0, 0] == int(np.argmin(dists))


def test_knn_candidates_shape_and_legal_indices():
    """knn_candidates returns (H*W, k) indices all in [0, N)."""
    rng = np.random.default_rng(0)
    tile_labs = rng.uniform(0, 100, size=(50, 3)).astype(np.float32)
    target_lab = rng.uniform(0, 100, size=(4, 6, 3)).astype(np.float32)
    index = build_faiss_index(tile_labs)
    result = knn_candidates(target_lab, index, k=8)
    assert result.shape == (4 * 6, 8)
    assert result.min() >= 0
    assert result.max() < 50
```

- [ ] **Step 2: Run — fail**

Run: `pytest tests/test_matching.py -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `build_faiss_index` + `knn_candidates`**

Append to `mosaic_core.py`:

```python
# ---------- matching ----------

def build_faiss_index(tile_labs: np.ndarray):
    """Build a flat L2 faiss index over an N×3 LAB matrix."""
    import faiss  # lazy import to avoid paying at module load
    arr = np.ascontiguousarray(tile_labs, dtype=np.float32)
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    return index


def knn_candidates(target_lab: np.ndarray, faiss_index, k: int = 32) -> np.ndarray:
    """Return top-k tile indices per target cell: int64[H*W, k]."""
    h, w, _ = target_lab.shape
    query = np.ascontiguousarray(target_lab.reshape(h * w, 3), dtype=np.float32)
    effective_k = min(k, faiss_index.ntotal)
    _dists, idxs = faiss_index.search(query, effective_k)
    return idxs.astype(np.int64)
```

- [ ] **Step 4: Run — pass**

Run: `pytest tests/test_matching.py -v`

Expected: 2/2 PASS.

- [ ] **Step 5: Commit + CHANGELOG**

Append:

```yaml
- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 build_faiss_index / knn_candidates,做 LAB 空间 top-k 候选查询
  rationale: N < 10k 底图量级,IndexFlatL2 就够用,无需 IVF;有 k > N 时自动降级为 N
  action: lazy import faiss;把 target 从 (H, W, 3) reshape 成 (H·W, 3) 批量查询
  result: 100 tile 查询 top-5 确定性且与 argmin L2 一致;4×6 grid 返回 (24, 8)
  validation: tests/test_matching.py::test_build_faiss_index_*, test_knn_candidates_* 绿
  status: stable
```

Commit:
```bash
git add chatgpt/mosaic_core.py chatgpt/tests/test_matching.py chatgpt/CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat: add faiss flat-L2 tile index + knn_candidates batch query

Flat L2 over LAB means. k > N degrades gracefully. Reshape-based
batch query avoids per-cell Python overhead.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Matching — `rerank`

**Files:**
- Modify: `chatgpt/mosaic_core.py`
- Modify: `chatgpt/tests/test_matching.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_matching.py`:

```python
def test_rerank_lambda_zero_mu_zero_is_argmin_delta_e():
    """With both penalties off, rerank returns the candidate with smallest ΔE."""
    from mosaic_core import rerank
    tile_labs = np.array(
        [[50.0, 0.0, 0.0], [50.0, 20.0, 0.0], [50.0, 40.0, 0.0]],
        dtype=np.float32,
    )
    target = np.array([50.0, 35.0, 0.0], dtype=np.float32)  # closest to index 2
    best = rerank(
        candidate_idxs=np.array([0, 1, 2], dtype=np.int64),
        tile_labs=tile_labs,
        target_lab_patch=target,
        usage_counts={},
        neighbor_tile_idxs=[],
        lambda_repeat=0.0,
        mu_neighbor=0.0,
    )
    assert best == 2


def test_rerank_lambda_penalizes_heavy_usage():
    """Heavy λ shifts choice away from an over-used tile toward an unused one."""
    from mosaic_core import rerank
    tile_labs = np.array(
        [[50.0, 0.0, 0.0], [50.0, 5.0, 0.0]],  # tile 0 slightly closer than tile 1
        dtype=np.float32,
    )
    target = np.array([50.0, 0.0, 0.0], dtype=np.float32)
    # λ=0: picks tile 0.
    best_cold = rerank(
        candidate_idxs=np.array([0, 1], dtype=np.int64),
        tile_labs=tile_labs,
        target_lab_patch=target,
        usage_counts={},
        neighbor_tile_idxs=[],
        lambda_repeat=0.0,
        mu_neighbor=0.0,
    )
    assert best_cold == 0
    # λ=100 with tile 0 used 1000 times: the log-penalty > ΔE gap → picks tile 1.
    best_hot = rerank(
        candidate_idxs=np.array([0, 1], dtype=np.int64),
        tile_labs=tile_labs,
        target_lab_patch=target,
        usage_counts={0: 1000},
        neighbor_tile_idxs=[],
        lambda_repeat=100.0,
        mu_neighbor=0.0,
    )
    assert best_hot == 1


def test_rerank_mu_penalizes_similar_neighbor():
    """Heavy μ shifts choice away from tile that looks like the left/up neighbor."""
    from mosaic_core import rerank
    # tile 0 is visually identical to tile 2 (same LAB); tile 1 is distinct.
    tile_labs = np.array(
        [[50.0, 0.0, 0.0], [50.0, 30.0, 0.0], [50.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    target = np.array([50.0, 0.0, 0.0], dtype=np.float32)  # tile 0 is closest
    # μ=0: picks tile 0.
    best_no_neighbor = rerank(
        candidate_idxs=np.array([0, 1], dtype=np.int64),
        tile_labs=tile_labs,
        target_lab_patch=target,
        usage_counts={},
        neighbor_tile_idxs=[],
        lambda_repeat=0.0,
        mu_neighbor=0.0,
    )
    assert best_no_neighbor == 0
    # μ=100 with tile 2 (identical LAB to tile 0) already placed next door: picks tile 1.
    best_with_clone_neighbor = rerank(
        candidate_idxs=np.array([0, 1], dtype=np.int64),
        tile_labs=tile_labs,
        target_lab_patch=target,
        usage_counts={},
        neighbor_tile_idxs=[2],
        lambda_repeat=0.0,
        mu_neighbor=100.0,
    )
    assert best_with_clone_neighbor == 1
```

- [ ] **Step 2: Run — fail**

Run: `pytest tests/test_matching.py -v`

Expected: 3 new FAILs with `ImportError: cannot import name 'rerank'`.

- [ ] **Step 3: Implement `rerank`**

Append to `mosaic_core.py`:

```python
def rerank(
    candidate_idxs: np.ndarray,
    tile_labs: np.ndarray,
    target_lab_patch: np.ndarray,
    usage_counts: dict,
    neighbor_tile_idxs: list[int],
    lambda_repeat: float,
    mu_neighbor: float,
) -> int:
    """Pick the best tile index from candidates via ΔE + usage + neighbor penalties.

    score = ΔE_CIEDE2000(tile, target) + λ·log(1+usage) + μ·max_sim_to_any_neighbor

    where neighbor_similarity = 1 / (1 + ΔE_CIEDE2000(tile, neighbor)).

    neighbor_tile_idxs is the list of already-filled left/up neighbor tile indices
    in scanline order; may be empty or have 1–2 entries.
    """
    best_idx = -1
    best_score = math.inf
    for raw_idx in candidate_idxs:
        idx = int(raw_idx)
        lab = tile_labs[idx]
        de = ciede2000(lab, target_lab_patch)
        usage_pen = lambda_repeat * math.log1p(usage_counts.get(idx, 0))
        if neighbor_tile_idxs:
            sim = max(
                1.0 / (1.0 + ciede2000(lab, tile_labs[n])) for n in neighbor_tile_idxs
            )
        else:
            sim = 0.0
        score = de + usage_pen + mu_neighbor * sim
        if score < best_score:
            best_score = score
            best_idx = idx
    return best_idx
```

- [ ] **Step 4: Run — pass**

Run: `pytest tests/test_matching.py -v`

Expected: all 5 matching tests PASS.

- [ ] **Step 5: Commit + CHANGELOG**

Append:

```yaml
- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 rerank(candidate_idxs, ..., λ, μ) -> tile_idx,结合 ΔE + λ·log(1+usage) + μ·max_neighbor_sim
  rationale: λ 摊开重复使用,μ 避免同色扎堆;比纯最近邻出图干净得多
  action: 在 top-k 候选里逐个算三项和,取最小;neighbor_sim = 1/(1+ΔE) 天然有界
  result: λ=μ=0 退化为 argmin ΔE;λ=100+usage=1000 可翻盘;μ=100+克隆邻居可翻盘
  validation: tests/test_matching.py::test_rerank_* (3 个) 绿
  status: stable
```

Commit:
```bash
git add chatgpt/mosaic_core.py chatgpt/tests/test_matching.py chatgpt/CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat: add rerank with λ usage-penalty and μ neighbor-penalty

Core selection step. λ amortizes popular tiles across the canvas;
μ prevents adjacent clones. Neighbor similarity is 1/(1+ΔE),
bounded and monotonic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Pool — `ensure_seed_tiles`

**Files:**
- Modify: `chatgpt/mosaic_core.py`
- Modify: `chatgpt/tests/test_transfer.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_transfer.py`:

```python
def test_ensure_seed_tiles_creates_files(tmp_path):
    """ensure_seed_tiles on an empty path creates n JPEG files."""
    from mosaic_core import ensure_seed_tiles
    target_dir = tmp_path / "seeds"
    ensure_seed_tiles(target_dir, n=10)
    jpgs = list(target_dir.glob("*.jpg"))
    assert len(jpgs) == 10
    # Each file should be a readable 64×64 image.
    from PIL import Image
    for p in jpgs:
        img = Image.open(p)
        assert img.size == (64, 64)


def test_ensure_seed_tiles_noop_when_nonempty(tmp_path):
    """If the dir already has images, ensure_seed_tiles must not add more."""
    from mosaic_core import ensure_seed_tiles
    target_dir = tmp_path / "existing"
    target_dir.mkdir()
    (target_dir / "user.jpg").write_bytes(b"fake")
    ensure_seed_tiles(target_dir, n=5)
    assert len(list(target_dir.glob("*"))) == 1
```

- [ ] **Step 2: Run — fail**

Run: `pytest tests/test_transfer.py -v`

Expected: 2 new FAILs with `ImportError`.

- [ ] **Step 3: Implement `ensure_seed_tiles`**

Append to `mosaic_core.py` (start the "# ---------- pool ----------" section):

```python
# ---------- pool / IO ----------

def ensure_seed_tiles(tile_dir: Path, n: int = 200) -> None:
    """If tile_dir is missing or empty, synthesize n 64×64 HSV color blocks."""
    tile_dir = Path(tile_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)
    if any(tile_dir.iterdir()):
        return
    rng = random.Random(0)
    for i in range(n):
        h = rng.random()
        s = 0.4 + rng.random() * 0.6
        v = 0.3 + rng.random() * 0.7
        rgb_float = _hsv_to_rgb(h, s, v)
        rgb = np.tile(
            (np.array(rgb_float) * 255).astype(np.uint8),
            (64, 64, 1),
        )
        Image.fromarray(rgb).save(tile_dir / f"seed_{i:04d}.jpg", quality=90)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Pure-stdlib HSV→RGB in [0,1]."""
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)
```

- [ ] **Step 4: Run — pass**

Run: `pytest tests/test_transfer.py -v`

Expected: all 6 transfer tests PASS.

- [ ] **Step 5: Commit + CHANGELOG**

Append:

```yaml
- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 ensure_seed_tiles(dir, n=200),空目录自动生成 64×64 HSV 色块 JPG
  rationale: 零配置跑通;Cell 3 即使底图目录空也能继续,不崩
  action: 固定 random.Random(0) 保证可复现;已有内容时 no-op,不覆盖用户真实照片
  result: 10 张 case 测试 shape 与数量;已有文件 case 验证 no-op
  validation: tests/test_transfer.py::test_ensure_seed_tiles_* (2 个) 绿
  status: stable
```

Commit:
```bash
git add chatgpt/mosaic_core.py chatgpt/tests/test_transfer.py chatgpt/CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat: add ensure_seed_tiles fallback for empty tile dirs

Deterministic HSV seed generator. No-op when the dir has any
existing content — won't overwrite a user's real photos.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Pool — `scan_tile_pool`

**Files:**
- Modify: `chatgpt/mosaic_core.py`
- Modify: `chatgpt/tests/test_transfer.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_transfer.py`:

```python
def test_scan_tile_pool_happy_path(tmp_path):
    """Scan returns TileRecord list and empty bad_files on clean JPGs."""
    from mosaic_core import scan_tile_pool, ensure_seed_tiles
    ensure_seed_tiles(tmp_path / "tiles", n=5)
    cache = tmp_path / "cache.pkl"
    tiles, bad = scan_tile_pool(tmp_path / "tiles", cache)
    assert len(tiles) == 5
    assert bad == []
    for t in tiles:
        assert t.lab_mean.shape == (3,)
        assert t.rgb_thumb.shape == (64, 64, 3)


def test_scan_tile_pool_skips_corrupt_files(tmp_path):
    """Corrupt JPG goes into bad_files and does not crash the scan."""
    from mosaic_core import scan_tile_pool, ensure_seed_tiles
    d = tmp_path / "tiles"
    ensure_seed_tiles(d, n=3)
    # Inject a file that looks like a JPG but isn't decodable.
    (d / "broken.jpg").write_bytes(b"not a real jpeg")
    cache = tmp_path / "cache.pkl"
    tiles, bad = scan_tile_pool(d, cache)
    assert len(tiles) == 3
    assert len(bad) == 1
    assert bad[0].name == "broken.jpg"


def test_scan_tile_pool_uses_cache(tmp_path):
    """Second call with same cache is trivial — cache file exists after first run."""
    from mosaic_core import scan_tile_pool, ensure_seed_tiles
    d = tmp_path / "tiles"
    ensure_seed_tiles(d, n=4)
    cache = tmp_path / "cache.pkl"
    tiles1, _ = scan_tile_pool(d, cache)
    assert cache.exists()
    tiles2, _ = scan_tile_pool(d, cache)
    assert len(tiles1) == len(tiles2)
    assert (tiles1[0].lab_mean == tiles2[0].lab_mean).all()
```

- [ ] **Step 2: Run — fail**

Run: `pytest tests/test_transfer.py -v`

Expected: 3 new FAILs with `ImportError: cannot import name 'scan_tile_pool'`.

- [ ] **Step 3: Implement `scan_tile_pool`**

Append to `mosaic_core.py`:

```python
TILE_CACHE_VERSION = 1
_TILE_EXTS = {".jpg", ".jpeg", ".png"}


def scan_tile_pool(
    tile_dir: Path,
    cache_path: Path,
) -> tuple[list[TileRecord], list[Path]]:
    """Recursively scan tile_dir, compute LAB means + 64×64 thumbs, cache to pickle.

    Returns: (tile_records, bad_files).
    """
    tile_dir = Path(tile_dir)
    cache_path = Path(cache_path)
    paths = sorted(
        p for p in tile_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _TILE_EXTS
    )

    cached: dict[Path, TileRecord] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as fh:
                payload = pickle.load(fh)
            if payload.get("version") == TILE_CACHE_VERSION:
                cached = payload["records"]
        except Exception:
            cached = {}

    records: list[TileRecord] = []
    bad_files: list[Path] = []
    for p in paths:
        if p in cached:
            records.append(cached[p])
            continue
        try:
            with Image.open(p) as img:
                img = img.convert("RGB").resize((64, 64), Image.BILINEAR)
                rgb = np.asarray(img, dtype=np.uint8)
            rec = TileRecord(path=p, lab_mean=lab_mean(rgb), rgb_thumb=rgb)
            records.append(rec)
        except Exception:
            bad_files.append(p)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(
            {"version": TILE_CACHE_VERSION, "records": {r.path: r for r in records}},
            fh,
        )
    return records, bad_files
```

- [ ] **Step 4: Run — pass**

Run: `pytest tests/test_transfer.py -v`

Expected: all 9 transfer tests PASS.

- [ ] **Step 5: Commit + CHANGELOG**

Append:

```yaml
- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 scan_tile_pool(dir, cache_path) -> (records, bad_files),递归扫图建 LAB+64×64 thumb 并 pickle 缓存
  rationale: 真实相册几千张照片二次跑时不用重读像素;缓存版本号不匹配直接重算
  action: rglob JPG/PNG/JPEG,PIL 读取后 resize 64×64;损坏文件进 bad_files 不中断;版本化 pickle
  result: 5 张清图全记录、坏 JPG 进 bad_files 不崩、缓存命中后内容字节级一致
  validation: tests/test_transfer.py::test_scan_tile_pool_* (3 个) 绿
  status: stable
```

Commit:
```bash
git add chatgpt/mosaic_core.py chatgpt/tests/test_transfer.py chatgpt/CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat: add scan_tile_pool with versioned pickle cache

Returns (records, bad_files) tuple. Corrupt files divert to
bad_files without aborting. Cache version mismatch triggers a
clean recompute (no migration).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Report — `build_report`

**Files:**
- Modify: `chatgpt/mosaic_core.py`
- Modify: `chatgpt/tests/test_transfer.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_transfer.py`:

```python
def test_build_report_structural_fields(tmp_path):
    """build_report returns a ReportBundle whose text contains key headlines."""
    from mosaic_core import TileRecord, build_report, ensure_seed_tiles, scan_tile_pool
    ensure_seed_tiles(tmp_path / "tiles", n=20)
    records, bad = scan_tile_pool(tmp_path / "tiles", tmp_path / "cache.pkl")
    assignment = np.zeros((4, 5), dtype=np.int64)  # all cells use tile 0
    bundle = build_report(assignment, records, elapsed_seconds=3.14, bad_files=bad)
    assert "扫到" in bundle.text or "tiles" in bundle.text
    assert "冷宫" in bundle.text or "cold" in bundle.text
    assert "3.14" in bundle.text or "3.1" in bundle.text
    # Figures are matplotlib Figures.
    import matplotlib
    assert isinstance(bundle.usage_bar_fig, matplotlib.figure.Figure)
    assert isinstance(bundle.cold_wall_fig, matplotlib.figure.Figure)
```

- [ ] **Step 2: Run — fail**

Run: `pytest tests/test_transfer.py -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `build_report`**

Append to `mosaic_core.py`:

```python
# ---------- report ----------

def build_report(
    assignment: np.ndarray,
    tile_records: list[TileRecord],
    elapsed_seconds: float,
    bad_files: list[Path],
) -> ReportBundle:
    """Produce a self-deprecating-style textual report + usage bar + cold-photo wall."""
    import matplotlib.pyplot as plt
    from collections import Counter

    flat = assignment.ravel()
    counts = Counter(int(x) for x in flat)
    total_cells = flat.size
    used_set = set(counts.keys())
    cold_idxs = [i for i in range(len(tile_records)) if i not in used_set]

    top_used = counts.most_common(5)
    lines = []
    lines.append(f"本次扫到 {len(tile_records)} 张 tile,跑了 {elapsed_seconds:.2f} 秒,生成了 {total_cells} 个格子。")
    lines.append(f"坏图 {len(bad_files)} 张,冷宫照片 {len(cold_idxs)} 张。")
    lines.append("")
    lines.append("TOP 5 最常被贴:")
    for idx, count in top_used:
        name = tile_records[idx].path.name if tile_records[idx].path else f"<tile {idx}>"
        lines.append(f"  {name}: {count} 次")
    lines.append("")
    lines.append(f"冷宫照片(前 5):")
    for idx in cold_idxs[:5]:
        name = tile_records[idx].path.name if tile_records[idx].path else f"<tile {idx}>"
        lines.append(f"  {name}")
    if bad_files:
        lines.append("")
        lines.append("坏图列表:")
        for p in bad_files:
            lines.append(f"  {p.name}")

    # Usage bar figure.
    bar_fig, bar_ax = plt.subplots(figsize=(10, 4))
    sorted_counts = sorted(counts.values(), reverse=True)
    bar_ax.bar(range(len(sorted_counts)), sorted_counts, color="#5b8cff")
    bar_ax.set_title("Tile usage (sorted desc)")
    bar_ax.set_xlabel("tile rank")
    bar_ax.set_ylabel("uses")

    # Cold wall figure — up to 25 thumbs in a 5×5 grid (or fewer if no cold).
    wall_n = min(25, len(cold_idxs))
    cols = 5 if wall_n >= 5 else max(1, wall_n)
    rows = max(1, (wall_n + cols - 1) // cols)
    wall_fig, wall_axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes_flat = np.atleast_1d(wall_axes).ravel()
    for i, ax in enumerate(axes_flat):
        if i < wall_n:
            ax.imshow(tile_records[cold_idxs[i]].rgb_thumb)
        ax.axis("off")
    wall_fig.suptitle(f"冷宫照片 ({len(cold_idxs)} 张未被使用)")
    wall_fig.tight_layout()

    return ReportBundle(
        text="\n".join(lines),
        usage_bar_fig=bar_fig,
        cold_wall_fig=wall_fig,
    )
```

- [ ] **Step 4: Run — pass**

Run: `pytest tests/test_transfer.py -v`

Expected: all 10 transfer tests PASS.

- [ ] **Step 5: Commit + CHANGELOG**

Append:

```yaml
- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 build_report(assignment, tile_records, elapsed, bad_files) -> ReportBundle(文字 + 使用柱状 + 冷宫墙)
  rationale: 报告是这个玩具"发朋友群秒上桌"的核心;TOP 5 + 冷宫 + 坏图三段式
  action: Counter 统计使用次数;matplotlib 画排序柱状图;至多 5×5=25 张冷宫缩略图墙
  result: 20 tile + 4×5=20 格 case 结构字段齐全,text 含"扫到""冷宫""耗时数字";两图都是 matplotlib.figure.Figure
  validation: tests/test_transfer.py::test_build_report_structural_fields 绿
  status: stable
```

Commit:
```bash
git add chatgpt/mosaic_core.py chatgpt/tests/test_transfer.py chatgpt/CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat: add build_report with usage bar and cold-photo wall

Three-section report (headline / TOP 5 / cold wall / bad files).
Returns a ReportBundle holding text + two matplotlib Figures the
notebook can display inline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: DeepZoom — `export_deepzoom`

**Files:**
- Modify: `chatgpt/mosaic_core.py`

No test — this is a thin IO wrapper covered by the end-to-end smoke in Task 14. Spec §8 explicitly says "不测 notebook / UI / IO".

- [ ] **Step 1: Implement `export_deepzoom`**

Append to `mosaic_core.py`:

```python
# ---------- deepzoom ----------

_OPENSEADRAGON_HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Photomosaic — DeepZoom</title>
    <style>
        html, body {{ margin: 0; height: 100%; background: #111; }}
        #viewer {{ width: 100vw; height: 100vh; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/openseadragon.min.js"></script>
</head>
<body>
    <div id="viewer"></div>
    <script>
        OpenSeadragon({{
            id: "viewer",
            prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/images/",
            tileSources: "{dzi_name}"
        }});
    </script>
</body>
</html>
"""


def export_deepzoom(png_path: Path, out_dir: Path) -> Path:
    """Slice a PNG into a DeepZoom pyramid and drop an index.html next to it.

    Returns the path to the generated index.html.
    """
    import deepzoom

    png_path = Path(png_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dzi_name = "mosaic.dzi"
    dzi_path = out_dir / dzi_name

    creator = deepzoom.ImageCreator(
        tile_size=256,
        tile_overlap=1,
        tile_format="jpg",
        image_quality=0.85,
        resize_filter="antialias",
    )
    creator.create(str(png_path), str(dzi_path))

    index_path = out_dir / "index.html"
    index_path.write_text(_OPENSEADRAGON_HTML.format(dzi_name=dzi_name), encoding="utf-8")
    return index_path
```

- [ ] **Step 2: Smoke-verify module imports cleanly**

Run from `chatgpt/`:
```bash
python -c "from mosaic_core import export_deepzoom; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit + CHANGELOG**

Append:

```yaml
- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 export_deepzoom(png_path, out_dir) -> index.html 路径,用 deepzoom.ImageCreator 切金字塔 + 写 OpenSeadragon HTML
  rationale: plan.md 明确说 DeepZoom 是整个项目 ROI 最高的功能;浏览器里无限缩放到单张底图可辨
  action: tile_size=256 overlap=1 JPG 85;HTML 通过 jsdelivr CDN 加载 OpenSeadragon 4,无离线 vendor 文件
  result: import 不报错;实际输出留给 Task 14 端到端冒烟
  validation: python -c 'from mosaic_core import export_deepzoom' 不抛
  status: stable
```

Commit:
```bash
git add chatgpt/mosaic_core.py chatgpt/CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat: add export_deepzoom for OpenSeadragon viewer

Thin wrapper over deepzoom.ImageCreator plus an OpenSeadragon-
from-CDN index.html. Untested at unit level per spec; covered by
end-to-end smoke in the notebook exec step.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Notebook source — `_build_notebook.py`

**Files:**
- Create: `chatgpt/_build_notebook.py`
- Create: `chatgpt/mosaic.ipynb` (generated artifact)

This task generates the 8-cell notebook from a Python source-of-truth script. Subsequent notebook edits happen by editing `_build_notebook.py` and re-running it; never hand-edit `mosaic.ipynb`.

- [ ] **Step 1: Write `_build_notebook.py`**

Path: `chatgpt/_build_notebook.py`. Exact content:

```python
"""Source-of-truth for mosaic.ipynb. Regenerate with:

    python _build_notebook.py

Never hand-edit mosaic.ipynb — edit this script and re-run.
"""

from pathlib import Path

import nbformat as nbf

CELLS = [
    # Cell 1 — install & imports
    nbf.v4.new_code_cell(
        "# Cell 1 — install deps (first run only) + imports + seed\n"
        "%pip install -q -r requirements.txt\n"
        "import time\n"
        "from pathlib import Path\n"
        "\n"
        "import ipywidgets as widgets\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "from IPython.display import display\n"
        "from PIL import Image\n"
        "from tqdm.auto import tqdm\n"
        "\n"
        "from mosaic_core import (\n"
        "    MosaicConfig,\n"
        "    build_faiss_index,\n"
        "    build_report,\n"
        "    ensure_seed_tiles,\n"
        "    export_deepzoom,\n"
        "    knn_candidates,\n"
        "    render_mosaic,\n"
        "    rerank,\n"
        "    scan_tile_pool,\n"
        "    split_target,\n"
        ")\n"
        "\n"
        "np.random.seed(0)"
    ),
    # Cell 2 — config + widgets
    nbf.v4.new_code_cell(
        "# Cell 2 — config & interactive sliders\n"
        "config = MosaicConfig()\n"
        "\n"
        "tile_dir_w = widgets.Text(value=str(config.tile_dir), description='tile_dir')\n"
        "target_w = widgets.Text(value=str(config.target_path), description='target')\n"
        "grid_w_w = widgets.IntSlider(value=config.grid_w, min=20, max=240, step=4, description='grid_w')\n"
        "grid_h_w = widgets.IntSlider(value=config.grid_h, min=12, max=135, step=2, description='grid_h')\n"
        "lambda_w = widgets.FloatSlider(value=config.lambda_repeat, min=0.0, max=5.0, step=0.05, description='λ repeat')\n"
        "mu_w = widgets.FloatSlider(value=config.mu_neighbor, min=0.0, max=5.0, step=0.05, description='μ neighbor')\n"
        "tau_w = widgets.FloatSlider(value=config.tau_transfer, min=0.0, max=1.0, step=0.02, description='τ transfer')\n"
        "\n"
        "def _sync(change=None):\n"
        "    config.tile_dir = Path(tile_dir_w.value)\n"
        "    config.target_path = Path(target_w.value) if target_w.value else None\n"
        "    config.grid_w = grid_w_w.value\n"
        "    config.grid_h = grid_h_w.value\n"
        "    config.lambda_repeat = lambda_w.value\n"
        "    config.mu_neighbor = mu_w.value\n"
        "    config.tau_transfer = tau_w.value\n"
        "\n"
        "for w in (tile_dir_w, target_w, grid_w_w, grid_h_w, lambda_w, mu_w, tau_w):\n"
        "    w.observe(_sync, names='value')\n"
        "_sync()\n"
        "\n"
        "display(widgets.VBox([tile_dir_w, target_w, grid_w_w, grid_h_w, lambda_w, mu_w, tau_w]))"
    ),
    # Cell 3 — tile pool
    nbf.v4.new_code_cell(
        "# Cell 3 — load / seed tile pool\n"
        "ensure_seed_tiles(config.tile_dir)\n"
        "tile_records, bad_files = scan_tile_pool(config.tile_dir, config.cache_path)\n"
        "print(f'扫到 {len(tile_records)} 张 tile,坏图 {len(bad_files)} 张。')"
    ),
    # Cell 4 — target + grid
    nbf.v4.new_code_cell(
        "# Cell 4 — load target image, fallback to gradient if missing\n"
        "def _fallback_target(w=768, h=432):\n"
        "    grad = np.linspace(0, 255, w, dtype=np.uint8)[None, :, None].repeat(h, axis=0).repeat(3, axis=2)\n"
        "    grad[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]\n"
        "    return Image.fromarray(grad)\n"
        "\n"
        "if config.target_path and config.target_path.exists():\n"
        "    target_img = Image.open(config.target_path).convert('RGB')\n"
        "else:\n"
        "    print(f'⚠️  目标图 {config.target_path} 不存在,使用内置渐变兜底')\n"
        "    target_img = _fallback_target()\n"
        "\n"
        "target_lab = split_target(target_img, config.grid_w, config.grid_h)\n"
        "print(f'目标 LAB grid shape = {target_lab.shape}')\n"
        "plt.figure(figsize=(6, 4))\n"
        "plt.imshow(target_img)\n"
        "plt.title('target')\n"
        "plt.axis('off')\n"
        "plt.show()"
    ),
    # Cell 5 — match loop
    nbf.v4.new_code_cell(
        "# Cell 5 — KNN candidates + per-cell rerank with visible reasoning\n"
        "tile_labs = np.stack([t.lab_mean for t in tile_records]).astype(np.float32)\n"
        "index = build_faiss_index(tile_labs)\n"
        "candidates = knn_candidates(target_lab, index, k=config.k_candidates)\n"
        "\n"
        "grid_h, grid_w = target_lab.shape[:2]\n"
        "assignment = np.zeros((grid_h, grid_w), dtype=np.int64)\n"
        "usage_counts: dict[int, int] = {}\n"
        "t0 = time.time()\n"
        "total = grid_h * grid_w\n"
        "\n"
        "for flat in tqdm(range(total), desc='matching'):\n"
        "    r, c = divmod(flat, grid_w)\n"
        "    cand = candidates[flat]\n"
        "    neighbors = []\n"
        "    if c > 0: neighbors.append(int(assignment[r, c - 1]))\n"
        "    if r > 0: neighbors.append(int(assignment[r - 1, c]))\n"
        "    best = rerank(cand, tile_labs, target_lab[r, c], usage_counts, neighbors,\n"
        "                  config.lambda_repeat, config.mu_neighbor)\n"
        "    assignment[r, c] = best\n"
        "    usage_counts[best] = usage_counts.get(best, 0) + 1\n"
        "    if flat % 50 == 0 and tile_records[best].path is not None:\n"
        "        print(f'  ({r:>3},{c:>3}) -> {tile_records[best].path.name}')\n"
        "\n"
        "elapsed = time.time() - t0\n"
        "print(f'匹配完成,耗时 {elapsed:.2f} 秒')"
    ),
    # Cell 6 — render
    nbf.v4.new_code_cell(
        "# Cell 6 — render mosaic\n"
        "mosaic_img = render_mosaic(assignment, tile_records, config.tile_px,\n"
        "                           config.tau_transfer, target_lab)\n"
        "config.out_dir.mkdir(parents=True, exist_ok=True)\n"
        "mosaic_path = config.out_dir / 'mosaic.png'\n"
        "mosaic_img.save(mosaic_path)\n"
        "print(f'saved {mosaic_path}')\n"
        "display(mosaic_img)"
    ),
    # Cell 7 — report
    nbf.v4.new_code_cell(
        "# Cell 7 — self-deprecating report\n"
        "report = build_report(assignment, tile_records, elapsed, bad_files)\n"
        "(config.out_dir / 'report.txt').write_text(report.text, encoding='utf-8')\n"
        "print(report.text)\n"
        "plt.show()  # flushes any pending figure\n"
        "display(report.usage_bar_fig)\n"
        "display(report.cold_wall_fig)"
    ),
    # Cell 8 — deepzoom
    nbf.v4.new_code_cell(
        "# Cell 8 — DeepZoom export\n"
        "index_html = export_deepzoom(mosaic_path, config.out_dir / 'deepzoom')\n"
        "print(f'✅ 已生成 {index_html},在浏览器打开即可无限缩放')"
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    out_path = Path(__file__).parent / "mosaic.ipynb"
    nbf.write(nb, str(out_path))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate `mosaic.ipynb`**

Run from `chatgpt/`:
```bash
python _build_notebook.py
```

Expected: prints `wrote /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/chatgpt/mosaic.ipynb` and the file appears.

- [ ] **Step 3: Verify notebook is valid**

Run:
```bash
python -c "import nbformat; nb = nbformat.read('mosaic.ipynb', as_version=4); print(f'{len(nb.cells)} cells')"
```

Expected: prints `8 cells`.

- [ ] **Step 4: Commit + CHANGELOG**

Append:

```yaml
- date: 2026-04-17
  type: feat
  target: chatgpt/_build_notebook.py + chatgpt/mosaic.ipynb
  change: 实现 _build_notebook.py(nbformat 拼 8 个 cell)+ 生成产物 mosaic.ipynb
  rationale: 直接手写 .ipynb JSON 不可维护;改 cell 内容只改 _build_notebook.py 然后重跑
  action: Cell 1 安装+导入;Cell 2 widgets+config 双向同步;Cell 3-5 pool/target/match;Cell 6 render;Cell 7 report;Cell 8 deepzoom
  result: nbformat 读回 8 cells 无报错;实际执行留给 Task 14
  validation: python -c 'import nbformat; nb = nbformat.read(...); assert len(nb.cells) == 8'
  status: stable
```

Commit:
```bash
git add chatgpt/_build_notebook.py chatgpt/mosaic.ipynb chatgpt/CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat: add _build_notebook.py source + generated mosaic.ipynb

Notebook is regenerated from _build_notebook.py via nbformat.
Hand-editing .ipynb JSON is forbidden; edit the script and rerun.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: End-to-end smoke test

**Files:**
- None to create. This task runs the full pipeline in a fresh state and verifies output artifacts.

- [ ] **Step 1: Run the full pytest suite**

Run from `chatgpt/`:
```bash
pytest tests/ -v
```

Expected: **15 tests PASS** (color 5 + matching 5 + transfer 10 — note: transfer has grown beyond spec's 4 entries; that's fine). If any fail, stop and fix the offending task before proceeding.

Tally check: color (5) + matching (5) + transfer (test_split_target_shape_and_dtype, test_split_target_uniform_gray_constant_lab, test_render_mosaic_output_size, test_render_mosaic_tau_zero_preserves_tile_bytes, test_ensure_seed_tiles_creates_files, test_ensure_seed_tiles_noop_when_nonempty, test_scan_tile_pool_happy_path, test_scan_tile_pool_skips_corrupt_files, test_scan_tile_pool_uses_cache, test_build_report_structural_fields = 10) = **20 tests total**. The spec said "~15 cases"; we landed at 20 because spec-scope testing (bad files, cache reuse, seed no-op, ensure_seed_tiles existence) added natural edges. Acceptable.

- [ ] **Step 2: Clean previous runtime artifacts**

Run from `chatgpt/`:
```bash
rm -rf .cache out my_tiles target.jpg
```

Purpose: exercise the "zero-config first run" path (tile_dir missing → seed fallback; target missing → gradient fallback).

- [ ] **Step 3: Execute the notebook headless**

Run from `chatgpt/`:
```bash
jupyter nbconvert --to notebook --execute --inplace mosaic.ipynb --ExecutePreprocessor.timeout=600
```

Expected:
- Exit code 0
- `mosaic.ipynb` updates in place with cell outputs
- `my_tiles/` directory contains 200 `seed_*.jpg` files
- `out/mosaic.png` exists
- `out/report.txt` exists
- `out/deepzoom/index.html` exists
- `out/deepzoom/mosaic.dzi` exists
- `out/deepzoom/mosaic_files/` directory exists with pyramid tiles

If widget-related warnings appear, they are non-fatal. If the notebook errors out, the cell traceback will be in the output — read it and fix the offending module.

- [ ] **Step 4: Verify output artifacts**

Run from `chatgpt/`:
```bash
ls -la out/mosaic.png out/report.txt out/deepzoom/index.html out/deepzoom/mosaic.dzi
cat out/report.txt | head -20
```

Expected: all four files listed with non-zero size; report.txt shows the headline + TOP 5 + cold-wall sections.

- [ ] **Step 5: Manual browser check (optional but recommended)**

Open `out/deepzoom/index.html` in a browser, pan and zoom. Seeing the pyramid render and scroll fluidly = DeepZoom is correctly wired. This cannot be automated — skip if running in a headless agent context.

- [ ] **Step 6: Re-run to exercise cache path**

Run again:
```bash
jupyter nbconvert --to notebook --execute --inplace mosaic.ipynb --ExecutePreprocessor.timeout=600
```

Expected: second run is faster in Cell 3 (cache hit), output artifacts still generated.

- [ ] **Step 7: Commit + final CHANGELOG entry**

Append:

```yaml
- date: 2026-04-17
  type: feat
  target: chatgpt/
  change: 端到端冒烟通过——pytest 全绿 (20/20),nbconvert headless 跑完 mosaic.ipynb 产出 mosaic.png + report.txt + deepzoom/index.html
  rationale: 验证 MVP 验收标准;零配置首跑 + 缓存二跑两条路径都覆盖
  action: 清空 .cache/out/my_tiles/target.jpg 后 nbconvert --execute;再二次跑验证缓存命中
  result: 首跑生成 200 张 seed tile,目标用渐变兜底;out/ 下四件产物齐全;二跑更快且产物再生
  validation: pytest tests/ 全绿;ls out/mosaic.png out/report.txt out/deepzoom/index.html out/deepzoom/mosaic.dzi 均存在且非零字节
  status: stable
```

Commit:
```bash
git add chatgpt/CHANGELOG.md chatgpt/mosaic.ipynb
git commit -m "$(cat <<'EOF'
test: end-to-end smoke — pytest 20/20 + nbconvert --execute green

Headless notebook run produces mosaic.png, report.txt, and
deepzoom/index.html in the zero-config first-run path. Re-run
exercises the pickle cache path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Verification checklist (spec §11)

After all tasks complete, confirm each MVP acceptance criterion from the spec:

- [ ] **§11.1** Fresh `pip install -r requirements.txt` + `jupyter lab mosaic.ipynb` + Run All → success (verified in Task 14 step 3 via nbconvert).
- [ ] **§11.2** `pytest tests/` all green (Task 14 step 1).
- [ ] **§11.3** Replacing `my_tiles/` with a real photo directory + re-running still produces output (manual follow-up — drop photos in `my_tiles/` and re-exec; not scripted).
- [ ] **§11.4** Moving a slider in the notebook + re-executing Cell 5 and Cell 6 produces visibly different output (manual — requires interactive jupyter session).
- [ ] **§11.5** Opening `out/deepzoom/index.html` in a browser and zooming all the way in shows single underlying tiles (Task 14 step 5, manual).

Items §11.3–§11.5 are deferred to owner's first real session with a photo library; they are not part of the agent-driven acceptance.

---

## Self-review notes

**Spec coverage:**
- §2 goals 1–6: all implemented (Tasks 2–13).
- §4 module table: every function has its task (lab_mean T2, ciede2000 T3, reinhard T4, split T5, render T6, faiss T7, knn T7, rerank T8, ensure_seed T9, scan T10, build_report T11, export_deepzoom T12).
- §4 dataclasses: TileRecord + MosaicConfig + ReportBundle defined in T2 skeleton.
- §5 data flow: realized by notebook cells T13.
- §6 notebook cells: all 8 produced by `_build_notebook.py` in T13.
- §7 error handling: tile_dir empty → T9 seed fallback; target missing → T13 Cell 4 gradient; corrupt JPG → T10 bad_files; cache version mismatch → T10 try/except + recompute; grid > tiles → T7 effective_k clamp.
- §8 tests: color 5 (target was 4, we added test_ciede2000_nonzero) + matching 5 + transfer 10 (target was 4 — we expanded to cover ensure_seed no-op, scan cache reuse, seed file count, build_report). **Overshoot is acceptable**: spec said "~15 cases" as a rough size estimate, not a ceiling.
- §9 dependencies: installed in T1.
- §10 CHANGELOG: initialized in T1 with first entry, every feature task appends a fresh entry.
- §11 acceptance: verified automatable parts in T14; manual parts deferred as noted.

**No placeholders:** every code block is complete. No "TBD" / "TODO" / "etc." / "similar to…" strings.

**Type consistency:** function signatures in later tasks match their introduction:
- `TileRecord.lab_mean: np.ndarray`, `.rgb_thumb: np.ndarray` — used consistently in scan, render, report.
- `scan_tile_pool -> tuple[list[TileRecord], list[Path]]` — consumed correctly by `build_report`.
- `rerank(candidate_idxs, tile_labs, target_lab_patch, usage_counts, neighbor_tile_idxs, lambda_repeat, mu_neighbor)` — matches Cell 5 call site in `_build_notebook.py`.
- `render_mosaic(assignment, tile_records, tile_px, tau, target_lab)` — matches Cell 6 call site.
- `build_report(assignment, tile_records, elapsed_seconds, bad_files)` — matches Cell 7 call site (Cell 7 passes `elapsed` as the 3rd positional arg, matching the `elapsed_seconds` parameter name).
- `export_deepzoom(png_path, out_dir) -> Path` — Cell 8 captures the returned path.

All consistent. Plan is ready for execution.
