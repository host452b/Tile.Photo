# Photomosaic Toy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `doubao/` 下实现一个本地 photomosaic 玩具：把目标图分块，用用户照片库匹配 + 渲染成"远看是目标图、近看是一堆小照片"的马赛克，同时产出自嘲式报告和 DeepZoom HTML。

**Architecture:** Thin library (`src/mosaic/` 6 模块) + Jupyter notebook (`photomosaic.ipynb` 8 cell) 编排调用。所有参数集中于 `MosaicConfig` dataclass；核心匹配循环 numpy 暴力 top-K（spec 原写 faiss，实施期为避免 arm64 wheel 风险改用 numpy argpartition，性能在 N≤20k 时等价，见 Task 1 备注）；渲染支持可选 Reinhard LAB 色调迁移。

**Tech Stack:** Python 3.11+, uv (env), Pillow, numpy, scikit-image, tqdm, matplotlib, pytest, OpenSeadragon (CDN)。

---

## Implementation Notes (Deviations from Spec)

**Spec § 4.3 said `faiss-cpu`，计划改为 numpy。** 原因：
1. N_tiles ≤ 5k × top-K=50 查询，`np.argpartition` 在 LAB 距离矩阵上毫秒级，faiss 无性能优势
2. faiss-cpu 在 macOS arm64 的 pip wheel 历史上偶发失败（spec §13 把这条列为 open question）
3. numpy 方案 3 行，faiss 方案 10+ 行 + 安装风险；符合"玩具优先"原则

**Spec § 4.6 说 `deepzoom` pypi 包，计划改为手写 DZI。** 原因：
1. `deepzoom` pypi 自 2018 后无更新
2. 手写 DZI XML + 瓦片金字塔 ~80 行，零外部依赖
3. OpenSeadragon 从 CDN 加载

其余严格按 spec（section 1-14）实施。

---

## File Structure

| 文件 | 责任 | Task |
|------|------|------|
| `pyproject.toml` | uv/hatchling 元信息 + deps | 1 |
| `.gitignore` | 忽略 .venv .cache output __pycache__ | 1 |
| `README.md` | 怎么跑 | 1 (stub) / 12 (正式) |
| `CHANGELOG.md` | agent-oriented changelog | 12 |
| `src/mosaic/__init__.py` | 包 re-export | 1 (空) |
| `src/mosaic/config.py` | `MosaicConfig` dataclass | 2 |
| `src/mosaic/tiles.py` | 扫描 + LAB 均色 + pickle 缓存 | 3 |
| `src/mosaic/match.py` | `split_target` + `match_all_tiles` | 4, 5, 6 |
| `src/mosaic/render.py` | `reinhard_transfer` + `render_mosaic` | 7 |
| `src/mosaic/report.py` | 文字 + 柱状图 + 冷宫墙 | 8 |
| `src/mosaic/deepzoom.py` | DZI 切片 + OpenSeadragon HTML | 9 |
| `tests/__init__.py` | 空 | 1 |
| `tests/test_match.py` | match.py 单元测试 | 4, 5, 6 |
| `tests/test_render.py` | render.py 单元测试 | 7 |
| `tests/test_e2e.py` | 合成数据端到端 smoke | 11 |
| `photomosaic.ipynb` | 8-cell 编排 notebook | 10 |

---

## Task 1: 项目骨架 + uv 环境

**Files:**
- Create: `doubao/pyproject.toml`
- Create: `doubao/.gitignore`
- Create: `doubao/README.md`
- Create: `doubao/src/mosaic/__init__.py`
- Create: `doubao/tests/__init__.py`

- [ ] **Step 1: 创建 `doubao/pyproject.toml`**

```toml
[project]
name = "mosaic"
version = "0.1.0"
description = "Photomosaic toy: reconstruct one photo from many"
requires-python = ">=3.11"
dependencies = [
    "pillow>=10.0",
    "numpy>=1.24",
    "scikit-image>=0.22",
    "tqdm>=4.60",
    "matplotlib>=3.5",
    "ipykernel>=6.0",
]

[dependency-groups]
dev = [
    "pytest>=7.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mosaic"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: 创建 `doubao/.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.cache/
output/
.ipynb_checkpoints/
.DS_Store
```

- [ ] **Step 3: 创建 `doubao/README.md` (stub)**

```markdown
# Photomosaic Toy

本地玩具：用你的照片库重组一张目标图。

## 跑起来

```bash
cd doubao
uv sync
uv run jupyter lab photomosaic.ipynb
```

Cell 2 里填 `tile_source_dir` 和 `target_image` 路径后从上往下跑。

详细设计见 `docs/superpowers/specs/2026-04-17-photomosaic-toy-design.md`。
```

- [ ] **Step 4: 创建 `doubao/src/mosaic/__init__.py`**

```python
"""Photomosaic toy package."""
```

- [ ] **Step 5: 创建 `doubao/tests/__init__.py`**

空文件：

```python
```

- [ ] **Step 6: 创建环境并验证**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv sync --all-groups
```
Expected: 成功创建 `.venv/`，安装依赖，无报错。

- [ ] **Step 7: 验证包可导入**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run python -c "import mosaic; print(mosaic.__doc__)"
```
Expected: 打印 `Photomosaic toy package.`

- [ ] **Step 8: 验证 pytest 工作（空用例）**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run pytest
```
Expected: `collected 0 items` 无错误退出。

- [ ] **Step 9: Commit**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo
git add doubao/pyproject.toml doubao/.gitignore doubao/README.md \
        doubao/src/mosaic/__init__.py doubao/tests/__init__.py
git commit -m "$(cat <<'EOF'
chore(doubao): scaffold mosaic package with uv

Create src/mosaic + tests, pyproject with hatchling src-layout,
runtime deps (pillow, numpy, skimage, tqdm, matplotlib) + dev group (pytest).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `config.py` — MosaicConfig dataclass

**Files:**
- Create: `doubao/src/mosaic/config.py`

- [ ] **Step 1: 创建 `doubao/src/mosaic/config.py`**

```python
"""Single source of truth for all mosaic parameters."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MosaicConfig:
    # paths
    tile_source_dir: Path
    target_image: Path
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))
    output_dir: Path = field(default_factory=lambda: Path("output"))

    # grid
    grid: tuple[int, int] = (120, 68)   # (cols, rows)
    tile_px: int = 16                    # 每 tile 渲染像素

    # matching
    candidate_k: int = 50       # 先取 top-k 颜色候选
    lambda_reuse: float = 0.3   # 重复惩罚
    mu_neighbor: float = 0.2    # 邻居相似度惩罚

    # rendering
    tau_tone: float = 0.5       # 色调迁移强度 0..1

    # behavior
    verbose: bool = True
    mode: str = "classic"       # 预留给 cursed_* 模式

    def validate(self) -> None:
        """Call before running the pipeline; raise on misconfiguration."""
        if not self.tile_source_dir.exists():
            raise ValueError(
                f"tile_source_dir does not exist: {self.tile_source_dir}. "
                "Please set it in cell 2."
            )
        if not self.target_image.exists():
            raise ValueError(
                f"target_image does not exist: {self.target_image}. "
                "Please set it in cell 2."
            )
        if not 0.0 <= self.tau_tone <= 1.0:
            raise ValueError(f"tau_tone must be in [0, 1], got {self.tau_tone}")
        if self.tile_px < 4:
            raise ValueError(f"tile_px too small: {self.tile_px}")
        cols, rows = self.grid
        if cols < 1 or rows < 1:
            raise ValueError(f"grid must be positive, got {self.grid}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: 冒烟 import**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run python -c "
from mosaic.config import MosaicConfig
from pathlib import Path
c = MosaicConfig(tile_source_dir=Path('/tmp'), target_image=Path('/etc/hosts'))
c.validate()
print('OK:', c.grid, c.tau_tone)
"
```
Expected: `OK: (120, 68) 0.5`

- [ ] **Step 3: 冒烟 validate 错误路径**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run python -c "
from mosaic.config import MosaicConfig
from pathlib import Path
c = MosaicConfig(tile_source_dir=Path('/no/such/path'), target_image=Path('/etc/hosts'))
try:
    c.validate()
except ValueError as e:
    print('OK raised:', str(e)[:60])
"
```
Expected: `OK raised: tile_source_dir does not exist: /no/such/path...`

- [ ] **Step 4: Commit**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo
git add doubao/src/mosaic/config.py
git commit -m "$(cat <<'EOF'
feat(mosaic): add MosaicConfig dataclass

All parameters (paths, grid, λ/μ/τ, candidate_k, verbose, mode) in one
dataclass with validate() — notebook cell 2 becomes single config entry point.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `tiles.py` — 扫描 + LAB 均色 + 缓存

**Files:**
- Create: `doubao/src/mosaic/tiles.py`

按 spec § 4.2，该模块不写单测（IO 密集，fixture 成本高）；正确性靠 Task 11 端到端 smoke 覆盖。

- [ ] **Step 1: 创建 `doubao/src/mosaic/tiles.py`**

```python
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
```

- [ ] **Step 2: 冒烟 import**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run python -c "from mosaic.tiles import TilePool, scan_tile_pool, load_or_build; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: 冒烟 build（临时合成数据）**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run python - <<'PY'
import tempfile, numpy as np
from pathlib import Path
from PIL import Image
from mosaic.tiles import load_or_build

with tempfile.TemporaryDirectory() as td:
    src = Path(td) / "src"
    src.mkdir()
    for i, color in enumerate([(255,0,0),(0,255,0),(0,0,255)]):
        Image.new("RGB", (50,50), color).save(src / f"t{i}.png")
    pool = load_or_build(src, tile_px=16, cache_dir=Path(td)/"cache")
    print(f"N={len(pool)}, lab_means shape={pool.lab_means.shape}, "
          f"thumbs shape={pool.thumbnails.shape}")
PY
```
Expected: `N=3, lab_means shape=(3, 3), thumbs shape=(3, 16, 16, 3)`

- [ ] **Step 4: Commit**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo
git add doubao/src/mosaic/tiles.py
git commit -m "$(cat <<'EOF'
feat(mosaic): add tile pool scanner with pickle cache

Recursive scan with junk-dir skips, center-crop+resize to tile_px²,
LAB mean via skimage, mtime-keyed pickle cache in cache_dir/tiles_{px}.pkl.
Corrupted tiles warn+skip per toy-level error policy.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `match.py:split_target` (TDD)

**Files:**
- Create: `doubao/src/mosaic/match.py`
- Create: `doubao/tests/test_match.py`

- [ ] **Step 1: 写失败测试**

Create `doubao/tests/test_match.py`:

```python
"""Tests for src/mosaic/match.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def two_tone_target(tmp_path: Path) -> Path:
    """100×100 image: left half red (255,0,0), right half blue (0,0,255)."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :50] = [255, 0, 0]
    arr[:, 50:] = [0, 0, 255]
    p = tmp_path / "target.png"
    Image.fromarray(arr).save(p)
    return p


def test_split_target_shape_and_lab(two_tone_target: Path) -> None:
    from mosaic.match import split_target

    cells = split_target(two_tone_target, grid=(4, 2))  # 4 cols × 2 rows
    assert cells.shape == (2, 4, 3)  # (rows, cols, LAB)
    # Left two columns are red, right two columns are blue — L differs, a differs strongly
    left_a = cells[:, :2, 1].mean()   # red: +a
    right_a = cells[:, 2:, 1].mean()  # blue: near 0 a but +b negative
    assert left_a > 20  # red is strongly +a
    assert right_a < 0  # blue is -a
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run pytest tests/test_match.py -v
```
Expected: `ImportError: cannot import name 'split_target' from 'mosaic.match'` 或 `ModuleNotFoundError`

- [ ] **Step 3: 写 `split_target` 实现**

Create `doubao/src/mosaic/match.py`:

```python
"""Split target + match tiles to cells."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def split_target(target_path: Path, grid: tuple[int, int]) -> np.ndarray:
    """Load target, resize to (cols, rows) via block averaging, return LAB cells.

    Returns array of shape (rows, cols, 3) with LAB mean per cell.
    """
    cols, rows = grid
    with Image.open(target_path) as im:
        im = im.convert("RGB")
        # Resize so each pixel is one cell's mean; PIL LANCZOS approximates mean
        im = im.resize((cols, rows), Image.BOX)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    lab = rgb2lab(arr)  # (rows, cols, 3)
    return lab.astype(np.float32)
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run pytest tests/test_match.py::test_split_target_shape_and_lab -v
```
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo
git add doubao/src/mosaic/match.py doubao/tests/test_match.py
git commit -m "$(cat <<'EOF'
feat(mosaic): add match.split_target with LAB cell grid

BOX resize to (cols, rows) approximates per-cell mean; rgb2lab converts.
Shape (rows, cols, 3). Tested with two-tone target on 4×2 grid.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `match.py:match_all_tiles` 基础版（无惩罚，TDD）

**Files:**
- Modify: `doubao/src/mosaic/match.py`
- Modify: `doubao/tests/test_match.py`

- [ ] **Step 1: 追加失败测试**

Append to `doubao/tests/test_match.py`:

```python
import numpy as np
from mosaic.tiles import TilePool


def _synthetic_pool(tile_px: int = 8) -> TilePool:
    """5 solid-color tiles — gray, warm-light, cool-dark, red, green."""
    colors = [
        (128, 128, 128),   # gray
        (230, 200, 180),   # warm light
        (40, 50, 70),      # cool dark
        (220, 40, 40),     # red
        (40, 200, 60),     # green
    ]
    thumbs = np.stack([np.full((tile_px, tile_px, 3), c, dtype=np.uint8) for c in colors])
    from skimage.color import rgb2lab
    lab_means = np.stack([rgb2lab(t.astype(np.float32) / 255.0).reshape(-1, 3).mean(0)
                          for t in thumbs]).astype(np.float32)
    return TilePool(
        paths=[f"tile_{i}.png" for i in range(len(colors))],
        lab_means=lab_means,
        thumbnails=thumbs,
    )


def test_match_all_tiles_picks_nearest_color_when_no_penalty() -> None:
    """With λ=μ=0, each cell should pick the color-closest tile."""
    from mosaic.match import match_all_tiles
    from mosaic.config import MosaicConfig

    pool = _synthetic_pool()
    # Build target cells that each match one tile exactly
    target = pool.lab_means.reshape(1, 5, 3).copy()  # 1×5 grid, each cell = one tile's LAB

    cfg = MosaicConfig(
        tile_source_dir=Path("/tmp"),
        target_image=Path("/tmp"),
        grid=(5, 1),
        tile_px=8,
        candidate_k=5,
        lambda_reuse=0.0,
        mu_neighbor=0.0,
        verbose=False,
    )
    assignment, use_count = match_all_tiles(target, pool, cfg)
    assert assignment.shape == (1, 5)
    # Each cell must pick its own tile (identity mapping)
    np.testing.assert_array_equal(assignment[0], np.arange(5))
    assert sum(use_count.values()) == 5
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run pytest tests/test_match.py::test_match_all_tiles_picks_nearest_color_when_no_penalty -v
```
Expected: `ImportError` 或 `AttributeError: module 'mosaic.match' has no attribute 'match_all_tiles'`

- [ ] **Step 3: 实现 `match_all_tiles`（无惩罚版）**

Append to `doubao/src/mosaic/match.py`:

```python
from collections import defaultdict

from .config import MosaicConfig
from .tiles import TilePool


def _top_k_candidates(target_lab: np.ndarray, pool_lab: np.ndarray, k: int) -> np.ndarray:
    """Return indices of k nearest pool entries to target_lab by L2 in LAB.

    target_lab shape (3,), pool_lab shape (N, 3). Returns (min(k, N),) int array.
    """
    d2 = ((pool_lab - target_lab) ** 2).sum(axis=1)
    k = min(k, len(pool_lab))
    idx = np.argpartition(d2, k - 1)[:k]
    # Sort by distance ascending for deterministic selection
    return idx[np.argsort(d2[idx])]


def match_all_tiles(
    target_cells: np.ndarray,
    pool: TilePool,
    cfg: MosaicConfig,
) -> tuple[np.ndarray, dict[int, int]]:
    """Row-major match each target cell to a pool tile.

    Returns (assignment (rows, cols) int32, use_count dict[tile_idx -> count]).
    """
    rows, cols, _ = target_cells.shape
    assignment = np.full((rows, cols), -1, dtype=np.int32)
    use_count: dict[int, int] = defaultdict(int)

    for r in range(rows):
        for c in range(cols):
            t_lab = target_cells[r, c]
            cand = _top_k_candidates(t_lab, pool.lab_means, cfg.candidate_k)
            color_dist = np.linalg.norm(pool.lab_means[cand] - t_lab, axis=1)
            scores = color_dist.copy()
            # (penalties added in Task 6)
            best_local = int(np.argmin(scores))
            best = int(cand[best_local])
            assignment[r, c] = best
            use_count[best] += 1
            if cfg.verbose:
                print(f"[{r:3d},{c:3d}] -> tile#{best} dist={color_dist[best_local]:.1f} "
                      f"reuse={use_count[best]}")

    return assignment, dict(use_count)
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run pytest tests/test_match.py::test_match_all_tiles_picks_nearest_color_when_no_penalty -v
```
Expected: `1 passed`

- [ ] **Step 5: 跑所有已有测试确认无回归**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run pytest tests/ -v
```
Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo
git add doubao/src/mosaic/match.py doubao/tests/test_match.py
git commit -m "$(cat <<'EOF'
feat(mosaic): add match_all_tiles with nearest-color selection

Row-major sweep; per cell takes top-k candidates by LAB L2 via
argpartition, picks min color_dist. Penalty hooks wired but zeroed
until Task 6. Verbose flag prints the decision per cell.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `match.py` — 重复 + 邻居惩罚（TDD）

**Files:**
- Modify: `doubao/src/mosaic/match.py`
- Modify: `doubao/tests/test_match.py`

- [ ] **Step 1: 追加失败测试**

Append to `doubao/tests/test_match.py`:

```python
def test_reuse_penalty_spreads_usage() -> None:
    """With 10 identical target cells and λ high, no single tile should dominate."""
    from mosaic.match import match_all_tiles
    from mosaic.config import MosaicConfig

    pool = _synthetic_pool()
    # All 10 cells want the same color (tile 1: warm light)
    target_lab = pool.lab_means[1]
    target = np.tile(target_lab, (1, 10, 1)).astype(np.float32)  # (1, 10, 3)

    cfg_no_pen = MosaicConfig(
        tile_source_dir=Path("/tmp"), target_image=Path("/tmp"),
        grid=(10, 1), tile_px=8, candidate_k=5,
        lambda_reuse=0.0, mu_neighbor=0.0, verbose=False,
    )
    assign_no, use_no = match_all_tiles(target, pool, cfg_no_pen)
    assert use_no.get(1, 0) == 10, f"expected all tile 1 without penalty, got {use_no}"

    cfg_pen = MosaicConfig(
        tile_source_dir=Path("/tmp"), target_image=Path("/tmp"),
        grid=(10, 1), tile_px=8, candidate_k=5,
        lambda_reuse=5.0, mu_neighbor=0.0, verbose=False,
    )
    assign_pen, use_pen = match_all_tiles(target, pool, cfg_pen)
    assert max(use_pen.values()) < 10, f"reuse penalty failed: {use_pen}"
    assert len(use_pen) >= 2, f"penalty should use ≥2 distinct tiles: {use_pen}"


def test_neighbor_penalty_differentiates_adjacent() -> None:
    """With μ high and 2-cell target, the two cells should differ."""
    from mosaic.match import match_all_tiles
    from mosaic.config import MosaicConfig

    pool = _synthetic_pool()
    # Target: one cell midway between tile 0 (gray) and tile 1 (warm light)
    mid = (pool.lab_means[0] + pool.lab_means[1]) / 2.0
    target = np.tile(mid, (1, 2, 1)).astype(np.float32)  # (1, 2, 3)

    cfg_no_mu = MosaicConfig(
        tile_source_dir=Path("/tmp"), target_image=Path("/tmp"),
        grid=(2, 1), tile_px=8, candidate_k=5,
        lambda_reuse=0.0, mu_neighbor=0.0, verbose=False,
    )
    assign_no, _ = match_all_tiles(target, pool, cfg_no_mu)

    cfg_mu = MosaicConfig(
        tile_source_dir=Path("/tmp"), target_image=Path("/tmp"),
        grid=(2, 1), tile_px=8, candidate_k=5,
        lambda_reuse=0.0, mu_neighbor=50.0, verbose=False,
    )
    assign_mu, _ = match_all_tiles(target, pool, cfg_mu)
    # Without μ both cells may (or may not) pick same; with large μ they must differ
    assert assign_mu[0, 0] != assign_mu[0, 1], \
        f"neighbor penalty failed, both cells picked {assign_mu[0, 0]}"
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run pytest tests/test_match.py::test_reuse_penalty_spreads_usage tests/test_match.py::test_neighbor_penalty_differentiates_adjacent -v
```
Expected: 两项失败 —— reuse 期望 use_pen 分散但实际仍全部 tile 1；neighbor 期望 [0,0]≠[0,1] 但实际相等。

- [ ] **Step 3: 加入惩罚项到 `match_all_tiles`**

Replace the `# (penalties added in Task 6)` comment block in `doubao/src/mosaic/match.py` by editing the function body. The full updated `match_all_tiles` should read:

```python
def match_all_tiles(
    target_cells: np.ndarray,
    pool: TilePool,
    cfg: MosaicConfig,
) -> tuple[np.ndarray, dict[int, int]]:
    """Row-major match each target cell to a pool tile.

    Score = color_dist
          + λ · log(1 + use_count[cand])
          + μ · mean LAB dist to already-placed left + top neighbor

    Returns (assignment (rows, cols) int32, use_count dict[tile_idx -> count]).
    """
    rows, cols, _ = target_cells.shape
    assignment = np.full((rows, cols), -1, dtype=np.int32)
    use_count: dict[int, int] = defaultdict(int)
    lam = float(cfg.lambda_reuse)
    mu = float(cfg.mu_neighbor)

    for r in range(rows):
        for c in range(cols):
            t_lab = target_cells[r, c]
            cand = _top_k_candidates(t_lab, pool.lab_means, cfg.candidate_k)
            cand_lab = pool.lab_means[cand]
            color_dist = np.linalg.norm(cand_lab - t_lab, axis=1)

            reuse_term = np.zeros_like(color_dist)
            if lam > 0:
                uses = np.array([use_count.get(int(i), 0) for i in cand], dtype=np.float32)
                reuse_term = lam * np.log1p(uses)

            neigh_term = np.zeros_like(color_dist)
            if mu > 0:
                neigh_labs: list[np.ndarray] = []
                if c > 0 and assignment[r, c - 1] >= 0:
                    neigh_labs.append(pool.lab_means[assignment[r, c - 1]])
                if r > 0 and assignment[r - 1, c] >= 0:
                    neigh_labs.append(pool.lab_means[assignment[r - 1, c]])
                if neigh_labs:
                    neigh_arr = np.stack(neigh_labs)  # (M, 3)
                    # For each candidate, mean L2 to neighbors; HIGH dist = DIVERSE, we want
                    # to ENCOURAGE difference from neighbor, so subtract? No — μ penalizes
                    # SIMILARITY to neighbor. Penalty = -mean_dist (closer -> higher penalty).
                    # We negate so small dist (similar) -> high score; we want diversity,
                    # so penalty = -mean_dist * mu scaled... Reformulate: similarity = 1/(1+dist).
                    mean_dist = np.linalg.norm(
                        cand_lab[:, None, :] - neigh_arr[None, :, :], axis=2
                    ).mean(axis=1)
                    similarity = 1.0 / (1.0 + mean_dist)
                    neigh_term = mu * similarity

            scores = color_dist + reuse_term + neigh_term
            best_local = int(np.argmin(scores))
            best = int(cand[best_local])
            assignment[r, c] = best
            use_count[best] += 1
            if cfg.verbose:
                print(f"[{r:3d},{c:3d}] -> tile#{best} dist={color_dist[best_local]:.1f} "
                      f"reuse={use_count[best]} pen={reuse_term[best_local] + neigh_term[best_local]:.2f}")

    return assignment, dict(use_count)
```

- [ ] **Step 4: 运行新增测试**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run pytest tests/test_match.py::test_reuse_penalty_spreads_usage tests/test_match.py::test_neighbor_penalty_differentiates_adjacent -v
```
Expected: `2 passed`

- [ ] **Step 5: 跑所有测试确认无回归**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run pytest tests/ -v
```
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo
git add doubao/src/mosaic/match.py doubao/tests/test_match.py
git commit -m "$(cat <<'EOF'
feat(mosaic): add reuse + neighbor penalties to match_all_tiles

λ·log(1+use_count) spreads usage across pool; μ·(1/(1+mean_neighbor_dist))
discourages adjacent repeats (only checks immediate left + top per spec).
Test: 10 identical cells with λ=5 yields ≥2 distinct tiles; 2-cell grid
with μ=50 forces differentiation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `render.py` — Reinhard 色调迁移 + 贴图（TDD）

**Files:**
- Create: `doubao/src/mosaic/render.py`
- Create: `doubao/tests/test_render.py`

- [ ] **Step 1: 写失败测试**

Create `doubao/tests/test_render.py`:

```python
"""Tests for src/mosaic/render.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _single_tile_pool():
    from mosaic.tiles import TilePool
    from skimage.color import rgb2lab
    # One red tile, 8×8
    thumb = np.full((8, 8, 3), [200, 40, 40], dtype=np.uint8)
    lab_mean = rgb2lab(thumb.astype(np.float32) / 255.0).reshape(-1, 3).mean(0).astype(np.float32)
    return TilePool(paths=["red.png"], lab_means=lab_mean.reshape(1, 3), thumbnails=thumb[None, ...])


def test_render_tau_zero_preserves_tile() -> None:
    """τ=0 means no tone transfer — output block equals tile's original pixels."""
    from mosaic.render import render_mosaic
    from mosaic.config import MosaicConfig

    pool = _single_tile_pool()
    target_cells = np.array([[[50.0, 0.0, 0.0]]], dtype=np.float32)  # 1×1 gray target
    assignment = np.array([[0]], dtype=np.int32)
    cfg = MosaicConfig(
        tile_source_dir=Path("/tmp"), target_image=Path("/tmp"),
        grid=(1, 1), tile_px=8, tau_tone=0.0, verbose=False,
    )
    img = render_mosaic(assignment, pool, target_cells, cfg)
    arr = np.asarray(img)
    assert arr.shape == (8, 8, 3)
    np.testing.assert_array_equal(arr, pool.thumbnails[0])


def test_render_tau_one_matches_target_lab() -> None:
    """τ=1 means full tone transfer — output LAB mean ≈ target LAB."""
    from mosaic.render import render_mosaic
    from mosaic.config import MosaicConfig
    from skimage.color import rgb2lab

    pool = _single_tile_pool()
    target_lab = np.array([50.0, 0.0, 0.0], dtype=np.float32)  # neutral gray
    target_cells = target_lab.reshape(1, 1, 3)
    assignment = np.array([[0]], dtype=np.int32)
    cfg = MosaicConfig(
        tile_source_dir=Path("/tmp"), target_image=Path("/tmp"),
        grid=(1, 1), tile_px=8, tau_tone=1.0, verbose=False,
    )
    img = render_mosaic(assignment, pool, target_cells, cfg)
    arr = np.asarray(img).astype(np.float32) / 255.0
    out_lab_mean = rgb2lab(arr).reshape(-1, 3).mean(0)
    np.testing.assert_allclose(out_lab_mean, target_lab, atol=2.0)
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run pytest tests/test_render.py -v
```
Expected: `ModuleNotFoundError: No module named 'mosaic.render'`

- [ ] **Step 3: 实现 `render.py`**

Create `doubao/src/mosaic/render.py`:

```python
"""Render the mosaic: optional Reinhard tone transfer + paste tiles."""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab

from .config import MosaicConfig
from .tiles import TilePool


def reinhard_transfer(
    tile_rgb: np.ndarray,
    target_lab: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Shift tile's LAB mean + std toward target_lab by factor tau.

    tile_rgb: (H, W, 3) uint8
    target_lab: (3,) float — target cell's LAB mean
    tau: 0..1 — 0 means no change, 1 means fully match target LAB mean
    Returns (H, W, 3) uint8
    """
    if tau <= 0.0:
        return tile_rgb
    lab = rgb2lab(tile_rgb.astype(np.float32) / 255.0)
    cur_mean = lab.reshape(-1, 3).mean(0)
    shifted = lab + tau * (target_lab - cur_mean)
    # Clip L to [0, 100]; a,b can be wide but clip to a safe range
    shifted[..., 0] = np.clip(shifted[..., 0], 0, 100)
    shifted[..., 1] = np.clip(shifted[..., 1], -128, 127)
    shifted[..., 2] = np.clip(shifted[..., 2], -128, 127)
    rgb = lab2rgb(shifted)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def render_mosaic(
    assignment: np.ndarray,
    pool: TilePool,
    target_cells: np.ndarray,
    cfg: MosaicConfig,
) -> Image.Image:
    """Compose the final mosaic.

    assignment: (rows, cols) int — tile index per cell
    target_cells: (rows, cols, 3) — LAB mean per cell (for tone transfer)
    Returns PIL.Image of size (cols * tile_px, rows * tile_px).
    """
    rows, cols = assignment.shape
    tile_px = cfg.tile_px
    canvas = np.zeros((rows * tile_px, cols * tile_px, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            idx = int(assignment[r, c])
            thumb = pool.thumbnails[idx]
            if cfg.tau_tone > 0.0:
                thumb = reinhard_transfer(thumb, target_cells[r, c], cfg.tau_tone)
            canvas[r * tile_px : (r + 1) * tile_px,
                   c * tile_px : (c + 1) * tile_px] = thumb

    return Image.fromarray(canvas)
```

- [ ] **Step 4: 运行 render 测试**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run pytest tests/test_render.py -v
```
Expected: `2 passed`

- [ ] **Step 5: 跑所有测试**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run pytest tests/ -v
```
Expected: `6 passed`

- [ ] **Step 6: Commit**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo
git add doubao/src/mosaic/render.py doubao/tests/test_render.py
git commit -m "$(cat <<'EOF'
feat(mosaic): add Reinhard tone transfer + mosaic render

τ=0 short-circuits to original tile (near-view preserves photo);
τ=1 shifts LAB mean fully to target (far-view matches target image);
linear blend in between. render_mosaic pastes tiles onto a
(cols*tile_px × rows*tile_px) canvas. Tests cover both τ extremes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `report.py` — 自嘲文字 + 柱状图 + 冷宫墙

**Files:**
- Create: `doubao/src/mosaic/report.py`

按 spec § 4.5 不写单测；smoke 由 Task 11 覆盖。

- [ ] **Step 1: 创建 `doubao/src/mosaic/report.py`**

```python
"""Self-deprecating report + usage histogram + cold-photo wall."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .tiles import TilePool


def generate_text_report(
    use_count: dict[int, int],
    pool: TilePool,
    total_cells: int,
) -> str:
    """Self-deprecating ASCII report."""
    n_used = sum(1 for v in use_count.values() if v > 0)
    n_pool = len(pool)

    # Top used
    top5 = sorted(use_count.items(), key=lambda kv: kv[1], reverse=True)[:5]
    # Cold wall: unused tiles
    used_idx = {i for i, v in use_count.items() if v > 0}
    cold_idx = [i for i in range(n_pool) if i not in used_idx]

    lines = [
        f"本次使用了你 {n_pool:,} 张照片里的 {n_used:,} 张。",
        f"总格数 {total_cells:,}，平均每张底图被用 {total_cells / max(n_used, 1):.1f} 次。",
        "",
        "用得最多的 TOP 5：",
    ]
    for idx, cnt in top5:
        name = Path(str(pool.paths[idx])).name
        lines.append(f"  - {name} ({cnt} 次)")
    lines.append("")
    cold_show = cold_idx[:5]
    if cold_show:
        lines.append(f"冷宫照片 TOP 5（共 {len(cold_idx)} 张一次都没被用）：")
        for idx in cold_show:
            name = Path(str(pool.paths[idx])).name
            lines.append(f"  - {name}")
    else:
        lines.append("无冷宫照片 —— 每张都至少被用了一次。")
    return "\n".join(lines)


def plot_usage_histogram(use_count: dict[int, int]):
    """Return a matplotlib Figure of usage count distribution."""
    import matplotlib.pyplot as plt

    counts = sorted(use_count.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(counts)), counts)
    ax.set_xlabel("底图（按使用次数降序）")
    ax.set_ylabel("使用次数")
    ax.set_title(f"底图使用分布（{len(counts)} 张被用）")
    fig.tight_layout()
    return fig


def build_cold_wall(
    pool: TilePool,
    use_count: dict[int, int],
    max_shown: int = 64,
) -> Image.Image:
    """Grid image of tiles that were never used."""
    used = {i for i, v in use_count.items() if v > 0}
    cold = [i for i in range(len(pool)) if i not in used]
    if not cold:
        # Return a 1×1 blank so caller doesn't blow up
        return Image.new("RGB", (1, 1), (0, 0, 0))

    cold = cold[:max_shown]
    n = len(cold)
    side = int(math.ceil(math.sqrt(n)))
    tile_px = pool.thumbnails.shape[1]
    canvas = np.full((side * tile_px, side * tile_px, 3), 255, dtype=np.uint8)
    for k, idx in enumerate(cold):
        r, c = k // side, k % side
        canvas[r * tile_px : (r + 1) * tile_px,
               c * tile_px : (c + 1) * tile_px] = pool.thumbnails[idx]
    return Image.fromarray(canvas)
```

- [ ] **Step 2: 冒烟 import + 空 use_count 防御**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run python - <<'PY'
import numpy as np
from mosaic.tiles import TilePool
from mosaic.report import generate_text_report, plot_usage_histogram, build_cold_wall

pool = TilePool(
    paths=[f"t{i}.png" for i in range(3)],
    lab_means=np.zeros((3, 3), dtype=np.float32),
    thumbnails=np.zeros((3, 8, 8, 3), dtype=np.uint8),
)
uc = {0: 5, 1: 2}  # tile 2 is in cold wall
print(generate_text_report(uc, pool, total_cells=7))
print("---")
fig = plot_usage_histogram(uc)
print("fig:", fig)
wall = build_cold_wall(pool, uc)
print("wall:", wall.size)
PY
```
Expected: 文字报告输出、fig 对象、wall 尺寸约 (8, 8)（1 个冷宫 tile 1×1 grid）。

- [ ] **Step 3: Commit**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo
git add doubao/src/mosaic/report.py
git commit -m "$(cat <<'EOF'
feat(mosaic): add text report + usage histogram + cold-photo wall

generate_text_report builds the self-deprecating summary (used/cold TOP5).
plot_usage_histogram returns a matplotlib Figure. build_cold_wall lays
unused tiles on a square grid canvas. All three consume TilePool + use_count.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `deepzoom.py` — DZI 金字塔 + OpenSeadragon HTML

**Files:**
- Create: `doubao/src/mosaic/deepzoom.py`

按 spec § 4.6 不写单测；手写 DZI 以避开 `deepzoom` pypi 无维护风险。

- [ ] **Step 1: 创建 `doubao/src/mosaic/deepzoom.py`**

```python
"""Manual DZI (Deep Zoom Image) export + OpenSeadragon viewer HTML."""
from __future__ import annotations

import math
import shutil
from pathlib import Path

from PIL import Image

TILE_SIZE = 256
OVERLAP = 1


def _save_level_tiles(img: Image.Image, level_dir: Path) -> None:
    """Split img into TILE_SIZE × TILE_SIZE tiles with OVERLAP pixel overlap."""
    level_dir.mkdir(parents=True, exist_ok=True)
    w, h = img.size
    cols = math.ceil(w / TILE_SIZE)
    rows = math.ceil(h / TILE_SIZE)
    for r in range(rows):
        for c in range(cols):
            x = c * TILE_SIZE
            y = r * TILE_SIZE
            left = max(0, x - OVERLAP)
            top = max(0, y - OVERLAP)
            right = min(w, x + TILE_SIZE + OVERLAP)
            bottom = min(h, y + TILE_SIZE + OVERLAP)
            tile = img.crop((left, top, right, bottom))
            tile.save(level_dir / f"{c}_{r}.jpg", "JPEG", quality=85)


def _write_dzi(dzi_path: Path, width: int, height: int) -> None:
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="jpg"
       Overlap="{OVERLAP}"
       TileSize="{TILE_SIZE}">
  <Size Width="{width}" Height="{height}"/>
</Image>
"""
    dzi_path.write_text(xml, encoding="utf-8")


def _write_html(html_path: Path, dzi_rel: str) -> None:
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Photomosaic — zoom in</title>
  <style>
    html, body, #viewer {{ margin: 0; padding: 0; width: 100vw; height: 100vh; background: #000; }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/openseadragon.min.js"></script>
</head>
<body>
  <div id="viewer"></div>
  <script>
    OpenSeadragon({{
      id: "viewer",
      prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/images/",
      tileSources: "{dzi_rel}",
      showNavigator: true,
      maxZoomPixelRatio: 4,
    }});
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def export_deepzoom(mosaic: Image.Image, output_dir: Path) -> Path:
    """Generate DZI pyramid + tiles + index.html in output_dir/deepzoom/.

    Returns absolute path to index.html.
    """
    dz_dir = Path(output_dir) / "deepzoom"
    if dz_dir.exists():
        shutil.rmtree(dz_dir)
    dz_dir.mkdir(parents=True)

    base_name = "mosaic"
    tiles_dir = dz_dir / f"{base_name}_files"
    tiles_dir.mkdir()

    orig_w, orig_h = mosaic.size
    max_dim = max(orig_w, orig_h)
    n_levels = math.ceil(math.log2(max_dim)) + 1

    # Level n_levels-1 is full resolution; level 0 is 1×1 (approximately)
    current = mosaic
    for level in reversed(range(n_levels)):
        level_dir = tiles_dir / str(level)
        _save_level_tiles(current, level_dir)
        # Downscale for next (lower) level
        w, h = current.size
        new_w = max(1, w // 2)
        new_h = max(1, h // 2)
        if new_w == w and new_h == h:
            break
        current = current.resize((new_w, new_h), Image.LANCZOS)

    _write_dzi(dz_dir / f"{base_name}.dzi", orig_w, orig_h)
    html_path = dz_dir / "index.html"
    _write_html(html_path, f"{base_name}.dzi")
    return html_path.resolve()
```

- [ ] **Step 2: 冒烟 export**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run python - <<'PY'
import tempfile, numpy as np
from pathlib import Path
from PIL import Image
from mosaic.deepzoom import export_deepzoom

with tempfile.TemporaryDirectory() as td:
    img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    html = export_deepzoom(img, Path(td))
    print("html:", html)
    print("exists:", html.exists())
    tiles = list((Path(td) / "deepzoom" / "mosaic_files").iterdir())
    print("levels:", len(tiles))
    dzi = (Path(td) / "deepzoom" / "mosaic.dzi").read_text()
    assert "Width=\"512\"" in dzi
    print("dzi ok")
PY
```
Expected: html 路径存在、levels ≥ 9、dzi ok

- [ ] **Step 3: Commit**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo
git add doubao/src/mosaic/deepzoom.py
git commit -m "$(cat <<'EOF'
feat(mosaic): add manual DZI export + OpenSeadragon viewer HTML

~80 lines, zero external deps (CDN OpenSeadragon). Halves resolution
per level down to 1px, writes 256² JPEG tiles with 1px overlap plus
DZI XML. index.html loads the pyramid for infinite-zoom viewing.
Chosen over unmaintained `deepzoom` pypi package.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `photomosaic.ipynb` — 8-cell 编排

**Files:**
- Create: `doubao/photomosaic.ipynb`

写一个 JSON 格式的 notebook。`nbformat 4.5` 最小结构。

- [ ] **Step 1: 创建 `doubao/photomosaic.ipynb`**

```python
# Use this script to generate the notebook file:
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run python - <<'PY'
import json

cells_src = [
    # Cell 1: environment self-check
    [
        "# Cell 1 — env check\n",
        "import sys\n",
        "print('python:', sys.version.split()[0])\n",
        "import mosaic\n",
        "print('mosaic package loaded')\n",
    ],
    # Cell 2: config
    [
        "# Cell 2 — config (fill paths below)\n",
        "from pathlib import Path\n",
        "from mosaic.config import MosaicConfig\n",
        "\n",
        "cfg = MosaicConfig(\n",
        "    tile_source_dir=Path('/path/to/your/photos'),   # ← 填这里\n",
        "    target_image=Path('/path/to/target.jpg'),        # ← 填这里\n",
        "    grid=(120, 68),\n",
        "    tile_px=16,\n",
        "    candidate_k=50,\n",
        "    lambda_reuse=0.3,\n",
        "    mu_neighbor=0.2,\n",
        "    tau_tone=0.5,\n",
        "    verbose=True,\n",
        ")\n",
        "cfg.validate()\n",
        "print('config ok')\n",
    ],
    # Cell 3: tile pool
    [
        "# Cell 3 — build / load tile pool (cached)\n",
        "from mosaic.tiles import load_or_build\n",
        "\n",
        "tile_pool = load_or_build(cfg.tile_source_dir, cfg.tile_px, cfg.cache_dir)\n",
        "print(f'{len(tile_pool):,} tiles ready')\n",
    ],
    # Cell 4: split target
    [
        "# Cell 4 — split target into LAB cell grid\n",
        "from mosaic.match import split_target\n",
        "\n",
        "target_cells = split_target(cfg.target_image, cfg.grid)\n",
        "print('target cells shape:', target_cells.shape)\n",
    ],
    # Cell 5: match
    [
        "# Cell 5 — match each cell to a tile (verbose prints the thinking)\n",
        "from mosaic.match import match_all_tiles\n",
        "\n",
        "assignment, use_count = match_all_tiles(target_cells, tile_pool, cfg)\n",
        "print(f'done. used {sum(1 for v in use_count.values() if v > 0)} / {len(tile_pool)} tiles.')\n",
    ],
    # Cell 6: render
    [
        "# Cell 6 — render + save PNG\n",
        "from mosaic.render import render_mosaic\n",
        "\n",
        "mosaic_img = render_mosaic(assignment, tile_pool, target_cells, cfg)\n",
        "out_png = cfg.output_dir / 'mosaic.png'\n",
        "mosaic_img.save(out_png)\n",
        "print('saved:', out_png)\n",
        "mosaic_img\n",
    ],
    # Cell 7: report
    [
        "# Cell 7 — self-deprecating report\n",
        "from mosaic.report import generate_text_report, plot_usage_histogram, build_cold_wall\n",
        "\n",
        "print(generate_text_report(use_count, tile_pool, total_cells=cfg.grid[0]*cfg.grid[1]))\n",
        "plot_usage_histogram(use_count)\n",
        "build_cold_wall(tile_pool, use_count)\n",
    ],
    # Cell 8: deepzoom
    [
        "# Cell 8 — DeepZoom HTML export\n",
        "from mosaic.deepzoom import export_deepzoom\n",
        "\n",
        "html = export_deepzoom(mosaic_img, cfg.output_dir)\n",
        "print(f'open: file://{html}')\n",
    ],
]

cells = [
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': src,
    }
    for src in cells_src
]

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

with open('photomosaic.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print('notebook written')
PY
```
Expected: `notebook written`

- [ ] **Step 2: 验证 notebook 合法**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run python -c "
import nbformat
nb = nbformat.read('photomosaic.ipynb', as_version=4)
nbformat.validate(nb)
print(f'valid, {len(nb.cells)} cells')
"
```
Expected: `valid, 8 cells`

- [ ] **Step 3: 试跑 cell 1 离线**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run python -c "
import sys
print('python:', sys.version.split()[0])
import mosaic
print('mosaic package loaded')
"
```
Expected: 打印 python 版本 + `mosaic package loaded`

- [ ] **Step 4: Commit**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo
git add doubao/photomosaic.ipynb
git commit -m "$(cat <<'EOF'
feat(mosaic): add 8-cell photomosaic notebook

Cell 1 env check, Cell 2 config (placeholder paths), Cell 3 tile pool,
Cell 4 target split, Cell 5 match w/ verbose, Cell 6 render+save,
Cell 7 report, Cell 8 DeepZoom export. Thin wiring — all logic lives
in src/mosaic/*.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: 端到端 smoke test（合成数据）

**Files:**
- Create: `doubao/tests/test_e2e.py`

覆盖 tiles.py / report.py / deepzoom.py 这些没单测的模块，以及 full pipeline 的集成。

- [ ] **Step 1: 创建 `doubao/tests/test_e2e.py`**

```python
"""End-to-end smoke: synthetic 100-tile pool → 16×9 target → verify all outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def test_full_pipeline_on_synthetic_data(tmp_path: Path) -> None:
    from mosaic.config import MosaicConfig
    from mosaic.tiles import load_or_build
    from mosaic.match import split_target, match_all_tiles
    from mosaic.render import render_mosaic
    from mosaic.report import generate_text_report, build_cold_wall
    from mosaic.deepzoom import export_deepzoom

    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(42)
    # 100 random-color tiles
    for i in range(100):
        color = rng.integers(0, 255, 3, dtype=np.uint8)
        Image.fromarray(np.full((32, 32, 3), color, dtype=np.uint8)).save(src / f"t{i:03d}.png")

    target_path = tmp_path / "target.png"
    # Gradient target
    grad = np.zeros((128, 128, 3), dtype=np.uint8)
    grad[..., 0] = np.linspace(0, 255, 128, dtype=np.uint8)[None, :]
    grad[..., 2] = np.linspace(0, 255, 128, dtype=np.uint8)[:, None]
    Image.fromarray(grad).save(target_path)

    cfg = MosaicConfig(
        tile_source_dir=src,
        target_image=target_path,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        grid=(16, 9),
        tile_px=16,
        candidate_k=20,
        lambda_reuse=0.3,
        mu_neighbor=0.2,
        tau_tone=0.4,
        verbose=False,
    )
    cfg.validate()

    pool = load_or_build(cfg.tile_source_dir, cfg.tile_px, cfg.cache_dir)
    assert len(pool) == 100

    target_cells = split_target(cfg.target_image, cfg.grid)
    assert target_cells.shape == (9, 16, 3)

    assignment, use_count = match_all_tiles(target_cells, pool, cfg)
    assert assignment.shape == (9, 16)
    assert (assignment >= 0).all()
    assert (assignment < 100).all()
    assert sum(use_count.values()) == 9 * 16

    mosaic = render_mosaic(assignment, pool, target_cells, cfg)
    assert mosaic.size == (16 * 16, 9 * 16)

    out_png = cfg.output_dir / "mosaic.png"
    mosaic.save(out_png)
    assert out_png.exists()

    report_text = generate_text_report(use_count, pool, total_cells=9 * 16)
    assert "本次使用了" in report_text
    assert "TOP 5" in report_text

    wall = build_cold_wall(pool, use_count)
    assert wall.size[0] >= 1

    html = export_deepzoom(mosaic, cfg.output_dir)
    assert html.exists()
    assert html.name == "index.html"
    # DZI file + tiles dir should exist
    assert (cfg.output_dir / "deepzoom" / "mosaic.dzi").exists()
    assert (cfg.output_dir / "deepzoom" / "mosaic_files").is_dir()


def test_reuse_cache_on_second_build(tmp_path: Path) -> None:
    """Second load_or_build on the same dir should hit the cache without rebuilding."""
    from mosaic.tiles import load_or_build

    src = tmp_path / "src"
    src.mkdir()
    for i in range(5):
        Image.fromarray(np.full((32, 32, 3), i * 50, dtype=np.uint8)).save(src / f"t{i}.png")

    cache = tmp_path / "cache"
    pool1 = load_or_build(src, tile_px=8, cache_dir=cache)
    pool2 = load_or_build(src, tile_px=8, cache_dir=cache)
    np.testing.assert_array_equal(pool1.lab_means, pool2.lab_means)
    np.testing.assert_array_equal(pool1.thumbnails, pool2.thumbnails)
    assert (cache / "tiles_8.pkl").exists()
```

- [ ] **Step 2: 运行 e2e 测试**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run pytest tests/test_e2e.py -v
```
Expected: `2 passed`

- [ ] **Step 3: 跑所有测试确认整体绿**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run pytest tests/ -v
```
Expected: `8 passed`

- [ ] **Step 4: Commit**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo
git add doubao/tests/test_e2e.py
git commit -m "$(cat <<'EOF'
test(mosaic): add end-to-end smoke with synthetic data

100 random-color tiles + gradient 128² target → full pipeline
(tiles → split → match → render → report → deepzoom). Also covers
tile pickle cache reuse on second build. Fills the coverage gap
left by tiles.py / report.py / deepzoom.py not having unit tests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: README 正式版 + CHANGELOG 首条 + 最终提交

**Files:**
- Modify: `doubao/README.md`
- Create: `doubao/CHANGELOG.md`
- Create: `doubao/CHANGELOG.archive.md`

按用户定的 agent-oriented changelog 规范，正式记录第一条。

- [ ] **Step 1: 重写 `doubao/README.md`**

```markdown
# Photomosaic Toy

一个本地跑的玩具：把你硬盘里的照片重组成另一张照片。

> 定位：**玩具**，不是 SaaS / API / 产品。速度不重要，稳定性不重要，可解释性重要，彩蛋比功能值钱。

## 跑起来

```bash
cd doubao
uv sync --all-groups
uv run jupyter lab photomosaic.ipynb
```

打开 notebook 后编辑 **Cell 2** 把 `tile_source_dir` 指向你的照片目录、`target_image` 指向目标图，然后从上到下依次运行。

## 输出

- `output/mosaic.png` —— 最终马赛克
- `output/deepzoom/index.html` —— 用浏览器打开可无限缩放，放到底能看清每张小图
- 控制台打印自嘲式报告（"用了 X 张里的 Y 张，冷宫 TOP 5..."）
- matplotlib 显示使用次数柱状图和冷宫照片墙

## 旋钮

全部在 Cell 2 的 `MosaicConfig` 里：

| 参数 | 含义 | 默认 |
|------|-----|-----|
| `grid` | 网格 (cols, rows) | (120, 68) |
| `tile_px` | 每格渲染像素 | 16 |
| `candidate_k` | 每格先筛 top-K 颜色候选 | 50 |
| `lambda_reuse` | 重复惩罚（λ）：大→使用更分散 | 0.3 |
| `mu_neighbor` | 邻居惩罚（μ）：大→相邻不同 | 0.2 |
| `tau_tone` | 色调迁移（τ）：0 原色，1 完全贴合目标 | 0.5 |
| `verbose` | 匹配每格时打印决策 | True |

## 架构

```
src/mosaic/
  config.py    MosaicConfig dataclass（全部参数）
  tiles.py     扫描 + LAB 均色 + pickle 缓存
  match.py     split_target + match_all_tiles（含 λ/μ 惩罚）
  render.py    Reinhard 色调迁移 + 贴图
  report.py    自嘲文字 + 柱状图 + 冷宫墙
  deepzoom.py  DZI 金字塔 + OpenSeadragon HTML
photomosaic.ipynb   8 cell 编排
```

详见 `docs/superpowers/specs/2026-04-17-photomosaic-toy-design.md`。

## 测试

```bash
uv run pytest
```

## 不在 MVP 的（已为增量预留接口）

CLIP 语义匹配 · cursed mode · Gradio 滑条 UI · rembg 主体 mask · 底图标签叙事。
```

- [ ] **Step 2: 创建 `doubao/CHANGELOG.md`**

注意：changelog 条目要在实际验证完成（8/8 tests pass, notebook cell 1 importable）之后写，把实际 result / validation 填进去。

```markdown
# CHANGELOG

> 按 agent-oriented 规范维护：刻意啰嗦、保留 try-failed 轨迹、50 条或 6 月触发压缩。
> 压缩归档见 `CHANGELOG.archive.md`。

## 活跃条目

- date: 2026-04-17
  type: feat
  target: doubao/
  change: 初始化照片马赛克 MVP 骨架 —— 8-cell notebook + src/mosaic/ 六模块（config/tiles/match/render/report/deepzoom）+ tests/（test_match.py, test_render.py, test_e2e.py）
  rationale: 用户明确玩具定位；8 cell 是 MVP 底线。模块化是为了将来加 CLIP / cursed mode / Gradio 的增量不需要重构 —— 不是为了产品化。关键决策：(1) 用 numpy argpartition 替 faiss 因为 N≤5k 无性能差距 + arm64 wheel 有风险；(2) 手写 DZI 替 `deepzoom` pypi 因为后者 2018 后无维护。两处均在 plan 中显式记录。
  action: 新建 pyproject.toml (hatchling src-layout) + .gitignore + 包骨架 + 实现 config(MosaicConfig dataclass w/ validate) + tiles(递归扫描 + LAB 均值 + mtime-keyed pickle 缓存) + match(split_target + match_all_tiles w/ λ·log(1+uses) 重复惩罚 + μ·1/(1+neighbor_dist) 邻居惩罚) + render(Reinhard LAB 均值迁移 + 贴图) + report(自嘲文字 + 使用分布柱状图 + 冷宫墙) + deepzoom(手写 DZI 金字塔 + OpenSeadragon CDN HTML) + 8-cell photomosaic.ipynb
  result: `uv sync --all-groups` 通过；`uv run pytest` 8 pass（test_match 4 + test_render 2 + test_e2e 2）；notebook cell 1 可导入 mosaic；端到端 smoke 在 100 张合成底图 × 16×9 网格下产出 mosaic.png + deepzoom/index.html
  validation: pytest 全绿 + test_e2e.py 覆盖未单测的 tiles/report/deepzoom 三模块 + DZI XML 含正确 Width/Height
  status: experimental
```

- [ ] **Step 3: 创建空的 `doubao/CHANGELOG.archive.md`**

```markdown
# CHANGELOG Archive

> 压缩后的历史条目迁移至此。当前为空（项目新建未触发压缩阈值：50 条或 6 个月）。
```

- [ ] **Step 4: 最后一次全量验证**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run pytest tests/ -v
```
Expected: `8 passed`

- [ ] **Step 5: 验证 notebook 自检 cell 能跑**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo/doubao
uv run python -c "
import sys; print('python:', sys.version.split()[0])
import mosaic
from mosaic.config import MosaicConfig
from mosaic.tiles import load_or_build
from mosaic.match import split_target, match_all_tiles
from mosaic.render import render_mosaic
from mosaic.report import generate_text_report, plot_usage_histogram, build_cold_wall
from mosaic.deepzoom import export_deepzoom
print('all modules importable')
"
```
Expected: 打印 python 版本 + `all modules importable`

- [ ] **Step 6: Commit**

```bash
cd /Users/joejiang/chrome_extension_perf_monitor/Tile.Photo
git add doubao/README.md doubao/CHANGELOG.md doubao/CHANGELOG.archive.md
git commit -m "$(cat <<'EOF'
docs(doubao): finalize README + day-1 CHANGELOG entry

README: how-to-run, knob reference, architecture overview, MVP-out items.
CHANGELOG: agent-oriented entry per user convention — records numpy-over-faiss
and manual-over-pypi-deepzoom decisions with rationale; status experimental
until first real photo library run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Success Criteria (from spec § 14)

执行完 Task 12 后验收：

1. ✅ `uv sync --all-groups` 成功，无手动配置
2. ✅ `uv run pytest tests/` 8 pass 全绿
3. ⏳ 用户填路径后，`photomosaic.ipynb` 从 cell 1 跑到 cell 8 不报错 —— 需要用户真跑
4. ⏳ Cell 5 verbose 输出能看到"思考过程"（每格一行）—— 需要用户真跑
5. ⏳ `output/mosaic.png` 肉眼像目标图的马赛克 —— 需要用户真跑
6. ⏳ `output/deepzoom/index.html` 能无限缩放 —— 需要用户真跑
7. ✅ Cell 7 文字报告含"用了 X 张里的 Y 张，冷宫 TOP5" —— test_e2e.py 覆盖
8. ✅ CHANGELOG.md 第一条已写好

Tasks 1-12 完成后，1/2/7/8 已通过自动化验证；3/4/5/6 需要用户在真实照片库上跑一次才能确认。
