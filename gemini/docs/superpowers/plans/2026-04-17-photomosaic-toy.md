# Photomosaic Toy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local-only photomosaic toy (Jupyter notebook + thin Python modules) that re-composes a target image from the user's personal photo library, with visible algorithm narration, self-mocking usage report, tone-transfer/diversity/neighbor sliders, and a DeepZoom HTML output friends can infinitely zoom into.

**Architecture:** 8-cell notebook is the UI layer. All logic lives in `src/` modules so it is testable: `config` (dataclass), `tile_pool` (scan + LAB mean + optional CLIP cache), `matcher` (FAISS color kNN + repeat/neighbor/CLIP rerank + live-thinking callback), `renderer` (Reinhard LAB tone transfer + paste + usage tracking), `reporter` (self-mocking text + usage chart + cold-photo wall), `deepzoom` (pyvips pyramid + OpenSeadragon HTML). Cache tile metadata in `./.cache/` to pickle; next run is instant.

**Tech Stack:** Python 3.12, Pillow, NumPy, scikit-image, FAISS-CPU (color kNN), pyvips (DeepZoom), jupytext (author notebook as `.py` with `# %%` cell markers then convert), open_clip_torch (optional, CPU), matplotlib (report charts), tqdm, pytest.

**Non-goals (per user's "toy" positioning):**
- No Gradio/Streamlit UI in v1 — the notebook Cell 2 config is the UI.
- No speed optimization beyond FAISS for color kNN; 10-minute runs are fine.
- No SaaS, no API, no packaging for distribution.

**Changelog discipline:** Per user's memory rules, this project maintains `CHANGELOG.md` with YAML entries (date/type/target/change/rationale/action/result/validation/status). Every non-trivial change commits with a matching changelog entry. See Task 1 for format.

---

## File Structure

```
gemini/
├── photomosaic.py              # jupytext-managed source (authored with # %% cells)
├── photomosaic.ipynb           # generated from photomosaic.py via jupytext --to ipynb
├── src/
│   ├── __init__.py
│   ├── config.py               # PhotomosaicConfig dataclass
│   ├── tile_pool.py            # scan + LAB mean + pickle cache + CLIP (optional) + tags
│   ├── matcher.py              # FAISS color kNN + repeat/neighbor/CLIP rerank + callback
│   ├── renderer.py             # Reinhard LAB tone transfer + paste + usage counter
│   ├── reporter.py             # self-mocking text + usage histogram + cold-photo wall
│   └── deepzoom.py             # pyvips pyramid + OpenSeadragon HTML template
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_tile_pool.py
│   ├── test_matcher.py
│   ├── test_renderer.py
│   ├── test_reporter.py
│   └── fixtures/               # tiny synthetic images for deterministic tests
│       └── (generated in tests)
├── samples/                    # example target + tiny tile pool for smoke test
│   └── (user adds or task creates tiny synthetic set)
├── .cache/                     # pickled tile pool metadata (gitignored)
├── .gitignore
├── requirements.txt
├── README.md
├── CHANGELOG.md
└── docs/superpowers/plans/2026-04-17-photomosaic-toy.md
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `gemini/.gitignore`
- Create: `gemini/requirements.txt`
- Create: `gemini/CHANGELOG.md`
- Create: `gemini/README.md`
- Create: `gemini/src/__init__.py` (empty)
- Create: `gemini/tests/__init__.py` (empty)
- Create: `gemini/samples/.gitkeep` (empty)

- [ ] **Step 1: Create `.gitignore`**

```
__pycache__/
*.py[cod]
.cache/
.venv/
.ipynb_checkpoints/
*.egg-info/
.pytest_cache/
.DS_Store
samples/**/*.jpg
samples/**/*.jpeg
samples/**/*.png
!samples/.gitkeep
output/
```

- [ ] **Step 2: Create `requirements.txt`**

```
pillow>=10.0
numpy>=1.26
scikit-image>=0.22
faiss-cpu>=1.7.4
pyvips>=2.2.1
jupytext>=1.16
tqdm>=4.66
matplotlib>=3.8
pytest>=8.0
jupyter>=1.0
# Optional (guarded by flag in config):
open_clip_torch>=2.24
torch>=2.2
```

- [ ] **Step 3: Create `CHANGELOG.md`**

```markdown
# CHANGELOG

> Convention: entries are YAML with fields date(ISO)/type/target/change/rationale/action/result/validation/status.
> `try-failed` entries are never deleted, only compressed. See `docs/superpowers/plans/2026-04-17-photomosaic-toy.md` for full convention.

## 活跃条目

- date: 2026-04-17
  type: feat
  target: repo
  change: Initial scaffolding (requirements, gitignore, module skeletons, plan doc)
  rationale: Kick off photomosaic toy per user spec (local ipynb + fun features + DeepZoom output)
  action: Create requirements.txt, .gitignore, README.md, src/tests skeletons
  result: Repo has no code yet, only structure
  validation: Tree inspection + pytest collects zero tests successfully
  status: stable
```

- [ ] **Step 4: Create `README.md`**

```markdown
# Photomosaic Toy

Re-compose a target image from your personal photo library. Local-only, deliberately slow, deliberately explainable.

## Quick start

```bash
pip install -r requirements.txt
# (macOS) brew install vips  # for pyvips
jupytext --to ipynb photomosaic.py
jupyter notebook photomosaic.ipynb
```

Edit Cell 2 (config) to point at your target image and tile-pool directory, then run all cells.

## What you get

- A rendered mosaic PNG
- A self-mocking text report (which photos got used, which are in the cold palace)
- A usage histogram
- An `output/deepzoom/index.html` you can open in a browser and infinitely zoom

## Configuration knobs (Cell 2)

- `grid`: target grid in tiles (e.g. 120×68 for 16:9)
- `tile_px`: rendered tile size in pixels
- `lambda_repeat`: penalty per prior use of a tile (0 = anything-goes, higher = force diversity)
- `mu_neighbor`: penalty if a neighbor cell used the same tile (prevents clumping)
- `tau_tone`: 0..1 Reinhard LAB tone transfer strength (0 = keep photo colors, 1 = perfect match, 0.4–0.6 is the sweet spot)
- `use_clip`: enable CLIP semantic reranking (slower, more "intentional" matches)
- `mode`: `normal` | `cursed` | `time_capsule`

See `docs/superpowers/plans/2026-04-17-photomosaic-toy.md` for the full spec.
```

- [ ] **Step 5: Create empty package files**

Create `gemini/src/__init__.py`, `gemini/tests/__init__.py`, `gemini/samples/.gitkeep` — each an empty file.

- [ ] **Step 6: Verify pytest collects zero tests**

Run: `cd gemini && python -m pytest --collect-only 2>&1 | tail -5`
Expected: `no tests ran` or `collected 0 items`.

- [ ] **Step 7: Commit**

```bash
cd gemini
git add .gitignore requirements.txt CHANGELOG.md README.md src/ tests/ samples/ docs/
git commit -m "chore: scaffold photomosaic toy project

Initial structure: requirements, gitignore, README, empty src/tests packages,
and the detailed implementation plan at docs/superpowers/plans/."
```

---

## Task 2: Config Dataclass

**Files:**
- Create: `gemini/src/config.py`
- Test: `gemini/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`gemini/tests/test_config.py`:
```python
import pytest
from src.config import PhotomosaicConfig


def test_config_defaults_are_sane():
    cfg = PhotomosaicConfig(target="target.jpg", tile_dir="tiles/")
    assert cfg.grid == (120, 68)
    assert cfg.tile_px == 16
    assert cfg.lambda_repeat == 0.3
    assert cfg.mu_neighbor == 0.5
    assert 0.0 <= cfg.tau_tone <= 1.0
    assert cfg.use_clip is False
    assert cfg.mode == "normal"


def test_config_rejects_tau_out_of_range():
    with pytest.raises(ValueError, match="tau_tone"):
        PhotomosaicConfig(target="t.jpg", tile_dir="tiles/", tau_tone=1.5)


def test_config_rejects_nonpositive_grid():
    with pytest.raises(ValueError, match="grid"):
        PhotomosaicConfig(target="t.jpg", tile_dir="tiles/", grid=(0, 10))


def test_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        PhotomosaicConfig(target="t.jpg", tile_dir="tiles/", mode="bogus")
```

- [ ] **Step 2: Run tests, confirm ImportError / FAIL**

Run: `cd gemini && python -m pytest tests/test_config.py -v`
Expected: ImportError for `src.config`.

- [ ] **Step 3: Implement `src/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Tuple

Mode = Literal["normal", "cursed", "time_capsule"]


@dataclass
class PhotomosaicConfig:
    target: str
    tile_dir: str
    grid: Tuple[int, int] = (120, 68)
    tile_px: int = 16
    lambda_repeat: float = 0.3
    mu_neighbor: float = 0.5
    tau_tone: float = 0.5
    use_clip: bool = False
    clip_weight: float = 0.15
    mode: Mode = "normal"
    cache_dir: str = ".cache"
    output_dir: str = "output"
    topk_color: int = 32
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not (0.0 <= self.tau_tone <= 1.0):
            raise ValueError(f"tau_tone must be in [0, 1], got {self.tau_tone}")
        if self.grid[0] <= 0 or self.grid[1] <= 0:
            raise ValueError(f"grid must have positive dims, got {self.grid}")
        if self.mode not in ("normal", "cursed", "time_capsule"):
            raise ValueError(f"mode must be one of normal/cursed/time_capsule, got {self.mode!r}")
        if self.tile_px <= 0:
            raise ValueError(f"tile_px must be positive, got {self.tile_px}")
        if self.topk_color <= 0:
            raise ValueError(f"topk_color must be positive, got {self.topk_color}")
```

- [ ] **Step 4: Run tests, confirm PASS**

Run: `cd gemini && python -m pytest tests/test_config.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit + changelog**

Append to `CHANGELOG.md` under `## 活跃条目`:
```yaml
- date: 2026-04-17
  type: feat
  target: src/config.py
  change: PhotomosaicConfig dataclass with validation (tau in [0,1], positive grid/tile_px, mode enum)
  rationale: Centralize the 3 sliders (lambda/mu/tau) + modes so Cell 2 of notebook is the single UI surface
  action: Dataclass + __post_init__ validation; 4 tests for defaults + each invariant
  result: 4/4 tests pass
  validation: pytest tests/test_config.py -v
  status: stable
```

```bash
git add src/config.py tests/test_config.py CHANGELOG.md
git commit -m "feat(config): PhotomosaicConfig dataclass with validation"
```

---

## Task 3: Tile Pool — Scan Filesystem

**Files:**
- Create: `gemini/src/tile_pool.py`
- Test: `gemini/tests/test_tile_pool.py`

- [ ] **Step 1: Write the failing test**

`gemini/tests/test_tile_pool.py`:
```python
from pathlib import Path
from PIL import Image
import pytest
from src.tile_pool import scan_tile_dir


def _make_img(path: Path, color: tuple[int, int, int], size=(64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_scan_finds_jpg_png_recursively(tmp_path: Path):
    _make_img(tmp_path / "a.jpg", (255, 0, 0))
    _make_img(tmp_path / "sub" / "b.png", (0, 255, 0))
    _make_img(tmp_path / "sub" / "c.jpeg", (0, 0, 255))
    (tmp_path / "readme.txt").write_text("not an image")
    (tmp_path / "broken.jpg").write_bytes(b"not a real jpeg")

    paths = scan_tile_dir(str(tmp_path))
    assert len(paths) == 3
    assert all(p.endswith((".jpg", ".jpeg", ".png")) for p in paths)


def test_scan_skips_too_small(tmp_path: Path):
    _make_img(tmp_path / "ok.jpg", (100, 100, 100), size=(64, 64))
    _make_img(tmp_path / "tiny.jpg", (100, 100, 100), size=(8, 8))
    paths = scan_tile_dir(str(tmp_path), min_side=32)
    assert len(paths) == 1
    assert paths[0].endswith("ok.jpg")
```

- [ ] **Step 2: Run, confirm FAIL**

Run: `cd gemini && python -m pytest tests/test_tile_pool.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `scan_tile_dir` in `src/tile_pool.py`**

```python
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
```

- [ ] **Step 4: Run, confirm PASS**

Run: `cd gemini && python -m pytest tests/test_tile_pool.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit + changelog**

Append changelog entry (same YAML shape as Task 2):
```yaml
- date: 2026-04-17
  type: feat
  target: src/tile_pool.py
  change: scan_tile_dir(path, min_side=32) recursively returns valid jpg/png paths, skipping broken/tiny
  rationale: Tile pool input is user's chaotic photo dump — must survive broken files and thumbnails silently
  action: Pillow Image.verify + size check; skip UnidentifiedImageError/OSError
  result: 2/2 tests pass (recursive glob + min-side filter)
  validation: pytest tests/test_tile_pool.py -v
  status: stable
```

```bash
git add src/tile_pool.py tests/test_tile_pool.py CHANGELOG.md
git commit -m "feat(tile_pool): scan_tile_dir with recursive glob + broken-file skip"
```

---

## Task 4: Tile Pool — LAB Mean + Pickle Cache

**Files:**
- Modify: `gemini/src/tile_pool.py` (append functions)
- Modify: `gemini/tests/test_tile_pool.py` (append tests)

- [ ] **Step 1: Append failing tests**

Append to `tests/test_tile_pool.py`:
```python
import numpy as np
from src.tile_pool import build_tile_index, load_or_build_index


def test_build_tile_index_computes_lab_mean(tmp_path: Path):
    _make_img(tmp_path / "red.jpg", (200, 30, 30))
    _make_img(tmp_path / "blue.jpg", (30, 30, 200))
    idx = build_tile_index(str(tmp_path))
    assert set(idx.paths) == {str(tmp_path / "red.jpg"), str(tmp_path / "blue.jpg")}
    assert idx.lab_mean.shape == (2, 3)
    # red's a* should be > blue's a*; blue's b* should be < red's b*
    red_i = idx.paths.index(str(tmp_path / "red.jpg"))
    blue_i = idx.paths.index(str(tmp_path / "blue.jpg"))
    assert idx.lab_mean[red_i, 1] > idx.lab_mean[blue_i, 1]
    assert idx.lab_mean[red_i, 2] > idx.lab_mean[blue_i, 2]


def test_load_or_build_uses_cache_on_second_call(tmp_path: Path):
    _make_img(tmp_path / "a.jpg", (100, 100, 100))
    cache = tmp_path / "_cache"
    idx1 = load_or_build_index(str(tmp_path), cache_dir=str(cache))
    # Delete source, second call must still work from cache
    (tmp_path / "a.jpg").unlink()
    idx2 = load_or_build_index(str(tmp_path), cache_dir=str(cache))
    assert idx1.paths == idx2.paths
    np.testing.assert_array_equal(idx1.lab_mean, idx2.lab_mean)
```

- [ ] **Step 2: Run, confirm FAIL**

Run: `cd gemini && python -m pytest tests/test_tile_pool.py -v`
Expected: ImportError for `build_tile_index`, `load_or_build_index`.

- [ ] **Step 3: Append implementation to `src/tile_pool.py`**

```python
import hashlib
import pickle
from dataclasses import dataclass
import numpy as np
from skimage.color import rgb2lab


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
```

- [ ] **Step 4: Run, confirm PASS**

Run: `cd gemini && python -m pytest tests/test_tile_pool.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit + changelog**

```yaml
- date: 2026-04-17
  type: feat
  target: src/tile_pool.py
  change: TileIndex dataclass + build_tile_index (LAB mean via skimage.rgb2lab on 64x64 thumbnail) + load_or_build_index with pickle cache keyed by sha1(abspath::min_side)
  rationale: LAB mean is perceptually better than RGB for color distance; pickle cache makes re-runs instant (user will tweak sliders many times)
  action: 64x64 LANCZOS thumbnail → rgb2lab → mean over pixels; cache file name is stable hash of tile_dir abspath
  result: 4/4 tests pass, cache survives source deletion
  validation: pytest tests/test_tile_pool.py -v
  status: stable
```

```bash
git add src/tile_pool.py tests/test_tile_pool.py CHANGELOG.md
git commit -m "feat(tile_pool): LAB mean index + pickle cache"
```

---

## Task 5: Tile Pool — Optional CLIP Embeddings

**Files:**
- Modify: `gemini/src/tile_pool.py` (add `add_clip_embeddings`)
- Modify: `gemini/tests/test_tile_pool.py` (add guarded test)

- [ ] **Step 1: Append test with optional-skip**

Append to `tests/test_tile_pool.py`:
```python
import importlib


def _has_clip() -> bool:
    try:
        importlib.import_module("open_clip")
        importlib.import_module("torch")
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_clip(), reason="open_clip/torch not installed")
def test_add_clip_embeddings_shapes(tmp_path: Path):
    from src.tile_pool import add_clip_embeddings
    _make_img(tmp_path / "a.jpg", (100, 100, 100))
    _make_img(tmp_path / "b.jpg", (200, 50, 50))
    idx = build_tile_index(str(tmp_path))
    idx2 = add_clip_embeddings(idx, model_name="ViT-B-32", pretrained="openai")
    assert idx2.clip_emb is not None
    assert idx2.clip_emb.shape[0] == 2
    assert idx2.clip_emb.shape[1] > 0
    # L2-normalized
    norms = np.linalg.norm(idx2.clip_emb, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)
```

- [ ] **Step 2: Run, confirm ImportError / skip**

Run: `cd gemini && python -m pytest tests/test_tile_pool.py::test_add_clip_embeddings_shapes -v`
Expected: either ImportError (function missing) or skip (no clip).

- [ ] **Step 3: Append CLIP pipeline**

```python
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
```

- [ ] **Step 4: Run — PASS if clip installed, skipped otherwise**

Run: `cd gemini && python -m pytest tests/test_tile_pool.py -v`
Expected: all green (with CLIP test either passing or skipped).

- [ ] **Step 5: Commit + changelog**

```yaml
- date: 2026-04-17
  type: feat
  target: src/tile_pool.py
  change: add_clip_embeddings(idx, model_name, pretrained) returns new TileIndex with L2-normalized CLIP image embeddings
  rationale: Enables "semantic" matching (玩点 A) — blue regions can prefer actual sea photos over blue walls
  action: open_clip.create_model_and_transforms → encode_image batched → L2 normalize → cpu().numpy()
  result: Test passes if open_clip installed, else skipped
  validation: pytest tests/test_tile_pool.py -v
  status: stable
```

```bash
git add src/tile_pool.py tests/test_tile_pool.py CHANGELOG.md
git commit -m "feat(tile_pool): optional CLIP image-embedding pipeline"
```

---

## Task 6: Tile Pool — Tags (Narrative Feature)

**Files:**
- Modify: `gemini/src/tile_pool.py` (add `load_tags`)
- Modify: `gemini/tests/test_tile_pool.py`

Tag format: a single JSON file `tags.json` at the tile root, mapping glob patterns to tag strings:
```json
{
  "2019_Japan/**/*": "2019 Japan trip",
  "work/**/*": "ex-coworkers"
}
```

- [ ] **Step 1: Append failing test**

```python
def test_load_tags_matches_glob_patterns(tmp_path: Path):
    from src.tile_pool import load_tags
    _make_img(tmp_path / "2019_Japan" / "a.jpg", (10, 10, 10))
    _make_img(tmp_path / "2019_Japan" / "sub" / "b.jpg", (20, 20, 20))
    _make_img(tmp_path / "work" / "c.jpg", (30, 30, 30))
    _make_img(tmp_path / "random.jpg", (40, 40, 40))
    import json
    (tmp_path / "tags.json").write_text(json.dumps({
        "2019_Japan/**/*": "Japan trip",
        "work/**/*": "ex-coworkers",
    }))
    paths = [
        str(tmp_path / "2019_Japan" / "a.jpg"),
        str(tmp_path / "2019_Japan" / "sub" / "b.jpg"),
        str(tmp_path / "work" / "c.jpg"),
        str(tmp_path / "random.jpg"),
    ]
    tags = load_tags(str(tmp_path), paths)
    assert tags[paths[0]] == "Japan trip"
    assert tags[paths[1]] == "Japan trip"
    assert tags[paths[2]] == "ex-coworkers"
    assert tags[paths[3]] == "untagged"


def test_load_tags_missing_file_returns_all_untagged(tmp_path: Path):
    from src.tile_pool import load_tags
    _make_img(tmp_path / "a.jpg", (10, 10, 10))
    tags = load_tags(str(tmp_path), [str(tmp_path / "a.jpg")])
    assert tags == {str(tmp_path / "a.jpg"): "untagged"}
```

- [ ] **Step 2: Run, confirm FAIL**

Run: `cd gemini && python -m pytest tests/test_tile_pool.py -v`
Expected: 2 new failures/import errors.

- [ ] **Step 3: Append implementation**

```python
import fnmatch
import json as _json
from typing import Dict


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
            if fnmatch.fnmatch(rel, pat):
                match = name
                break
        out[p] = match
    return out
```

- [ ] **Step 4: Run, confirm PASS**

Run: `cd gemini && python -m pytest tests/test_tile_pool.py -v`
Expected: all passing.

- [ ] **Step 5: Commit + changelog**

```yaml
- date: 2026-04-17
  type: feat
  target: src/tile_pool.py
  change: load_tags(tile_dir, paths) reads tags.json (glob-pattern → tag name), defaults to 'untagged'
  rationale: Powers the "23% from 2019 Japan trip" narrative in final report — emotional payoff for user
  action: fnmatch on relative path against ordered JSON patterns; first match wins
  result: 2/2 new tests pass
  validation: pytest tests/test_tile_pool.py -v
  status: stable
```

```bash
git add src/tile_pool.py tests/test_tile_pool.py CHANGELOG.md
git commit -m "feat(tile_pool): tags.json glob pattern loader"
```

---

## Task 7: Target Image → Grid Patches

**Files:**
- Create: `gemini/src/target.py`
- Test: `gemini/tests/test_target.py`

- [ ] **Step 1: Write the failing test**

`gemini/tests/test_target.py`:
```python
from pathlib import Path
import numpy as np
from PIL import Image
from src.target import split_into_patches


def test_split_returns_expected_shape(tmp_path: Path):
    img_path = tmp_path / "t.jpg"
    Image.new("RGB", (240, 136), (128, 128, 128)).save(img_path)
    patches_lab, cell_rgb = split_into_patches(str(img_path), grid=(120, 68))
    # grid is (cols, rows)
    assert patches_lab.shape == (68, 120, 3)  # (rows, cols, lab)
    assert cell_rgb.shape == (68, 120, 2, 2, 3)  # each patch 2x2 RGB


def test_split_handles_non_divisible_sizes(tmp_path: Path):
    img_path = tmp_path / "t.jpg"
    # 241 × 137 doesn't divide evenly by 120 × 68 — function must resize
    Image.new("RGB", (241, 137), (200, 50, 50)).save(img_path)
    patches_lab, _ = split_into_patches(str(img_path), grid=(120, 68))
    assert patches_lab.shape == (68, 120, 3)
    # mean of every cell's L*/a*/b* should reflect a strong red
    assert patches_lab[..., 1].mean() > 30  # a* positive = red
```

- [ ] **Step 2: Run, confirm FAIL**

Run: `cd gemini && python -m pytest tests/test_target.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/target.py`**

```python
from __future__ import annotations
from typing import Tuple
import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def split_into_patches(target_path: str, grid: Tuple[int, int], patch_px: int = 2
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      patches_lab: (rows, cols, 3) LAB mean per grid cell
      cell_rgb:    (rows, cols, patch_px, patch_px, 3) uint8 source RGB per cell
                   (used by tone-transfer to know what the cell 'should look like')
    grid is (cols, rows) to match image-native ordering.
    """
    cols, rows = grid
    target_w = cols * patch_px
    target_h = rows * patch_px
    with Image.open(target_path) as im:
        im = im.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.uint8)  # (target_h, target_w, 3)
    # Reshape into cells
    cell_rgb = arr.reshape(rows, patch_px, cols, patch_px, 3).transpose(0, 2, 1, 3, 4)
    # LAB mean per cell
    rgb_f = arr.astype(np.float32) / 255.0
    lab = rgb2lab(rgb_f)  # (target_h, target_w, 3)
    patches_lab = lab.reshape(rows, patch_px, cols, patch_px, 3).mean(axis=(1, 3))
    return patches_lab.astype(np.float32), cell_rgb
```

- [ ] **Step 4: Run, confirm PASS**

Run: `cd gemini && python -m pytest tests/test_target.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit + changelog**

```yaml
- date: 2026-04-17
  type: feat
  target: src/target.py
  change: split_into_patches(path, grid, patch_px=2) returns (LAB mean per cell, source RGB per cell)
  rationale: Separates target parsing from matching; LAB mean drives match distance, source RGB drives tone transfer
  action: Resize to grid × patch_px, rgb2lab, reshape to (rows, patch_px, cols, patch_px, 3), mean over patch axes
  result: 2/2 tests pass, non-divisible sizes handled by LANCZOS resize
  validation: pytest tests/test_target.py -v
  status: stable
```

```bash
git add src/target.py tests/test_target.py CHANGELOG.md
git commit -m "feat(target): split target image into grid of LAB-mean patches"
```

---

## Task 8: Matcher — FAISS Color kNN Baseline

**Files:**
- Create: `gemini/src/matcher.py`
- Test: `gemini/tests/test_matcher.py`

- [ ] **Step 1: Write the failing test**

`gemini/tests/test_matcher.py`:
```python
import numpy as np
from src.matcher import color_topk


def test_color_topk_picks_nearest():
    tile_lab = np.array([
        [50.0, 0.0, 0.0],      # neutral gray
        [50.0, 80.0, 60.0],    # vivid red
        [50.0, -80.0, -60.0],  # vivid blue-green
    ], dtype=np.float32)
    # Query = one cell that's close to the red tile
    patches_lab = np.array([[[50.0, 78.0, 58.0]]], dtype=np.float32)  # (1, 1, 3)
    idx, dist = color_topk(patches_lab, tile_lab, k=2)
    assert idx.shape == (1, 1, 2)
    assert dist.shape == (1, 1, 2)
    assert idx[0, 0, 0] == 1  # red tile wins
    # Distance to index 1 must be smaller than to the runner-up
    assert dist[0, 0, 0] < dist[0, 0, 1]


def test_color_topk_k_capped_to_tile_count():
    tile_lab = np.random.randn(5, 3).astype(np.float32) * 30 + 50
    patches_lab = np.random.randn(2, 3, 3).astype(np.float32) * 30 + 50
    idx, _ = color_topk(patches_lab, tile_lab, k=10)
    assert idx.shape == (2, 3, 5)  # capped to N=5
```

- [ ] **Step 2: Run, confirm FAIL**

Run: `cd gemini && python -m pytest tests/test_matcher.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/matcher.py`**

```python
from __future__ import annotations
import numpy as np
import faiss


def color_topk(patches_lab: np.ndarray, tile_lab: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    patches_lab: (rows, cols, 3) — target cells in LAB
    tile_lab:    (N, 3)          — tile pool LAB means
    Returns (topk_idx, topk_dist) each (rows, cols, k_effective) where k_effective = min(k, N).
    """
    rows, cols, _ = patches_lab.shape
    n_tiles = tile_lab.shape[0]
    k_eff = min(k, n_tiles)
    index = faiss.IndexFlatL2(3)
    index.add(tile_lab.astype(np.float32))
    q = patches_lab.reshape(-1, 3).astype(np.float32)
    dist, idx = index.search(q, k_eff)
    return idx.reshape(rows, cols, k_eff), dist.reshape(rows, cols, k_eff)
```

- [ ] **Step 4: Run, confirm PASS**

Run: `cd gemini && python -m pytest tests/test_matcher.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit + changelog**

```yaml
- date: 2026-04-17
  type: feat
  target: src/matcher.py
  change: color_topk(patches_lab, tile_lab, k) returns (idx, dist) via faiss IndexFlatL2 in LAB space
  rationale: Exact nearest-neighbor in LAB is fast enough (tens of thousands of tiles) and perceptually correct; returns top-k so later reranking can consider alternatives
  action: faiss IndexFlatL2 on (N, 3) LAB means; batch search over flattened patches
  result: 2/2 tests pass (nearest wins + k capped to N)
  validation: pytest tests/test_matcher.py -v
  status: stable
```

```bash
git add src/matcher.py tests/test_matcher.py CHANGELOG.md
git commit -m "feat(matcher): FAISS LAB kNN baseline"
```

---

## Task 9: Matcher — Repeat + Neighbor Penalty Rerank

**Files:**
- Modify: `gemini/src/matcher.py` (add `assign_with_penalties`)
- Modify: `gemini/tests/test_matcher.py`

- [ ] **Step 1: Append failing tests**

```python
from src.matcher import assign_with_penalties


def test_repeat_penalty_spreads_usage():
    # 4 cells all want tile 0 (identical LAB), 3 tiles available
    tile_lab = np.array([[50, 0, 0], [50, 5, 5], [50, -5, -5]], dtype=np.float32)
    patches_lab = np.full((2, 2, 3), [50, 0, 0], dtype=np.float32)
    idx, dist = color_topk(patches_lab, tile_lab, k=3)
    # With no penalty, every cell picks tile 0
    no_pen = assign_with_penalties(idx, dist, lambda_repeat=0.0, mu_neighbor=0.0)
    assert (no_pen == 0).all()
    # With heavy repeat penalty, at least one other tile must appear
    with_pen = assign_with_penalties(idx, dist, lambda_repeat=100.0, mu_neighbor=0.0)
    assert len(set(with_pen.ravel().tolist())) > 1


def test_neighbor_penalty_breaks_adjacency():
    # 2 tiles: both roughly equal distance to all cells
    tile_lab = np.array([[50, 10, 0], [50, 0, 10]], dtype=np.float32)
    patches_lab = np.full((2, 4, 3), [50, 5, 5], dtype=np.float32)
    idx, dist = color_topk(patches_lab, tile_lab, k=2)
    with_pen = assign_with_penalties(idx, dist, lambda_repeat=0.0, mu_neighbor=1000.0)
    # No two horizontally adjacent cells should share the same tile
    for r in range(with_pen.shape[0]):
        for c in range(with_pen.shape[1] - 1):
            assert with_pen[r, c] != with_pen[r, c + 1]


def test_assign_emits_callback_per_cell():
    tile_lab = np.array([[50, 0, 0], [50, 30, 30]], dtype=np.float32)
    patches_lab = np.full((1, 2, 3), [50, 15, 15], dtype=np.float32)
    idx, dist = color_topk(patches_lab, tile_lab, k=2)
    events = []
    assign_with_penalties(idx, dist, lambda_repeat=0.1, mu_neighbor=0.0,
                          on_cell=lambda r, c, chosen, candidates: events.append((r, c, chosen)))
    assert len(events) == 2
    assert [e[:2] for e in events] == [(0, 0), (0, 1)]
```

- [ ] **Step 2: Run, confirm FAIL**

Run: `cd gemini && python -m pytest tests/test_matcher.py -v`
Expected: 3 new failures.

- [ ] **Step 3: Append implementation**

```python
from typing import Callable, Optional

CellCallback = Callable[[int, int, int, list[tuple[int, float]]], None]


def assign_with_penalties(topk_idx: np.ndarray, topk_dist: np.ndarray,
                          lambda_repeat: float, mu_neighbor: float,
                          on_cell: Optional[CellCallback] = None) -> np.ndarray:
    """
    Greedy left-to-right, top-to-bottom assignment minimizing:
      score(tile) = sqrt(dist) + lambda_repeat * log1p(usage[tile])
                    + mu_neighbor * 1_{tile in {left_neighbor, top_neighbor}}
    Returns assignment array of shape (rows, cols).
    """
    rows, cols, k = topk_idx.shape
    assignment = np.full((rows, cols), -1, dtype=np.int64)
    usage: dict[int, int] = {}
    for r in range(rows):
        for c in range(cols):
            best_tile = -1
            best_score = float("inf")
            candidate_log: list[tuple[int, float]] = []
            left = int(assignment[r, c - 1]) if c > 0 else -1
            up = int(assignment[r - 1, c]) if r > 0 else -1
            for cand_rank in range(k):
                t = int(topk_idx[r, c, cand_rank])
                d = float(topk_dist[r, c, cand_rank])
                score = np.sqrt(max(d, 0.0))
                score += lambda_repeat * np.log1p(usage.get(t, 0))
                if t == left or t == up:
                    score += mu_neighbor
                candidate_log.append((t, float(score)))
                if score < best_score:
                    best_score = score
                    best_tile = t
            assignment[r, c] = best_tile
            usage[best_tile] = usage.get(best_tile, 0) + 1
            if on_cell is not None:
                on_cell(r, c, best_tile, candidate_log)
    return assignment
```

- [ ] **Step 4: Run, confirm PASS**

Run: `cd gemini && python -m pytest tests/test_matcher.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit + changelog**

```yaml
- date: 2026-04-17
  type: feat
  target: src/matcher.py
  change: assign_with_penalties(topk_idx, topk_dist, lambda_repeat, mu_neighbor, on_cell) — greedy raster-scan with usage + neighbor penalties + per-cell callback
  rationale: lambda solves "one photo dominates" (玩点 B); mu solves "same photo clumps in a region"; on_cell gives the notebook its live-thinking printout (玩点 "能看见算法在思考")
  action: Greedy O(rows*cols*k), score = sqrt(L2 LAB dist) + lambda*log1p(usage) + mu*neighbor_clash
  result: 3/3 new tests pass; zero-penalty regression still passes
  validation: pytest tests/test_matcher.py -v
  status: stable
```

```bash
git add src/matcher.py tests/test_matcher.py CHANGELOG.md
git commit -m "feat(matcher): repeat+neighbor penalty rerank + live-cell callback"
```

---

## Task 10: Matcher — CLIP Semantic Rerank (Optional)

**Files:**
- Modify: `gemini/src/matcher.py` (add `assign_with_clip`)
- Modify: `gemini/tests/test_matcher.py`

The rerank adds `-clip_weight * cosine(tile_emb, target_cell_emb)` to the score. Target cell embeddings come from CLIPping an upscaled patch of the target.

- [ ] **Step 1: Append failing test (no CLIP dep — uses synthetic emb arrays)**

```python
def test_clip_rerank_prefers_semantically_similar_when_tie():
    # Two tiles equally close in color, but tile 1 semantically closer
    tile_lab = np.array([[50, 0, 0], [50, 0, 0]], dtype=np.float32)
    tile_clip = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    patches_lab = np.array([[[50, 0, 0]]], dtype=np.float32)   # (1,1,3)
    patch_clip = np.array([[[0.0, 1.0]]], dtype=np.float32)     # (1,1,2) — matches tile 1
    idx, dist = color_topk(patches_lab, tile_lab, k=2)
    from src.matcher import assign_with_clip
    out = assign_with_clip(idx, dist, patches_lab, tile_lab,
                           tile_clip=tile_clip, patch_clip=patch_clip,
                           lambda_repeat=0.0, mu_neighbor=0.0, clip_weight=1.0)
    assert out[0, 0] == 1


def test_clip_rerank_noop_when_weight_zero():
    tile_lab = np.array([[50, 0, 0], [50, 0, 0]], dtype=np.float32)
    tile_clip = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    patches_lab = np.array([[[50, 0, 0]]], dtype=np.float32)
    patch_clip = np.array([[[0.0, 1.0]]], dtype=np.float32)
    idx, dist = color_topk(patches_lab, tile_lab, k=2)
    from src.matcher import assign_with_clip
    out_zero = assign_with_clip(idx, dist, patches_lab, tile_lab,
                                tile_clip=tile_clip, patch_clip=patch_clip,
                                lambda_repeat=0.0, mu_neighbor=0.0, clip_weight=0.0)
    out_plain = assign_with_penalties(idx, dist, lambda_repeat=0.0, mu_neighbor=0.0)
    np.testing.assert_array_equal(out_zero, out_plain)
```

- [ ] **Step 2: Run, confirm FAIL**

Run: `cd gemini && python -m pytest tests/test_matcher.py -v`
Expected: 2 new failures.

- [ ] **Step 3: Append implementation**

```python
def assign_with_clip(topk_idx: np.ndarray, topk_dist: np.ndarray,
                     patches_lab: np.ndarray, tile_lab: np.ndarray,
                     tile_clip: np.ndarray, patch_clip: np.ndarray,
                     lambda_repeat: float, mu_neighbor: float, clip_weight: float,
                     on_cell: Optional[CellCallback] = None) -> np.ndarray:
    """
    Same greedy scan as assign_with_penalties, plus a cosine-similarity bonus:
      score -= clip_weight * cosine(tile_clip[t], patch_clip[r,c])
    Assumes both tile_clip and patch_clip are L2-normalized (cosine = dot).
    """
    rows, cols, k = topk_idx.shape
    assignment = np.full((rows, cols), -1, dtype=np.int64)
    usage: dict[int, int] = {}
    for r in range(rows):
        for c in range(cols):
            best_tile = -1
            best_score = float("inf")
            candidate_log: list[tuple[int, float]] = []
            left = int(assignment[r, c - 1]) if c > 0 else -1
            up = int(assignment[r - 1, c]) if r > 0 else -1
            p_emb = patch_clip[r, c]
            for cand_rank in range(k):
                t = int(topk_idx[r, c, cand_rank])
                d = float(topk_dist[r, c, cand_rank])
                score = np.sqrt(max(d, 0.0))
                score += lambda_repeat * np.log1p(usage.get(t, 0))
                if t == left or t == up:
                    score += mu_neighbor
                score -= clip_weight * float(np.dot(tile_clip[t], p_emb))
                candidate_log.append((t, float(score)))
                if score < best_score:
                    best_score = score
                    best_tile = t
            assignment[r, c] = best_tile
            usage[best_tile] = usage.get(best_tile, 0) + 1
            if on_cell is not None:
                on_cell(r, c, best_tile, candidate_log)
    return assignment
```

- [ ] **Step 4: Run, confirm PASS**

Run: `cd gemini && python -m pytest tests/test_matcher.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit + changelog**

```yaml
- date: 2026-04-17
  type: feat
  target: src/matcher.py
  change: assign_with_clip(..., tile_clip, patch_clip, clip_weight) adds cosine similarity bonus to the penalty rerank
  rationale: 玩点 A — "blue regions pick real sea photos over blue walls"; weight=0 regression guarantees it never hurts plain mode
  action: Extends greedy scan with -clip_weight * dot(tile_emb, patch_emb); assumes L2-normalized inputs
  result: 2/2 new tests pass (tie-break + zero-weight regression)
  validation: pytest tests/test_matcher.py -v
  status: stable
```

```bash
git add src/matcher.py tests/test_matcher.py CHANGELOG.md
git commit -m "feat(matcher): CLIP semantic cosine rerank"
```

---

## Task 11: Renderer — Reinhard LAB Tone Transfer

**Files:**
- Create: `gemini/src/renderer.py`
- Test: `gemini/tests/test_renderer.py`

- [ ] **Step 1: Write the failing test**

`gemini/tests/test_renderer.py`:
```python
import numpy as np
from PIL import Image
from pathlib import Path
from src.renderer import reinhard_tone_transfer


def _solid_rgb(color: tuple[int, int, int], size=(16, 16)) -> np.ndarray:
    return np.full((size[1], size[0], 3), color, dtype=np.uint8)


def test_tau_zero_returns_source_unchanged():
    src = _solid_rgb((150, 40, 40))
    target = _solid_rgb((40, 40, 150))
    out = reinhard_tone_transfer(src, target, tau=0.0)
    np.testing.assert_array_equal(out, src)


def test_tau_one_matches_target_mean_closely():
    src = _solid_rgb((150, 40, 40))
    target = _solid_rgb((40, 40, 150))
    out = reinhard_tone_transfer(src, target, tau=1.0)
    # After full transfer, mean color should be very close to target's
    src_mean = src.reshape(-1, 3).mean(axis=0)
    out_mean = out.reshape(-1, 3).mean(axis=0)
    target_mean = target.reshape(-1, 3).mean(axis=0)
    # Closer to target than to source
    assert np.linalg.norm(out_mean - target_mean) < np.linalg.norm(out_mean - src_mean)


def test_tau_interpolates_linearly_in_lab():
    src = _solid_rgb((150, 40, 40))
    target = _solid_rgb((40, 40, 150))
    half = reinhard_tone_transfer(src, target, tau=0.5)
    full = reinhard_tone_transfer(src, target, tau=1.0)
    # Half-strength output should lie between src and full-transfer
    src_mean = src.reshape(-1, 3).mean(axis=0)
    half_mean = half.reshape(-1, 3).mean(axis=0)
    full_mean = full.reshape(-1, 3).mean(axis=0)
    assert np.linalg.norm(half_mean - src_mean) < np.linalg.norm(full_mean - src_mean)
    assert np.linalg.norm(half_mean - full_mean) < np.linalg.norm(src_mean - full_mean)
```

- [ ] **Step 2: Run, confirm FAIL**

Run: `cd gemini && python -m pytest tests/test_renderer.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/renderer.py`**

```python
from __future__ import annotations
import numpy as np
from skimage.color import rgb2lab, lab2rgb


def reinhard_tone_transfer(src_rgb: np.ndarray, target_rgb: np.ndarray, tau: float) -> np.ndarray:
    """
    Reinhard color transfer in LAB space, interpolated by tau ∈ [0, 1].
    tau=0 returns src unchanged; tau=1 shifts src's per-channel mean/std toward target's.
    Inputs are uint8 HxWx3; output uint8 HxWx3.
    """
    if tau <= 0.0:
        return src_rgb.copy()
    src_lab = rgb2lab(src_rgb.astype(np.float32) / 255.0)
    tgt_lab = rgb2lab(target_rgb.astype(np.float32) / 255.0)
    src_mean = src_lab.reshape(-1, 3).mean(axis=0)
    src_std = src_lab.reshape(-1, 3).std(axis=0) + 1e-6
    tgt_mean = tgt_lab.reshape(-1, 3).mean(axis=0)
    tgt_std = tgt_lab.reshape(-1, 3).std(axis=0) + 1e-6
    # Full Reinhard: shifted = (src - src_mean) * (tgt_std/src_std) + tgt_mean
    shifted = (src_lab - src_mean) * (tgt_std / src_std) + tgt_mean
    blended = (1.0 - tau) * src_lab + tau * shifted
    out_rgb = np.clip(lab2rgb(blended), 0.0, 1.0)
    return (out_rgb * 255.0 + 0.5).astype(np.uint8)
```

- [ ] **Step 4: Run, confirm PASS**

Run: `cd gemini && python -m pytest tests/test_renderer.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit + changelog**

```yaml
- date: 2026-04-17
  type: feat
  target: src/renderer.py
  change: reinhard_tone_transfer(src, target, tau) — LAB-space per-channel mean/std shift, interpolated by tau
  rationale: 玩点 C — default commercial tools run tau≈0.9 (over-tinted); we expose tau so 0.4–0.6 sweet spot is reachable
  action: rgb2lab on both → shift src stats toward target stats → interpolate by tau → lab2rgb back
  result: 3/3 tests (identity at 0, target-dominant at 1, monotonic interpolation)
  validation: pytest tests/test_renderer.py -v
  status: stable
```

```bash
git add src/renderer.py tests/test_renderer.py CHANGELOG.md
git commit -m "feat(renderer): Reinhard LAB tone transfer with tau blend"
```

---

## Task 12: Renderer — Paste Tiles + Usage Tracking

**Files:**
- Modify: `gemini/src/renderer.py` (append `render_mosaic`)
- Modify: `gemini/tests/test_renderer.py`

- [ ] **Step 1: Append failing tests**

```python
def test_render_mosaic_writes_expected_shape(tmp_path: Path):
    # 3 tiles (solid R, G, B), 2x3 grid
    tiles = [tmp_path / f"t{i}.jpg" for i in range(3)]
    colors = [(200, 20, 20), (20, 200, 20), (20, 20, 200)]
    for p, c in zip(tiles, colors):
        Image.fromarray(_solid_rgb(c, size=(32, 32))).save(p)
    assignment = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
    from src.renderer import render_mosaic
    # Passing tau=0 means source tile colors are preserved
    cell_rgb = np.zeros((2, 3, 2, 2, 3), dtype=np.uint8)  # placeholder target, irrelevant at tau=0
    img, usage = render_mosaic(assignment, [str(p) for p in tiles], cell_rgb, tile_px=16, tau=0.0)
    assert img.size == (3 * 16, 2 * 16)
    assert usage == {0: 2, 1: 2, 2: 2}


def test_render_mosaic_uses_tone_transfer_when_tau_nonzero(tmp_path: Path):
    # One vivid red tile; target cell is blue. At tau=1, the rendered pixel should be bluish.
    tile = tmp_path / "red.jpg"
    Image.fromarray(_solid_rgb((200, 20, 20), size=(32, 32))).save(tile)
    assignment = np.array([[0]], dtype=np.int64)
    cell_rgb = np.full((1, 1, 2, 2, 3), (20, 20, 200), dtype=np.uint8)
    from src.renderer import render_mosaic
    img, _ = render_mosaic(assignment, [str(tile)], cell_rgb, tile_px=16, tau=1.0)
    arr = np.asarray(img)
    # Mean B channel should now exceed mean R channel (original was opposite)
    assert arr[..., 2].mean() > arr[..., 0].mean()
```

- [ ] **Step 2: Run, confirm FAIL**

Run: `cd gemini && python -m pytest tests/test_renderer.py -v`
Expected: 2 new failures.

- [ ] **Step 3: Append implementation**

```python
from PIL import Image as _Image
from pathlib import Path as _Path
from collections import Counter


def _load_and_fit(path: str, size: int) -> np.ndarray:
    with _Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        # center-crop to square, then resize to size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        im = im.crop((left, top, left + s, top + s)).resize((size, size), _Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def render_mosaic(assignment: np.ndarray, tile_paths: list[str],
                  cell_rgb: np.ndarray, tile_px: int, tau: float
                  ) -> tuple[_Image.Image, dict[int, int]]:
    """
    assignment: (rows, cols) tile indices
    tile_paths: list of file paths, indexed by assignment values
    cell_rgb:   (rows, cols, patch_px, patch_px, 3) uint8 — target color per cell (for tone transfer)
    Returns (PIL image, usage_counter).
    """
    rows, cols = assignment.shape
    out = np.zeros((rows * tile_px, cols * tile_px, 3), dtype=np.uint8)
    tile_cache: dict[int, np.ndarray] = {}
    usage: Counter[int] = Counter()
    for r in range(rows):
        for c in range(cols):
            t = int(assignment[r, c])
            usage[t] += 1
            if t not in tile_cache:
                tile_cache[t] = _load_and_fit(tile_paths[t], tile_px)
            tile_img = tile_cache[t]
            if tau > 0.0:
                # Use the target cell's mean color as a (1x1) reference, tiled
                tgt_mean = cell_rgb[r, c].reshape(-1, 3).mean(axis=0).astype(np.uint8)
                tgt_patch = np.full((tile_px, tile_px, 3), tgt_mean, dtype=np.uint8)
                tile_img = reinhard_tone_transfer(tile_img, tgt_patch, tau=tau)
            out[r*tile_px:(r+1)*tile_px, c*tile_px:(c+1)*tile_px] = tile_img
    return _Image.fromarray(out), dict(usage)
```

- [ ] **Step 4: Run, confirm PASS**

Run: `cd gemini && python -m pytest tests/test_renderer.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit + changelog**

```yaml
- date: 2026-04-17
  type: feat
  target: src/renderer.py
  change: render_mosaic(assignment, tile_paths, cell_rgb, tile_px, tau) — paste center-cropped tiles with optional Reinhard tone transfer toward target cell mean; returns (Image, usage Counter)
  rationale: Keeps "load tile once" caching inside the renderer; usage dict feeds the self-mocking report
  action: Center-crop to square, LANCZOS resize to tile_px, tone-transfer using target cell's mean RGB if tau>0
  result: 2/2 new tests pass (shape + usage + tone-transfer direction)
  validation: pytest tests/test_renderer.py -v
  status: stable
```

```bash
git add src/renderer.py tests/test_renderer.py CHANGELOG.md
git commit -m "feat(renderer): paste tiles with usage tracking + per-cell tone transfer"
```

---

## Task 13: Reporter — Self-Mocking Text Report

**Files:**
- Create: `gemini/src/reporter.py`
- Test: `gemini/tests/test_reporter.py`

- [ ] **Step 1: Write the failing test**

`gemini/tests/test_reporter.py`:
```python
from src.reporter import build_text_report


def test_text_report_contains_key_stats():
    tile_paths = [f"/tile/{i}.jpg" for i in range(5)]
    usage = {0: 89, 1: 5, 2: 2, 3: 0, 4: 0}
    total_cells = sum(usage.values())
    tags = {
        "/tile/0.jpg": "2019 Japan trip",
        "/tile/1.jpg": "2019 Japan trip",
        "/tile/2.jpg": "work",
        "/tile/3.jpg": "untagged",
        "/tile/4.jpg": "untagged",
    }
    report = build_text_report(tile_paths, usage, tags, total_cells)
    # Must mention total photos used vs pool size
    assert "5" in report  # pool size
    assert "3" in report  # actually-used count
    # Must mention the top offender
    assert "/tile/0.jpg" in report
    assert "89" in report
    # Must mention cold photos
    assert "冷宫" in report or "cold" in report.lower()
    # Must mention top tag breakdown
    assert "Japan" in report
```

- [ ] **Step 2: Run, confirm FAIL**

Run: `cd gemini && python -m pytest tests/test_reporter.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/reporter.py`**

```python
from __future__ import annotations
from typing import Mapping
from pathlib import Path


def build_text_report(tile_paths: list[str], usage: Mapping[int, int],
                      tags: Mapping[str, str], total_cells: int) -> str:
    pool_size = len(tile_paths)
    used = [(i, n) for i, n in usage.items() if n > 0]
    used_count = len(used)
    cold = [p for i, p in enumerate(tile_paths) if usage.get(i, 0) == 0]
    # Top 5 most-used
    top = sorted(used, key=lambda x: -x[1])[:5]
    # Tag shares (weighted by usage count)
    tag_counts: dict[str, int] = {}
    for i, n in usage.items():
        tag = tags.get(tile_paths[i], "untagged")
        tag_counts[tag] = tag_counts.get(tag, 0) + n
    tag_share = sorted(tag_counts.items(), key=lambda x: -x[1])

    lines: list[str] = []
    lines.append(f"本次使用了你 {pool_size} 张照片里的 {used_count} 张 ({used_count / max(pool_size, 1):.0%}).")
    lines.append("")
    lines.append("最勤奋的拼豆 (用得最多的 5 张):")
    for i, n in top:
        name = Path(tile_paths[i]).name
        lines.append(f"  - {tile_paths[i]} ({name}): 被用了 {n} 次 ({n / total_cells:.1%} 的格子)")
    lines.append("")
    lines.append(f"冷宫照片 ({len(cold)} 张一次都没被用上):")
    for p in cold[:5]:
        lines.append(f"  - {p}")
    if len(cold) > 5:
        lines.append(f"  ... 还有 {len(cold) - 5} 张")
    lines.append("")
    lines.append("构成配方 (按标签):")
    for tag, n in tag_share:
        lines.append(f"  - {tag}: {n / total_cells:.1%} ({n} / {total_cells} 格)")
    return "\n".join(lines)
```

- [ ] **Step 4: Run, confirm PASS**

Run: `cd gemini && python -m pytest tests/test_reporter.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit + changelog**

```yaml
- date: 2026-04-17
  type: feat
  target: src/reporter.py
  change: build_text_report(tile_paths, usage, tags, total_cells) — top-5 offenders, cold-palace photos, tag-weighted composition breakdown
  rationale: This IS the shareable artifact (朋友圈 one-shot), far more viral than the mosaic itself
  action: Sort usage desc, enumerate zero-use paths, tag-weight by cell count
  result: 1/1 test passes; output contains all key stats
  validation: pytest tests/test_reporter.py -v
  status: stable
```

```bash
git add src/reporter.py tests/test_reporter.py CHANGELOG.md
git commit -m "feat(reporter): self-mocking text report"
```

---

## Task 14: Reporter — Usage Histogram + Cold-Photo Wall

**Files:**
- Modify: `gemini/src/reporter.py` (append `save_usage_plot`, `save_cold_wall`)
- Modify: `gemini/tests/test_reporter.py`

- [ ] **Step 1: Append failing tests**

```python
def test_save_usage_plot_writes_png(tmp_path: Path):
    from src.reporter import save_usage_plot
    usage = {i: (100 - i * 5) for i in range(20)}
    out_path = tmp_path / "hist.png"
    save_usage_plot(usage, str(out_path))
    assert out_path.exists() and out_path.stat().st_size > 0


def test_save_cold_wall_writes_png(tmp_path: Path):
    from src.reporter import save_cold_wall
    tile_paths = []
    for i in range(6):
        p = tmp_path / f"t{i}.jpg"
        Image.new("RGB", (64, 64), (i * 40, 100, 100)).save(p)
        tile_paths.append(str(p))
    usage = {0: 10, 1: 5}  # 2..5 cold
    out_path = tmp_path / "cold.png"
    save_cold_wall(tile_paths, usage, str(out_path), cols=2, tile_px=32)
    assert out_path.exists() and out_path.stat().st_size > 0
    img = Image.open(out_path)
    # 4 cold tiles, cols=2 → 2 rows × 2 cols
    assert img.size == (2 * 32, 2 * 32)
```

- [ ] **Step 2: Run, confirm FAIL**

Run: `cd gemini && python -m pytest tests/test_reporter.py -v`
Expected: 2 new failures.

- [ ] **Step 3: Append implementation**

```python
def save_usage_plot(usage: Mapping[int, int], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = sorted(usage.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(counts)), counts, width=1.0)
    ax.set_xlabel("tile rank (most-used → least-used)")
    ax.set_ylabel("use count")
    ax.set_title("Tile usage distribution (long tail = healthy diversity)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_cold_wall(tile_paths: list[str], usage: Mapping[int, int],
                   out_path: str, cols: int = 10, tile_px: int = 64) -> None:
    from PIL import Image as _I
    cold_paths = [p for i, p in enumerate(tile_paths) if usage.get(i, 0) == 0]
    if not cold_paths:
        # Still write a placeholder so caller can rely on file presence
        _I.new("RGB", (tile_px, tile_px), (40, 40, 40)).save(out_path)
        return
    rows = (len(cold_paths) + cols - 1) // cols
    canvas = _I.new("RGB", (cols * tile_px, rows * tile_px), (0, 0, 0))
    for i, p in enumerate(cold_paths):
        with _I.open(p) as im:
            im = im.convert("RGB")
            s = min(im.size)
            left = (im.size[0] - s) // 2
            top = (im.size[1] - s) // 2
            im = im.crop((left, top, left + s, top + s)).resize((tile_px, tile_px), _I.LANCZOS)
        canvas.paste(im, ((i % cols) * tile_px, (i // cols) * tile_px))
    canvas.save(out_path)
```

- [ ] **Step 4: Run, confirm PASS**

Run: `cd gemini && python -m pytest tests/test_reporter.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit + changelog**

```yaml
- date: 2026-04-17
  type: feat
  target: src/reporter.py
  change: save_usage_plot (matplotlib long-tail bar) + save_cold_wall (grid of unused tiles)
  rationale: Visual companions to the text report; cold-photo wall is the "here are your forgotten memories" payoff
  action: matplotlib Agg backend for headless rendering; tile wall center-crops each cold photo to square
  result: 2/2 new tests pass
  validation: pytest tests/test_reporter.py -v
  status: stable
```

```bash
git add src/reporter.py tests/test_reporter.py CHANGELOG.md
git commit -m "feat(reporter): usage histogram + cold-photo wall"
```

---

## Task 15: DeepZoom — Pyramid + OpenSeadragon HTML

**Files:**
- Create: `gemini/src/deepzoom.py`
- Test: `gemini/tests/test_deepzoom.py`

- [ ] **Step 1: Write the failing test**

`gemini/tests/test_deepzoom.py`:
```python
from pathlib import Path
import numpy as np
from PIL import Image
from src.deepzoom import export_deepzoom


def test_export_deepzoom_writes_files(tmp_path: Path):
    src = tmp_path / "big.png"
    Image.fromarray((np.random.rand(512, 512, 3) * 255).astype(np.uint8)).save(src)
    out_dir = tmp_path / "out"
    export_deepzoom(str(src), str(out_dir), title="Test")
    # Required outputs
    assert (out_dir / "index.html").exists()
    # Some pyramid tiles must exist somewhere under out_dir
    tiles = list(out_dir.rglob("*.jpeg")) + list(out_dir.rglob("*.jpg")) + list(out_dir.rglob("*.png"))
    assert len(tiles) > 0
    html = (out_dir / "index.html").read_text()
    # Wired to OpenSeadragon + the dzi descriptor
    assert "openseadragon" in html.lower()
    assert ".dzi" in html
    assert "Test" in html
```

- [ ] **Step 2: Run, confirm FAIL**

Run: `cd gemini && python -m pytest tests/test_deepzoom.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/deepzoom.py`**

Note: pyvips requires libvips installed (macOS: `brew install vips`). The function falls back to the `deepzoom` pure-Python package if pyvips import fails — both pin to the same DZI layout so the HTML template is identical.

```python
from __future__ import annotations
from pathlib import Path


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  html, body {{ margin:0; padding:0; height:100%; background:#111; color:#ddd; font-family:system-ui; }}
  #meta {{ position:fixed; left:12px; bottom:12px; opacity:.7; font-size:12px; z-index:10; }}
  #viewer {{ width:100%; height:100%; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/openseadragon.min.js"></script>
</head>
<body>
<div id="viewer"></div>
<div id="meta">{title} — zoom in. that one tile is the photo from that day.</div>
<script>
OpenSeadragon({{
  id: "viewer",
  prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/images/",
  tileSources: "mosaic.dzi",
  showNavigator: true,
  defaultZoomLevel: 0,
  minZoomLevel: 0.2
}});
</script>
</body>
</html>
"""


def export_deepzoom(src_image: str, out_dir: str, title: str = "Photomosaic") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dzi_stem = out / "mosaic"
    try:
        import pyvips  # type: ignore
        pyvips.Image.new_from_file(src_image, access="sequential").dzsave(
            str(dzi_stem), layout="dz", suffix=".jpeg[Q=85]"
        )
    except (ImportError, OSError):
        # Fallback: deepzoom.py
        import deepzoom  # type: ignore
        creator = deepzoom.ImageCreator(
            tile_size=254, tile_overlap=1, tile_format="jpg",
            image_quality=0.85, resize_filter="antialias",
        )
        creator.create(src_image, f"{dzi_stem}.dzi")
    (out / "index.html").write_text(_HTML_TEMPLATE.format(title=title))
```

Because the `deepzoom` Python package is not in `requirements.txt` (optional fallback), document this in the function docstring and README.

- [ ] **Step 4: Run, confirm PASS (requires either pyvips or deepzoom)**

Run: `cd gemini && python -m pytest tests/test_deepzoom.py -v`
If both libs missing, `pip install pyvips` or `pip install deepzoom`; pyvips additionally needs `brew install vips` on macOS.
Expected: 1 passed.

- [ ] **Step 5: Commit + changelog**

```yaml
- date: 2026-04-17
  type: feat
  target: src/deepzoom.py
  change: export_deepzoom(src, out_dir, title) writes DZI pyramid + OpenSeadragon index.html
  rationale: The "send a link, friend zooms for 30 min" feature — cheapest per-line-of-code fun (玩点 "缩放惊喜")
  action: pyvips.dzsave preferred; falls back to deepzoom.py if pyvips unavailable; CDN-hosted OpenSeadragon loader
  result: 1/1 test passes; HTML references mosaic.dzi
  validation: pytest tests/test_deepzoom.py -v
  status: stable
```

```bash
git add src/deepzoom.py tests/test_deepzoom.py CHANGELOG.md
git commit -m "feat(deepzoom): DZI pyramid + OpenSeadragon HTML export"
```

---

## Task 16: Notebook — Wire 8 Cells

**Files:**
- Create: `gemini/photomosaic.py` (jupytext source with `# %%` cells)
- Generate: `gemini/photomosaic.ipynb`

- [ ] **Step 1: Write `photomosaic.py` with 8 cells**

```python
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Photomosaic Toy
# Local-only. Deliberately slow. Deliberately explainable.
# Edit Cell 2 config, then "Run All".

# %%
# Cell 1 — Imports
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
from PIL import Image
from src.config import PhotomosaicConfig
from src.tile_pool import load_or_build_index, add_clip_embeddings, load_tags
from src.target import split_into_patches
from src.matcher import color_topk, assign_with_penalties, assign_with_clip
from src.renderer import render_mosaic
from src.reporter import build_text_report, save_usage_plot, save_cold_wall
from src.deepzoom import export_deepzoom
print("imports ok")

# %%
# Cell 2 — Config (your only UI)
cfg = PhotomosaicConfig(
    target="samples/target.jpg",
    tile_dir="samples/tiles",
    grid=(120, 68),
    tile_px=16,
    lambda_repeat=0.3,
    mu_neighbor=0.5,
    tau_tone=0.5,
    use_clip=False,
    mode="normal",
    output_dir="output",
)
Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
print(cfg)

# %%
# Cell 3 — Scan tile pool
idx = load_or_build_index(cfg.tile_dir, cache_dir=cfg.cache_dir)
tags = load_tags(cfg.tile_dir, idx.paths)
print(f"Tile pool: {len(idx.paths)} photos")
if cfg.use_clip:
    idx = add_clip_embeddings(idx)
    print(f"CLIP embeddings: {idx.clip_emb.shape}")

# %%
# Cell 4 — Target split
patches_lab, cell_rgb = split_into_patches(cfg.target, grid=cfg.grid)
print(f"Target split into {patches_lab.shape[0]} x {patches_lab.shape[1]} cells")

# %%
# Cell 5 — Match (with live narration)
topk_idx, topk_dist = color_topk(patches_lab, idx.lab_mean, k=cfg.topk_color)
total_cells = patches_lab.shape[0] * patches_lab.shape[1]

def narrate(r, c, chosen, candidates):
    if (r * patches_lab.shape[1] + c) % max(total_cells // 40, 1) == 0:
        top3 = sorted(candidates, key=lambda x: x[1])[:3]
        chosen_name = Path(idx.paths[chosen]).name
        runners = ", ".join(f"{Path(idx.paths[t]).name}({s:.1f})" for t, s in top3)
        print(f"[{r:3d},{c:3d}] picked {chosen_name} — top3 by score: {runners}")

if cfg.use_clip and idx.clip_emb is not None:
    # Compute per-cell CLIP emb from upscaled target patch
    import open_clip, torch
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    rows, cols = patches_lab.shape[:2]
    # Use the larger rescaled target to extract meaningful patches
    full = Image.open(cfg.target).convert("RGB").resize((cols * 32, rows * 32), Image.LANCZOS)
    patch_embs = np.zeros((rows, cols, idx.clip_emb.shape[1]), dtype=np.float32)
    with torch.no_grad():
        for r in range(rows):
            batch = [preprocess(full.crop((c*32, r*32, (c+1)*32, (r+1)*32))) for c in range(cols)]
            feat = model.encode_image(torch.stack(batch).to(device))
            feat = feat / feat.norm(dim=-1, keepdim=True)
            patch_embs[r] = feat.cpu().numpy()
    assignment = assign_with_clip(topk_idx, topk_dist, patches_lab, idx.lab_mean,
                                  tile_clip=idx.clip_emb, patch_clip=patch_embs,
                                  lambda_repeat=cfg.lambda_repeat, mu_neighbor=cfg.mu_neighbor,
                                  clip_weight=cfg.clip_weight, on_cell=narrate)
else:
    assignment = assign_with_penalties(topk_idx, topk_dist,
                                       lambda_repeat=cfg.lambda_repeat,
                                       mu_neighbor=cfg.mu_neighbor,
                                       on_cell=narrate)

# %%
# Cell 6 — Render
mosaic, usage = render_mosaic(assignment, idx.paths, cell_rgb,
                               tile_px=cfg.tile_px, tau=cfg.tau_tone)
mosaic_path = Path(cfg.output_dir) / "mosaic.png"
mosaic.save(mosaic_path)
print(f"wrote {mosaic_path}  ({mosaic.size[0]}×{mosaic.size[1]} px)")
mosaic

# %%
# Cell 7 — Report
report = build_text_report(idx.paths, usage, tags, total_cells)
print(report)
save_usage_plot(usage, str(Path(cfg.output_dir) / "usage_histogram.png"))
save_cold_wall(idx.paths, usage, str(Path(cfg.output_dir) / "cold_wall.png"))
(Path(cfg.output_dir) / "report.txt").write_text(report)

# %%
# Cell 8 — DeepZoom HTML (send this folder to a friend)
dz_dir = Path(cfg.output_dir) / "deepzoom"
export_deepzoom(str(mosaic_path), str(dz_dir), title=f"Photomosaic: {Path(cfg.target).stem}")
print(f"open {dz_dir}/index.html in a browser (or serve the folder)")
```

- [ ] **Step 2: Convert to notebook**

Run: `cd gemini && jupytext --to ipynb photomosaic.py`
Expected: creates `photomosaic.ipynb`.

- [ ] **Step 3: Syntax-check the Python source**

Run: `cd gemini && python -c "import ast; ast.parse(Path('photomosaic.py').read_text()); print('ok')" | cat`
Actually simpler: `cd gemini && python -m py_compile photomosaic.py`
Expected: no output (successful compile).

- [ ] **Step 4: Commit + changelog**

```yaml
- date: 2026-04-17
  type: feat
  target: photomosaic.py, photomosaic.ipynb
  change: 8-cell jupytext notebook wiring all modules, with live narration hook in Cell 5 and DeepZoom export in Cell 8
  rationale: Cell 2 is the user's entire UI surface (per "toy" positioning); narrate() shows the algorithm thinking (玩点 "能看见算法在思考")
  action: .py authored with # %% percent-format cells, converted to .ipynb via jupytext --to ipynb
  result: py_compile passes; ipynb generated
  validation: python -m py_compile photomosaic.py && ls photomosaic.ipynb
  status: stable
```

```bash
git add photomosaic.py photomosaic.ipynb CHANGELOG.md
git commit -m "feat(notebook): 8-cell photomosaic pipeline with live narration"
```

---

## Task 17: End-to-End Smoke Test

**Files:**
- Create: `gemini/tests/test_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
"""End-to-end smoke: generate tiny synthetic target + tile pool, run full pipeline, check artifacts."""
from pathlib import Path
import numpy as np
from PIL import Image


def test_full_pipeline_produces_outputs(tmp_path: Path):
    # 1) Build tiny tile pool: 30 solid-color tiles
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(30):
        color = tuple(int(x) for x in rng.integers(0, 256, size=3))
        Image.new("RGB", (64, 64), color).save(tile_dir / f"t{i:03d}.jpg")

    # 2) Build a small target (gradient)
    xs = np.linspace(0, 255, 96, dtype=np.uint8)
    ys = np.linspace(0, 255, 56, dtype=np.uint8)
    tgt = np.stack([
        np.tile(xs, (56, 1)),
        np.tile(ys[:, None], (1, 96)),
        np.full((56, 96), 128, dtype=np.uint8),
    ], axis=-1)
    target_path = tmp_path / "target.png"
    Image.fromarray(tgt).save(target_path)

    out_dir = tmp_path / "output"

    # 3) Run pipeline
    from src.config import PhotomosaicConfig
    from src.tile_pool import load_or_build_index, load_tags
    from src.target import split_into_patches
    from src.matcher import color_topk, assign_with_penalties
    from src.renderer import render_mosaic
    from src.reporter import build_text_report, save_usage_plot, save_cold_wall

    cfg = PhotomosaicConfig(
        target=str(target_path),
        tile_dir=str(tile_dir),
        grid=(48, 28),           # small grid
        tile_px=8,
        lambda_repeat=0.5,
        mu_neighbor=0.5,
        tau_tone=0.4,
        cache_dir=str(tmp_path / ".cache"),
        output_dir=str(out_dir),
        topk_color=8,
    )
    idx = load_or_build_index(cfg.tile_dir, cache_dir=cfg.cache_dir)
    tags = load_tags(cfg.tile_dir, idx.paths)
    patches_lab, cell_rgb = split_into_patches(cfg.target, grid=cfg.grid)
    topk_idx, topk_dist = color_topk(patches_lab, idx.lab_mean, k=cfg.topk_color)
    assignment = assign_with_penalties(topk_idx, topk_dist,
                                       lambda_repeat=cfg.lambda_repeat,
                                       mu_neighbor=cfg.mu_neighbor)
    out_dir.mkdir(parents=True, exist_ok=True)
    mosaic, usage = render_mosaic(assignment, idx.paths, cell_rgb,
                                  tile_px=cfg.tile_px, tau=cfg.tau_tone)
    mosaic_path = out_dir / "mosaic.png"
    mosaic.save(mosaic_path)

    total_cells = patches_lab.shape[0] * patches_lab.shape[1]
    report = build_text_report(idx.paths, usage, tags, total_cells)
    (out_dir / "report.txt").write_text(report)
    save_usage_plot(usage, str(out_dir / "usage.png"))
    save_cold_wall(idx.paths, usage, str(out_dir / "cold.png"))

    # 4) Assertions
    assert mosaic_path.exists() and mosaic_path.stat().st_size > 0
    assert mosaic.size == (48 * 8, 28 * 8)
    assert (out_dir / "report.txt").read_text().strip() != ""
    assert (out_dir / "usage.png").exists()
    assert (out_dir / "cold.png").exists()
    # Diversity penalty should spread usage
    assert len([n for n in usage.values() if n > 0]) >= 5
```

- [ ] **Step 2: Run, confirm PASS**

Run: `cd gemini && python -m pytest tests/test_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 3: Run the full suite**

Run: `cd gemini && python -m pytest -v`
Expected: all green (with CLIP test skipped if open_clip not installed; deepzoom test passes if pyvips or deepzoom.py installed).

- [ ] **Step 4: Commit + changelog**

```yaml
- date: 2026-04-17
  type: test
  target: tests/test_smoke.py
  change: End-to-end smoke: synthetic 30-tile pool + gradient target → run matcher → renderer → reporter; assert mosaic shape + usage diversity
  rationale: Catches wiring regressions between modules that unit tests miss; lets user sanity-check on a throwaway pool before a real run
  action: tmp_path fixture builds tile pool and target; runs full pipeline end-to-end
  result: 1/1 test passes; full suite green
  validation: pytest -v
  status: stable
```

```bash
git add tests/test_smoke.py CHANGELOG.md
git commit -m "test: end-to-end smoke test of full mosaic pipeline"
```

---

## Task 18: README Polish + Example `samples/`

**Files:**
- Modify: `gemini/README.md`
- Create: `gemini/samples/README.md`

Add to README: step-by-step first-run, what each output file means, knob-tuning intuition (tau sweet spot 0.4–0.6, lambda when one photo dominates, mu when clumping), how to set up `tags.json`, how to enable CLIP mode, cursed mode idea gallery ("用所有表情包拼证件照" etc.).

- [ ] **Step 1: Rewrite README**

Replace `gemini/README.md` with the expanded version (keep it concise — ~100 lines — and include the knob-tuning intuition table):

```markdown
# Photomosaic Toy

Re-compose a target image from your personal photo library. Local-only, deliberately slow, deliberately explainable.

## Install

```bash
pip install -r requirements.txt
# macOS DeepZoom dep:
brew install vips
# Optional CLIP mode:
# already in requirements.txt (open_clip_torch, torch)
```

## First run (with sample data)

1. Put a target image at `samples/target.jpg` (1080p works well).
2. Put 500–5000 photos under `samples/tiles/` (any subfolder layout; `.jpg`/`.png`).
3. `jupytext --to ipynb photomosaic.py && jupyter notebook photomosaic.ipynb`
4. Run all cells. First run takes longer (caches tile-pool LAB means); subsequent runs are fast.

## Outputs (in `output/`)

| File | What |
|---|---|
| `mosaic.png` | The final image |
| `report.txt` | Self-mocking stats ("IMG_0217.jpg was used 89 times, mostly for sky") |
| `usage_histogram.png` | Long-tail bar chart of tile usage |
| `cold_wall.png` | Grid of photos that were never chosen ("the cold palace") |
| `deepzoom/index.html` | Open in a browser, infinite zoom |

## Tuning the knobs (Cell 2)

| Knob | Effect | When to raise |
|---|---|---|
| `lambda_repeat` | Spreads usage — higher means no single photo dominates | You see one face 200 times in the mosaic |
| `mu_neighbor` | Prevents the same tile appearing in adjacent cells | You see tile clusters/clumps |
| `tau_tone` | Reinhard LAB tone transfer strength (0..1) | 0 = photos untouched, 1 = full target match. **Sweet spot: 0.4–0.6** |
| `use_clip` | Semantic reranking — blue regions prefer real sea photos | You want "this is weirdly correct" moments |

## Tagging your pool (optional)

Create `tile_dir/tags.json`:
```json
{
  "2019_Japan/**/*": "2019 Japan trip",
  "family/**/*": "family photos",
  "work/**/*": "ex-coworkers"
}
```
Report will then tell you "23% of your mosaic came from the 2019 Japan trip".

## Modes (`cfg.mode`)

- `normal`: default
- `cursed`: (v2 idea) use all emoji as tiles to build a serious portrait
- `time_capsule`: (v2 idea) bucket tiles by EXIF year, report the per-year composition

## What each module does

- `src/config.py` — the one dataclass that holds all UI state
- `src/tile_pool.py` — scan, LAB-mean cache, optional CLIP embeddings, tag loading
- `src/target.py` — target → grid of LAB means + source RGB patches
- `src/matcher.py` — FAISS color kNN + repeat/neighbor/CLIP rerank + live narration callback
- `src/renderer.py` — Reinhard LAB tone transfer + tile paste + usage counter
- `src/reporter.py` — self-mocking text + usage histogram + cold-photo wall
- `src/deepzoom.py` — DZI pyramid + OpenSeadragon HTML

## Run tests

```bash
pytest -v
```

## Plan / changelog

- Implementation plan: `docs/superpowers/plans/2026-04-17-photomosaic-toy.md`
- Change history: `CHANGELOG.md` (agent-oriented; verbose by design)
```

- [ ] **Step 2: Create `samples/README.md`**

```markdown
# Sample data directory

Put your target image at `samples/target.jpg` and your tile pool under `samples/tiles/` before running the notebook.

For a self-contained demo, run `pytest tests/test_smoke.py -v` — it synthesizes a tile pool and target in a tmp dir and exercises the full pipeline.
```

- [ ] **Step 3: Commit + changelog**

```yaml
- date: 2026-04-17
  type: docs
  target: README.md, samples/README.md
  change: Expanded README with install (incl. brew vips), knob-tuning table, tag setup, per-module purpose
  rationale: README is the user's only onboarding surface since no GUI; tuning table turns 3 abstract sliders into actionable intuition
  action: Rewrite README; add samples/README.md pointer
  result: README reads end-to-end in under 3 minutes
  validation: manual read-through
  status: stable
```

```bash
git add README.md samples/README.md CHANGELOG.md
git commit -m "docs: expanded README with knob-tuning intuition"
```

---

## Self-Review

**Spec coverage:**
- ✅ Local ipynb + config as UI → Task 16 (photomosaic.py with 8 cells)
- ✅ "See it thinking" narration → Task 9's `on_cell` callback, wired in Task 16 Cell 5
- ✅ Self-mocking report → Task 13 (`build_text_report`)
- ✅ Usage histogram + cold-photo wall → Task 14
- ✅ Tile pool narrative/tags → Task 6 (`load_tags`) + Task 13 tag breakdown
- ✅ DeepZoom HTML output → Task 15
- ✅ λ repeat penalty → Task 9
- ✅ μ neighbor penalty → Task 9
- ✅ τ Reinhard LAB tone transfer → Tasks 11 + 12
- ✅ CLIP semantic match (玩点 A) → Tasks 5 + 10
- ✅ Tile pool caching (`.cache/`) → Task 4 pickle cache
- ✅ FAISS for color kNN → Task 8

**Cursed modes** — intentionally dropped from v1 as they are just config presets (which photos you feed as `tile_dir`); README mentions them as v2 ideas. The infrastructure (tags + tau + the three knobs) already supports running "所有表情包拼证件照" by pointing `tile_dir` at an emoji folder and tuning `tau_tone` low.

**Type consistency check:**
- `TileIndex` has `.paths: List[str]`, `.lab_mean: (N, 3)`, `.clip_emb: (N, D) | None`. Used consistently in Tasks 3, 4, 5, and in notebook Cell 3 / 5.
- `assignment: np.ndarray (rows, cols) int64` — used same way in Tasks 9, 10, 12, 16, 17.
- `cell_rgb: (rows, cols, patch_px, patch_px, 3) uint8` — same shape in Task 7 (producer), Task 12 (consumer), Task 16 wiring, Task 17 smoke.
- `usage: dict[int, int]` — produced by Task 12, consumed by Tasks 13, 14, 17.
- `patches_lab: (rows, cols, 3) float32` — consistent across Tasks 7, 8, 9, 10.
- Callback signature `on_cell(r, c, chosen, candidates)` — same in Tasks 9 and 10.

**Placeholder scan:** no TBD / "handle edge cases" / "similar to Task N" / "add validation" phrases; every code block is complete.

**One known caveat:** Task 15 (DeepZoom) requires either pyvips (needs `brew install vips` on macOS) or the `deepzoom` PyPI package. The test will fail without one of them installed. This is called out in Task 15 Step 4 and in the README install section. If the executing engineer hits a pyvips import error, the remedy is `pip install deepzoom` (pure-Python fallback) — no code change needed.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-17-photomosaic-toy.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
