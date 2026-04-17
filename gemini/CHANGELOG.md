# CHANGELOG

> Convention: entries are YAML with fields date(ISO)/type/target/change/rationale/action/result/validation/status.
> `try-failed` entries are never deleted, only compressed. See `docs/superpowers/plans/2026-04-17-photomosaic-toy.md` for full convention.

## 活跃条目

- date: 2026-04-17
  type: feat
  target: src/tile_pool.py
  change: TileIndex dataclass + build_tile_index (LAB mean via skimage.rgb2lab on 64x64 thumbnail) + load_or_build_index with pickle cache keyed by sha1(abspath::min_side)
  rationale: LAB mean is perceptually better than RGB for color distance; pickle cache makes re-runs instant (user will tweak sliders many times)
  action: 64x64 LANCZOS thumbnail → rgb2lab → mean over pixels; cache file name is stable hash of tile_dir abspath
  result: 4/4 tests pass, cache survives source deletion
  validation: pytest tests/test_tile_pool.py -v
  status: stable

- date: 2026-04-17
  type: feat
  target: src/tile_pool.py
  change: scan_tile_dir(path, min_side=32) recursively returns valid jpg/png paths, skipping broken/tiny
  rationale: Tile pool input is user's chaotic photo dump — must survive broken files and thumbnails silently
  action: Pillow Image.verify + size check; skip UnidentifiedImageError/OSError
  result: 2/2 tests pass (recursive glob + min-side filter)
  validation: pytest tests/test_tile_pool.py -v
  status: stable

- date: 2026-04-17
  type: feat
  target: repo
  change: Initial scaffolding (requirements, gitignore, module skeletons, plan doc)
  rationale: Kick off photomosaic toy per user spec (local ipynb + fun features + DeepZoom output)
  action: Create requirements.txt, .gitignore, README.md, src/tests skeletons
  result: Repo has no code yet, only structure
  validation: Tree inspection + pytest collects zero tests successfully
  status: stable

- date: 2026-04-17
  type: fix
  target: requirements.txt, requirements-clip.txt, README.md, .gitignore
  change: Split torch/open_clip_torch into optional requirements-clip.txt; added samples/**/*.webp to gitignore; README quick-start notes the notebook is Task 16 deliverable
  rationale: Code review flagged that unconditional torch install (~2 GB, Apple Silicon wheel risk) is user-hostile for the non-CLIP default path; webp is common on modern phones/macOS screenshots
  action: Remove torch + open_clip_torch from requirements.txt into requirements-clip.txt; add webp glob to gitignore; annotate README quick-start
  result: Core install trims to ~200 MB; CLIP install is opt-in
  validation: pytest --collect-only still returns 0 tests; diff inspected
  status: stable

- date: 2026-04-17
  type: feat
  target: src/config.py
  change: PhotomosaicConfig dataclass with validation (tau in [0,1], positive grid/tile_px, mode enum)
  rationale: Centralize the 3 sliders (lambda/mu/tau) + modes so Cell 2 of notebook is the single UI surface
  action: Dataclass + __post_init__ validation; 4 tests for defaults + each invariant
  result: 4/4 tests pass
  validation: pytest tests/test_config.py -v
  status: stable

- date: 2026-04-17
  type: try-failed
  target: conftest.py
  change: Removed 41-line conftest.py that installed a custom MetaPathFinder to force src/ package resolution
  rationale: Implementer added it thinking global sys.path entries (gitlab-mr-analyzer/src, auto-yes/.../src) were hijacking `from src.config` imports
  action: Deleted conftest.py; verified pytest tests/test_config.py -v still reports 4/4 passed from gemini/ cwd
  result: Tests pass without the workaround — pytest's default rootdir-based import resolution handles the local src/ correctly when invoked from gemini/
  problem_context: ImportError for src.config when tests were (apparently) invoked from an unexpected cwd during implementer's initial pass
  workaround_reason: A 41-line MetaPathFinder is wildly over-engineered for a toy project; the actual issue (if any) would have been fixable with 3 lines of pyproject.toml pythonpath config
  next_action: If a future task sees genuine import collision, add `[tool.pytest.ini_options] pythonpath = ["."]` to pyproject.toml — not a MetaPathFinder
  next_result: pending (no collision seen yet)
  status: reverted
