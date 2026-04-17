# CHANGELOG

> Convention: entries are YAML with fields date(ISO)/type/target/change/rationale/action/result/validation/status.
> `try-failed` entries are never deleted, only compressed. See `docs/superpowers/plans/2026-04-17-photomosaic-toy.md` for full convention.

## 活跃条目

- date: 2026-04-17
  type: feat
  target: photomosaic.py, photomosaic.ipynb
  change: 8-cell jupytext notebook wiring all modules, with live narration hook in Cell 5 and DeepZoom export in Cell 8
  rationale: Cell 2 is the user's entire UI surface (per "toy" positioning); narrate() shows the algorithm thinking (玩点 "能看见算法在思考")
  action: .py authored with # %% percent-format cells, converted to .ipynb via jupytext --to ipynb
  result: py_compile passes; ipynb generated and parses as JSON
  validation: python -m py_compile photomosaic.py && python -c "import json, pathlib; json.loads(pathlib.Path('photomosaic.ipynb').read_text())"
  status: stable

- date: 2026-04-17
  type: fix
  target: src/deepzoom.py, tests/test_deepzoom.py
  change: Corrected DZI Width/Height to PIL-native (width, height) ordering; test used Image.new(size=(300,200)) so assertions match PIL convention
  rationale: Previous Task 15 commit had implementer swap width↔height in DZI XML to compensate for a buggy test that used np.random.rand(300,200,3) and then asserted Width="300". The swap would have produced wrong aspect ratio in real OpenSeadragon output for any non-square image
  action: Test now uses Image.new("RGB", (300, 200)) — unambiguous PIL (w, h) = (300, 200); implementation writes DZI Width=w, Height=h straight from im.size
  result: 3/3 deepzoom tests still pass; full suite 30/1 unchanged; mosaic.dzi now correctly reflects source image dimensions
  validation: pytest tests/test_deepzoom.py -v
  status: stable

- date: 2026-04-17
  type: feat
  target: src/deepzoom.py
  change: Pure-Python DZI pyramid + OpenSeadragon HTML export using only Pillow (no pyvips, no external deepzoom package)
  rationale: Plan originally specified pyvips.dzsave with deepzoom.py fallback; on this machine pyvips+libvips are absent and the `deepzoom` PyPI package has been withdrawn. Pillow-based implementation (~50 lines) aligns with "toy, no heavy deps" user positioning
  action: For each pyramid level 0..ceil(log2(max_dim)), LANCZOS-resize then crop tile_size+overlap squares; write mosaic.dzi XML + mosaic_files/<level>/<c>_<r>.jpg; ship an OpenSeadragon-loading index.html template
  result: 3/3 tests pass — files exist, pyramid has expected level count (0..log2), DZI XML has correct attrs
  validation: pytest tests/test_deepzoom.py -v
  status: stable

- date: 2026-04-17
  type: try-failed
  target: requirements.txt, deepzoom install path
  change: Planned pyvips + deepzoom-package dependency chain was unusable
  rationale: Plan's assumption (either pyvips via brew vips, or pip install deepzoom) failed: libvips not installed + deepzoom not on PyPI
  action: Attempted pip install deepzoom → "No matching distribution found"; checked which vips → not found
  result: Failed both paths; pivoted to pure-Python Pillow-only DZI generator
  problem_context: Task 15 needs a DZI pyramid + OpenSeadragon HTML output
  workaround_reason: For a local toy, asking user to install libvips (brew + build) or a non-existent pypi package is user-hostile
  next_action: Implement DZI pyramid directly in Python using Pillow (see sibling feat entry)
  next_result: Success — 3/3 tests pass with zero external binaries
  status: reverted

- date: 2026-04-17
  type: feat
  target: src/reporter.py
  change: save_usage_plot (matplotlib long-tail bar) + save_cold_wall (grid of unused tiles)
  rationale: Visual companions to the text report; cold-photo wall is the "here are your forgotten memories" payoff
  action: matplotlib Agg backend for headless rendering; tile wall center-crops each cold photo to square
  result: 2/2 new tests pass
  validation: pytest tests/test_reporter.py -v
  status: stable

- date: 2026-04-17
  type: feat
  target: src/reporter.py
  change: build_text_report(tile_paths, usage, tags, total_cells) — top-5 offenders, cold photos, tag-weighted composition breakdown
  rationale: This IS the shareable artifact (朋友圈 one-shot), far more viral than the mosaic itself
  action: Sort usage desc, enumerate zero-use paths, tag-weight by cell count
  result: 1/1 test passes; output contains all key stats
  validation: pytest tests/test_reporter.py -v
  status: stable

- date: 2026-04-17
  type: feat
  target: src/renderer.py
  change: render_mosaic(assignment, tile_paths, cell_rgb, tile_px, tau) — paste center-cropped tiles with optional Reinhard tone transfer toward target cell mean; returns (Image, usage Counter)
  rationale: Keeps "load tile once" caching inside the renderer; usage dict feeds the self-mocking report
  action: Center-crop to square, LANCZOS resize to tile_px, tone-transfer using target cell's mean RGB if tau>0
  result: 2/2 new tests pass (shape + usage + tone-transfer direction)
  validation: pytest tests/test_renderer.py -v
  status: stable

- date: 2026-04-17
  type: feat
  target: src/renderer.py
  change: reinhard_tone_transfer(src, target, tau) — LAB-space per-channel mean/std shift, interpolated by tau
  rationale: 玩点 C — default commercial tools run tau≈0.9 (over-tinted); we expose tau so 0.4–0.6 sweet spot is reachable
  action: rgb2lab on both → shift src stats toward target stats → interpolate by tau → lab2rgb back
  result: 3/3 tests (identity at 0, target-dominant at 1, monotonic interpolation)
  validation: pytest tests/test_renderer.py -v
  status: stable

- date: 2026-04-17
  type: feat
  target: src/matcher.py
  change: assign_with_clip(..., tile_clip, patch_clip, clip_weight) adds cosine similarity bonus to the penalty rerank
  rationale: 玩点 A — "blue regions pick real sea photos over blue walls"; weight=0 regression guarantees it never hurts plain mode
  action: Extends greedy scan with -clip_weight * dot(tile_emb, patch_emb); assumes L2-normalized inputs
  result: 2/2 new tests pass (tie-break + zero-weight regression)
  validation: pytest tests/test_matcher.py -v
  status: stable

- date: 2026-04-17
  type: feat
  target: src/matcher.py
  change: assign_with_penalties(topk_idx, topk_dist, lambda_repeat, mu_neighbor, on_cell) — greedy raster-scan with usage + neighbor penalties + per-cell callback
  rationale: lambda solves "one photo dominates" (玩点 B); mu solves "same photo clumps in a region"; on_cell gives the notebook its live-thinking printout (玩点 "能看见算法在思考")
  action: Greedy O(rows*cols*k), score = sqrt(L2 LAB dist) + lambda*log1p(usage) + mu*neighbor_clash
  result: 3/3 new tests pass; zero-penalty regression still passes
  validation: pytest tests/test_matcher.py -v
  status: stable

- date: 2026-04-17
  type: feat
  target: pyproject.toml
  change: Add 3-line pytest config pinning pythonpath=["."] and testpaths=["tests"]
  rationale: Preemptive fix against ambient sys.path collisions — machine has other /src dirs (gitlab-mr-analyzer/src with __init__.py, auto-yes/.../src) that are one same-named module away from hijacking our imports. The earlier conftest.py try-failed entry predicted pyproject.toml pythonpath as the correct escape hatch; Task 8's implementer flagged that with 6+ source modules now, collision is increasingly likely
  action: Create pyproject.toml with [tool.pytest.ini_options] pythonpath=["."] testpaths=["tests"]
  result: 14 passed + 1 skipped (unchanged); now robust to future module-name collisions
  validation: pytest tests/ -v
  status: stable

- date: 2026-04-17
  type: feat
  target: src/matcher.py
  change: color_topk(patches_lab, tile_lab, k) returns (idx, dist) via faiss IndexFlatL2 in LAB space
  rationale: Exact nearest-neighbor in LAB is fast enough (tens of thousands of tiles) and perceptually correct; returns top-k so later reranking can consider alternatives
  action: faiss IndexFlatL2 on (N, 3) LAB means; batch search over flattened patches; cap k to N
  result: 2/2 tests pass (nearest wins + k capped to N)
  validation: pytest tests/test_matcher.py -v
  status: stable

- date: 2026-04-17
  type: feat
  target: src/target.py
  change: split_into_patches(path, grid, patch_px=2) returns (LAB mean per cell, source RGB per cell)
  rationale: Separates target parsing from matching; LAB mean drives match distance, source RGB drives tone transfer
  action: Resize to grid × patch_px, rgb2lab, reshape to (rows, patch_px, cols, patch_px, 3), mean over patch axes
  result: 2/2 tests pass, non-divisible sizes handled by LANCZOS resize
  validation: pytest tests/test_target.py -v
  status: stable

- date: 2026-04-17
  type: feat
  target: src/tile_pool.py
  change: load_tags(tile_dir, paths) reads tags.json (glob-pattern → tag name), defaults to 'untagged'
  rationale: Powers the "23% from 2019 Japan trip" narrative in final report — emotional payoff for user
  action: fnmatch on relative path against ordered JSON patterns; first match wins; ** patterns handled via regex translation (_glob_match helper)
  result: 2/2 new tests pass
  validation: pytest tests/test_tile_pool.py -v
  status: stable

- date: 2026-04-17
  type: feat
  target: src/tile_pool.py
  change: add_clip_embeddings(idx, model_name, pretrained) returns new TileIndex with L2-normalized CLIP image embeddings
  rationale: Enables "semantic" matching (玩点 A) — blue regions can prefer actual sea photos over blue walls
  action: open_clip.create_model_and_transforms → encode_image batched → L2 normalize → cpu().numpy(); imports are function-local so module stays importable without CLIP
  result: Test passes if open_clip installed, else skipped (currently skipped on this machine by design)
  validation: pytest tests/test_tile_pool.py -v
  status: stable

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
