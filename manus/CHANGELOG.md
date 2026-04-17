# CHANGELOG

> Maintained per agent-oriented convention: verbose, preserve try-failed chains, ISO dates, compress on 50 entries or 6 months.

## 活跃条目

- date: 2026-04-17
  type: feat
  target: entire project (manus/)
  change: Scaffold Tier B photomosaic toy (mosaic/ package + mosaic.ipynb) per spec docs/superpowers/specs/2026-04-17-tile-photo-mosaic-design.md
  rationale: User wants a local ipynb that builds a photo-of-photos mosaic, with interactive sliders, self-deprecating stats, and DeepZoom HTML export. Explicit vetoes: no torch, no CLIP, no Gradio ("不要太复杂").
  action: Create mosaic/ helper package (pool/target/match/render/report/dzi), mosaic.ipynb with 8 cells, pytest suite with fixtures
  result: Scaffolding in place; awaiting user's real pool + target to produce first real mosaic
  validation: pytest suite passes; smoke test produces a 20×12 px mosaic from 20-fixture pool
  status: experimental

- date: 2026-04-17
  type: try-failed
  target: mosaic/dzi.py
  change: Tried to depend on pypi `deepzoom` package for DZI pyramid export
  rationale: Plan originally listed `deepzoom` as a drop-in library to avoid implementing the DZI spec ourselves
  action: `pip install deepzoom` on Python 3.12 miniconda
  result: No matching distribution. Package appears abandoned or missing 3.12 wheels.
  validation: pip index versions deepzoom → ERROR: No matching distribution found for deepzoom
  problem_context: Need tile pyramid + XML descriptor + folder layout compatible with OpenSeadragon
  workaround_reason: Adding brew-install libvips (`pyvips`) violates the "不要太复杂" constraint — no non-pip setup steps.
  next_action: Write a ~50 LOC pure-PIL DZI generator directly in mosaic/dzi.py (DZI format is simple XML + halved-resolution tile pyramid)
  next_result: Succeeded (see 2026-04-17 feat(dzi) entry below)
  status: reverted

- date: 2026-04-17
  type: feat
  target: mosaic/dzi.py
  change: Pure-PIL DZI pyramid generator replacing the abandoned pypi `deepzoom` dep
  rationale: Replaces the failed `deepzoom` dep attempt (see try-failed entry above); keeps zero non-pip setup cost
  action: Implement level-by-level halving with `Image.resize(LANCZOS)`, emit `image.dzi` XML + `image_files/<level>/<col>_<row>.jpg` tiles with 1px overlap. `index.html` loads OpenSeadragon from jsdelivr CDN.
  result: Smoke test produces valid DZI + index.html; tested folder opens in browser
  validation: tests/test_dzi.py — verifies .dzi XML + image_files/ dir + html references dzi + OpenSeadragon CDN
  status: stable

- date: 2026-04-17
  type: validation
  target: entire project (manus/)
  change: Full pytest suite run after initial scaffold complete
  rationale: Confirm all module interfaces match spec and the end-to-end smoke pipeline produces a mosaic before handing off to the user
  action: `pytest -v` from manus/ (Python 3.12.4 miniconda on macOS arm64)
  result: 24 passed, 0 failed in 0.75s — covers lab_mean, scan_pool + cache + invalidation, load_and_slice center-crop, cKDTree top-K, diversity/neighbor penalty reranking, Reinhard tau=0/1 endpoints, compose per-tile tone transfer, text_report/cold_wall/hist_figure/describe_lab, dzi pyramid descriptor + tiles + html, end-to-end pipeline smoke
  validation: see commit message of final commit in this scaffold batch for full pytest output
  status: stable (modules); experimental (whole notebook — needs user's real pool + target for first real mosaic)
