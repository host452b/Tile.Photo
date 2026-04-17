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
