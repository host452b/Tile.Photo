# Tile.Photo — Photomosaic Toy (manus branch)

- **Date**: 2026-04-17
- **Status**: Approved, pending implementation plan
- **Scope tier**: B — core + social-media-worthy extras
- **Location**: `Tile.Photo/manus/`
- **Hard constraints from user**: no torch, no CLIP, no deep learning, "不要太复杂"

## 1. Product Stance (why this design looks the way it does)

This is a **toy**, not a product. Design choices consciously invert typical product tradeoffs:

- **Speed doesn't matter** — 10-min runtime per image is fine
- **Generality doesn't matter** — it must run on this user's Mac, nothing else
- **Stability doesn't matter** — crashes are an acceptable UX ("崩了重跑")
- **Interpretability matters a lot** — "watching it think" is the core fun
- **Social shareability matters a lot** — output artifacts should be fun to send to friends

If you find yourself adding defensive error handling, retry logic, or config validation beyond the bare minimum, you're building the wrong thing.

## 2. Target User Stories

1. "I point it at a target photo + a folder of my photos → I get a mosaic PNG + a zoomable HTML I can send to the group chat."
2. "I tweak sliders (diversity, neighbor-spread, tone transfer) in the notebook and re-render; I can see the mosaic change in real time."
3. "After a run, I get a self-deprecating text report ('IMG_0217 was used 89 times, mostly for sky') that's good for a laugh."
4. "I see which of my photos were never picked (the 'cold wall') — these are mostly my blurry selfies, and I find that funny."

## 3. Architecture

### 3.1 File layout

```
manus/
├── mosaic.ipynb               # single entrypoint, 8 cells
├── mosaic/                    # helper package (cells stay short, modules stay testable)
│   ├── __init__.py
│   ├── pool.py                # scan pool dir → LAB mean + 32px thumb → pickle cache
│   ├── match.py               # cKDTree top-K + score reranking
│   ├── render.py              # Reinhard LAB tone transfer + tile paste
│   ├── report.py              # text report, usage histogram, cold wall
│   └── dzi.py                 # DeepZoom pyramid + OpenSeadragon index.html
├── cache/                     # tile feature cache, gitignored
├── out/                       # output artifacts per-target, gitignored
├── fixtures/                  # tiny test pool for smoke tests (20 images)
├── tests/                     # pytest smoke test
├── requirements.txt
├── CHANGELOG.md               # maintained per user's agent-oriented convention
└── .gitignore
```

### 3.2 The 8 notebook cells

| Cell | Purpose | Re-runs? |
|:---:|---|:---:|
| 1 | `%pip install -q -r requirements.txt` + imports | once |
| 2 | Config + ipywidgets (paths, grid, sliders λ/μ/τ, tile_px) | rarely |
| 3 | Pool scan → LAB means + 32px thumbs → pickle to `cache/pool_<hash>.pkl`, skip if cache hit | once per pool change |
| 4 | Load target image, fit to grid aspect (center-crop), slice → cell LAB means | per target |
| 5 | **Match** — for each cell in row-major order: cKDTree top-K=20 candidates → rerank by `score = LAB_dist + λ·log(1+uses) + μ·neighbor_sim` → pick. Prints e.g. `[2304/8160] cell (34,28) picked IMG_0217 (rank 4/20, uses=12, reason=low-usage)` every 50 cells. | per slider change |
| 6 | Render: pull thumb, optional Reinhard LAB transfer (strength τ), paste to canvas. Save `out/<stem>/mosaic.png` | per match |
| 7 | Report: text to `out/<stem>/report.txt`, matplotlib usage histogram, `cold_wall.png` (bottom-20 unused thumbs as a grid) | per match |
| 8 | DeepZoom export: pyramid + `out/<stem>/dzi/index.html` with OpenSeadragon | on demand |

A "Regenerate" button in Cell 2 triggers cells 5–7 (not 3). Cell 8 is a separate button.

## 4. Algorithm Details

### 4.1 Color space

All matching happens in **LAB** (perceptual uniformity). Convert once at pool-scan time (`skimage.color.rgb2lab`), store as float32.

### 4.2 Index

`scipy.spatial.cKDTree` on the (N, 3) LAB-means array. Pure python/C, no native brew deps, no torch.

### 4.3 Scoring

For cell `c` with LAB mean `L_c`, candidates are the top-K=20 color-nearest tiles.

```
score(t) = dist(L_c, L_t)                         # LAB Euclidean
        + λ · log(1 + uses[t])                    # diversity penalty
        + μ · avg_{n ∈ already-filled 8-neighbors of c} sim(t, chosen[n])
```

`sim(t, n)` = `1 - cos(L_t, L_n)` (LAB cosine dissimilarity). Pick the lowest-score candidate.

**Scan order**: row-major top-to-bottom. Early cells have no filled neighbors → `μ` term is 0. Acceptable asymmetry for a toy.

### 4.4 Reinhard tone transfer

Given cell LAB mean `μ_c`, cell LAB std `σ_c`, tile LAB mean `μ_t`, tile LAB std `σ_t`:

```
tile_lab_shifted = (tile_lab - μ_t) · (σ_c / σ_t) · τ + μ_t + (μ_c - μ_t) · τ
```

- `τ=0` → tile untouched (photos near-recognizable up close, mosaic fuzzy from afar)
- `τ=1` → full Reinhard match (mosaic perfect from afar, photos heavily tinted up close)
- Default **`τ=0.5`** per user's "sweet spot" observation.

### 4.5 Default parameters

| Param | Default | Range | Meaning |
|---|---:|---|---|
| `grid_w × grid_h` | 120 × 68 | 40–240 | grid columns × rows (16:9) |
| `tile_px` | 16 | 8–48 | pixel size of each tile in output |
| `λ` (diversity) | 2.0 | 0–10 | usage penalty weight |
| `μ` (neighbor) | 0.5 | 0–5 | neighbor similarity weight |
| `τ` (tone) | 0.5 | 0–1 | Reinhard transfer strength |
| `K` (candidates) | 20 | 5–50 | kNN shortlist size |

Output canvas default = 120·16 × 68·16 = **1920 × 1088 px**.

## 5. Outputs

Per run, in `out/<target_stem>/`:

- `mosaic.png` — the main artifact
- `report.txt` — self-deprecating stats (see §6)
- `cold_wall.png` — 20 least-used thumbs in a 4×5 matplotlib grid
- `dzi/index.html` + `dzi/image_files/` — zoomable HTML (shareable by zipping `dzi/`)

If a run overwrites a prior run for the same target stem, prior files are clobbered. No timestamping — this is a toy.

## 6. Report Format

```
Tile.Photo — mosaic of {target_name}
Generated 2026-04-17 21:15

Pool scanned: 3,241 images from /Users/joejiang/Pictures
Tiles placed: 8,160 (120 × 68)
Unique tiles used: 847 (26% of pool)

Most used (top 5):
  IMG_0217.jpg — 89 uses  (reason: mean LAB ≈ sky blue)
  IMG_4502.jpg — 71 uses
  ...

Never used (cold wall — 394 tiles):
  IMG_8810.jpg, IMG_8811.jpg, ... (see cold_wall.png for top 20)

Parameters: grid=120×68, λ=2.0, μ=0.5, τ=0.5, K=20
```

The "reason" for top uses is a heuristic: `describe_lab(tile.mean_lab)` → one of {`sky blue`, `skin tone`, `foliage green`, `warm brown`, `neutral gray`, `shadow`, …}. Thresholds are approximate; this is for comic effect, not accuracy.

## 7. DeepZoom Export

Library: `deepzoom` (pure python, pip-installable, no brew).

Produces `image.dzi` + `image_files/<level>/<col>_<row>.jpg`. We bundle an `index.html` with the OpenSeadragon CDN script pointing at the DZI. The entire `out/<stem>/dzi/` folder is self-contained and shareable by copying or zipping.

## 8. Widgets (ipywidgets)

In Cell 2 (one layout box):

| Widget | Type | Bound to |
|---|---|---|
| target path | `Text` | config |
| pool dir | `Text` | config |
| grid_w | `IntSlider(40–240, step 4, default 120)` | recompute grid |
| tile_px | `IntSlider(8–48, default 16)` | recompute output size |
| λ | `FloatSlider(0–10, step 0.1, default 2.0)` | match only |
| μ | `FloatSlider(0–5, step 0.1, default 0.5)` | match only |
| τ | `FloatSlider(0–1, step 0.05, default 0.5)` | render only |
| Regenerate | `Button` | runs cells 5–7 |
| Export DZI | `Button` | runs cell 8 |
| log | `Output` | print sink |
| preview | `Image` | final PNG bytes |

## 9. Error Handling

Only handle what will actually happen. No defensive theater.

| Condition | Behavior |
|---|---|
| Pool contains corrupted/unreadable image | skip + warning, continue |
| Pool dir doesn't exist | `FileNotFoundError`, let it propagate |
| Target image doesn't exist | same |
| Target aspect ≠ grid aspect | center-crop target to grid aspect |
| Cache hash mismatch (new files in pool) | auto re-scan |
| Widget slider out of range | not possible, range-bounded |

## 10. Testing

Single pytest smoke test in `tests/test_smoke.py`:

1. Fixture: 20 tiny solid-color images + a 10×10-pixel target, all in `fixtures/`
2. Run full pipeline with grid=5×5, tile_px=4
3. Assert: `out/smoke/mosaic.png` exists, is 20×20 px, is not all-black, `report.txt` contains "Tiles placed: 25"

No coverage goal. No CI. If it runs and the assertions pass, it ships.

## 11. CHANGELOG Discipline

Per the user's agent-oriented convention (stored in memory). Every substantive change → a YAML entry with `date / type / target / change / rationale / action / result / validation / status`. try-failed entries preserved; 50-entry / 6-month compression threshold.

Initial entry on first commit: `type: feat, target: entire project, change: scaffold tier-B photomosaic toy, status: experimental`.

## 12. Out of Scope (explicitly NOT doing)

- CLIP / semantic matching (user vetoed torch)
- Gradio / Streamlit UI (ipywidgets enough per "不要太复杂")
- Cursed / WeChat / TimeCapsule preset modes (tier C, later)
- Pool tagging + attribution narrative (tier C, later)
- Multi-target batching
- GPU / MPS / parallel render
- Upload to remote
- Pretty printing / progress bars beyond `tqdm`
- Retry logic anywhere

## 13. Dependencies

```
numpy
pillow
scikit-image
scipy
tqdm
ipywidgets
matplotlib
deepzoom
```

Zero: torch, transformers, open_clip, faiss, gradio, streamlit, rembg.

## 14. Assumptions (call out if wrong)

- Mac is M-series or Intel — both fine without torch (pure numpy path)
- Pool size ≤ ~30k images (cKDTree handles comfortably; cache pickle stays < 100MB)
- User will supply real target + pool paths at Cell 2 time; fixtures are only for smoke test
- User is OK with `%pip install` inside the notebook (vs. requiring a venv setup) — matches "toy" stance

---

## Appendix A — Self-review log (2026-04-17)

- ✅ No TBDs or placeholders
- ✅ Defaults for every parameter
- ✅ Error handling decisions explicit (§9)
- ✅ Tests scoped appropriately for a toy (§10)
- ✅ Out-of-scope list prevents scope creep (§12)
- ✅ Assumptions flagged (§14)
- ✅ Algorithm math written out (§4) so the implementation plan is unambiguous
- ⚠ Scan order in §4.3 creates top-left asymmetry in neighbor penalty — accepted as toy-acceptable; not worth a second pass
- ⚠ `describe_lab` heuristic in §6 is hand-wavy — intentional, it's a punchline generator, not analytics
