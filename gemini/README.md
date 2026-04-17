# Photomosaic Toy

Re-compose a target image from your personal photo library. Local-only, deliberately slow, deliberately explainable.

## Install

```bash
pip install -r requirements.txt
# Optional, only if you want CLIP semantic matching (~2 GB):
# pip install -r requirements-clip.txt
```

No system dependencies required. DeepZoom export uses a pure-Python DZI generator (Pillow only).

## First run

1. Put a target image at `samples/target.jpg` (1080p works well).
2. Put 500–5000 photos under `samples/tiles/` (any subfolder layout; `.jpg`/`.png`).
3. Generate the notebook and launch Jupyter:

```bash
jupytext --to ipynb photomosaic.py
jupyter notebook photomosaic.ipynb
```

4. Run all cells. First run takes longer (caches tile-pool LAB means); subsequent runs are fast.

For a no-setup smoke test that synthesizes a tiny tile pool + target:

```bash
pytest tests/test_smoke.py -v
```

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
| `use_clip` | Semantic reranking — blue regions prefer real sea photos | You want "this is weirdly correct" moments (requires `requirements-clip.txt`) |

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
- `src/deepzoom.py` — Pure-Python DZI pyramid + OpenSeadragon HTML

## Run tests

```bash
pytest -v
```

31 unit + 1 smoke test; 1 CLIP test is skipped unless `requirements-clip.txt` is installed.

## Plan / changelog

- Implementation plan: `docs/superpowers/plans/2026-04-17-photomosaic-toy.md`
- Change history: `CHANGELOG.md` (agent-oriented; verbose by design)
