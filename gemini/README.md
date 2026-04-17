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
