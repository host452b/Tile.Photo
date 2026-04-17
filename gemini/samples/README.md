# Sample data directory

Put your target image at `samples/target.jpg` and your tile pool under `samples/tiles/` before running the notebook.

For a self-contained demo with no setup, run `pytest tests/test_smoke.py -v` — it synthesizes a tile pool and target in a tmp dir and exercises the full pipeline.

Tile-pool organization is free-form (recursive glob). Optional: add a `tags.json` next to the tiles to power the "23% from 2019 Japan trip" narrative in the final report. See the main README for the tag format.
