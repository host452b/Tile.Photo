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
