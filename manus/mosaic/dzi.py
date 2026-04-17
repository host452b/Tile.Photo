"""DeepZoom (DZI) pyramid export, pure PIL.

Implements a subset of the DZI spec sufficient for OpenSeadragon:
  - `image.dzi`  — XML descriptor with final image size
  - `image_files/<level>/<col>_<row>.<ext>` — one tile per (col, row) per level

Level 0 is a 1x1 thumbnail; max_level is ceil(log2(max(W, H))) and contains the
full-resolution image. Each intermediate level is the image resized to
(W >> (max_level - level)) x (H >> (max_level - level)).
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image


_HTML = """<!doctype html>
<meta charset="utf-8">
<title>Tile.Photo mosaic</title>
<style>html,body{margin:0;padding:0;height:100%;background:#111;color:#ccc;font:13px sans-serif}#v{width:100%;height:100vh}</style>
<div id="v"></div>
<script src="https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/openseadragon.min.js"></script>
<script>
OpenSeadragon({
  id: "v",
  prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/images/",
  tileSources: "image.dzi"
});
</script>
"""


def _write_dzi_descriptor(out_dir: Path, width: int, height: int, tile_size: int, overlap: int, fmt: str) -> None:
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"\n'
        f'       Format="{fmt}"\n'
        f'       Overlap="{overlap}"\n'
        f'       TileSize="{tile_size}">\n'
        f'  <Size Width="{width}" Height="{height}"/>\n'
        "</Image>\n"
    )
    (out_dir / "image.dzi").write_text(xml, encoding="utf-8")


def _emit_level(img: Image.Image, level_dir: Path, tile_size: int, overlap: int, fmt: str) -> None:
    level_dir.mkdir(parents=True, exist_ok=True)
    w, h = img.size
    cols = math.ceil(w / tile_size)
    rows = math.ceil(h / tile_size)
    ext = "jpg" if fmt == "jpg" else fmt
    save_kwargs = {"quality": 90} if fmt == "jpg" else {}
    for c in range(cols):
        for r in range(rows):
            x0 = max(c * tile_size - overlap, 0)
            y0 = max(r * tile_size - overlap, 0)
            x1 = min((c + 1) * tile_size + overlap, w)
            y1 = min((r + 1) * tile_size + overlap, h)
            tile = img.crop((x0, y0, x1, y1))
            if fmt == "jpg":
                tile = tile.convert("RGB")
            tile.save(level_dir / f"{c}_{r}.{ext}", **save_kwargs)


def export_dzi(image: Image.Image, out_dir: Path, tile_size: int = 254, overlap: int = 1, fmt: str = "jpg") -> Path:
    """Write a DZI pyramid + index.html into out_dir. Returns path to index.html."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w, h = image.size
    max_level = int(math.ceil(math.log2(max(w, h, 1))))
    tiles_dir = out_dir / "image_files"
    tiles_dir.mkdir(exist_ok=True)

    for level in range(max_level + 1):
        scale = 2 ** (max_level - level)
        lw = max(1, (w + scale - 1) // scale)
        lh = max(1, (h + scale - 1) // scale)
        level_img = image if (lw, lh) == (w, h) else image.resize((lw, lh), Image.LANCZOS)
        _emit_level(level_img, tiles_dir / str(level), tile_size, overlap, fmt)

    _write_dzi_descriptor(out_dir, w, h, tile_size, overlap, fmt)

    html = out_dir / "index.html"
    html.write_text(_HTML, encoding="utf-8")
    return html
