"""DeepZoom 金字塔切片 + 自包含 OpenSeadragon HTML 查看器。纯 Pillow 实现。"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

from PIL import Image

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Photomosaic</title>
<style>html,body{{margin:0;padding:0;background:#111;}}#viewer{{width:100vw;height:100vh;}}</style>
<script src="https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/openseadragon.min.js"></script>
</head>
<body>
<div id="viewer"></div>
<script>
OpenSeadragon({{
  id: "viewer",
  tileSources: "{dzi_name}",
  prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/images/",
  showNavigator: true,
  animationTime: 0.5,
  maxZoomPixelRatio: 8
}});
</script>
</body>
</html>
"""

DZI_TEMPLATE = '''<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="{fmt}" Overlap="{overlap}" TileSize="{tile_size}">
  <Size Width="{w}" Height="{h}"/>
</Image>'''


def _write_level_tiles(level_img: Image.Image, level_dir: Path, tile_size: int, overlap: int, fmt: str, quality: int) -> None:
    lw, lh = level_img.size
    cols = max(1, math.ceil(lw / tile_size))
    rows = max(1, math.ceil(lh / tile_size))
    for col in range(cols):
        for row in range(rows):
            left = col * tile_size - (overlap if col > 0 else 0)
            top = row * tile_size - (overlap if row > 0 else 0)
            right = min(col * tile_size + tile_size + overlap, lw)
            bottom = min(row * tile_size + tile_size + overlap, lh)
            tile = level_img.crop((left, top, right, bottom))
            tile.save(level_dir / f"{col}_{row}.{fmt}", quality=quality)


def export_deepzoom(
    png_path: Path,
    output_dir: Path,
    tile_size: int = 254,
    overlap: int = 1,
    fmt: str = "jpg",
    quality: int = 90,
) -> dict:
    png_path = Path(png_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    base_name = "mosaic"
    img = Image.open(png_path).convert("RGB")
    w, h = img.size
    max_level = int(math.ceil(math.log2(max(w, h))))

    files_dir = output_dir / f"{base_name}_files"
    files_dir.mkdir()

    for level in range(max_level + 1):
        scale = 2 ** (max_level - level)
        lw = max(1, math.ceil(w / scale))
        lh = max(1, math.ceil(h / scale))
        level_img = img.resize((lw, lh), Image.LANCZOS)
        level_dir = files_dir / str(level)
        level_dir.mkdir()
        _write_level_tiles(level_img, level_dir, tile_size, overlap, fmt, quality)

    dzi_path = output_dir / f"{base_name}.dzi"
    dzi_path.write_text(
        DZI_TEMPLATE.format(fmt=fmt, overlap=overlap, tile_size=tile_size, w=w, h=h),
        encoding="utf-8",
    )

    html_path = output_dir / "index.html"
    html_path.write_text(HTML_TEMPLATE.format(dzi_name=dzi_path.name), encoding="utf-8")

    return {"html": html_path, "dzi": dzi_path}
