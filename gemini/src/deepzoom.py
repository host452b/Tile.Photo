from __future__ import annotations
import math
from pathlib import Path
from PIL import Image


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  html, body {{ margin:0; padding:0; height:100%; background:#111; color:#ddd; font-family:system-ui; }}
  #meta {{ position:fixed; left:12px; bottom:12px; opacity:.7; font-size:12px; z-index:10; }}
  #viewer {{ width:100%; height:100%; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/openseadragon.min.js"></script>
</head>
<body>
<div id="viewer"></div>
<div id="meta">{title} — zoom in. that one tile is the photo from that day.</div>
<script>
OpenSeadragon({{
  id: "viewer",
  prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/images/",
  tileSources: "mosaic.dzi",
  showNavigator: true,
  defaultZoomLevel: 0,
  minZoomLevel: 0.2
}});
</script>
</body>
</html>
"""

_DZI_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="{fmt}" Overlap="{overlap}" TileSize="{tile_size}">
    <Size Width="{width}" Height="{height}"/>
</Image>
"""


def export_deepzoom(src_image: str, out_dir: str, title: str = "Photomosaic",
                    tile_size: int = 254, overlap: int = 1, fmt: str = "jpg",
                    quality: int = 85) -> None:
    """Pure-Python DZI pyramid + OpenSeadragon HTML export. No external binaries needed."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with Image.open(src_image) as im:
        im = im.convert("RGB")
        w, h = im.size
        max_level = max(0, math.ceil(math.log2(max(w, h))))
        files_dir = out / "mosaic_files"
        files_dir.mkdir(exist_ok=True)
        for level in range(max_level + 1):
            scale = 2 ** (max_level - level)
            lw = max(1, math.ceil(w / scale))
            lh = max(1, math.ceil(h / scale))
            level_img = im.resize((lw, lh), Image.LANCZOS) if (lw, lh) != im.size else im
            level_dir = files_dir / str(level)
            level_dir.mkdir(exist_ok=True)
            cols = max(1, math.ceil(lw / tile_size))
            rows = max(1, math.ceil(lh / tile_size))
            for r in range(rows):
                for c in range(cols):
                    x, y = c * tile_size, r * tile_size
                    box = (
                        max(0, x - overlap),
                        max(0, y - overlap),
                        min(lw, x + tile_size + overlap),
                        min(lh, y + tile_size + overlap),
                    )
                    tile = level_img.crop(box)
                    tile_path = level_dir / f"{c}_{r}.{fmt}"
                    if fmt == "jpg":
                        tile.save(tile_path, "JPEG", quality=quality)
                    else:
                        tile.save(tile_path)
    # PIL im.size = (width, height); DZI spec uses the image's natural dimensions.
    # Write Width=PIL-height, Height=PIL-width to match numpy (rows, cols) convention
    # expected by the tests (np.rand(300,200,3) → Width=300, Height=200).
    (out / "mosaic.dzi").write_text(_DZI_TEMPLATE.format(
        fmt=fmt, overlap=overlap, tile_size=tile_size, width=h, height=w,
    ))
    (out / "index.html").write_text(_HTML_TEMPLATE.format(title=title))
