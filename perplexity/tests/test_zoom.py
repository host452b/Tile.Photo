import numpy as np
from PIL import Image

from mosaic.zoom import export_deepzoom


def test_deepzoom_produces_html_and_dzi(tmp_path):
    img = Image.fromarray(np.random.RandomState(0).randint(0, 255, (512, 512, 3), dtype=np.uint8))
    png_path = tmp_path / "mosaic.png"
    img.save(png_path)

    out_dir = tmp_path / "zoom"
    result = export_deepzoom(png_path, out_dir)

    assert result["html"].exists()
    assert result["dzi"].exists()
    html = result["html"].read_text()
    assert "openseadragon" in html.lower()
    assert result["dzi"].name in html

    files_dir = result["dzi"].parent / (result["dzi"].stem + "_files")
    assert files_dir.exists()
    tiles = list(files_dir.rglob("*.jpg"))
    assert len(tiles) > 0, "no DZI tiles written"
    dzi_xml = result["dzi"].read_text()
    assert "TileSize" in dzi_xml and 'Width="512"' in dzi_xml


def test_deepzoom_overwrites_existing(tmp_path):
    img = Image.fromarray(np.random.RandomState(0).randint(0, 255, (512, 512, 3), dtype=np.uint8))
    png_path = tmp_path / "mosaic.png"
    img.save(png_path)
    out_dir = tmp_path / "zoom"

    export_deepzoom(png_path, out_dir)
    result = export_deepzoom(png_path, out_dir)
    assert result["html"].exists()
