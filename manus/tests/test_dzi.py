import numpy as np
from PIL import Image

from mosaic.dzi import export_dzi


def test_export_dzi_produces_html_and_dzi(tmp_path):
    rgb = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(rgb)
    out_dir = tmp_path / "dzi"
    html = export_dzi(img, out_dir)
    assert html.exists()
    assert (out_dir / "image.dzi").exists()
    assert (out_dir / "image_files").is_dir()
    assert "image.dzi" in html.read_text()
    assert "openseadragon" in html.read_text().lower()


def test_export_dzi_has_tile_at_every_level(tmp_path):
    rgb = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(rgb)
    out_dir = tmp_path / "dzi"
    export_dzi(img, out_dir, tile_size=32, overlap=1)
    tiles = out_dir / "image_files"
    # 64 -> max_level=6; each level should have at least one tile 0_0.jpg
    for level in range(7):
        assert (tiles / str(level) / "0_0.jpg").exists(), f"missing level {level}"


def test_export_dzi_descriptor_has_size(tmp_path):
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    out_dir = tmp_path / "dzi"
    export_dzi(img, out_dir)
    descriptor = (out_dir / "image.dzi").read_text()
    assert 'Width="200"' in descriptor
    assert 'Height="100"' in descriptor
