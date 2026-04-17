from pathlib import Path
import numpy as np
from PIL import Image
from src.deepzoom import export_deepzoom


def test_export_deepzoom_writes_files(tmp_path: Path):
    src = tmp_path / "big.png"
    Image.fromarray((np.random.rand(512, 512, 3) * 255).astype(np.uint8)).save(src)
    out_dir = tmp_path / "out"
    export_deepzoom(str(src), str(out_dir), title="Test")
    # Required outputs
    assert (out_dir / "index.html").exists()
    # DZI descriptor
    assert (out_dir / "mosaic.dzi").exists()
    # Tile pyramid folder
    tiles_root = out_dir / "mosaic_files"
    assert tiles_root.is_dir()
    tiles = list(tiles_root.rglob("*.jpg")) + list(tiles_root.rglob("*.jpeg")) + list(tiles_root.rglob("*.png"))
    assert len(tiles) > 0
    # HTML wiring
    html = (out_dir / "index.html").read_text()
    assert "openseadragon" in html.lower()
    assert ".dzi" in html
    assert "Test" in html


def test_export_deepzoom_pyramid_has_multiple_levels(tmp_path: Path):
    src = tmp_path / "mid.png"
    Image.fromarray((np.random.rand(1024, 768, 3) * 255).astype(np.uint8)).save(src)
    out_dir = tmp_path / "out"
    export_deepzoom(str(src), str(out_dir))
    tiles_root = out_dir / "mosaic_files"
    levels = sorted(int(d.name) for d in tiles_root.iterdir() if d.is_dir())
    # 1024 → log2 = 10, so levels 0..10 should exist
    assert levels[0] == 0
    assert levels[-1] == 10
    # Top level (max) must contain at least one tile
    assert any((tiles_root / str(levels[-1])).rglob("*.jpg"))


def test_export_deepzoom_dzi_xml_is_valid(tmp_path: Path):
    src = tmp_path / "tiny.png"
    Image.fromarray((np.random.rand(300, 200, 3) * 255).astype(np.uint8)).save(src)
    out_dir = tmp_path / "out"
    export_deepzoom(str(src), str(out_dir), tile_size=128, overlap=2)
    dzi = (out_dir / "mosaic.dzi").read_text()
    # Basic XML sanity
    assert 'TileSize="128"' in dzi
    assert 'Overlap="2"' in dzi
    assert 'Format="jpg"' in dzi
    assert 'Width="300"' in dzi
    assert 'Height="200"' in dzi
