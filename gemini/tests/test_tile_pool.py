from pathlib import Path
from PIL import Image
import pytest
from src.tile_pool import scan_tile_dir


def _make_img(path: Path, color: tuple[int, int, int], size=(64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_scan_finds_jpg_png_recursively(tmp_path: Path):
    _make_img(tmp_path / "a.jpg", (255, 0, 0))
    _make_img(tmp_path / "sub" / "b.png", (0, 255, 0))
    _make_img(tmp_path / "sub" / "c.jpeg", (0, 0, 255))
    (tmp_path / "readme.txt").write_text("not an image")
    (tmp_path / "broken.jpg").write_bytes(b"not a real jpeg")

    paths = scan_tile_dir(str(tmp_path))
    assert len(paths) == 3
    assert all(p.endswith((".jpg", ".jpeg", ".png")) for p in paths)


def test_scan_skips_too_small(tmp_path: Path):
    _make_img(tmp_path / "ok.jpg", (100, 100, 100), size=(64, 64))
    _make_img(tmp_path / "tiny.jpg", (100, 100, 100), size=(8, 8))
    paths = scan_tile_dir(str(tmp_path), min_side=32)
    assert len(paths) == 1
    assert paths[0].endswith("ok.jpg")
