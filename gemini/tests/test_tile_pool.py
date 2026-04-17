from pathlib import Path
from PIL import Image
import pytest
import numpy as np
from src.tile_pool import scan_tile_dir, build_tile_index, load_or_build_index


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


def test_build_tile_index_computes_lab_mean(tmp_path: Path):
    _make_img(tmp_path / "red.jpg", (200, 30, 30))
    _make_img(tmp_path / "blue.jpg", (30, 30, 200))
    idx = build_tile_index(str(tmp_path))
    assert set(idx.paths) == {str(tmp_path / "red.jpg"), str(tmp_path / "blue.jpg")}
    assert idx.lab_mean.shape == (2, 3)
    # red's a* should be > blue's a*; blue's b* should be < red's b*
    red_i = idx.paths.index(str(tmp_path / "red.jpg"))
    blue_i = idx.paths.index(str(tmp_path / "blue.jpg"))
    assert idx.lab_mean[red_i, 1] > idx.lab_mean[blue_i, 1]
    assert idx.lab_mean[red_i, 2] > idx.lab_mean[blue_i, 2]


def test_load_or_build_uses_cache_on_second_call(tmp_path: Path):
    _make_img(tmp_path / "a.jpg", (100, 100, 100))
    cache = tmp_path / "_cache"
    idx1 = load_or_build_index(str(tmp_path), cache_dir=str(cache))
    # Delete source, second call must still work from cache
    (tmp_path / "a.jpg").unlink()
    idx2 = load_or_build_index(str(tmp_path), cache_dir=str(cache))
    assert idx1.paths == idx2.paths
    np.testing.assert_array_equal(idx1.lab_mean, idx2.lab_mean)
