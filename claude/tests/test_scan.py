from pathlib import Path

import numpy as np
from PIL import Image

from src import scan


def test_demo_mode_returns_500_tiles(tmp_path):
    pool = scan.build_pool(
        base_dir=tmp_path / "nonexistent",
        cache_dir=tmp_path / "cache",
        tile_px=24,
        demo_mode=True,
    )
    assert pool.lab.shape == (500, 3)
    assert pool.lab.dtype == np.float32
    assert len(pool.thumbs_paths) == 500
    assert len(pool.source_paths) == 500
    for p in pool.thumbs_paths[:5]:
        img = Image.open(p)
        assert img.size == (24, 24)


def test_demo_mode_covers_hue_spectrum(tmp_path):
    pool = scan.build_pool(
        base_dir=tmp_path / "nonexistent",
        cache_dir=tmp_path / "cache",
        tile_px=24,
        demo_mode=True,
    )
    a_channel = pool.lab[:, 1]
    b_channel = pool.lab[:, 2]
    assert a_channel.max() - a_channel.min() > 50
    assert b_channel.max() - b_channel.min() > 50


def _write_solid_jpg(path: Path, rgb: tuple[int, int, int], size: int = 128) -> None:
    arr = np.full((size, size, 3), rgb, dtype=np.uint8)
    Image.fromarray(arr).save(path, quality=95)


def test_scans_real_files(tmp_path):
    base = tmp_path / "photos"
    base.mkdir()
    _write_solid_jpg(base / "red.jpg", (230, 20, 20))
    _write_solid_jpg(base / "green.jpg", (20, 200, 20))
    _write_solid_jpg(base / "blue.jpg", (20, 20, 220))

    pool = scan.build_pool(
        base_dir=base,
        cache_dir=tmp_path / "cache",
        tile_px=24,
        demo_mode=False,
    )
    assert pool.lab.shape == (3, 3)
    assert len(pool.source_paths) == 3
    assert all(Path(p).exists() for p in pool.thumbs_paths)
    for p in pool.thumbs_paths:
        assert Image.open(p).size == (24, 24)


def test_skips_corrupt_files(tmp_path, caplog):
    base = tmp_path / "photos"
    base.mkdir()
    _write_solid_jpg(base / "good.jpg", (100, 100, 100))
    (base / "bad.jpg").write_bytes(b"not a jpeg")

    pool = scan.build_pool(
        base_dir=base,
        cache_dir=tmp_path / "cache",
        tile_px=24,
        demo_mode=False,
    )
    assert pool.lab.shape == (1, 3)
