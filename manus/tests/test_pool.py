import pickle

import numpy as np
from PIL import Image

from mosaic.pool import lab_mean, scan_pool


def test_lab_mean_pure_red(fixtures_dir):
    img = np.array(Image.open(fixtures_dir / "red.png").convert("RGB"))
    lab = lab_mean(img)
    assert lab.shape == (3,)
    assert 50 < lab[0] < 56
    assert 75 < lab[1] < 85
    assert 60 < lab[2] < 72


def test_lab_mean_pure_blue(fixtures_dir):
    img = np.array(Image.open(fixtures_dir / "blue.png").convert("RGB"))
    lab = lab_mean(img)
    assert lab.shape == (3,)
    assert 28 < lab[0] < 36
    assert -115 < lab[2] < -100


def test_scan_pool_reads_images(tmp_path, fixtures_dir):
    pool = tmp_path / "pool"
    pool.mkdir()
    for name in ("red.png", "blue.png"):
        (pool / name).write_bytes((fixtures_dir / name).read_bytes())

    cache_dir = tmp_path / "cache"
    tiles = scan_pool(pool, cache_dir, thumb_px=8)

    assert len(tiles) == 2
    paths = {t.path.name for t in tiles}
    assert paths == {"red.png", "blue.png"}
    for t in tiles:
        assert t.thumb.shape == (8, 8, 3)
        assert t.lab.shape == (3,)


def test_scan_pool_uses_cache(tmp_path, fixtures_dir):
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "red.png").write_bytes((fixtures_dir / "red.png").read_bytes())
    cache_dir = tmp_path / "cache"

    scan_pool(pool, cache_dir, thumb_px=8)
    cache_files = list(cache_dir.glob("*.pkl"))
    assert len(cache_files) == 1

    with cache_files[0].open("rb") as f:
        data = pickle.load(f)
    assert len(data["tiles"]) == 1


def test_scan_pool_invalidates_on_new_file(tmp_path, fixtures_dir):
    pool = tmp_path / "pool"
    pool.mkdir()
    cache_dir = tmp_path / "cache"
    (pool / "red.png").write_bytes((fixtures_dir / "red.png").read_bytes())
    first = scan_pool(pool, cache_dir, thumb_px=8)
    assert len(first) == 1

    (pool / "blue.png").write_bytes((fixtures_dir / "blue.png").read_bytes())
    second = scan_pool(pool, cache_dir, thumb_px=8)
    assert len(second) == 2
