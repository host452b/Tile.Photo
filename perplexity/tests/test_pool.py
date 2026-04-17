import pickle
import time

import numpy as np
import pytest
from PIL import Image

from mosaic.pool import scan_pool, load_cache


def test_scan_returns_entry_per_image(tmp_pool_dir, tmp_path):
    cache_path = tmp_path / "pool.pkl"
    features = scan_pool(tmp_pool_dir, cache_path)
    assert len(features) == 16
    for path, entry in features.items():
        assert "lab_mean" in entry
        assert "thumbnail" in entry
        assert "mtime" in entry
        assert entry["lab_mean"].shape == (3,)
        assert entry["thumbnail"].shape == (16, 16, 3)


def test_scan_caches_to_pickle(tmp_pool_dir, tmp_path):
    cache_path = tmp_path / "pool.pkl"
    scan_pool(tmp_pool_dir, cache_path)
    assert cache_path.exists()
    with open(cache_path, "rb") as f:
        loaded = pickle.load(f)
    assert len(loaded) == 16


def test_scan_is_incremental_on_mtime(tmp_pool_dir, tmp_path):
    cache_path = tmp_path / "pool.pkl"
    first = scan_pool(tmp_pool_dir, cache_path)
    first_lab = next(iter(first.values()))["lab_mean"].copy()

    target = next(iter(tmp_pool_dir.glob("*.jpg")))
    time.sleep(0.01)
    Image.new("RGB", (64, 64), (10, 20, 30)).save(target)

    second = scan_pool(tmp_pool_dir, cache_path)
    changed = second[str(target)]["lab_mean"]
    assert not np.allclose(changed, first_lab)


def test_scan_drops_deleted_files(tmp_pool_dir, tmp_path):
    cache_path = tmp_path / "pool.pkl"
    scan_pool(tmp_pool_dir, cache_path)

    victim = next(iter(tmp_pool_dir.glob("*.jpg")))
    victim.unlink()

    second = scan_pool(tmp_pool_dir, cache_path)
    assert str(victim) not in second
    assert len(second) == 15


def test_scan_skips_non_images(tmp_pool_dir, tmp_path):
    (tmp_pool_dir / "notes.txt").write_text("not an image")
    (tmp_pool_dir / "broken.jpg").write_bytes(b"not really a jpg")
    cache_path = tmp_path / "pool.pkl"
    features = scan_pool(tmp_pool_dir, cache_path)
    assert len(features) == 16  # 16 real, 2 skipped


def test_load_cache_returns_empty_when_missing(tmp_path):
    features = load_cache(tmp_path / "nonexistent.pkl")
    assert features == {}
