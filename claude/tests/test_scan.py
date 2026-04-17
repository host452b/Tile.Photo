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
