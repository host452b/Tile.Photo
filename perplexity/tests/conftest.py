import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tmp_pool_dir(tmp_path):
    """16 张纯色 tile（4x4 颜色网格）写到临时目录。"""
    pool = tmp_path / "pool"
    pool.mkdir()
    for i in range(4):
        for j in range(4):
            color = (i * 64, j * 64, ((i + j) * 32) % 256)
            img = Image.new("RGB", (64, 64), color)
            img.save(pool / f"tile_{i}_{j}.jpg")
    return pool


@pytest.fixture
def tmp_target_img(tmp_path):
    """4x4 四象限纯色目标图（每象限 64x64）。"""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:128, :128] = (255, 0, 0)
    arr[:128, 128:] = (0, 255, 0)
    arr[128:, :128] = (0, 0, 255)
    arr[128:, 128:] = (255, 255, 0)
    path = tmp_path / "target.png"
    Image.fromarray(arr).save(path)
    return path
