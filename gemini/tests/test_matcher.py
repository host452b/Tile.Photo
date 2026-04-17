import numpy as np
from src.matcher import color_topk


def test_color_topk_picks_nearest():
    tile_lab = np.array([
        [50.0, 0.0, 0.0],      # neutral gray
        [50.0, 80.0, 60.0],    # vivid red
        [50.0, -80.0, -60.0],  # vivid blue-green
    ], dtype=np.float32)
    # Query = one cell that's close to the red tile
    patches_lab = np.array([[[50.0, 78.0, 58.0]]], dtype=np.float32)  # (1, 1, 3)
    idx, dist = color_topk(patches_lab, tile_lab, k=2)
    assert idx.shape == (1, 1, 2)
    assert dist.shape == (1, 1, 2)
    assert idx[0, 0, 0] == 1  # red tile wins
    # Distance to index 1 must be smaller than to the runner-up
    assert dist[0, 0, 0] < dist[0, 0, 1]


def test_color_topk_k_capped_to_tile_count():
    tile_lab = np.random.randn(5, 3).astype(np.float32) * 30 + 50
    patches_lab = np.random.randn(2, 3, 3).astype(np.float32) * 30 + 50
    idx, _ = color_topk(patches_lab, tile_lab, k=10)
    assert idx.shape == (2, 3, 5)  # capped to N=5
