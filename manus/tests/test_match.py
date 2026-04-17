import numpy as np

from mosaic.match import build_index, match_grid, topk_candidates


def _fake_tiles_lab():
    return np.array(
        [
            [50.0, 0.0, 0.0],
            [53.0, 80.0, 67.0],
            [32.0, 79.0, -108.0],
            [88.0, -79.0, 81.0],
            [97.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


def test_topk_returns_nearest_first():
    labs = _fake_tiles_lab()
    idx = build_index(labs)
    query = np.array([53.0, 80.0, 67.0], dtype=np.float32)
    cand = topk_candidates(idx, query, k=3)
    assert cand[0] == 1
    assert len(cand) == 3


def test_topk_k_clamps_to_pool_size():
    labs = _fake_tiles_lab()
    idx = build_index(labs)
    cand = topk_candidates(idx, labs[0], k=100)
    assert len(cand) == 5


def test_match_grid_without_penalties_picks_nearest():
    labs = np.array(
        [
            [50.0, 0.0, 0.0],
            [53.0, 80.0, 67.0],
            [32.0, 79.0, -108.0],
            [88.0, -79.0, 81.0],
        ],
        dtype=np.float32,
    )
    cells = labs.reshape(2, 2, 3)
    idx = build_index(labs)
    grid = match_grid(idx, labs, cells, k=4, lambda_=0.0, mu=0.0)
    assert grid.shape == (2, 2)
    assert grid[0, 0] == 0
    assert grid[0, 1] == 1
    assert grid[1, 0] == 2
    assert grid[1, 1] == 3


def test_match_grid_diversity_forces_spread():
    labs = np.array(
        [[50.0, 0.0, 0.0], [50.01, 0.0, 0.0]],
        dtype=np.float32,
    )
    cells = np.array([[[50.0, 0.0, 0.0], [50.0, 0.0, 0.0]]], dtype=np.float32)
    idx = build_index(labs)
    grid = match_grid(idx, labs, cells, k=2, lambda_=10.0, mu=0.0)
    assert set(grid.ravel()) == {0, 1}


def test_match_grid_shape_matches_input():
    labs = np.random.default_rng(0).uniform(-50, 50, size=(30, 3)).astype(np.float32)
    cells = np.random.default_rng(1).uniform(-50, 50, size=(4, 6, 3)).astype(np.float32)
    idx = build_index(labs)
    grid = match_grid(idx, labs, cells, k=5, lambda_=1.0, mu=0.5)
    assert grid.shape == (4, 6)
    assert grid.dtype == np.int64
