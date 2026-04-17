import numpy as np

from src import match


def test_match_picks_nearest_lab():
    pool_lab = np.array(
        [
            [50.0, 0.0, 0.0],
            [50.0, 60.0, 40.0],
            [50.0, -40.0, 50.0],
            [50.0, 20.0, -60.0],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [[50.0, 0.0, 0.0], [50.0, 60.0, 40.0]],
            [[50.0, -40.0, 50.0], [50.0, 20.0, -60.0]],
        ],
        dtype=np.float32,
    )
    idx = match.match_grid(target, pool_lab)
    assert idx.shape == (2, 2)
    assert idx.dtype == np.int32
    np.testing.assert_array_equal(idx, np.array([[0, 1], [2, 3]], dtype=np.int32))


def test_match_picks_closest_when_no_exact():
    pool_lab = np.array(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    target = np.array([[[30.0, 0.0, 0.0]]], dtype=np.float32)
    idx = match.match_grid(target, pool_lab)
    assert idx[0, 0] == 0


def test_match_raises_on_empty_pool():
    import pytest

    pool_lab = np.zeros((0, 3), dtype=np.float32)
    target = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
    with pytest.raises(ValueError):
        match.match_grid(target, pool_lab)
