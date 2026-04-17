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


def test_lambda_zero_matches_phase1_broadcast():
    rng = np.random.default_rng(0)
    pool = rng.uniform(0, 100, size=(30, 3)).astype(np.float32)
    target = rng.uniform(0, 100, size=(5, 7, 3)).astype(np.float32)

    old = match.match_grid(target, pool)
    new = match.match_grid(target, pool, lambda_=0.0, mu=0.0)
    np.testing.assert_array_equal(old, new)


def test_lambda_reduces_max_usage():
    pool = np.array(
        [
            [50.0, 0.0, 0.0],
            [50.0, 10.0, 10.0],
            [50.0, -10.0, -10.0],
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(1)
    target_ab = rng.uniform(-15, 15, size=(10, 10, 2)).astype(np.float32)
    target_l = np.full((10, 10, 1), 50.0, dtype=np.float32)
    target = np.concatenate([target_l, target_ab], axis=-1)

    idx0 = match.match_grid(target, pool, lambda_=0.0)
    idx_penalty = match.match_grid(target, pool, lambda_=50.0)

    max0 = np.bincount(idx0.ravel(), minlength=3).max()
    maxp = np.bincount(idx_penalty.ravel(), minlength=3).max()
    assert maxp < max0, f"expected λ=50 to flatten usage; max0={max0}, maxp={maxp}"
    usage_p = np.bincount(idx_penalty.ravel(), minlength=3)
    assert (usage_p >= 1).all(), f"expected all 3 tiles used; got {usage_p}"


def test_mu_penalizes_adjacent_duplicates():
    pool = np.array(
        [
            [50.0, 0.0, 0.0],
            [50.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )
    target = np.full((6, 6, 3), [50.0, 2.5, 2.5], dtype=np.float32)

    idx = match.match_grid(target, pool, lambda_=0.0, mu=500.0)
    for r in range(6):
        for c in range(6):
            if r > 0:
                assert idx[r, c] != idx[r - 1, c], f"top neighbor match at ({r},{c})"
            if c > 0:
                assert idx[r, c] != idx[r, c - 1], f"left neighbor match at ({r},{c})"


def test_mu_zero_does_not_affect_result():
    rng = np.random.default_rng(2)
    pool = rng.uniform(0, 100, size=(10, 3)).astype(np.float32)
    target = rng.uniform(0, 100, size=(4, 5, 3)).astype(np.float32)

    no_mu = match.match_grid(target, pool, lambda_=3.0, mu=0.0)
    with_zero = match.match_grid(target, pool, lambda_=3.0, mu=0.0)
    np.testing.assert_array_equal(no_mu, with_zero)
