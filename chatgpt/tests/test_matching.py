import numpy as np
import pytest

from mosaic_core import build_faiss_index, knn_candidates


def test_build_faiss_index_returns_top_k_deterministic():
    """Same query must give identical top-5 on repeated calls."""
    rng = np.random.default_rng(42)
    tile_labs = rng.uniform(0, 100, size=(100, 3)).astype(np.float32)
    index = build_faiss_index(tile_labs)
    query = np.array([[50.0, 0.0, 0.0]], dtype=np.float32)
    d1, i1 = index.search(query, 5)
    d2, i2 = index.search(query, 5)
    np.testing.assert_array_equal(i1, i2)
    # Top hit must be the closest tile by L2.
    dists = np.linalg.norm(tile_labs - query, axis=1)
    assert i1[0, 0] == int(np.argmin(dists))


def test_knn_candidates_shape_and_legal_indices():
    """knn_candidates returns (H*W, k) indices all in [0, N)."""
    rng = np.random.default_rng(0)
    tile_labs = rng.uniform(0, 100, size=(50, 3)).astype(np.float32)
    target_lab = rng.uniform(0, 100, size=(4, 6, 3)).astype(np.float32)
    index = build_faiss_index(tile_labs)
    result = knn_candidates(target_lab, index, k=8)
    assert result.shape == (4 * 6, 8)
    assert result.min() >= 0
    assert result.max() < 50


def test_rerank_lambda_zero_mu_zero_is_argmin_delta_e():
    """With both penalties off, rerank returns the candidate with smallest ΔE."""
    from mosaic_core import rerank
    tile_labs = np.array(
        [[50.0, 0.0, 0.0], [50.0, 20.0, 0.0], [50.0, 40.0, 0.0]],
        dtype=np.float32,
    )
    target = np.array([50.0, 35.0, 0.0], dtype=np.float32)  # closest to index 2
    best = rerank(
        candidate_idxs=np.array([0, 1, 2], dtype=np.int64),
        tile_labs=tile_labs,
        target_lab_patch=target,
        usage_counts={},
        neighbor_tile_idxs=[],
        lambda_repeat=0.0,
        mu_neighbor=0.0,
    )
    assert best == 2


def test_rerank_lambda_penalizes_heavy_usage():
    """Heavy λ shifts choice away from an over-used tile toward an unused one."""
    from mosaic_core import rerank
    tile_labs = np.array(
        [[50.0, 0.0, 0.0], [50.0, 5.0, 0.0]],  # tile 0 slightly closer than tile 1
        dtype=np.float32,
    )
    target = np.array([50.0, 0.0, 0.0], dtype=np.float32)
    best_cold = rerank(
        candidate_idxs=np.array([0, 1], dtype=np.int64),
        tile_labs=tile_labs,
        target_lab_patch=target,
        usage_counts={},
        neighbor_tile_idxs=[],
        lambda_repeat=0.0,
        mu_neighbor=0.0,
    )
    assert best_cold == 0
    best_hot = rerank(
        candidate_idxs=np.array([0, 1], dtype=np.int64),
        tile_labs=tile_labs,
        target_lab_patch=target,
        usage_counts={0: 1000},
        neighbor_tile_idxs=[],
        lambda_repeat=100.0,
        mu_neighbor=0.0,
    )
    assert best_hot == 1


def test_rerank_mu_penalizes_similar_neighbor():
    """Heavy μ shifts choice away from tile that looks like the left/up neighbor."""
    from mosaic_core import rerank
    tile_labs = np.array(
        [[50.0, 0.0, 0.0], [50.0, 30.0, 0.0], [50.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    target = np.array([50.0, 0.0, 0.0], dtype=np.float32)
    best_no_neighbor = rerank(
        candidate_idxs=np.array([0, 1], dtype=np.int64),
        tile_labs=tile_labs,
        target_lab_patch=target,
        usage_counts={},
        neighbor_tile_idxs=[],
        lambda_repeat=0.0,
        mu_neighbor=0.0,
    )
    assert best_no_neighbor == 0
    best_with_clone_neighbor = rerank(
        candidate_idxs=np.array([0, 1], dtype=np.int64),
        tile_labs=tile_labs,
        target_lab_patch=target,
        usage_counts={},
        neighbor_tile_idxs=[2],
        lambda_repeat=0.0,
        mu_neighbor=100.0,
    )
    assert best_with_clone_neighbor == 1


def test_rerank_empty_candidates_raises():
    """rerank with no candidates must raise, not silently return -1."""
    from mosaic_core import rerank
    tile_labs = np.array([[50.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        rerank(
            candidate_idxs=np.array([], dtype=np.int64),
            tile_labs=tile_labs,
            target_lab_patch=np.array([50.0, 0.0, 0.0], dtype=np.float32),
            usage_counts={},
            neighbor_tile_idxs=[],
            lambda_repeat=0.0,
            mu_neighbor=0.0,
        )
