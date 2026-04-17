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
