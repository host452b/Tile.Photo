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


from src.matcher import assign_with_penalties


def test_repeat_penalty_spreads_usage():
    # 4 cells all want tile 0 (identical LAB), 3 tiles available
    tile_lab = np.array([[50, 0, 0], [50, 5, 5], [50, -5, -5]], dtype=np.float32)
    patches_lab = np.full((2, 2, 3), [50, 0, 0], dtype=np.float32)
    idx, dist = color_topk(patches_lab, tile_lab, k=3)
    # With no penalty, every cell picks tile 0
    no_pen = assign_with_penalties(idx, dist, lambda_repeat=0.0, mu_neighbor=0.0)
    assert (no_pen == 0).all()
    # With heavy repeat penalty, at least one other tile must appear
    with_pen = assign_with_penalties(idx, dist, lambda_repeat=100.0, mu_neighbor=0.0)
    assert len(set(with_pen.ravel().tolist())) > 1


def test_neighbor_penalty_breaks_adjacency():
    # 2 tiles: both roughly equal distance to all cells
    tile_lab = np.array([[50, 10, 0], [50, 0, 10]], dtype=np.float32)
    patches_lab = np.full((2, 4, 3), [50, 5, 5], dtype=np.float32)
    idx, dist = color_topk(patches_lab, tile_lab, k=2)
    with_pen = assign_with_penalties(idx, dist, lambda_repeat=0.0, mu_neighbor=1000.0)
    # No two horizontally adjacent cells should share the same tile
    for r in range(with_pen.shape[0]):
        for c in range(with_pen.shape[1] - 1):
            assert with_pen[r, c] != with_pen[r, c + 1]


def test_assign_emits_callback_per_cell():
    tile_lab = np.array([[50, 0, 0], [50, 30, 30]], dtype=np.float32)
    patches_lab = np.full((1, 2, 3), [50, 15, 15], dtype=np.float32)
    idx, dist = color_topk(patches_lab, tile_lab, k=2)
    events = []
    assign_with_penalties(idx, dist, lambda_repeat=0.1, mu_neighbor=0.0,
                          on_cell=lambda r, c, chosen, candidates: events.append((r, c, chosen)))
    assert len(events) == 2
    assert [e[:2] for e in events] == [(0, 0), (0, 1)]


def test_clip_rerank_prefers_semantically_similar_when_tie():
    # Two tiles equally close in color, but tile 1 semantically closer
    tile_lab = np.array([[50, 0, 0], [50, 0, 0]], dtype=np.float32)
    tile_clip = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    patches_lab = np.array([[[50, 0, 0]]], dtype=np.float32)   # (1,1,3)
    patch_clip = np.array([[[0.0, 1.0]]], dtype=np.float32)     # (1,1,2) — matches tile 1
    idx, dist = color_topk(patches_lab, tile_lab, k=2)
    from src.matcher import assign_with_clip
    out = assign_with_clip(idx, dist, patches_lab, tile_lab,
                           tile_clip=tile_clip, patch_clip=patch_clip,
                           lambda_repeat=0.0, mu_neighbor=0.0, clip_weight=1.0)
    assert out[0, 0] == 1


def test_clip_rerank_noop_when_weight_zero():
    tile_lab = np.array([[50, 0, 0], [50, 0, 0]], dtype=np.float32)
    tile_clip = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    patches_lab = np.array([[[50, 0, 0]]], dtype=np.float32)
    patch_clip = np.array([[[0.0, 1.0]]], dtype=np.float32)
    idx, dist = color_topk(patches_lab, tile_lab, k=2)
    from src.matcher import assign_with_clip
    out_zero = assign_with_clip(idx, dist, patches_lab, tile_lab,
                                tile_clip=tile_clip, patch_clip=patch_clip,
                                lambda_repeat=0.0, mu_neighbor=0.0, clip_weight=0.0)
    out_plain = assign_with_penalties(idx, dist, lambda_repeat=0.0, mu_neighbor=0.0)
    np.testing.assert_array_equal(out_zero, out_plain)
