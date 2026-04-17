import numpy as np

from mosaic.match import solve_assignment


def _synthetic_setup():
    """造 4 种纯色 tile + 4 格目标（每格颜色对应一种 tile）。"""
    pool = {}
    for i, color_lab in enumerate([
        np.array([50, 80, 60], dtype=np.float32),   # red-ish
        np.array([90, -80, 80], dtype=np.float32),  # green-ish
        np.array([30, 70, -100], dtype=np.float32), # blue-ish
        np.array([95, -20, 90], dtype=np.float32),  # yellow-ish
    ]):
        pool[f"tile_{i}.jpg"] = {
            "mtime": 0.0,
            "lab_mean": color_lab,
            "thumbnail": np.full((16, 16, 3), i * 50, dtype=np.uint8),
        }

    cells = [
        {"row": 0, "col": 0, "lab_mean": np.array([50, 80, 60], dtype=np.float32), "variance": 0.0},
        {"row": 0, "col": 1, "lab_mean": np.array([90, -80, 80], dtype=np.float32), "variance": 0.0},
        {"row": 1, "col": 0, "lab_mean": np.array([30, 70, -100], dtype=np.float32), "variance": 0.0},
        {"row": 1, "col": 1, "lab_mean": np.array([95, -20, 90], dtype=np.float32), "variance": 0.0},
    ]
    return pool, cells


def test_no_penalty_picks_nearest_color():
    pool, cells = _synthetic_setup()
    assignment = solve_assignment(
        pool, cells, grid_shape=(2, 2),
        lambda_reuse=0.0, mu_neighbor=0.0, topk=4,
    )
    assert assignment[(0, 0)] == "tile_0.jpg"
    assert assignment[(0, 1)] == "tile_1.jpg"
    assert assignment[(1, 0)] == "tile_2.jpg"
    assert assignment[(1, 1)] == "tile_3.jpg"


def test_high_lambda_forces_unique_tiles():
    pool, _ = _synthetic_setup()
    # 4 cells all want tile_0 color, but λ should spread
    same_color = np.array([50, 80, 60], dtype=np.float32)
    cells = [
        {"row": 0, "col": 0, "lab_mean": same_color, "variance": 0.0},
        {"row": 0, "col": 1, "lab_mean": same_color, "variance": 0.0},
        {"row": 1, "col": 0, "lab_mean": same_color, "variance": 0.0},
        {"row": 1, "col": 1, "lab_mean": same_color, "variance": 0.0},
    ]
    assignment = solve_assignment(
        pool, cells, grid_shape=(2, 2),
        lambda_reuse=1e6, mu_neighbor=0.0, topk=4,
    )
    used = list(assignment.values())
    assert len(set(used)) == 4


def test_zero_lambda_allows_repetition():
    pool, _ = _synthetic_setup()
    same_color = np.array([50, 80, 60], dtype=np.float32)
    cells = [
        {"row": 0, "col": 0, "lab_mean": same_color, "variance": 0.0},
        {"row": 0, "col": 1, "lab_mean": same_color, "variance": 0.0},
    ]
    assignment = solve_assignment(
        pool, cells, grid_shape=(1, 2),
        lambda_reuse=0.0, mu_neighbor=0.0, topk=4,
    )
    # both should pick tile_0 (nearest color), no diversity pressure
    assert assignment[(0, 0)] == "tile_0.jpg"
    assert assignment[(0, 1)] == "tile_0.jpg"


def test_variance_ordering_picks_rare_color_first():
    """高方差格子先选，rare color 不会被平凡格抢走。"""
    # 2 tiles only: rare_color (1 copy) + common_color (easy match for both)
    rare = np.array([50, 100, 0], dtype=np.float32)
    common = np.array([50, 0, 0], dtype=np.float32)
    pool = {
        "rare.jpg": {"mtime": 0.0, "lab_mean": rare, "thumbnail": np.zeros((16,16,3), np.uint8)},
        "common.jpg": {"mtime": 0.0, "lab_mean": common, "thumbnail": np.zeros((16,16,3), np.uint8)},
    }
    # two cells: one exactly wants rare (variance high), one wants common (variance low)
    cells = [
        {"row": 0, "col": 0, "lab_mean": common, "variance": 0.1},
        {"row": 0, "col": 1, "lab_mean": rare, "variance": 99.0},
    ]
    assignment = solve_assignment(
        pool, cells, grid_shape=(1, 2),
        lambda_reuse=1e6, mu_neighbor=0.0, topk=2,
    )
    # high λ + variance-first means (0,1) claims rare.jpg first, (0,0) left with common
    assert assignment[(0, 1)] == "rare.jpg"
    assert assignment[(0, 0)] == "common.jpg"
