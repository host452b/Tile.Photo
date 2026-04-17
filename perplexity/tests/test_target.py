import numpy as np

from mosaic.target import load_and_grid


def test_grid_produces_cols_x_rows_cells(tmp_target_img):
    grid = load_and_grid(tmp_target_img, grid_cols=4, grid_rows=4)
    assert grid["shape"] == (4, 4)
    assert len(grid["cells"]) == 16


def test_each_cell_has_lab_mean_and_variance(tmp_target_img):
    grid = load_and_grid(tmp_target_img, grid_cols=4, grid_rows=4)
    for cell in grid["cells"]:
        assert cell["lab_mean"].shape == (3,)
        assert np.isscalar(cell["variance"]) or cell["variance"].shape == ()
        assert "row" in cell and "col" in cell


def test_quadrants_have_distinct_colors(tmp_target_img):
    grid = load_and_grid(tmp_target_img, grid_cols=2, grid_rows=2)
    means = [c["lab_mean"] for c in grid["cells"]]
    # four cells, all pairs should differ
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.allclose(means[i], means[j], atol=1.0)


def test_center_crops_non_matching_aspect(tmp_path):
    # 320x256 target, grid 2x2 (asks for 1:1) → should center-crop to 256x256
    from PIL import Image
    arr = np.zeros((256, 320, 3), dtype=np.uint8)
    arr[:, :160] = (255, 0, 0)
    arr[:, 160:] = (0, 255, 0)
    path = tmp_path / "wide.png"
    Image.fromarray(arr).save(path)

    grid = load_and_grid(path, grid_cols=2, grid_rows=2)
    # After center crop to 256x256, split into 2x2, left and right halves different
    cells = {(c["row"], c["col"]): c["lab_mean"] for c in grid["cells"]}
    assert not np.allclose(cells[(0, 0)], cells[(0, 1)], atol=1.0)
