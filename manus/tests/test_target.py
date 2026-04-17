from mosaic.target import load_and_slice


def test_load_and_slice_shape(fixtures_dir):
    grid = load_and_slice(fixtures_dir / "target.png", grid_w=10, grid_h=5)
    assert grid.lab_means.shape == (5, 10, 3)
    assert grid.canvas.shape[2] == 3


def test_load_and_slice_left_is_red_right_is_blue(fixtures_dir):
    grid = load_and_slice(fixtures_dir / "target.png", grid_w=10, grid_h=5)
    left = grid.lab_means[:, :5].mean(axis=(0, 1))
    right = grid.lab_means[:, 5:].mean(axis=(0, 1))
    # red has large +a*, blue has very -b*
    assert left[1] > 50
    assert right[2] < -50
