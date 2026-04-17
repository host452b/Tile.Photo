import numpy as np
import pytest
from PIL import Image

from mosaic_core import split_target


def test_split_target_shape_and_dtype():
    """split_target(img, grid_w=10, grid_h=5) must return ndarray shape (5, 10, 3) float32."""
    img = Image.new("RGB", (100, 50), (128, 128, 128))  # 100 px wide, 50 px tall
    lab_grid = split_target(img, grid_w=10, grid_h=5)
    assert lab_grid.shape == (5, 10, 3)
    assert lab_grid.dtype == np.float32


def test_split_target_uniform_gray_constant_lab():
    """Uniform gray input -> all cells share the same LAB value (L~54, a~0, b~0)."""
    img = Image.new("RGB", (80, 40), (128, 128, 128))
    lab_grid = split_target(img, grid_w=8, grid_h=4)
    L_plane = lab_grid[..., 0]
    # All cells equal within float noise.
    assert L_plane.max() - L_plane.min() < 0.01
    # L ~= 54 for sRGB 128 gray.
    assert 52 < L_plane.mean() < 56
