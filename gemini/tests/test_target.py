from pathlib import Path
import numpy as np
from PIL import Image
from src.target import split_into_patches


def test_split_returns_expected_shape(tmp_path: Path):
    img_path = tmp_path / "t.jpg"
    Image.new("RGB", (240, 136), (128, 128, 128)).save(img_path)
    patches_lab, cell_rgb = split_into_patches(str(img_path), grid=(120, 68))
    # grid is (cols, rows)
    assert patches_lab.shape == (68, 120, 3)  # (rows, cols, lab)
    assert cell_rgb.shape == (68, 120, 2, 2, 3)  # each patch 2x2 RGB


def test_split_handles_non_divisible_sizes(tmp_path: Path):
    img_path = tmp_path / "t.jpg"
    # 241 × 137 doesn't divide evenly by 120 × 68 — function must resize
    Image.new("RGB", (241, 137), (200, 50, 50)).save(img_path)
    patches_lab, _ = split_into_patches(str(img_path), grid=(120, 68))
    assert patches_lab.shape == (68, 120, 3)
    # mean of every cell's L*/a*/b* should reflect a strong red
    assert patches_lab[..., 1].mean() > 30  # a* positive = red
