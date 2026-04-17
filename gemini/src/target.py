from __future__ import annotations
from typing import Tuple
import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def split_into_patches(target_path: str, grid: Tuple[int, int], patch_px: int = 2
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      patches_lab: (rows, cols, 3) LAB mean per grid cell
      cell_rgb:    (rows, cols, patch_px, patch_px, 3) uint8 source RGB per cell
                   (used by tone-transfer to know what the cell 'should look like')
    grid is (cols, rows) to match image-native ordering.
    """
    cols, rows = grid
    target_w = cols * patch_px
    target_h = rows * patch_px
    with Image.open(target_path) as im:
        im = im.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.uint8)  # (target_h, target_w, 3)
    # Reshape into cells
    cell_rgb = arr.reshape(rows, patch_px, cols, patch_px, 3).transpose(0, 2, 1, 3, 4)
    # LAB mean per cell
    rgb_f = arr.astype(np.float32) / 255.0
    lab = rgb2lab(rgb_f)  # (target_h, target_w, 3)
    patches_lab = lab.reshape(rows, patch_px, cols, patch_px, 3).mean(axis=(1, 3))
    return patches_lab.astype(np.float32), cell_rgb
