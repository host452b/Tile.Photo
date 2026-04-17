"""Split target + match tiles to cells."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def split_target(target_path: Path, grid: tuple[int, int]) -> np.ndarray:
    """Load target, resize to (cols, rows) via block averaging, return LAB cells.

    Returns array of shape (rows, cols, 3) with LAB mean per cell.
    """
    cols, rows = grid
    with Image.open(target_path) as im:
        im = im.convert("RGB")
        # Resize so each pixel is one cell's mean; PIL LANCZOS approximates mean
        im = im.resize((cols, rows), Image.BOX)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    lab = rgb2lab(arr)  # (rows, cols, 3)
    return lab.astype(np.float32)
