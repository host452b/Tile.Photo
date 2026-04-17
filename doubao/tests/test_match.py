"""Tests for src/mosaic/match.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def two_tone_target(tmp_path: Path) -> Path:
    """100×100 image: left half red (255,0,0), right half blue (0,0,255)."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :50] = [255, 0, 0]
    arr[:, 50:] = [0, 0, 255]
    p = tmp_path / "target.png"
    Image.fromarray(arr).save(p)
    return p


def test_split_target_shape_and_lab(two_tone_target: Path) -> None:
    from mosaic.match import split_target

    cells = split_target(two_tone_target, grid=(4, 2))  # 4 cols × 2 rows
    assert cells.shape == (2, 4, 3)  # (rows, cols, LAB)
    # Left two columns are red, right two columns are blue.
    # In CIE LAB: red has +b (~67), blue has strongly -b (~-108); both have +a,
    # so b is the clean separator.
    left_b = cells[:, :2, 2].mean()
    right_b = cells[:, 2:, 2].mean()
    assert left_b > 20  # red has positive b
    assert right_b < -20  # blue has strongly negative b
