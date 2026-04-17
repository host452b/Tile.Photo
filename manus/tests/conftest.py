from pathlib import Path

import numpy as np
import pytest
from PIL import Image


FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES


def _write_solid(path: Path, rgb: tuple[int, int, int], size: int = 8):
    arr = np.full((size, size, 3), rgb, dtype=np.uint8)
    Image.fromarray(arr).save(path)


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures():
    FIXTURES.mkdir(exist_ok=True)
    if not (FIXTURES / "red.png").exists():
        _write_solid(FIXTURES / "red.png", (255, 0, 0))
    if not (FIXTURES / "blue.png").exists():
        _write_solid(FIXTURES / "blue.png", (0, 0, 255))
    if not (FIXTURES / "target.png").exists():
        arr = np.zeros((90, 160, 3), dtype=np.uint8)
        arr[:, :80] = (255, 0, 0)
        arr[:, 80:] = (0, 0, 255)
        Image.fromarray(arr).save(FIXTURES / "target.png")

    smoke = FIXTURES / "smoke_pool"
    smoke.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(20):
        p = smoke / f"tile_{i:02d}.png"
        if p.exists():
            continue
        rgb = tuple(int(x) for x in rng.integers(0, 255, size=3))
        _write_solid(p, rgb, size=8)
