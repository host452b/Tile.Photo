import pytest
from src.config import PhotomosaicConfig


def test_config_defaults_are_sane():
    cfg = PhotomosaicConfig(target="target.jpg", tile_dir="tiles/")
    assert cfg.grid == (120, 68)
    assert cfg.tile_px == 16
    assert cfg.lambda_repeat == 0.3
    assert cfg.mu_neighbor == 0.5
    assert 0.0 <= cfg.tau_tone <= 1.0
    assert cfg.use_clip is False
    assert cfg.mode == "normal"


def test_config_rejects_tau_out_of_range():
    with pytest.raises(ValueError, match="tau_tone"):
        PhotomosaicConfig(target="t.jpg", tile_dir="tiles/", tau_tone=1.5)


def test_config_rejects_nonpositive_grid():
    with pytest.raises(ValueError, match="grid"):
        PhotomosaicConfig(target="t.jpg", tile_dir="tiles/", grid=(0, 10))


def test_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        PhotomosaicConfig(target="t.jpg", tile_dir="tiles/", mode="bogus")
