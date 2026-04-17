import numpy as np

from src import tone


def _solid(rgb: tuple[int, int, int], size: int = 8) -> np.ndarray:
    return np.full((size, size, 3), rgb, dtype=np.uint8)


def test_strength_zero_returns_source_unchanged():
    src = _solid((120, 80, 40))
    tgt = _solid((20, 200, 200))
    out = tone.reinhard_transfer(src, tgt, strength=0.0)
    np.testing.assert_array_equal(out, src)


def test_strength_one_moves_source_mean_toward_target_mean():
    from skimage.color import rgb2lab

    src = _solid((120, 80, 40))
    tgt = _solid((20, 200, 200))
    out = tone.reinhard_transfer(src, tgt, strength=1.0)

    out_lab = rgb2lab(out.astype(np.float32) / 255.0).mean(axis=(0, 1))
    tgt_lab = rgb2lab(tgt.astype(np.float32) / 255.0).mean(axis=(0, 1))
    diff = np.linalg.norm(out_lab - tgt_lab)
    assert diff < 5.0, f"expected ΔE < 5, got {diff}"


def test_strength_half_is_between_source_and_target():
    from skimage.color import rgb2lab

    src = _solid((120, 80, 40))
    tgt = _solid((20, 200, 200))
    src_lab = rgb2lab(src.astype(np.float32) / 255.0).mean(axis=(0, 1))
    tgt_lab = rgb2lab(tgt.astype(np.float32) / 255.0).mean(axis=(0, 1))

    out = tone.reinhard_transfer(src, tgt, strength=0.5)
    out_lab = rgb2lab(out.astype(np.float32) / 255.0).mean(axis=(0, 1))

    to_src = np.linalg.norm(out_lab - src_lab)
    to_tgt = np.linalg.norm(out_lab - tgt_lab)
    full = np.linalg.norm(tgt_lab - src_lab)
    assert 0.2 * full < to_src < 0.8 * full
    assert 0.2 * full < to_tgt < 0.8 * full


def test_output_shape_and_dtype_preserved():
    src = np.zeros((12, 16, 3), dtype=np.uint8)
    tgt = np.full((12, 16, 3), 200, dtype=np.uint8)
    out = tone.reinhard_transfer(src, tgt, strength=0.5)
    assert out.shape == (12, 16, 3)
    assert out.dtype == np.uint8
