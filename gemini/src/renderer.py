from __future__ import annotations
import numpy as np
from skimage.color import rgb2lab, lab2rgb


def reinhard_tone_transfer(src_rgb: np.ndarray, target_rgb: np.ndarray, tau: float) -> np.ndarray:
    """
    Reinhard color transfer in LAB space, interpolated by tau ∈ [0, 1].
    tau=0 returns src unchanged; tau=1 shifts src's per-channel mean/std toward target's.
    Inputs are uint8 HxWx3; output uint8 HxWx3.
    """
    if tau <= 0.0:
        return src_rgb.copy()
    src_lab = rgb2lab(src_rgb.astype(np.float32) / 255.0)
    tgt_lab = rgb2lab(target_rgb.astype(np.float32) / 255.0)
    src_mean = src_lab.reshape(-1, 3).mean(axis=0)
    src_std = src_lab.reshape(-1, 3).std(axis=0) + 1e-6
    tgt_mean = tgt_lab.reshape(-1, 3).mean(axis=0)
    tgt_std = tgt_lab.reshape(-1, 3).std(axis=0) + 1e-6
    # Full Reinhard: shifted = (src - src_mean) * (tgt_std/src_std) + tgt_mean
    shifted = (src_lab - src_mean) * (tgt_std / src_std) + tgt_mean
    blended = (1.0 - tau) * src_lab + tau * shifted
    out_rgb = np.clip(lab2rgb(blended), 0.0, 1.0)
    return (out_rgb * 255.0 + 0.5).astype(np.uint8)
