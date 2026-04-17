from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Tuple

Mode = Literal["normal", "cursed", "time_capsule"]


@dataclass
class PhotomosaicConfig:
    target: str
    tile_dir: str
    grid: Tuple[int, int] = (120, 68)
    tile_px: int = 16
    lambda_repeat: float = 0.3
    mu_neighbor: float = 0.5
    tau_tone: float = 0.5
    use_clip: bool = False
    clip_weight: float = 0.15
    mode: Mode = "normal"
    cache_dir: str = ".cache"
    output_dir: str = "output"
    topk_color: int = 32
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not (0.0 <= self.tau_tone <= 1.0):
            raise ValueError(f"tau_tone must be in [0, 1], got {self.tau_tone}")
        if self.grid[0] <= 0 or self.grid[1] <= 0:
            raise ValueError(f"grid must have positive dims, got {self.grid}")
        if self.mode not in ("normal", "cursed", "time_capsule"):
            raise ValueError(f"mode must be one of normal/cursed/time_capsule, got {self.mode!r}")
        if self.tile_px <= 0:
            raise ValueError(f"tile_px must be positive, got {self.tile_px}")
        if self.topk_color <= 0:
            raise ValueError(f"topk_color must be positive, got {self.topk_color}")
