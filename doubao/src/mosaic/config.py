"""Single source of truth for all mosaic parameters."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MosaicConfig:
    # paths
    tile_source_dir: Path
    target_image: Path
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))
    output_dir: Path = field(default_factory=lambda: Path("output"))

    # grid
    grid: tuple[int, int] = (120, 68)   # (cols, rows)
    tile_px: int = 16                    # 每 tile 渲染像素

    # matching
    candidate_k: int = 50       # 先取 top-k 颜色候选
    lambda_reuse: float = 0.3   # 重复惩罚
    mu_neighbor: float = 0.2    # 邻居相似度惩罚

    # rendering
    tau_tone: float = 0.5       # 色调迁移强度 0..1

    # behavior
    verbose: bool = True
    mode: str = "classic"       # 预留给 cursed_* 模式

    def validate(self) -> None:
        """Call before running the pipeline; raise on misconfiguration."""
        if not self.tile_source_dir.exists():
            raise ValueError(
                f"tile_source_dir does not exist: {self.tile_source_dir}. "
                "Please set it in cell 2."
            )
        if not self.target_image.exists():
            raise ValueError(
                f"target_image does not exist: {self.target_image}. "
                "Please set it in cell 2."
            )
        if not 0.0 <= self.tau_tone <= 1.0:
            raise ValueError(f"tau_tone must be in [0, 1], got {self.tau_tone}")
        if self.tile_px < 4:
            raise ValueError(f"tile_px too small: {self.tile_px}")
        cols, rows = self.grid
        if cols < 1 or rows < 1:
            raise ValueError(f"grid must be positive, got {self.grid}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
