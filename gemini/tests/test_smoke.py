"""End-to-end smoke: generate tiny synthetic target + tile pool, run full pipeline, check artifacts."""
from pathlib import Path
import numpy as np
from PIL import Image


def test_full_pipeline_produces_outputs(tmp_path: Path):
    # 1) Build tiny tile pool: 30 solid-color tiles
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(30):
        color = tuple(int(x) for x in rng.integers(0, 256, size=3))
        Image.new("RGB", (64, 64), color).save(tile_dir / f"t{i:03d}.jpg")

    # 2) Build a small target (gradient)
    xs = np.linspace(0, 255, 96, dtype=np.uint8)
    ys = np.linspace(0, 255, 56, dtype=np.uint8)
    tgt = np.stack([
        np.tile(xs, (56, 1)),
        np.tile(ys[:, None], (1, 96)),
        np.full((56, 96), 128, dtype=np.uint8),
    ], axis=-1)
    target_path = tmp_path / "target.png"
    Image.fromarray(tgt).save(target_path)

    out_dir = tmp_path / "output"

    # 3) Run pipeline
    from src.config import PhotomosaicConfig
    from src.tile_pool import load_or_build_index, load_tags
    from src.target import split_into_patches
    from src.matcher import color_topk, assign_with_penalties
    from src.renderer import render_mosaic
    from src.reporter import build_text_report, save_usage_plot, save_cold_wall

    cfg = PhotomosaicConfig(
        target=str(target_path),
        tile_dir=str(tile_dir),
        grid=(48, 28),           # small grid
        tile_px=8,
        lambda_repeat=0.5,
        mu_neighbor=0.5,
        tau_tone=0.4,
        cache_dir=str(tmp_path / ".cache"),
        output_dir=str(out_dir),
        topk_color=8,
    )
    idx = load_or_build_index(cfg.tile_dir, cache_dir=cfg.cache_dir)
    tags = load_tags(cfg.tile_dir, idx.paths)
    patches_lab, cell_rgb = split_into_patches(cfg.target, grid=cfg.grid)
    topk_idx, topk_dist = color_topk(patches_lab, idx.lab_mean, k=cfg.topk_color)
    assignment = assign_with_penalties(topk_idx, topk_dist,
                                       lambda_repeat=cfg.lambda_repeat,
                                       mu_neighbor=cfg.mu_neighbor)
    out_dir.mkdir(parents=True, exist_ok=True)
    mosaic, usage = render_mosaic(assignment, idx.paths, cell_rgb,
                                  tile_px=cfg.tile_px, tau=cfg.tau_tone)
    mosaic_path = out_dir / "mosaic.png"
    mosaic.save(mosaic_path)

    total_cells = patches_lab.shape[0] * patches_lab.shape[1]
    report = build_text_report(idx.paths, usage, tags, total_cells)
    (out_dir / "report.txt").write_text(report)
    save_usage_plot(usage, str(out_dir / "usage.png"))
    save_cold_wall(idx.paths, usage, str(out_dir / "cold.png"))

    # 4) Assertions
    assert mosaic_path.exists() and mosaic_path.stat().st_size > 0
    assert mosaic.size == (48 * 8, 28 * 8)
    assert (out_dir / "report.txt").read_text().strip() != ""
    assert (out_dir / "usage.png").exists()
    assert (out_dir / "cold.png").exists()
    # Diversity penalty should spread usage
    assert len([n for n in usage.values() if n > 0]) >= 5
