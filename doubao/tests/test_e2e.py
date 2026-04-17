"""End-to-end smoke: synthetic 100-tile pool → 16×9 target → verify all outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def test_full_pipeline_on_synthetic_data(tmp_path: Path) -> None:
    from mosaic.config import MosaicConfig
    from mosaic.deepzoom import export_deepzoom
    from mosaic.match import match_all_tiles, split_target
    from mosaic.render import render_mosaic
    from mosaic.report import build_cold_wall, generate_text_report
    from mosaic.tiles import load_or_build

    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(42)
    # 100 random-color tiles
    for i in range(100):
        color = rng.integers(0, 255, 3, dtype=np.uint8)
        Image.fromarray(np.full((32, 32, 3), color, dtype=np.uint8)).save(
            src / f"t{i:03d}.png"
        )

    target_path = tmp_path / "target.png"
    # Gradient target
    grad = np.zeros((128, 128, 3), dtype=np.uint8)
    grad[..., 0] = np.linspace(0, 255, 128, dtype=np.uint8)[None, :]
    grad[..., 2] = np.linspace(0, 255, 128, dtype=np.uint8)[:, None]
    Image.fromarray(grad).save(target_path)

    cfg = MosaicConfig(
        tile_source_dir=src,
        target_image=target_path,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        grid=(16, 9),
        tile_px=16,
        candidate_k=20,
        lambda_reuse=0.3,
        mu_neighbor=0.2,
        tau_tone=0.4,
        verbose=False,
    )
    cfg.validate()

    pool = load_or_build(cfg.tile_source_dir, cfg.tile_px, cfg.cache_dir)
    assert len(pool) == 100

    target_cells = split_target(cfg.target_image, cfg.grid)
    assert target_cells.shape == (9, 16, 3)

    assignment, use_count = match_all_tiles(target_cells, pool, cfg)
    assert assignment.shape == (9, 16)
    assert (assignment >= 0).all()
    assert (assignment < 100).all()
    assert sum(use_count.values()) == 9 * 16

    mosaic = render_mosaic(assignment, pool, target_cells, cfg)
    assert mosaic.size == (16 * 16, 9 * 16)

    out_png = cfg.output_dir / "mosaic.png"
    mosaic.save(out_png)
    assert out_png.exists()

    report_text = generate_text_report(use_count, pool, total_cells=9 * 16)
    assert "本次使用了" in report_text
    assert "TOP 5" in report_text

    wall = build_cold_wall(pool, use_count)
    assert wall.size[0] >= 1

    html = export_deepzoom(mosaic, cfg.output_dir)
    assert html.exists()
    assert html.name == "index.html"
    assert (cfg.output_dir / "deepzoom" / "mosaic.dzi").exists()
    assert (cfg.output_dir / "deepzoom" / "mosaic_files").is_dir()


def test_reuse_cache_on_second_build(tmp_path: Path) -> None:
    """Second load_or_build on the same dir should hit the cache without rebuilding."""
    from mosaic.tiles import load_or_build

    src = tmp_path / "src"
    src.mkdir()
    for i in range(5):
        Image.fromarray(np.full((32, 32, 3), i * 50, dtype=np.uint8)).save(
            src / f"t{i}.png"
        )

    cache = tmp_path / "cache"
    pool1 = load_or_build(src, tile_px=8, cache_dir=cache)
    pool2 = load_or_build(src, tile_px=8, cache_dir=cache)
    np.testing.assert_array_equal(pool1.lab_means, pool2.lab_means)
    np.testing.assert_array_equal(pool1.thumbnails, pool2.thumbnails)
    assert (cache / "tiles_8.pkl").exists()
