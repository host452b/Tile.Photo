from mosaic.pipeline import run_pipeline


def test_preview_pipeline_produces_image(tmp_pool_dir, tmp_target_img, tmp_path):
    result = run_pipeline(
        target_path=tmp_target_img,
        pool_dir=tmp_pool_dir,
        grid_cols=4, grid_rows=4,
        tile_px=16,
        lambda_reuse=1.0, mu_neighbor=0.5, tau_transfer=0.5,
        cache_path=tmp_path / "cache.pkl",
        output_dir=tmp_path / "out",
        do_deepzoom=False,
        do_report=False,
    )
    assert result["image"].size == (64, 64)
    assert sum(result["usage"].values()) == 16


def test_full_pipeline_produces_all_artifacts(tmp_pool_dir, tmp_target_img, tmp_path):
    out_dir = tmp_path / "out"
    result = run_pipeline(
        target_path=tmp_target_img,
        pool_dir=tmp_pool_dir,
        grid_cols=4, grid_rows=4,
        tile_px=16,
        lambda_reuse=1.0, mu_neighbor=0.5, tau_transfer=0.5,
        cache_path=tmp_path / "cache.pkl",
        output_dir=out_dir,
        do_deepzoom=True,
        do_report=True,
    )
    assert result["png_path"].exists()
    assert result["report_path"].exists()
    assert result["chart_path"].exists()
    assert result["cold_path"].exists()
    assert result["deepzoom_html"].exists()
