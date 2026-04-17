from mosaic.config import DEFAULT_CONFIG, build_widgets


def test_default_config_has_expected_keys():
    required = {
        "target_path", "pool_dir", "grid_cols", "grid_rows", "tile_px",
        "preview_grid_cols", "preview_grid_rows", "preview_tile_px",
        "lambda_reuse", "mu_neighbor", "tau_transfer",
        "topk_candidates", "neighbor_sigma", "cache_dir", "output_dir", "seed",
    }
    assert required.issubset(DEFAULT_CONFIG.keys())


def test_default_grid_is_16_9_ish():
    assert DEFAULT_CONFIG["grid_cols"] == 120
    assert DEFAULT_CONFIG["grid_rows"] == 68


def test_build_widgets_returns_container_and_values():
    result = build_widgets()
    assert "container" in result
    assert "get_params" in result
    params = result["get_params"]()
    assert set(params.keys()) == {"lambda_reuse", "mu_neighbor", "tau_transfer"}
    assert params["lambda_reuse"] == DEFAULT_CONFIG["lambda_reuse"]
