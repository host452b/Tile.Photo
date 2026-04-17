from mosaic.render import render_mosaic


def test_render_outputs_correct_dimensions(tmp_pool_dir, tmp_target_img):
    from mosaic.pool import scan_pool
    from mosaic.target import load_and_grid

    pool = scan_pool(tmp_pool_dir, tmp_target_img.parent / "cache.pkl")
    grid = load_and_grid(tmp_target_img, grid_cols=4, grid_rows=4)

    first_tile = next(iter(pool.keys()))
    assignment = {(c["row"], c["col"]): first_tile for c in grid["cells"]}

    result = render_mosaic(assignment, pool, grid, tile_px=32, tau=0.0)

    assert result["image"].size == (128, 128)  # 4 cols * 32 px
    assert result["usage"][first_tile] == 16
    for path in pool:
        if path != first_tile:
            assert result["usage"][path] == 0


def test_render_applies_tau_transfer(tmp_pool_dir, tmp_target_img, monkeypatch):
    from mosaic.pool import scan_pool
    from mosaic.target import load_and_grid

    calls = []

    import mosaic.render
    original = mosaic.render.reinhard_transfer

    def spy(tile, patch, tau):
        calls.append(tau)
        return original(tile, patch, tau)

    monkeypatch.setattr(mosaic.render, "reinhard_transfer", spy)

    pool = scan_pool(tmp_pool_dir, tmp_target_img.parent / "cache.pkl")
    grid = load_and_grid(tmp_target_img, grid_cols=2, grid_rows=2)
    first_tile = next(iter(pool.keys()))
    assignment = {(c["row"], c["col"]): first_tile for c in grid["cells"]}

    render_mosaic(assignment, pool, grid, tile_px=32, tau=0.75)

    assert len(calls) == 4
    assert all(c == 0.75 for c in calls)
