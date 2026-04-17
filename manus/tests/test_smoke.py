import numpy as np
from PIL import Image

from mosaic.dzi import export_dzi
from mosaic.match import build_index, match_grid
from mosaic.pool import scan_pool
from mosaic.render import compose
from mosaic.report import cold_wall, text_report
from mosaic.target import load_and_slice


def test_full_pipeline(tmp_path, fixtures_dir):
    pool_dir = fixtures_dir / "smoke_pool"
    target = fixtures_dir / "target.png"

    tiles = scan_pool(pool_dir, tmp_path / "cache", thumb_px=8)
    assert len(tiles) == 20

    grid = load_and_slice(target, grid_w=5, grid_h=3)
    assert grid.lab_means.shape == (3, 5, 3)

    tile_labs = np.stack([t.lab for t in tiles])
    idx = build_index(tile_labs)
    choices = match_grid(idx, tile_labs, grid.lab_means, k=5, lambda_=1.0, mu=0.3)
    assert choices.shape == (3, 5)

    mosaic = compose(tiles, choices, grid.lab_means, tile_px=4, tau=0.5)
    assert mosaic.shape == (12, 20, 3)
    assert mosaic.dtype == np.uint8
    assert mosaic.sum() > 0

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    Image.fromarray(mosaic).save(out_dir / "mosaic.png")

    uses = np.bincount(choices.ravel(), minlength=len(tiles))
    report = text_report(
        tiles,
        uses,
        grid_shape=choices.shape,
        params={"lambda_": 1.0, "mu": 0.3, "tau": 0.5},
    )
    (out_dir / "report.txt").write_text(report)
    assert "Tiles placed: 15" in report

    wall = cold_wall(tiles, uses, n=5, thumb_px=8, cols=5)
    Image.fromarray(wall).save(out_dir / "cold_wall.png")

    html = export_dzi(Image.fromarray(mosaic), out_dir / "dzi")
    assert html.exists()
