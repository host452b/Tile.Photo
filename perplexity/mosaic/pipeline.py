"""端到端 pipeline: scan pool → grid target → solve → render → (report + deepzoom)。"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

from mosaic.match import solve_assignment
from mosaic.pool import scan_pool
from mosaic.render import render_mosaic
from mosaic.report import generate_report
from mosaic.target import load_and_grid
from mosaic.zoom import export_deepzoom


def run_pipeline(
    target_path: Path,
    pool_dir: Path,
    grid_cols: int,
    grid_rows: int,
    tile_px: int,
    lambda_reuse: float,
    mu_neighbor: float,
    tau_transfer: float,
    cache_path: Path,
    output_dir: Path,
    topk_candidates: int = 64,
    neighbor_sigma: float = 20.0,
    do_deepzoom: bool = True,
    do_report: bool = True,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    pool = scan_pool(Path(pool_dir), Path(cache_path))
    grid = load_and_grid(Path(target_path), grid_cols=grid_cols, grid_rows=grid_rows)

    assignment = solve_assignment(
        pool=pool, cells=grid["cells"], grid_shape=grid["shape"],
        lambda_reuse=lambda_reuse, mu_neighbor=mu_neighbor,
        topk=topk_candidates, neighbor_sigma=neighbor_sigma,
    )
    rendered = render_mosaic(assignment, pool, grid, tile_px=tile_px, tau=tau_transfer)

    png_path = output_dir / f"mosaic_{ts}.png"
    rendered["image"].save(png_path)
    result = {"image": rendered["image"], "usage": rendered["usage"], "png_path": png_path}

    if do_report:
        positions = defaultdict(list)
        for (r, c), tile_path in assignment.items():
            positions[tile_path].append((r, c))
        report_path = output_dir / f"report_{ts}.md"
        chart_path = output_dir / f"chart_{ts}.png"
        cold_path = output_dir / f"cold_{ts}.png"
        generate_report(
            pool=pool, usage=rendered["usage"], positions=dict(positions),
            grid_shape=grid["shape"],
            output_md=report_path, output_chart=chart_path, output_cold=cold_path,
        )
        result.update({"report_path": report_path, "chart_path": chart_path, "cold_path": cold_path})

    if do_deepzoom:
        dzi_dir = output_dir / f"deepzoom_{ts}"
        dzi = export_deepzoom(png_path, dzi_dir)
        result["deepzoom_html"] = dzi["html"]
        result["deepzoom_dir"] = dzi_dir

    return result
