"""One-shot helper: regenerate mosaic.ipynb from inline cell sources."""

from __future__ import annotations

import json
from pathlib import Path


CELLS: list[tuple[str, str]] = [
    # Cell 1 — imports (re-run this first; the %pip line is safe to keep)
    (
        "code",
        """\
%pip install -q -r requirements.txt
from pathlib import Path
import numpy as np
from PIL import Image
import ipywidgets as W
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

from mosaic.pool import scan_pool
from mosaic.target import load_and_slice
from mosaic.match import build_index, match_grid
from mosaic.render import compose
from mosaic.report import text_report, cold_wall, usage_hist_figure
from mosaic.dzi import export_dzi
""",
    ),
    # Cell 2 — config + widgets
    (
        "code",
        """\
target_path = W.Text(value="", description="target:", layout=W.Layout(width="90%"))
pool_path = W.Text(value="", description="pool dir:", layout=W.Layout(width="90%"))
grid_w = W.IntSlider(value=120, min=40, max=240, step=4, description="grid_w")
tile_px = W.IntSlider(value=16, min=8, max=48, step=1, description="tile_px")
lam = W.FloatSlider(value=2.0, min=0.0, max=10.0, step=0.1, description="\u03bb diversity")
mu = W.FloatSlider(value=0.5, min=0.0, max=5.0, step=0.1, description="\u03bc neighbor")
tau = W.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05, description="\u03c4 tone")
regen = W.Button(description="Regenerate", button_style="primary")
dzi_btn = W.Button(description="Export DeepZoom")
log = W.Output(layout=W.Layout(height="200px", border="1px solid #888", overflow="auto"))
preview = W.Image(layout=W.Layout(width="100%"))

display(W.VBox([
    target_path, pool_path,
    W.HBox([grid_w, tile_px]),
    W.HBox([lam, mu, tau]),
    W.HBox([regen, dzi_btn]),
    log, preview,
]))

state = {"tiles": None, "grid": None, "choices": None, "mosaic": None, "out_dir": None}
""",
    ),
    # Cell 3 — scan pool
    (
        "code",
        """\
def _scan():
    with log:
        clear_output()
        if not pool_path.value:
            print("fill pool dir path above, then run this cell")
            return
        print(f"scanning {pool_path.value} ...")
        tiles = scan_pool(Path(pool_path.value), Path("cache"), thumb_px=32)
        print(f"got {len(tiles)} tiles")
        state["tiles"] = tiles

_scan()
""",
    ),
    # Cell 4 — load & slice target
    (
        "code",
        """\
def _slice():
    with log:
        clear_output()
        if not target_path.value:
            print("fill target path above, then run this cell")
            return
        gh = max(1, int(grid_w.value * 9 / 16))
        tg = load_and_slice(Path(target_path.value), grid_w=grid_w.value, grid_h=gh)
        state["grid"] = tg
        print(f"target sliced into {tg.lab_means.shape[1]} \u00d7 {tg.lab_means.shape[0]} cells")

_slice()
""",
    ),
    # Cell 5 — match
    (
        "code",
        """\
def _match():
    with log:
        clear_output()
        tiles = state["tiles"]
        tg = state["grid"]
        if not tiles or tg is None:
            print("run cells 3 and 4 first")
            return
        tile_labs = np.stack([t.lab for t in tiles])
        idx = build_index(tile_labs)
        total = int(tg.lab_means.shape[0] * tg.lab_means.shape[1])
        choices = match_grid(
            idx, tile_labs, tg.lab_means,
            k=20, lambda_=lam.value, mu=mu.value,
            log_every=max(50, total // 20),
        )
        state["choices"] = choices
        print(f"match complete: {choices.shape}")

_match()
""",
    ),
    # Cell 6 — render
    (
        "code",
        """\
def _render():
    with log:
        clear_output()
        tiles = state["tiles"]
        tg = state["grid"]
        choices = state.get("choices")
        if not tiles or tg is None or choices is None:
            print("run cells 3\u20135 first")
            return
        out_dir = Path("out") / Path(target_path.value).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        state["out_dir"] = out_dir
        mosaic = compose(tiles, choices, tg.lab_means, tile_px=tile_px.value, tau=tau.value)
        state["mosaic"] = mosaic
        Image.fromarray(mosaic).save(out_dir / "mosaic.png")
        with open(out_dir / "mosaic.png", "rb") as f:
            preview.value = f.read()
        print(f"wrote {out_dir / 'mosaic.png'}")

_render()
""",
    ),
    # Cell 7 — report
    (
        "code",
        """\
def _report():
    with log:
        clear_output()
        tiles = state["tiles"]
        choices = state.get("choices")
        out_dir = state.get("out_dir")
        if not tiles or choices is None or out_dir is None:
            print("run cells 3\u20136 first")
            return
        uses = np.bincount(choices.ravel(), minlength=len(tiles))
        report = text_report(
            tiles, uses, grid_shape=choices.shape,
            params={"lambda_": lam.value, "mu": mu.value, "tau": tau.value},
        )
        (out_dir / "report.txt").write_text(report)
        print(report)

        wall = cold_wall(tiles, uses, n=20, thumb_px=32, cols=5)
        Image.fromarray(wall).save(out_dir / "cold_wall.png")

        fig = usage_hist_figure(tiles, uses)
        fig.savefig(out_dir / "usage_hist.png", dpi=110)
        plt.close(fig)

_report()
""",
    ),
    # Cell 8 — DeepZoom + button wiring
    (
        "code",
        """\
def _dzi():
    with log:
        clear_output()
        mosaic = state.get("mosaic")
        out_dir = state.get("out_dir")
        if mosaic is None or out_dir is None:
            print("run cell 6 first")
            return
        html = export_dzi(Image.fromarray(mosaic), out_dir / "dzi")
        print(f"wrote {html}")
        print(f"share the folder {out_dir / 'dzi'} to give someone a zoomable version")

regen.on_click(lambda _: (_match(), _render(), _report()))
dzi_btn.on_click(lambda _: _dzi())
""",
    ),
]


def main() -> None:
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": src,
                "outputs": [],
                "execution_count": None,
            }
            for _, src in CELLS
        ],
        "metadata": {
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent.parent / "mosaic.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
