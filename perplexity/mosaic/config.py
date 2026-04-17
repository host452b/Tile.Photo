"""默认参数 + ipywidgets 交互面板工厂（三滑条 λ/μ/τ）。"""
from __future__ import annotations

import ipywidgets as widgets

DEFAULT_CONFIG = {
    "target_path": "target.jpg",
    "pool_dir": "pool/",
    "grid_cols": 120,
    "grid_rows": 68,
    "tile_px": 16,
    "preview_grid_cols": 48,
    "preview_grid_rows": 27,
    "preview_tile_px": 6,
    "lambda_reuse": 1.0,
    "mu_neighbor": 0.5,
    "tau_transfer": 0.5,
    "topk_candidates": 64,
    "neighbor_sigma": 20.0,
    "cache_dir": ".cache",
    "output_dir": "output",
    "seed": 42,
}


def build_widgets() -> dict:
    lam = widgets.FloatSlider(
        value=DEFAULT_CONFIG["lambda_reuse"], min=0.0, max=5.0, step=0.1,
        description="λ reuse:", style={"description_width": "initial"},
    )
    mu = widgets.FloatSlider(
        value=DEFAULT_CONFIG["mu_neighbor"], min=0.0, max=5.0, step=0.1,
        description="μ neighbor:", style={"description_width": "initial"},
    )
    tau = widgets.FloatSlider(
        value=DEFAULT_CONFIG["tau_transfer"], min=0.0, max=1.0, step=0.05,
        description="τ transfer:", style={"description_width": "initial"},
    )
    preview_btn = widgets.Button(description="预览 48×27", button_style="info")
    render_btn = widgets.Button(description="正式渲染 120×68", button_style="primary")

    container = widgets.VBox([
        widgets.HTML("<h4>参数</h4>"), lam, mu, tau,
        widgets.HBox([preview_btn, render_btn]),
    ])

    def get_params() -> dict:
        return {"lambda_reuse": lam.value, "mu_neighbor": mu.value, "tau_transfer": tau.value}

    return {
        "container": container,
        "get_params": get_params,
        "preview_button": preview_btn,
        "render_button": render_btn,
    }
