"""Source-of-truth for mosaic.ipynb. Regenerate with:

    python _build_notebook.py

Never hand-edit mosaic.ipynb — edit this script and re-run.
"""

from pathlib import Path

import nbformat as nbf

CELLS = [
    # Cell 1 — install & imports
    nbf.v4.new_code_cell(
        "# Cell 1 — install deps (first run only) + imports + seed\n"
        "%pip install -q -r requirements.txt\n"
        "import time\n"
        "from pathlib import Path\n"
        "\n"
        "import ipywidgets as widgets\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "from IPython.display import display\n"
        "from PIL import Image\n"
        "from tqdm.auto import tqdm\n"
        "\n"
        "from mosaic_core import (\n"
        "    MosaicConfig,\n"
        "    build_faiss_index,\n"
        "    build_report,\n"
        "    ensure_seed_tiles,\n"
        "    export_deepzoom,\n"
        "    knn_candidates,\n"
        "    render_mosaic,\n"
        "    rerank,\n"
        "    scan_tile_pool,\n"
        "    split_target,\n"
        ")\n"
        "\n"
        "np.random.seed(0)"
    ),
    # Cell 2 — config + widgets
    nbf.v4.new_code_cell(
        "# Cell 2 — config & interactive sliders\n"
        "config = MosaicConfig()\n"
        "\n"
        "tile_dir_w = widgets.Text(value=str(config.tile_dir), description='tile_dir')\n"
        "target_w = widgets.Text(value=str(config.target_path), description='target')\n"
        "grid_w_w = widgets.IntSlider(value=config.grid_w, min=20, max=240, step=4, description='grid_w')\n"
        "grid_h_w = widgets.IntSlider(value=config.grid_h, min=12, max=135, step=2, description='grid_h')\n"
        "lambda_w = widgets.FloatSlider(value=config.lambda_repeat, min=0.0, max=5.0, step=0.05, description='λ repeat')\n"
        "mu_w = widgets.FloatSlider(value=config.mu_neighbor, min=0.0, max=5.0, step=0.05, description='μ neighbor')\n"
        "tau_w = widgets.FloatSlider(value=config.tau_transfer, min=0.0, max=1.0, step=0.02, description='τ transfer')\n"
        "\n"
        "def _sync(change=None):\n"
        "    config.tile_dir = Path(tile_dir_w.value)\n"
        "    config.target_path = Path(target_w.value) if target_w.value else None\n"
        "    config.grid_w = grid_w_w.value\n"
        "    config.grid_h = grid_h_w.value\n"
        "    config.lambda_repeat = lambda_w.value\n"
        "    config.mu_neighbor = mu_w.value\n"
        "    config.tau_transfer = tau_w.value\n"
        "\n"
        "for w in (tile_dir_w, target_w, grid_w_w, grid_h_w, lambda_w, mu_w, tau_w):\n"
        "    w.observe(_sync, names='value')\n"
        "_sync()\n"
        "\n"
        "display(widgets.VBox([tile_dir_w, target_w, grid_w_w, grid_h_w, lambda_w, mu_w, tau_w]))"
    ),
    # Cell 3 — tile pool
    nbf.v4.new_code_cell(
        "# Cell 3 — load / seed tile pool\n"
        "ensure_seed_tiles(config.tile_dir)\n"
        "tile_records, bad_files = scan_tile_pool(config.tile_dir, config.cache_path)\n"
        "print(f'扫到 {len(tile_records)} 张 tile,坏图 {len(bad_files)} 张。')"
    ),
    # Cell 4 — target + grid
    nbf.v4.new_code_cell(
        "# Cell 4 — load target image, fallback to gradient if missing\n"
        "def _fallback_target(w=768, h=432):\n"
        "    grad = np.linspace(0, 255, w, dtype=np.uint8)[None, :, None].repeat(h, axis=0).repeat(3, axis=2)\n"
        "    grad[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]\n"
        "    return Image.fromarray(grad)\n"
        "\n"
        "if config.target_path and config.target_path.exists():\n"
        "    target_img = Image.open(config.target_path).convert('RGB')\n"
        "else:\n"
        "    print(f'⚠️  目标图 {config.target_path} 不存在,使用内置渐变兜底')\n"
        "    target_img = _fallback_target()\n"
        "\n"
        "target_lab = split_target(target_img, config.grid_w, config.grid_h)\n"
        "print(f'目标 LAB grid shape = {target_lab.shape}')\n"
        "plt.figure(figsize=(6, 4))\n"
        "plt.imshow(target_img)\n"
        "plt.title('target')\n"
        "plt.axis('off')\n"
        "plt.show()"
    ),
    # Cell 5 — match loop
    nbf.v4.new_code_cell(
        "# Cell 5 — KNN candidates + per-cell rerank with visible reasoning\n"
        "tile_labs = np.stack([t.lab_mean for t in tile_records]).astype(np.float32)\n"
        "index = build_faiss_index(tile_labs)\n"
        "candidates = knn_candidates(target_lab, index, k=config.k_candidates)\n"
        "\n"
        "grid_h, grid_w = target_lab.shape[:2]\n"
        "assignment = np.zeros((grid_h, grid_w), dtype=np.int64)\n"
        "usage_counts: dict[int, int] = {}\n"
        "t0 = time.time()\n"
        "total = grid_h * grid_w\n"
        "\n"
        "for flat in tqdm(range(total), desc='matching'):\n"
        "    r, c = divmod(flat, grid_w)\n"
        "    cand = candidates[flat]\n"
        "    neighbors = []\n"
        "    if c > 0: neighbors.append(int(assignment[r, c - 1]))\n"
        "    if r > 0: neighbors.append(int(assignment[r - 1, c]))\n"
        "    best = rerank(cand, tile_labs, target_lab[r, c], usage_counts, neighbors,\n"
        "                  config.lambda_repeat, config.mu_neighbor)\n"
        "    assignment[r, c] = best\n"
        "    usage_counts[best] = usage_counts.get(best, 0) + 1\n"
        "    if flat % 50 == 0 and tile_records[best].path is not None:\n"
        "        print(f'  ({r:>3},{c:>3}) -> {tile_records[best].path.name}')\n"
        "\n"
        "elapsed = time.time() - t0\n"
        "print(f'匹配完成,耗时 {elapsed:.2f} 秒')"
    ),
    # Cell 6 — render
    nbf.v4.new_code_cell(
        "# Cell 6 — render mosaic\n"
        "mosaic_img = render_mosaic(assignment, tile_records, config.tile_px,\n"
        "                           config.tau_transfer, target_lab)\n"
        "config.out_dir.mkdir(parents=True, exist_ok=True)\n"
        "mosaic_path = config.out_dir / 'mosaic.png'\n"
        "mosaic_img.save(mosaic_path)\n"
        "print(f'saved {mosaic_path}')\n"
        "display(mosaic_img)"
    ),
    # Cell 7 — report
    nbf.v4.new_code_cell(
        "# Cell 7 — self-deprecating report\n"
        "report = build_report(assignment, tile_records, elapsed, bad_files)\n"
        "(config.out_dir / 'report.txt').write_text(report.text, encoding='utf-8')\n"
        "print(report.text)\n"
        "plt.show()  # flushes any pending figure\n"
        "display(report.usage_bar_fig)\n"
        "display(report.cold_wall_fig)"
    ),
    # Cell 8 — deepzoom
    nbf.v4.new_code_cell(
        "# Cell 8 — DeepZoom export\n"
        "index_html = export_deepzoom(mosaic_path, config.out_dir / 'deepzoom')\n"
        "print(f'✅ 已生成 {index_html},在浏览器打开即可无限缩放')"
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    out_path = Path(__file__).parent / "mosaic.ipynb"
    nbf.write(nb, str(out_path))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
