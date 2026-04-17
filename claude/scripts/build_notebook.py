"""Build bead_mosaic.ipynb via nbformat. Run: python scripts/build_notebook.py"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bead_mosaic.ipynb"

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Bead Mosaic — Phase 2\n"
    "Zero-config demo: run all cells top-to-bottom.\n"
    "To use your own photos: set `CONFIG['BASE_DIR']` + `CONFIG['TARGET_PATH']` in Cell 2 and flip `DEMO_MODE` to `False`.\n"
    "\n"
    "**Phase 2 adds three knobs:**\n"
    "- `LAMBDA` — repetition penalty (0 = no penalty, large = force tile diversity)\n"
    "- `MU` — neighbor penalty (0 = ignore, large = no adjacent duplicates)\n"
    "- `TAU` — tone transfer strength [0, 1] (0 = original tile colors, 1 = tiles adopt target patch's LAB mean)\n"
))

cells.append(nbf.v4.new_code_cell(
    "%pip install -q -r requirements.txt\n"
    "import sys\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, str(Path.cwd()))\n"
    "\n"
    "import numpy as np\n"
    "import ipywidgets as widgets\n"
    "from PIL import Image, ImageOps\n"
    "from skimage import data\n"
    "from skimage.color import rgb2lab\n"
    "from IPython.display import display\n"
    "\n"
    "from src import match, render, scan"
))

cells.append(nbf.v4.new_code_cell(
    "# --- edit these for real usage ---\n"
    "CONFIG = {\n"
    "    'BASE_DIR': Path.cwd() / 'my_photos',\n"
    "    'TARGET_PATH': None,  # None -> use skimage astronaut\n"
    "    'GRID_W': 120,\n"
    "    'GRID_H': 68,\n"
    "    'TILE_PX': 24,\n"
    "    'CACHE_DIR': Path.cwd() / '.cache',\n"
    "    'OUTPUT_PATH': Path.cwd() / 'output.png',\n"
    "    'DEMO_MODE': True,\n"
    "    'LAMBDA': 0.0,  # repetition penalty\n"
    "    'MU': 0.0,      # neighbor penalty\n"
    "    'TAU': 0.0,     # tone transfer strength [0, 1]\n"
    "}\n"
    "CONFIG"
))

cells.append(nbf.v4.new_code_cell(
    "pool = scan.build_pool(\n"
    "    base_dir=CONFIG['BASE_DIR'],\n"
    "    cache_dir=CONFIG['CACHE_DIR'],\n"
    "    tile_px=CONFIG['TILE_PX'],\n"
    "    demo_mode=CONFIG['DEMO_MODE'],\n"
    ")\n"
    "print(f\"pool size: {pool.lab.shape[0]} tiles\")"
))

cells.append(nbf.v4.new_code_cell(
    "def load_target(path, grid_w, grid_h, tile_px):\n"
    "    if path is None:\n"
    "        arr = data.astronaut()\n"
    "        img = Image.fromarray(arr)\n"
    "    else:\n"
    "        img = Image.open(path)\n"
    "        img = ImageOps.exif_transpose(img).convert('RGB')\n"
    "    out_w, out_h = grid_w * tile_px, grid_h * tile_px\n"
    "    src_w, src_h = img.size\n"
    "    scale = min(out_w / src_w, out_h / src_h)\n"
    "    new_w, new_h = int(src_w * scale), int(src_h * scale)\n"
    "    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)\n"
    "    canvas = Image.new('RGB', (out_w, out_h), (0, 0, 0))\n"
    "    canvas.paste(resized, ((out_w - new_w) // 2, (out_h - new_h) // 2))\n"
    "    return np.asarray(canvas, dtype=np.uint8)\n"
    "\n"
    "target_rgb = load_target(\n"
    "    CONFIG['TARGET_PATH'], CONFIG['GRID_W'], CONFIG['GRID_H'], CONFIG['TILE_PX']\n"
    ")\n"
    "target_lab = rgb2lab(target_rgb.astype(np.float32) / 255.0)\n"
    "target_lab_grid = target_lab.reshape(\n"
    "    CONFIG['GRID_H'], CONFIG['TILE_PX'],\n"
    "    CONFIG['GRID_W'], CONFIG['TILE_PX'], 3\n"
    ").mean(axis=(1, 3)).astype(np.float32)\n"
    "target_lab_grid.shape"
))

cells.append(nbf.v4.new_code_cell(
    "idx = match.match_grid(\n"
    "    target_lab_grid, pool.lab,\n"
    "    lambda_=CONFIG['LAMBDA'], mu=CONFIG['MU'],\n"
    ")\n"
    "idx.shape"
))

cells.append(nbf.v4.new_code_cell(
    "img, usage = render.render_mosaic_with_usage(\n"
    "    idx, pool, CONFIG['TILE_PX'], CONFIG['OUTPUT_PATH'],\n"
    "    target_rgb=target_rgb, tone_strength=CONFIG['TAU'],\n"
    ")\n"
    "display(img)\n"
    "print(f\"\\noutput written to {CONFIG['OUTPUT_PATH']}\")\n"
    "print(f\"tiles used: {len(usage)} distinct / {sum(usage.values())} total placements\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 交互滑条\n"
    "拖动滑条不会自动重跑——点 **重跑** 按钮触发一次 match+render。"
))

cells.append(nbf.v4.new_code_cell(
    "lambda_slider = widgets.FloatSlider(value=CONFIG['LAMBDA'], min=0, max=200, step=5, description='λ (重复)', continuous_update=False)\n"
    "mu_slider = widgets.FloatSlider(value=CONFIG['MU'], min=0, max=1000, step=10, description='μ (邻居)', continuous_update=False)\n"
    "tau_slider = widgets.FloatSlider(value=CONFIG['TAU'], min=0, max=1, step=0.05, description='τ (色调)', continuous_update=False)\n"
    "rerun_btn = widgets.Button(description='重跑', button_style='primary')\n"
    "out = widgets.Output()\n"
    "\n"
    "def _rerun(_):\n"
    "    with out:\n"
    "        out.clear_output()\n"
    "        idx = match.match_grid(\n"
    "            target_lab_grid, pool.lab,\n"
    "            lambda_=lambda_slider.value, mu=mu_slider.value,\n"
    "        )\n"
    "        img, usage = render.render_mosaic_with_usage(\n"
    "            idx, pool, CONFIG['TILE_PX'], CONFIG['OUTPUT_PATH'],\n"
    "            target_rgb=target_rgb, tone_strength=tau_slider.value,\n"
    "        )\n"
    "        display(img)\n"
    "        print(f\"tiles used: {len(usage)} distinct / {sum(usage.values())} total\")\n"
    "\n"
    "rerun_btn.on_click(_rerun)\n"
    "display(widgets.VBox([lambda_slider, mu_slider, tau_slider, rerun_btn, out]))"
))

nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
OUT.write_text(nbf.writes(nb))
print(f"wrote {OUT}")
