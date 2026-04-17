# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Photomosaic Toy
# Local-only. Deliberately slow. Deliberately explainable.
# Edit Cell 2 config, then "Run All".

# %%
# Cell 1 — Imports
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
from PIL import Image
from src.config import PhotomosaicConfig
from src.tile_pool import load_or_build_index, add_clip_embeddings, load_tags
from src.target import split_into_patches
from src.matcher import color_topk, assign_with_penalties, assign_with_clip
from src.renderer import render_mosaic
from src.reporter import build_text_report, save_usage_plot, save_cold_wall
from src.deepzoom import export_deepzoom
print("imports ok")

# %%
# Cell 2 — Config (your only UI)
cfg = PhotomosaicConfig(
    target="samples/target.jpg",
    tile_dir="samples/tiles",
    grid=(120, 68),
    tile_px=16,
    lambda_repeat=0.3,
    mu_neighbor=0.5,
    tau_tone=0.5,
    use_clip=False,
    mode="normal",
    output_dir="output",
)
Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
print(cfg)

# %%
# Cell 3 — Scan tile pool
idx = load_or_build_index(cfg.tile_dir, cache_dir=cfg.cache_dir)
tags = load_tags(cfg.tile_dir, idx.paths)
print(f"Tile pool: {len(idx.paths)} photos")
if cfg.use_clip:
    idx = add_clip_embeddings(idx)
    print(f"CLIP embeddings: {idx.clip_emb.shape}")

# %%
# Cell 4 — Target split
patches_lab, cell_rgb = split_into_patches(cfg.target, grid=cfg.grid)
print(f"Target split into {patches_lab.shape[0]} x {patches_lab.shape[1]} cells")

# %%
# Cell 5 — Match (with live narration)
topk_idx, topk_dist = color_topk(patches_lab, idx.lab_mean, k=cfg.topk_color)
total_cells = patches_lab.shape[0] * patches_lab.shape[1]

def narrate(r, c, chosen, candidates):
    if (r * patches_lab.shape[1] + c) % max(total_cells // 40, 1) == 0:
        top3 = sorted(candidates, key=lambda x: x[1])[:3]
        chosen_name = Path(idx.paths[chosen]).name
        runners = ", ".join(f"{Path(idx.paths[t]).name}({s:.1f})" for t, s in top3)
        print(f"[{r:3d},{c:3d}] picked {chosen_name} — top3 by score: {runners}")

if cfg.use_clip and idx.clip_emb is not None:
    # Compute per-cell CLIP emb from upscaled target patch
    import open_clip, torch
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    rows, cols = patches_lab.shape[:2]
    # Use the larger rescaled target to extract meaningful patches
    full = Image.open(cfg.target).convert("RGB").resize((cols * 32, rows * 32), Image.LANCZOS)
    patch_embs = np.zeros((rows, cols, idx.clip_emb.shape[1]), dtype=np.float32)
    with torch.no_grad():
        for r in range(rows):
            batch = [preprocess(full.crop((c*32, r*32, (c+1)*32, (r+1)*32))) for c in range(cols)]
            feat = model.encode_image(torch.stack(batch).to(device))
            feat = feat / feat.norm(dim=-1, keepdim=True)
            patch_embs[r] = feat.cpu().numpy()
    assignment = assign_with_clip(topk_idx, topk_dist, patches_lab, idx.lab_mean,
                                  tile_clip=idx.clip_emb, patch_clip=patch_embs,
                                  lambda_repeat=cfg.lambda_repeat, mu_neighbor=cfg.mu_neighbor,
                                  clip_weight=cfg.clip_weight, on_cell=narrate)
else:
    assignment = assign_with_penalties(topk_idx, topk_dist,
                                       lambda_repeat=cfg.lambda_repeat,
                                       mu_neighbor=cfg.mu_neighbor,
                                       on_cell=narrate)

# %%
# Cell 6 — Render
mosaic, usage = render_mosaic(assignment, idx.paths, cell_rgb,
                               tile_px=cfg.tile_px, tau=cfg.tau_tone)
mosaic_path = Path(cfg.output_dir) / "mosaic.png"
mosaic.save(mosaic_path)
print(f"wrote {mosaic_path}  ({mosaic.size[0]}×{mosaic.size[1]} px)")
mosaic

# %%
# Cell 7 — Report
report = build_text_report(idx.paths, usage, tags, total_cells)
print(report)
save_usage_plot(usage, str(Path(cfg.output_dir) / "usage_histogram.png"))
save_cold_wall(idx.paths, usage, str(Path(cfg.output_dir) / "cold_wall.png"))
(Path(cfg.output_dir) / "report.txt").write_text(report)

# %%
# Cell 8 — DeepZoom HTML (send this folder to a friend)
dz_dir = Path(cfg.output_dir) / "deepzoom"
export_deepzoom(str(mosaic_path), str(dz_dir), title=f"Photomosaic: {Path(cfg.target).stem}")
print(f"open {dz_dir}/index.html in a browser (or serve the folder)")
