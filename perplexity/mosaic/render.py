"""组装最终 mosaic：对每格 tile 做色调迁移后贴到画布，累计 usage。"""
from __future__ import annotations

from collections import Counter

import numpy as np
from PIL import Image

from mosaic.transfer import reinhard_transfer


def render_mosaic(assignment: dict, pool: dict, grid: dict, tile_px: int, tau: float) -> dict:
    """
    assignment: {(row, col): tile_path}
    pool: {tile_path: {lab_mean, thumbnail, mtime}}
    grid: {'shape': (rows, cols), 'cells': [...], 'image': PIL.Image, 'cell_size': (h, w)}
    返回 {'image': PIL.Image (RGB), 'usage': Counter[tile_path]}
    """
    rows, cols = grid["shape"]
    target_image = grid["image"]
    target_h, target_w = target_image.size[1], target_image.size[0]
    cell_h = target_h // rows
    cell_w = target_w // cols

    out_h = tile_px * rows
    out_w = tile_px * cols
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    target_arr = np.asarray(target_image, dtype=np.uint8)
    usage: Counter = Counter()

    tile_cache = {}
    for r in range(rows):
        for c in range(cols):
            tile_path = assignment[(r, c)]
            if tile_path not in tile_cache:
                img = Image.open(tile_path).convert("RGB")
                tile_cache[tile_path] = np.asarray(
                    img.resize((tile_px, tile_px), Image.LANCZOS), dtype=np.uint8
                )
            tile_arr = tile_cache[tile_path]

            target_patch = target_arr[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            if target_patch.shape[0] != tile_px or target_patch.shape[1] != tile_px:
                patch_img = Image.fromarray(target_patch).resize((tile_px, tile_px), Image.LANCZOS)
                target_patch = np.asarray(patch_img, dtype=np.uint8)

            transferred = reinhard_transfer(tile_arr, target_patch, tau=tau)
            canvas[r * tile_px : (r + 1) * tile_px, c * tile_px : (c + 1) * tile_px] = transferred
            usage[tile_path] += 1

    return {"image": Image.fromarray(canvas), "usage": usage}
