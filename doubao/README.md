# Photomosaic Toy

一个本地跑的玩具：把你硬盘里的照片重组成另一张照片。

> 定位：**玩具**，不是 SaaS / API / 产品。速度不重要，稳定性不重要，可解释性重要，彩蛋比功能值钱。

## 跑起来

    cd doubao
    uv sync --all-groups
    uv run jupyter lab photomosaic.ipynb

打开 notebook 后编辑 **Cell 2** 把 `tile_source_dir` 指向你的照片目录、`target_image` 指向目标图，然后从上到下依次运行。

## 输出

- `output/mosaic.png` —— 最终马赛克
- `output/deepzoom/index.html` —— 用浏览器打开可无限缩放，放到底能看清每张小图
- 控制台打印自嘲式报告（"用了 X 张里的 Y 张，冷宫 TOP 5..."）
- matplotlib 显示使用次数柱状图和冷宫照片墙

## 旋钮

全部在 Cell 2 的 `MosaicConfig` 里：

| 参数 | 含义 | 默认 |
|------|-----|-----|
| `grid` | 网格 (cols, rows) | (120, 68) |
| `tile_px` | 每格渲染像素 | 16 |
| `candidate_k` | 每格先筛 top-K 颜色候选 | 50 |
| `lambda_reuse` | 重复惩罚（λ）：大→使用更分散 | 0.3 |
| `mu_neighbor` | 邻居惩罚（μ）：大→相邻不同 | 0.2 |
| `tau_tone` | 色调迁移（τ）：0 原色，1 完全贴合目标 | 0.5 |
| `verbose` | 匹配每格时打印决策 | True |

## 架构

    src/mosaic/
      config.py    MosaicConfig dataclass（全部参数）
      tiles.py     扫描 + LAB 均色 + pickle 缓存
      match.py     split_target + match_all_tiles（含 λ/μ 惩罚）
      render.py    Reinhard 色调迁移 + 贴图
      report.py    自嘲文字 + 柱状图 + 冷宫墙
      deepzoom.py  DZI 金字塔 + OpenSeadragon HTML
    photomosaic.ipynb   8 cell 编排

详见 `docs/superpowers/specs/2026-04-17-photomosaic-toy-design.md`，实施计划 `docs/superpowers/plans/2026-04-17-photomosaic-toy.md`。

## 测试

    uv run pytest

## 不在 MVP 的（已为增量预留接口）

CLIP 语义匹配 · cursed mode · Gradio 滑条 UI · rembg 主体 mask · 底图标签叙事。
