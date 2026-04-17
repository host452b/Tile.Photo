# Photomosaic ipynb 生成器 — 设计文档

- **日期**: 2026-04-17
- **作者**: joejiang@nvidia.com
- **状态**: 已批准，待进入 writing-plans

## 1. 定位与非目标

**是什么**: 本地 Mac 上跑的 Jupyter notebook，把硬盘里一堆照片重组成另一张照片（photomosaic）。

**不是什么**: 不是 SaaS、不是 API、不是产品、不是作品集级代码。

**反直觉选择**（源于"玩具"定位）:
- 速度不重要：跑 10 分钟出一张可接受。
- 普适性不重要：只在 joejiang 的 Mac 上跑。
- 稳定性不重要：崩了重跑。
- **可解释性很重要**：好玩的本质是"看它怎么思考"。
- **彩蛋比功能值钱**：后续 V2/V3 会加 Cursed Mode、时间胶囊等玩法。

## 2. MVP / V2 / V3 范围划分

### MVP（本 spec 覆盖）

1. 8-cell ipynb 主流程
2. 底图池扫描 + LAB 平均色 + pickle 缓存（mtime 失效）
3. faiss top-K 颜色候选 + λ 重复惩罚 + μ 邻居惩罚（贪心匹配）
4. Reinhard LAB 色调迁移（τ 强度可调）
5. ipywidgets 内嵌交互（三滑条 λ/μ/τ + 预览 + 正式渲染）
6. 自嘲式报告（文字模板 + 柱状图 + 冷宫照片墙）
7. DeepZoom 导出（OpenSeadragon HTML + 金字塔瓦片）
8. 匹配过程日志（tqdm + 每 100 格 print 决策）

### V2（后续）

- CLIP 语义匹配（通过 `semantic_reranker` 钩子接入，MVP 已预留参数）
- 底图池打标签 + 报告叙事（"23% 来自 2019 日本旅行"）
- Cursed Mode 预设（"用表情包拼证件照" 等几个预配置 recipe）

### V3（彩蛋）

- WeChat 聊天截图专用模式（可读小字版）
- `rembg` / saliency 主体 mask，主体区更密
- 实时 canvas 看它一格一格填的动画

## 3. 架构

### 目录结构

```
perplexity/
  mosaic.ipynb              # 8-cell 主入口
  mosaic/                   # 核心 Python package（从 cell 抽出以便测试/复用）
    __init__.py
    config.py               # 默认参数 + ipywidgets UI 工厂
    pool.py                 # 扫描底图、LAB 均值、pickle 缓存
    target.py               # 读目标图、裁切、分网格
    match.py                # faiss top-K + λ/μ 重排 + semantic_reranker 钩子
    transfer.py             # Reinhard LAB 色调迁移
    render.py               # 贴图 + usage 计数
    report.py               # 文字报告 + 柱状图 + 冷宫墙
    zoom.py                 # DeepZoom + OpenSeadragon HTML
  tests/
    test_match.py
    test_transfer.py
    test_pool.py
  .cache/                   # pool_features.pkl（自动创建，gitignore）
  output/                   # mosaic_*.png / deepzoom_*/ / report_*.md
  pool/                     # 用户提供的底图目录（gitignore）
  target.jpg                # 用户提供的目标图（gitignore）
```

**约束**: 每个模块文件 <300 行，单一职责。Cell 里只做 import + 调用 + display，业务逻辑全在 package 里。

### 数据流

```
底图目录 ──▶ pool.scan() ──▶ .cache/pool_features.pkl
                              (tile_path, LAB_mean, thumbnail_16px, mtime)
                                     │
目标图 ──▶ target.grid() ──▶ 每格 (row, col, LAB_mean, variance)
                                                     │
                              match.solve(λ, μ) ────┤
                                                     │
                                          assignment: {(row,col): tile_path}
                                                     │
                                          transfer.reinhard(τ)
                                                     ▼
                                          render.paste ──▶ numpy array ──▶ PIL save
                                                     │                     (mosaic_<ts>.png)
                              ┌──────────────────────┴───────────────────┐
                              ▼                                          ▼
                       report.generate()                         zoom.deepzoom(png_path)
                       (report_<ts>.md + charts)                 (deepzoom_<ts>/)
```

## 4. 匹配算法

### 优化目标

对每个网格位置 `g`，选 tile `t` 最小化：

```
cost(g, t) = ‖LAB_mean(g) − LAB_mean(t)‖²      # 颜色主项
           + λ · log(1 + usage_count[t])       # 重复惩罚
           + μ · Σ_{n∈N4(g)} sim(t, t_n)       # 邻居相似惩罚（4 邻域已决策的）
           + ω · semantic_dist(g, t)           # V2 CLIP 钩子（MVP ω=0）
```

其中 `sim(a, b) = exp(−‖LAB(a) − LAB(b)‖² / σ²)`，4 邻域只看已经决策过的上方和左方（扫描顺序的自然先行者）。

### 求解顺序

按目标网格的**颜色方差降序**扫描（方差大的稀有色先选）。动机：让"稀有色区域"先抢到最匹配的 tile，而不是被平凡区域用光。

**邻居惩罚的兼容处理**：因为扫描顺序不再是行优先，"4 邻域"指的是网格几何上的上下左右；只把其中**已经决策过**的格子算入邻居相似项，尚未决策的邻居跳过。这让 μ 项自然随扫描进度生效。

### faiss 使用

- `faiss.IndexFlatL2`（3 维 LAB 空间，池规模 ~3K，秒级建索引，精确 NN）
- 每格取 `top_k=64` 颜色候选 → Python 里按完整 cost 重排 → argmin
- 为什么不 ILP：严格最优是 NP-hard 分配问题，玩具场景肉眼看不出差距，贪心省 10× 时间

### 可见性（"看它思考"）

- 整体 `tqdm` 进度条
- 每 100 格 `print` 一行：`(42, 17) → IMG_2019_03_14_217.jpg | color=8.2, used=3x, neigh=0.12`
- 匹配完印一段 summary：候选被 top-1 命中的比例、平均 cost、最高 usage tile

## 5. 色调迁移（Reinhard）

对每块 tile，贴入前：

```python
# LAB 空间逐通道均值-标准差迁移
tile_lab[c] = (tile_lab[c] − μ_tile[c]) * (σ_target[c] / σ_tile[c]) + μ_target[c]
# 再与原 tile 按 τ 混合
out = (1 − τ) * tile_orig + τ * tile_transferred
```

`τ ∈ [0, 1]` 默认 0.5。τ=0 完全原色（近看清晰，远看色差重），τ=1 完全贴合（远看完美，近看染色重）。甜区通常在 [0.4, 0.6]——这也是商业工具默认 τ≈0.9 被 Reddit 抱怨"染色太重"的反面。

## 6. ipywidgets 交互（Cell 2）

```
┌────────────────────────────────────────────┐
│ λ (reuse penalty):      [====|----] 1.0    │
│ μ (neighbor penalty):   [==|------] 0.5    │
│ τ (tonal transfer):     [====|----] 0.5    │
│                                             │
│ [预览 48×27]   [正式渲染 120×68]           │
└────────────────────────────────────────────┘
[预览图区域 / 正式图区域]
```

- **预览按钮**：用 48×27 网格 + 6px tile（= 288×162 输出），faiss 索引复用，出图 ~5s
- **正式渲染按钮**：完整 120×68 + 16px（= 1920×1088），跑完自动触发 §7 报告 + §8 DeepZoom

## 7. 报告

**输出**: `output/report_<ts>.md` + notebook 内 `IPython.display`

**结构**:

1. **文字段**（template，按用户原文口吻）:
   ```
   本次使用了你 {pool_total} 张照片里的 {used_count} 张。
   其中 {top_tile_name} 被用了 {top_count} 次（主要用于填充 {top_region_guess}）。
   冷宫照片 TOP 5 是：{cold_top5_names}（都是你 {cold_guess}）。
   ```
   `top_region_guess` 是启发式标签：计算该 tile 被使用位置的质心 `(ȳ, x̄)` 归一化到 [0,1]：
   - `ȳ < 0.33` → "天空"
   - `ȳ > 0.67` → "地面"
   - 其它 + `|x̄ − 0.5| < 0.2` → "主体"
   - 其它 → "填充"

   标签不追求准确，追求好笑。

2. **使用次数柱状图**: matplotlib，top-30 tile，x 轴文件名（截断），y 轴次数。

3. **冷宫照片墙**: `usage_count=0` 的 tile 随机抽 20 张，4×5 平铺缩略图。

## 8. DeepZoom 导出

- 库：`deepzoom`（pure-Python，pip 安装，不依赖 `libvips` brew 安装）
- 切金字塔瓦片到 `output/deepzoom_<ts>/`
- 生成自包含 `index.html`，内嵌 OpenSeadragon CDN 链接 + 初始化 JS
- 用户双击打开就是一个能无限放大的 photomosaic

**html 结构（简化）**:
```html
<!DOCTYPE html>
<script src="https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/openseadragon.min.js"></script>
<div id="viewer" style="width:100vw;height:100vh"></div>
<script>
OpenSeadragon({ id: "viewer", tileSources: "mosaic.dzi" });
</script>
```

## 9. 缓存策略

**`.cache/pool_features.pkl`** 结构：
```python
{
  "<tile_path>": {
    "mtime": <float>,          # os.path.getmtime
    "lab_mean": np.array([L, a, b]),
    "thumbnail": np.array(16, 16, 3),  # 用作邻居相似度判断
  },
  ...
}
```

**增量扫描逻辑**:
- 启动时 load pickle
- 遍历 pool 目录，对每个文件：
  - 不在缓存 → 新图，计算并加入
  - 在缓存但 mtime 不同 → 重算并更新
  - 在缓存且 mtime 一致 → skip
- 缓存中存在但文件已删 → 移除
- 全扫完 dump 回 pickle

## 10. 默认参数

```python
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
    "neighbor_sigma": 20.0,      # LAB space, 邻居相似度衰减
    "cache_dir": ".cache",
    "output_dir": "output",
    "seed": 42,
}
```

## 11. 依赖

**MVP pip 依赖**（全部可 pure pip，无 brew 依赖）:
```
pillow
numpy
scikit-image          # rgb2lab / lab2rgb
faiss-cpu
tqdm
ipywidgets
matplotlib
deepzoom              # pure-python DZI 切片
```

V2 追加 `open_clip_torch` + `torch`（MVP 不安装）。

## 12. 错误处理（玩具级）

- 底图池扫描遇到坏图 / 非图片文件：`try/except`，skip，收集到 `errors` 列表，扫描完后 print "skipped N files"
- 空池 / 目标图不存在：直接 `raise FileNotFoundError` 让 cell 崩
- faiss 异常 / 渲染异常：不吞，直接 raise
- 原则：**只在边界（文件 IO）做容错，内部逻辑不加防御**

## 13. 测试

**轻量测试**，不追求覆盖率：

### `tests/test_match.py`
- 造 4×4 合成目标（纯色 quadrant） + 16 张纯色 tile
- 断言 assignment 每格都挑到最近颜色的 tile
- 断言 λ=0 时同一 tile 可被复用；λ 足够大时不复用

### `tests/test_transfer.py`
- τ=0：`np.allclose(out, tile_orig)`
- τ=1：`out` 的 LAB 均值 ≈ target LAB 均值（容差 1.0）

### `tests/test_pool.py`
- 扫一个临时目录，pickle 命中
- 改一个文件的 mtime，再扫一次，那个文件被重算（其它被 skip）
- 删一个文件，再扫一次，缓存里对应条目消失

### 不测
- `render.py`（IO + PIL，肉眼验）
- `zoom.py`（IO + 第三方库，肉眼验 html 能打开）
- `report.py`（模板字符串，肉眼验）

## 14. 实施顺序建议（给 writing-plans 的 hint）

1. `pool.py` + `test_pool.py`（基础，无依赖）
2. `target.py`（简单分网格）
3. `match.py` + `test_match.py`（核心算法）
4. `transfer.py` + `test_transfer.py`
5. `render.py`（组装）
6. `report.py`（模板 + matplotlib）
7. `zoom.py`（第三方库封装）
8. `config.py` + `mosaic.ipynb`（最后组装所有 cell）

每一步都应能独立 import 并在 REPL 里试跑。

## 15. 未解决 / 延后

- **目标图默认比例**: spec 假设 16:9（120×68 ≈ 1.76）。如果用户给的目标图不是 16:9，MVP 做 center-crop + 提示。V2 可做智能 crop。
- **tile 宽高比**: MVP 假设底图可被 center-crop 成正方形。非常规比例（panoramic）直接当普通方图用。
- **性能上限**: 当池超过 ~50K 张时 pickle 加载变慢，届时考虑换 sqlite 或 npy + jsonl 索引。MVP 不处理。
