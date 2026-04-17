# Photomosaic Toy — Design Spec

Date: 2026-04-17
Status: approved for implementation planning
Owner: joejiang

## 1. 定位

一个本地跑的 ipynb 玩具,把硬盘里的照片重组成另一张照片。

明确不是:SaaS、API、Etsy 店、作品集项目。定位决定了以下反直觉选择:
- 速度不重要,跑 10 分钟出一张可以接受
- 普适性不重要,能在 owner 的 Mac 上跑就够
- 稳定性不重要,崩了重跑
- **可解释性很重要**——"看它怎么思考的"是玩具乐趣的核心

## 2. 目标与非目标

### MVP 目标

1. 本地 ipynb run-all 跑通:输入目标图 + 底图目录 → 输出拼贴 PNG + DeepZoom 可缩放 HTML + 自嘲式报告。
2. 可调的三滑条(实时改,不需要重跑上游):
   - λ 重复惩罚:`score += λ · log(1 + usage_count)`
   - μ 邻居惩罚:`score += μ · neighbor_similarity(选中 tile, 已填邻居)`
   - τ 色调迁移强度:Reinhard LAB 均值迁移,τ∈[0,1] 与原图混合
3. 零配置可跑:
   - 底图目录为空或不存在 → 自动生成 200 张 64×64 合成色块 tile
   - 目标图缺失 → 用内置渐变图
4. 算法过程可观察:逐格 `print` 选中 tile 路径 + 色差 + 惩罚项数值(tqdm 节流,每 50 格 print 一次理由)
5. 自嘲式报告含:冷宫照片 TOP N 缩略图墙、使用次数柱状图、目录分布百分比、总耗时、坏图列表
6. DeepZoom 产物:`out/deepzoom/index.html` + OpenSeadragon CDN + tiles 金字塔,浏览器打开即可无限缩放

### 非目标(v2 再议,本版本绝不做)

- CLIP 语义匹配
- 人脸识别 / 人脸聚类("前三人"那句不做)
- Cursed mode 预设("用 500 张吃的拼体检报告")
- Pool 标签叙事("2019 日本旅行 23%")
- Gradio / Streamlit Web UI
- 跨平台(Windows/Linux)/ GPU / 分布式
- 性能优化到秒级

## 3. 架构(方案 II:Notebook + 薄 helper)

```
Tile.Photo/chatgpt/
├── plan.md                    # 已有的产品讨论
├── mosaic.ipynb               # 8 个 cell 的"剧本",只做配置/可视化/报告
├── mosaic_core.py             # 纯函数 helper,pytest 覆盖
├── tests/
│   ├── test_color.py
│   ├── test_matching.py
│   └── test_transfer.py
├── requirements.txt
├── .cache/                    # tile LAB embedding pickle,.gitignore
├── out/                       # 渲染结果,.gitignore
│   ├── mosaic.png
│   ├── report.txt
│   └── deepzoom/
├── CHANGELOG.md               # agent-friendly 规范
├── CHANGELOG.archive.md
└── .gitignore
```

### 技术栈

- Python 3.11+(owner Mac 现成)
- `pillow`, `numpy`, `scipy`, `scikit-image`(LAB 色差 / CIEDE2000)
- `faiss-cpu`(top-k 最近邻)
- `tqdm`, `ipywidgets`, `matplotlib`
- `deepzoom`(纯 pip,不用 pyvips,不 `brew install`)
- `pytest`(测试)

所有依赖 `pip install` 即可,零 native 编译、零 brew。

## 4. 模块设计(`mosaic_core.py`)

每个函数单一职责,边界清晰:

| 函数 | 签名 | 纯函数 | 说明 |
|---|---|---|---|
| `scan_tile_pool` | `(dir: Path, cache_path: Path) -> list[TileRecord]` | IO,确定性 | 递归扫 JPG/PNG,读 LAB 均值 + 64×64 缩略图,走/建 pickle 缓存 |
| `ensure_seed_tiles` | `(dir: Path, n: int = 200) -> None` | IO | 若 dir 不存在或为空,生成 n 张 64×64 随机 HSV 色块 |
| `split_target` | `(img: PIL.Image, grid_w: int, grid_h: int) -> np.ndarray[H, W, 3]` | 纯 | 切 LAB 空间的 patch 均值 |
| `build_faiss_index` | `(tile_labs: np.ndarray) -> faiss.Index` | 纯 | L2 平面索引(底图量级 < 10k,不需要 IVF) |
| `knn_candidates` | `(target_lab: np.ndarray, faiss_index, k: int = 32) -> np.ndarray[H·W, k]` | 纯 | 每个 patch 的 top-k tile index |
| `rerank` | `(candidate_idxs, tile_labs, target_lab_patch, usage_counts, neighbor_tile_idxs: list[int], λ, μ) -> int` | 纯 | 在 top-k 里 `score = ΔE_CIEDE2000 + λ·log(1+usage) + μ·max(similarity to each neighbor tile)`,返回最小 score 的 tile idx。`neighbor_tile_idxs` 是当前格左/上两个已填格的 tile index(扫描线顺序,边界格邻居列表可为空/单元素) |
| `reinhard_transfer` | `(tile_rgb, target_lab_mean, τ) -> rgb` | 纯 | LAB 空间均值迁移后按 τ 混合 |
| `render_mosaic` | `(assignment, tile_records, tile_px, τ, target_lab) -> PIL.Image` | 纯 | 按 assignment 拼大图,每块可选做色调迁移 |
| `build_report` | `(assignment, tile_records, elapsed, bad_files) -> ReportBundle` | 纯 | 返回 `(text, usage_bar_fig, cold_wall_fig)` |
| `export_deepzoom` | `(png_path: Path, out_dir: Path) -> None` | IO | 调 `deepzoom.ImageCreator` 切金字塔,写 `index.html`(带 OpenSeadragon CDN) |

### 数据结构

```python
@dataclass
class TileRecord:
    path: Path
    lab_mean: np.ndarray  # float32[3]
    rgb_thumb: np.ndarray  # uint8[64, 64, 3]

@dataclass
class MosaicConfig:
    target_path: Path | None
    tile_dir: Path
    grid_w: int = 120
    grid_h: int = 68       # 16:9 ~= 1080p 档
    tile_px: int = 16
    k_candidates: int = 32
    lambda_repeat: float = 0.5
    mu_neighbor: float = 0.3
    tau_transfer: float = 0.4
    cache_path: Path = Path(".cache/tiles.pkl")
    out_dir: Path = Path("out")

@dataclass
class ReportBundle:
    text: str
    usage_bar_fig: matplotlib.figure.Figure
    cold_wall_fig: matplotlib.figure.Figure
```

## 5. 数据流

```
[底图目录]                           [目标图]
      │                                   │
  ensure_seed_tiles                  (fallback 渐变图)
      │                                   │
  scan_tile_pool                    split_target
      │                                   │
  [list[TileRecord]]                [grid H×W LAB patches]
      │                                   │
  build_faiss_index ────────┐             │
                            ▼             ▼
                      knn_candidates  (top-32)
                            │
                            ▼
                    逐格 rerank 循环(扫描线顺序 top-left → bottom-right)
                    (usage_counts 累积;neighbor = 左邻 + 上邻已填 tile)
                            │
                            ▼
                  assignment: ndarray[H, W] → tile_idx
                            │
           ┌────────────────┼─────────────────┐
           ▼                ▼                 ▼
    render_mosaic    build_report     export_deepzoom
      (+τ transfer)                   (deepzoom.py)
           │                ▼                 │
           ▼          report 文本 +           ▼
       mosaic.png     使用柱状图 +          deepzoom/index.html
                      冷宫墙图
```

## 6. Notebook Cell 设计

8 个 cell,每个都薄,保留"剧本感":

| # | 内容 | 预期行数 |
|---|---|---|
| 1 | `%pip install -r requirements.txt` + imports + 设随机种子 | ~10 |
| 2 | 定义 `MosaicConfig`,创建 `ipywidgets` 滑条(λ/μ/τ/grid_w/grid_h)+ 路径输入,显示 widgets | ~30 |
| 3 | `ensure_seed_tiles(config.tile_dir)` → `scan_tile_pool(...)` → print `"扫到 N 张 tile,缓存命中 X 张"` | ~8 |
| 4 | 读目标图(缺失用渐变)→ `split_target` → 画网格 overlay 预览 | ~12 |
| 5 | `build_faiss_index` → `knn_candidates` → 逐格 `rerank` 循环(tqdm;每 50 格 print 一次选中理由) | ~25 |
| 6 | `render_mosaic(..., τ=config.tau_transfer)` → `display(img)` | ~6 |
| 7 | `build_report(...)` → `display(text)` + `plt.show(bar_fig)` + `plt.show(cold_wall_fig)` | ~10 |
| 8 | `export_deepzoom(...)` → print `"✅ 已生成 out/deepzoom/index.html,在浏览器打开即可无限缩放"` | ~6 |

## 7. 错误处理

玩具定位 → **不写防御性代码,只写清楚的边界行为**:

| 场景 | 行为 |
|---|---|
| 底图目录不存在或为空 | `ensure_seed_tiles` 自动生成 200 张合成 tile(兜底) |
| 目标图路径为 None / 文件不存在 | 用内置 768×432 渐变图 |
| 单张 tile 读取失败(损坏 JPG)| skip + 加入 `bad_files` 列表,报告里列出,不中断 |
| 缓存 pickle 版本不匹配 | 删缓存重算,不做 migration |
| `grid_w * grid_h > len(tiles)` | 允许重复,用 λ 惩罚自然摊开;tile 不足 32 张时 k 降级 |
| faiss 查询失败 | 让异常抛,不吞 |

## 8. 测试策略

**不测 notebook / UI / IO**,只测 `mosaic_core.py` 的纯函数。约 15 个 case:

### `test_color.py`
- `lab_mean` 对纯红图像返回已知值
- `ciede2000(a, a) == 0`
- `reinhard_transfer(rgb, target_mean, τ=0)` 返回原图(误差 < 1e-6)
- `reinhard_transfer(rgb, target_mean, τ=1)` 图像均值等于 target_mean

### `test_matching.py`
- `rerank(λ=0, μ=0)` 退化为最小 ΔE 选择
- 增大 λ 后,频繁使用的 tile 被惩罚,选择改变
- 增大 μ 后,与邻居高度相似的 tile 被惩罚
- `knn_candidates` 返回 shape `(H*W, k)`,且索引合法
- `build_faiss_index` 对 100 个 tile 建索引后查询 top-5 返回确定值

### `test_transfer.py`
- `split_target(img, 10, 5)` 返回 shape `(5, 10, 3)`
- `render_mosaic` 输出尺寸 `= (grid_w * tile_px, grid_h * tile_px)`
- τ=0 时 `render_mosaic` 输出与 tile 原色 byte-exact
- 坏图在 `scan_tile_pool` 中被归入 `bad_files` 而不崩

## 9. 依赖与打包

`requirements.txt`:

```
pillow>=10
numpy>=1.26
scipy>=1.11
scikit-image>=0.22
faiss-cpu>=1.7
tqdm>=4.66
jupyter>=1.0
ipywidgets>=8.1
matplotlib>=3.8
deepzoom>=0.2
pytest>=7.4
```

无 native 依赖,`pip install -r requirements.txt` 一把梭。

## 10. CHANGELOG 约定

遵循 owner 的 agent-friendly 规范(已存入 memory):
- YAML 字段:`date / type / target / change / rationale / action / result / validation / status`
- try-failed 链条是金子,只压缩不删除
- 50 条或 6 个月触发压缩到 `CHANGELOG.archive.md`
- 禁止抽象措辞、相对时间、省略 rationale

首条条目由实现阶段初始化提交时写入。

## 11. 验收

MVP 完成的标志:
1. 在 owner 的 Mac 上 clone 后 `pip install -r requirements.txt` + `jupyter lab mosaic.ipynb` + 点 Run All → 不崩、出 `out/mosaic.png` + `out/deepzoom/index.html` + 控制台报告
2. `pytest tests/` 全绿
3. 替换底图目录为真实相册路径后,重跑仍能出图
4. 调 λ/μ/τ 滑条重新触发 Cell 5/6 后,结果可见变化
5. 浏览器打开 `out/deepzoom/index.html`,可无限放大到单张底图清晰可辨

## 12. 未来扩展(不在本次范围)

- v2:CLIP 语义匹配(加 `open_clip_torch`,用 faiss 语义索引与颜色分分数加权)
- v2:人脸聚类 + 报告里"前三人"
- v2:Cursed mode 预设 + Pool 标签叙事
- v3:Gradio 分享版
