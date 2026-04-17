# 照片马赛克玩具 — 设计稿

- **Date:** 2026-04-17
- **Author:** joejiang (设计由 Claude 协同)
- **Status:** Draft → awaiting review
- **Target Location:** `Tile.Photo/doubao/`
- **Architecture:** Thin library + Jupyter notebook (方案 2)

---

## 1. 定位（不可妥协）

本项目是**一个本地跑的玩具**，不是 SaaS / API / 产品 / 作品集。定位决定了以下反直觉的选择：

- **速度不重要** —— 跑 10 分钟出一张可接受
- **普适性不重要** —— 在 joejiang 的 Mac 上跑就行
- **稳定性不重要** —— 崩了重跑
- **可解释性重要** —— 好玩的本质是"看它怎么思考"
- **彩蛋比功能值钱** —— MVP 不做，但接口预留

任何设计分歧回到"对玩具有利 vs 对产品有利"时，选玩具。

---

## 2. MVP 范围

MVP = 8 个 cell 骨架全部：

| Cell | 职责 |
|------|------|
| a | 扫描底图池 + LAB 均色索引（带缓存） |
| b | 目标图分块 + 每格均色 |
| c | 色调迁移（Reinhard，参数 τ） |
| d | 重复惩罚（参数 λ） |
| e | 邻居惩罚（参数 μ） |
| f | 渲染 + 保存 PNG |
| g | 自嘲式报告（使用次数 / 冷宫 TOP5 / 柱状图 / 冷宫墙） |
| h | DeepZoom HTML 导出（OpenSeadragon） |

**MVP 刻意不做，留作增量：**

- CLIP 语义匹配（`match.py` 可加分支）
- Cursed mode（"用 A 拼 B"、全表情包拼证件照等）
- Gradio / Streamlit 滑条 UI（MVP 用 ipynb 手调参数）
- 底图池标签叙事（"23% 来自 2019 日本旅行"）
- rembg / saliency 主体 mask
- 实时可视化动画（`verbose` print 当前格和挑中理由即算）

---

## 3. 目录结构

```
Tile.Photo/doubao/
├── plan.md                         # 既有研究文档（不动）
├── README.md                       # 怎么跑
├── CHANGELOG.md                    # agent-oriented 规范，Day 1 起维护
├── CHANGELOG.archive.md            # 压缩归档（初始为空）
├── pyproject.toml                  # uv 管理
├── .gitignore                      # 忽略 .venv/ .cache/ output/
├── photomosaic.ipynb               # 8-cell 编排
├── src/mosaic/
│   ├── __init__.py
│   ├── config.py                   # MosaicConfig dataclass
│   ├── tiles.py                    # 扫描 + LAB 均色 + pickle 缓存
│   ├── match.py                    # faiss + λ/μ 惩罚 + verbose 打印
│   ├── render.py                   # Reinhard 色调迁移 + 贴图
│   ├── report.py                   # 文字报告 + 柱状图 + 冷宫墙
│   └── deepzoom.py                 # DeepZoom 切片 + HTML 导出
├── tests/
│   ├── test_match.py
│   └── test_render.py
├── docs/superpowers/specs/
│   └── 2026-04-17-photomosaic-toy-design.md  # 本文档
├── .cache/                         # 运行时生成，git 忽略
└── output/                         # 运行时生成，git 忽略
```

每个模块目标 ≤ 200 行。超过说明职责混了，要拆。

---

## 4. 组件设计

### 4.1 `config.py`

单一入口 dataclass，所有参数集中一处：

```python
@dataclass
class MosaicConfig:
    # paths
    tile_source_dir: Path       # 底图目录（递归扫描）
    target_image: Path          # 目标图
    cache_dir: Path = Path(".cache")
    output_dir: Path = Path("output")

    # grid
    grid: tuple[int, int] = (120, 68)   # (cols, rows)
    tile_px: int = 16                    # 每 tile 渲染像素

    # matching
    candidate_k: int = 50       # faiss 先取 top-k 颜色候选
    lambda_reuse: float = 0.3   # 重复惩罚
    mu_neighbor: float = 0.2    # 邻居相似度惩罚

    # rendering
    tau_tone: float = 0.5       # 色调迁移强度 0..1

    # behavior
    verbose: bool = True        # 打印思考过程
    mode: str = "classic"       # 预留给 cursed_* 模式

    def validate(self) -> None:
        """运行前调用；路径不存在直接抛"""
```

**Why dataclass 而不是 argparse/dict：** notebook cell 2 里 `cfg = MosaicConfig(...)` 是最清晰的"所有旋钮都在这"入口，IDE 补全也友好。

### 4.2 `tiles.py`

```python
def scan_tile_pool(root: Path) -> list[Path]:
    """递归扫描 root 下所有支持的图像文件。
    跳过: .DS_Store, .app bundle, @eaDir（群晖缩略图）, 隐藏目录"""

def compute_tile_features(paths: list[Path], tile_px: int,
                          cache_dir: Path) -> TilePool:
    """对每张图: 等比缩放 + 中心裁剪到 tile_px²，计算 LAB 均色。
    缓存 key = (path, mtime, tile_px)，增量更新，损坏 pickle 警告后重建。"""

@dataclass
class TilePool:
    paths: list[Path]
    lab_means: np.ndarray       # shape (N, 3)
    thumbnails: list[np.ndarray]  # shape (N, tile_px, tile_px, 3) uint8
```

**缓存策略：** 单个 pickle `cache_dir/tiles_{tile_px}.pkl`，里面是 `{path_str: {mtime, lab_mean, thumb_bytes}}`。每次 `load_or_build` 先读 pickle → 对缺失或 mtime 改的重算 → 写回。

### 4.3 `match.py`

```python
def split_target(target: Path, grid: tuple[int, int]) -> np.ndarray:
    """返回 shape (rows, cols, 3) 的 LAB 均色数组"""

def build_faiss_index(tile_pool: TilePool) -> faiss.IndexFlatL2:
    """L2 索引 on lab_means"""

def match_all_tiles(target_cells: np.ndarray,
                    tile_pool: TilePool,
                    faiss_index,
                    cfg: MosaicConfig) -> tuple[np.ndarray, dict]:
    """
    行优先遍历每格，对每格：
      1. faiss 取 candidate_k 颜色候选
      2. 对每候选算 score = color_dist
                          + λ_reuse * log(1 + use_count[cand])
                          + μ_neighbor * avg_lab_dist(cand, 紧邻左/上 tile)
         注: 邻居只看当前行左一格 + 上一行同列一格（行优先遍历保证已放置），
             缺邻居（第一行/第一列）不参与该项。
      3. 选最低分
      4. 更新 use_count；记到 assignment[row][col] = cand_idx
      5. verbose=True 时 print(f"[{r:3d},{c:3d}] -> tile#{idx} "
                                f"dist={d:.1f} reuse={u} neigh={n:.1f}")
    返回 (assignment: (rows,cols) int, use_count: dict[int,int])
    """
```

**为什么是暴力 O(N_cells × K) 而不是更聪明的全局优化？**
因为玩具。N_cells = 120×68 = 8160，K=50，一次循环 40 万次评分，纯 numpy 秒出。Hungarian / MIP 会让项目从周末项目变成月度项目。

### 4.4 `render.py`

```python
def reinhard_transfer(tile_bgr: np.ndarray,
                      target_patch_lab: np.ndarray,
                      tau: float) -> np.ndarray:
    """Reinhard LAB 均值+标准差迁移，强度 τ∈[0,1] 线性混合原色与迁移色。
    τ=0 完全原色（近看照片清晰），τ=1 完全贴合（远看完美，近看被染色）。"""

def render_mosaic(assignment: np.ndarray,
                  tile_pool: TilePool,
                  target_cells: np.ndarray,
                  cfg: MosaicConfig) -> PIL.Image.Image:
    """新建 (cols*tile_px, rows*tile_px) 画布。
    对每格: 取 tile 缩略图 → 可选 reinhard_transfer(τ) → paste。
    返回最终 Image。"""
```

### 4.5 `report.py`

```python
def generate_text_report(use_count: dict, tile_pool: TilePool,
                         total_cells: int) -> str:
    """自嘲式文本，示例:
    '本次使用了你 3,241 张照片里的 847 张。
     其中 IMG_2019_03_14_217.jpg 被用了 89 次（主要用于填充天空）。
     冷宫照片 TOP 5 是：
       1. selfie_blur_2020_01_02.jpg (0 次)
       ...'"""

def plot_usage_histogram(use_count: dict) -> matplotlib.figure.Figure:
    """使用次数直方图"""

def build_cold_wall(tile_pool: TilePool, use_count: dict,
                    max_shown: int = 64) -> PIL.Image.Image:
    """冷宫照片缩略图墙（未被使用的 tiles）"""
```

### 4.6 `deepzoom.py`

```python
def export_deepzoom(mosaic: PIL.Image.Image,
                    output_dir: Path) -> Path:
    """用 `deepzoom` 包切金字塔 → output_dir/tiles/
    生成 output_dir/index.html，用 CDN 上的 OpenSeadragon 加载。
    返回 index.html 的绝对路径。"""
```

**为什么不用 pyvips：** pyvips 需要 brew 装 libvips，装机体验差。`deepzoom` 纯 Python 慢一点但零额外系统依赖 —— 玩具优先的选择。

---

## 5. 数据流

```
tile_source_dir ──scan──> paths ──features──> TilePool
                                               │      （缓存到 .cache/）
                                               ▼
target_image ──split_target──> target_cells (LAB 均色网格)
                                               │
                          ┌────────────────────┤
                          ▼                    ▼
                   faiss_index         match_all_tiles
                                          │   │
                                  assignment   use_count
                                          │   │
                                          ▼   ▼
                                     render_mosaic (+tone transfer)
                                               │
                                     PIL.Image (final mosaic)
                                      │    │       │
              ┌───────────────────────┘    │       └───────────────┐
              ▼                            ▼                       ▼
     output/mosaic.png          report: text + histogram    deepzoom/index.html
                                         + cold wall        + tiles/_files/
```

---

## 6. 8-cell notebook 编排

每个 cell 目标 ≤ 15 行，主要是 import + 调用：

```
Cell 1  环境自检 + sys.path 注入 src/
Cell 2  cfg = MosaicConfig(
            tile_source_dir=Path("<待填>"),
            target_image=Path("<待填>"),
            grid=(120, 68), tile_px=16,
            lambda_reuse=0.3, mu_neighbor=0.2, tau_tone=0.5,
            verbose=True,
        ); cfg.validate()
Cell 3  tile_pool = tiles.load_or_build(cfg)
        print(f"loaded {len(tile_pool.paths)} tiles")
Cell 4  target_cells = match.split_target(cfg.target_image, cfg.grid)
Cell 5  faiss_index = match.build_faiss_index(tile_pool)
        assignment, use_count = match.match_all_tiles(
            target_cells, tile_pool, faiss_index, cfg)
Cell 6  mosaic = render.render_mosaic(assignment, tile_pool, target_cells, cfg)
        mosaic.save(cfg.output_dir / "mosaic.png")
        mosaic
Cell 7  print(report.generate_text_report(use_count, tile_pool,
                                          total_cells=cfg.grid[0]*cfg.grid[1]))
        report.plot_usage_histogram(use_count)
        report.build_cold_wall(tile_pool, use_count).show()
Cell 8  html = deepzoom.export_deepzoom(mosaic, cfg.output_dir)
        print(f"open: file://{html}")
```

---

## 7. 错误处理（玩具级粗糙）

- **损坏底图**：`try/except` + `warnings.warn` + 跳过，不打断主流程
- **pickle 缓存损坏**：warn + 重建
- **路径不存在**：`cfg.validate()` 抛 `ValueError("please set tile_source_dir in cell 2")`
- **其他异常**：让它崩，用户重跑

**刻意不做：** retry / log 文件 / 异常上报 / 优雅降级。

---

## 8. 测试策略（最小）

### 8.1 包含
- **`tests/test_match.py`**
  - 5 张纯色底图 + 10 格目标：验证无惩罚（λ=μ=0）时选颜色最近
  - λ>0 时不再全选同一张
  - μ>0 时相邻格更多样
- **`tests/test_render.py`**
  - τ=0：输出 pixel 与 tile 原图一致
  - τ=1：输出 LAB 均值与 target patch 一致

### 8.2 不测
- `tiles.py`：IO 密集，fixture 维护成本高
- `deepzoom.py`：输出是 HTML + 目录 png，肉眼验证
- `report.py`：纯字符串/图，肉眼验证

跑命令：`cd doubao && uv run pytest`

---

## 9. 依赖

### 9.1 运行时（`pyproject.toml`）

```
pillow
numpy
scikit-image        # LAB 转换
faiss-cpu           # ANN 索引
tqdm                # 进度条
matplotlib          # 报告图
deepzoom            # DeepZoom 切片（零系统依赖）
ipykernel           # notebook 运行
```

### 9.2 开发
```
pytest
```

### 9.3 刻意不装（留给增量）
- `open_clip_torch` / `torch` —— CLIP 语义匹配（体积大，MVP 不需要）
- `rembg` —— 主体 mask
- `gradio` —— 滑条 UI
- `pyvips` —— 可选 DeepZoom 加速（需 brew libvips）

---

## 10. 未来增量接口（MVP 先留好，不实现）

- **CLIP 语义**：`match.py` 的打分函数拆成 `score(candidate, context) -> float`，新加一项 `ν * (1 - cos_sim(clip_vec, target_patch_clip))`
- **Cursed mode**：`MosaicConfig.mode = "cursed_pool_swap"` 等，`match.py` 里 if 分支
- **Gradio**：所有模块已是纯函数，直接包一层 callback
- **底图标签**：`TilePool` 加 `tags: dict[Path, list[str]]` 字段，`report.py` 读
- **实时可视化**：把 `match_all_tiles` 改成 generator，cell 5 外面包 tqdm + matplotlib 动画

---

## 11. CHANGELOG 策略

按用户定的 agent-oriented 规范（详见 feedback memory）。

Day 1 第一条：
```yaml
- date: 2026-04-17
  type: feat
  target: doubao/
  change: 初始化照片马赛克 MVP 骨架（8 cell notebook + src/mosaic/ 5 模块）
  rationale: 用户明确玩具定位，8-cell 结构是 MVP 底线；模块化是为 CLIP / cursed / Gradio 增量做准备，不是产品化
  action: 新建目录结构、pyproject.toml、config/tiles/match/render/report/deepzoom 六模块骨架、photomosaic.ipynb 8 个 cell、tests 两个文件
  result: uv sync + pytest 通过，notebook cell 1 可运行
  validation: pytest tests/ 通过；notebook cell 1 自检通过
  status: experimental
```

后续每改一个模块都追加条目。压缩阈值 50 条或 6 个月。

---

## 12. 假设与已做决定

| 项 | 决定 | 可推翻的前提 |
|----|-----|-------------|
| 硬件 | M 系列 Mac | CPU faiss 已够用，不上 CUDA |
| 环境 | `uv` 创建 `.venv` | 若没装 uv，README 写 pip 备选 |
| 底图规模 | 500–5k 张 | 超过 20k 需换索引策略（IVF / OPQ） |
| 首跑素材 | 用户填路径再跑 | MVP 不绑定具体数据集 |
| 目标图尺寸 | 120×68 × 16px ≈ 1920×1088 | 配置即可改 |
| 主体 mask | 不做 | 增量加 |
| UI | 纯 notebook | Gradio 作为增量 |

---

## 13. Open Questions（spec 阶段暂缓，实现时回来看）

- `deepzoom` pypi 包有多老？是否仍维护？—— 不维护就换 `pyvips` 或自写 DZI XML
- faiss-cpu 在 Mac arm64 是否 wheel 即装即用？—— uv sync 验证
- notebook cell 2 的路径默认值怎么给？占位符 `Path("/path/to/photos")` 触发 validate 报错，提示用户填。

---

## 14. Success Criteria

MVP 完成的判定：

1. `uv sync` 成功，无手动配置
2. `pytest tests/` 全绿
3. 用户填路径后，`photomosaic.ipynb` 从 cell 1 跑到 cell 8 不报错
4. Cell 5 的 verbose 输出能看到"思考过程"（每格打印一行）
5. `output/mosaic.png` 肉眼像目标图的马赛克
6. `output/index.html` 用浏览器打开能无限缩放，放到底能看清单张小图
7. Cell 7 文字报告含"用了 X 张里的 Y 张，冷宫 TOP5"
8. CHANGELOG.md 第一条已写好

任一不满足：MVP 未完成。
