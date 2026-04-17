# Bead Mosaic — Phase 1 Design

**Date:** 2026-04-17
**Scope:** 核心闭环。其他功能分 Phase 2–6 独立 spec。
**Positioning:** 本地玩具（ipynb），不是 SaaS。速度/普适性/稳定性不重要；可解释性和"可迭代性"重要。

---

## 1. 目标

一个 Jupyter notebook，输入 1 张目标图 + 1 个底图目录，输出一张 photomosaic PNG。

**完成定义 (DoD)：**

1. 在任何未配置过的 Mac（Apple Silicon，Python 3.12）上，clone 本仓库后直接打开 notebook，全部 cell 从上到下执行无报错，最后 cell 显示一张 photomosaic。
2. 零配置模式下使用内置演示数据：目标图 = `skimage.data.astronaut()`，底图池 = 运行时合成的 500 张色块。
3. 用户只需在 Cell 2 的 `CONFIG` dict 里改两个路径即可换成自己的目标图 + 照片目录，其余无需改动。
4. 底图扫描支持**断点续扫**：中断后重跑只扫新增/变更的文件。
5. smoke test `tests/test_pipeline.py` 通过。

**非目标（留给后续 Phase）：**

- 任何惩罚项（λ 重复、μ 邻居）→ Phase 2
- 色调迁移（τ）→ Phase 2
- 实时"思考"可视化 / 自嘲报告 / 冷宫照片墙 → Phase 3
- DeepZoom HTML 导出 → Phase 4
- CLIP 语义匹配 → Phase 5
- 彩蛋模式（Cursed / TimeCapsule / WeChat）/ saliency mask / Gradio UI → Phase 6

---

## 2. 关键技术决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 网格 | 120 × 68 (16:9) | 沿用 plan.md |
| Tile 像素 | 24 px | 16 近看偏糊；24 是甜区 |
| 输出尺寸 | 2880 × 1632 | 120·24 × 68·24 |
| 色彩空间 | LAB (D65) | 感知均匀 |
| 距离度量 | ΔE76（LAB 欧氏） | 足够简单；ΔE2000 留 Phase 5 |
| 匹配算法 | numpy 暴力最近邻 | ≤ 5k 张底图无压力；faiss 只在需要时引入 |
| 是否允许重复贴图 | 允许 | λ 惩罚属 Phase 2 |
| Tile 形状 | 正方形 | 简化 |
| UI | 纯 Jupyter cell | Gradio/Streamlit 属 Phase 6 |
| EXIF 方向 | 读图时 `ImageOps.exif_transpose` normalize | 否则横竖图出乱 |
| HEIC 支持 | `pillow-heif` 注册 opener | iPhone 导出友好 |
| 目标图长宽比不匹配 | letterbox 留黑边 | 保留用户原图构图；不 crop |
| 缓存失效键 | (文件路径, mtime, TILE_PX) 的哈希 | 改 TILE_PX 会触发重算 |

---

## 3. 依赖

```
pillow>=10
numpy>=1.26
scikit-image>=0.22
tqdm>=4.66
pillow-heif>=0.15
```

全部在 Apple Silicon 上秒装。不引入 torch / faiss / CLIP（属 Phase 5）。

---

## 4. 文件结构

```
claude/
├── plan.md
├── bead_mosaic.ipynb            # 主 notebook（仅负责编排 + 显示）
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── scan.py                  # 扫描底图池 + LAB 均值 + 缩略图缓存
│   ├── match.py                 # 最近邻匹配
│   └── render.py                # 拼接渲染
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py         # 纯色 smoke test
├── docs/superpowers/specs/      # 本 spec 文件所在
├── .gitignore
└── .cache/                      # gitignored
    ├── pool.npz                 # N 条 (lab[3], path_hash, mtime)
    └── thumbs/
        └── <sha1>.jpg           # 24×24 缩略图
```

**为什么拆 src：** Phase 2–6 需要往每个模块加新能力（match 加 λ/μ 惩罚，render 加色调迁移，scan 加 CLIP embedding）。拆成文件后增量改干净；notebook 只做编排，后续 Phase 改 notebook 的量最小。

---

## 5. 模块接口

### 5.1 `src/scan.py`

```python
@dataclass
class TilePool:
    lab: np.ndarray          # (N, 3) float32, LAB 均值
    thumbs_paths: list[str]  # N 个缩略图文件路径
    source_paths: list[str]  # N 个原图路径（用于 Phase 3 报告）

def build_pool(
    base_dir: Path,
    cache_dir: Path,
    tile_px: int,
    demo_mode: bool = False,
) -> TilePool: ...
```

**行为：**

1. 若 `demo_mode=True` 或 `base_dir` 不存在/为空 → 合成 500 张 24×24 色块（HSV 均匀采样），写入缓存，返回 TilePool。
2. 否则递归扫 `base_dir`，支持扩展名：`.jpg/.jpeg/.png/.webp/.heic`。
3. 每张图：
   - 读 + `exif_transpose` normalize
   - 中心 crop 到方形
   - 缩放到 `tile_px × tile_px`
   - 转 LAB，取均值（3 个 float）
   - 保存缩略图到 `cache_dir/thumbs/<sha1(source_path)>.jpg`
4. 断点续扫：持久化一个 `cache_dir/pool.npz`，下次启动时读入，只扫新增/变更文件（基于 mtime + 路径哈希）。
5. 坏图（解码失败/尺寸 0）→ log warning，跳过，不中断。

**依赖的外部状态：** 文件系统（`base_dir` 是只读，`cache_dir` 可写）。

### 5.2 `src/match.py`

```python
def match_grid(
    target_lab: np.ndarray,  # (H, W, 3)
    pool_lab: np.ndarray,    # (N, 3)
) -> np.ndarray:             # (H, W) int32, 每格对应的 pool index
```

**行为：** 对每格 LAB 与 pool_lab 算 L2 欧氏距离，取 argmin。纯 numpy broadcast，无副作用。

### 5.3 `src/render.py`

```python
def render_mosaic(
    index_grid: np.ndarray,    # (H, W) int32
    pool: TilePool,
    tile_px: int,
    output_path: Path,
) -> Image.Image: ...          # 返回 PIL Image，同时写文件
```

**行为：** 分配 `(H·tile_px, W·tile_px, 3)` uint8 画布；按 index 读 thumbs_paths 对应缩略图贴进去；保存 PNG；返回 PIL Image 方便 notebook 内联显示。

### 5.4 notebook cell 结构

| # | 名称 | 内容 |
|---|---|---|
| 1 | 安装 + import | `%pip install -r requirements.txt` + 模块 import |
| 2 | 配置 | `CONFIG = {...}` dict；注释标清"改这两行"|
| 3 | 扫描底图池 | `pool = scan.build_pool(...)` |
| 4 | 目标图分块 | 读 target → exif_transpose → letterbox resize → reshape → LAB 均值 → `target_lab_grid` |
| 5 | 匹配 | `idx = match.match_grid(target_lab_grid, pool.lab)` |
| 6 | 渲染 | `img = render.render_mosaic(idx, pool, TILE_PX, output_path)` |
| 7 | 显示 | `display(img)` + 打印 `tile usage dict`（为 Phase 3 报告埋接口） |

---

## 6. 数据流

```
BASE_DIR ─┐
          ├→ scan.build_pool ──→ TilePool(lab, thumbs_paths)
          │                          │
  cache ──┘                          ├→ match.match_grid ─→ index_grid
                                     │         ↑                │
TARGET_PATH → 分块 → target_lab_grid ┘         │                │
                                               │                ↓
                                       render.render_mosaic → output.png
```

---

## 7. 测试策略

### 7.1 Smoke test — `tests/test_pipeline.py`

合成 10 种纯色的底图 tiles + 一张 10 色块拼成的目标图 → 跑完整 `scan → match → render` → 断言每格匹配到同色底图（LAB 距离 < 1）。这个测试守住"匹配算法不瞎"的底线。

### 7.2 手动验证

- **demo 模式**：空目录 → notebook 全跑 → 出 astronaut 的色块 mosaic
- **真实照片**：切换到用户自己的目录 → 全跑一次；肉眼看像不像

Phase 1 **不做** UI 自动化测试 / 性能 benchmark / 大规模数据集测试。

---

## 8. 已考虑的风险和边界

| 风险 | 应对 |
|---|---|
| 用户相册很大（5 万+） | Phase 1 仍用 numpy 暴力；若实际跑觉得慢，Phase 2 再上 faiss。spec 明确不优化。 |
| HEIC 文件 | `pillow-heif.register_heif_opener()` 在 `__init__.py` 里调一次 |
| EXIF 方向 | 每次读图统一 `exif_transpose` |
| 坏图 / 零字节文件 | `try/except` 包裹，log warning 跳过 |
| 极端长宽比目标图 | letterbox 留黑边（不 crop） |
| 用户改了 TILE_PX 后缓存过期 | 缓存 key 含 tile_px；mismatch 触发重扫 |
| 底图池 < GRID_W·GRID_H | 允许重复贴图（Phase 1 允许）→ 自然解决 |
| 缓存目录损坏 | scan 启动时校验 npz schema；失败则删除重建 |

---

## 9. 面向 Phase 2–6 的扩展接口预埋

| Phase | 需要的扩展点 | Phase 1 如何预埋 |
|---|---|---|
| 2 | λ 重复惩罚 | `match_grid` 调用点传 `None`，签名后续加可选参数 |
| 2 | μ 邻居惩罚 | 同上 |
| 2 | τ 色调迁移 | `render_mosaic` 签名后续加 `tone_strength` 可选参数，默认 0 = 不迁移 |
| 3 | 自嘲报告 / 冷宫照片墙 | render 返回 `tile_usage: dict[int, int]`；notebook Cell 7 已打印 |
| 4 | DeepZoom | render 的 PIL Image 本身即可喂给 pyvips，不需要改接口 |
| 5 | CLIP 语义匹配 | `TilePool` 可加 `clip_emb: np.ndarray \| None`；`match_grid` 签名加 `pool_clip/target_clip/weight` 可选参数 |

Phase 1 **不提前实现** 任何这些参数，只保证加的时候不需要翻修接口。

---

## 10. 变更记录

项目第一次代码改动时创建 `CHANGELOG.md`，按用户约定（面向 agent 的 YAML 格式）维护。本 spec 文件本身不进 changelog。
