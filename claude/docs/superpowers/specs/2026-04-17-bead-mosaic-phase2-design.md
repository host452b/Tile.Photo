# Bead Mosaic — Phase 2 Design (λ/μ/τ)

**Date:** 2026-04-17
**Scope:** 在 Phase 1 核心管线上加三个"甜区"旋钮：λ 重复惩罚、μ 邻居惩罚、τ 色调迁移。不碰 CLIP、DeepZoom、报告、Gradio、彩蛋模式（属 Phase 3+）。
**Prerequisite:** Phase 1 已落地，12 个测试全绿，接口为 Phase 2 预埋过。

---

## 1. 目标

三个 float 旋钮（都在 [0, 1] 或类似尺度）让用户在 notebook 里交互式地调节 mosaic 的"效果甜区"。

**完成定义 (DoD)：**

1. Phase 1 的 12 个测试继续全绿（λ=μ=τ=0 时行为 bitwise-equal Phase 1）。
2. Notebook 新增一个 ipywidgets UI cell，包含 3 个 FloatSlider + 一个"重跑"按钮；按钮点一次重跑 match+render。
3. `CONFIG['LAMBDA']` / `CONFIG['MU']` / `CONFIG['TAU']` 三个 key 作为初始值，默认全 0。
4. 新增测试覆盖 λ/μ/τ 各自的行为预期。
5. 端到端：λ=2, μ=10, τ=0.5 下 demo mode notebook 跑通且输出合理（肉眼查）。

**非目标（属 Phase 3+）：**

- CLIP 语义匹配、自嘲报告、冷宫照片墙、DeepZoom、Gradio、彩蛋模式、saliency mask
- Hungarian 或其他全局最优匹配（Phase 2 保持贪心）
- τ 的 mean+std Reinhard（Phase 2 只迁 mean）

---

## 2. 三个旋钮的语义

### 2.1 λ — 重复惩罚

**问题它解决**：无惩罚时一张"万能照"（平均色接近灰色/中性）会被反复贴到很多格子，视觉单调。

**惩罚项：** `λ · log(1 + uses[i])`

加在每个 candidate tile 的平方距离上。`log` 使得惩罚随使用次数增长但饱和——λ 小时效果柔和；λ 大时强制所有 tile 被用到。

**合理范围：** 0（无惩罚） … 200（强制多样性，可能牺牲色彩准确度）。默认 0。

### 2.2 μ — 邻居惩罚

**问题它解决**：某片大面积均匀色（天空、墙）容易连续贴同一 tile，近看很假。

**惩罚项：** 扫描到格子 (r, c) 时，检查 (r-1, c) 和 (r, c-1) 的已放置 index。对每一个 candidate tile i：
```
neighbor_penalty(i) = μ · (i == top_index) + μ · (i == left_index)
```

**为什么只看 top/left：** 扫描顺序是行优先（左→右，上→下），scan 到 (r, c) 时，top 和 left 已定；bottom 和 right 尚未扫到。4-connected 里这两个就是已知邻居。

**合理范围：** 0 … 1000（极强，几乎不允许邻居相同）。默认 0。

### 2.3 τ — 色调迁移（Reinhard mean-only）

**问题它解决**：即使最近邻匹配，tile 色调通常不完全等于目标 patch，贴上去后远观有小"色斑"。

**算法：** 每张 tile 在贴到画布之前，做一次向目标 patch 的 LAB 均值迁移：

```
tile_lab = rgb2lab(tile_rgb)
delta = (target_patch_lab_mean - tile_lab.mean()) * tau
tile_lab_adjusted = tile_lab + delta
tile_rgb_adjusted = np.clip(lab2rgb(tile_lab_adjusted) * 255, 0, 255).astype(uint8)
```

只迁 mean 不迁 std：std 迁移会把高对比 tile 拍扁成低对比，视觉上等同于"染色染过头"，Phase 2 保守选择 mean-only。

**τ=0**：`delta=0`，tile 不变（= Phase 1 行为，bitwise-equal）。
**τ=1**：tile 完全采纳 target patch 的 LAB 均值（但保留内部结构/纹理）。
**τ=0.5**：甜区（用户 plan.md 的经验值）。

**合理范围：** [0.0, 1.0]。默认 0。

---

## 3. 接口变化

### 3.1 `src/match.py` — `match_grid` 扩展

```python
def match_grid(
    target_lab: np.ndarray,    # (H, W, 3) float32
    pool_lab: np.ndarray,      # (N, 3) float32
    *,
    lambda_: float = 0.0,
    mu: float = 0.0,
) -> np.ndarray:               # (H, W) int32
```

**行为：**
- 当 `lambda_ == 0 and mu == 0`：等价于 Phase 1 broadcast argmin（bitwise-equal）。
- 否则：贪心行优先扫描。维护 `uses: dict[int, int]` 和 `placed: np.ndarray (H, W)`。每格对所有 N 个 pool tile 计算：
  ```
  score[i] = ‖target[r,c] - pool[i]‖² + λ·log(1+uses[i]) + μ·((i==top) + (i==left))
  ```
  取 argmin，更新 uses 和 placed。

**兼容性：** 位置参数 `target_lab, pool_lab` 不变；新参数为 kwonly。Phase 1 的 `match.match_grid(target, pool)` 调用处不改。

### 3.2 `src/render.py` — `render_mosaic_with_usage` 扩展

```python
def render_mosaic_with_usage(
    index_grid: np.ndarray,
    pool: TilePool,
    tile_px: int,
    output_path: Path,
    *,
    target_rgb: np.ndarray | None = None,     # (H·tile_px, W·tile_px, 3) uint8
    tone_strength: float = 0.0,
) -> tuple[Image.Image, dict[int, int]]:
```

**行为：**
- 当 `tone_strength == 0.0` 或 `target_rgb is None`：等价于 Phase 1（bitwise-equal）。
- 否则：for 每个格子 `(r, c)`：
  1. 读对应 thumb（按 Phase 1 的缓存方式）
  2. 切出 target_rgb 里对应 patch（`target_rgb[r·tile_px:(r+1)·tile_px, c·tile_px:(c+1)·tile_px]`）
  3. 调 `tone.reinhard_transfer(thumb, target_patch, tone_strength)` 得到 adjusted thumb
  4. 贴到画布

### 3.3 `src/tone.py` — 新模块

```python
def reinhard_transfer(
    source_rgb: np.ndarray,    # (H, W, 3) uint8
    target_rgb: np.ndarray,    # (H, W, 3) uint8
    strength: float,           # [0.0, 1.0]
) -> np.ndarray:               # (H, W, 3) uint8
```

**实现：**
```python
source_lab = rgb2lab(source_rgb.astype(np.float32) / 255.0)
target_lab_mean = rgb2lab(target_rgb.astype(np.float32) / 255.0).mean(axis=(0, 1))
source_lab_mean = source_lab.mean(axis=(0, 1))
delta = (target_lab_mean - source_lab_mean) * strength
adjusted_lab = source_lab + delta
adjusted_rgb = lab2rgb(adjusted_lab)
return np.clip(adjusted_rgb * 255, 0, 255).astype(np.uint8)
```

拆成独立模块是因为：
- 可独立测试（不需要 scan/match 参与）
- Phase 5 做 CLIP 匹配时色调迁移可能要换到 CNN-based；tone.py 作为稳定接口
- 单个函数独立文件开销低（< 30 行）

### 3.4 Notebook 变化

Cell 2 (CONFIG) 新增三个 key：
```python
'LAMBDA': 0.0,
'MU': 0.0,
'TAU': 0.0,
```

Cell 5 (match) 改成：
```python
idx = match.match_grid(target_lab_grid, pool.lab, lambda_=CONFIG['LAMBDA'], mu=CONFIG['MU'])
```

Cell 6 (render) 改成：
```python
img, usage = render.render_mosaic_with_usage(
    idx, pool, CONFIG['TILE_PX'], CONFIG['OUTPUT_PATH'],
    target_rgb=target_rgb, tone_strength=CONFIG['TAU'],
)
```

**新增** Cell 7 (slider UI)：
```python
import ipywidgets as widgets

lambda_slider = widgets.FloatSlider(value=CONFIG['LAMBDA'], min=0, max=50, step=0.5, description='λ (重复)')
mu_slider = widgets.FloatSlider(value=CONFIG['MU'], min=0, max=200, step=5, description='μ (邻居)')
tau_slider = widgets.FloatSlider(value=CONFIG['TAU'], min=0, max=1, step=0.05, description='τ (色调)')
rerun_btn = widgets.Button(description='重跑', button_style='primary')
out = widgets.Output()

def rerun(_):
    with out:
        out.clear_output()
        idx = match.match_grid(
            target_lab_grid, pool.lab,
            lambda_=lambda_slider.value, mu=mu_slider.value,
        )
        img, usage = render.render_mosaic_with_usage(
            idx, pool, CONFIG['TILE_PX'], CONFIG['OUTPUT_PATH'],
            target_rgb=target_rgb, tone_strength=tau_slider.value,
        )
        display(img)
        print(f"tiles used: {len(usage)} distinct / {sum(usage.values())} total")

rerun_btn.on_click(rerun)
display(widgets.VBox([lambda_slider, mu_slider, tau_slider, rerun_btn, out]))
```

Cell 8 保留显示最终图（非交互方式）。

---

## 4. 依赖

新增：
```
ipywidgets>=8
```

在 Apple Silicon 上秒装。Phase 1 其他依赖不动。

---

## 5. 文件结构增量

```
claude/
├── src/
│   ├── match.py          # 改：加 λ/μ 参数 + 贪心路径
│   ├── render.py         # 改：加 target_rgb/tone_strength 参数
│   └── tone.py           # 新
├── tests/
│   ├── test_match.py     # 改：加 λ/μ 测试
│   ├── test_render.py    # 改：加 τ 测试
│   └── test_tone.py      # 新
├── scripts/
│   └── build_notebook.py # 改：加 CONFIG 字段 + slider cell
├── requirements.txt      # 加 ipywidgets
└── bead_mosaic.ipynb     # 重新生成
```

---

## 6. 关键算法细节

### 6.1 贪心匹配的性能

- 朴素双循环 Python：H·W = 8160 次 per-cell argmin，每次对 N=500 个 tile 算距离。Python 循环会慢（~10s），不可接受。
- **加速策略**：把贪心外层留 Python，但每个格子内部的 `score = dist² + penalty` 用 numpy 向量化计算。单格 argmin ~几十微秒。总时长 <1s。

```python
def _greedy_match(target_lab, pool_lab, lambda_, mu):
    H, W, _ = target_lab.shape
    N = pool_lab.shape[0]
    placed = np.full((H, W), -1, dtype=np.int32)
    uses = np.zeros(N, dtype=np.int64)

    for r in range(H):
        for c in range(W):
            diff = target_lab[r, c] - pool_lab  # (N, 3)
            dist2 = (diff * diff).sum(axis=-1)  # (N,)
            score = dist2 + lambda_ * np.log1p(uses)
            if mu > 0:
                if r > 0:
                    score[placed[r - 1, c]] += mu
                if c > 0:
                    score[placed[r, c - 1]] += mu
            idx = int(score.argmin())
            placed[r, c] = idx
            uses[idx] += 1
    return placed
```

### 6.2 τ=0 的 bitwise-equal 保证

`render.render_mosaic_with_usage` 里，如果 `tone_strength == 0 or target_rgb is None`：
- 走原 Phase 1 路径（直接从 cache 读 thumb 贴到画布）
- 不进入任何 LAB 转换流程（避免浮点 round-trip）
- 保证像素 bitwise-equal Phase 1 输出

### 6.3 τ 算法的 broadcast-friendly 实现

因为每格 target patch 不同，对每格单独做 rgb2lab 开销大。优化：**整幅 target_rgb 先转 LAB 一次**，然后用 `target_lab.reshape(H, tile_px, W, tile_px, 3).mean(axis=(1, 3))` 一次性拿到每格的 LAB 均值（这跟 Phase 1 notebook Cell 4 做的事情完全一样，可以复用结果，但 render 模块不能假设——所以 render 内部需要自己算一次）。

---

## 7. 测试策略

### 7.1 兼容性（零惩罚/零迁移）
- `match_grid(target, pool, lambda_=0, mu=0)` 输出 bitwise-equal `match_grid(target, pool)`
- `render_mosaic_with_usage(..., tone_strength=0)` 像素 bitwise-equal Phase 1
- Phase 1 的 12 个测试原样跑

### 7.2 λ 行为
- 小 pool（3 个差异大的 tile）+ 大 grid（10×10）
- 跑两次：λ=0 和 λ=5
- 断言：λ=5 时 `max(uses) < max_uses_at_lambda_0`，且 `min(uses) >= 1`（所有 tile 都被用到至少一次）

### 7.3 μ 行为
- 同样小 pool + 大 grid
- μ=0 和 μ=500 分别跑
- μ=500 时遍历 grid 断言 `placed[r,c] != placed[r-1,c]` 且 `placed[r,c] != placed[r,c-1]`
- 除非某格附近"可用 tile 已耗尽"（pool 太小且 λ 高）——可通过取小 μ 或配合 λ=0 测试

### 7.4 tone (τ)
- 单元测试在 `tests/test_tone.py`：纯色 source + 纯色 target，τ=1，验证输出 ≈ target（LAB mean 距离 < 5）
- 单元测试 τ=0 输出 bitwise-equal source
- render 级：小规模 2×2 grid + 纯色 tile + 不同目标色，τ=1，验证每格输出 LAB mean 接近目标 patch LAB mean（ΔE < 5）

### 7.5 Smoke（集成）
扩展 `tests/test_pipeline.py`：同 Phase 1 palette，加一个 `test_end_to_end_with_knobs` 跑 λ=2, μ=10, τ=0.3 端到端，断言不抛异常、输出尺寸正确。不断言像素精确值（贪心路径不保证与 argmin 一致）。

---

## 8. 风险与取舍

| 风险 | 应对 |
|---|---|
| 贪心产生行方向色带（"上半吃蓝色"） | Phase 2 不优化到 Hungarian；让用户调 λ 使分布均匀。若效果实在不可接受，Phase 3 再考虑局部 shuffle/swapping |
| τ 在极端色差时溢出 | `np.clip(0, 255)` 处理；极端情况会产生饱和色——这是用户调 τ 的可见反馈 |
| 贪心 Python 外循环性能 | numpy 内循环向量化后单次 match < 1s；120×68=8160 格可接受 |
| ipywidgets 在 jupyter nbconvert 批量执行时不交互 | 批量执行只生成 UI（slider 初始状态 = CONFIG 默认值 = Phase 1 行为）；交互只在人工打开 notebook 时生效 |
| `slide 拖动没触发重算` 体验不好 | Phase 2 故意——避免拖动时连续触发 2-3s 级别重渲染。用"重跑"按钮显式触发 |

---

## 9. 扩展点预埋（面向 Phase 3+）

| 未来 Phase | 需要的扩展 | Phase 2 如何预埋 |
|---|---|---|
| 3（报告） | tile_usage 统计 | Phase 1 已有，不动 |
| 3（冷宫墙） | 被 penalty 挤出的 tile 记录 | Phase 2 match 内部已有 `uses` 数组，可导出 |
| 5（CLIP） | 语义距离加入 score | `match_grid` 签名可继续加 `clip_target`, `clip_pool`, `clip_weight` kwonly |
| 6（saliency mask） | 按区域重要性调节 λ | 可扩展 `match_grid` 接受可选 `cell_weight: (H, W)` |

Phase 2 不提前实现任何这些；只保证签名不翻修。

---

## 10. CHANGELOG 条目预期

完成后在 `CHANGELOG.md` 活跃条目区加一条 Phase 2 feat 条目，按 agent-friendly 格式：
- date / type=feat / target / change(具体到函数签名 + 新模块)
- rationale（"甜区三滑条"）
- validation（新测试 + Phase 1 回归 + 手动 demo mode）
- status=stable
- spec + plan 引用
