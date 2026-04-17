# CHANGELOG

> Format: YAML entries, written for future agents (see `memory/feedback_changelog_for_agents.md`).
> Compression trigger: 50 entries OR 6 months.

## 活跃条目

- date: 2026-04-17
  type: feat
  target: Phase 2 knobs (src/match.py, src/render.py, src/tone.py, bead_mosaic.ipynb)
  change: 三个交互旋钮落地。match_grid 加 kwonly `lambda_`（重复惩罚，score += λ·log1p(uses)）与 `mu`（邻居惩罚，score += μ·(top_match + left_match)）；λ=μ=0 时保留 Phase 1 broadcast fast-path 确保 bitwise-equal。render_mosaic_with_usage 加 kwonly `target_rgb` + `tone_strength`，τ>0 时每 tile 贴图前跑 Reinhard LAB 均值迁移（新模块 src/tone.py，只迁 mean 不迁 std）。notebook 加 ipywidgets 三滑条 + "重跑" 按钮 cell；滑条 λ∈[0,200]、μ∈[0,1000]、τ∈[0,1]。
  rationale: Phase 1 brainstorm 锁定的"甜区三滑条"。λ 治"万能照被狂贴"；μ 治"大面积同色扎堆"；τ 治"近看染色过重 / 远看色斑"。全 kwonly 默认 0 让 Phase 1 的 12 个测试和 demo mode 行为不动。新增 src/tone.py 独立模块因为 Phase 5 (CLIP) 可能要换色调算法。
  action: 8 个 TDD 任务（含 1 个后续 slider 范围修订）；新增 src/tone.py；match_grid 引入 _greedy_match 私有路径（Python 外循环 + numpy 内向量化，120×68=8160 格实测 <0.5s）；render fast-path 在 τ=0 或 target_rgb=None 时完全跳过 LAB 转换。
  result: Phase 1 的 12 个测试零修改全绿；新增 11 个测试全过（tone=4、match λ/μ=4、render τ=2、pipeline knobs=1）；demo notebook 默认参数下 output.png 与 Phase 1 输出肉眼一致（已 Read 验证）；手动跑 λ=5/τ=0.4 视觉可见改善（astronaut 面部肤色平滑、国旗条纹更可辨）；λ 参数实测：λ=0→105 distinct tiles/max_use=4077，λ=100→203/1476，λ=500→349/279。
  validation: pytest 23/23；jupyter nbconvert --execute bead_mosaic.ipynb 无报错；Read output.png 可辨识；手动 python 脚本验证 λ 扫描效应。
  status: stable
  try-failed-inline: 第一版 test_lambda_reduces_max_usage 测试数据把 target a/b 值放在 [40, 60]，远离 pool 的 [-10, 10] 范围，导致某个 tile 以巨大优势被选中（所有 100 格都选它），λ=20 惩罚不足以翻转。改为 target a/b ∈ [-15, 15]、λ=50 才让惩罚可见。教训：随机 fixture 的分布区间必须与 pool 重叠，否则测试的不是算法行为而是极端数据。
  try-failed-inline-2: 第一版 build_notebook.py 的 λ 滑条 max=50，实测 λ 需要 ~100 才看到明显 tile 分布变化（λ=50 只让 distinct tiles 从 105→143），滑条天花板没意义。改 max=200 且 spec §2.1 "合理范围 0…200" 也是这个值。教训：滑条默认范围必须和算法实际敏感区匹配，不能想当然定。
  spec: docs/superpowers/specs/2026-04-17-bead-mosaic-phase2-design.md
  plan: docs/superpowers/plans/2026-04-17-bead-mosaic-phase2.md

- date: 2026-04-17
  type: feat
  target: Phase 1 core pipeline (src/scan.py, src/match.py, src/render.py, bead_mosaic.ipynb)
  change: 从零搭建 photomosaic 最小闭环：扫描底图池 (LAB 均值 + mtime/tile_px 增量缓存) → numpy 暴力 ΔE76 最近邻匹配 → 贴图渲染到 2880×1632 PNG。Demo 模式用 skimage.data.astronaut + 合成 500 HSV 色块，零配置跑通。
  rationale: 按 2026-04-17 brainstorm 结论，Phase 1 只做核心闭环，不引入 λ/μ 惩罚、色调迁移、CLIP、DeepZoom、Gradio（依次分到 Phase 2-6 独立 spec/plan）。可迭代性优先于一次性完成。
  action: 9 个 TDD 任务完成；7 cell notebook 通过 nbformat 脚本生成；src/ 拆成 scan/match/render 三模块为 Phase 2-6 预埋扩展点（TilePool 可扩 clip_emb；render_mosaic_with_usage 已返回 usage dict 供 Phase 3 报告用；match_grid 签名稳定可加 λ/μ 参数不破坏调用点）。
  result: pytest 全绿 12/12（smoke test 对纯色 palette 做端到端 round-trip 验证每格 ΔRGB < 8）；demo mode notebook 产出 output.png 肉眼可辨 astronaut（橙色太空服、人脸、左侧美国国旗、右侧头盔、letterbox 黑边正确）。
  validation: tests/test_scan.py (6) + test_match.py (3) + test_render.py (2) + test_pipeline.py (1) 全过；`jupyter nbconvert --execute bead_mosaic.ipynb` demo mode 无报错；Read output.png 验证可辨识度。
  status: stable
  spec: docs/superpowers/specs/2026-04-17-bead-mosaic-phase1-design.md
  plan: docs/superpowers/plans/2026-04-17-bead-mosaic-phase1.md

- date: 2026-04-17
  type: decision
  target: tests/test_render.py
  change: 测试 fixture 用 PNG 替代 JPG 保存 solid-color tiles。
  rationale: 第一次跑 render 单测时 JPEG 有损压缩把 (255,0,0) 压成 (254,0,0)，array_equal 断言挂掉。render 本身没问题，是测试 fixture 的选择问题。
  action: _solid_thumb helper 从 `.save(path, quality=92)` 改成 `.save(path)` 并把路径从 .jpg 改成 .png。
  result: 两个 render 测试通过。
  validation: pytest tests/test_render.py
  status: stable
  lesson: 像素级 array_equal 断言必须用无损格式。实际业务代码里 render 从 cache 读的是 JPG 缩略图——那是 90%+ 质量可接受的损失，不影响马赛克视觉效果。
