# CHANGELOG

> Format: YAML entries, written for future agents (see `memory/feedback_changelog_for_agents.md`).
> Compression trigger: 50 entries OR 6 months.

## 活跃条目

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
