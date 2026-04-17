# CHANGELOG

> 按 agent-oriented 规范维护：刻意啰嗦、保留 try-failed 轨迹、50 条或 6 月触发压缩。
> 压缩归档见 `CHANGELOG.archive.md`。

## 活跃条目

- date: 2026-04-17
  type: feat
  target: doubao/
  change: 初始化照片马赛克 MVP 骨架 —— 8-cell notebook + src/mosaic/ 六模块（config/tiles/match/render/report/deepzoom）+ tests/ 三文件（test_match.py, test_render.py, test_e2e.py）
  rationale: 用户明确玩具定位；8-cell 是 MVP 底线。模块化是为了将来加 CLIP / cursed mode / Gradio 的增量不需要重构 —— 不是为了产品化。关键决策：(1) 用 numpy argpartition 替 faiss，因为 N≤5k 无性能差距 + arm64 wheel 有风险；(2) 手写 DZI（~100 行）替 `deepzoom` pypi，因为后者 2018 后无维护。两处决策均在 plan 的 Implementation Notes 节显式记录。
  action: 按 12-task plan 依次实现。pyproject.toml 用 hatchling src-layout + pytest pythonpath=["src"]；tiles.py 递归扫描（跳过 .git/__pycache__/.cache/@eaDir/.ipynb_checkpoints/*.app）+ 中心裁剪 + rgb2lab 均值 + mtime-keyed pickle 缓存；match.py split_target 用 PIL Image.BOX resize 做块均值再 rgb2lab + match_all_tiles 行优先 + argpartition top-k 候选 + score = color_dist + λ·log(1+uses) + μ·1/(1+mean_neighbor_dist)；render.py Reinhard LAB 均值迁移（τ 线性混合）+ paste 到画布；report.py 自嘲文字 + matplotlib 柱状图 + 冷宫缩略图墙；deepzoom.py 手写 DZI 金字塔（每级减半到 1px）+ 256² JPEG tiles w/ 1px overlap + OpenSeadragon CDN HTML；photomosaic.ipynb 8 cell 纯 import+调用。
  result: `uv sync --all-groups` 通过（49 包，含 numpy/scikit-image/matplotlib/pillow/tqdm/ipykernel）；`uv run pytest` 8 pass（test_match 4 + test_render 2 + test_e2e 2），全部 <0.5s；notebook cell 1 import 成功；端到端 smoke 在 100 张合成底图 × 16×9 网格下产出 256×144 mosaic.png + deepzoom/index.html + mosaic.dzi。提交哈希链：b839331 scaffold → d9ff388 config → c5c13eb tiles → 11a066f match.split_target → 61e90a4 match_all_tiles → 20b3ea2 penalties → 3fe4698 render → 2e0fb6a report → a6c368d deepzoom → 69995fe notebook → 05a6b28 e2e。
  validation: pytest 8/8 绿 + test_e2e.py 覆盖未单测的 tiles/report/deepzoom 三模块 + DZI XML 含正确 Width/Height + 10 级金字塔（log2(512)+1）+ cache 二次 build 命中验证。
  status: experimental
  notes: success criteria 中 3/4/5/6（notebook 在真实照片库跑通、verbose 思考输出、mosaic 视觉合理、deepzoom 可缩放）需要用户提供真实素材后人工验证才能标 stable。

- date: 2026-04-17
  type: try-failed
  target: doubao/tests/test_match.py
  change: 最初用 LAB `a` 通道断言区分 red 和 blue target
  rationale: 想选一个能强对比红蓝的 LAB 通道；naive 直觉认为 `a` 是 green↔red 轴，blue 应该 a<0
  action: 写 `assert right_a < 0  # blue is -a`（right half 是纯蓝区）
  result: 失败 —— `AssertionError: assert np.float32(79.18559) < 0`
  validation: pytest 首次运行断言失败；深挖后确认 pure blue RGB (0,0,255) 在 CIE LAB 中 `a ≈ 79`（正值，偏红），并非负值
  problem_context: 设计 split_target 的第一个单元测试，需要一个能稳健区分两种纯色的 LAB 通道断言
  workaround_reason: RGB 纯色到 LAB 的映射有反直觉之处：red 和 blue 在 `a` 通道都是正值且数量级相近（red a≈+80, blue a≈+79）；真正的分界线在 `b` 通道（red +67 yellow 方向, blue -108 blue 方向）。**未来再写 LAB 断言时，测试前先 `from skimage.color import rgb2lab; rgb2lab([[[0,0,255]]])` 打印一次真实值，不要靠直觉。**
  next_action: 改成 `assert left_b > 20 and right_b < -20`（用 `b` 通道）
  next_result: 成功，见同日 feat 条目
  status: reverted

- date: 2026-04-17
  type: try-failed
  target: doubao/tests/test_match.py
  change: 用 `lambda_reuse=5.0` 测试 10 个相同 target cell 下重复惩罚能分散选择
  rationale: 想选一个"温和"的 λ 看看是否有效
  action: `assert max(use_pen.values()) < 10 with lambda_reuse=5.0`
  result: 失败 —— `{1: 10}`，tile 1 仍被选 10 次
  validation: pytest 首次运行 `assert 10 < 10` 失败
  problem_context: 验证 λ·log(1+use_count) 惩罚项确实能在 target 完全匹配某 tile 时也分散选择
  workaround_reason: λ 必须大到能和候选之间的最小色差竞争。本合成 pool 中 tile 1 (warm light LAB ≈ (83,3,16)) 到第二近邻 tile 0 (gray LAB ≈ (54,0,0)) 的距离约 33；λ·log(1+n) 在 n=1 时需要 ≥ 33 才能触发切换，即 λ ≳ 33/log(2) ≈ 48。λ=5 下 5·log(2) ≈ 3.47 根本不是对手。**未来调 λ 的经验法则：λ 的量级应与候选对间 LAB 色差的量级相当（常见 RGB 相似色差 10-30，相异色差 30-80）。**
  next_action: 把 λ 提升到 60
  next_result: 成功，见同日 feat 条目
  status: reverted
