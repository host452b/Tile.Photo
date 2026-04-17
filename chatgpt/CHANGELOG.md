# CHANGELOG

> Format: agent-friendly YAML per feedback_changelog_for_agents memory.
> Compression triggers at 50 entries or 6 months (archived in CHANGELOG.archive.md).

## 活跃条目

- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 lab_mean(rgb) -> float32[3],基于 skimage.color.rgb2lab 对 uint8 RGB 求 LAB 均值
  rationale: 色差匹配的基础;后续 reinhard_transfer、ciede2000、scan_tile_pool 都依赖它
  action: 新建 mosaic_core.py 含 TileRecord/MosaicConfig/ReportBundle 三个 dataclass 骨架 + lab_mean
  result: 纯红图像测试通过,返回值与 sRGB → CIELAB 理论值 ± 0.5 内吻合
  validation: tests/test_color.py::test_lab_mean_pure_red 绿
  status: stable

- date: 2026-04-17
  type: feat
  target: chatgpt/
  change: 初始化 photomosaic toy 项目骨架,产出 requirements.txt / .gitignore / CHANGELOG.md / tests/ 空目录
  rationale: 按已批准的 spec (docs/superpowers/specs/2026-04-17-photomosaic-toy-design.md) 开始实现;MVP 定位为本地 ipynb 玩具,λ/μ/τ 三滑条 + 自嘲式报告 + DeepZoom。
  action: 创建 requirements.txt (11 个 pip-only 依赖,零 native/brew)、.gitignore (忽略 .cache/out/my_tiles/target.jpg)、CHANGELOG 头,建 tests/ 目录。
  result: 骨架就绪,进入 TDD 模块实现阶段。
  validation: 文件存在;pip install -r requirements.txt 能装;git status 干净。
  status: stable

- date: 2026-04-17
  type: try-failed
  target: chatgpt/requirements.txt + plan Task 12
  change: 从 requirements.txt 移除 deepzoom>=0.2,改为在 Task 12 自写 DeepZoom 编码器
  rationale: plan 设计时假定 PyPI 上存在 deepzoom 包 (参照 openzoom/deepzoom.py),实际 pip 已无此包 (也无 deep-zoom-tools / pydeepzoom),只有 git+ 安装可用但那会违反"零额外工具"的玩具定位
  action: 删 requirements.txt 第 10 行 deepzoom>=0.2;更新 plan §Task 12 步骤,改为纯 PIL + 手写 DZI XML 的 export_deepzoom 实现
  result: 剩 11 个依赖全部 pip 可装;Task 12 将用 ~80 行自写 DeepZoom 金字塔 + OpenSeadragon HTML
  validation: pip install --dry-run 对每个剩余依赖都 ok;DZI 格式规范见 https://learn.microsoft.com/en-us/previous-versions/windows/silverlight/dotnet-windows-silverlight/cc645077(v=vs.95)
  problem_context: Task 1 的实现 agent 跑 pip install 时发现 deepzoom>=0.2 装不上,flagged 为 DONE_WITH_CONCERNS
  workaround_reason: git+url 安装要求用户机器装 git (本身没问题,但增加了"一条 pip install 解决所有依赖"的摩擦);DZI 格式简单,80 行自写也清晰
  next_action: Task 12 执行时用纯 PIL + XML 生成 .dzi + _files/ 金字塔 + OpenSeadragon CDN 的 HTML
  status: stable
