# CHANGELOG

> 本 changelog 的主要读者是 agent，不是人。规则见 `~/.claude/projects/.../memory/feedback_changelog_for_agents.md`。
> 要求刻意啰嗦、保留 try-failed 链条、ISO 日期、达到 50 条/6 个月触发压缩。

## 活跃条目

- date: 2026-04-17
  type: feat
  target: perplexity/ (全 MVP)
  change: 初始实现 photomosaic 生成器 MVP
  rationale: |
    按 spec docs/superpowers/specs/2026-04-17-photomosaic-ipynb-design.md
    和 plan docs/superpowers/plans/2026-04-17-photomosaic-ipynb.md 的
    MVP 范围实现。定位是本地 Mac 玩具：速度/普适性/稳定性都不追求，
    可解释性和"玩的质感"第一。
  action: |
    - mosaic/pool.py: 扫描底图 + LAB 均值 + pickle 缓存（mtime 增量 + 坏图 skip + 失效清理）
    - mosaic/target.py: 读目标图 + 按网格宽高比 center-crop + 分网格（每格 LAB 均值 + 方差）
    - mosaic/match.py: faiss IndexFlatL2 top-K + λ*log(1+usage) 重复惩罚 + μ*Σexp(-‖Δ‖²/σ²) 邻居惩罚 + 方差降序贪心扫描 + semantic_reranker V2 钩子
    - mosaic/transfer.py: Reinhard LAB 均值-标准差迁移 + τ 线性混合（EPS=1e-6 防零标准差）
    - mosaic/render.py: 组装 canvas + 每 tile 过 Reinhard + usage Counter
    - mosaic/report.py: 自嘲式文字模板 + top-30 柱状图 + 冷宫 5x4 缩略图墙 + region label 启发式
    - mosaic/zoom.py: 纯 Pillow 内联 DZI 金字塔 + OpenSeadragon CDN HTML
    - mosaic/config.py: DEFAULT_CONFIG (16 keys) + ipywidgets 三滑条 λ/μ/τ + preview/render 按钮工厂
    - mosaic/pipeline.py: run_pipeline 端到端（scan → grid → solve → render → report + deepzoom，do_report/do_deepzoom 开关）
    - mosaic.ipynb: 5-cell notebook 入口（markdown / imports / config / widgets handlers / 提示）
    - README.md: setup + 使用 + 产物 + V2/V3 roadmap
  result: |
    本地 smoke test 通过：32 张 tile + 20×12 网格 → PNG + report.md + DeepZoom html 全部生成。
    报告文本里的统计数字和 top tile 正确，DeepZoom tiles 金字塔写出 11 张瓦片。
  validation: |
    31 单测全过（pytest tests/ -v）：
      test_pool.py (6) + test_target.py (4) + test_match.py (4) + test_transfer.py (4)
      + test_render.py (2) + test_report.py (4) + test_zoom.py (2) + test_config.py (3)
      + test_pipeline.py (2) = 31 passed
    外加一次 realistic-input smoke（32 tile pool / 640×480 target）人工验证产物可读
  status: stable

- date: 2026-04-17
  type: try-failed
  target: perplexity/requirements.txt
  change: 最初计划依赖 `deepzoom>=0.5` 作为 PyPI 包
  rationale: spec 草稿假设存在 "deepzoom pure-Python pip package"
  action: 把 `deepzoom>=0.5` 写进 requirements.txt 里
  result: |
    Task 0 实施时 pip install 报错 "No matching distribution found for deepzoom>=0.5"。
    核实 PyPI：https://pypi.org/pypi/deepzoom/json 返回 404，包不存在。
    `pydeepzoom` / `deep-zoom` / `dzi` 等变体也都 404。
    `pyvips` 倒是存在但需要 brew install vips，违反"pure pip, no brew"约束。
  validation: |
    curl -s https://pypi.org/pypi/deepzoom/json -o /dev/null -w "%{http_code}" → 404
    pip index versions deepzoom → ERROR: No matching distribution
  problem_context: |
    需要一个把 PNG 切成 DeepZoom 金字塔瓦片的工具，让 OpenSeadragon 能无限放大
  workaround_reason: |
    考虑过 git+https://github.com/openzoom/deepzoom.py.git，但依赖的长期可用性不确定；
    pyvips 违反依赖策略；
    对一个 ~50 行的 DZI 切片算法来说，引第三方依赖比内联实现成本更高。
  next_action: Task 7 改为纯 Pillow 内联实现 DZI 金字塔（math.ceil levels + crop 切片 + DZI XML template）
  next_result: |
    实现了 zoom.py export_deepzoom()，2 个单测通过（tile 写出数 + DZI XML 含预期宽高）。
    端到端 smoke 生成 11 张金字塔瓦片，index.html 能通过 CDN 加载 OpenSeadragon。
  status: reverted

- date: 2026-04-17
  type: observation
  target: perplexity/mosaic/report.py
  change: matplotlib 柱状图里的中文字符（"使用次数" / "Top 30 被使用最多的 tile"）显示为 tofu
  rationale: |
    DejaVu Sans（matplotlib 默认字体）不含 CJK glyph。
    warning 类似 "Glyph 20351 (使) missing from font(s) DejaVu Sans"。
  action: |
    未在 MVP 里修复——spec 没要求 CJK 渲染，单测不验证字形渲染，markdown 文字本身没事（只是 chart png 有 tofu）。
    后续如果要修，两条路：
      (1) 在 report.py 顶部加 `matplotlib.rcParams['font.family'] = 'PingFang SC'` 或其它 macOS 内置 CJK 字体
      (2) 把 chart 标签换成英文
  status: experimental
  workaround_reason: 玩具定位，不追求细节打磨；MVP 优先功能完整，tofu 字不影响 usage 数据可读性
