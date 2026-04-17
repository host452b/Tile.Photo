# CHANGELOG

> Format: agent-friendly YAML per feedback_changelog_for_agents memory.
> Compression triggers at 50 entries or 6 months (archived in CHANGELOG.archive.md).

## 活跃条目

- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 ensure_seed_tiles(dir, n=200),空目录自动生成 64×64 HSV 色块 JPG
  rationale: 零配置跑通;Cell 3 即使底图目录空也能继续,不崩
  action: 固定 random.Random(0) 保证可复现;已有内容时 no-op,不覆盖用户真实照片
  result: 10 张 case 测试 shape 与数量;已有文件 case 验证 no-op
  validation: tests/test_transfer.py::test_ensure_seed_tiles_* (2 个) 绿
  status: stable

- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 rerank(candidate_idxs, ..., λ, μ) -> tile_idx,结合 ΔE + λ·log(1+usage) + μ·max_neighbor_sim
  rationale: λ 摊开重复使用,μ 避免同色扎堆;比纯最近邻出图干净得多
  action: 在 top-k 候选里逐个算三项和,取最小;neighbor_sim = 1/(1+ΔE) 天然有界
  result: λ=μ=0 退化为 argmin ΔE;λ=100+usage=1000 可翻盘;μ=100+克隆邻居可翻盘
  validation: tests/test_matching.py::test_rerank_* (3 个) 绿
  status: stable

- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 build_faiss_index / knn_candidates,做 LAB 空间 top-k 候选查询
  rationale: N < 10k 底图量级,IndexFlatL2 就够用,无需 IVF;有 k > N 时自动降级为 N
  action: lazy import faiss;把 target 从 (H, W, 3) reshape 成 (H·W, 3) 批量查询
  result: 100 tile 查询 top-5 确定性且与 argmin L2 一致;4×6 grid 返回 (24, 8)
  validation: tests/test_matching.py::test_build_faiss_index_*, test_knn_candidates_* 绿
  status: stable

- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 render_mosaic(assignment, tile_records, tile_px, τ, target_lab) -> PIL.Image
  rationale: 把 assignment 表变成可视化像素矩阵;τ>0 时每 tile 贴前做 reinhard_transfer
  action: 双层 for 循环按格贴图,尺寸不匹配时 resize;短路 τ=0 省去 LAB 往返开销
  result: 4×3 grid 输出 64×48 PIL Image;τ=0 块内字节完全等于 tile 原图
  validation: tests/test_transfer.py::test_render_mosaic_* 绿
  status: stable

- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 split_target(img, grid_w, grid_h) -> float32[H, W, 3],输出每 patch LAB 均值
  rationale: 匹配阶段需要"每格目标颜色"作为 KNN 查询向量
  action: resize 到 grid_w*patch_w × grid_h*patch_h 后 reshape+mean 向量化计算,无 Python 循环
  result: 100×50 图切 10×5 返回正确 shape;灰度图各 cell 的 LAB 一致且 L~54
  validation: tests/test_transfer.py::test_split_target_* 绿
  status: stable

- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 reinhard_transfer(tile_rgb, target_lab_mean, τ) -> rgb,按 τ 线性混合 LAB 均值迁移后的 RGB 与原图
  rationale: τ 是"有质感层"的核心滑条;原 Reinhard 含 std 缩放,此处只做 mean-shift 够用且稳定
  action: τ=0 短路返回原 tile;τ=1 返回纯迁移结果;中间做 (1-τ)*orig + τ*transferred 线性混
  result: τ=0 byte-exact;τ=1 后新均值与目标误差 < 4 LAB 单位(sRGB 裁剪导致的正常漂移)
  validation: tests/test_color.py::test_reinhard_tau_{zero,one}_* 绿
  status: stable

- date: 2026-04-17
  type: feat
  target: chatgpt/mosaic_core.py
  change: 实现 ciede2000(lab_a, lab_b) -> float,包装 skimage.color.deltaE_ciede2000 返回标量
  rationale: ΔE_CIEDE2000 是 rerank 里的主色差项;纯 LAB 欧氏会在深色/饱和色区域失真
  action: 新增函数,内部 reshape 到 (1,1,3) 后 squeeze 出标量
  result: 同点 ΔE < 1e-6,红-绿对比 > 10 两条测试均通过
  validation: tests/test_color.py::test_ciede2000_* 绿
  status: stable

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
