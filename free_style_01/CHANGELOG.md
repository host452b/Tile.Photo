# CHANGELOG · `free_style`

> 写给下一个接手这个目录的 agent / 作者自己:刻意啰嗦,保留 try-failed 链条。

---

## 2026-04-17 · patch 3 · 故障风视觉

用户反馈"不够艺术,要带点故障风"。全套 CRT / glitch aesthetic 注入,逻辑零改动。

- **CRT scanlines**:`body::before` 固定叠层 `repeating-linear-gradient` 2px 透明 + 1px 微亮,`mix-blend-mode: overlay`。整页扫描线。
- **Chromatic aberration on logo**:`Tile.Photo` 标题用 mono + `text-shadow: -1px 0 cyan, 1px 0 magenta`,再加 9s 一次的 `logo-glitch` keyframes,瞬间错位 100ms。
- **Frame shudder**:`body::after` 每 11s 触发 `crt-shudder` — 一帧 cyan 0.04 + 一帧 magenta 0.04 跳闪,仿 VHS 掉帧。
- **System status LED**(新)——header 右侧加了 sys-status 指示器,5 态:READY/INGEST/COMPUTE/DONE/FAULT,每态专属色 + 脉冲 dot,DONE 时 dot 静止。app.js 在 handleTileFiles / startBuild / handleDone / cancelBuild / error 处切换。
- **Mono 字体漫延**:button、slider-name、slider-hint、step-hint、tagline、log、footer 全上 mono,增加"机械"观感。中文主要内容保持 sans,可读性不牺牲。
- **锐角优先**:全局圆角从 `10px / 6px` 降到 `2px` 或 `0`。按钮、slider-value、padded 元素都是近直角。
- **Cyan + magenta 辅助色**:step-n、slider-value、徽章、log-stage 前缀、drop icon、focus ring 全部 cyan;selection 背景 magenta;hover accent 见于 按钮 chromatic split。
- **Viewfinder 角标**:`.step::before/::after` 每个 step 左右各一个 10×10 直角,仿取景器定位框。
- **Progress bar 扫光**:`.bar-fill::after` 一个线性 overlay,1.6s 循环横扫,表达"进度中"的机械感。
- **Log 终端光标**:`.log-stream .log-line:last-child::after` 在最后一行末尾追加绿色 `_`,blink 动画。
- **Blink cursor on step headings**:`h2::after = '_'`,橙色 1.1s blink,像等待终端输入。
- **Button chromatic aberration on hover**:`.primary-btn:hover text-shadow: -1px cyan, 1px magenta`,配合 ::after 的亮条扫过。
- **HTML 改动**:只在 header 加了 sys-status 三行 div;logo 字符从 `◇` 换 `▣`(更"像素化");其他结构不动。

用户还能感觉到的副作用:
- 滑条 thumb 改成方块(橙色发光),轨道也从圆 → 直。
- 表格/列表的圆角统统消失。报告里的 `kbd` 徽章现在是深底 + 橙色 + 橙色光晕。
- 冷宫照片 hover 时描边变 magenta。
- 文字选择默认是 magenta 背景。

如果觉得太重,所有效果都是单行 CSS;最容易降温的几个开关:
- `body::before` 删掉 → 去 scanline
- `.brand h1 { animation: none; text-shadow: none; }` → 去 logo 错位
- `body::after { animation: none; }` → 去整页 shudder
- `.step-head h2::after { content: none; }` → 去标题光标
- `.log-stream .log-line:last-child::after { content: none; }` → 去日志光标

---

## 2026-04-17 · patch 2 · 修"重跑不响应" + worker 常驻

**症状**:用户第一次跑完后调滑条,再点"开始拼",没反应。

**根因**:`startBuild` 结尾写了 `for (const t of state.tiles) t.workerBitmap = null;` —— 这是因为 transfer 后 workerBitmap 在主线程被 detach,我用 null 标记避免后续 close(null) 的 noop。但第二次 startBuild 时 `tileBitmaps = state.tiles.map(t => t.workerBitmap)` 全是 null。传给 `postMessage(msg, [null, null, ...])` 时,transferables 里有 null 不合规 —— Chrome 静默抛 DataCloneError。因为 `startBuild` 是 async 且没 try/catch,错误被 promise 吸收,`state.running` 永远 true,按钮永远 disabled。用户只看到"点了没反应"。

**修(最小版)**:每次重跑前 `createImageBitmap(mainBitmap)` 克隆一份新 workerBitmap。

**修(做到位)**:顺手重构 worker 协议。原先 `build` 消息一条龙:tile 分析 + 匹配。现在拆 `load-tiles` / `match` / `reset`,worker 缓存 tile labs,改参后 startBuild 只发 `match`。效果:

- 第一次跑:和以前一样开销(分析 + 匹配)
- 改 τ / λ / μ / grid 再点"开始拼":直接跳到匹配。对 5600 格 × 1000 底图,从 3-4s → 0.5-1s
- 改底图池:自动 `reset` worker 缓存,下次 startBuild 会走完整路径

**相关小修**:
- `ensureWorkerBitmaps()`:从 mainBitmap 克隆 workerBitmap,用于二次运行
- `waitForWorkerMessage(worker, type)`:拿一个 Promise 等 sentinel 消息,期间 handleWorkerMessage 还是正常处理进度事件
- `invalidateTilesCache()`:上传新底图 / resetTiles 时调用,发 `reset` 给 worker 并把 `state.tilesLoadedInWorker` 翻 false
- `startBuild` 加 try/catch,让 async 错误不再被吞
- `cancelBuild` 终止 worker 时把 `tilesLoadedInWorker` 翻回 false(缓存随 worker 消失)

**试过但没做**:
- 把 worker 的 tile LAB 缓存暴露给主线程用 BroadcastChannel / SharedArrayBuffer → 不需要,主线程根本不用 LAB,只用 RGB 做渲染。
- 在 handleTileFiles 里等 `reset-ack` 确认 worker 真的清了缓存 → 没必要,reset 是 fire-and-forget,下次 load-tiles 会覆盖。

---

## 2026-04-17 · patch 1 · 修算法对称 + 推荐数量 UI

用户反馈两个问题,都命中:

### 问题 1 · 底图数量没有动态推荐

**症状**:Step 02 的提示只写死了"建议 300+ 张",但用户改 grid 滑条到 200(= 200×113 = 22,600 格)时,300 张远远不够。

**修**:
- index.html Step 02 的 step-hint 加 `id="tiles-rec"`。
- app.js 新增 `tileCountThresholds()`:基于目标图纵横比 + grid 滑条算出 `{cells, min, rec, ideal}`。
  - 最低 = `max(50, cells/30)` — 能跑,严重重复
  - 推荐 = `max(200, cells/8)` — 够跑有点多样
  - 理想 = `max(800, cells/3)` — 几乎无重复
- `updateTileRecommendation()` 在 target 变、grid 滑条变、tiles 变、reset 等所有节点都调一遍。
- step-hint 根据 `data-level` 上色:warn=红,ok=黄,good=绿。
- 启动时提示"先把目标图丢进上面的 01,我好告诉你到底需要多少张"。

### 问题 2 · 颜色匹配算法 bug(严重)

**症状**:用户说生成图的颜色/色度匹配不对。

**根因**:tile 和 target 两边算"平均色"的方式**不一致**:

- **Target** (`app.js: computeTargetPatches`):
  - `drawImage(bitmap, 0, 0, gridW, gridH)` — 浏览器下采样,**每个输出像素 = 一个 patch 的 sRGB 平均色**
  - 读 RGB → `rgbToLab`
  - 每个 patch 的 LAB = 真实平均色的 LAB

- **Tile (旧版 `worker.js: analyzeTiles`)**:
  - `drawImage(tile, 0, 0, 32, 32)`,**读 32×32 = 1024 像素的 RGB**
  - **每个像素单独 rgbToLab,然后把 1024 个 LAB 值算术平均**
  - 每个 tile 的 LAB = 一个"虚拟色"的 LAB,通常在物理上**不存在**

**为什么错**:LAB 是感知空间,非线性的。对"上半红 下半蓝"的 tile:
- LAB 平均给:`[L=43, a=80, b=-20]` — 看起来像暗红偏紫,但这是 `LAB(红)+LAB(蓝)/2`,不对应任何真实色。
- RGB 平均给:`(127, 0, 127) → LAB=[30, 59, -36]` — 纯紫,tile 做小显示时眼睛看到的色。

Target 用的是 RGB 平均(浏览器下采样)。所以 tile 也必须用 RGB 平均。匹配才有意义。

**修**:
- worker.js `analyzeTiles`:改为 `drawImage(tile, 0, 0, 4, 4)` → 16 像素 RGB 均值 → `rgbToLab`。
- 4×4 而不是 1×1:drawImage 到 1×1 在部分实现里会走 nearest-neighbor。4×4 bilinear 下采样再 JS 平均,结果和浏览器到 1×1 的理想行为等价,更保险。
- 成本:分析单 tile 从 1024 次 LAB 变 1 次 + 16 次 RGB 加法,快 100×。

### 问题 3(顺手) · 重复惩罚曲线

**症状**:λ slider 拉到 0.30 感觉没啥用,一张万能照还在狂贴。

**根因**:`lambda * log(1 + uses)` 对 `uses=50` 才到 `log(51)=3.9`,乘 λ=0.3 只有 1.17。而 top-K 候选之间 ΔE 差 1-5,1.17 的惩罚几乎不改变谁胜出。

**修**:换 `lambda * sqrt(uses)`。`uses=50 → 7.07 → λ=0.3 → 2.12`(约 log 的 1.8×,在 uses>20 时差异更大)。让默认 λ=0.3 也开始有感觉。

### 没改的

- Neighbor penalty 仍是 `mu * (matches in 4 causal neighbors)`。用户没反馈,先不动。
- 默认 λ 滑条值仍 30。如果 sqrt 已经够,保留用户可以拉满。
- τ 色调迁移的 alpha-blend 做法。如果用户觉得 τ 依然偏染色,v2 换成 per-tile LAB mean shift。

---

## 2026-04-17 · 首次端到端 v1

### 选型决策(在用户给的两条路中挑)

**备选 A:ipynb + Gradio + Python**(open_clip, faiss, pyvips,周末能写完)
**备选 B:React + WebAssembly(Rust wasm-bindgen)**(浏览器原生、100% local、实时)
**选定 C(新):纯浏览器 JS + Web Worker + OffscreenCanvas** — 既不是 A 也不是 B

**为什么不选 A(ipynb)**:
- 用户的交付标准是"下次交互时可以尝试产品效果"。ipynb 要求对方起 conda 环境、装 faiss-cpu(Mac arm64 编译经常断)、等 CLIP 模型下载。不可即开即玩。
- 调参反馈循环差:每改一次 λ / τ 都要重跑 Cell 5-6。而用户在上文明确说"拖一下滑条等 10 秒看新图"是体验核心。ipynb 给不了。

**为什么不选 B(Rust+WASM)**:
- 一次会话端到端实现 Rust + wasm-pack + React + transformers.js 流水线风险太高。任何一个点 blocked 整个交付失败。
- CLIP 的 ONNX 量化 + transformers.js 在首包加载/内存/浏览器兼容上有一堆尖角,v1 不值得。
- "卖点三个字:浏览器本地跑" — 纯 JS 同样能做到,体验 90% 等价,开发成本 20%。

**选定方案理由**:
- 同样是 "100% local",同样是 Web Worker + OffscreenCanvas + Transferable ImageBitmap。
- 朴素 JS 对 200×113 格 × 3000 底图的匹配任务 5-10 秒完成,可接受。
- 没有构建步骤,用户起 `python3 -m http.server` 就能玩。
- CLIP / DeepZoom / 彩蛋模式推到 v2,v1 先把骨干跑通。

### 架构关键点(方便下次不再想)

1. **Tile ImageBitmap 创建两份**:一份 transfer 给 worker 做像素分析,另一份主线程保留用于实时渲染。Transfer 语义会"掏空"主线程的 bitmap,不能复用。两份创建 = 两次 `createImageBitmap(full, sx, sy, size, size, { resizeWidth: 40, resizeHeight: 40 })`,共享同一个 full decode。
2. **Target patches 主线程计算**:主线程把 target scale 到 `gridW × gridH` 的 OffscreenCanvas,getImageData,每个像素 = 一个 patch 的 RGB/LAB 均值。传 Float32Array 给 worker。主线程自己留 RGB 拷贝用于渲染期色调迁移。
3. **色调迁移**:v1 用最简做法 `ctx.fillStyle = target_mean_rgb; ctx.globalAlpha = tau; ctx.fillRect(...)`。这等价于 `final = lerp(tile, target, tau)` — 数学上比较粗糙(不是 LAB 均值迁移也不是 Reinhard),但视觉上 80% 对,且是 Canvas 原生合成,不花 CPU。
4. **Worker 不做渲染**:worker 只算匹配 + 发事件。主线程随事件流 `drawImage` 到输出 canvas。这样用户看到的是"一格一格填"的动画 — 呼应用户"能看见它思考"的需求。
5. **topK 里再算惩罚**:匹配是 `top-K by 颜色 → 在 K 里重排 by 颜色+λ·log(uses)+μ·邻居匹配数`。全集 N 可能 3000,topK=50。惩罚只加到 50 个候选上,数值可控。
6. **邻居:4 个因果邻居**(左上 / 上 / 右上 / 左)。不用 8 邻居是因为还没填到下面,用 causal 比较干净。

### 试过 / 没做 / 选择留存

- **没用 `createImageBitmap(imageBitmap)` 克隆**:API 需要先完整 decode 一份 full bitmap。我采取"decode 一次 → 调两次带 resize 的 createImageBitmap"。如果未来要再调整 tile 尺寸,改 `TILE_BITMAP_SIZE` 常量就好。
- **没用 KD-Tree**:N=3000 的线性扫描一次 ~0.3ms(3000 次 LAB 距离),整个 22k 格共 ~7s。KD-tree 能到 0.5s,但得加依赖或自己实现 BBF。v2 再说。
- **没用 CIEDE2000**:公式贼长,对 photomosaic 排序用处不明显(反正都是相对距离)。用了 LAB 欧氏。
- **没做 saliency / 人脸密格**:MediaPipe 要模型,v2。
- **没做实时参数重跑**:滑条目前只显示值,不会重跑。如果希望滑条改变后秒级预览 —— 需要把 worker 改成"可中断 + 可重入",v1 不做。用户要重跑,需要再点一次"开始拼"(下次迭代里加一键重跑)。

### 可能翻车的点(如果用户反馈)

- **拖文件夹在 Safari 上**:`webkitGetAsEntry` 兼容性 OK,但 folder input 的 `webkitdirectory` 属性在 iOS Safari 无效。v1 默认 Chrome/Firefox/Edge。
- **`createImageBitmap` 的 resize 选项**:Firefox 曾经一段时间没实现 `resizeWidth/resizeHeight`,但 ≥ 94 应该都有。
- **超大 target(>10MP)**:`createImageBitmap(file)` decode 全尺寸,如果 target 是 100MP 会占内存。没做 resize — 反正立刻 scale 到 gridW×gridH 的 OffscreenCanvas,full bitmap 很快会被 GC。但决解码那一瞬峰值内存高。够用了不优化。
- **λ 默认 0.3 + log 曲线**:对高重用度 tile 的惩罚偏软(uses=50 时 penalty ≈ 1.2,ΔE 通常 5-20)。如果看到"填空王"很明显,把滑条拉到 80-100。再不够的话 v2 换 linear。

### v2 待办(按冲击力排序)

1. **"重新拼"按钮**:不重新上传,只重跑 worker 匹配。给快速参数调优。
2. **CLIP 语义**:transformers.js + CLIP-ViT-B/32 ONNX 量化,作为可选第 5 个滑条。
3. **导出可分享 HTML**:OpenSeadragon + DeepZoom tiles。
4. **预设彩蛋**:Cursed / TimeCapsule / WeChat,每个预设锁定底图池类型 + 参数组。
5. **WebGPU 匹配路径**:10k+ 底图时启用。
