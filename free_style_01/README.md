# Tile.Photo · `free_style`

把硬盘里一堆照片重组成另一张照片的本地玩具。

- **100% 本地**:照片不离开浏览器,没有服务器,没有上传。
- **即开即玩**:不装依赖、不编译、不 npm install。
- **能看见它思考**:一格一格地填,每格告诉你为什么挑这张、刷掉了哪些候选。
- **自嘲式战报**:谁被用了 89 次当填空王、谁被打入冷宫。

## 跑起来

两种方式任选:

```bash
# A. Python 自带的静态服务器
cd free_style
python3 -m http.server 8000
open http://localhost:8000
```

```bash
# B. Node 一行起
cd free_style
npx --yes serve .
```

然后:
1. 拖一张目标图进 **Step 01**。
2. 拖一个装着 300 张以上照片的**文件夹**进 **Step 02**。
3. 动一下 **Step 03** 的三个滑条(或用默认值,都是甜区)。
4. 点**开始拼**。

> 不建议直接双击 `index.html` — 现代浏览器对 `file://` 下的 Worker 和 OffscreenCanvas 限制越来越多。起个 HTTP 服务再看。

## 三个滑条到底在干嘛

| | 意思 | 建议 |
|---|---|---|
| `τ` · 色调迁移 | 每块底图贴上去之前,多大程度把它染成目标色 | 40-60% 是甜区。100% 就是在用底图当"彩色像素"。 |
| `λ` · 重复惩罚 | 一张底图被重复用的代价 | 越大越强迫它多样。0.3-0.5 通常够。 |
| `μ` · 邻居惩罚 | 相邻格子撞脸的代价 | 0.5-1.0 能显著减少"团块"。 |

技术上的匹配分数:
```
score(tile, cell) = LAB距离(tile, cell) + λ·log(1+uses[tile]) + μ·(邻居里同款数)
```
对每格:先在 **Top-50 颜色最近**的候选里挑这个 `score` 最小的那张,而不是全局扫描。这比商业工具的"最近邻 + 染色拉满"风格更值得玩。

## 技术栈 (v1)

- 纯 HTML/CSS/JS 模块 + Web Worker + OffscreenCanvas
- 没有 Rust,没有 WASM,没有 React,没有构建步骤
- LAB 颜色空间 + 加权最近邻

## v1 里**不做**的(已选入 v2)

| 砍掉的东西 | 为什么 | 什么时候回来 |
|---|---|---|
| CLIP 语义匹配 | 要引入 transformers.js + ~80 MB 的 ONNX 权重;v1 先把骨干跑通 | v2:作为可选 checkbox,默认关 |
| DeepZoom / OpenSeadragon | CSS transform zoom 已经够玩,金字塔切片是另一套基建 | v2:加**导出可分享 HTML**按钮时 |
| 彩蛋模式(Cursed / TimeCapsule / WeChat) | 每个都值得单独设计,现在做会稀释 v1 | v2:作为预设(preset),底图池 + 目标都给定模式 |
| 底图池打标签 + 叙事统计 | 要文件名解析规则,v1 先看算法能不能跑 | v2:识别 EXIF 日期聚类 |
| 主体 saliency(人脸密格) | rembg / MediaPipe 是单独的模型 | v3 |
| GPU 匹配 | WebGPU 能把 5s 干到 0.3s | v3:遇到 10k+ 底图再做 |

## 文件结构

```
free_style/
├── index.html       # 页面骨架 + 5 个 step section
├── css/style.css    # 深色主题
├── js/
│   ├── app.js       # 主线程:UI / 文件 / Worker 协调 / 渲染 / 报告
│   ├── worker.js    # 计算线程:LAB 分析 + 匹配循环
│   └── lab.js       # 主线程用的 LAB 工具(worker 内联复制)
├── CHANGELOG.md     # 设计决策 + 试错记录
└── README.md
```

## 浏览器要求

Chrome ≥ 94 / Edge ≥ 94 / Firefox ≥ 105 / Safari ≥ 16.4。
需要 OffscreenCanvas、`createImageBitmap` 的 resize 选项、Transferable ImageBitmap 全套。

## 限制

- 默认最多 8000 张底图(硬编码 `MAX_TILES`,可改)。
- 目标图建议不超过 4000px,内部只需要网格分块的低分辨率副本。
- 纯 JS,没有 SIMD。200×113 网格 × 3000 底图 估算 5-10 秒。
