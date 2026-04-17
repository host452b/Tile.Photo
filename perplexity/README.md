# Photomosaic 生成器

本地 Mac 玩具：把硬盘里一堆照片重组成另一张照片。

## Setup

```bash
cd perplexity
pip install -r requirements.txt
jupyter notebook mosaic.ipynb
```

## 使用

1. 把一堆照片放到 `perplexity/pool/` 目录（递归扫描，支持 jpg/png/webp/bmp）
2. 把目标图放到 `perplexity/target.jpg`
3. 打开 `mosaic.ipynb`，Run All
4. 看 Cell 4 的滑条：
   - **λ reuse**: tile 重复惩罚。大 → 每张 tile 最多用一次；0 → 谁近用谁
   - **μ neighbor**: 邻居相似惩罚。大 → 相邻 tile 颜色必须差异大
   - **τ transfer**: 色调迁移强度。0 → 照片原色（近看清晰），1 → 贴合目标（远看完美）
5. 点"预览 48×27"看效果，满意后点"正式渲染 120×68"

## 产物

- `output/mosaic_<ts>.png` — 成品图
- `output/report_<ts>.md` — 自嘲式报告（哪张 tile 被用最多、哪些冷宫）
- `output/deepzoom_<ts>/index.html` — 用浏览器双击打开，能无限放大

## 性能

第一次扫底图池要算 LAB 均值，`.cache/pool_features.pkl` 保存。之后按 mtime 增量。
3K 张池 + 120×68 网格 ≈ 10-30 秒完整渲染。

## 砍掉的东西（后续 V2/V3）

- CLIP 语义匹配（有 `semantic_reranker` 钩子预留）
- 底图打标签 + 叙事报告（"23% 来自 2019 日本旅行"）
- Cursed Mode 预设
- 实时"看它思考"动画
- 主体 saliency mask
- matplotlib CJK 字体（目前柱状图里 Chinese 是 tofu，MD 文字没事）

## 开发

```bash
pytest tests/ -v
```
