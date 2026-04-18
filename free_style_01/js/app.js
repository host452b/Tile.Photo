// Tile.Photo — main-thread app.
// Owns: UI, file intake, target patch analysis, worker coordination, live rendering, report, zoom/pan.

import { rgbToLab } from './lab.js';

// ============================================================
//   Constants
// ============================================================
const TILE_BITMAP_SIZE = 40;   // resolution of each tile's ImageBitmap (used for both worker analysis + main render)
const TILE_RENDER_PX = 20;     // per-cell output pixel size on the final canvas
const TOPK = 50;               // color-neighbor candidates before penalty re-ranking
const MAX_TILES = 8000;        // hard cap
const UPLOAD_CONCURRENCY = 6;  // simultaneous image decodes

// ============================================================
//   State
// ============================================================
const state = {
  target: null,          // { file, bitmap, fullBitmap, width, height }
  tiles: [],             // [{ name, mainBitmap, workerBitmap }]
  worker: null,
  tilesLoadedInWorker: false, // true once 'load-tiles' round-trip completed
  running: false,
  output: null,          // { gridW, gridH, ctx, canvas, patchRgb }
  lastResult: null,      // { grid, uses }
  view: { zoom: 1, panX: 0, panY: 0, fitScale: 1 },
  params: {
    grid: 100,
    tau: 0.4,
    lambda: 0.3,
    mu: 0.5,
  },
  verbose: false,
};

// ============================================================
//   DOM shorthands
// ============================================================
const $ = (id) => document.getElementById(id);
const els = {
  targetDrop: $('target-drop'),
  targetInput: $('target-input'),
  targetPick: $('target-pick'),
  targetPlaceholder: $('target-placeholder'),
  targetPreviewWrap: $('target-preview-wrap'),
  targetCanvas: $('target-canvas'),
  targetMeta: $('target-meta'),
  targetReset: $('target-reset'),

  tilesDrop: $('tiles-drop'),
  tilesInput: $('tiles-input'),
  tilesDirInput: $('tiles-dir-input'),
  tilesPickFiles: $('tiles-pick-files'),
  tilesPickDir: $('tiles-pick-dir'),
  tilesPlaceholder: $('tiles-placeholder'),
  tilesStatsWrap: $('tiles-stats-wrap'),
  tilesCount: $('tiles-count'),
  tilesBytes: $('tiles-bytes'),
  tilesReset: $('tiles-reset'),
  tilesGrid: $('tiles-grid'),
  tilesProgress: $('tiles-progress'),
  tilesBar: $('tiles-bar'),
  tilesProgressText: $('tiles-progress-text'),
  tilesRec: $('tiles-rec'),

  sysStatus: $('sys-status'),
  sysStatusText: $('sys-status-text'),

  gridInput: $('grid-input'),
  gridValue: $('grid-value'),
  tauInput: $('tau-input'),
  tauValue: $('tau-value'),
  lambdaInput: $('lambda-input'),
  lambdaValue: $('lambda-value'),
  muInput: $('mu-input'),
  muValue: $('mu-value'),

  runBtn: $('run-btn'),
  cancelBtn: $('cancel-btn'),
  actionStatus: $('action-status'),

  stepOutput: $('step-output'),
  stepReport: $('step-report'),
  canvasWrap: $('canvas-wrap'),
  outputCanvas: $('output-canvas'),
  zoomFit: $('zoom-fit'),
  zoom1x: $('zoom-1x'),
  zoom4x: $('zoom-4x'),
  zoomReadout: $('zoom-readout'),
  savePng: $('save-png'),

  progressText: $('progress-text'),
  logStream: $('log-stream'),
  logVerbose: $('log-verbose'),

  reportBody: $('report-body'),
};

// ============================================================
//   Utility
// ============================================================
function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }

// Reflect the high-level phase via the header system-status LED.
// States: ready, ingest, compute, done, fault.
function setSys(state, text) {
  if (!els.sysStatus) return;
  els.sysStatus.dataset.state = state;
  if (text) els.sysStatusText.textContent = text;
}

async function extractFilesFromDrop(dataTransfer) {
  const items = dataTransfer.items;
  if (!items) return [...(dataTransfer.files || [])];
  const out = [];
  const promises = [];
  for (const item of items) {
    if (item.kind !== 'file') continue;
    const entry = typeof item.webkitGetAsEntry === 'function' ? item.webkitGetAsEntry() : null;
    if (entry) {
      promises.push(walkEntry(entry, out));
    } else {
      const f = item.getAsFile();
      if (f) out.push(f);
    }
  }
  await Promise.all(promises);
  return out;
}

async function walkEntry(entry, acc) {
  if (entry.isFile) {
    await new Promise((resolve) => {
      entry.file((file) => { acc.push(file); resolve(); }, () => resolve());
    });
  } else if (entry.isDirectory) {
    const reader = entry.createReader();
    // readEntries returns at most ~100 entries at once; loop until empty.
    while (true) {
      const batch = await new Promise((resolve) => {
        reader.readEntries(resolve, () => resolve([]));
      });
      if (!batch.length) break;
      for (const e of batch) await walkEntry(e, acc);
    }
  }
}

function isImageFile(f) {
  if (!f) return false;
  if (f.type && f.type.startsWith('image/')) return true;
  // some dragged files have empty type; fall back to extension
  return /\.(png|jpe?g|webp|gif|bmp|avif)$/i.test(f.name || '');
}

// Decode one image file into two 40×40 ImageBitmaps, center-cropped to square.
async function decodeTile(file) {
  const full = await createImageBitmap(file);
  try {
    const size = Math.min(full.width, full.height);
    const sx = ((full.width - size) / 2) | 0;
    const sy = ((full.height - size) / 2) | 0;
    const [mainBm, workerBm] = await Promise.all([
      createImageBitmap(full, sx, sy, size, size, {
        resizeWidth: TILE_BITMAP_SIZE,
        resizeHeight: TILE_BITMAP_SIZE,
        resizeQuality: 'high',
      }),
      createImageBitmap(full, sx, sy, size, size, {
        resizeWidth: TILE_BITMAP_SIZE,
        resizeHeight: TILE_BITMAP_SIZE,
        resizeQuality: 'high',
      }),
    ]);
    return { name: file.name, mainBitmap: mainBm, workerBitmap: workerBm, bytes: file.size };
  } finally {
    full.close?.();
  }
}

// ============================================================
//   Target upload
// ============================================================
async function handleTargetFiles(files) {
  const f = files.find(isImageFile);
  if (!f) return;
  const bitmap = await createImageBitmap(f);
  state.target = { file: f, bitmap, width: bitmap.width, height: bitmap.height };
  renderTargetPreview();
  updateGridLabel();
  updateTileRecommendation();
  updateRunButton();
}

function renderTargetPreview() {
  const t = state.target;
  els.targetPlaceholder.hidden = true;
  els.targetPreviewWrap.hidden = false;
  const canvas = els.targetCanvas;
  const ctx = canvas.getContext('2d');
  const maxW = 320, maxH = 220;
  const scale = Math.min(maxW / t.width, maxH / t.height);
  canvas.width = Math.round(t.width * scale);
  canvas.height = Math.round(t.height * scale);
  ctx.drawImage(t.bitmap, 0, 0, canvas.width, canvas.height);
  els.targetMeta.innerHTML =
    `<strong>${escapeHtml(t.file.name)}</strong><br>` +
    `${t.width} × ${t.height} px · ${fmtBytes(t.file.size)}`;
}

function resetTarget() {
  if (state.target?.bitmap) state.target.bitmap.close?.();
  state.target = null;
  els.targetInput.value = '';
  els.targetPlaceholder.hidden = false;
  els.targetPreviewWrap.hidden = true;
  updateGridLabel();
  updateTileRecommendation();
  updateRunButton();
}

// ============================================================
//   Tile upload
// ============================================================
async function handleTileFiles(files) {
  const imgs = files.filter(isImageFile);
  if (!imgs.length) return;
  if (imgs.length > MAX_TILES) {
    log('warn', `⚠ 底图太多(${imgs.length}),截断到 ${MAX_TILES}。`);
    imgs.length = MAX_TILES;
  }

  els.tilesPlaceholder.hidden = true;
  els.tilesProgress.hidden = false;
  els.tilesStatsWrap.hidden = true;

  setSys('ingest', 'INGEST');

  // Pool is about to change — any cached LAB analysis in the worker is now stale.
  invalidateTilesCache();

  const total = imgs.length;
  let done = 0;
  let totalBytes = 0;
  state.tiles = [];

  // Concurrency-limited decode
  const queue = imgs.slice();
  const runners = Array.from({ length: UPLOAD_CONCURRENCY }, async () => {
    while (queue.length) {
      const file = queue.shift();
      if (!file) break;
      try {
        const tile = await decodeTile(file);
        state.tiles.push(tile);
        totalBytes += tile.bytes;
      } catch (err) {
        console.warn('tile decode failed:', file.name, err);
      }
      done++;
      if (done % 8 === 0 || done === total) {
        els.tilesBar.style.width = `${(100 * done / total).toFixed(1)}%`;
        els.tilesProgressText.textContent = `解码中 · ${done} / ${total}`;
      }
    }
  });
  await Promise.all(runners);

  els.tilesProgress.hidden = true;
  renderTilePalette(totalBytes);
  updateTileRecommendation();
  updateRunButton();
  setSys('ready', 'READY');
}

function renderTilePalette(totalBytes) {
  els.tilesStatsWrap.hidden = false;
  els.tilesCount.textContent = state.tiles.length;
  els.tilesBytes.textContent = totalBytes ? `· ${fmtBytes(totalBytes)}` : '';

  const grid = els.tilesGrid;
  grid.innerHTML = '';
  // Show up to 160 thumbnails; rest is memory only
  const showN = Math.min(state.tiles.length, 160);
  const frag = document.createDocumentFragment();
  for (let i = 0; i < showN; i++) {
    const c = document.createElement('canvas');
    c.width = TILE_BITMAP_SIZE;
    c.height = TILE_BITMAP_SIZE;
    c.title = state.tiles[i].name;
    c.getContext('2d').drawImage(state.tiles[i].mainBitmap, 0, 0);
    frag.appendChild(c);
  }
  grid.appendChild(frag);
}

function resetTiles() {
  for (const t of state.tiles) {
    t.mainBitmap?.close?.();
    t.workerBitmap?.close?.();
  }
  state.tiles = [];
  els.tilesInput.value = '';
  els.tilesDirInput.value = '';
  els.tilesPlaceholder.hidden = false;
  els.tilesStatsWrap.hidden = true;
  els.tilesProgress.hidden = true;
  invalidateTilesCache();
  updateTileRecommendation();
  updateRunButton();
}

// ============================================================
//   Params — sliders
// ============================================================
function updateGridLabel() {
  const w = Number(els.gridInput.value);
  const ratio = state.target ? state.target.width / state.target.height : 16 / 9;
  const h = Math.max(4, Math.round(w / ratio));
  els.gridValue.textContent = `${w} × ${h}`;
  state.params.grid = w;
  updateTileRecommendation();
}

// Size recommendations derived from (# of cells the mosaic will have).
// Heuristics are deliberately conservative — it's fine to run with fewer,
// just uglier. Thresholds are: minimum to "not look like a handful of tiles
// photocopied everywhere", recommended for decent variety, ideal for ≤3× reuse.
function tileCountThresholds() {
  if (!state.target) return null;
  const w = Number(els.gridInput.value);
  const ratio = state.target.width / state.target.height;
  const h = Math.max(4, Math.round(w / ratio));
  const cells = w * h;
  return {
    gridW: w,
    gridH: h,
    cells,
    min: Math.max(50, Math.round(cells / 30)),
    rec: Math.max(200, Math.round(cells / 8)),
    ideal: Math.max(800, Math.round(cells / 3)),
  };
}

function updateTileRecommendation() {
  const el = els.tilesRec;
  if (!el) return;
  const th = tileCountThresholds();
  const have = state.tiles.length;

  if (!th) {
    el.textContent = have
      ? `已有 ${have} 张 · 把目标图丢进 01,再告诉你具体要多少张。`
      : '先把目标图丢进上面的 01,我好告诉你到底需要多少张。';
    el.dataset.level = have ? 'info' : 'info';
    return;
  }

  const { gridW, gridH, cells, min, rec, ideal } = th;
  const base = `目标 ${gridW}×${gridH} = ${cells} 格 · 最低 ${min} / 推荐 ${rec} / 理想 ${ideal}`;

  if (have === 0) {
    el.textContent = base + ' · 建议整文件夹拖进来。';
    el.dataset.level = 'info';
  } else if (have < min) {
    el.textContent = `${base} · ⚠ 你只有 ${have} 张,远低于最低 ${min},会严重重复。`;
    el.dataset.level = 'warn';
  } else if (have < rec) {
    el.textContent = `${base} · · 你有 ${have} 张,够跑但不够多样,再加到 ${rec}+ 会好很多。`;
    el.dataset.level = 'ok';
  } else if (have < ideal) {
    el.textContent = `${base} · ✓ 你有 ${have} 张,多样性够用。`;
    el.dataset.level = 'good';
  } else {
    el.textContent = `${base} · ✓ 你有 ${have} 张,充裕。`;
    el.dataset.level = 'good';
  }
}

function bindSliders() {
  els.gridInput.addEventListener('input', updateGridLabel);

  els.tauInput.addEventListener('input', () => {
    state.params.tau = Number(els.tauInput.value) / 100;
    els.tauValue.textContent = `${els.tauInput.value}%`;
  });
  els.lambdaInput.addEventListener('input', () => {
    state.params.lambda = Number(els.lambdaInput.value) / 100;
    els.lambdaValue.textContent = state.params.lambda.toFixed(2);
  });
  els.muInput.addEventListener('input', () => {
    state.params.mu = Number(els.muInput.value) / 100;
    els.muValue.textContent = state.params.mu.toFixed(2);
  });
}

function updateRunButton() {
  const ok = state.target && state.tiles.length >= 8 && !state.running;
  els.runBtn.disabled = !ok;
  if (!state.target && !state.tiles.length) els.actionStatus.textContent = '等两边都就绪';
  else if (!state.target) els.actionStatus.textContent = '还差目标图';
  else if (state.tiles.length < 8) els.actionStatus.textContent = `还差底图(至少 8 张,当前 ${state.tiles.length})`;
  else els.actionStatus.textContent = `已就绪 · ${state.tiles.length} 张底图`;
}

// ============================================================
//   Build pipeline
// ============================================================

// Lazily spawn a long-lived worker. A single worker persists across runs so
// the tile LAB cache survives; terminating + respawning would defeat the
// whole point of caching.
function ensureWorker() {
  if (state.worker) return state.worker;
  const w = new Worker('js/worker.js', { type: 'classic' });
  w.onmessage = (ev) => handleWorkerMessage(ev.data);
  w.onerror = (e) => {
    log('error', `⚠ Worker error: ${e.message}`);
    stopBuild();
  };
  state.worker = w;
  return w;
}

// Any workerBitmap that's been transferred to the worker on a prior run is
// now detached (null). Clone fresh copies from mainBitmap so we can send
// them again. mainBitmap stays on this side untouched.
async function ensureWorkerBitmaps() {
  const missing = state.tiles.filter((t) => !t.workerBitmap);
  if (!missing.length) return;
  const total = missing.length;
  let done = 0;
  els.actionStatus.textContent = `准备 worker 副本… 0 / ${total}`;

  const queue = missing.slice();
  const runners = Array.from({ length: UPLOAD_CONCURRENCY }, async () => {
    while (queue.length) {
      const t = queue.shift();
      if (!t) break;
      try {
        t.workerBitmap = await createImageBitmap(t.mainBitmap);
      } catch (e) {
        console.warn('clone failed:', t.name, e);
      }
      done++;
      if (done % 16 === 0 || done === total) {
        els.actionStatus.textContent = `准备 worker 副本… ${done} / ${total}`;
      }
    }
  });
  await Promise.all(runners);

  const failed = state.tiles.filter((t) => !t.workerBitmap);
  if (failed.length) {
    throw new Error(`${failed.length} 张底图无法生成 worker 副本,请点 Step 02 的"清空重传"。`);
  }
}

// Wait for a specific message type on a worker. The main onmessage handler
// still fires for every message — this listener only resolves/rejects
// at the sentinel, it does NOT reprocess anything.
function waitForWorkerMessage(worker, matchType) {
  return new Promise((resolve, reject) => {
    const onMsg = (ev) => {
      if (ev.data?.type === matchType) {
        worker.removeEventListener('message', onMsg);
        resolve(ev.data);
      } else if (ev.data?.type === 'error') {
        worker.removeEventListener('message', onMsg);
        reject(new Error(ev.data.message));
      }
    };
    worker.addEventListener('message', onMsg);
  });
}

function invalidateTilesCache() {
  state.tilesLoadedInWorker = false;
  if (state.worker) {
    state.worker.postMessage({ type: 'reset' });
  }
}

async function startBuild() {
  if (state.running || !state.target || !state.tiles.length) return;
  state.running = true;
  els.runBtn.disabled = true;
  els.cancelBtn.hidden = false;
  els.stepOutput.hidden = false;
  els.stepReport.hidden = true;
  els.savePng.disabled = true;
  clearLog();
  setSys('compute', 'COMPUTE');

  try {
    const { gridW, gridH } = computeGridDims();
    log('stage', `目标 ${state.target.width}×${state.target.height}px → 网格 ${gridW}×${gridH}`);

    // Phase A (main): derive target patch means. Cheap; always re-done.
    const patchLab = new Float32Array(gridW * gridH * 3);
    const patchRgb = new Uint8ClampedArray(gridW * gridH * 3);
    computeTargetPatches(state.target.bitmap, gridW, gridH, patchLab, patchRgb);
    log('stage', `目标分块完成,${gridW * gridH} 格。`);

    // Phase B (main): init output canvas with the target as a faint backdrop.
    const cW = gridW * TILE_RENDER_PX;
    const cH = gridH * TILE_RENDER_PX;
    const canvas = els.outputCanvas;
    canvas.width = cW;
    canvas.height = cH;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(state.target.bitmap, 0, 0, cW, cH);
    ctx.fillStyle = 'rgba(10,10,12,0.45)';
    ctx.fillRect(0, 0, cW, cH);

    state.output = { gridW, gridH, ctx, canvas, patchRgb, patchLab };
    fitCanvasToViewport();

    // Phase C: ensure worker + tile cache.
    const worker = ensureWorker();
    if (!state.tilesLoadedInWorker) {
      log('stage', `首次运行:把 ${state.tiles.length} 张底图送到 worker。`);
      await ensureWorkerBitmaps();
      const tileBitmaps = state.tiles.map((t) => t.workerBitmap);
      worker.postMessage(
        { type: 'load-tiles', tiles: tileBitmaps },
        tileBitmaps
      );
      for (const t of state.tiles) t.workerBitmap = null;
      await waitForWorkerMessage(worker, 'tiles-ready');
      state.tilesLoadedInWorker = true;
    } else {
      log('stage', `沿用上一轮的底图缓存,直接重新匹配。`);
    }

    // Phase D: fire the match.
    worker.postMessage(
      {
        type: 'match',
        targetPatchLabs: patchLab,
        params: {
          gridW, gridH,
          lambda: state.params.lambda,
          mu: state.params.mu,
          topK: Math.min(TOPK, state.tiles.length),
        },
      },
      [patchLab.buffer]
    );
  } catch (err) {
    log('error', `⚠ ${err.message || err}`);
    stopBuild();
  }
}

function computeGridDims() {
  const w = state.params.grid;
  const ratio = state.target.width / state.target.height;
  const h = Math.max(4, Math.round(w / ratio));
  return { gridW: w, gridH: h };
}

function computeTargetPatches(bitmap, gridW, gridH, outLab, outRgb) {
  const oc = new OffscreenCanvas(gridW, gridH);
  const c = oc.getContext('2d');
  c.imageSmoothingEnabled = true;
  c.imageSmoothingQuality = 'high';
  c.drawImage(bitmap, 0, 0, gridW, gridH);
  const { data } = c.getImageData(0, 0, gridW, gridH);
  for (let i = 0; i < gridW * gridH; i++) {
    const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
    outRgb[i * 3] = r;
    outRgb[i * 3 + 1] = g;
    outRgb[i * 3 + 2] = b;
    const [L, aa, bb] = rgbToLab(r, g, b);
    outLab[i * 3] = L;
    outLab[i * 3 + 1] = aa;
    outLab[i * 3 + 2] = bb;
  }
}

function stopBuild() {
  state.running = false;
  els.runBtn.disabled = false;
  els.cancelBtn.hidden = true;
  updateRunButton();
}

function cancelBuild() {
  if (!state.running) return;
  if (state.worker) state.worker.terminate();
  state.worker = null;
  state.tilesLoadedInWorker = false;  // cache died with the worker
  log('warn', '⨉ 已中断。');
  setSys('ready', 'READY');
  stopBuild();
}

// ============================================================
//   Worker message handling + live render
// ============================================================
function handleWorkerMessage(msg) {
  switch (msg.type) {
    case 'stage':
      log('stage', msg.msg);
      break;
    case 'tile-progress':
      els.progressText.textContent = `底图分析 · ${msg.done} / ${msg.total}`;
      break;
    case 'tiles-ready':
      log('stage', `底图分析完成 · ${msg.count} 张已缓存,后续改参秒级重跑。`);
      break;
    case 'match-batch':
      renderMatchBatch(msg.batch);
      els.progressText.textContent = `拼图中 · ${msg.done} / ${msg.total}`;
      break;
    case 'done':
      handleDone(msg);
      break;
    case 'error':
      log('error', `⚠ ${msg.message}`);
      setSys('fault', 'FAULT');
      stopBuild();
      break;
  }
}

function renderMatchBatch(batch) {
  const { ctx, patchRgb } = state.output;
  const tau = state.params.tau;

  for (const cell of batch) {
    const { x, y, cellIdx, pickedIdx, pickedUses, candidates } = cell;
    const tile = state.tiles[pickedIdx];
    if (!tile || !tile.mainBitmap) continue;

    const cx = x * TILE_RENDER_PX;
    const cy = y * TILE_RENDER_PX;
    ctx.drawImage(tile.mainBitmap, cx, cy, TILE_RENDER_PX, TILE_RENDER_PX);

    if (tau > 0) {
      const r = patchRgb[cellIdx * 3];
      const g = patchRgb[cellIdx * 3 + 1];
      const b = patchRgb[cellIdx * 3 + 2];
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.globalAlpha = tau;
      ctx.fillRect(cx, cy, TILE_RENDER_PX, TILE_RENDER_PX);
      ctx.globalAlpha = 1;
    }

    if (cellIdx % 17 === 0 || candidates[0].total > 20) {
      const pick = candidates[0];
      log(
        'pick',
        `(${pad(x, 3)},${pad(y, 3)}) ← ${trunc(tile.name, 28)}  ` +
        `ΔE=${pick.colorDist.toFixed(1)} uses=${pickedUses} ` +
        `λ=${pick.usePen.toFixed(2)} μ=${pick.neighPen.toFixed(2)}`
      );
      if (state.verbose && candidates.length > 1) {
        for (let i = 1; i < candidates.length; i++) {
          const c = candidates[i];
          const nm = state.tiles[c.idx]?.name ?? `#${c.idx}`;
          log(
            'reject',
            `            ⨯ ${trunc(nm, 28)}  ` +
            `ΔE=${c.colorDist.toFixed(1)} λ=${c.usePen.toFixed(2)} μ=${c.neighPen.toFixed(2)} tot=${c.total.toFixed(1)}`
          );
        }
      }
    }
  }
}

function handleDone(msg) {
  const { grid, uses } = msg;
  state.lastResult = { grid, uses };
  log('done', `✓ 完成。${grid.length} 格 / ${state.tiles.length} 底图。`);
  els.savePng.disabled = false;
  setSys('done', 'DONE');
  stopBuild();
  generateReport(grid, uses);
  els.stepReport.hidden = false;
}

// ============================================================
//   Logging
// ============================================================
const pendingLogs = [];
let logFlushScheduled = false;

function log(kind, text) {
  pendingLogs.push({ kind, text });
  if (!logFlushScheduled) {
    logFlushScheduled = true;
    requestAnimationFrame(flushLogs);
  }
}

function flushLogs() {
  logFlushScheduled = false;
  if (!pendingLogs.length) return;
  const frag = document.createDocumentFragment();
  for (const { kind, text } of pendingLogs) {
    const div = document.createElement('div');
    div.className = `log-line log-${kind}`;
    div.textContent = text;
    frag.appendChild(div);
  }
  pendingLogs.length = 0;
  const stream = els.logStream;
  stream.appendChild(frag);
  // cap lines
  const lines = stream.children;
  const MAX_LINES = 1200;
  if (lines.length > MAX_LINES) {
    const excess = lines.length - MAX_LINES;
    for (let i = 0; i < excess; i++) stream.removeChild(stream.firstChild);
  }
  stream.scrollTop = stream.scrollHeight;
}

function clearLog() {
  els.logStream.innerHTML = '';
  pendingLogs.length = 0;
}

function pad(n, w) {
  return String(n).padStart(w, ' ');
}
function trunc(s, n) {
  if (s.length <= n) return s.padEnd(n, ' ');
  return s.slice(0, n - 1) + '…';
}
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ============================================================
//   Canvas zoom + pan
// ============================================================
function applyTransform() {
  const c = els.outputCanvas;
  c.style.transform = `translate(${state.view.panX}px, ${state.view.panY}px) scale(${state.view.zoom})`;
  els.zoomReadout.textContent = `${Math.round(state.view.zoom * 100)}%`;
}

function fitCanvasToViewport() {
  const wrap = els.canvasWrap;
  const c = els.outputCanvas;
  if (!c.width || !c.height) return;
  const rect = wrap.getBoundingClientRect();
  const scale = Math.min(rect.width / c.width, rect.height / c.height) * 0.98;
  state.view.zoom = scale;
  state.view.fitScale = scale;
  state.view.panX = (rect.width - c.width * scale) / 2;
  state.view.panY = (rect.height - c.height * scale) / 2;
  applyTransform();
}

function bindCanvasInteractions() {
  const wrap = els.canvasWrap;

  wrap.addEventListener('wheel', (e) => {
    if (!els.outputCanvas.width) return;
    e.preventDefault();
    const rect = wrap.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY > 0 ? 1 / 1.12 : 1.12;
    const newZoom = clamp(state.view.zoom * factor, 0.1, 12);
    state.view.panX = mx - (mx - state.view.panX) * (newZoom / state.view.zoom);
    state.view.panY = my - (my - state.view.panY) * (newZoom / state.view.zoom);
    state.view.zoom = newZoom;
    applyTransform();
  }, { passive: false });

  let dragging = false, startX = 0, startY = 0, startPanX = 0, startPanY = 0;
  wrap.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    dragging = true;
    wrap.classList.add('is-dragging');
    startX = e.clientX; startY = e.clientY;
    startPanX = state.view.panX; startPanY = state.view.panY;
  });
  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    state.view.panX = startPanX + (e.clientX - startX);
    state.view.panY = startPanY + (e.clientY - startY);
    applyTransform();
  });
  window.addEventListener('mouseup', () => {
    dragging = false;
    wrap.classList.remove('is-dragging');
  });

  els.zoomFit.addEventListener('click', fitCanvasToViewport);
  els.zoom1x.addEventListener('click', () => {
    const rect = wrap.getBoundingClientRect();
    state.view.zoom = 1;
    state.view.panX = (rect.width - els.outputCanvas.width) / 2;
    state.view.panY = (rect.height - els.outputCanvas.height) / 2;
    applyTransform();
  });
  els.zoom4x.addEventListener('click', () => {
    const rect = wrap.getBoundingClientRect();
    state.view.zoom = 4;
    state.view.panX = (rect.width - els.outputCanvas.width * 4) / 2;
    state.view.panY = (rect.height - els.outputCanvas.height * 4) / 2;
    applyTransform();
  });

  window.addEventListener('resize', () => {
    if (els.outputCanvas.width) fitCanvasToViewport();
  });
}

// ============================================================
//   Report
// ============================================================
function generateReport(grid, uses) {
  const N = state.tiles.length;
  const total = grid.length;
  const used = new Set();
  for (let i = 0; i < grid.length; i++) used.add(grid[i]);
  const usedN = used.size;
  const frozenN = N - usedN;

  // Rankings
  const entries = [];
  for (let i = 0; i < N; i++) entries.push({ idx: i, uses: uses[i], name: state.tiles[i].name });
  entries.sort((a, b) => b.uses - a.uses);
  const hot = entries.slice(0, 10);
  const frozen = entries.filter((e) => e.uses === 0).slice(0, 50);

  // Self-deprecating story
  const topHot = hot[0];
  const storyParts = [];
  storyParts.push(`本次共摆了 <span class="kbd">${total}</span> 格,动用了 <span class="kbd">${usedN} / ${N}</span> 张底图。`);
  if (topHot && topHot.uses > 0) {
    const pct = (100 * topHot.uses / total).toFixed(1);
    storyParts.push(`劳模:<span class="kbd">${escapeHtml(trunc(topHot.name, 36).trim())}</span> 被用了 <span class="kbd">${topHot.uses}</span> 次(${pct}%),估计是万能填空。`);
  }
  if (frozenN > 0) {
    storyParts.push(`被打入冷宫的有 <span class="kbd">${frozenN}</span> 张。它们可能太糊、太黑、或者色彩太单一。`);
  } else {
    storyParts.push(`神奇,所有 ${N} 张底图都被用上了,一张都没浪费。`);
  }

  // Render
  els.reportBody.innerHTML = `
    <div class="report-story">${storyParts.join(' ')}</div>
    <div class="report-grid">
      <div class="report-block">
        <h3>劳模 TOP 10</h3>
        <ul class="rank-list" id="rank-hot"></ul>
      </div>
      <div class="report-block">
        <h3>使用次数热力图 (${state.output.gridW} × ${state.output.gridH})</h3>
        <div class="heatmap-wrap">
          <canvas id="heatmap-canvas"></canvas>
          <div class="heatmap-legend">
            <span>0</span><span>→</span><span id="heatmap-max">0</span>
          </div>
        </div>
      </div>
      <div class="report-block" style="grid-column: 1 / -1;">
        <h3>冷宫照片墙 (${frozenN})</h3>
        <div class="frozen-wall" id="frozen-wall"></div>
      </div>
    </div>
  `;

  // Hot list with thumbnails
  const hotUL = $('rank-hot');
  hot.filter((e) => e.uses > 0).forEach((e, i) => {
    const li = document.createElement('li');
    li.innerHTML = `
      <span class="rank-num">${i + 1}</span>
      <canvas class="rank-thumb" width="${TILE_BITMAP_SIZE}" height="${TILE_BITMAP_SIZE}"></canvas>
      <span class="rank-name" title="${escapeHtml(e.name)}">${escapeHtml(e.name)}</span>
      <span class="rank-count">${e.uses}</span>
    `;
    const cv = li.querySelector('canvas');
    cv.getContext('2d').drawImage(state.tiles[e.idx].mainBitmap, 0, 0);
    hotUL.appendChild(li);
  });

  // Heatmap
  renderHeatmap(grid, uses);

  // Frozen wall
  const wall = $('frozen-wall');
  frozen.forEach((e) => {
    const c = document.createElement('canvas');
    c.width = TILE_BITMAP_SIZE;
    c.height = TILE_BITMAP_SIZE;
    c.title = e.name;
    c.getContext('2d').drawImage(state.tiles[e.idx].mainBitmap, 0, 0);
    wall.appendChild(c);
  });
}

function renderHeatmap(grid, uses) {
  const { gridW, gridH } = state.output;
  const canvas = $('heatmap-canvas');
  const displayScale = Math.max(2, Math.min(6, Math.floor(720 / gridW)));
  canvas.width = gridW * displayScale;
  canvas.height = gridH * displayScale;
  const ctx = canvas.getContext('2d');
  let maxU = 1;
  for (let i = 0; i < grid.length; i++) {
    const u = uses[grid[i]];
    if (u > maxU) maxU = u;
  }
  $('heatmap-max').textContent = String(maxU);
  const img = ctx.createImageData(gridW, gridH);
  for (let i = 0; i < grid.length; i++) {
    const u = uses[grid[i]];
    const t = Math.min(1, u / maxU);
    // Dark → accent orange gradient
    const r = Math.round(20 + (255 - 20) * t);
    const g = Math.round(20 + (107 - 20) * t);
    const b = Math.round(25 + (53 - 25) * t);
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  // Draw small image then scale up with nearest-neighbor
  const tmp = new OffscreenCanvas(gridW, gridH);
  tmp.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
}

// ============================================================
//   Save PNG
// ============================================================
function savePNG() {
  const c = els.outputCanvas;
  c.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const stamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 13);
    a.download = `tilephoto-${stamp}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }, 'image/png');
}

// ============================================================
//   Wire up
// ============================================================
function bindDropzone(el, handler, { allowFolders = false } = {}) {
  el.addEventListener('dragover', (e) => {
    e.preventDefault();
    el.classList.add('drag-over');
  });
  el.addEventListener('dragleave', () => el.classList.remove('drag-over'));
  el.addEventListener('drop', async (e) => {
    e.preventDefault();
    el.classList.remove('drag-over');
    const files = allowFolders
      ? await extractFilesFromDrop(e.dataTransfer)
      : [...(e.dataTransfer.files || [])];
    handler(files);
  });
}

function bindUploads() {
  // Target
  els.targetInput.addEventListener('change', (e) => {
    handleTargetFiles([...e.target.files]);
  });
  els.targetPick.addEventListener('click', (e) => {
    e.preventDefault();
    els.targetInput.click();
  });
  // Clicking anywhere on the target drop (except buttons) opens picker.
  els.targetDrop.addEventListener('click', (e) => {
    if (e.target.closest('button')) return;
    if (state.target) return;
    els.targetInput.click();
  });
  bindDropzone(els.targetDrop, handleTargetFiles);
  els.targetReset.addEventListener('click', resetTarget);

  // Tiles
  els.tilesInput.addEventListener('change', (e) => handleTileFiles([...e.target.files]));
  els.tilesDirInput.addEventListener('change', (e) => handleTileFiles([...e.target.files]));
  els.tilesPickFiles.addEventListener('click', (e) => { e.preventDefault(); els.tilesInput.click(); });
  els.tilesPickDir.addEventListener('click', (e) => { e.preventDefault(); els.tilesDirInput.click(); });
  els.tilesDrop.addEventListener('click', (e) => {
    if (e.target.closest('button')) return;
    if (state.tiles.length) return;
    els.tilesInput.click();
  });
  bindDropzone(els.tilesDrop, handleTileFiles, { allowFolders: true });
  els.tilesReset.addEventListener('click', resetTiles);
}

function bindActions() {
  els.runBtn.addEventListener('click', startBuild);
  els.cancelBtn.addEventListener('click', cancelBuild);
  els.savePng.addEventListener('click', savePNG);
  els.logVerbose.addEventListener('change', (e) => {
    state.verbose = e.target.checked;
  });
}

function init() {
  bindUploads();
  bindSliders();
  bindActions();
  bindCanvasInteractions();

  // Initialize slider derived state
  state.params.tau = Number(els.tauInput.value) / 100;
  state.params.lambda = Number(els.lambdaInput.value) / 100;
  state.params.mu = Number(els.muInput.value) / 100;
  els.tauValue.textContent = `${els.tauInput.value}%`;
  els.lambdaValue.textContent = state.params.lambda.toFixed(2);
  els.muValue.textContent = state.params.mu.toFixed(2);
  updateGridLabel();
  updateTileRecommendation();
  updateRunButton();

  // Feature probe
  if (typeof OffscreenCanvas === 'undefined') {
    els.actionStatus.textContent = '⚠ 浏览器不支持 OffscreenCanvas,请用 Chrome/Edge/Firefox 最新版。';
  }
}

init();
