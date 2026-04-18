// Matching worker.
//
// Stateful: keeps tile LAB cache between runs so the user can tweak sliders
// and re-run without paying for tile analysis each time.
//
// Protocol (main → worker):
//   { type:'load-tiles', tiles:ImageBitmap[] }
//     → analyze, cache LABs, close bitmaps, reply 'tiles-ready'.
//   { type:'match', targetPatchLabs:Float32Array(W*H*3),
//     params:{gridW,gridH,lambda,mu,topK} }
//     → use cached LABs to run the match, reply 'done'.
//   { type:'reset' }
//     → drop the LAB cache (used when the tile pool changes).
//
// Protocol (worker → main):
//   { type:'stage', msg }
//   { type:'tile-progress', done, total }
//   { type:'tiles-ready', count }
//   { type:'match-batch', batch:[{cellIdx,x,y,pickedIdx,candidates:[{idx,colorDist,usePen,neighPen,total}]}] }
//   { type:'done', grid:Int32Array(W*H), uses:Int32Array(N) }
//   { type:'error', message }

// ---------- LAB utilities (duplicated from lab.js; worker has no import) ----------
const SRGB_TO_LINEAR = (() => {
  const lut = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const v = i / 255;
    lut[i] = v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  }
  return lut;
})();

const XN = 0.95047, YN = 1.0, ZN = 1.08883;
const LAB_EPS = 216 / 24389;
const LAB_KAPPA = 24389 / 27;

function fLab(t) {
  return t > LAB_EPS ? Math.cbrt(t) : (LAB_KAPPA * t + 16) / 116;
}

function rgbToLab(r, g, b) {
  const R = SRGB_TO_LINEAR[r];
  const G = SRGB_TO_LINEAR[g];
  const B = SRGB_TO_LINEAR[b];
  const X = 0.4124564 * R + 0.3575761 * G + 0.1804375 * B;
  const Y = 0.2126729 * R + 0.7151522 * G + 0.072175 * B;
  const Z = 0.0193339 * R + 0.119192 * G + 0.9503041 * B;
  const fx = fLab(X / XN);
  const fy = fLab(Y / YN);
  const fz = fLab(Z / ZN);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

// ---------- Top-K helper ----------
function topKSmallestIdx(scores, k) {
  const n = scores.length;
  if (k >= n) {
    const idx = new Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    return idx.sort((a, b) => scores[a] - scores[b]);
  }
  // Max-heap of size k
  const heap = new Int32Array(k);
  let size = 0;
  const swap = (i, j) => { const t = heap[i]; heap[i] = heap[j]; heap[j] = t; };
  const siftUp = (i) => {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (scores[heap[p]] < scores[heap[i]]) { swap(p, i); i = p; } else break;
    }
  };
  const siftDown = (i) => {
    while (true) {
      const l = 2 * i + 1, r = 2 * i + 2;
      let largest = i;
      if (l < size && scores[heap[l]] > scores[heap[largest]]) largest = l;
      if (r < size && scores[heap[r]] > scores[heap[largest]]) largest = r;
      if (largest === i) break;
      swap(i, largest);
      i = largest;
    }
  };
  for (let i = 0; i < n; i++) {
    if (size < k) { heap[size++] = i; siftUp(size - 1); }
    else if (scores[i] < scores[heap[0]]) { heap[0] = i; siftDown(0); }
  }
  return Array.from(heap.slice(0, size)).sort((a, b) => scores[a] - scores[b]);
}

// ---------- Tile analysis ----------
// Critical: must compute "mean color" the SAME way the target side does.
// Target side uses `drawImage(target, 0, 0, gridW, gridH)` — browser-native
// downsample, each output pixel = one patch's mean sRGB. We mirror that here:
// drawImage the tile to a small N×N canvas, then average those pixels' sRGB
// (pre-LAB). Averaging in sRGB space matches the target's downsample path and
// yields the tile's true mean color. Averaging in LAB (what v1 did) produces
// a virtual color that never existed in the tile — a half-red/half-blue tile
// would map to a purplish hue in LAB-mean but to real magenta in RGB-mean,
// and only the latter matches how the target patch was sampled.
async function analyzeTiles(tiles) {
  const N = tiles.length;
  const labs = new Float32Array(N * 3);
  const ANAL = 4;
  const canvas = new OffscreenCanvas(ANAL, ANAL);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('OffscreenCanvas 2d context unavailable');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  const TOTAL = ANAL * ANAL;

  for (let i = 0; i < N; i++) {
    const tile = tiles[i];
    ctx.clearRect(0, 0, ANAL, ANAL);
    ctx.drawImage(tile, 0, 0, ANAL, ANAL);
    const { data } = ctx.getImageData(0, 0, ANAL, ANAL);

    let sumR = 0, sumG = 0, sumB = 0;
    for (let p = 0; p < data.length; p += 4) {
      sumR += data[p];
      sumG += data[p + 1];
      sumB += data[p + 2];
    }
    const r = Math.round(sumR / TOTAL);
    const g = Math.round(sumG / TOTAL);
    const b = Math.round(sumB / TOTAL);
    const [L, aa, bb] = rgbToLab(r, g, b);
    labs[i * 3] = L;
    labs[i * 3 + 1] = aa;
    labs[i * 3 + 2] = bb;

    if ((i & 31) === 31 || i === N - 1) {
      self.postMessage({ type: 'tile-progress', done: i + 1, total: N });
    }
  }
  return labs;
}

// ---------- Worker-level cache ----------
let cachedTileLabs = null;  // Float32Array(N*3); null until 'load-tiles' completes.

// ---------- Matching loop ----------
function runMatch(tileLabs, targetLabs, gridW, gridH, lambda, mu, topK) {
  const N = tileLabs.length / 3;
  const cells = gridW * gridH;
  const grid = new Int32Array(cells);
  const uses = new Int32Array(N);
  const scores = new Float32Array(N);

  const BATCH = 32;
  let batch = [];

  self.postMessage({ type: 'stage', msg: `匹配中 · ${cells} 格 × ${N} 底图,top-${topK} 候选` });

  for (let cellIdx = 0; cellIdx < cells; cellIdx++) {
    const x = cellIdx % gridW;
    const y = (cellIdx / gridW) | 0;

    const tL = targetLabs[cellIdx * 3];
    const ta = targetLabs[cellIdx * 3 + 1];
    const tb = targetLabs[cellIdx * 3 + 2];

    // color distance to every tile
    for (let j = 0; j < N; j++) {
      const dL = tileLabs[j * 3] - tL;
      const da = tileLabs[j * 3 + 1] - ta;
      const db = tileLabs[j * 3 + 2] - tb;
      scores[j] = Math.sqrt(dL * dL + da * da + db * db);
    }

    const candIdxs = topKSmallestIdx(scores, topK);

    // neighbor ids (4-neighborhood, causal)
    const nTL = x > 0 && y > 0 ? grid[(y - 1) * gridW + (x - 1)] : -1;
    const nT = y > 0 ? grid[(y - 1) * gridW + x] : -1;
    const nTR = x < gridW - 1 && y > 0 ? grid[(y - 1) * gridW + (x + 1)] : -1;
    const nL = x > 0 ? grid[y * gridW + (x - 1)] : -1;

    let bestTotal = Infinity;
    let bestIdx = candIdxs[0];
    let bestUsePen = 0, bestNeighPen = 0, bestColor = scores[bestIdx];

    const rankedCandidates = [];
    for (const j of candIdxs) {
      const colorDist = scores[j];
      // sqrt(uses) rather than log(1+uses): the log curve is too flat past
      // a dozen reuses, letting a "universal filler" tile get stamped
      // everywhere with only mild pushback. sqrt grows noticeably faster
      // in the 10-100 range, which is where the slider should bite.
      const usePen = lambda * Math.sqrt(uses[j]);
      let matches = 0;
      if (j === nTL) matches++;
      if (j === nT) matches++;
      if (j === nTR) matches++;
      if (j === nL) matches++;
      const neighPen = mu * matches;
      const total = colorDist + usePen + neighPen;
      rankedCandidates.push({ idx: j, colorDist, usePen, neighPen, total });
      if (total < bestTotal) {
        bestTotal = total;
        bestIdx = j;
        bestUsePen = usePen;
        bestNeighPen = neighPen;
        bestColor = colorDist;
      }
    }
    rankedCandidates.sort((a, b) => a.total - b.total);

    grid[cellIdx] = bestIdx;
    uses[bestIdx]++;

    batch.push({
      cellIdx, x, y,
      pickedIdx: bestIdx,
      pickedUses: uses[bestIdx],
      candidates: rankedCandidates.slice(0, 3),
    });

    if (batch.length >= BATCH || cellIdx === cells - 1) {
      self.postMessage({ type: 'match-batch', batch, done: cellIdx + 1, total: cells });
      batch = [];
    }
  }

  return { grid, uses };
}

// ---------- Message dispatch ----------
self.onmessage = async (e) => {
  const msg = e.data;
  if (!msg) return;

  try {
    if (msg.type === 'load-tiles') {
      const { tiles } = msg;
      self.postMessage({ type: 'stage', msg: `分析 ${tiles.length} 张底图的平均色…` });
      cachedTileLabs = await analyzeTiles(tiles);
      // Bitmaps are done being read; release GPU memory.
      for (const bm of tiles) { try { bm.close?.(); } catch {} }
      self.postMessage({ type: 'tiles-ready', count: tiles.length });
      return;
    }

    if (msg.type === 'match') {
      if (!cachedTileLabs) {
        self.postMessage({ type: 'error', message: '底图还没加载到 worker' });
        return;
      }
      const { targetPatchLabs, params } = msg;
      const { gridW, gridH, lambda, mu, topK } = params;
      const { grid, uses } = runMatch(
        cachedTileLabs, targetPatchLabs, gridW, gridH, lambda, mu, topK
      );
      self.postMessage(
        { type: 'done', grid, uses },
        [grid.buffer, uses.buffer]
      );
      return;
    }

    if (msg.type === 'reset') {
      cachedTileLabs = null;
      return;
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
