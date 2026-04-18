// LAB color space + small numerical utilities.
// Exposed in two flavors: ES module (main thread) and classic worker-global (see worker.js).

// sRGB [0..255] → linear RGB [0..1] lookup table (hot path).
const SRGB_TO_LINEAR = (() => {
  const lut = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const v = i / 255;
    lut[i] = v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  }
  return lut;
})();

// D65 reference white
const XN = 0.95047;
const YN = 1.0;
const ZN = 1.08883;

// LAB f(t) threshold — (6/29)^3
const LAB_EPS = 216 / 24389; // 0.00885645...
const LAB_KAPPA = 24389 / 27; // 903.2962963...

function fLab(t) {
  return t > LAB_EPS ? Math.cbrt(t) : (LAB_KAPPA * t + 16) / 116;
}

export function rgbToLab(r, g, b) {
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

// Given an RGBA ImageData.data-like Uint8ClampedArray (4 bytes per pixel),
// compute mean LAB and mean RGB over the whole array.
// Returns {L, a, b, r, g, _b} — `_b` because `b` collides with LAB.
export function meanLabAndRgbFromRGBA(data) {
  const n = data.length / 4;
  let sumR = 0, sumG = 0, sumB = 0;
  let sumL = 0, sumA = 0, sumBB = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    sumR += r;
    sumG += g;
    sumB += b;
    const [L, aa, bb] = rgbToLab(r, g, b);
    sumL += L;
    sumA += aa;
    sumBB += bb;
  }
  return {
    L: sumL / n,
    a: sumA / n,
    b: sumBB / n,
    r: sumR / n,
    g: sumG / n,
    _b: sumB / n,
  };
}

// ΔE — plain euclidean distance in LAB. CIEDE2000 is better but much heavier;
// for photomosaic ranking this is sufficient.
export function labDistance(l1, a1, b1, l2, a2, b2) {
  const dL = l1 - l2;
  const da = a1 - a2;
  const db = b1 - b2;
  return Math.sqrt(dL * dL + da * da + db * db);
}

// Top-K smallest by `score`. Returns indices sorted by score ascending.
// Uses a bounded max-heap sized K; O(N log K).
export function topKSmallest(scores, k) {
  const n = scores.length;
  if (k >= n) {
    const idx = new Int32Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    return [...idx].sort((a, b) => scores[a] - scores[b]);
  }
  // Max-heap of size K, root = biggest (so we can evict when smaller arrives).
  const heap = new Int32Array(k);
  let size = 0;
  const swap = (i, j) => {
    const t = heap[i]; heap[i] = heap[j]; heap[j] = t;
  };
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
    if (size < k) {
      heap[size++] = i;
      siftUp(size - 1);
    } else if (scores[i] < scores[heap[0]]) {
      heap[0] = i;
      siftDown(0);
    }
  }
  // Sort ascending for deterministic output.
  return Array.from(heap.slice(0, size)).sort((a, b) => scores[a] - scores[b]);
}

// Clamp [0..255], byte-safe.
export function clamp255(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v | 0;
}
