// Mask post-processing: turn the low-res saliency map into a crisp,
// full-resolution alpha matte, plus a slightly larger hole mask for inpainting.

/** Keep only the significant connected blobs of a saliency map (in place). */
export function keepMainBlobs(map, w, h, { threshold = 0.5, minShare = 0.15 } = {}) {
  const n = w * h;
  const labels = new Int32Array(n).fill(-1);
  const areas = [];
  const stack = new Int32Array(n);
  for (let start = 0; start < n; start++) {
    if (labels[start] !== -1 || map[start] < threshold) continue;
    const id = areas.length;
    let area = 0;
    let top = 0;
    stack[top++] = start;
    labels[start] = id;
    while (top) {
      const i = stack[--top];
      area++;
      const x = i % w;
      const y = (i / w) | 0;
      if (x > 0) visit(i - 1);
      if (x < w - 1) visit(i + 1);
      if (y > 0) visit(i - w);
      if (y < h - 1) visit(i + w);
    }
    areas.push(area);
    function visit(j) {
      if (labels[j] === -1 && map[j] >= threshold) {
        labels[j] = id;
        stack[top++] = j;
      }
    }
  }
  const largest = Math.max(0, ...areas);
  if (largest < n * 0.005) return 0;

  const keep = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const id = labels[i];
    if (id !== -1 && areas[id] >= largest * minShare) keep[i] = 1;
  }
  // Let soft edges around kept blobs survive, but drop distant speckles.
  const near = dilate(keep, w, h, 3);
  for (let i = 0; i < n; i++) map[i] *= near[i];
  return largest / n;
}

export function resizeBilinear(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh);
  const sx = sw / dw;
  const sy = sh / dh;
  for (let y = 0; y < dh; y++) {
    const fy = Math.min(Math.max((y + 0.5) * sy - 0.5, 0), sh - 1);
    const y0 = fy | 0;
    const y1 = Math.min(y0 + 1, sh - 1);
    const ty = fy - y0;
    for (let x = 0; x < dw; x++) {
      const fx = Math.min(Math.max((x + 0.5) * sx - 0.5, 0), sw - 1);
      const x0 = fx | 0;
      const x1 = Math.min(x0 + 1, sw - 1);
      const tx = fx - x0;
      const a = src[y0 * sw + x0] + (src[y0 * sw + x1] - src[y0 * sw + x0]) * tx;
      const b = src[y1 * sw + x0] + (src[y1 * sw + x1] - src[y1 * sw + x0]) * tx;
      out[y * dw + x] = a + (b - a) * ty;
    }
  }
  return out;
}

/** Mean filter over a (2r+1)² window, normalised at the borders. */
function boxBlur(src, w, h, r) {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let sum = 0;
    for (let x = 0; x <= Math.min(r, w - 1); x++) sum += src[row + x];
    for (let x = 0; x < w; x++) {
      const lo = Math.max(x - r, 0);
      const hi = Math.min(x + r, w - 1);
      tmp[row + x] = sum / (hi - lo + 1);
      if (x + r + 1 < w) sum += src[row + x + r + 1];
      if (x - r >= 0) sum -= src[row + x - r];
    }
  }
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = 0; y <= Math.min(r, h - 1); y++) sum += tmp[y * w + x];
    for (let y = 0; y < h; y++) {
      const lo = Math.max(y - r, 0);
      const hi = Math.min(y + r, h - 1);
      out[y * w + x] = sum / (hi - lo + 1);
      if (y + r + 1 < h) sum += tmp[(y + r + 1) * w + x];
      if (y - r >= 0) sum -= tmp[(y - r) * w + x];
    }
  }
  return out;
}

/**
 * Guided filter (He et al.) with a greyscale guide: snaps the blurry,
 * upsampled mask `p` to edges in the photo `guide`.
 */
export function guidedFilter(guide, p, w, h, r, eps) {
  const n = w * h;
  const ip = new Float32Array(n);
  const ii = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    ip[i] = guide[i] * p[i];
    ii[i] = guide[i] * guide[i];
  }
  const meanI = boxBlur(guide, w, h, r);
  const meanP = boxBlur(p, w, h, r);
  const corrIP = boxBlur(ip, w, h, r);
  const corrII = boxBlur(ii, w, h, r);
  const a = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const varI = corrII[i] - meanI[i] * meanI[i];
    const covIP = corrIP[i] - meanI[i] * meanP[i];
    a[i] = covIP / (varI + eps);
    b[i] = meanP[i] - a[i] * meanI[i];
  }
  const meanA = boxBlur(a, w, h, r);
  const meanB = boxBlur(b, w, h, r);
  const q = new Float32Array(n);
  for (let i = 0; i < n; i++) q[i] = meanA[i] * guide[i] + meanB[i];
  return q;
}

/** Binary dilation by a square of radius r (separable running max). */
export function dilate(mask, w, h, r) {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let last = -Infinity;
    for (let x = 0; x < w; x++) {
      if (mask[row + Math.min(x + r, w - 1)] > 0) last = Math.min(x + r, w - 1);
      tmp[row + x] = last >= x - r ? 1 : 0;
    }
  }
  for (let x = 0; x < w; x++) {
    let last = -Infinity;
    for (let y = 0; y < h; y++) {
      const yy = Math.min(y + r, h - 1);
      if (tmp[yy * w + x] > 0) last = yy;
      out[y * w + x] = last >= y - r ? 1 : 0;
    }
  }
  return out;
}

export function luminance(rgba, n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = (0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2]) / 255;
  }
  return out;
}

const smoothstep = (e0, e1, x) => {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
  return t * t * (3 - 2 * t);
};

/**
 * Build the subject's alpha matte and the inpainting hole at full resolution.
 *
 * The hole is the matte grown by a couple of pixels, so the subject's fringe
 * (partly subject-coloured pixels) doesn't get left behind on the background,
 * while staying thin enough not to show as a halo at rest.
 *
 * @returns {{ matte: Float32Array, hole: Float32Array }}
 */
export function buildMasks(saliency, segSize, rgba, w, h) {
  const n = w * h;
  const up = resizeBilinear(saliency, segSize, segSize, w, h);
  const r = Math.max(4, Math.round(Math.max(w, h) / 110));
  const refined = guidedFilter(luminance(rgba, n), up, w, h, r, 1e-3);
  const matte = new Float32Array(n);
  const solid = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    matte[i] = smoothstep(0.3, 0.7, refined[i]);
    solid[i] = matte[i] > 0.02 ? 1 : 0;
  }
  const hole = dilate(solid, w, h, Math.max(2, Math.round(Math.max(w, h) / 400)));
  return { matte, hole };
}
