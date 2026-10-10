// Algorithmic background fill: "push-pull" pyramid interpolation.
//
// Push: build an image pyramid where each coarser pixel is the weighted mean
// of the *known* pixels beneath it, so known colours spread inwards as the
// resolution drops. Pull: walk back up, filling each level's unknown pixels
// from a bilinear sample of the level above. The result is a smooth, seamless
// fill of the hole from its surroundings; it is blurry rather than textured,
// which is fine since the jelly covers most of it most of the time.

const TAP = [1 / 8, 3 / 8, 3 / 8, 1 / 8];

/**
 * @param {Uint8ClampedArray} rgba  w×h image
 * @param {Float32Array} hole       w×h, 1 where pixels must be filled
 * @returns {Uint8ClampedArray}     new RGBA image with the hole filled
 */
export function inpaint(rgba, hole, w, h) {
  const n = w * h;
  const base = { w, h, c: new Float32Array(n * 3), wt: new Float32Array(n) };
  for (let i = 0; i < n; i++) {
    const known = hole[i] > 0 ? 0 : 1;
    base.wt[i] = known;
    for (let k = 0; k < 3; k++) base.c[i * 3 + k] = rgba[i * 4 + k] * known;
  }

  const levels = [base];
  while (levels.at(-1).w > 1 || levels.at(-1).h > 1) {
    levels.push(push(levels.at(-1)));
  }

  const top = levels.at(-1);
  for (let i = 0; i < top.wt.length; i++) {
    for (let k = 0; k < 3; k++) {
      top.c[i * 3 + k] = top.wt[i] > 0 ? top.c[i * 3 + k] / top.wt[i] : 128; // 128: nothing known at all
    }
    top.wt[i] = 1;
  }

  for (let l = levels.length - 2; l >= 0; l--) pull(levels[l], levels[l + 1]);

  const out = new Uint8ClampedArray(rgba);
  for (let i = 0; i < n; i++) {
    if (hole[i] > 0) for (let k = 0; k < 3; k++) out[i * 4 + k] = base.c[i * 3 + k];
  }
  return out;
}

function push(fine) {
  const w = Math.max(1, Math.ceil(fine.w / 2));
  const h = Math.max(1, Math.ceil(fine.h / 2));
  const c = new Float32Array(w * h * 3);
  const wt = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sw = 0;
      let r = 0;
      let g = 0;
      let b = 0;
      for (let j = 0; j < 4; j++) {
        const fy = Math.min(Math.max(2 * y - 1 + j, 0), fine.h - 1);
        for (let i = 0; i < 4; i++) {
          const fx = Math.min(Math.max(2 * x - 1 + i, 0), fine.w - 1);
          const fi = fy * fine.w + fx;
          const k = TAP[i] * TAP[j] * fine.wt[fi];
          // Colours are stored premultiplied by weight.
          sw += k;
          r += TAP[i] * TAP[j] * fine.c[fi * 3];
          g += TAP[i] * TAP[j] * fine.c[fi * 3 + 1];
          b += TAP[i] * TAP[j] * fine.c[fi * 3 + 2];
        }
      }
      const o = y * w + x;
      if (sw > 0) {
        // Re-premultiply by the (boosted, clamped) coarse weight.
        const nw = Math.min(1, sw * 2);
        c[o * 3] = (r / sw) * nw;
        c[o * 3 + 1] = (g / sw) * nw;
        c[o * 3 + 2] = (b / sw) * nw;
        wt[o] = nw;
      }
    }
  }
  return { w, h, c, wt };
}

function pull(fine, coarse) {
  const { w, h, c, wt } = fine;
  for (let y = 0; y < h; y++) {
    const fy = Math.min(Math.max((y + 0.5) / 2 - 0.5, 0), coarse.h - 1);
    const y0 = fy | 0;
    const y1 = Math.min(y0 + 1, coarse.h - 1);
    const ty = fy - y0;
    for (let x = 0; x < w; x++) {
      const o = y * w + x;
      const a = wt[o];
      if (a >= 1) continue;
      const fx = Math.min(Math.max((x + 0.5) / 2 - 0.5, 0), coarse.w - 1);
      const x0 = fx | 0;
      const x1 = Math.min(x0 + 1, coarse.w - 1);
      const tx = fx - x0;
      const i00 = (y0 * coarse.w + x0) * 3;
      const i01 = (y0 * coarse.w + x1) * 3;
      const i10 = (y1 * coarse.w + x0) * 3;
      const i11 = (y1 * coarse.w + x1) * 3;
      for (let k = 0; k < 3; k++) {
        const top = coarse.c[i00 + k] + (coarse.c[i01 + k] - coarse.c[i00 + k]) * tx;
        const bot = coarse.c[i10 + k] + (coarse.c[i11 + k] - coarse.c[i10 + k]) * tx;
        const up = top + (bot - top) * ty;
        // `c` is premultiplied by `a`, `up` is fully opaque.
        c[o * 3 + k] = c[o * 3 + k] + (1 - a) * up;
      }
      wt[o] = 1;
    }
  }
}
