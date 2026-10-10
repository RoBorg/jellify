// Monocular depth estimation with Depth Anything V2 Small (int8, 27 MB),
// loaded only when depth mode is used. The model outputs relative inverse
// depth (disparity): larger values are closer to the camera.
import { model, imageTensor } from './ort.js';

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
// Long side of the model input. Must be a multiple of the ViT patch size
// (14); smaller is faster, and the jelly only needs coarse depth.
const LONG_SIDE = 364;
const PATCH = 14;

const depthAnything = model('models/depth-anything-v2-small-int8.onnx');
export const loadDepth = depthAnything.load;

/**
 * @param {CanvasImageSource & { width: number, height: number }} source
 * @returns {Promise<{ map: Float32Array, w: number, h: number }>} disparity map
 */
export async function estimateDepth(source) {
  const scale = LONG_SIDE / Math.max(source.width, source.height);
  const w = Math.max(PATCH, Math.round((source.width * scale) / PATCH) * PATCH);
  const h = Math.max(PATCH, Math.round((source.height * scale) / PATCH) * PATCH);
  const input = imageTensor(source, w, h, MEAN, STD);
  const out = await depthAnything.run(input);
  const [oh, ow] = out.dims.slice(-2);
  return { map: out.data, w: ow, h: oh };
}

/**
 * Per-vertex looseness profile from depth: 0 for the parts of the subject
 * nearest the camera (they wobble most), 1 for the farthest (held firm).
 * Each vertex averages the disparity of subject pixels around it, so edge
 * vertices aren't dragged towards the background's depth.
 */
export function depthProfile(depth, matte, mw, mh, rest, radius) {
  const { map, w, h } = depth;
  const sx = w / mw;
  const sy = h / mh;
  const r = Math.max(1, Math.round(radius * sx));
  const n = rest.length / 2;
  const values = new Float32Array(n);
  const valid = new Uint8Array(n);
  for (let v = 0; v < n; v++) {
    const cx = Math.round(rest[v * 2] * sx);
    const cy = Math.round(rest[v * 2 + 1] * sy);
    let sum = 0;
    let weight = 0;
    for (let y = Math.max(0, cy - r); y <= Math.min(h - 1, cy + r); y++) {
      const my = Math.min(mh - 1, Math.floor((y + 0.5) / sy));
      for (let x = Math.max(0, cx - r); x <= Math.min(w - 1, cx + r); x++) {
        const a = matte[my * mw + Math.min(mw - 1, Math.floor((x + 0.5) / sx))];
        sum += map[y * w + x] * a;
        weight += a;
      }
    }
    if (weight > 0.01) {
      values[v] = sum / weight;
      valid[v] = 1;
    }
  }

  // Normalise with percentiles so a few outliers don't squash the range.
  const sorted = values.filter((_, i) => valid[i]).sort();
  if (!sorted.length) return null;
  const near = sorted[Math.floor((sorted.length - 1) * 0.95)];
  const far = sorted[Math.floor((sorted.length - 1) * 0.05)];
  const range = near - far || 1;
  const profile = new Float32Array(n);
  for (let v = 0; v < n; v++) {
    profile[v] = valid[v] ? Math.min(1, Math.max(0, (near - values[v]) / range)) : 1;
  }
  return profile;
}
