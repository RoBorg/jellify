// Salient-object segmentation with U²-Netp (4.6 MB). The model takes a
// 320×320 image and returns a per-pixel "this is the main subject"
// probability map.
import { model, imageTensor } from './ort.js';

export const SEG_SIZE = 320;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const u2netp = model('models/u2netp.onnx');
export const loadSegmenter = u2netp.load;

/**
 * @param {CanvasImageSource} source  image to segment (any size)
 * @returns {Promise<Float32Array>} SEG_SIZE×SEG_SIZE saliency map in [0, 1]
 */
export async function segment(source) {
  // Same preprocessing as rembg: scale by the image max, then ImageNet-normalise.
  const input = imageTensor(source, SEG_SIZE, SEG_SIZE, MEAN, STD, { scaleByMax: true });
  const { data: out } = await u2netp.run(input);

  const n = SEG_SIZE * SEG_SIZE;
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < n; i++) {
    if (out[i] < lo) lo = out[i];
    if (out[i] > hi) hi = out[i];
  }
  const range = hi - lo || 1;
  const map = new Float32Array(n);
  for (let i = 0; i < n; i++) map[i] = (out[i] - lo) / range;
  return map;
}
