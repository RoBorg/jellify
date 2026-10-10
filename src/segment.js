// Salient-object segmentation with U²-Netp (4.6 MB) on onnxruntime-web's WASM
// backend. The model takes a 320×320 image and returns a per-pixel
// "this is the main subject" probability map.
import * as ort from 'onnxruntime-web/wasm';

const MODEL_URL = `${import.meta.env.BASE_URL}models/u2netp.onnx`;
export const SEG_SIZE = 320;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let sessionPromise = null;

export function loadSegmenter() {
  if (!sessionPromise) {
    // Multithreading needs cross-origin isolation, which a static host
    // usually can't provide; the model is small enough not to need it.
    ort.env.wasm.numThreads = 1;
    sessionPromise = ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    }).catch((err) => {
      sessionPromise = null;
      throw err;
    });
  }
  return sessionPromise;
}

/**
 * @param {CanvasImageSource} source  image to segment (any size)
 * @returns {Promise<Float32Array>} SEG_SIZE×SEG_SIZE saliency map in [0, 1]
 */
export async function segment(source) {
  const session = await loadSegmenter();

  const canvas = new OffscreenCanvas(SEG_SIZE, SEG_SIZE);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, SEG_SIZE, SEG_SIZE);
  const { data } = ctx.getImageData(0, 0, SEG_SIZE, SEG_SIZE);

  // Same preprocessing as rembg: scale by the image max, then ImageNet-normalise.
  let max = 1;
  for (let i = 0; i < data.length; i++) if ((i & 3) !== 3 && data[i] > max) max = data[i];
  const n = SEG_SIZE * SEG_SIZE;
  const input = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      input[c * n + i] = (data[i * 4 + c] / max - MEAN[c]) / STD[c];
    }
  }

  const feeds = {
    [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, SEG_SIZE, SEG_SIZE]),
  };
  const results = await session.run(feeds, [session.outputNames[0]]);
  const out = results[session.outputNames[0]].data;

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
