// Face detection with YuNet (OpenCV Zoo, 230 KB), for placing googly eyes.
// It takes a fixed 640×640 BGR image (0–255, not normalised) and predicts,
// on grids of stride 8, 16 and 32, a face score, a box and five landmarks:
// right eye, left eye, nose tip and the two mouth corners.
import { model, imageTensor } from './ort.js';

const SIZE = 640;
const STRIDES = [8, 16, 32];
const SCORE_MIN = 0.6;
const NMS_IOU = 0.3;

const yunet = model('models/face_detection_yunet_2023mar.onnx');
export const loadFaceDetector = yunet.load;

/**
 * @param {CanvasImageSource & { width: number, height: number }} source
 * @returns {Promise<Array<{ x: number, y: number, w: number, h: number, score: number, eyes: number[][] }>>}
 *   faces in `source` pixels, best first; `eyes` is [[x, y], [x, y]]
 */
export async function detectFaces(source) {
  // Letterbox: scale the long side to 640 and pad the rest with black.
  const scale = SIZE / Math.max(source.width, source.height);
  const canvas = new OffscreenCanvas(SIZE, SIZE);
  const ctx = canvas.getContext('2d');
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, source.width * scale, source.height * scale);
  const input = imageTensor(canvas, SIZE, SIZE, [0, 0, 0], [1 / 255, 1 / 255, 1 / 255], { bgr: true });
  return decodeFaces(await yunet.runAll(input), scale);
}

/** Turn YuNet's raw outputs into faces, in the coordinates of an image scaled by `scale` to fit. */
export function decodeFaces(outputs, scale) {
  const faces = [];
  for (const stride of STRIDES) {
    const cls = outputs[`cls_${stride}`].data;
    const obj = outputs[`obj_${stride}`].data;
    const bbox = outputs[`bbox_${stride}`].data;
    const kps = outputs[`kps_${stride}`].data;
    const cols = SIZE / stride;
    for (let i = 0; i < cls.length; i++) {
      const clamp = (v) => Math.min(1, Math.max(0, v));
      const score = Math.sqrt(clamp(cls[i]) * clamp(obj[i]));
      if (score < SCORE_MIN) continue;
      const c = i % cols;
      const r = (i / cols) | 0;
      const cx = (c + bbox[i * 4]) * stride;
      const cy = (r + bbox[i * 4 + 1]) * stride;
      const w = Math.exp(bbox[i * 4 + 2]) * stride;
      const h = Math.exp(bbox[i * 4 + 3]) * stride;
      const point = (n) => [((kps[i * 10 + n * 2] + c) * stride) / scale, ((kps[i * 10 + n * 2 + 1] + r) * stride) / scale];
      faces.push({
        x: (cx - w / 2) / scale,
        y: (cy - h / 2) / scale,
        w: w / scale,
        h: h / scale,
        score,
        eyes: [point(0), point(1)],
      });
    }
  }

  // Non-maximum suppression: neighbouring cells all fire for the same face.
  faces.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const f of faces) if (kept.every((k) => iou(k, f) < NMS_IOU)) kept.push(f);
  return kept;
}

function iou(a, b) {
  const w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
  const h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  return inter / (a.w * a.h + b.w * b.h - inter);
}
