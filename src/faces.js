// Face detection with YuNet (OpenCV Zoo, 230 KB), for placing googly eyes.
// It takes a fixed 640×640 BGR image (0–255, not normalised) and predicts,
// on grids of stride 8, 16 and 32, a face score, a box and five landmarks:
// right eye, left eye, nose tip and the two mouth corners.
//
// YuNet is trained on mostly upright faces: it still finds faces tilted by
// 45° or more, but its landmarks drift back towards upright, so the eyes
// come out level. So the photo is run through it at several rotations, and
// each face is kept from the rotation that scored it highest, which is the
// one where it was closest to upright and its landmarks are trustworthy.
import { model, imageTensor } from './ort.js';

const SIZE = 640;
const STRIDES = [8, 16, 32];
const ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]; // degrees
const SCORE_MIN = 0.6;
// Rotated copies give more chances for a false positive, so ask for more.
const ROTATED_SCORE_MIN = 0.7;
const NMS_IOU = 0.3;

const yunet = model('models/face_detection_yunet_2023mar.onnx');
export const loadFaceDetector = yunet.load;

/**
 * @param {CanvasImageSource & { width: number, height: number }} source
 * @returns {Promise<Array<{ x: number, y: number, w: number, h: number, score: number, eyes: number[][] }>>}
 *   faces in `source` pixels, best first; `eyes` is [[x, y], [x, y]]
 */
export async function detectFaces(source) {
  const { width: w, height: h } = source;
  const canvas = new OffscreenCanvas(SIZE, SIZE);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  const found = [];
  for (const angle of ANGLES) {
    // Rotate about the photo's centre, scaled so the rotated photo fits,
    // and pad the rest with black.
    const { theta, scale } = frame(angle, w, h);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.translate(SIZE / 2, SIZE / 2);
    ctx.rotate(theta);
    ctx.scale(scale, scale);
    ctx.drawImage(source, -w / 2, -h / 2, w, h);
    const input = imageTensor(canvas, SIZE, SIZE, [0, 0, 0], [1 / 255, 1 / 255, 1 / 255], { bgr: true });
    const faces = decodeFaces(await yunet.runAll(input));
    for (const f of faces) {
      if (angle && f.score < ROTATED_SCORE_MIN) continue;
      found.push(unrotate(f, angle, w, h));
    }
  }
  return mergeFaces(found);
}

/** The rotation (radians, clockwise on screen) and scale for a detection pass. */
export function frame(angle, w, h) {
  const theta = (angle * Math.PI) / 180;
  const c = Math.abs(Math.cos(theta));
  const s = Math.abs(Math.sin(theta));
  const scale = SIZE / Math.max(w * c + h * s, w * s + h * c);
  return { theta, scale };
}

/** Map a face found in a pass rotated by `angle` back to the w×h photo. */
export function unrotate(face, angle, w, h) {
  const { theta, scale } = frame(angle, w, h);
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const back = ([mx, my]) => {
    const u = (mx - SIZE / 2) / scale;
    const v = (my - SIZE / 2) / scale;
    return [u * c + v * s + w / 2, -u * s + v * c + h / 2];
  };
  const [cx, cy] = back([face.x + face.w / 2, face.y + face.h / 2]);
  const size = Math.max(face.w, face.h) / scale;
  return {
    x: cx - size / 2,
    y: cy - size / 2,
    w: size,
    h: size,
    score: face.score,
    eyes: face.eyes.map(back),
  };
}

/** The same face is usually found in several passes: keep its best-scoring one. */
export function mergeFaces(faces) {
  faces.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const f of faces) {
    const cx = f.x + f.w / 2;
    const cy = f.y + f.h / 2;
    const dup = kept.some((k) => Math.hypot(k.x + k.w / 2 - cx, k.y + k.h / 2 - cy) < 0.5 * Math.max(k.w, f.w));
    if (!dup) kept.push(f);
  }
  return kept;
}

/** Turn YuNet's raw outputs into faces, in the 640×640 input's pixels. */
export function decodeFaces(outputs) {
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
      const point = (n) => [(kps[i * 10 + n * 2] + c) * stride, (kps[i * 10 + n * 2 + 1] + r) * stride];
      faces.push({ x: cx - w / 2, y: cy - h / 2, w, h, score, eyes: [point(0), point(1)] });
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
