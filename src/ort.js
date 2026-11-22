// Main-thread side of the model worker (see ort-worker.js).

const worker = new Worker(new URL('./ort-worker.js', import.meta.url), { type: 'module' });
const pending = new Map();
let nextId = 0;

worker.onmessage = ({ data }) => {
  const { resolve, reject } = pending.get(data.id);
  pending.delete(data.id);
  if (data.error) reject(new Error(data.error));
  else resolve(data);
};

function call(msg, transfer = []) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, ...msg }, transfer);
  });
}

/**
 * A model at `path` (relative to the site root), loaded on first use.
 * `run` resolves to the first output; `runAll` to every output, by name.
 * @returns {{ load: () => Promise<void>, run: (input) => Promise<{ data: Float32Array, dims: number[] }>, runAll: (input) => Promise<Record<string, { data: Float32Array, dims: number[] }>> }}
 */
export function model(path) {
  const url = new URL(`${import.meta.env.BASE_URL}${path}`, location.href).href;
  let loaded = null;
  const load = () => {
    if (!loaded) {
      loaded = call({ url }).catch((err) => {
        loaded = null; // allow a retry
        throw err;
      });
    }
    return loaded;
  };
  const run = async (input) => {
    await load();
    return call({ url, input }, [input.data.buffer]);
  };
  const runAll = async (input) => {
    await load();
    return (await call({ url, input, all: true }, [input.data.buffer])).outputs;
  };
  return { load, run, runAll };
}

/** Draw `source` at w×h and return it as a normalised NCHW float tensor. */
export function imageTensor(source, w, h, mean, std, { scaleByMax = false, bgr = false } = {}) {
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, w, h);

  let max = 255;
  if (scaleByMax) {
    max = 1;
    for (let i = 0; i < data.length; i++) if ((i & 3) !== 3 && data[i] > max) max = data[i];
  }
  const n = w * h;
  const input = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const src = bgr ? 2 - c : c;
      input[c * n + i] = (data[i * 4 + src] / max - mean[c]) / std[c];
    }
  }
  return { data: input, dims: [1, 3, h, w] };
}
