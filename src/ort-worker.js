// Runs the ONNX models off the main thread, so the jelly keeps animating
// during inference. Messages: { id, url, input? } where `input` is
// { data: Float32Array, dims } for the model's first input; with no input
// the model is just loaded. Replies { id, data, dims } or { id, error }.
import * as ort from 'onnxruntime-web/wasm';

// Multithreading needs cross-origin isolation, which a static host usually
// can't provide; the models are small enough not to need it.
ort.env.wasm.numThreads = 1;

const sessions = new Map();

function load(url) {
  if (!sessions.has(url)) {
    const promise = ort.InferenceSession.create(url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    promise.catch(() => sessions.delete(url)); // allow a retry
    sessions.set(url, promise);
  }
  return sessions.get(url);
}

async function handle({ url, input }) {
  const session = await load(url);
  if (!input) return {};
  const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', input.data, input.dims) };
  const results = await session.run(feeds);
  const out = results[session.outputNames[0]];
  return { data: out.data, dims: out.dims };
}

// The WASM runtime is single-threaded: handle one request at a time.
let queue = Promise.resolve();
self.onmessage = ({ data: msg }) => {
  queue = queue.then(async () => {
    try {
      const reply = await handle(msg);
      self.postMessage({ id: msg.id, ...reply }, reply.data ? [reply.data.buffer] : []);
    } catch (err) {
      self.postMessage({ id: msg.id, error: err?.message ?? String(err) });
    }
  });
};
