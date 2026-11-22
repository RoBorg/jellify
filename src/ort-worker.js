// Runs the ONNX models off the main thread, so the jelly keeps animating
// during inference. Messages: { id, url, input?, all? } where `input` is
// { data: Float32Array, dims } for the model's first input; with no input
// the model is just loaded. Replies { id, data, dims } for the first output,
// or { id, outputs: { name: { data, dims } } } for every output when `all`
// is set, or { id, error }.
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

async function handle({ url, input, all }) {
  const session = await load(url);
  if (!input) return {};
  const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', input.data, input.dims) };
  const results = await session.run(feeds);
  if (all) {
    const outputs = {};
    for (const name of session.outputNames) outputs[name] = { data: results[name].data, dims: results[name].dims };
    return { outputs };
  }
  const out = results[session.outputNames[0]];
  return { data: out.data, dims: out.dims };
}

const buffers = (reply) =>
  reply.outputs ? Object.values(reply.outputs).map((o) => o.data.buffer) : reply.data ? [reply.data.buffer] : [];

// The WASM runtime is single-threaded: handle one request at a time.
let queue = Promise.resolve();
self.onmessage = ({ data: msg }) => {
  queue = queue.then(async () => {
    try {
      const reply = await handle(msg);
      self.postMessage({ id: msg.id, ...reply }, buffers(reply));
    } catch (err) {
      self.postMessage({ id: msg.id, error: err?.message ?? String(err) });
    }
  });
};
