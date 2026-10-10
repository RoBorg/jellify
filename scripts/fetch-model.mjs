// Downloads the models into public/models if they aren't there already, and
// verifies their checksums.
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const MODELS = [
  {
    // U²-Netp salient-object segmentation, 4.6 MB.
    file: 'u2netp.onnx',
    url: 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx',
    sha256: '309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8',
  },
  {
    // Depth Anything V2 Small, int8, 27 MB. Only loaded in depth mode.
    file: 'depth-anything-v2-small-int8.onnx',
    url: 'https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/4472b7362082ad9968fee890ca0f1e5aca36b93d/onnx/model_int8.onnx',
    sha256: '01aa7a23de3f4a0ee1a2bb9997e6918104c85a9f95dea46d27b9b3fb0c6b9001',
  },
];

const dir = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'models');
const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

for (const { file, url, sha256: expected } of MODELS) {
  const dest = join(dir, file);
  if (existsSync(dest) && sha256(readFileSync(dest)) === expected) continue;

  console.log(`Downloading ${url}`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Model download failed: HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (sha256(buf) !== expected) throw new Error(`Checksum mismatch for ${file}`);
  mkdirSync(dir, { recursive: true });
  writeFileSync(dest, buf);
  console.log(`Saved ${dest} (${(buf.length / 1e6).toFixed(1)} MB)`);
}
