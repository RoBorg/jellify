// Downloads the U²-Netp salient-object model (4.6 MB) into public/models if it
// isn't there already, and verifies its checksum.
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const MODEL_URL = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx';
const MODEL_SHA256 = '309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8';
const dest = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'models', 'u2netp.onnx');

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

if (existsSync(dest) && sha256(readFileSync(dest)) === MODEL_SHA256) process.exit(0);

console.log(`Downloading ${MODEL_URL}`);
const res = await fetch(MODEL_URL);
if (!res.ok) throw new Error(`Model download failed: HTTP ${res.status}`);
const buf = Buffer.from(await res.arrayBuffer());
if (sha256(buf) !== MODEL_SHA256) throw new Error('Model checksum mismatch');
mkdirSync(dirname(dest), { recursive: true });
writeFileSync(dest, buf);
console.log(`Saved ${dest} (${(buf.length / 1e6).toFixed(1)} MB)`);
