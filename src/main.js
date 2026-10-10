import { loadSegmenter, segment, SEG_SIZE } from './segment.js';
import { loadDepth, estimateDepth, depthProfile, depthImages, INFERNO_CSS } from './depth.js';
import { keepMainBlobs, buildMasks } from './mask.js';
import { inpaint } from './inpaint.js';
import { buildMesh, Jelly } from './jelly.js';
import { Renderer } from './renderer.js';
import { attachPointer, WindowTracker, MotionTracker } from './input.js';

const MAX_SIDE = 1024; // working resolution, px on the long side
const MOTION_GAIN = 0.6; // px/s² of jelly push per m/s² of device acceleration, per px of image diagonal
const MOTION_DEADZONE = 0.25; // m/s², ignores sensor noise while the phone is still

const $ = (id) => document.getElementById(id);
const canvas = $('view');
const fileInput = $('file');
const statusEl = $('status');
const settingsForm = $('settings');
const settingsBtn = $('settings-btn');
const hintEl = $('hint');
const legendEl = $('depth-legend');
legendEl.querySelector('.legend-bar').style.background = INFERNO_CSS;

const settings = { wobble: 0.4, damping: 0.35, sensitivity: 1, hideJelly: false, mode: 'height', showDepth: false };
const windowTracker = new WindowTracker();
const motion = new MotionTracker();

let renderer = null;
let jelly = null;
let scene = null; // the current photo's working data, for depth mode
let imageDiag = 1;
let busy = false;
let awake = false;
let needsDraw = false;

class UserError extends Error {}

try {
  renderer = new Renderer(canvas);
} catch (err) {
  setStatus(err.message, { error: true });
}

// Start fetching the model straight away so it's ready by the time a photo is picked.
loadSegmenter().catch(() => {});

function setStatus(text, { error = false } = {}) {
  statusEl.hidden = !text;
  statusEl.classList.toggle('error', error);
  $('status-text').textContent = text || '';
}

const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

async function decode(file) {
  try {
    return await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch {
    // Older Safari: fall back to an <img>.
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      return img;
    } catch {
      throw new UserError("That file doesn't look like an image this browser can open.");
    } finally {
      URL.revokeObjectURL(url);
    }
  }
}

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

async function openImage(file) {
  if (busy || !renderer || !file) return;
  busy = true;
  hintEl.hidden = true;
  try {
    setStatus('Opening photo…');
    const bitmap = await decode(file);
    const srcW = bitmap.width;
    const srcH = bitmap.height;
    const scale = Math.min(1, MAX_SIDE / Math.max(srcW, srcH));
    const w = Math.max(1, Math.round(srcW * scale));
    const h = Math.max(1, Math.round(srcH * scale));

    const photo = makeCanvas(w, h);
    const pctx = photo.getContext('2d', { willReadFrequently: true });
    pctx.imageSmoothingQuality = 'high';
    pctx.drawImage(bitmap, 0, 0, w, h);
    bitmap.close?.();
    const rgba = pctx.getImageData(0, 0, w, h).data;

    // Show the photo while we work on it.
    jelly = null;
    scene = null;
    $('intro').hidden = true;
    canvas.hidden = false;
    renderer.setBackground(photo, w, h);
    renderer.setJelly(null);
    renderer.setDepthView(null);
    updateDepthView();
    imageDiag = Math.hypot(w, h);
    needsDraw = true;

    setStatus('Loading the cut-out model…');
    await loadSegmenter().catch(() => {
      throw new UserError("Couldn't load the cut-out model. Check your connection and try again.");
    });

    setStatus('Finding the subject…');
    await nextFrame();
    const saliency = await segment(photo);
    if (!keepMainBlobs(saliency, SEG_SIZE, SEG_SIZE)) {
      throw new UserError("Couldn't find a clear subject in that photo. Try one with a single, obvious subject.");
    }

    setStatus('Cutting it out…');
    await nextFrame();
    const { matte, hole } = buildMasks(saliency, SEG_SIZE, rgba, w, h);
    const mesh = buildMesh(matte, w, h);
    if (!mesh) throw new UserError("Couldn't find a clear subject in that photo.");

    setStatus('Filling in the background…');
    await nextFrame();
    const filled = inpaint(rgba, hole, w, h);

    const bg = makeCanvas(w, h);
    bg.getContext('2d').putImageData(new ImageData(filled, w, h), 0, 0);
    const cutout = new Uint8ClampedArray(rgba);
    for (let i = 0; i < w * h; i++) cutout[i * 4 + 3] = matte[i] * 255;
    const fg = makeCanvas(w, h);
    fg.getContext('2d').putImageData(new ImageData(cutout, w, h), 0, 0);

    renderer.setBackground(bg, w, h);
    renderer.setJelly(fg, mesh);
    jelly = new Jelly(mesh);
    jelly.setParams(settings);
    scene = { photo, matte, hole, w, h, mesh, depth: null };

    // A little boing so it's obvious what happened.
    const size = mesh.bounds.y1 - mesh.bounds.y0;
    jelly.kick(size * 0.6, size * 1.2);
    wake();

    setStatus('');
    settingsBtn.hidden = false;
    $('new-btn').textContent = 'New photo';
    hintEl.textContent = matchMedia('(pointer: coarse)').matches
      ? 'Drag it, or shake your phone'
      : 'Drag it, or shake the browser window';
    hintEl.hidden = false;
    hintEl.style.animation = 'none';
    void hintEl.offsetWidth; // restart the fade-out
    hintEl.style.animation = '';
    if (settings.mode === 'depth') applyMode();
  } catch (err) {
    console.error(err);
    setStatus(err instanceof UserError ? err.message : `Something went wrong: ${err.message}`, { error: true });
  } finally {
    busy = false;
  }
}

/**
 * Apply the firmness mode: "height" (the top wobbles most) or "depth" (the
 * parts nearest the camera wobble most). Depth is estimated once per photo,
 * the first time it's needed.
 */
async function applyMode() {
  const current = scene;
  if (!jelly || !current) return;
  if (settings.mode !== 'depth') {
    jelly.setProfile(null);
    wake();
    return;
  }
  // Only touch the status pill while this photo is still the one on screen.
  const status = (text, opts) => current === scene && setStatus(text, opts);
  try {
    if (!current.depth) {
      current.depth = (async () => {
        status('Loading the depth model…');
        await loadDepth();
        status('Estimating depth…');
        await nextFrame();
        const depth = await estimateDepth(current.photo);
        const { mesh, matte, hole, w, h } = current;
        const images = depthImages(depth, matte, hole, w, h);
        const toCanvas = (img) => {
          const c = makeCanvas(img.width, img.height);
          c.getContext('2d').putImageData(img, 0, 0);
          return c;
        };
        return {
          profile: depthProfile(depth, matte, w, h, mesh.rest, mesh.spacing * 1.5),
          view: { bg: toCanvas(images.bg), fg: toCanvas(images.fg) },
        };
      })();
      current.depth.catch(() => {
        current.depth = null; // allow a retry
      });
    }
    const { profile, view } = await current.depth;
    status('');
    if (current !== scene || settings.mode !== 'depth') return;
    if (!current.viewShown) {
      renderer.setDepthView(view.bg, view.fg);
      current.viewShown = true;
    }
    updateDepthView();
    jelly.setProfile(profile);
    // A small boing so the change is visible.
    const size = current.mesh.bounds.y1 - current.mesh.bounds.y0;
    jelly.kick(size * 0.4, size * 0.6);
    wake();
  } catch (err) {
    console.error(err);
    status("Couldn't estimate depth. Check your connection and try again.", { error: true });
  }
}

/** Show the depth map when it's asked for, in depth mode, and ready. */
function updateDepthView() {
  if (!renderer) return;
  renderer.showDepth = settings.showDepth && settings.mode === 'depth' && !!renderer.depthView;
  legendEl.hidden = !renderer.showDepth;
  needsDraw = true;
}

function wake() {
  awake = true;
}

let last = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min((now - last) / 1000, 1 / 30);
  last = now;
  if (!renderer) return;

  const [wx, wy] = windowTracker.poll();
  if (!jelly) {
    if (needsDraw) renderer.draw(null, null);
    needsDraw = false;
    return;
  }

  // The photo moves with the window; the jelly's inertia leaves it behind.
  if (wx || wy) {
    const k = settings.sensitivity / renderer.fit.scale;
    const max = imageDiag * 0.2;
    const clamp = (v) => Math.max(-max, Math.min(max, v));
    jelly.jolt(clamp(-wx * k), clamp(-wy * k));
    wake();
  }

  let ax = 0;
  let ay = 0;
  if (motion.active) {
    const mag = Math.hypot(motion.ax, motion.ay);
    if (mag > MOTION_DEADZONE) {
      const g = (-settings.sensitivity * MOTION_GAIN * imageDiag * (mag - MOTION_DEADZONE)) / mag;
      ax = motion.ax * g;
      ay = motion.ay * g;
      wake();
    }
  }

  if (!awake && !needsDraw) return;
  jelly.step(dt, ax, ay);
  renderer.draw(jelly.pos, jelly.strain);
  needsDraw = false;
  if (jelly.isAtRest()) awake = false;
}
requestAnimationFrame(frame);

attachPointer(canvas, {
  toImage: (x, y) => renderer.toImage(x, y),
  grab: (id, x, y) => {
    if (!jelly) return false;
    wake();
    return jelly.grab(id, x, y);
  },
  move: (id, x, y) => {
    jelly?.moveGrab(id, x, y);
    wake();
  },
  release: (id) => jelly?.release(id),
});

new ResizeObserver(() => {
  renderer?.resize();
  needsDraw = true;
}).observe(canvas);

// --- Photo sources: picker, drag and drop, paste ---

const pick = () => fileInput.click();
$('new-btn').addEventListener('click', pick);
$('intro-btn').addEventListener('click', pick);
fileInput.addEventListener('change', () => {
  openImage(fileInput.files[0]);
  fileInput.value = '';
});

let dragDepth = 0;
window.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragDepth++;
  document.body.classList.add('dragging');
});
window.addEventListener('dragleave', () => {
  if (--dragDepth <= 0) {
    dragDepth = 0;
    document.body.classList.remove('dragging');
  }
});
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove('dragging');
  const file = [...(e.dataTransfer?.files || [])].find((f) => f.type.startsWith('image/'));
  if (file) openImage(file);
});
window.addEventListener('paste', (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
  if (item) openImage(item.getAsFile());
});

// --- Device motion ---

if (MotionTracker.supported()) {
  if (MotionTracker.needsPermission()) {
    // iOS only grants motion access from inside a tap, so ask on the first
    // tap on the photo. Asking from the photo-picker tap would clash with the
    // file dialog.
    let asking = false; // one tap fires both touchend and click
    const ask = async () => {
      if (asking) return;
      asking = true;
      try {
        await motion.start();
      } catch (err) {
        console.error(err);
        return; // not counted as a user gesture; try again next tap
      } finally {
        asking = false;
      }
      canvas.removeEventListener('touchend', ask);
      canvas.removeEventListener('click', ask);
    };
    canvas.addEventListener('touchend', ask);
    canvas.addEventListener('click', ask);
  } else {
    motion.start();
  }
}

// --- Settings ---

settingsBtn.addEventListener('click', () => {
  settingsForm.hidden = !settingsForm.hidden;
  settingsBtn.setAttribute('aria-expanded', String(!settingsForm.hidden));
});
settingsForm.addEventListener('input', () => {
  const f = settingsForm.elements;
  settings.wobble = Number(f.wobble.value);
  settings.damping = Number(f.damping.value);
  settings.sensitivity = Number(f.sensitivity.value);
  settings.hideJelly = f.hideJelly.checked;
  settings.showDepth = f.showDepth.checked;
  const mode = f.mode.value;
  f.showDepth.disabled = mode !== 'depth';
  jelly?.setParams(settings);
  renderer.showJelly = !settings.hideJelly;
  needsDraw = true;
  if (mode !== settings.mode) {
    settings.mode = mode;
    applyMode();
  }
  updateDepthView();
});
settingsForm.addEventListener('submit', (e) => e.preventDefault());
