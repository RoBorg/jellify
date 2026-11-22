import { loadSegmenter, segment, SEG_SIZE } from './segment.js';
import { loadDepth, estimateDepth, depthProfile, depthImages, INFERNO_CSS } from './depth.js';
import { keepMainBlobs, buildMasks } from './mask.js';
import { inpaint } from './inpaint.js';
import { buildMesh, Jelly } from './jelly.js';
import { Renderer } from './renderer.js';
import { attachPointer, WindowTracker, MotionTracker } from './input.js';
import { loadFaceDetector, detectFaces } from './faces.js';
import { Googly, placeEyes } from './googly.js';
import { Sound } from './sound.js';

const MAX_SIDE = 1024; // working resolution, px on the long side
const MAX_FACES = 40;
const MOTION_GAIN = 0.6; // px/s² of jelly push per m/s² of device acceleration, per px of image diagonal
const MOTION_DEADZONE = 0.25; // m/s², ignores sensor noise while the phone is still
const DOUBLE_TAP = 400; // ms
const EFFECTS = ['jelly', 'googly', 'both'];

const $ = (id) => document.getElementById(id);
const canvas = $('view');
const eyesCanvas = $('eyes');
const eyesCtx = eyesCanvas.getContext('2d');
const fileInput = $('file');
const statusEl = $('status');
const settingsForm = $('settings');
const settingsBtn = $('settings-btn');
const hintEl = $('hint');
const legendEl = $('depth-legend');
legendEl.querySelector('.legend-bar').style.background = INFERNO_CSS;

const settings = {
  effect: 'jelly', // 'jelly', 'googly' (eyes only) or 'both' (eyes stuck to the jelly)
  wobble: 0.4,
  damping: 0.35,
  sensitivity: 1,
  hideJelly: false,
  mode: 'height',
  showDepth: false,
  eyeSize: 1,
  eyeOverlap: false,
};
const windowTracker = new WindowTracker();
const motion = new MotionTracker();
const sound = new Sound();
const hasJelly = () => settings.effect !== 'googly';
const hasEyes = () => settings.effect !== 'jelly';

let renderer = null;
let photo = null; // the current photo, and what's been worked out from it so far
let processing = 0; // bumped for each new photo or effect, so stale work can bail out
let jelly = null;
let scene = null; // the current photo's jelly data, for depth mode
let googly = null;
let imageDiag = 1;
let awake = false;
let needsDraw = false;

class UserError extends Error {}

try {
  renderer = new Renderer(canvas);
} catch (err) {
  setStatus(err.message, { error: true });
}

// Start fetching the models straight away so they're ready by the time a
// photo is picked. Googly eyes fall back to the subject when there's no face.
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
  if (!renderer || !file) return;
  const run = ++processing;
  hintEl.hidden = true;
  try {
    setStatus('Opening photo…');
    const bitmap = await decode(file);
    if (run !== processing) return;
    const srcW = bitmap.width;
    const srcH = bitmap.height;
    const scale = Math.min(1, MAX_SIDE / Math.max(srcW, srcH));
    const w = Math.max(1, Math.round(srcW * scale));
    const h = Math.max(1, Math.round(srcH * scale));

    const image = makeCanvas(w, h);
    const pctx = image.getContext('2d', { willReadFrequently: true });
    pctx.imageSmoothingQuality = 'high';
    pctx.drawImage(bitmap, 0, 0, w, h);
    bitmap.close?.();
    const rgba = pctx.getImageData(0, 0, w, h).data;

    photo = { canvas: image, rgba, w, h, subject: null, jelly: null, eyes: null };
    $('intro').hidden = true;
    imageDiag = Math.hypot(w, h);
    await applyEffect();
  } catch (err) {
    if (run !== processing) return;
    console.error(err);
    setStatus(err instanceof UserError ? err.message : `Something went wrong: ${err.message}`, { error: true });
  }
}

/** Show the current photo with the current effect, working it out if needed. */
async function applyEffect() {
  const current = photo;
  if (!current) return;
  const run = ++processing;
  const stale = () => run !== processing;

  // Show the plain photo while we work on it.
  jelly = null;
  scene = null;
  googly = null;
  hintEl.hidden = true;
  canvas.hidden = false;
  canvas.style.cursor = '';
  eyesCanvas.hidden = !hasEyes();
  eyesCtx.clearRect(0, 0, eyesCanvas.width, eyesCanvas.height);
  renderer.setBackground(current.canvas, current.w, current.h);
  renderer.setJelly(null);
  renderer.setDepthView(null);
  updateDepthView();
  needsDraw = true;

  try {
    if (hasJelly()) await showJelly(current, stale);
    if (stale()) return;
    const note = hasEyes() ? await showEyes(current, stale) : '';
    if (stale()) return;
    const hint = {
      jelly: `Drag it, or ${shakeHint()}`,
      googly: `${note}Drag the eyes, tap to add one, double-tap to remove`,
      both: `${note}Drag the jelly or the eyes, tap to add eyes, or ${shakeHint()}`,
    }[settings.effect];
    setStatus('');
    settingsBtn.hidden = false;
    $('new-btn').textContent = 'New photo';
    hintEl.textContent = hint;
    hintEl.hidden = false;
    hintEl.style.animation = 'none';
    void hintEl.offsetWidth; // restart the fade-out
    hintEl.style.animation = '';
    wake();
  } catch (err) {
    if (stale()) return;
    console.error(err);
    setStatus(err instanceof UserError ? err.message : `Something went wrong: ${err.message}`, { error: true });
  }
}

const shakeHint = () => (matchMedia('(pointer: coarse)').matches ? 'shake your phone' : 'shake the browser window');

/** The photo's main subject: a SEG_SIZE² saliency map with only its main blobs kept, or null. */
function findSubject(current) {
  if (!current.subject) {
    current.subject = (async () => {
      setStatus('Loading the cut-out model…');
      await loadSegmenter().catch(() => {
        throw new UserError("Couldn't load the cut-out model. Check your connection and try again.");
      });
      setStatus('Finding the subject…');
      await nextFrame();
      const map = await segment(current.canvas);
      return keepMainBlobs(map, SEG_SIZE, SEG_SIZE) ? map : null;
    })();
    current.subject.catch(() => {
      current.subject = null; // allow a retry
    });
  }
  return current.subject;
}

async function showJelly(current, stale) {
  if (!current.jelly) {
    const saliency = await findSubject(current);
    if (stale()) return;
    if (!saliency) {
      throw new UserError("Couldn't find a clear subject in that photo. Try one with a single, obvious subject.");
    }

    const { rgba, w, h } = current;
    setStatus('Cutting it out…');
    await nextFrame();
    if (stale()) return;
    const { matte, hole } = buildMasks(saliency, SEG_SIZE, rgba, w, h);
    const mesh = buildMesh(matte, w, h);
    if (!mesh) throw new UserError("Couldn't find a clear subject in that photo.");

    setStatus('Filling in the background…');
    await nextFrame();
    if (stale()) return;
    const filled = inpaint(rgba, hole, w, h);

    const bg = makeCanvas(w, h);
    bg.getContext('2d').putImageData(new ImageData(filled, w, h), 0, 0);
    const cutout = new Uint8ClampedArray(rgba);
    for (let i = 0; i < w * h; i++) cutout[i * 4 + 3] = matte[i] * 255;
    const fg = makeCanvas(w, h);
    fg.getContext('2d').putImageData(new ImageData(cutout, w, h), 0, 0);
    current.jelly = { photo: current.canvas, matte, hole, w, h, mesh, bg, fg, depth: null };
  }
  if (stale()) return;

  const js = current.jelly;
  renderer.setBackground(js.bg, js.w, js.h);
  renderer.setJelly(js.fg, js.mesh);
  js.viewShown = false;
  jelly = new Jelly(js.mesh);
  jelly.setParams(settings);
  scene = js;

  // A little boing so it's obvious what happened.
  const size = js.mesh.bounds.y1 - js.mesh.bounds.y0;
  jelly.kick(size * 0.6, size * 1.2);
  sound.boing(0.5);
  if (settings.mode === 'depth') applyMode();
}

/** Sets up the googly eyes; returns a note for the hint, if any. */
async function showEyes(current, stale) {
  if (!current.eyes) {
    setStatus('Loading the face finder…');
    await loadFaceDetector().catch(() => {
      throw new UserError("Couldn't load the face finder. Check your connection and try again.");
    });
    setStatus('Looking for faces…');
    await nextFrame();
    const faces = (await detectFaces(current.canvas)).slice(0, MAX_FACES);
    if (stale()) return;
    let subject = null;
    if (!faces.length) {
      const map = await findSubject(current);
      if (stale()) return;
      if (map) subject = { map, size: SEG_SIZE };
    }
    current.eyesHome = placeEyes(faces, subject, current.w, current.h);
    current.eyes = current.eyesHome.map((e) => ({ ...e }));
    current.eyesNote = faces.length ? '' : 'No faces found, so they went on top. ';
  }
  googly = new Googly(current.eyes, current.eyesHome, current.w, current.h);
  googly.setParams(settings);
  if (jelly) googly.setSurface(jelly.surface());
  googly.onHit = (speed, radius) => {
    // Small eyes tick higher.
    const pitch = Math.min(2, Math.max(0.6, Math.sqrt(40 / Math.max(1, radius * renderer.fit.scale))));
    sound.tick(speed / 0.8, pitch);
  };
  return current.eyesNote;
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

/** Device acceleration felt by the scene (m/s², screen axes), past the deadzone and scaled by sensitivity. */
function motionForce() {
  if (!motion.active) return [0, 0];
  const mag = Math.hypot(motion.ax, motion.ay);
  if (mag <= MOTION_DEADZONE) return [0, 0];
  // The photo moves with the device; what's loose on it is left behind.
  const f = (-settings.sensitivity * (mag - MOTION_DEADZONE)) / mag;
  return [motion.ax * f, motion.ay * f];
}

function drawEyes() {
  // Match the drawing buffer to the overlay's CSS size, as the renderer does.
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(eyesCanvas.clientWidth * dpr));
  const h = Math.max(1, Math.round(eyesCanvas.clientHeight * dpr));
  if (eyesCanvas.width !== w || eyesCanvas.height !== h) {
    eyesCanvas.width = w;
    eyesCanvas.height = h;
  }
  eyesCtx.setTransform(1, 0, 0, 1, 0, 0);
  eyesCtx.clearRect(0, 0, eyesCanvas.width, eyesCanvas.height);
  googly.draw(eyesCtx, renderer.fit, dpr);
}

let last = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min((now - last) / 1000, 1 / 30);
  last = now;
  if (!renderer) return;

  const [wx, wy] = windowTracker.poll();
  const [fx, fy] = motionForce();
  if (!jelly && !googly) {
    if (needsDraw) renderer.draw(null, null);
    needsDraw = false;
    return;
  }

  // The photo moves with the window, and whatever's loose on it (the jelly,
  // the pupils) is left behind by its inertia.
  const k = settings.sensitivity / renderer.fit.scale;
  if (wx || wy || fx || fy) wake();
  let ax = 0;
  let ay = 0;
  if (jelly) {
    if (wx || wy) {
      const max = imageDiag * 0.2;
      const clamp = (v) => Math.max(-max, Math.min(max, v));
      const jx = clamp(-wx * k);
      const jy = clamp(-wy * k);
      jelly.jolt(jx, jy);
      const jolt = Math.hypot(jx, jy) / imageDiag;
      if (jolt > 0.015) sound.boing(jolt * 8);
    }
    ax = fx * MOTION_GAIN * imageDiag;
    ay = fy * MOTION_GAIN * imageDiag;
    const shake = Math.hypot(fx, fy);
    if (shake > 3) sound.boing(shake / 15);
  }

  if (!awake && !needsDraw) return;
  if (jelly) {
    jelly.step(dt, ax, ay);
    renderer.draw(jelly.pos, jelly.strain);
  } else if (needsDraw) {
    renderer.draw(null, null);
  }
  if (googly) {
    googly.step(dt, fx, fy, wx * k, wy * k);
    drawEyes();
  }
  needsDraw = false;
  if ((!jelly || jelly.isAtRest()) && (!googly || googly.isAtRest())) awake = false;
}
requestAnimationFrame(frame);

const grabbed = new Map(); // pointer id → what it's holding
let lastTap = null; // { eye, time }, for double-tap to remove

attachPointer(canvas, {
  toImage: (x, y) => renderer.toImage(x, y),
  // Eyes sit on top of the jelly, so they get first pick.
  grab: (id, x, y) => {
    const target = [googly, jelly].find((t) => t?.grab(id, x, y));
    if (!target) return false;
    grabbed.set(id, target);
    wake();
    return true;
  },
  move: (id, x, y) => {
    grabbed.get(id)?.moveGrab(id, x, y);
    wake();
  },
  release: (id) => {
    const target = grabbed.get(id);
    grabbed.delete(id);
    if (!target) return;
    const dragged = target.release(id);
    if (target === jelly && dragged) {
      const size = jelly.mesh.bounds.y1 - jelly.mesh.bounds.y0;
      sound.boing(dragged / size);
    }
  },
  tap: (x, y) => {
    if (!googly) return;
    const eye = googly.eyeAt(x, y);
    const now = performance.now();
    if (eye) {
      if (lastTap && lastTap.eye === eye && now - lastTap.time < DOUBLE_TAP) {
        googly.removeEye(eye);
        lastTap = null;
      } else {
        lastTap = { eye, time: now };
      }
    } else {
      googly.addEye(x, y);
      lastTap = null;
    }
    wake();
    updateCursor(x, y);
  },
  hover: (x, y) => updateCursor(x, y),
});

/** Show what a click would do: grab an eye or the jelly, or add an eye. */
function updateCursor(x, y) {
  if (googly?.eyeAt(x, y) || jelly?.hits(x, y)) canvas.style.cursor = 'grab';
  else canvas.style.cursor = googly ? 'copy' : 'default';
}

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
  settings.eyeSize = Number(f.eyeSize.value);
  settings.eyeOverlap = f.eyeOverlap.checked;
  if (f.sound.checked !== sound.enabled) sound.setEnabled(f.sound.checked);
  const mode = f.mode.value;
  f.showDepth.disabled = mode !== 'depth';
  jelly?.setParams(settings);
  googly?.setParams(settings);
  wake();
  renderer.showJelly = !settings.hideJelly;
  needsDraw = true;
  if (mode !== settings.mode) {
    settings.mode = mode;
    applyMode();
  }
  updateDepthView();
});
settingsForm.addEventListener('submit', (e) => e.preventDefault());
$('reset-eyes').addEventListener('click', () => {
  googly?.reset();
  wake();
});

// --- Effect: jelly or googly eyes ---

const EFFECT_KEY = 'jellify:effect';
const effectInputs = document.querySelectorAll('input[name="effect"]');

function setEffect(effect) {
  settings.effect = effect;
  for (const input of effectInputs) input.checked = input.value === effect;
  for (const el of document.querySelectorAll('[data-effect]')) {
    el.hidden = !el.dataset.effect.split(' ').includes(effect);
  }
  $('brand-mark').textContent = effect === 'jelly' ? '🍮' : '👀';
  if (effect !== 'jelly') loadFaceDetector().catch(() => {});
  try {
    localStorage.setItem(EFFECT_KEY, effect);
  } catch {
    // Storage blocked; the choice just won't be remembered.
  }
}

for (const input of effectInputs) {
  input.addEventListener('change', () => {
    if (!input.checked) return;
    setEffect(input.value);
    applyEffect();
  });
}

try {
  const saved = localStorage.getItem(EFFECT_KEY);
  if (EFFECTS.includes(saved)) setEffect(saved);
} catch {
  // Storage blocked; start with jelly.
}
