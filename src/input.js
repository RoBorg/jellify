// Input sources that make the jelly wobble: pointer drags (mouse/touch/pen),
// the browser window being dragged around the desktop, and device motion.

/** Grab and drag the jelly with any number of pointers. */
export function attachPointer(canvas, { toImage, grab, move, release }) {
  const local = (e) => {
    const r = canvas.getBoundingClientRect();
    return toImage(e.clientX - r.left, e.clientY - r.top);
  };
  canvas.addEventListener('pointerdown', (e) => {
    const [x, y] = local(e);
    if (grab(e.pointerId, x, y)) {
      canvas.setPointerCapture(e.pointerId);
      canvas.classList.add('grabbing');
      e.preventDefault();
    }
  });
  canvas.addEventListener('pointermove', (e) => {
    if (canvas.hasPointerCapture(e.pointerId)) move(e.pointerId, ...local(e));
  });
  const end = (e) => {
    release(e.pointerId);
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    canvas.classList.remove('grabbing');
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
}

/**
 * Tracks the browser window's position on screen. There's no event for a
 * window being moved, so poll screenX/screenY once per frame.
 */
export class WindowTracker {
  constructor() {
    this.reset();
  }

  reset() {
    this.x = window.screenX;
    this.y = window.screenY;
    this.w = window.outerWidth;
    this.h = window.outerHeight;
  }

  /** CSS px the window moved since the last poll. */
  poll() {
    const { screenX: x, screenY: y, outerWidth: w, outerHeight: h } = window;
    let dx = x - this.x;
    let dy = y - this.y;
    // Resizing from the left/top edge also changes screenX/Y; that isn't a shake.
    if (w !== this.w || h !== this.h) dx = dy = 0;
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    return [dx, dy];
  }
}

/**
 * Device acceleration in screen space (x right, y down, m/s²), excluding
 * gravity.
 */
export class MotionTracker {
  constructor() {
    this.ax = 0;
    this.ay = 0;
    this.gravity = null;
    this.active = false;
    this.onMotion = this.onMotion.bind(this);
  }

  /** iOS needs an explicit permission request, from inside a user gesture. */
  static needsPermission() {
    return typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function';
  }

  static supported() {
    return typeof DeviceMotionEvent !== 'undefined' && window.isSecureContext;
  }

  async start() {
    if (this.active || !MotionTracker.supported()) return this.active;
    if (MotionTracker.needsPermission()) {
      const state = await DeviceMotionEvent.requestPermission();
      if (state !== 'granted') return false;
    }
    window.addEventListener('devicemotion', this.onMotion);
    this.active = true;
    return true;
  }

  onMotion(e) {
    let x;
    let y;
    const a = e.acceleration;
    if (a && a.x != null) {
      x = a.x;
      y = a.y;
    } else {
      // No gravity-free reading: high-pass the raw one instead.
      const g = e.accelerationIncludingGravity;
      if (!g || g.x == null) return;
      if (!this.gravity) this.gravity = { x: g.x, y: g.y };
      this.gravity.x += (g.x - this.gravity.x) * 0.1;
      this.gravity.y += (g.y - this.gravity.y) * 0.1;
      x = g.x - this.gravity.x;
      y = g.y - this.gravity.y;
    }
    // Device axes (x right, y up the screen in its natural orientation) to
    // current screen axes (x right, y down).
    const angle = ((screen.orientation?.angle ?? window.orientation ?? 0) * Math.PI) / 180;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    this.ax = x * c - y * s;
    this.ay = -(x * s + y * c);
  }
}
