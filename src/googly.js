// Googly eyes: each pupil is a loose disc rattling around inside a clear
// dome. It falls under gravity, lags behind when its eye is moved (by a drag,
// the browser window moving, the phone being shaken, or the jelly it's stuck
// to wobbling), and bounces and rolls along the rim.
//
// Motion is simulated in each eye's own frame, in image pixels. To keep the
// feel the same for big and small eyes, accelerations given in m/s² are
// scaled as if every eye were EYE_METRES across.

const EYE_METRES = 0.05;
const GRAVITY = 9.81; // m/s²
const PUPIL = 0.55; // pupil radius / eye radius
const RIM = 0.05; // rim thickness / eye radius
const SUBSTEP = 1 / 480;
const RESTITUTION = 0.5;
const BOUNCE_MIN = 0.08; // m/s; slower impacts don't bounce, so pupils settle
const AIR_DRAG = 1.2; // 1/s
const ROLL_DRAG = 4; // 1/s, while touching the rim
const HOUSING_SMOOTHING = 0.03; // s, evens out jumpy window/pointer updates
const REST_SPEED = 0.003; // m/s
const LAYOUT_ITERATIONS = 24;
const HIT_MIN = 0.12; // m/s; softer impacts are silent
const HIT_INTERVAL = 0.05; // s between impact reports from one eye

/**
 * A surface the eyes are stuck to, such as the jelly. Eye positions are kept
 * in the surface's rest frame (the original photo's pixels) and mapped to
 * where the surface currently is for drawing.
 * @typedef {{
 *   bind: (x: number, y: number) => unknown,
 *   place: (binding: unknown) => number[],
 *   toRest: (x: number, y: number) => number[],
 * }} Surface
 */

/**
 * @param {Array<{ x: number, y: number, r: number }>} eyes  where each eye
 *   wants to be (photo px) and its radius at size 1. Dragging, adding and
 *   removing eyes changes this array and its objects in place, so changes
 *   outlive this instance.
 * @param {Array<{ x: number, y: number, r: number }>} home  the eyes as first
 *   placed, for reset()
 */
export class Googly {
  constructor(eyes, home, w, h) {
    this.eyes = eyes;
    this.home = home;
    this.w = w;
    this.h = h;
    this.size = 1;
    this.overlap = false;
    this.surface = null;
    this.grabs = new Map();
    this.state = new Map(); // eye → simulation state
    this.time = 0;
    /** Called with (speed in m/s, eye radius in px) when a pupil hits the rim hard. */
    this.onHit = null;
    for (const e of eyes) this.track(e);
    this.layout();
  }

  track(eye) {
    this.state.set(eye, {
      // Where it sits after layout: in the rest frame (rx, ry) and as drawn (x, y).
      rx: eye.x,
      ry: eye.y,
      x: eye.x,
      y: eye.y,
      binding: null,
      // Pupil offset from the eye's centre, and its velocity, px and px/s.
      // Start with a little toss so it's obvious they're loose.
      px: 0,
      py: 0,
      vx: (Math.random() - 0.5) * eye.r * 12,
      vy: -Math.random() * eye.r * 6,
      lastX: null,
      lastY: null,
      hvx: 0, // the eye's own (smoothed) velocity
      hvy: 0,
      lastHit: -Infinity,
    });
  }

  setParams({ eyeSize = this.size, eyeOverlap = this.overlap }) {
    this.size = eyeSize;
    this.overlap = eyeOverlap;
    this.layout();
  }

  /** Stick the eyes to a deforming surface, or unstick them with null. */
  setSurface(surface) {
    this.surface = surface;
    this.layout();
  }

  radius(eye) {
    return eye.r * this.size;
  }

  /** Put back the eyes as they were first placed, undoing drags, additions and removals. */
  reset() {
    this.grabs.clear();
    this.state.clear();
    this.eyes.length = 0;
    for (const e of this.home) {
      const eye = { ...e };
      this.eyes.push(eye);
      this.track(eye);
    }
    this.layout();
  }

  /** Add an eye centred on (x, y) as drawn, the same size as the others. */
  addEye(x, y) {
    const radii = this.eyes.map((e) => e.r).sort((a, b) => a - b);
    const r = radii.length ? radii[radii.length >> 1] : Math.min(this.w, this.h) * 0.06;
    const [rx, ry] = this.surface ? this.surface.toRest(x, y) : [x, y];
    const eye = { x: rx, y: ry, r };
    this.eyes.push(eye);
    this.track(eye);
    this.layout();
    return eye;
  }

  removeEye(eye) {
    const i = this.eyes.indexOf(eye);
    if (i === -1) return;
    this.eyes.splice(i, 1);
    this.state.delete(eye);
    for (const [id, g] of this.grabs) if (g.eye === eye) this.grabs.delete(id);
    this.layout();
  }

  /** The topmost eye drawn under (x, y), or null. */
  eyeAt(x, y) {
    this.updatePlacement();
    for (let i = this.eyes.length - 1; i >= 0; i--) {
      const e = this.eyes[i];
      const s = this.state.get(e);
      if (Math.hypot(s.x - x, s.y - y) <= this.radius(e)) return e;
    }
    return null;
  }

  /**
   * Work out where each eye sits. Unless overlapping is allowed, eyes that
   * would overlap are pushed apart, starting afresh from where they want to
   * be each time, so shrinking them again lets them return. Eyes being
   * dragged stay under the pointer and push the others out of the way.
   */
  layout() {
    const eyes = this.eyes;
    const n = eyes.length;
    const st = eyes.map((e) => this.state.get(e));
    for (let i = 0; i < n; i++) {
      st[i].rx = eyes[i].x;
      st[i].ry = eyes[i].y;
    }
    if (!this.overlap && n > 1) {
      const held = new Set([...this.grabs.values()].map((g) => g.eye));
      for (let it = 0; it < LAYOUT_ITERATIONS; it++) {
        let moved = false;
        for (let i = 0; i < n; i++) {
          for (let j = i + 1; j < n; j++) {
            let dx = st[j].rx - st[i].rx;
            let dy = st[j].ry - st[i].ry;
            const gap = this.radius(eyes[i]) + this.radius(eyes[j]);
            if (Math.abs(dx) >= gap || Math.abs(dy) >= gap) continue;
            let d = Math.hypot(dx, dy);
            if (d >= gap) continue;
            const hi = held.has(eyes[i]);
            const hj = held.has(eyes[j]);
            if (hi && hj) continue;
            if (d < 1e-6) {
              // Exactly on top of each other: split them sideways.
              dx = 1;
              dy = 0;
              d = 1;
            }
            // Share the push, unless one of them is held still.
            const wi = hi ? 0 : hj ? 1 : 0.5;
            const wj = hj ? 0 : hi ? 1 : 0.5;
            const push = (gap - d) / d;
            st[i].rx -= dx * push * wi;
            st[i].ry -= dy * push * wi;
            st[j].rx += dx * push * wj;
            st[j].ry += dy * push * wj;
            moved = true;
          }
        }
        for (const s of st) {
          s.rx = Math.min(this.w, Math.max(0, s.rx));
          s.ry = Math.min(this.h, Math.max(0, s.ry));
        }
        if (!moved) break;
      }
    }
    for (const s of st) s.binding = this.surface ? this.surface.bind(s.rx, s.ry) : null;
    this.updatePlacement();
  }

  /** Move each eye to where its spot on the surface currently is. */
  updatePlacement() {
    for (const e of this.eyes) {
      const s = this.state.get(e);
      if (s.binding) [s.x, s.y] = this.surface.place(s.binding);
      else [s.x, s.y] = [s.rx, s.ry];
    }
  }

  /** Start dragging the eye under (x, y), if there is one. */
  grab(id, x, y) {
    const eye = this.eyeAt(x, y);
    if (!eye) return false;
    // Pick it up from where it is, even if it had been pushed aside.
    const s = this.state.get(eye);
    eye.x = s.rx;
    eye.y = s.ry;
    this.grabs.set(id, { eye, dx: s.x - x, dy: s.y - y });
    this.layout();
    return true;
  }

  moveGrab(id, x, y) {
    const g = this.grabs.get(id);
    if (!g) return;
    const tx = Math.min(this.w, Math.max(0, x + g.dx));
    const ty = Math.min(this.h, Math.max(0, y + g.dy));
    [g.eye.x, g.eye.y] = this.surface ? this.surface.toRest(tx, ty) : [tx, ty];
    this.layout();
  }

  release(id) {
    this.grabs.delete(id);
  }

  /**
   * Advance the simulation.
   * @param {number} dt   seconds since the last frame
   * @param {number} ax   uniform inertial acceleration in m/s² (screen axes), e.g. from device motion
   * @param {number} ay
   * @param {number} mx   how far every eye moved with the window this frame, image px
   * @param {number} my
   */
  step(dt, ax = 0, ay = 0, mx = 0, my = 0) {
    if (dt <= 0) return;
    this.time += dt;
    this.updatePlacement();
    const smooth = 1 - Math.exp(-dt / HOUSING_SMOOTHING);
    const steps = Math.min(40, Math.ceil(dt / SUBSTEP));
    const h = dt / steps;
    for (const e of this.eyes) {
      const s = this.state.get(e);
      const R = this.radius(e);
      const reach = R * (1 - PUPIL - RIM);
      const scale = (2 * R) / EYE_METRES; // px per metre

      // When the eye speeds up or slows down, the pupil's inertia carries it
      // the other way relative to the eye.
      if (s.lastX === null) {
        s.lastX = s.x;
        s.lastY = s.y;
      }
      const hvx = s.hvx + ((s.x - s.lastX + mx) / dt - s.hvx) * smooth;
      const hvy = s.hvy + ((s.y - s.lastY + my) / dt - s.hvy) * smooth;
      s.vx -= hvx - s.hvx;
      s.vy -= hvy - s.hvy;
      s.hvx = hvx;
      s.hvy = hvy;
      s.lastX = s.x;
      s.lastY = s.y;

      const gx = ax * scale;
      const gy = (GRAVITY + ay) * scale;
      const bounceMin = BOUNCE_MIN * scale;
      let { px, py, vx, vy } = s;
      let hit = 0;
      for (let k = 0; k < steps; k++) {
        vx += gx * h;
        vy += gy * h;
        const air = 1 - AIR_DRAG * h;
        vx *= air;
        vy *= air;
        px += vx * h;
        py += vy * h;
        const d = Math.hypot(px, py);
        if (d > reach) {
          const nx = px / d;
          const ny = py / d;
          px = nx * reach;
          py = ny * reach;
          const vn = vx * nx + vy * ny;
          if (vn > 0) {
            const bounce = vn > bounceMin ? RESTITUTION : 0;
            vx -= (1 + bounce) * vn * nx;
            vy -= (1 + bounce) * vn * ny;
            if (vn > hit) hit = vn;
          }
          const roll = 1 - ROLL_DRAG * h;
          vx *= roll;
          vy *= roll;
        }
      }
      Object.assign(s, { px, py, vx, vy });

      const speed = hit / scale;
      if (speed > HIT_MIN && this.time - s.lastHit > HIT_INTERVAL && this.onHit) {
        s.lastHit = this.time;
        this.onHit(speed, R);
      }
    }
  }

  /** True when nothing is visibly moving, so rendering can pause. */
  isAtRest() {
    if (this.grabs.size) return false;
    return this.eyes.every((e) => {
      const s = this.state.get(e);
      const scale = (2 * this.radius(e)) / EYE_METRES;
      return Math.hypot(s.vx, s.vy) < REST_SPEED * scale && Math.hypot(s.hvx, s.hvy) < REST_SPEED * scale;
    });
  }

  /**
   * Draw the eyes on a 2D canvas laid over the photo.
   * @param {CanvasRenderingContext2D} ctx
   * @param {{ scale: number, x: number, y: number }} fit  image px → CSS px
   * @param {number} dpr  device px per CSS px
   */
  draw(ctx, fit, dpr) {
    this.updatePlacement();
    const k = fit.scale * dpr; // device px per image px
    ctx.setTransform(k, 0, 0, k, fit.x * dpr, fit.y * dpr);
    for (const e of this.eyes) {
      const s = this.state.get(e);
      const { x, y } = s;
      const R = this.radius(e);

      // White dome, standing slightly proud of the photo.
      ctx.save();
      ctx.shadowColor = 'rgb(0 0 0 / 0.35)';
      ctx.shadowBlur = R * 0.25 * k;
      ctx.shadowOffsetY = R * 0.08 * k;
      const white = ctx.createRadialGradient(x - R * 0.25, y - R * 0.3, 0, x, y, R);
      white.addColorStop(0, '#ffffff');
      white.addColorStop(0.65, '#f3f2ef');
      white.addColorStop(1, '#d6d4cf');
      ctx.fillStyle = white;
      circle(ctx, x, y, R);
      ctx.fill();
      ctx.restore();
      ctx.lineWidth = R * 0.035;
      ctx.strokeStyle = 'rgb(0 0 0 / 0.3)';
      ctx.stroke();

      // The pupil, with a soft shadow on the white below it.
      const pr = R * PUPIL;
      const cx = x + s.px;
      const cy = y + s.py;
      ctx.save();
      ctx.shadowColor = 'rgb(0 0 0 / 0.3)';
      ctx.shadowBlur = pr * 0.2 * k;
      ctx.shadowOffsetY = pr * 0.06 * k;
      const black = ctx.createRadialGradient(cx - pr * 0.3, cy - pr * 0.35, 0, cx, cy, pr);
      black.addColorStop(0, '#3a3a3a');
      black.addColorStop(1, '#050505');
      ctx.fillStyle = black;
      circle(ctx, cx, cy, pr);
      ctx.fill();
      ctx.restore();

      // Glare on the plastic, which stays put while the pupil moves under it.
      ctx.fillStyle = 'rgb(255 255 255 / 0.8)';
      ctx.beginPath();
      ctx.ellipse(x - R * 0.38, y - R * 0.42, R * 0.26, R * 0.13, -Math.PI / 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'rgb(255 255 255 / 0.45)';
      circle(ctx, x + R * 0.42, y + R * 0.44, R * 0.07);
      ctx.fill();
    }
  }
}

function circle(ctx, x, y, r) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
}

/**
 * Where to put the eyes: two per face, sized to the face. With no faces, a
 * pair goes near the top of the main subject (or the middle of the photo if
 * there's no subject either).
 * @param {Array<{ w: number, eyes: number[][] }>} faces  from detectFaces()
 * @param {{ map: Float32Array, size: number } | null} subject  saliency map over the whole photo
 * @returns {Array<{ x: number, y: number, r: number }>}
 */
export function placeEyes(faces, subject, w, h) {
  if (faces.length) {
    return faces.flatMap(({ w: faceW, eyes: [a, b] }) => {
      // Size from the gap between the eyes, or from the face's width when the
      // eyes look close together because the head is turned.
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const d = Math.hypot(dx, dy);
      const r = Math.max(4, 0.45 * Math.max(d, 0.4 * faceW));
      // Spread them along the line between the eyes, so they follow the
      // head's tilt without overlapping.
      const ux = d > 1e-3 ? dx / d : 1;
      const uy = d > 1e-3 ? dy / d : 0;
      const half = Math.max(d, 2.1 * r) / 2;
      const mx = (a[0] + b[0]) / 2;
      const my = (a[1] + b[1]) / 2;
      return [-1, 1].map((side) => ({ x: mx + ux * half * side, y: my + uy * half * side, r }));
    });
  }

  let cx = w / 2;
  let cy = h / 2;
  let r = Math.min(w, h) * 0.08;
  if (subject) {
    const { map, size } = subject;
    let y0 = size;
    let y1 = -1;
    for (let i = 0; i < map.length; i++) {
      if (map[i] < 0.5) continue;
      const y = (i / size) | 0;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
    if (y1 >= 0) {
      // Find the subject's width a little way down from its top.
      const row = Math.round(y0 + (y1 - y0) * 0.22);
      let x0 = size;
      let x1 = -1;
      for (let x = 0; x < size; x++) {
        if (map[row * size + x] >= 0.5) {
          if (x < x0) x0 = x;
          x1 = x;
        }
      }
      const sx = w / size;
      const sy = h / size;
      cx = ((x0 + x1 + 1) / 2) * sx;
      cy = (row + 0.5) * sy;
      r = Math.max(6, Math.min((x1 - x0 + 1) * sx * 0.17, (y1 - y0 + 1) * sy * 0.12));
    }
  }
  return [cx - r * 1.1, cx + r * 1.1].map((x) => ({ x, y: cy, r }));
}
