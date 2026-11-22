// Googly eyes: each pupil is a loose disc rattling around inside a clear
// dome. It falls under gravity, lags behind when its eye is moved (by a drag,
// the browser window moving, or the phone being shaken), and bounces and
// rolls along the rim.
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

/**
 * @param {Array<{ x: number, y: number, r: number, homeX: number, homeY: number }>} eyes
 *   from placeEyes(): where each eye wants to be, its radius at size 1, and
 *   where it was first placed, in image px. Dragging moves these objects, so
 *   moves outlive this instance.
 */
export class Googly {
  constructor(eyes, w, h) {
    this.eyes = eyes;
    this.w = w;
    this.h = h;
    this.size = 1;
    this.overlap = false;
    this.grabs = new Map();
    // Where each eye is actually drawn: its own position, unless that would
    // overlap another eye and overlapping isn't allowed.
    this.pos = new Float64Array(eyes.length * 2);
    this.layout();
    this.state = eyes.map((e, i) => ({
      // Pupil offset from the eye's centre, and its velocity, px and px/s.
      px: 0,
      py: 0,
      // Start each one with a little toss so it's obvious they're loose.
      vx: (Math.random() - 0.5) * e.r * 12,
      vy: -Math.random() * e.r * 6,
      lastX: this.pos[i * 2],
      lastY: this.pos[i * 2 + 1],
      hvx: 0, // the eye's own (smoothed) velocity
      hvy: 0,
    }));
  }

  setParams({ eyeSize = this.size, eyeOverlap = this.overlap }) {
    this.size = eyeSize;
    this.overlap = eyeOverlap;
    this.layout();
  }

  radius(i) {
    return this.eyes[i].r * this.size;
  }

  /** Put every eye back where it was first placed. */
  reset() {
    for (const e of this.eyes) {
      e.x = e.homeX;
      e.y = e.homeY;
    }
    this.layout();
  }

  /**
   * Work out where to draw the eyes. Unless overlapping is allowed, eyes that
   * would overlap are pushed apart, starting afresh from where they want to
   * be each time, so shrinking them again lets them return. Eyes being
   * dragged stay under the pointer and push the others out of the way.
   */
  layout() {
    const { eyes, pos } = this;
    const n = eyes.length;
    for (let i = 0; i < n; i++) {
      pos[i * 2] = eyes[i].x;
      pos[i * 2 + 1] = eyes[i].y;
    }
    if (this.overlap || n < 2) return;
    const held = new Uint8Array(n);
    for (const g of this.grabs.values()) held[g.i] = 1;
    for (let it = 0; it < LAYOUT_ITERATIONS; it++) {
      let moved = false;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          let dx = pos[j * 2] - pos[i * 2];
          let dy = pos[j * 2 + 1] - pos[i * 2 + 1];
          const gap = this.radius(i) + this.radius(j);
          if (Math.abs(dx) >= gap || Math.abs(dy) >= gap) continue;
          let d = Math.hypot(dx, dy);
          if (d >= gap) continue;
          if (d < 1e-6) {
            // Exactly on top of each other: split them sideways.
            dx = j > i ? 1 : -1;
            dy = 0;
            d = 1;
          }
          // Share the push, unless one of them is held still.
          const wi = held[i] ? 0 : held[j] ? 1 : 0.5;
          const wj = held[j] ? 0 : held[i] ? 1 : 0.5;
          const push = (gap - d) / d;
          pos[i * 2] -= dx * push * wi;
          pos[i * 2 + 1] -= dy * push * wi;
          pos[j * 2] += dx * push * wj;
          pos[j * 2 + 1] += dy * push * wj;
          moved = true;
        }
      }
      for (let i = 0; i < n; i++) {
        pos[i * 2] = Math.min(this.w, Math.max(0, pos[i * 2]));
        pos[i * 2 + 1] = Math.min(this.h, Math.max(0, pos[i * 2 + 1]));
      }
      if (!moved) break;
    }
  }

  /** Start dragging the eye under (x, y), if there is one. */
  grab(id, x, y) {
    const { pos } = this;
    for (let i = this.eyes.length - 1; i >= 0; i--) {
      const ex = pos[i * 2];
      const ey = pos[i * 2 + 1];
      if (Math.hypot(ex - x, ey - y) <= this.radius(i)) {
        // Pick it up from where it's drawn, even if it had been pushed aside.
        const e = this.eyes[i];
        e.x = ex;
        e.y = ey;
        this.grabs.set(id, { i, dx: ex - x, dy: ey - y });
        this.layout();
        return true;
      }
    }
    return false;
  }

  moveGrab(id, x, y) {
    const g = this.grabs.get(id);
    if (!g) return;
    const e = this.eyes[g.i];
    e.x = Math.min(this.w, Math.max(0, x + g.dx));
    e.y = Math.min(this.h, Math.max(0, y + g.dy));
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
    const smooth = 1 - Math.exp(-dt / HOUSING_SMOOTHING);
    const steps = Math.min(40, Math.ceil(dt / SUBSTEP));
    const h = dt / steps;
    for (let i = 0; i < this.eyes.length; i++) {
      const ex = this.pos[i * 2];
      const ey = this.pos[i * 2 + 1];
      const s = this.state[i];
      const R = this.radius(i);
      const reach = R * (1 - PUPIL - RIM);
      const scale = (2 * R) / EYE_METRES; // px per metre

      // When the eye speeds up or slows down, the pupil's inertia carries it
      // the other way relative to the eye.
      const hvx = s.hvx + ((ex - s.lastX + mx) / dt - s.hvx) * smooth;
      const hvy = s.hvy + ((ey - s.lastY + my) / dt - s.hvy) * smooth;
      s.vx -= hvx - s.hvx;
      s.vy -= hvy - s.hvy;
      s.hvx = hvx;
      s.hvy = hvy;
      s.lastX = ex;
      s.lastY = ey;

      const gx = ax * scale;
      const gy = (GRAVITY + ay) * scale;
      const bounceMin = BOUNCE_MIN * scale;
      let { px, py, vx, vy } = s;
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
            const e_ = vn > bounceMin ? RESTITUTION : 0;
            vx -= (1 + e_) * vn * nx;
            vy -= (1 + e_) * vn * ny;
          }
          const roll = 1 - ROLL_DRAG * h;
          vx *= roll;
          vy *= roll;
        }
      }
      Object.assign(s, { px, py, vx, vy });
    }
  }

  /** True when nothing is visibly moving, so rendering can pause. */
  isAtRest() {
    if (this.grabs.size) return false;
    return this.state.every((s, i) => {
      const scale = (2 * this.radius(i)) / EYE_METRES;
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
    const k = fit.scale * dpr; // device px per image px
    ctx.setTransform(k, 0, 0, k, fit.x * dpr, fit.y * dpr);
    for (let i = 0; i < this.eyes.length; i++) {
      const x = this.pos[i * 2];
      const y = this.pos[i * 2 + 1];
      const R = this.radius(i);
      const s = this.state[i];

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
 * Where to put the eyes: two per face, sized to the gap between them. With
 * no faces, a pair goes near the top of the main subject (or the middle of
 * the photo if there's no subject either).
 * @param {Array<{ eyes: number[][] }>} faces  from detectFaces()
 * @param {{ map: Float32Array, size: number } | null} subject  saliency map over the whole photo
 */
export function placeEyes(faces, subject, w, h) {
  if (faces.length) {
    return faces.flatMap(({ eyes: [a, b] }) => {
      const r = Math.max(4, Math.hypot(b[0] - a[0], b[1] - a[1]) * 0.45);
      return [a, b].map(([x, y]) => ({ x, y, r, homeX: x, homeY: y }));
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
  return [cx - r * 1.1, cx + r * 1.1].map((x) => ({ x, y: cy, r, homeX: x, homeY: cy }));
}
