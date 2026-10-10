// Soft-body "jelly": a grid mesh covering the subject's mask, simulated as a
// mass-spring network.
//
// Every vertex is tied to its rest position by an anchor spring: stiff at the
// bottom of the subject and loose at the top, like a pudding stuck to a
// plate. Edge, diagonal and bend springs between neighbours carry waves
// through the body, and each triangle resists changes to its area, which
// keeps the lattice from folding over itself and makes stretched jelly thin
// out. All forces are expressed as accelerations (unit mass) and integrated
// with semi-implicit Euler in small fixed substeps.

const TAU = Math.PI * 2;
const SUBSTEP = 1 / 1200; // small enough to stay stable at the firmest setting
const MAX_SPEED = 30000; // px/s, guards against blow-ups

/** Build a grid mesh over the pixels of `alpha` that are part of the subject. */
export function buildMesh(alpha, w, h, cellsAcross = 26) {
  let x0 = w;
  let y0 = h;
  let x1 = -1;
  let y1 = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (alpha[y * w + x] > 0.05) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return null;
  x1 += 1;
  y1 += 1;

  const spacing = Math.max(6, Math.max(x1 - x0, y1 - y0) / cellsAcross);
  const nx = Math.ceil((x1 - x0) / spacing);
  const ny = Math.ceil((y1 - y0) / spacing);

  const covered = new Uint8Array(nx * ny);
  for (let y = y0; y < y1; y++) {
    const cy = Math.min(((y - y0) / spacing) | 0, ny - 1);
    for (let x = x0; x < x1; x++) {
      if (alpha[y * w + x] > 0.05) covered[cy * nx + Math.min(((x - x0) / spacing) | 0, nx - 1)] = 1;
    }
  }

  const vid = new Int32Array((nx + 1) * (ny + 1)).fill(-1);
  const rest = [];
  const vertex = (i, j) => {
    const k = j * (nx + 1) + i;
    if (vid[k] === -1) {
      vid[k] = rest.length / 2;
      rest.push(x0 + i * spacing, y0 + j * spacing);
    }
    return vid[k];
  };

  const tris = [];
  const springs = new Map();
  const spring = (a, b, kind) => {
    const key = a < b ? a * 1e6 + b : b * 1e6 + a;
    if (!springs.has(key)) springs.set(key, [a, b, kind]);
  };
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      if (!covered[j * nx + i]) continue;
      const a = vertex(i, j);
      const b = vertex(i + 1, j);
      const c = vertex(i, j + 1);
      const d = vertex(i + 1, j + 1);
      tris.push(a, b, d, a, d, c);
      spring(a, b, 0);
      spring(c, d, 0);
      spring(a, c, 0);
      spring(b, d, 0);
      spring(a, d, 1);
      spring(b, c, 1);
    }
  }
  // Bend springs span two cells and stop the lattice from crumpling.
  for (let j = 0; j <= ny; j++) {
    for (let i = 0; i <= nx; i++) {
      const a = vid[j * (nx + 1) + i];
      if (a === -1) continue;
      if (i + 2 <= nx) {
        const m = vid[j * (nx + 1) + i + 1];
        const b = vid[j * (nx + 1) + i + 2];
        if (m !== -1 && b !== -1) spring(a, b, 2);
      }
      if (j + 2 <= ny) {
        const m = vid[(j + 1) * (nx + 1) + i];
        const b = vid[(j + 2) * (nx + 1) + i];
        if (m !== -1 && b !== -1) spring(a, b, 2);
      }
    }
  }

  return {
    rest: new Float32Array(rest),
    tris: new Uint32Array(tris),
    springs: [...springs.values()],
    spacing,
    bounds: { x0, y0, x1, y1 },
  };
}

function signedArea(p, a, b, c) {
  return 0.5 * ((p[b * 2] - p[a * 2]) * (p[c * 2 + 1] - p[a * 2 + 1]) - (p[c * 2] - p[a * 2]) * (p[b * 2 + 1] - p[a * 2 + 1]));
}

export class Jelly {
  constructor(mesh) {
    this.mesh = mesh;
    const n = mesh.rest.length / 2;
    this.count = n;
    this.rest = mesh.rest;
    this.pos = new Float32Array(mesh.rest);
    this.vel = new Float32Array(n * 2);
    this.force = new Float32Array(n * 2);
    this.strain = new Float32Array(n);

    const m = mesh.springs.length;
    this.springA = new Uint32Array(m);
    this.springB = new Uint32Array(m);
    this.springKind = new Uint8Array(m);
    this.restLen = new Float32Array(m);
    mesh.springs.forEach(([a, b, kind], s) => {
      this.springA[s] = a;
      this.springB[s] = b;
      this.springKind[s] = kind;
      this.restLen[s] = Math.hypot(
        mesh.rest[b * 2] - mesh.rest[a * 2],
        mesh.rest[b * 2 + 1] - mesh.rest[a * 2 + 1],
      );
    });

    this.tris = mesh.tris;
    this.restArea = new Float32Array(mesh.tris.length / 3);
    for (let t = 0; t < this.restArea.length; t++) {
      this.restArea[t] = signedArea(mesh.rest, mesh.tris[t * 3], mesh.tris[t * 3 + 1], mesh.tris[t * 3 + 2]);
    }

    // 0 at the top of the subject, 1 at its base.
    const { y0, y1 } = mesh.bounds;
    this.height = new Float32Array(n);
    for (let i = 0; i < n; i++) this.height[i] = (mesh.rest[i * 2 + 1] - y0) / Math.max(1, y1 - y0);

    this.grabs = new Map();
    this.setParams({ wobble: 0.4, damping: 0.35 });
  }

  /** wobble: 0 = firm, 1 = very floppy. damping: 0 = boings forever, 1 = settles fast. */
  setParams({ wobble = this.wobble, damping = this.damping }) {
    this.wobble = wobble;
    this.damping = damping;
    const firm = 2 ** (2 - 3 * wobble); // frequency multiplier: 4× at wobble 0, ½× at 1
    const omega = (hz) => TAU * hz * firm;

    this.anchorK = new Float32Array(this.count);
    for (let i = 0; i < this.count; i++) {
      const t = this.height[i];
      const hz = 0.9 + 7 * t ** 3;
      this.anchorK[i] = omega(hz) ** 2;
    }
    this.springK = [omega(8) ** 2, omega(6) ** 2, omega(2) ** 2];
    this.springC = this.springK.map((k) => 2 * (0.04 + 0.12 * damping) * Math.sqrt(k));
    this.areaK = omega(7) ** 2;
    // Scale with stiffness so firm jelly settles in a similar number of wobbles.
    this.drag = 2 * (0.05 + 0.6 * damping) * omega(1);
  }

  grab(id, x, y) {
    const { spacing, bounds } = this.mesh;
    let bestD = Infinity;
    for (let i = 0; i < this.count; i++) {
      bestD = Math.min(bestD, Math.hypot(this.pos[i * 2] - x, this.pos[i * 2 + 1] - y));
    }
    if (bestD > spacing * 2) return false;

    // Hold a wide, soft-edged patch so the pull spreads smoothly instead of
    // creasing the mesh around a single point.
    const sigma = Math.max(spacing * 2, Math.max(bounds.x1 - bounds.x0, bounds.y1 - bounds.y0) * 0.12);
    const verts = [];
    for (let i = 0; i < this.count; i++) {
      const dx = this.pos[i * 2] - x;
      const dy = this.pos[i * 2 + 1] - y;
      const w = Math.exp(-(dx * dx + dy * dy) / (sigma * sigma));
      if (w > 0.02) verts.push({ i, dx, dy, w });
    }
    this.grabs.set(id, { x, y, verts });
    return true;
  }

  moveGrab(id, x, y) {
    const g = this.grabs.get(id);
    if (g) {
      g.x = x;
      g.y = y;
    }
  }

  release(id) {
    this.grabs.delete(id);
  }

  /** Teleport the body relative to its anchors, e.g. when the window jumps. */
  jolt(dx, dy) {
    for (let i = 0; i < this.count; i++) {
      this.pos[i * 2] += dx;
      this.pos[i * 2 + 1] += dy;
    }
  }

  /** Give the body a velocity kick, falling off towards its base. */
  kick(vx, vy) {
    for (let i = 0; i < this.count; i++) {
      const f = 1 - 0.8 * this.height[i];
      this.vel[i * 2] += vx * f;
      this.vel[i * 2 + 1] += vy * f;
    }
  }

  /**
   * Advance the simulation.
   * @param {number} dt     seconds since last frame
   * @param {number} ax     uniform external acceleration (px/s²), e.g. from device motion
   * @param {number} ay
   */
  step(dt, ax = 0, ay = 0) {
    const steps = Math.min(80, Math.ceil(dt / SUBSTEP));
    const h = dt / steps;
    for (let s = 0; s < steps; s++) this.substep(h, ax, ay);
    this.updateStrain();
  }

  substep(h, ax, ay) {
    const { pos, vel, force, rest, anchorK, count } = this;
    const drag = this.drag;
    for (let i = 0; i < count; i++) {
      const k = anchorK[i];
      force[i * 2] = ax - k * (pos[i * 2] - rest[i * 2]) - drag * vel[i * 2];
      force[i * 2 + 1] = ay - k * (pos[i * 2 + 1] - rest[i * 2 + 1]) - drag * vel[i * 2 + 1];
    }

    const { springA, springB, springKind, restLen, springK, springC } = this;
    for (let s = 0; s < springA.length; s++) {
      const a = springA[s] * 2;
      const b = springB[s] * 2;
      const dx = pos[b] - pos[a];
      const dy = pos[b + 1] - pos[a + 1];
      const len = Math.hypot(dx, dy) || 1e-6;
      const nx = dx / len;
      const ny = dy / len;
      const relV = (vel[b] - vel[a]) * nx + (vel[b + 1] - vel[a + 1]) * ny;
      const kind = springKind[s];
      const f = springK[kind] * (len - restLen[s]) + springC[kind] * relV;
      force[a] += f * nx;
      force[a + 1] += f * ny;
      force[b] -= f * nx;
      force[b + 1] -= f * ny;
    }

    // Area preservation: push each triangle's vertices along the gradient of
    // its area, scaled so a vertex displaced by d feels roughly areaK·d.
    const { tris, restArea, areaK } = this;
    for (let t = 0; t < restArea.length; t++) {
      const a = tris[t * 3] * 2;
      const b = tris[t * 3 + 1] * 2;
      const c = tris[t * 3 + 2] * 2;
      const ax_ = pos[a];
      const ay_ = pos[a + 1];
      const bx = pos[b];
      const by = pos[b + 1];
      const cx = pos[c];
      const cy = pos[c + 1];
      const area = 0.5 * ((bx - ax_) * (cy - ay_) - (cx - ax_) * (by - ay_));
      const f = (-areaK * (area - restArea[t])) / restArea[t]; // × 2 × ½ from the gradient
      force[a] += f * (by - cy);
      force[a + 1] += f * (cx - bx);
      force[b] += f * (cy - ay_);
      force[b + 1] += f * (ax_ - cx);
      force[c] += f * (ay_ - by);
      force[c + 1] += f * (bx - ax_);
    }

    if (this.grabs.size) {
      const kg = (TAU * 4) ** 2;
      const cg = 2 * 0.8 * Math.sqrt(kg);
      for (const g of this.grabs.values()) {
        for (const { i, dx, dy, w } of g.verts) {
          force[i * 2] += w * (kg * (g.x + dx - pos[i * 2]) - cg * vel[i * 2]);
          force[i * 2 + 1] += w * (kg * (g.y + dy - pos[i * 2 + 1]) - cg * vel[i * 2 + 1]);
        }
      }
    }

    for (let i = 0; i < count * 2; i++) {
      let v = vel[i] + force[i] * h;
      if (v > MAX_SPEED) v = MAX_SPEED;
      else if (v < -MAX_SPEED) v = -MAX_SPEED;
      vel[i] = v;
      pos[i] += v * h;
    }
  }

  /** Per-vertex average stretch (+) or squash (−) of the attached springs, for shading. */
  updateStrain() {
    const { pos, strain, springA, springB, restLen, springKind } = this;
    strain.fill(0);
    const counts = new Uint8Array(this.count);
    for (let s = 0; s < springA.length; s++) {
      if (springKind[s] !== 0) continue;
      const a = springA[s];
      const b = springB[s];
      const len = Math.hypot(pos[b * 2] - pos[a * 2], pos[b * 2 + 1] - pos[a * 2 + 1]);
      const e = len / restLen[s] - 1;
      strain[a] += e;
      strain[b] += e;
      counts[a]++;
      counts[b]++;
    }
    for (let i = 0; i < this.count; i++) if (counts[i]) strain[i] /= counts[i];
  }

  /** True when nothing is visibly moving, so rendering can pause. */
  isAtRest() {
    if (this.grabs.size) return false;
    const { pos, vel, rest } = this;
    for (let i = 0; i < pos.length; i++) {
      if (Math.abs(vel[i]) > 0.5 || Math.abs(pos[i] - rest[i]) > 0.05) return false;
    }
    return true;
  }
}
