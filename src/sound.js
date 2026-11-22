// Synthesised sound effects with Web Audio: a wobbly "boing" for the jelly
// and a plastic tick when a googly pupil hits its rim. No audio files.
// Browsers only let audio start from a user gesture, so the context is
// created when sound is first switched on.

export class Sound {
  constructor() {
    this.ctx = null;
    this.enabled = false;
    this.lastTick = 0;
    this.lastBoing = 0;
  }

  /** Call from inside a user gesture (e.g. a checkbox change). */
  setEnabled(on) {
    this.enabled = on;
    if (!on) return;
    if (!this.ctx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return;
      this.ctx = new Ctx();
      // A compressor keeps a crowd of rattling eyes from clipping.
      this.out = this.ctx.createDynamicsCompressor();
      this.out.connect(this.ctx.destination);
      const n = Math.round(this.ctx.sampleRate * 0.05);
      this.noise = this.ctx.createBuffer(1, n, this.ctx.sampleRate);
      const data = this.noise.getChannelData(0);
      for (let i = 0; i < n; i++) data[i] = Math.random() * 2 - 1;
    }
    this.ctx.resume();
  }

  ready() {
    return this.enabled && this.ctx && this.ctx.state === 'running';
  }

  /**
   * A googly pupil hitting the rim.
   * @param {number} strength  0–1
   * @param {number} pitch     1 for a typical eye; higher for smaller eyes
   */
  tick(strength, pitch = 1) {
    if (!this.ready()) return;
    const ctx = this.ctx;
    const t = ctx.currentTime;
    if (t - this.lastTick < 0.018) return; // a shaken crowd would be a roar
    this.lastTick = t;
    const level = 0.05 + 0.35 * Math.min(1, strength);

    const src = ctx.createBufferSource();
    src.buffer = this.noise;
    const band = ctx.createBiquadFilter();
    band.type = 'bandpass';
    band.frequency.value = (2200 + Math.random() * 1200) * pitch;
    band.Q.value = 5;
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(level, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.035);
    src.connect(band).connect(gain).connect(this.out);
    src.start(t);
    src.stop(t + 0.05);

    // A short tone gives it the hollow, plastic "tock".
    const osc = ctx.createOscillator();
    osc.frequency.value = (900 + Math.random() * 300) * pitch;
    const tone = ctx.createGain();
    tone.gain.setValueAtTime(level * 0.5, t);
    tone.gain.exponentialRampToValueAtTime(0.001, t + 0.025);
    osc.connect(tone).connect(this.out);
    osc.start(t);
    osc.stop(t + 0.03);
  }

  /**
   * A jelly boing: a falling, warbling tone.
   * @param {number} strength  0–1
   */
  boing(strength) {
    if (!this.ready()) return;
    const ctx = this.ctx;
    const t = ctx.currentTime;
    if (t - this.lastBoing < 0.25) return;
    this.lastBoing = t;
    const s = Math.min(1, strength);
    const level = 0.08 + 0.3 * s;
    const length = 0.35 + 0.4 * s;
    const base = 95 + Math.random() * 40;

    const osc = ctx.createOscillator();
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(base * (1.8 + s), t);
    osc.frequency.exponentialRampToValueAtTime(base, t + length * 0.6);

    // Vibrato that fades out, for the wobble.
    const lfo = ctx.createOscillator();
    lfo.frequency.value = 11 + Math.random() * 4;
    const depth = ctx.createGain();
    depth.gain.setValueAtTime(base * 0.35, t);
    depth.gain.exponentialRampToValueAtTime(1, t + length);
    lfo.connect(depth).connect(osc.frequency);

    const low = ctx.createBiquadFilter();
    low.type = 'lowpass';
    low.frequency.value = 1400;
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(level, t + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, t + length);
    osc.connect(low).connect(gain).connect(this.out);
    for (const o of [osc, lfo]) {
      o.start(t);
      o.stop(t + length + 0.05);
    }
  }
}
