// WebGL2 renderer: the filled-in background as a quad, with the jelly mesh,
// textured with the cut-out subject, drawn over it.

const VERT = `#version 300 es
in vec2 aPos;    // image pixels
in vec2 aUv;
in float aStrain;
uniform vec2 uImage;   // image size, px
uniform vec4 uFit;     // scale.xy, offset.xy in clip space
out vec2 vUv;
out float vStrain;
void main() {
  vec2 p = aPos / uImage * 2.0 - 1.0;
  gl_Position = vec4(p * uFit.xy + uFit.zw, 0.0, 1.0);
  gl_Position.y = -gl_Position.y;
  vUv = aUv;
  vStrain = aStrain;
}`;

const FRAG = `#version 300 es
precision mediump float;
in vec2 vUv;
in float vStrain;
uniform sampler2D uTex;
uniform float uSheen;
out vec4 outColor;
void main() {
  vec4 c = texture(uTex, vUv); // premultiplied
  // Squashed jelly darkens a touch, stretched jelly goes glossy and pale.
  float s = clamp(vStrain * 1.6, -0.25, 0.35) * uSheen;
  vec3 rgb = s > 0.0 ? mix(c.rgb, vec3(c.a), s) : c.rgb * (1.0 + s);
  outColor = vec4(rgb, c.a);
}`;

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { premultipliedAlpha: true, antialias: true, alpha: true });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;
    this.prog = program(gl, VERT, FRAG);
    this.loc = {
      aPos: gl.getAttribLocation(this.prog, 'aPos'),
      aUv: gl.getAttribLocation(this.prog, 'aUv'),
      aStrain: gl.getAttribLocation(this.prog, 'aStrain'),
      uImage: gl.getUniformLocation(this.prog, 'uImage'),
      uFit: gl.getUniformLocation(this.prog, 'uFit'),
      uTex: gl.getUniformLocation(this.prog, 'uTex'),
      uSheen: gl.getUniformLocation(this.prog, 'uSheen'),
    };
    this.bg = null;
    this.jelly = null;
    this.showJelly = true;
    this.fit = { scale: 1, x: 0, y: 0 };
  }

  /** Background-only scene, e.g. the raw photo while it's being processed. */
  setBackground(source, w, h) {
    const gl = this.gl;
    this.width = w;
    this.height = h;
    if (this.bg) this.disposeVao(this.bg);
    const quad = new Float32Array([0, 0, w, 0, 0, h, w, h]);
    this.bg = this.makeVao(quad, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), null, null);
    this.bg.count = 4;
    this.bg.tex = texture(gl, source, false);
    this.resize();
  }

  /** The jelly layer: `source` is the RGBA cut-out, `mesh` from buildMesh(). */
  setJelly(source, mesh) {
    const gl = this.gl;
    if (this.jelly) this.disposeVao(this.jelly);
    if (!source) {
      this.jelly = null;
      return;
    }
    const uv = new Float32Array(mesh.rest.length);
    for (let i = 0; i < uv.length; i += 2) {
      uv[i] = mesh.rest[i] / this.width;
      uv[i + 1] = mesh.rest[i + 1] / this.height;
    }
    this.jelly = this.makeVao(mesh.rest, uv, new Float32Array(mesh.rest.length / 2), mesh.tris);
    this.jelly.count = mesh.tris.length;
    this.jelly.tex = texture(gl, source, true);
  }

  makeVao(pos, uv, strain, indices) {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buf = (data, loc, size, usage) => {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, data, usage);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      return b;
    };
    const out = {
      vao,
      pos: buf(pos, this.loc.aPos, 2, gl.DYNAMIC_DRAW),
      uv: buf(uv, this.loc.aUv, 2, gl.STATIC_DRAW),
    };
    if (strain) {
      out.strain = buf(strain, this.loc.aStrain, 1, gl.DYNAMIC_DRAW);
    } else {
      gl.disableVertexAttribArray(this.loc.aStrain);
      gl.vertexAttrib1f(this.loc.aStrain, 0);
    }
    if (indices) {
      out.index = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, out.index);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    }
    gl.bindVertexArray(null);
    return out;
  }

  disposeVao(v) {
    const gl = this.gl;
    gl.deleteVertexArray(v.vao);
    for (const b of [v.pos, v.uv, v.strain, v.index]) if (b) gl.deleteBuffer(b);
    gl.deleteTexture(v.tex);
  }

  /** Match the drawing buffer to the canvas's CSS size and refit the image. */
  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssW = this.canvas.clientWidth;
    const cssH = this.canvas.clientHeight;
    this.canvas.width = Math.max(1, Math.round(cssW * dpr));
    this.canvas.height = Math.max(1, Math.round(cssH * dpr));
    if (!this.width) return;
    const scale = Math.min(cssW / this.width, cssH / this.height);
    this.fit = {
      scale, // CSS px per image px
      x: (cssW - this.width * scale) / 2,
      y: (cssH - this.height * scale) / 2,
    };
  }

  /** Canvas-relative CSS px → image px. */
  toImage(cx, cy) {
    return [(cx - this.fit.x) / this.fit.scale, (cy - this.fit.y) / this.fit.scale];
  }

  draw(pos, strain) {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.bg) return;

    const cssW = this.canvas.clientWidth;
    const cssH = this.canvas.clientHeight;
    const sx = (this.width * this.fit.scale) / cssW;
    const sy = (this.height * this.fit.scale) / cssH;

    gl.useProgram(this.prog);
    gl.uniform2f(this.loc.uImage, this.width, this.height);
    gl.uniform4f(this.loc.uFit, sx, sy, 0, 0);
    gl.uniform1i(this.loc.uTex, 0);
    gl.activeTexture(gl.TEXTURE0);

    gl.disable(gl.BLEND);
    gl.uniform1f(this.loc.uSheen, 0);
    gl.bindVertexArray(this.bg.vao);
    gl.bindTexture(gl.TEXTURE_2D, this.bg.tex);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, this.bg.count);

    if (this.jelly && this.showJelly && pos) {
      // Keep the jelly inside the photo's frame.
      const dpr = this.canvas.width / cssW;
      gl.enable(gl.SCISSOR_TEST);
      const w = this.width * this.fit.scale;
      const h = this.height * this.fit.scale;
      gl.scissor(
        Math.round(this.fit.x * dpr),
        Math.round(this.canvas.height - (this.fit.y + h) * dpr), // GL's y axis points up
        Math.round(w * dpr),
        Math.round(h * dpr),
      );
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.uniform1f(this.loc.uSheen, 1);
      gl.bindVertexArray(this.jelly.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.jelly.pos);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, pos);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.jelly.strain);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, strain);
      gl.bindTexture(gl.TEXTURE_2D, this.jelly.tex);
      gl.drawElements(gl.TRIANGLES, this.jelly.count, gl.UNSIGNED_INT, 0);
      gl.disable(gl.SCISSOR_TEST);
    }
    gl.bindVertexArray(null);
  }
}

function program(gl, vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

function texture(gl, source, premultiply) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, premultiply);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}
