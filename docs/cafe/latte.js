/*
  latte.js
  A small stable-fluids solver (Jos Stam, "Stable Fluids", 1999) running on
  WebGL2, dressed up as a flat white. Velocity lives on a coarse grid, the
  crema and milk ("dye") on a finer one, and your cursor is the spoon.

  Each frame: vorticity confinement -> divergence -> Jacobi pressure solve ->
  subtract pressure gradient (so the liquid stays incompressible) -> advect
  velocity and dye along the flow.
*/

const SIM_RES = 128;
const PRESSURE_ITERATIONS = 20;
const VELOCITY_DISSIPATION = 1.3;
const PRESSURE_DECAY = 0.8;
const CURL = 12;
const SPOON_RADIUS = 0.0012;
const SPOON_FORCE = 5200;

const VERT = `#version 300 es
precision highp float;
in vec2 aPos;
uniform vec2 uTexel;
out vec2 vUv;
out vec2 vL;
out vec2 vR;
out vec2 vT;
out vec2 vB;
void main() {
  vUv = aPos * 0.5 + 0.5;
  vL = vUv - vec2(uTexel.x, 0.0);
  vR = vUv + vec2(uTexel.x, 0.0);
  vT = vUv + vec2(0.0, uTexel.y);
  vB = vUv - vec2(0.0, uTexel.y);
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAG_HEAD = `#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 vUv;
in vec2 vL;
in vec2 vR;
in vec2 vT;
in vec2 vB;
out vec4 fragColor;
`;

const SHADERS = {
  copy: `
    uniform sampler2D uTex;
    void main() { fragColor = texture(uTex, vUv); }`,

  scale: `
    uniform sampler2D uTex;
    uniform float uValue;
    void main() { fragColor = uValue * texture(uTex, vUv); }`,

  splat: `
    uniform sampler2D uTarget;
    uniform vec2 uPoint;
    uniform vec3 uValue;
    uniform float uRadius;
    void main() {
      vec2 p = vUv - uPoint;
      vec3 base = texture(uTarget, vUv).xyz;
      fragColor = vec4(base + uValue * exp(-dot(p, p) / uRadius), 1.0);
    }`,

  advect: `
    uniform sampler2D uVelocity;
    uniform sampler2D uSource;
    uniform vec2 uSimTexel;
    uniform float uDt;
    uniform float uDissipation;
    void main() {
      vec2 coord = vUv - uDt * texture(uVelocity, vUv).xy * uSimTexel;
      fragColor = texture(uSource, coord) / (1.0 + uDissipation * uDt);
    }`,

  divergence: `
    uniform sampler2D uVelocity;
    void main() {
      float L = texture(uVelocity, vL).x;
      float R = texture(uVelocity, vR).x;
      float T = texture(uVelocity, vT).y;
      float B = texture(uVelocity, vB).y;
      vec2 C = texture(uVelocity, vUv).xy;
      if (vL.x < 0.0) L = -C.x;
      if (vR.x > 1.0) R = -C.x;
      if (vT.y > 1.0) T = -C.y;
      if (vB.y < 0.0) B = -C.y;
      fragColor = vec4(0.5 * (R - L + T - B), 0.0, 0.0, 1.0);
    }`,

  curl: `
    uniform sampler2D uVelocity;
    void main() {
      float L = texture(uVelocity, vL).y;
      float R = texture(uVelocity, vR).y;
      float T = texture(uVelocity, vT).x;
      float B = texture(uVelocity, vB).x;
      fragColor = vec4(0.5 * (R - L - T + B), 0.0, 0.0, 1.0);
    }`,

  vorticity: `
    uniform sampler2D uVelocity;
    uniform sampler2D uCurl;
    uniform float uCurlAmount;
    uniform float uDt;
    void main() {
      float L = texture(uCurl, vL).x;
      float R = texture(uCurl, vR).x;
      float T = texture(uCurl, vT).x;
      float B = texture(uCurl, vB).x;
      float C = texture(uCurl, vUv).x;
      vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
      force /= length(force) + 0.0001;
      force *= uCurlAmount * C;
      force.y *= -1.0;
      vec2 vel = texture(uVelocity, vUv).xy + force * uDt;
      fragColor = vec4(clamp(vel, -1000.0, 1000.0), 0.0, 1.0);
    }`,

  pressure: `
    uniform sampler2D uPressure;
    uniform sampler2D uDivergence;
    void main() {
      float L = texture(uPressure, vL).x;
      float R = texture(uPressure, vR).x;
      float T = texture(uPressure, vT).x;
      float B = texture(uPressure, vB).x;
      float div = texture(uDivergence, vUv).x;
      fragColor = vec4((L + R + B + T - div) * 0.25, 0.0, 0.0, 1.0);
    }`,

  // subtract the pressure gradient, then let the cup wall stop the flow
  gradient: `
    uniform sampler2D uPressure;
    uniform sampler2D uVelocity;
    void main() {
      float L = texture(uPressure, vL).x;
      float R = texture(uPressure, vR).x;
      float T = texture(uPressure, vT).x;
      float B = texture(uPressure, vB).x;
      vec2 vel = texture(uVelocity, vUv).xy - vec2(R - L, T - B);
      vel *= 1.0 - smoothstep(0.43, 0.5, length(vUv - 0.5));
      fragColor = vec4(vel, 0.0, 1.0);
    }`,

  // The only shader you see. The fluid underneath is real; this pass paints it
  // in oil pastel, the way Isfar's drawings are made: crema laid in with short
  // diagonal strokes of umber, terracotta and a little rose, milk in cream with
  // the odd streak of pale blue, paper tooth showing through, and one loose ink
  // line wherever the milk meets the crema. Stirring drags the color along.
  display: `
    uniform sampler2D uDye;
    uniform vec2 uDyeTexel;
    uniform vec3 uInk;
    uniform float uWeight;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
    float vnoise(vec2 p) {
      vec2 i = floor(p), u = fract(p);
      u = u * u * (3.0 - 2.0 * u);
      return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
    }
    float lum(vec2 uv) { return dot(texture(uDye, uv).rgb, vec3(0.299, 0.587, 0.114)); }
    float contour(float l, float level, float width) {
      float fw = fwidth(l) + 1e-4;
      return 1.0 - smoothstep(width * 0.45, width, abs(l - level) / fw);
    }
    void main() {
      vec2 uv = vUv;
      vec2 wob = vec2(vnoise(uv * 6.0), vnoise(uv * 6.0 + 19.0)) - 0.5;
      vec2 suv = uv + wob * 0.006;
      vec2 t = uDyeTexel * 2.5;
      float l = (lum(suv) * 2.0 + lum(suv + vec2(t.x, 0.0)) + lum(suv - vec2(t.x, 0.0)) + lum(suv + vec2(0.0, t.y)) + lum(suv - vec2(0.0, t.y))) / 6.0;

      // pastel strokes: long thin noise along a diagonal, two directions layered
      mat2 rotA = mat2(0.82, -0.57, 0.57, 0.82);
      mat2 rotB = mat2(0.94, 0.34, -0.34, 0.94);
      float sA = vnoise(rotA * uv * vec2(220.0, 20.0));
      float sB = vnoise(rotB * uv * vec2(140.0, 14.0));
      float strokes = sA * 0.6 + sB * 0.4;
      float tone = l + (strokes - 0.5) * 0.16;

      vec3 umber = vec3(0.36, 0.21, 0.14);
      vec3 terracotta = vec3(0.72, 0.43, 0.30);
      vec3 peach = vec3(0.93, 0.70, 0.53);
      vec3 cream = vec3(0.98, 0.93, 0.82);
      vec3 col = umber;
      col = mix(col, terracotta, smoothstep(0.22, 0.34, tone));
      col = mix(col, peach, smoothstep(0.42, 0.56, tone));
      col = mix(col, cream, smoothstep(0.58, 0.68, tone));

      // accent streaks that follow the dye: rose in the crema, sky blue in the foam
      float blotch = vnoise(uv * 9.0 + 3.0);
      float crema = 1.0 - smoothstep(0.45, 0.6, tone);
      float foam = smoothstep(0.62, 0.75, tone);
      col = mix(col, vec3(0.90, 0.56, 0.60), crema * smoothstep(0.55, 0.9, sB) * smoothstep(0.45, 0.8, blotch) * 0.55);
      col = mix(col, vec3(0.64, 0.78, 0.90), foam * smoothstep(0.62, 0.95, sA) * smoothstep(0.5, 0.85, 1.0 - blotch) * 0.45);

      // paper tooth where the pastel skipped
      float tooth = smoothstep(0.66, 0.92, vnoise(uv * 480.0) * 0.55 + strokes * 0.45);
      col = mix(col, vec3(0.98, 0.95, 0.89), tooth * 0.55);

      // the ink line where milk meets crema
      float dry = 0.55 + 0.45 * smoothstep(0.2, 0.7, vnoise(uv * 34.0));
      float ink = contour(l, 0.6, 1.8 * uWeight) * dry;
      col = mix(col, uInk, ink * 0.9);

      // the colour stops a hair short of the rim, like it was coloured in by hand
      vec2 p = uv - 0.5;
      float ang = atan(p.y, p.x);
      float edge = 0.475 - 0.008 * vnoise(vec2(ang * 4.0, 2.0)) - 0.005 * sin(ang * 3.0 + 1.0);
      float a = 1.0 - smoothstep(edge - 0.004, edge + 0.004, length(p));
      a *= 0.93 + 0.07 * strokes;
      fragColor = vec4(col * a, a);
    }`,
};

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* The starting latte, painted on a 2D canvas: crema, mottling, and a rosetta. */
export function paintLatte(size = 512) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const x = c.getContext('2d');
  x.scale(size / 512, size / 512);
  const rand = mulberry32(2025);

  const crema = x.createRadialGradient(256, 270, 20, 256, 256, 256);
  crema.addColorStop(0, '#9c6a42');
  crema.addColorStop(0.5, '#7f5232');
  crema.addColorStop(0.82, '#643819');
  crema.addColorStop(1, '#3e2211');
  x.fillStyle = crema;
  x.fillRect(0, 0, 512, 512);

  for (let i = 0; i < 900; i++) {
    const a = rand() * Math.PI * 2;
    const r = Math.sqrt(rand()) * 250;
    x.fillStyle = rand() > 0.5 ? 'rgba(40,20,8,0.10)' : 'rgba(205,145,85,0.08)';
    x.beginPath();
    x.arc(256 + Math.cos(a) * r, 256 + Math.sin(a) * r, 1 + rand() * 5, 0, Math.PI * 2);
    x.fill();
  }

  const MILK = '#f1e3cb';
  const CREMA = 'rgba(128, 80, 45, 0.92)';

  // the milk body: a slightly lopsided teardrop with a soft, foamy edge
  const body = new Path2D();
  body.moveTo(256, 150);
  body.bezierCurveTo(340, 158, 378, 236, 372, 300);
  body.bezierCurveTo(366, 372, 318, 428, 258, 432);
  body.bezierCurveTo(196, 428, 142, 376, 140, 302);
  body.bezierCurveTo(138, 232, 176, 158, 256, 150);
  const head = new Path2D();
  head.ellipse(258, 124, 30, 25, -0.08, 0, Math.PI * 2);

  x.save();
  x.fillStyle = MILK;
  x.shadowColor = 'rgba(241, 227, 203, 0.9)';
  x.shadowBlur = 14;
  x.fill(body);
  x.fill(head);
  x.restore();

  // leaves: crescents of crema pushed into the milk, thick in the middle and
  // tapering at the tips, each one a little off like a real pour
  x.save();
  x.clip(body);
  x.fillStyle = CREMA;
  x.shadowColor = CREMA;
  x.shadowBlur = 3;
  for (let i = 0; i < 11; i++) {
    const y = 176 + i * 23 + (rand() - 0.5) * 5;
    const w = 44 + i * 12.5 + (rand() - 0.5) * 8;
    const lift = w * (0.34 + rand() * 0.08);
    const thick = 4.5 + rand() * 2.5 - i * 0.15;
    const sway = (rand() - 0.5) * 6 + (i - 5) * 0.6;
    const cx = 256 + sway;
    x.beginPath();
    x.moveTo(cx - w, y + w * 0.5);
    x.quadraticCurveTo(cx, y - lift, cx + w, y + w * 0.5 + (rand() - 0.5) * 6);
    x.quadraticCurveTo(cx, y - lift + thick * 2.2, cx - w, y + w * 0.5);
    x.fill();
  }
  x.restore();

  // the pull-through: fine at the top, a little heavier where the pitcher lifted off
  x.save();
  x.fillStyle = CREMA;
  x.shadowColor = CREMA;
  x.shadowBlur = 2;
  x.beginPath();
  x.moveTo(257, 92);
  x.quadraticCurveTo(255.5, 280, 254, 452);
  x.lineTo(260, 452);
  x.quadraticCurveTo(259, 280, 258.4, 92);
  x.closePath();
  x.fill();
  x.restore();

  // a little foam texture over the milk
  for (let i = 0; i < 500; i++) {
    const px = 150 + rand() * 220;
    const py = 110 + rand() * 330;
    const k = size / 512; // hit-testing happens in canvas pixels, not the scaled space
    if (!x.isPointInPath(body, px * k, py * k) && !x.isPointInPath(head, px * k, py * k)) continue;
    x.fillStyle = rand() > 0.5 ? 'rgba(255,250,240,0.25)' : 'rgba(150,110,70,0.06)';
    x.beginPath();
    x.arc(px, py, 0.6 + rand() * 1.6, 0, Math.PI * 2);
    x.fill();
  }

  return c;
}

export function createLatte(canvas, { onStir, reducedMotion = false } = {}) {
  const gl = canvas.getContext('webgl2', {
    alpha: true, depth: false, stencil: false, antialias: false, premultipliedAlpha: true,
  });
  if (!gl) return null;
  gl.getExtension('EXT_color_buffer_float');

  const dpr = () => Math.min(window.devicePixelRatio || 1, 2);
  const DYE_RES = canvas.clientWidth * dpr() > 640 ? 1024 : 512;

  /* ---------- programs ---------- */
  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }

  let programs;
  try {
    const vs = compile(gl.VERTEX_SHADER, VERT);
    programs = Object.fromEntries(Object.entries(SHADERS).map(([name, body]) => {
      const p = gl.createProgram();
      gl.attachShader(p, vs);
      gl.attachShader(p, compile(gl.FRAGMENT_SHADER, FRAG_HEAD + body));
      gl.bindAttribLocation(p, 0, 'aPos');
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      const u = {};
      const count = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
      for (let i = 0; i < count; i++) {
        const uname = gl.getActiveUniform(p, i).name;
        u[uname] = gl.getUniformLocation(p, uname);
      }
      return [name, () => { gl.useProgram(p); return u; }];
    }));
  } catch (err) {
    console.warn('[latte] shader setup failed, serving it still:', err);
    return null;
  }

  /* ---------- geometry: one full-screen quad ---------- */
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(0);

  function blit(target) {
    if (target) {
      gl.viewport(0, 0, target.w, target.h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    } else {
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  }

  /* ---------- framebuffers ---------- */
  function target(w, h, filter) {
    gl.activeTexture(gl.TEXTURE0);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);
    return {
      fbo, w, h, texel: [1 / w, 1 / h],
      attach(unit) { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); return unit; },
    };
  }

  function pair(w, h, filter) {
    let a = target(w, h, filter);
    let b = target(w, h, filter);
    return { get read() { return a; }, get write() { return b; }, swap() { [a, b] = [b, a]; }, texel: a.texel };
  }

  const probe = target(4, 4, gl.NEAREST);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) return null;
  gl.deleteFramebuffer(probe.fbo);

  const velocity = pair(SIM_RES, SIM_RES, gl.LINEAR);
  const dye = pair(DYE_RES, DYE_RES, gl.LINEAR);
  const pressure = pair(SIM_RES, SIM_RES, gl.NEAREST);
  const divergence = target(SIM_RES, SIM_RES, gl.NEAREST);
  const curl = target(SIM_RES, SIM_RES, gl.NEAREST);
  const simTexel = velocity.texel;

  /* ---------- pour: upload the painted latte, still the liquid ---------- */
  function pour() {
    const art = paintLatte(DYE_RES);
    gl.activeTexture(gl.TEXTURE0);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, art);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.uniform1i(programs.copy().uTex, 0);
    blit(dye.write);
    dye.swap();
    gl.deleteTexture(tex);

    for (const t of [velocity.read, velocity.write, pressure.read, pressure.write]) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    render();
  }

  /* ---------- solver ---------- */
  function splat(x, y, dx, dy) {
    const u = programs.splat();
    gl.uniform2f(u.uTexel, ...simTexel);
    gl.uniform1i(u.uTarget, velocity.read.attach(0));
    gl.uniform2f(u.uPoint, x, y);
    gl.uniform3f(u.uValue, dx * SPOON_FORCE, dy * SPOON_FORCE, 0);
    gl.uniform1f(u.uRadius, SPOON_RADIUS);
    blit(velocity.write);
    velocity.swap();
  }

  function step(dt) {
    let u = programs.curl();
    gl.uniform2f(u.uTexel, ...simTexel);
    gl.uniform1i(u.uVelocity, velocity.read.attach(0));
    blit(curl);

    u = programs.vorticity();
    gl.uniform2f(u.uTexel, ...simTexel);
    gl.uniform1i(u.uVelocity, velocity.read.attach(0));
    gl.uniform1i(u.uCurl, curl.attach(1));
    gl.uniform1f(u.uCurlAmount, CURL);
    gl.uniform1f(u.uDt, dt);
    blit(velocity.write);
    velocity.swap();

    u = programs.divergence();
    gl.uniform2f(u.uTexel, ...simTexel);
    gl.uniform1i(u.uVelocity, velocity.read.attach(0));
    blit(divergence);

    u = programs.scale();
    gl.uniform1i(u.uTex, pressure.read.attach(0));
    gl.uniform1f(u.uValue, PRESSURE_DECAY);
    blit(pressure.write);
    pressure.swap();

    u = programs.pressure();
    gl.uniform2f(u.uTexel, ...simTexel);
    gl.uniform1i(u.uDivergence, divergence.attach(0));
    for (let i = 0; i < PRESSURE_ITERATIONS; i++) {
      gl.uniform1i(u.uPressure, pressure.read.attach(1));
      blit(pressure.write);
      pressure.swap();
    }

    u = programs.gradient();
    gl.uniform2f(u.uTexel, ...simTexel);
    gl.uniform1i(u.uPressure, pressure.read.attach(0));
    gl.uniform1i(u.uVelocity, velocity.read.attach(1));
    blit(velocity.write);
    velocity.swap();

    u = programs.advect();
    gl.uniform2f(u.uSimTexel, ...simTexel);
    gl.uniform1f(u.uDt, dt);
    gl.uniform1i(u.uVelocity, velocity.read.attach(0));
    gl.uniform1i(u.uSource, velocity.read.attach(0));
    gl.uniform1f(u.uDissipation, VELOCITY_DISSIPATION);
    blit(velocity.write);
    velocity.swap();

    gl.uniform1i(u.uVelocity, velocity.read.attach(0));
    gl.uniform1i(u.uSource, dye.read.attach(1));
    gl.uniform1f(u.uDissipation, 0);
    blit(dye.write);
    dye.swap();
  }

  function render() {
    const u = programs.display();
    gl.uniform1i(u.uDye, dye.read.attach(0));
    gl.uniform2f(u.uDyeTexel, ...dye.texel);
    gl.uniform3f(u.uInk, 0.18, 0.13, 0.1);
    gl.uniform1f(u.uWeight, dpr());
    blit(null);
  }

  /* ---------- loop: only runs while the liquid is moving ---------- */
  let raf = 0;
  let last = 0;
  let awakeUntil = 0;
  let visible = true;
  let stirred = false;
  let script = null;
  const spoon = { x: 0.5, y: 0.5, dx: 0, dy: 0 };

  function wake(ms = 4500) {
    awakeUntil = Math.max(awakeUntil, performance.now() + ms);
    if (!raf && visible) {
      last = performance.now();
      raf = requestAnimationFrame(frame);
    }
  }

  function stir(x, y, dx, dy) {
    if (Math.hypot(x - 0.5, y - 0.5) > 0.47) return;
    splat(x, y, dx, dy);
  }

  function frame(now) {
    raf = 0;
    const dt = Math.min((now - last) / 1000, 1 / 30);
    last = now;

    if (script) {
      const k = (now - script.t0) / 1600;
      if (k >= 1) {
        script = null;
      } else {
        const e = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
        const a = -2.75 + e * 2.2;
        const x = 0.5 + Math.cos(a) * 0.29;
        const y = 0.5 + Math.sin(a) * 0.29;
        if (script.px !== undefined) stir(x, y, x - script.px, y - script.py);
        script.px = x;
        script.py = y;
      }
    }

    if (spoon.dx || spoon.dy) {
      stir(spoon.x, spoon.y, spoon.dx, spoon.dy);
      spoon.dx = spoon.dy = 0;
    }

    step(dt);
    render();
    if (now < awakeUntil && visible) raf = requestAnimationFrame(frame);
  }

  /* ---------- the spoon ---------- */
  const toUv = (e) => {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) / r.width, 1 - (e.clientY - r.top) / r.height];
  };
  const place = (e) => { [spoon.x, spoon.y] = toUv(e); };
  canvas.addEventListener('pointerenter', place);
  canvas.addEventListener('pointerdown', place);
  canvas.addEventListener('pointermove', (e) => {
    const [x, y] = toUv(e);
    spoon.dx += x - spoon.x;
    spoon.dy += y - spoon.y;
    spoon.x = x;
    spoon.y = y;
    script = null;
    if (!stirred) { stirred = true; onStir?.(); }
    wake();
  });

  /* ---------- sizing + visibility ---------- */
  function resize() {
    const w = Math.round(canvas.clientWidth * dpr());
    const h = Math.round(canvas.clientHeight * dpr());
    if (w && h && (canvas.width !== w || canvas.height !== h)) {
      canvas.width = w;
      canvas.height = h;
      render();
    }
  }
  new ResizeObserver(resize).observe(canvas);
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (visible && performance.now() < awakeUntil) wake(0);
  }).observe(canvas);

  canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    cancelAnimationFrame(raf);
    canvas.dispatchEvent(new CustomEvent('latte:lost', { bubbles: true }));
  });

  resize();
  pour();

  return {
    spec: { sim: SIM_RES, dye: DYE_RES, iterations: PRESSURE_ITERATIONS },
    pour() {
      script = null;
      pour();
    },
    demo() {
      if (reducedMotion || stirred || script) return;
      script = { t0: performance.now() };
      wake(3800);
    },
  };
}
