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

  // The only shader you see. The fluid underneath is real; this pass just shows
  // the photograph Isfar took of her latte, with a soft edge where the liquid
  // meets the cup. Stirring drags the foam along, the way it does in the cup.
  display: `
    uniform sampler2D uDye;
    uniform vec2 uDyeTexel;
    uniform float uWeight;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
    float vnoise(vec2 p) {
      vec2 i = floor(p), u = fract(p);
      u = u * u * (3.0 - 2.0 * u);
      return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
    }
    void main() {
      vec2 uv = vUv;
      vec3 col = texture(uDye, uv).rgb;

      // the photograph loses a little contrast through the simulation, so put it back
      col = clamp((col - 0.5) * 1.06 + 0.5, 0.0, 1.0);

      vec2 p = uv - 0.5;
      float d = length(p);
      float ang = atan(p.y, p.x);

      // the liquid sits a hair inside the cup, and darkens where it meets the wall
      float edge = 0.492 - 0.004 * vnoise(vec2(ang * 4.0, 2.0));
      col *= 1.0 - smoothstep(edge - 0.08, edge, d) * 0.22;

      // the shine the ceramic throws across the surface, steady while it swirls
      float shine = smoothstep(0.82, 0.995, vnoise(vec2(ang * 1.6 + 2.2, 0.5))) * smoothstep(0.30, 0.47, d);
      col += shine * 0.07;

      float a = 1.0 - smoothstep(edge - 0.006, edge + 0.004, d);
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

/* The starting latte, painted on a 2D canvas: rose crema, mottling, and a heart. */
export function paintLatte(size = 512) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const x = c.getContext('2d');
  x.scale(size / 512, size / 512);
  const rand = mulberry32(2025);

  const crema = x.createRadialGradient(256, 270, 20, 256, 256, 256);
  crema.addColorStop(0, '#cd8b72');
  crema.addColorStop(0.5, '#bd7359');
  crema.addColorStop(0.82, '#a75c44');
  crema.addColorStop(1, '#7c3b2b');
  x.fillStyle = crema;
  x.fillRect(0, 0, 512, 512);

  for (let i = 0; i < 900; i++) {
    const a = rand() * Math.PI * 2;
    const r = Math.sqrt(rand()) * 250;
    x.fillStyle = rand() > 0.5 ? 'rgba(70,24,14,0.10)' : 'rgba(226,160,134,0.09)';
    x.beginPath();
    x.arc(256 + Math.cos(a) * r, 256 + Math.sin(a) * r, 1 + rand() * 5, 0, Math.PI * 2);
    x.fill();
  }

  const MILK = '#f8efe8';
  const CREMA = 'rgba(166, 84, 62, 0.92)';

  // a heart, the way it lands when the pitcher comes in close and lifts at the end:
  // one wide lobe of milk, a couple of rings inside it, and a line pulled through
  const heart = (cx, cy, w, h) => {
    const p = new Path2D();
    p.moveTo(cx, cy + h * 0.5);
    p.bezierCurveTo(cx - w * 0.64, cy + h * 0.04, cx - w * 0.54, cy - h * 0.54, cx - w * 0.17, cy - h * 0.36);
    p.bezierCurveTo(cx - w * 0.07, cy - h * 0.31, cx - w * 0.03, cy - h * 0.25, cx, cy - h * 0.18);
    p.bezierCurveTo(cx + w * 0.03, cy - h * 0.25, cx + w * 0.07, cy - h * 0.31, cx + w * 0.17, cy - h * 0.36);
    p.bezierCurveTo(cx + w * 0.54, cy - h * 0.54, cx + w * 0.64, cy + h * 0.04, cx, cy + h * 0.5);
    return p;
  };
  const body = heart(256, 252, 340, 330);

  x.save();
  x.fillStyle = MILK;
  x.shadowColor = 'rgba(248, 239, 232, 0.9)';
  x.shadowBlur = 14;
  x.fill(body);
  x.restore();

  // the rings inside: each pass of the pitcher leaves a thin crescent of crema
  x.save();
  x.clip(body);
  x.strokeStyle = CREMA;
  x.lineJoin = 'round';
  x.shadowColor = CREMA;
  x.shadowBlur = 3;
  for (let i = 0; i < 2; i++) {
    const k = 0.66 - i * 0.3;
    const wob = (rand() - 0.5) * 7;
    x.lineWidth = 5.5 - i * 1.4 + rand();
    x.stroke(heart(256 + wob * 0.5, 250 - i * 12 + wob, 340 * k, 330 * k));
  }
  x.restore();

  // the pull-through: fine at the top, heavier where the pitcher lifted off
  x.save();
  x.clip(body);
  x.fillStyle = CREMA;
  x.shadowColor = CREMA;
  x.shadowBlur = 2;
  x.beginPath();
  x.moveTo(256.5, 70);
  x.quadraticCurveTo(254.5, 280, 252.5, 420);
  x.lineTo(260, 420);
  x.quadraticCurveTo(258.5, 280, 258.5, 70);
  x.closePath();
  x.fill();
  x.restore();

  // a little foam texture over the milk
  for (let i = 0; i < 500; i++) {
    const px = 150 + rand() * 220;
    const py = 110 + rand() * 330;
    const k = size / 512; // hit-testing happens in canvas pixels, not the scaled space
    if (!x.isPointInPath(body, px * k, py * k)) continue;
    x.fillStyle = rand() > 0.5 ? 'rgba(255,250,246,0.25)' : 'rgba(180,110,86,0.06)';
    x.beginPath();
    x.arc(px, py, 0.6 + rand() * 1.6, 0, Math.PI * 2);
    x.fill();
  }

  return c;
}

export function createLatte(canvas, { onStir, photo, reducedMotion = false } = {}) {
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

  /* ---------- the surface we pour: Isfar's photo, or the drawn one until it loads ---------- */
  let shot = null;
  if (photo) {
    const img = new Image();
    img.decoding = 'async';
    img.src = photo;
    (img.decode ? img.decode() : Promise.resolve()).then(() => { shot = img; pour(); }).catch(() => {});
  }

  function surface(size) {
    if (!shot) return paintLatte(size);
    const c = document.createElement('canvas');
    c.width = c.height = size;
    const x = c.getContext('2d');
    x.fillStyle = '#a9604a';                    // behind the corners the photo doesn't cover
    x.fillRect(0, 0, size, size);
    x.drawImage(shot, 0, 0, size, size);
    return c;
  }

  /* ---------- pour: upload the latte, still the liquid ---------- */
  function pour() {
    const art = surface(DYE_RES);
    stirAmount = 0;
    onStir?.(0);
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
    gl.uniform1f(u.uWeight, dpr());
    blit(null);
  }

  /* ---------- loop: only runs while the liquid is moving ---------- */
  let raf = 0;
  let last = 0;
  let awakeUntil = 0;
  let visible = true;
  let stirred = false;
  let stirAmount = 0;
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
        // a short, shallow pass near the rim: enough to show the surface moves,
        // not enough to take the heart out before anyone has seen it
        const a = -2.75 + e * 1.05;
        const x = 0.5 + Math.cos(a) * 0.35;
        const y = 0.5 + Math.sin(a) * 0.35;
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
    const moved = Math.hypot(x - spoon.x, y - spoon.y);
    spoon.dx += x - spoon.x;
    spoon.dy += y - spoon.y;
    spoon.x = x;
    spoon.y = y;
    script = null;
    stirred = true;
    // how far the spoon has travelled across the cup, in cup widths
    if (Math.hypot(x - 0.5, y - 0.5) < 0.47 && moved < 0.3) stirAmount += moved;
    onStir?.(stirAmount);
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
