/*
  sketch.js
  Every graphic on the page is drawn the way the to-go cup in Last Call is
  drawn: one loose pass of the pen per line, a little bowed, overshooting its
  corners, no fills, no shading. Where one drawn thing sits on top of another,
  a "knock" lets the page show through so the lines behind it disappear, the
  way they would on paper. Lines draw themselves in when they scroll into view.
  Everything is seeded, so the drawing is identical on every visit.
*/

const SVG = 'http://www.w3.org/2000/svg';
const f = (n) => Math.round(n * 10) / 10;

export function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const jit = (r, a) => (r() - 0.5) * 2 * a;

/* ---------- the pen ---------- */

/* one stroke from a to b: bowed, with a small overshoot at each end */
export function stroke(x1, y1, x2, y2, r, { bow = 1.2, over = 3, shake = 0.7 } = {}) {
  const len = Math.hypot(x2 - x1, y2 - y1) || 1;
  const ux = (x2 - x1) / len, uy = (y2 - y1) / len;
  const nx = -uy, ny = ux;
  const b = Math.min(len * 0.012, 3.4) * bow;
  const s0 = r() * over, s1 = r() * over;
  const ax = x1 - ux * s0 + nx * jit(r, shake), ay = y1 - uy * s0 + ny * jit(r, shake);
  const bx = x2 + ux * s1 + nx * jit(r, shake), by = y2 + uy * s1 + ny * jit(r, shake);
  const k1 = jit(r, b), k2 = jit(r, b);
  return `M${f(ax)} ${f(ay)}C${f(ax + (bx - ax) * 0.33 + nx * k1)} ${f(ay + (by - ay) * 0.33 + ny * k1)} ${f(ax + (bx - ax) * 0.66 + nx * k2)} ${f(ay + (by - ay) * 0.66 + ny * k2)} ${f(bx)} ${f(by)}`;
}

/* straight edges between points, each one its own stroke */
export function edges(pts, r, o = {}, closed = true) {
  let d = '';
  const n = closed ? pts.length : pts.length - 1;
  for (let k = 0; k < n; k++) {
    const [x1, y1] = pts[k], [x2, y2] = pts[(k + 1) % pts.length];
    d += stroke(x1, y1, x2, y2, r, o);
  }
  return d;
}

export const rectPts = (x, y, w, h) => [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];

export function smooth(pts) {
  let d = `M${f(pts[0][0])} ${f(pts[0][1])}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
    d += `C${f(p1[0] + (p2[0] - p0[0]) / 6)} ${f(p1[1] + (p2[1] - p0[1]) / 6)} ${f(p2[0] - (p3[0] - p1[0]) / 6)} ${f(p2[1] - (p3[1] - p1[1]) / 6)} ${f(p2[0])} ${f(p2[1])}`;
  }
  return d;
}

export function ringPts(cx, cy, rx, ry, r, { turns = 1, wobble = 0.02, steps = 40, start = r() * Math.PI * 2, drift = 0 } = {}) {
  const pts = [];
  const p1 = r() * 10, p2 = r() * 10;
  const n = Math.ceil(steps * turns);
  for (let i = 0; i <= n; i++) {
    const a = start + (i / steps) * Math.PI * 2;
    const k = 1 + Math.sin(a * 2 + p1) * wobble + Math.sin(a * 3 + p2) * wobble * 0.6 + (i / n) * drift;
    pts.push([cx + Math.cos(a) * rx * k, cy + Math.sin(a) * ry * k]);
  }
  return pts;
}
/* a circle drawn in one go: it runs a little past where it started */
export const circle = (cx, cy, rx, ry, r, o = {}) => smooth(ringPts(cx, cy, rx, ry, r, { turns: 1.08, drift: 0.02, ...o }));

/* oil pastel scribbled into a shape: one continuous zigzag, the way you fill a
   region with a crayon without lifting it */
export function scribble(pts, r, { angle = -35, gap = 9, jitter = 2.5 } = {}) {
  const a = (angle * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a);
  const rp = pts.map(([x, y]) => [x * ca + y * sa, -x * sa + y * ca]);
  const ys = rp.map((p) => p[1]);
  const maxY = Math.max(...ys);
  const out = [];
  let flip = false;
  for (let y = Math.min(...ys) + gap * 0.6; y < maxY; y += gap * (0.8 + r() * 0.4)) {
    const xs = [];
    for (let i = 0; i < rp.length; i++) {
      const [x1, y1] = rp[i], [x2, y2] = rp[(i + 1) % rp.length];
      if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) xs.push(x1 + ((y - y1) / (y2 - y1)) * (x2 - x1));
    }
    if (xs.length < 2) continue;
    xs.sort((m, n) => m - n);
    const x0 = xs[0] + r() * 6, x1 = xs[xs.length - 1] - r() * 6;
    const row = flip ? [[x1, y], [x0, y]] : [[x0, y], [x1, y]];
    flip = !flip;
    row.forEach(([x, yy]) => {
      const j = yy + jit(r, jitter);
      out.push([x * ca - j * sa, x * sa + j * ca]);
    });
  }
  return out.length ? `M${out.map(([x, y]) => `${f(x)} ${f(y)}`).join('L')}` : '';
}

const knock = (pts) => `M${pts.map(([x, y]) => `${f(x)} ${f(y)}`).join('L')}Z`;
const outlined = (pts, r, o) => [['knock', knock(pts)], ['ink', edges(pts, r, o)]];

export function dots(x1, y, x2, r, gap = 5.5) {
  let d = '';
  for (let x = x1 + r() * 3; x < x2 - 1; x += gap * (0.7 + r() * 0.6)) {
    d += `M${f(x)} ${f(y + jit(r, 0.6))}l${f(0.4 + r() * 0.8)} ${f(jit(r, 0.4))}`;
  }
  return d;
}
export function dashes(x1, y, x2, r, dash = 6, gap = 4) {
  let d = '';
  for (let x = x1 + r() * 2; x < x2 - 2; x += dash + gap * (0.6 + r() * 0.8)) {
    const len = dash * (0.6 + r() * 0.6);
    d += stroke(x, y + jit(r, 0.5), Math.min(x + len, x2), y + jit(r, 0.5), r, { over: 0.5, shake: 0.3, bow: 0.3 });
  }
  return d;
}

const tapePts = (w, h, r) => {
  const teeth = (x, dir) => {
    const pts = [];
    const n = Math.max(3, Math.round(h / 5));
    for (let i = 0; i <= n; i++) pts.push([x + dir * (i % 2 ? 2.5 + r() * 2 : r()), (h * i) / n]);
    return pts;
  };
  return [...teeth(w, -1), ...teeth(0, 1).reverse()];
};

/* ---------- what to draw for each kind of thing ---------- */

const flat = (y, w, r, o) => stroke(0, y, w, y, r, { over: 4, bow: 1.4, ...o });

export const RECIPES = {
  ruleTop: (w, h, r) => [['ink rule', flat(0.5, w, r)]],
  ruleBottom: (w, h, r) => [['ink rule', flat(h - 0.5, w, r)]],
  ruleTopBottom: (w, h, r) => [['ink rule', flat(0.5, w, r) + flat(h - 0.5, w, r)]],
  ruleLeft: (w, h, r) => [['ink rule', stroke(0.5, 3, 0.5, h - 3, r, { over: 2 })]],
  ruleMid: (w, h, r) => [['ink rule', stroke(0, h / 2, w, h / 2, r, { over: 2, bow: 1.6 })]],
  ruleDoubleBoth: (w, h, r) => [['ink rule', flat(0, w, r, { over: 5 }) + flat(4.5, w, r, { over: 5 }) + flat(h - 4.5, w, r, { over: 5 }) + flat(h, w, r, { over: 5 })]],
  ruleDoubleTop: (w, h, r) => [['ink rule', flat(0, w, r, { over: 5 }) + flat(4.5, w, r, { over: 5 })]],
  edition: (w, h, r) => [['ink', flat(0, w, r, { over: 6 }) + flat(4.5, w, r, { over: 6 })], ['ink rule', flat(h, w, r)]],
  columnRule: (w, h, r) => [['ink rule', stroke(w / 2, 0, w / 2, h - 40, r, { over: 6, bow: 1.4 })]],
  heavyBottom: (w, h, r) => [['ink', flat(h, w, r, { over: 3 })]],
  heavyTop: (w, h, r) => [['ink', flat(0, w, r, { over: 3 })]],
  underline: (w, h, r) => [['ink under', stroke(-2, h + 1.5, w + 2, h + 0.5 + jit(r, 1), r, { bow: 2.2, over: 3 })]],
  dots: (w, h, r) => [['ink dots', dots(0, h - 1, w, r)]],
  dashed: (w, h, r) => [['ink dash', dashes(0, h / 2, w, r)]],
  box: (w, h, r) => [['ink', edges(rectPts(0, 0, w, h), r, { over: 2, shake: 0.4 })]],
  paper: (w, h, r) => outlined(rectPts(0, 0, w, h), r),
  print: (w, h, r) => outlined(rectPts(0, 0, w, h), r),
  card: (w, h, r) => [...outlined(rectPts(0, 0, w, h), r, { over: 6 }), ['ink thin', edges(rectPts(16, 16, w - 32, h - 32), r, { over: 4, bow: 1.5 })]],
  receipt: (w, h, r) => {
    const zig = [];
    let i = 0;
    for (let x = w; x > 0; x -= 7.5, i++) zig.push([x, h - (i % 2 ? 0 : 8) - r() * 1.5]);
    zig.push([0, h - 8]);
    return [
      ['knock', knock([[0, 0], [w, 0], ...zig])],
      ['ink', edges([[0, h - 8], [0, 0], [w, 0], [w, h - 8]], r, {}, false) + edges([[w, h - 8], ...zig], r, { over: 0.6, shake: 0.3, bow: 0.2 }, false)],
    ];
  },
  check: (w, h, r) => {
    let holes = '';
    for (let x = 12; x < w - 8; x += 16) holes += circle(x + jit(r, 0.6), 9, 2.6, 2.6, r, { steps: 8, wobble: 0.08 });
    return [...outlined(rectPts(0, 0, w, h), r), ['ink thin', holes]];
  },
  tape: (w, h, r) => { const pts = tapePts(w, h, r); return [['knock', knock(pts)], ['ink faint', edges(pts, r, { over: 1, shake: 0.4, bow: 0.4 })]]; },
  pin: (w, h, r) => {
    const c = w / 2;
    return [
      ['knock', `${smooth(ringPts(c, c, c, c, r, { steps: 14, wobble: 0.02 }))}Z`],
      ['ink', circle(c, c, c, c, r, { steps: 16, wobble: 0.05 })],
      ['ink thin', stroke(c - c * 0.45, c + c * 0.05, c - c * 0.05, c - c * 0.45, r, { bow: 2, over: 0.5 })],
    ];
  },
  nail: (w, h, r) => [['knock', `${smooth(ringPts(w / 2, h / 2, w / 2, h / 2, r, { steps: 10 }))}Z`], ['ink', circle(w / 2, h / 2, w / 2, h / 2, r, { steps: 12, wobble: 0.06 })]],
  peg: (w, h, r) => [
    ...outlined(rectPts(0, 0, w, h), r, { over: 1.5, shake: 0.4 }),
    ['ink', stroke(-2.5, h * 0.37, w + 2.5, h * 0.37, r, { over: 1 }) + stroke(-2.5, h * 0.37 + 4, w + 2.5, h * 0.37 + 4, r, { over: 1 })],
  ],
  ledge: (w, h, r) => [...outlined(rectPts(0, 0, w, h), r, { over: 4 }), ['ink thin', stroke(0, 4.5, w, 4.5, r, { over: 3 })]],
  clip: (w, h, r) => {
    const body = [[w * 0.12, h * 0.48], [w * 0.88, h * 0.48], [w * 0.83, h * 0.97], [w * 0.17, h * 0.97]];
    const loop = (x1, x2, top) => smooth([[x1, h * 0.5], [x1 - 1, h * 0.3], [(x1 + x2) / 2, top], [x2 + 1, h * 0.3], [x2, h * 0.5]]);
    return [['ink', loop(w * 0.3, w * 0.7, h * 0.02) + loop(w * 0.37, w * 0.63, h * 0.2)], ...outlined(body, r, { over: 2 })];
  },
  frame: (w, h, r, el) => {
    const t = parseFloat(getComputedStyle(el).getPropertyValue('--frame')) || 11;
    const m = parseFloat(getComputedStyle(el).paddingTop) || 30;
    const miters = stroke(0, 0, t, t, r, { over: 1 }) + stroke(w, 0, w - t, t, r, { over: 1 }) + stroke(w, h, w - t, h - t, r, { over: 1 }) + stroke(0, h, t, h - t, r, { over: 1 });
    const layers = [
      ['knock', knock(rectPts(0, 0, w, h))],
      ['ink', edges(rectPts(0, 0, w, h), r) + edges(rectPts(t, t, w - 2 * t, h - 2 * t), r, { over: 2 }) + miters],
      ['ink thin', edges(rectPts(m - 1, m - 1, w - 2 * m + 2, h - 2 * m + 2), r, { over: 2 })],
    ];
    if (el.closest('.sways')) layers.push(['ink thin', stroke(w / 2, -30, w * 0.26, t * 0.6, r, { bow: 0.6 }) + stroke(w / 2, -30, w * 0.74, t * 0.6, r, { bow: 0.6 })]);
    return layers;
  },
  twine: (w, h, r) => {
    const pts = [];
    for (let i = 0; i <= 12; i++) {
      const t = i / 12;
      pts.push([-30 + (w + 60) * t, 6 + 80 * t * (1 - t) + jit(r, 0.8)]);
    }
    return [['ink', smooth(pts)]];
  },
  /* marker loops: the scribbled circle you put around the thing that matters */
  loop: (w, h, r) => [['ink marker', circle(w / 2, h / 2 + 1, w / 2 + 20, h / 2 + 13, r, { steps: 26, turns: 1.14, wobble: 0.05, drift: 0.07 })]],
  numeral: (w, h, r) => [['ink marker', circle(w / 2, h / 2, w / 2 + 10, h / 2 + 7, r, { steps: 18, turns: 1.12, wobble: 0.08, drift: 0.1 })]],
  oval: (w, h, r) => [['ink marker', circle(w / 2, h / 2, w / 2 + 34, h / 2 + 20, r, { steps: 34, turns: 1.1, wobble: 0.04, drift: 0.05 })]],
  /* a five-point star, the kind you draw next to something in a margin */
  star: (w, h, r) => {
    const c = w / 2, R = w / 2, k = R * 0.42;
    const pts = [];
    for (let i = 0; i < 10; i++) {
      const a = -Math.PI / 2 + (i * Math.PI) / 5;
      const rad = (i % 2 ? k : R) * (1 + jit(r, 0.08));
      pts.push([c + Math.cos(a) * rad, c + Math.sin(a) * rad]);
    }
    return [['ink star', edges(pts, r, { over: 1.2, shake: 0.3, bow: 0.4 })]];
  },

  /* chart axes through the middle, with arrowheads and ticks */
  axes: (w, h, r) => {
    const cx = w / 2, cy = h / 2, a = 11;
    let d = stroke(8, cy, w - 8, cy, r, { over: 2, bow: 1.6 }) + stroke(cx, 8, cx, h - 8, r, { over: 2, bow: 1.6 });
    d += stroke(w - 8, cy, w - 8 - a, cy - a * 0.6, r, { over: 0.5 }) + stroke(w - 8, cy, w - 8 - a, cy + a * 0.6, r, { over: 0.5 });
    d += stroke(8, cy, 8 + a, cy - a * 0.6, r, { over: 0.5 }) + stroke(8, cy, 8 + a, cy + a * 0.6, r, { over: 0.5 });
    d += stroke(cx, 8, cx - a * 0.6, 8 + a, r, { over: 0.5 }) + stroke(cx, 8, cx + a * 0.6, 8 + a, r, { over: 0.5 });
    d += stroke(cx, h - 8, cx - a * 0.6, h - 8 - a, r, { over: 0.5 }) + stroke(cx, h - 8, cx + a * 0.6, h - 8 - a, r, { over: 0.5 });
    let ticks = '';
    for (let i = 1; i < 10; i++) {
      if (i === 5) continue;
      const x = (w * i) / 10, y = (h * i) / 10;
      ticks += stroke(x, cy - 5, x, cy + 5, r, { over: 0.5, bow: 0.3 }) + stroke(cx - 5, y, cx + 5, y, r, { over: 0.5, bow: 0.3 });
    }
    return [['ink axis', d], ['ink thin', ticks]];
  },

  ring: (w, h, r) => {
    const c = w / 2;
    return [['ink faint', circle(c, c, c - 10, c - 12, r, { turns: 0.93, wobble: 0.03 })]];
  },
};

/* ---------- apply to the page ---------- */

const PLAN = [
  ['[data-sk]', (el) => el.dataset.sk],
  ['.box', 'box'],
  ['.leader', 'dots'],
  ['.pen-link', 'underline'],
  ['.contacts a, .qa > div, .name-line', 'ruleBottom'],
  ['.item', 'ruleTop'],
  ['.card-sub > span', 'ruleMid'],
];

const PAD = 40;
let counter = 1;

function paint(host) {
  const w = host.offsetWidth, h = host.offsetHeight;
  if (!w && !h) return;
  const key = `${w}x${h}`;
  if (host._skKey === key) return;
  host._skKey = key;
  const r = rng(host._skSeed);
  const layers = RECIPES[host._skRecipe](w, h, r, host);
  const svg = host._skSvg;
  svg.setAttribute('width', w + PAD * 2);
  svg.setAttribute('height', h + PAD * 2);
  svg.setAttribute('viewBox', `${-PAD} ${-PAD} ${w + PAD * 2} ${h + PAD * 2}`);
  svg.innerHTML = layers.filter(([, d]) => d).map(([cls, d]) => `<path class="${cls}" pathLength="1" d="${d}"/>`).join('');
}

export function sketchPage(root = document) {
  const hosts = [];
  for (const [selector, recipe] of PLAN) {
    root.querySelectorAll(selector).forEach((el) => {
      const name = typeof recipe === 'function' ? recipe(el) : recipe;
      if (el._skRecipe || !RECIPES[name]) return;
      el._skRecipe = name;
      el._skSeed = counter++ * 7919 + name.length * 131;
      if (getComputedStyle(el).position === 'static') el.style.position = 'relative';
      const svg = document.createElementNS(SVG, 'svg');
      svg.setAttribute('class', `sk sk-${name}`);
      svg.setAttribute('aria-hidden', 'true');
      svg.setAttribute('focusable', 'false');
      el._skSvg = svg;
      el.prepend(svg);
      hosts.push(el);
    });
  }
  const ro = new ResizeObserver((entries) => entries.forEach((e) => paint(e.target)));
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      e.target._skSvg.classList.add('drawn');
      io.unobserve(e.target);
    });
  }, { rootMargin: '0px 0px -8% 0px' });
  hosts.forEach((el) => { paint(el); ro.observe(el); io.observe(el); });
}

/* ---------- the cup, drawn from above ---------- */

export function drawCup(art, rim) {
  const r = rng(1907);
  const C = 500;
  const L = [];

  // a blue pastel ground scribbled under the saucer, like the croissant's
  L.push(['crayon blue', scribble(ringPts(C + 40, C + 60, 520, 470, r, { steps: 40, wobble: 0.06 }), r, { angle: -28, gap: 26, jitter: 9 })]);

  // saucer: outer edge and the well the cup sits in

  // handle, sticking out to the upper right, hidden where it meets the cup
  const handleLater = [];
  const ang = (-36 * Math.PI) / 180;
  const along = (d, off) => [C + Math.cos(ang) * d - Math.sin(ang) * off, C + Math.sin(ang) * d + Math.cos(ang) * off];
  const handle = [along(318, -44), along(440, -40), along(462, -20), along(466, 0), along(462, 20), along(440, 40), along(318, 44)];
  handleLater.push(['knock', knock(handle)]);
  handleLater.push(['crayon cream', scribble(handle, r, { angle: -36, gap: 12, jitter: 3 })]);
  handleLater.push(['ink', smooth(handle)]);
  handleLater.push(['ink thin', circle(...along(414, 0), 20, 13, r, { steps: 16, wobble: 0.06 })]);

  // the saucer, laid in with a pale butter pastel and a rosy lip
  L.push(['knock', `${smooth(ringPts(C, C, 476, 476, r, { steps: 60, wobble: 0.004 }))}Z`]);
  L.push(['crayon butter', scribble(ringPts(C, C, 468, 468, r, { steps: 50, wobble: 0.01 }), r, { angle: 55, gap: 22, jitter: 6 })]);
  L.push(['crayon rose', smooth(ringPts(C, C, 452, 452, r, { steps: 50, turns: 0.55, wobble: 0.012 }))]);
  L.push(['ink', circle(C, C, 476, 476, r, { steps: 60, wobble: 0.007, drift: 0.012 })]);
  L.push(['ink thin', circle(C, C, 396, 396, r, { steps: 50, turns: 0.86, wobble: 0.01, drift: 0 })]);

  L.push(...handleLater);

  // the mug
  L.push(['knock', `${smooth(ringPts(C, C, 338, 338, r, { steps: 44, wobble: 0.004 }))}Z`]);
  L.push(['crayon cream', scribble(ringPts(C, C, 330, 330, r, { steps: 40, wobble: 0.01 }), r, { angle: 20, gap: 20, jitter: 5 })]);
  L.push(['ink', circle(C, C, 338, 338, r, { steps: 56, wobble: 0.008, drift: 0.015 })]);

  art.innerHTML = L.map(([cls, d]) => `<path class="${cls}" pathLength="1" d="${d}"/>`).join('');

  // the inside rim, drawn over the edge of the coffee
  rim.innerHTML = `<path class="ink" pathLength="1" d="${circle(C, C, 298, 298, r, { steps: 56, wobble: 0.008, drift: 0.012 })}"/>`;
}
