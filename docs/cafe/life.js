/*
  life.js
  Goldfish cut out of Isfar's pastel, swimming on the last page of the
  sketchbook. Each one wanders on its own, steers off the edges of the page,
  and darts away from the cursor. The caller decides when the page is open.
*/
import { rng } from './sketch.js';

const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const TAU = Math.PI * 2;
const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const turnToward = (from, to, amount) => from + wrapAngle(to - from) * Math.min(1, amount);

/* sprite name, and which way the fish's head points in the drawing (degrees) */
const FISH = [
  ['goldfish-1', -55], ['goldfish-7', -65], ['goldfish-3', -125], ['goldfish-4', -60],
  ['goldfish-5', -120], ['goldfish-2', -50], ['goldfish-6', -120], ['goldfish-8', -115],
];

export function goldfish(tank, pointerArea) {
  const r = rng(88);
  const small = innerWidth < 700;
  const fish = FISH.slice(0, small ? 4 : 7).map(([name, art], i) => {
    const img = new Image();
    img.src = `images/art/bits/${name}.webp`;
    img.alt = '';
    img.decoding = 'async';
    const size = (small ? 40 : 54) + r() * (small ? 16 : 30);
    img.style.width = `${size}px`;
    tank.append(img);
    return {
      img, size, art: (art * Math.PI) / 180,
      x: 0, y: 0, heading: r() * TAU, shown: 0, speed: 30, turn: 0,
      seed: r() * 100, cruise: 26 + r() * 22, placed: false,
    };
  });

  const pointer = { x: -9999, y: -9999, until: 0 };
  let w = 0, h = 0;
  const measure = () => { w = tank.clientWidth; h = tank.clientHeight; };

  const scare = (x, y, radius) => {
    for (const f of fish) {
      const d = Math.hypot(f.x - x, f.y - y);
      if (d < radius) {
        f.heading = Math.atan2(f.y - y, f.x - x) + (r() - 0.5) * 0.8;
        f.speed = 200 + r() * 90;
      }
    }
  };
  const local = (e) => {
    const rect = tank.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  };
  pointerArea.addEventListener('pointermove', (e) => {
    [pointer.x, pointer.y] = local(e);
    pointer.until = performance.now() + 400;
  });
  pointerArea.addEventListener('pointerdown', (e) => scare(...local(e), 260));

  /* The angle the fish is drawn at trails the angle it is swimming at, so a
     sharp turn reads as a body swinging round rather than a snap. */
  const draw = (f, t, dt) => {
    const wiggle = Math.sin(t * (3.4 + Math.min(f.speed, 140) / 90) + f.seed) * (0.035 + Math.min(f.speed, 200) / 5200);
    const target = f.heading - f.art + wiggle;
    f.shown = dt ? turnToward(f.shown, target, dt * 7) : target;
    f.img.style.transform = `translate3d(${(f.x - f.size / 2).toFixed(2)}px, ${(f.y - f.size / 2).toFixed(2)}px, 0) rotate(${f.shown.toFixed(4)}rad)`;
  };

  let raf = 0, last = 0, running = false, step = 1 / 60;
  const frame = (now) => {
    /* frames never arrive evenly spaced, so the step is smoothed: the fish keep
       real time, without the per-frame stutter that raw deltas give them */
    const raw = Math.min(0.05, Math.max(0.004, (now - last) / 1000));
    step += (raw - step) * 0.1;
    const dt = step;
    last = now;
    const t = now / 1000;
    const near = now < pointer.until;
    for (const f of fish) {
      f.turn += ((Math.sin(t * 0.55 + f.seed) + Math.sin(t * 0.23 + f.seed * 2)) * 0.9 - f.turn) * dt;
      let heading = f.heading + f.turn * dt;

      const margin = 90;
      const edge = Math.max(0, 1 - Math.min(f.x, w - f.x, f.y, h - f.y) / margin);
      if (edge > 0) heading = turnToward(heading, Math.atan2(h / 2 - f.y, w / 2 - f.x), edge * dt * 4);

      if (near) {
        const dx = f.x - pointer.x, dy = f.y - pointer.y;
        const d = Math.hypot(dx, dy);
        if (d < 170) {
          const fear = 1 - d / 170;
          heading = turnToward(heading, Math.atan2(dy, dx), fear * dt * 6);
          f.speed = Math.min(300, f.speed + 900 * fear * dt);
        }
      }

      f.speed += (f.cruise - f.speed) * dt * 1.4;
      f.heading = heading;
      f.x = Math.min(w + 40, Math.max(-40, f.x + Math.cos(heading) * f.speed * dt));
      f.y = Math.min(h + 40, Math.max(-40, f.y + Math.sin(heading) * f.speed * dt));
      draw(f, t, dt);
    }
    raf = running ? requestAnimationFrame(frame) : 0;
  };

  const place = () => {
    measure();
    for (const f of fish) {
      if (f.placed) continue;
      f.x = 60 + r() * Math.max(1, w - 120);
      f.y = 60 + r() * Math.max(1, h - 120);
      f.placed = true;
      draw(f, 0);
    }
  };
  new ResizeObserver(() => { measure(); if (!fish[0].placed) place(); }).observe(tank);
  place();

  return {
    play(on) {
      running = on && !reduceMotion;
      if (running && !raf) { last = performance.now(); step = 1 / 60; raf = requestAnimationFrame(frame); }
    },
  };
}
