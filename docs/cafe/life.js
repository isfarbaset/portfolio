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
      x: 0, y: 0, heading: r() * TAU, speed: 30, turn: 0,
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
        f.speed = 260 + r() * 120;
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

  const draw = (f, t) => {
    const wiggle = Math.sin(t * (7 + f.speed / 25) + f.seed) * (0.05 + f.speed / 2200);
    const angle = f.heading - f.art + wiggle;
    f.img.style.transform = `translate3d(${(f.x - f.size / 2).toFixed(1)}px, ${(f.y - f.size / 2).toFixed(1)}px, 0) rotate(${angle.toFixed(3)}rad)`;
  };

  let raf = 0, last = 0, running = false;
  const frame = (now) => {
    const dt = Math.min(0.05, (now - last) / 1000);
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
          heading = turnToward(heading, Math.atan2(dy, dx), fear * dt * 10);
          f.speed = Math.min(360, f.speed + 1400 * fear * dt);
        }
      }

      f.speed += (f.cruise - f.speed) * dt * 1.4;
      f.heading = heading;
      f.x = Math.min(w + 40, Math.max(-40, f.x + Math.cos(heading) * f.speed * dt));
      f.y = Math.min(h + 40, Math.max(-40, f.y + Math.sin(heading) * f.speed * dt));
      draw(f, t);
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
      if (running && !raf) { last = performance.now(); raf = requestAnimationFrame(frame); }
    },
  };
}
