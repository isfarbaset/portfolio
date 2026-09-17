/* =========================================================
   Isfar's · one open page
   ========================================================= */
import { createLatte } from './latte.js';
import { drawCup, sketchPage } from './sketch.js';
import { goldfish } from './life.js';

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const finePointer = matchMedia('(hover: hover) and (pointer: fine)').matches;
const ET = 'America/New_York';

$$('[data-year]').forEach((el) => { el.textContent = new Date().getFullYear(); });

/* ---------- ink ---------- */
drawCup($('#cupArt'), $('#cupRim'));
$$('.plot .dot').forEach((d, i) => d.style.setProperty('--i', i));
sketchPage();

/* ---------- things appear as they come into view ---------- */
{
  const io = new IntersectionObserver((entries) => entries.forEach((e) => {
    if (!e.isIntersecting) return;
    e.target.classList.add('in');
    io.unobserve(e.target);
  }), { rootMargin: '0px 0px -12% 0px' });
  $$('[data-reveal], .roles, .plot, mark.marker').forEach((el) => io.observe(el));
}

/* ---------- the stamp in the corner: now, in Northern Virginia ---------- */
{
  const dateFmt = new Intl.DateTimeFormat('en-US', { timeZone: ET, weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' });
  const timeFmt = new Intl.DateTimeFormat('en-US', { timeZone: ET, hour: 'numeric', minute: '2-digit', second: '2-digit' });
  const tick = () => {
    const now = new Date();
    $('#stampDate').textContent = dateFmt.format(now);
    $('#stampTime').textContent = `${timeFmt.format(now)} ET`;
  };
  tick();
  setInterval(tick, 1000);

  $('#rDate').textContent = new Intl.DateTimeFormat('en-US', { timeZone: ET, month: '2-digit', day: '2-digit', year: '2-digit' }).format(new Date());

  const status = $('#status');
  const update = () => {
    const d = new Date();
    const time = new Intl.DateTimeFormat('en-US', { timeZone: ET, hour: 'numeric', minute: '2-digit' }).format(d)
      .toLowerCase().replace('am', 'a.m.').replace('pm', 'p.m.');
    const hour = Number(new Intl.DateTimeFormat('en-US', { timeZone: ET, hour: 'numeric', hourCycle: 'h23' }).format(d));
    status.textContent = hour >= 7 && hour < 23
      ? `It's ${time} in Northern Virginia. The kitchen's open, and I usually reply within a day.`
      : `It's ${time} in Northern Virginia. I'm asleep, or pretending to be. Leave a note and I'll get to it after coffee.`;
  };
  update();
  setInterval(update, 30_000);
}

/* ---------- the sticker that matches where you are ---------- */
{
  const links = $$('.sticker');
  const io = new IntersectionObserver((entries) => entries.forEach((e) => {
    if (!e.isIntersecting) return;
    links.forEach((a) => (a.hash === `#${e.target.id}` ? a.setAttribute('aria-current', 'true') : a.removeAttribute('aria-current')));
  }), { rootMargin: '-45% 0px -50% 0px' });
  ['#work', '#about', '#art', '#hi'].forEach((id) => io.observe($(id)));
}

/* ---------- the latte: Isfar's photo, poured into a fluid sim ---------- */
{
  const stage = $('#cup');
  const latte = createLatte($('#latte'), {
    reducedMotion: reduceMotion,
    photo: 'images/latte-photo.webp',
    /* amount is how far the spoon has travelled, in cup widths */
    onStir(amount) {
      if (amount > 0.05) stage.classList.add('stirring');
      stage.classList.toggle('spent', amount > 3.4);
    },
  });

  const again = () => {
    latte?.pour();
    stage.classList.remove('pour');
    void stage.offsetWidth;
    stage.classList.add('pour');
  };
  $('#repour').addEventListener('click', again);
  $('#refill').addEventListener('click', again);

  if (!latte) {
    $('#repour').hidden = true;
    $('#refill').hidden = true;
    $('#stirHint').hidden = true;
  } else {
    // a short swirl once it has been on screen a moment, so the surface is
    // visibly liquid. Late enough that the heart in the photo lands first.
    new IntersectionObserver(([e], io) => {
      if (!e.isIntersecting) return;
      io.disconnect();
      setTimeout(() => latte.demo(), 2400);
    }, { threshold: 0.6 }).observe($('#latte'));
  }
}

/* ---------- the menu: a print follows the cursor ---------- */
if (finePointer) {
  const peek = $('#peek');
  const box = $('#peekImgs');
  const imgs = new Map();
  const pos = { x: 0, y: 0, tx: 0, ty: 0, rot: 0, on: false, running: false };
  $$('.item').forEach((row) => {
    const src = row.dataset.img;
    if (!imgs.has(src)) {
      const img = new Image();
      img.src = src;
      img.alt = '';
      box.append(img);
      imgs.set(src, img);
    }
    row.addEventListener('pointerenter', (e) => {
      if (!pos.on) { pos.x = pos.tx = e.clientX; pos.y = pos.ty = e.clientY; }
      imgs.forEach((img, key) => img.classList.toggle('on', key === src));
      pos.on = true;
      peek.classList.add('on');
      start();
    });
    row.addEventListener('pointerleave', () => { pos.on = false; peek.classList.remove('on'); });
  });
  addEventListener('pointermove', (e) => { pos.tx = e.clientX; pos.ty = e.clientY; if (pos.on) start(); }, { passive: true });

  const tick = () => {
    pos.x += (pos.tx - pos.x) * 0.14;
    pos.y += (pos.ty - pos.y) * 0.14;
    const vx = pos.tx - pos.x;
    pos.rot += (clamp(vx * 0.05 - 3, -9, 5) - pos.rot) * 0.12;
    const x = Math.min(pos.x + 40, innerWidth - 320);
    const y = clamp(pos.y - 120, 70, innerHeight - 280);
    peek.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) rotate(${pos.rot.toFixed(2)}deg)`;
    if (pos.on || Math.abs(vx) > 0.4) requestAnimationFrame(tick);
    else pos.running = false;
  };
  const start = () => { if (!pos.running) { pos.running = true; requestAnimationFrame(tick); } };
}

/* ---------- the scatter plot: drag with a mouse, tap to look closer ---------- */
const openDrawing = (() => {
  const dlg = $('#lightbox');
  const img = $('#lbImg');
  const items = $$('[data-full]');
  let index = 0;
  const show = (i) => {
    index = (i + items.length) % items.length;
    const b = items[index];
    img.src = b.dataset.full;
    img.alt = b.querySelector('img').alt;
    $('#lbTitle').textContent = b.dataset.title;
    $('#lbMeta').textContent = `oil pastel on paper · ${index + 1} of ${items.length}`;
  };
  dlg.addEventListener('click', (e) => {
    const action = e.target.closest('[data-lb]')?.dataset.lb;
    if (action === 'close' || e.target === dlg || e.target.classList.contains('lb-stage')) dlg.close();
    else if (action === 'prev') show(index - 1);
    else if (action === 'next') show(index + 1);
  });
  dlg.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') show(index + 1);
    if (e.key === 'ArrowLeft') show(index - 1);
  });
  return (el) => { show(items.indexOf(el)); dlg.showModal(); };
})();

{
  const plot = $('#plot');
  $$('.dot', plot).forEach((dot) => {
    let drag = null;
    dot.addEventListener('pointerdown', (e) => {
      if (e.pointerType !== 'mouse' || e.button !== 0) return;
      const rect = plot.getBoundingClientRect();
      drag = {
        id: e.pointerId, sx: e.clientX, sy: e.clientY, moved: false, rect,
        x0: (parseFloat(getComputedStyle(dot).getPropertyValue('--x')) / 100) * rect.width,
        y0: (parseFloat(getComputedStyle(dot).getPropertyValue('--y')) / 100) * rect.height,
      };
      dot.setPointerCapture(e.pointerId);
    });
    dot.addEventListener('pointermove', (e) => {
      if (!drag || e.pointerId !== drag.id) return;
      const dx = e.clientX - drag.sx, dy = e.clientY - drag.sy;
      if (!drag.moved && Math.hypot(dx, dy) < 6) return;
      drag.moved = true;
      dot.classList.add('dragging');
      dot.style.setProperty('--x', `${clamp(((drag.x0 + dx) / drag.rect.width) * 100, 2, 98)}%`);
      dot.style.setProperty('--y', `${clamp(((drag.y0 + dy) / drag.rect.height) * 100, 2, 98)}%`);
    });
    const end = (e) => {
      if (!drag || e.pointerId !== drag.id) return;
      dot.classList.remove('dragging');
      dot.dataset.justDragged = drag.moved ? '1' : '';
      drag = null;
    };
    dot.addEventListener('pointerup', end);
    dot.addEventListener('pointercancel', end);
    dot.addEventListener('click', () => {
      if (dot.dataset.justDragged) { dot.dataset.justDragged = ''; return; }
      openDrawing(dot);
    });
  });
}

/* ---------- goldfish, only while you can see them ---------- */
{
  const fish = goldfish($('#tank'), $('#hi'));
  new IntersectionObserver(([e]) => fish.play(e.isIntersecting)).observe($('#tank'));
}

/* ---------- the note ---------- */
{
  const form = $('#orderForm');
  const link = $('#placeOrder');
  const LABELS = { analysis: 'data analysis', ml: 'machine learning', ai: 'an AI assistant', dash: 'a dashboard', story: 'data storytelling', chat: 'a coffee chat' };
  const list = (w) => (w.length < 2 ? w.join('') : `${w.slice(0, -1).join(', ')} & ${w.at(-1)}`);
  const update = () => {
    const picked = $$('input[name="want"]:checked', form).map((i) => LABELS[i.value]);
    const name = $('#cupName').value.trim();
    const subject = picked.length ? `An order for Isfar: ${list(picked)}` : 'Hi Isfar';
    const body = ['Hi Isfar,', '', picked.length ? `I'm after ${list(picked)}.` : '', name ? `This is ${name}.` : '', '', '']
      .filter((line, i, arr) => line !== '' || arr[i - 1] !== '').join('\n');
    link.href = `mailto:isfar.baset@gmail.com?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
  };
  form.addEventListener('input', update);
  form.addEventListener('change', update);
  update();
}
