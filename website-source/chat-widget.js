// ============================================
// POSTCARDS FROM ISFAR
// A tiny, no-AI corner of the site.
// Floating postage-stamp button → opens a paper postcard
// with rotating handwritten-style notes + a mailto "write back".
// ============================================

(function () {
  'use strict';

  // ---- Notes (lowercase on purpose — feels less corporate) ----
  const POSTCARDS = [
    "this week i'm romanticizing 4pm coffee with a thick novel & the way the kitchen light goes peach right before it disappears.",
    "currently obsessed with: oat milk cortados, the smell of new paperback books, & pretending i'll one day learn portuguese.",
    "a stranger held the elevator & asked if i'd seen the magnolias on 14th street. i hadn't. they were extraordinary. that was the whole tuesday.",
    "i think 'making a good chart' and 'making a good meal' use the same part of the brain — the part that wants to feed people.",
    "the bookstore on the corner has a striped cat who has decided the philosophy section is hers. i respect her completely.",
    "weekend plan: paint something bad on purpose, walk somewhere new, leave my phone face-down. that's the whole list.",
    "the chai near my apartment doesn't quite taste like dhaka but it tries, & i love it for trying."
  ];

  // ---- 1. Styles ----
  const css = `
  /* ---- Floating postage-stamp button ---- */
  #postcard-bubble {
    position: fixed;
    bottom: 28px;
    right: 28px;
    z-index: 99999;
    width: 64px;
    height: 64px;
    border: none;
    background: transparent;
    cursor: pointer;
    padding: 0;
    transition: transform 0.4s cubic-bezier(.34,1.56,.64,1);
    animation: postcardSway 6s ease-in-out infinite;
  }
  #postcard-bubble:hover {
    transform: rotate(-6deg) scale(1.08);
  }
  #postcard-bubble:focus { outline: none; }
  #postcard-bubble:focus-visible {
    outline: 2px dashed #a8b89a;
    outline-offset: 4px;
    border-radius: 8px;
  }
  #postcard-bubble svg {
    width: 100%; height: 100%;
    filter: drop-shadow(0 6px 14px rgba(58, 42, 32, 0.18));
  }
  @keyframes postcardSway {
    0%, 100% { transform: rotate(-2deg); }
    50%      { transform: rotate(2deg); }
  }
  #postcard-bubble.open {
    animation: none;
    transform: rotate(0deg) scale(0.9);
    opacity: 0.55;
  }

  /* Tooltip — only on hover, no auto-show pulse */
  #postcard-label {
    position: fixed;
    bottom: 100px;
    right: 28px;
    z-index: 99998;
    background: #fdf6e8;
    color: #3a2a20;
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-style: italic;
    font-size: 0.95rem;
    padding: 6px 14px;
    border: 1px solid rgba(168, 184, 154, 0.5);
    border-radius: 2px;
    white-space: nowrap;
    pointer-events: none;
    opacity: 0;
    transform: translateY(4px) rotate(-1deg);
    transition: opacity 0.3s ease, transform 0.3s ease;
    box-shadow: 0 4px 12px rgba(58, 42, 32, 0.08);
  }
  #postcard-bubble:hover + #postcard-label {
    opacity: 1;
    transform: translateY(0) rotate(-2deg);
  }
  #postcard-bubble.open + #postcard-label { opacity: 0 !important; }

  /* ---- Postcard panel ---- */
  #postcard-panel {
    position: fixed;
    bottom: 110px;
    right: 28px;
    z-index: 99998;
    width: 360px;
    max-width: calc(100vw - 40px);
    opacity: 0;
    transform: translateY(20px) rotate(-3deg) scale(0.94);
    transform-origin: bottom right;
    pointer-events: none;
    transition: opacity 0.45s cubic-bezier(.34,1.2,.64,1),
                transform 0.45s cubic-bezier(.34,1.2,.64,1);
  }
  #postcard-panel.open {
    opacity: 1;
    transform: translateY(0) rotate(-1.5deg) scale(1);
    pointer-events: auto;
  }

  /* Paper */
  .pc-paper {
    position: relative;
    background: #fdf6e8;
    background-image:
      radial-gradient(circle at 20% 30%, rgba(212, 181, 201, 0.06) 0%, transparent 50%),
      radial-gradient(circle at 80% 70%, rgba(168, 184, 154, 0.06) 0%, transparent 50%);
    color: #3a2a20;
    padding: 1.6rem 1.6rem 1.4rem;
    border-radius: 3px;
    box-shadow:
      0 1px 1px rgba(58, 42, 32, 0.04),
      0 8px 24px rgba(58, 42, 32, 0.15),
      0 18px 48px rgba(58, 42, 32, 0.12);
    font-family: 'Cormorant Garamond', Georgia, serif;
  }
  /* faint paper edge */
  .pc-paper::before {
    content: '';
    position: absolute;
    inset: 0;
    border: 1px solid rgba(168, 154, 130, 0.18);
    border-radius: 3px;
    pointer-events: none;
  }
  /* close button (small "×" in top right, hand-drawn feel) */
  .pc-close {
    position: absolute;
    top: 8px; right: 10px;
    background: transparent;
    border: none;
    width: 26px; height: 26px;
    cursor: pointer;
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.4rem;
    line-height: 1;
    color: #a8978a;
    opacity: 0.7;
    padding: 0;
    transition: opacity 0.2s ease, transform 0.2s ease;
  }
  .pc-close:hover { opacity: 1; transform: rotate(90deg); }
  .pc-close:focus { outline: none; }
  .pc-close:focus-visible { outline: 1px dotted #3a2a20; outline-offset: 2px; }

  /* Header row: from line + index */
  .pc-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.6rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(168, 154, 130, 0.25);
  }
  .pc-from {
    font-style: italic;
    font-size: 1.1rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    color: #3a2a20;
  }
  .pc-from .pc-mark {
    color: #a8b89a;
    margin-right: 0.25em;
  }
  .pc-index {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #a8978a;
    font-weight: 400;
  }

  /* Body text — the postcard message */
  .pc-body {
    min-height: 140px;
    display: flex;
    align-items: center;
    padding: 0.4rem 0 0.6rem;
  }
  .pc-body-text {
    font-style: italic;
    font-size: 1.08rem;
    font-weight: 400;
    line-height: 1.6;
    color: #3a2a20;
    transition: opacity 0.3s ease;
    margin: 0;
  }
  .pc-body-text.fading { opacity: 0; }

  /* Sign-off */
  .pc-signoff {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-style: italic;
    font-size: 0.95rem;
    color: #6a5a4a;
    text-align: right;
    margin: 0 0.2rem 0.8rem 0;
  }
  .pc-signoff::before {
    content: '── ';
    color: #a8978a;
  }

  /* Controls: prev / dots / next */
  .pc-controls {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.9rem;
    padding-top: 0.6rem;
    border-top: 1px dashed rgba(168, 154, 130, 0.3);
  }
  .pc-arrow {
    background: transparent;
    border: none;
    color: #a8978a;
    cursor: pointer;
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.3rem;
    line-height: 1;
    padding: 4px 8px;
    border-radius: 2px;
    transition: color 0.2s ease, transform 0.2s ease;
  }
  .pc-arrow:hover { color: #3a2a20; transform: translateY(-1px); }
  .pc-arrow:focus { outline: none; }
  .pc-arrow:focus-visible { outline: 1px dotted #3a2a20; outline-offset: 2px; }
  .pc-dots {
    display: flex;
    gap: 6px;
    align-items: center;
  }
  .pc-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: rgba(168, 154, 130, 0.35);
    transition: background 0.25s ease, transform 0.25s ease;
  }
  .pc-dot.active {
    background: #a8b89a;
    transform: scale(1.4);
  }

  /* Write back link */
  .pc-writeback {
    display: block;
    margin-top: 0.9rem;
    padding-top: 0.85rem;
    border-top: 1px solid rgba(168, 154, 130, 0.2);
    text-align: center;
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-style: italic;
    font-size: 0.95rem;
    color: #6a5a4a;
    text-decoration: none;
    letter-spacing: 0.01em;
    transition: color 0.25s ease, letter-spacing 0.3s ease;
  }
  .pc-writeback:hover {
    color: #3a2a20;
    letter-spacing: 0.04em;
  }
  .pc-writeback .pc-pen { color: #d4b5c9; margin-right: 0.2em; }

  /* Mobile */
  @media (max-width: 500px) {
    #postcard-panel {
      width: calc(100vw - 24px);
      right: 12px;
      bottom: 92px;
    }
    #postcard-bubble {
      bottom: 18px; right: 18px;
      width: 56px; height: 56px;
    }
    #postcard-label { bottom: 82px; right: 18px; font-size: 0.9rem; }
    .pc-paper { padding: 1.3rem 1.3rem 1.1rem; }
    .pc-from { font-size: 1rem; }
    .pc-body-text { font-size: 1rem; }
    .pc-body { min-height: 120px; }
  }
  `;

  const styleEl = document.createElement('style');
  styleEl.textContent = css;
  document.head.appendChild(styleEl);

  // ---- 2. HTML ----
  const wrap = document.createElement('div');
  wrap.id = 'postcard-root';
  wrap.innerHTML = `
    <button id="postcard-bubble" aria-label="Open a postcard from Isfar" aria-expanded="false">
      <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <!-- stamp body -->
        <rect x="3" y="3" width="58" height="58" rx="3" ry="3"
              fill="#fdf6e8"
              stroke="#a8b89a"
              stroke-width="1.4"
              stroke-dasharray="2 1.6"/>
        <!-- inner frame -->
        <rect x="9" y="9" width="46" height="46" rx="2" ry="2"
              fill="none"
              stroke="#a8b89a"
              stroke-width="0.8"
              opacity="0.55"/>
        <!-- tiny pressed flower in center -->
        <g transform="translate(32 32)" fill="#fdf6e8" stroke="#a8b89a" stroke-width="1.3">
          <ellipse cx="0" cy="-8" rx="3.6" ry="5"/>
          <ellipse cx="8" cy="0" rx="5" ry="3.6"/>
          <ellipse cx="0" cy="8" rx="3.6" ry="5"/>
          <ellipse cx="-8" cy="0" rx="5" ry="3.6"/>
          <circle cx="0" cy="0" r="2.6" fill="#d4b5c9" stroke="none"/>
        </g>
      </svg>
    </button>
    <div id="postcard-label">a postcard from isfar</div>

    <div id="postcard-panel" role="dialog" aria-label="A postcard from Isfar" aria-hidden="true">
      <div class="pc-paper">
        <button class="pc-close" aria-label="Close postcard">×</button>
        <div class="pc-header">
          <span class="pc-from"><span class="pc-mark">✿</span>a postcard from isfar</span>
          <span class="pc-index" id="pc-index">no. 1 / ${POSTCARDS.length}</span>
        </div>
        <div class="pc-body">
          <p class="pc-body-text" id="pc-body-text">${POSTCARDS[0]}</p>
        </div>
        <div class="pc-signoff">xo, isfar</div>
        <div class="pc-controls">
          <button class="pc-arrow pc-prev" aria-label="Previous postcard">‹ prev</button>
          <div class="pc-dots" id="pc-dots"></div>
          <button class="pc-arrow pc-next" aria-label="Next postcard">next ›</button>
        </div>
        <a class="pc-writeback"
           href="mailto:isfar.baset@gmail.com?subject=re%3A%20your%20postcard"
           aria-label="Write back to Isfar by email">
          <span class="pc-pen">✎</span>write back →
        </a>
      </div>
    </div>
  `;
  document.body.appendChild(wrap);

  // ---- 3. Wire up ----
  const bubble  = document.getElementById('postcard-bubble');
  const panel   = document.getElementById('postcard-panel');
  const closeBt = panel.querySelector('.pc-close');
  const prevBt  = panel.querySelector('.pc-prev');
  const nextBt  = panel.querySelector('.pc-next');
  const bodyEl  = document.getElementById('pc-body-text');
  const indexEl = document.getElementById('pc-index');
  const dotsEl  = document.getElementById('pc-dots');

  // Build dots
  POSTCARDS.forEach((_, i) => {
    const d = document.createElement('span');
    d.className = 'pc-dot' + (i === 0 ? ' active' : '');
    d.setAttribute('role', 'button');
    d.setAttribute('aria-label', 'Go to postcard ' + (i + 1));
    d.dataset.idx = String(i);
    dotsEl.appendChild(d);
  });
  const dotEls = dotsEl.querySelectorAll('.pc-dot');

  let current = 0;
  let isAnimating = false;

  function renderCard(idx) {
    if (isAnimating) return;
    isAnimating = true;
    bodyEl.classList.add('fading');
    setTimeout(() => {
      bodyEl.textContent = POSTCARDS[idx];
      indexEl.textContent = 'no. ' + (idx + 1) + ' / ' + POSTCARDS.length;
      dotEls.forEach((d, i) => d.classList.toggle('active', i === idx));
      bodyEl.classList.remove('fading');
      isAnimating = false;
    }, 250);
    current = idx;
  }

  function nudge(delta) {
    const next = (current + delta + POSTCARDS.length) % POSTCARDS.length;
    renderCard(next);
  }

  function togglePanel(force) {
    const open = typeof force === 'boolean'
      ? force
      : !panel.classList.contains('open');
    panel.classList.toggle('open', open);
    bubble.classList.toggle('open', open);
    bubble.setAttribute('aria-expanded', String(open));
    panel.setAttribute('aria-hidden', String(!open));
  }

  bubble.addEventListener('click', () => togglePanel());
  closeBt.addEventListener('click', () => togglePanel(false));
  prevBt.addEventListener('click', () => nudge(-1));
  nextBt.addEventListener('click', () => nudge(1));

  dotEls.forEach(d => {
    d.addEventListener('click', () => renderCard(parseInt(d.dataset.idx, 10)));
  });

  // Keyboard: Esc closes, arrows navigate when open
  document.addEventListener('keydown', (e) => {
    if (!panel.classList.contains('open')) return;
    if (e.key === 'Escape') togglePanel(false);
    if (e.key === 'ArrowLeft')  nudge(-1);
    if (e.key === 'ArrowRight') nudge(1);
  });

  // Start with a random card so each visit feels a little different
  const startIdx = Math.floor(Math.random() * POSTCARDS.length);
  if (startIdx !== 0) {
    current = startIdx;
    bodyEl.textContent = POSTCARDS[startIdx];
    indexEl.textContent = 'no. ' + (startIdx + 1) + ' / ' + POSTCARDS.length;
    dotEls.forEach((d, i) => d.classList.toggle('active', i === startIdx));
  }
})();
