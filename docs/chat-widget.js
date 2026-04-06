// ============================================
// FLOATING CHAT WIDGET - "Chat with Isfar"
// Self-contained: injects CSS, HTML, and wires up the chatbot AI
// Skip rendering on the dedicated chatbot page
// ============================================

(function () {
  'use strict';

  // Don't show the widget on the dedicated chatbot page
  if (window.location.pathname.includes('about-me-chatbot')) return;

  // ---- 1. Inject CSS ----
  const css = `
  /* ---- Floating bubble ---- */
  #chat-widget-bubble {
    position: fixed;
    bottom: 28px;
    right: 28px;
    z-index: 99999;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
    color: #fff;
    border: none;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(0,122,255,0.4), 0 2px 8px rgba(0,0,0,0.15);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.3s cubic-bezier(.16,1,.3,1), box-shadow 0.3s ease;
  }
  #chat-widget-bubble:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 28px rgba(0,122,255,0.5), 0 3px 12px rgba(0,0,0,0.18);
  }
  #chat-widget-bubble svg { pointer-events: none; }
  #chat-widget-bubble .bubble-close { display: none; }
  #chat-widget-bubble.open .bubble-open { display: none; }
  #chat-widget-bubble.open .bubble-close { display: block; }

  /* Pulse animation on first load */
  @keyframes widgetPulse {
    0%   { box-shadow: 0 4px 20px rgba(0,122,255,0.4), 0 0 0 0 rgba(0,122,255,0.5); }
    70%  { box-shadow: 0 4px 20px rgba(0,122,255,0.4), 0 0 0 14px rgba(0,122,255,0); }
    100% { box-shadow: 0 4px 20px rgba(0,122,255,0.4), 0 0 0 0 rgba(0,122,255,0); }
  }
  #chat-widget-bubble:not(.open) {
    animation: widgetPulse 2.5s ease-out 1s 3;
  }

  /* Tooltip label */
  #chat-widget-label {
    position: fixed;
    bottom: 96px;
    right: 28px;
    z-index: 99998;
    background: #1d1d1f;
    color: #fff;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 6px 14px;
    border-radius: 8px;
    white-space: nowrap;
    pointer-events: none;
    opacity: 0;
    transform: translateY(6px);
    transition: opacity 0.25s ease, transform 0.25s ease;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif;
  }
  #chat-widget-bubble:hover ~ #chat-widget-label,
  #chat-widget-label.show {
    opacity: 1;
    transform: translateY(0);
  }
  #chat-widget-bubble.open ~ #chat-widget-label { opacity: 0 !important; }

  /* ---- Chat panel ---- */
  #chat-widget-panel {
    position: fixed;
    bottom: 100px;
    right: 28px;
    z-index: 99998;
    width: 400px;
    max-height: 580px;
    border-radius: 20px;
    overflow: hidden;
    background: #fafafa;
    box-shadow: 0 12px 48px rgba(0,0,0,0.18), 0 2px 8px rgba(0,0,0,0.08);
    display: flex;
    flex-direction: column;
    opacity: 0;
    transform: translateY(20px) scale(0.95);
    pointer-events: none;
    transition: opacity 0.3s cubic-bezier(.16,1,.3,1),
                transform 0.3s cubic-bezier(.16,1,.3,1);
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif;
  }
  #chat-widget-panel.open {
    opacity: 1;
    transform: translateY(0) scale(1);
    pointer-events: auto;
  }

  /* Header */
  #chat-widget-panel .cw-header {
    background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
    color: #fff;
    padding: 1.1rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-shrink: 0;
  }
  #chat-widget-panel .cw-header-avatar {
    width: 38px; height: 38px; border-radius: 50%;
    background: rgba(255,255,255,0.2);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; flex-shrink: 0;
  }
  #chat-widget-panel .cw-header-text h4 {
    margin: 0; font-size: 1rem; font-weight: 600; letter-spacing: -0.01em;
  }
  #chat-widget-panel .cw-header-text p {
    margin: 2px 0 0 0; font-size: 0.78rem; opacity: 0.85; font-weight: 400;
  }

  /* Messages area */
  #cw-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1rem 1rem 0.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    background: #f5f5f7;
    min-height: 260px;
    max-height: 340px;
  }
  #cw-messages::-webkit-scrollbar { width: 4px; }
  #cw-messages::-webkit-scrollbar-track { background: transparent; }
  #cw-messages::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 2px; }

  .cw-msg { display: flex; animation: cwMsgIn 0.35s cubic-bezier(.16,1,.3,1); }
  .cw-msg.user { justify-content: flex-end; }
  .cw-msg.bot  { justify-content: flex-start; }

  .cw-msg-bubble {
    max-width: 82%;
    padding: 0.7rem 1rem;
    border-radius: 16px;
    font-size: 0.88rem;
    line-height: 1.55;
    font-weight: 400;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .cw-msg.user .cw-msg-bubble {
    background: #007AFF; color: #fff;
    border-bottom-right-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,122,255,0.2);
  }
  .cw-msg.bot .cw-msg-bubble {
    background: #fff; color: #1d1d1f;
    border-bottom-left-radius: 6px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.04);
  }

  @keyframes cwMsgIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* Typing dots */
  .cw-typing { display: flex; gap: 4px; align-items: center; padding: 4px 0; }
  .cw-typing span {
    width: 7px; height: 7px; border-radius: 50%; background: #9ca3af;
    animation: cwDot 1.4s infinite ease-in-out;
  }
  .cw-typing span:nth-child(1) { animation-delay: 0.2s; }
  .cw-typing span:nth-child(2) { animation-delay: 0.4s; }
  .cw-typing span:nth-child(3) { animation-delay: 0.6s; }
  @keyframes cwDot {
    0%,60%,100% { opacity: 0.3; transform: scale(0.8); }
    30%         { opacity: 1;   transform: scale(1);   }
  }

  /* Suggestion chips */
  #cw-suggestions {
    display: flex; gap: 0.5rem; flex-wrap: wrap;
    padding: 0.6rem 1rem; background: #fafafa;
  }
  .cw-chip {
    background: rgba(0,122,255,0.08); color: #007AFF;
    padding: 0.45rem 0.9rem; border-radius: 16px;
    font-size: 0.78rem; font-weight: 500;
    cursor: pointer; border: 1px solid rgba(0,122,255,0.12);
    transition: all 0.2s ease; white-space: nowrap;
  }
  .cw-chip:hover {
    background: #007AFF; color: #fff;
    box-shadow: 0 2px 10px rgba(0,122,255,0.25);
  }

  /* Input bar */
  #cw-input-bar {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.7rem 0.8rem;
    background: #fff;
    border-top: 1px solid rgba(0,0,0,0.06);
    flex-shrink: 0;
  }
  #cw-input {
    flex: 1;
    border: 1.5px solid rgba(0,0,0,0.08);
    border-radius: 22px;
    padding: 0.65rem 1rem;
    font-size: 0.88rem;
    font-family: inherit;
    outline: none;
    resize: none;
    min-height: 40px;
    max-height: 80px;
    line-height: 1.4;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }
  #cw-input:focus {
    border-color: rgba(0,122,255,0.35);
    box-shadow: 0 0 0 3px rgba(0,122,255,0.1);
  }
  #cw-input::placeholder { color: #86868b; }

  #cw-send {
    width: 38px; height: 38px; border-radius: 50%;
    background: #007AFF; color: #fff;
    border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    transition: background 0.2s ease, transform 0.2s ease;
    box-shadow: 0 2px 6px rgba(0,122,255,0.25);
  }
  #cw-send:hover { background: #0056CC; transform: scale(1.06); }
  #cw-send:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }

  /* Mobile */
  @media (max-width: 500px) {
    #chat-widget-panel {
      width: calc(100vw - 20px);
      right: 10px;
      bottom: 90px;
      max-height: 70vh;
    }
    #chat-widget-bubble { bottom: 18px; right: 18px; width: 54px; height: 54px; }
    #chat-widget-label  { bottom: 80px; right: 18px; }
  }
  `;

  const styleEl = document.createElement('style');
  styleEl.textContent = css;
  document.head.appendChild(styleEl);

  // ---- 2. Inject HTML ----
  const wrapper = document.createElement('div');
  wrapper.id = 'chat-widget-root';
  wrapper.innerHTML = `
    <button id="chat-widget-bubble" aria-label="Chat with Isfar">
      <svg class="bubble-open" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
      </svg>
      <svg class="bubble-close" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
      </svg>
    </button>
    <div id="chat-widget-label">Chat with Isfar</div>

    <div id="chat-widget-panel">
      <div class="cw-header">
        <div class="cw-header-avatar">💬</div>
        <div class="cw-header-text">
          <h4>Chat with Isfar</h4>
          <p>Ask about skills, projects, experience &amp; more</p>
        </div>
      </div>

      <div id="cw-messages">
        <div class="cw-msg bot">
          <div class="cw-msg-bubble">Hi! 👋 I'm an AI assistant for Isfar's portfolio. Ask me about her skills, projects, experience, or education!</div>
        </div>
      </div>

      <div id="cw-suggestions">
        <div class="cw-chip">Skills</div>
        <div class="cw-chip">Projects</div>
        <div class="cw-chip">Experience</div>
        <div class="cw-chip">Education</div>
      </div>

      <div id="cw-input-bar">
        <textarea id="cw-input" placeholder="Type a message…" rows="1"></textarea>
        <button id="cw-send" aria-label="Send">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m22 2-7 20-4-9-9-4z"/><path d="m22 2-11 10"/></svg>
        </button>
      </div>
    </div>
  `;
  document.body.appendChild(wrapper);

  // ---- 3. Load the chatbot AI class (chatbot.js) ----
  // Resolve path relative to the current page
  function resolveChatbotPath() {
    const scripts = document.querySelectorAll('script[src*="chat-widget"]');
    if (scripts.length) {
      const src = scripts[scripts.length - 1].src;
      return src.replace('chat-widget.js', 'chatbot.js');
    }
    // Fallback: assume same directory
    return 'chatbot.js';
  }

  function ensureChatbotAI(callback) {
    if (typeof IsfarChatbotAI !== 'undefined') {
      callback();
      return;
    }
    const s = document.createElement('script');
    s.src = resolveChatbotPath();
    s.onload = callback;
    s.onerror = function () {
      console.warn('⚠️ Could not load chatbot.js – widget will use a fallback.');
      callback();
    };
    document.head.appendChild(s);
  }

  // ---- 4. Wire everything up ----
  ensureChatbotAI(function () {
    var ai;
    try { ai = new IsfarChatbotAI(); } catch (e) { ai = null; }

    var isProcessing = false;
    var bubble = document.getElementById('chat-widget-bubble');
    var panel  = document.getElementById('chat-widget-panel');
    var msgBox = document.getElementById('cw-messages');
    var input  = document.getElementById('cw-input');
    var send   = document.getElementById('cw-send');
    var chips  = document.querySelectorAll('.cw-chip');
    var label  = document.getElementById('chat-widget-label');

    // Show tooltip briefly on load
    setTimeout(function () { label.classList.add('show'); }, 2000);
    setTimeout(function () { label.classList.remove('show'); }, 5500);

    // Toggle open/close
    bubble.addEventListener('click', function () {
      var isOpen = panel.classList.toggle('open');
      bubble.classList.toggle('open', isOpen);
      if (isOpen) input.focus();
    });

    // Close on Escape
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && panel.classList.contains('open')) {
        panel.classList.remove('open');
        bubble.classList.remove('open');
      }
    });

    // Add a message to the panel
    function addMsg(text, type) {
      var div = document.createElement('div');
      div.className = 'cw-msg ' + type;
      var bub = document.createElement('div');
      bub.className = 'cw-msg-bubble';
      bub.textContent = text;
      div.appendChild(bub);
      msgBox.appendChild(div);
      msgBox.scrollTop = msgBox.scrollHeight;
      return div;
    }

    function addTyping() {
      var div = document.createElement('div');
      div.className = 'cw-msg bot';
      div.innerHTML = '<div class="cw-msg-bubble"><div class="cw-typing"><span></span><span></span><span></span></div></div>';
      msgBox.appendChild(div);
      msgBox.scrollTop = msgBox.scrollHeight;
      return div;
    }

    function handleSend() {
      var text = input.value.trim();
      if (!text || isProcessing) return;
      isProcessing = true;

      addMsg(text, 'user');
      input.value = '';
      input.style.height = '40px';

      var typing = addTyping();

      var delay = 600 + text.length * 20 + Math.random() * 400;
      setTimeout(function () {
        typing.remove();
        var response;
        if (ai) {
          try { response = ai.generateResponse(text); } catch (e) { response = null; }
        }
        if (!response) {
          response = "Thanks for your question! For the best experience, visit the full Chat with Isfar page. You can ask me about skills, projects, experience, or education.";
        }
        addMsg(response, 'bot');
        isProcessing = false;
      }, delay);
    }

    send.addEventListener('click', handleSend);

    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });

    input.addEventListener('input', function () {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 80) + 'px';
    });

    chips.forEach(function (chip) {
      chip.addEventListener('click', function () {
        if (isProcessing) return;
        input.value = this.textContent.trim();
        handleSend();
      });
    });
  });
})();
