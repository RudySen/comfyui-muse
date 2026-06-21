// muse_chat_ui.js — DOM building, rendering, and event wiring for the Muse chat panel.
// Pure vanilla JS, no build step. Exposes window.MuseChatUI.create(container) which
// returns a controller object the LiteGraph widget mounts into the node body.

import api from "./muse_chat_api.js";

const DEFAULT_URLS = {
  lmstudio: "http://localhost:1234",
  ollama: "http://localhost:11434",
};

// Accent presets for the color picker. Each re-themes the whole panel by driving
// --muse-accent-rgb (used by every glow/dim/bubble) plus the two solid hexes.
const ACCENTS = [
  { name: "Amber", accent: "#ff9d3d", bright: "#ffb85c", rgb: "255, 157, 61" },
  { name: "Gold", accent: "#ffc14d", bright: "#ffd47a", rgb: "255, 193, 77" },
  { name: "Coral", accent: "#ff6f5e", bright: "#ff9182", rgb: "255, 111, 94" },
  { name: "Rose", accent: "#ff5ca8", bright: "#ff86bf", rgb: "255, 92, 168" },
  { name: "Violet", accent: "#a882ff", bright: "#c2a6ff", rgb: "168, 130, 255" },
  { name: "Blue", accent: "#5c94ff", bright: "#86b0ff", rgb: "92, 148, 255" },
  { name: "Cyan", accent: "#38d1ff", bright: "#7ce0ff", rgb: "56, 209, 255" },
  { name: "Mint", accent: "#3ddc84", bright: "#74e7a8", rgb: "61, 220, 132" },
];
const ACCENT_STORAGE_KEY = "museChatAccent";
const VRAM_STORAGE_KEY = "museChatFreeVram";

function loadVramPref() {
  try {
    return localStorage.getItem(VRAM_STORAGE_KEY) !== "0";
  } catch (e) {
    return true;
  }
}

// --- small DOM helpers -------------------------------------------------------

function el(tag, opts = {}, children = []) {
  const node = document.createElement(tag);
  if (opts.class) node.className = opts.class;
  if (opts.text != null) node.textContent = opts.text;
  if (opts.html != null) node.innerHTML = opts.html;
  if (opts.title) node.title = opts.title;
  if (opts.type) node.type = opts.type;
  if (opts.placeholder) node.placeholder = opts.placeholder;
  if (opts.value != null) node.value = opts.value;
  if (opts.attrs) for (const [k, v] of Object.entries(opts.attrs)) node.setAttribute(k, v);
  if (opts.on) for (const [ev, fn] of Object.entries(opts.on)) node.addEventListener(ev, fn);
  for (const c of children) if (c) node.appendChild(c);
  return node;
}

function relativeTime(iso) {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (isNaN(then)) return "";
  const s = Math.floor((Date.now() - then) / 1000);
  if (s < 60) return "just now";
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d < 7) return `${d}d ago`;
  return new Date(iso).toLocaleDateString();
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Minimal, safe markdown: code blocks, inline code, bold, italic, line breaks.
// Code blocks are pulled out behind private-use sentinels (.. —
// chars that can't appear in normal text) so the later inline rules and the
// digit-index restore can't accidentally match real content.
const MD_OPEN = "";
const MD_CLOSE = "";
function renderMarkdown(text) {
  const blocks = [];
  let src = String(text).replace(/```([\s\S]*?)```/g, (_, code) => {
    blocks.push(code.replace(/^\n/, ""));
    return MD_OPEN + (blocks.length - 1) + MD_CLOSE;
  });
  src = escapeHtml(src);
  src = src.replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`);
  src = src.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  src = src.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  src = src.replace(/\n/g, "<br>");
  src = src.replace(new RegExp(MD_OPEN + "(\\d+)" + MD_CLOSE, "g"), (_, i) =>
    `<pre class="muse-code"><code>${escapeHtml(blocks[+i])}</code></pre>`
  );
  return src;
}

// Split a content string into {thinking, answer} by extracting <think>...</think>.
function splitThinking(content) {
  let thinking = "";
  const answer = content.replace(/<think>([\s\S]*?)<\/think>/gi, (_, t) => {
    thinking += t;
    return "";
  });
  // Handle an unclosed <think> still streaming.
  const open = answer.indexOf("<think>");
  if (open !== -1) {
    thinking += answer.slice(open + 7);
    return { thinking, answer: answer.slice(0, open) };
  }
  return { thinking, answer };
}

// --- main controller ---------------------------------------------------------

function create(root) {
  root.classList.add("muse-panel");

  const state = {
    backend: "lmstudio",
    baseUrl: DEFAULT_URLS.lmstudio,
    model: null,
    models: [],
    modelInstanceId: null,
    statusState: "offline", // offline | not-loaded | loaded | error
    chat: null, // full active chat object
    chats: [], // lightweight list
    streaming: false,
    abortCtl: null,
    contextLength: null,
    sidebarCollapsed: false,
    freeComfyVram: loadVramPref(),
    availableGuides: [], // filenames present in ComfyUI/input/
    availableImages: [], // image filenames present in ComfyUI/input/
    pendingImages: [], // images attached to the next message (filenames)
  };

  // ---- structure ----
  const statusDot = el("span", { class: "muse-status-dot", title: "Disconnected" });

  const backendSelect = el("select", { class: "muse-select" }, [
    el("option", { value: "lmstudio", text: "LM Studio" }),
    el("option", { value: "ollama", text: "Ollama" }),
  ]);
  const baseUrlInput = el("input", {
    class: "muse-input muse-baseurl",
    value: state.baseUrl,
    placeholder: "base url",
    title: "Backend base URL",
  });
  const modelSelect = el("select", { class: "muse-select muse-model" });
  const refreshBtn = el("button", { class: "muse-icon-btn", title: "Refresh models", html: "&#x21bb;" });
  // Loading is automatic (the model JIT-loads on the first message), so there's
  // no Load button — only Unload, to free VRAM. Enabled only when loaded.
  const unloadBtn = el("button", {
    class: "muse-btn muse-load-btn",
    text: "Unload",
    title: "Unload the model to free VRAM",
  });

  // accent color picker swatches
  const swatchEls = [];
  const accentRow = el("div", { class: "muse-accent-row", title: "Accent color" });
  for (const preset of ACCENTS) {
    const sw = el("button", {
      class: "muse-swatch",
      title: preset.name,
      attrs: { "data-name": preset.name },
      on: { click: () => applyAccent(preset) },
    });
    sw.style.background = preset.accent;
    sw.style.color = preset.accent; // for the active glow (currentColor)
    swatchEls.push(sw);
    accentRow.appendChild(sw);
  }

  // "Free ComfyUI VRAM" toggle — unloads ComfyUI's models + clears cache before
  // each message so the LLM loads into freed VRAM.
  const vramToggle = el("input", { class: "muse-vram-toggle", type: "checkbox" });
  vramToggle.checked = state.freeComfyVram;
  const vramLabel = el("label", {
    class: "muse-vram-label",
    title: "Unload ComfyUI models and clear its cache before each message, so the LLM has free VRAM to load into.",
  }, [vramToggle, el("span", { text: "Free ComfyUI VRAM" })]);

  // "Unload on Run" toggle — frees the LLM's VRAM (confirmed) before ComfyUI's
  // queued generation runs. Global preference, read by the queue gate.
  const runToggle = el("input", { class: "muse-vram-toggle", type: "checkbox" });
  let runPref = true;
  try {
    runPref = localStorage.getItem("museChatUnloadOnRun") !== "0";
  } catch (e) {
    /* default on */
  }
  runToggle.checked = runPref;
  const runLabel = el("label", {
    class: "muse-vram-label",
    title: "Unload the chat model (and confirm VRAM is freed) before a ComfyUI Run, so the render isn't starved of memory.",
  }, [runToggle, el("span", { text: "Unload on Run" })]);

  const optionsRow = el("div", { class: "muse-topbar-row muse-options-row" }, [
    accentRow,
    el("div", { class: "muse-options-toggles" }, [vramLabel, runLabel]),
  ]);

  const topBar = el("div", { class: "muse-topbar" }, [
    el("div", { class: "muse-topbar-row" }, [
      statusDot,
      backendSelect,
      el("div", { class: "muse-model-wrap" }, [modelSelect, refreshBtn]),
      unloadBtn,
    ]),
    el("div", { class: "muse-topbar-row muse-topbar-row-2" }, [baseUrlInput]),
    optionsRow,
  ]);

  // sidebar
  const newChatBtn = el("button", { class: "muse-btn muse-newchat-btn", html: "+ New Chat" });
  const chatList = el("div", { class: "muse-chatlist" });
  const sidebar = el("div", { class: "muse-sidebar" }, [
    el("div", { class: "muse-sidebar-head" }, [newChatBtn]),
    chatList,
  ]);

  const sidebarToggle = el("button", {
    class: "muse-icon-btn muse-sidebar-toggle",
    title: "Toggle chat list",
    html: "&#9776;",
  });

  // system prompt
  const sysToggle = el("button", {
    class: "muse-sys-toggle",
    html: "<span class='muse-chevron'>&#9656;</span> System prompt",
    title: "Edit system prompt for this chat",
  });
  const sysTextarea = el("textarea", {
    class: "muse-sys-textarea",
    placeholder: "Optional system prompt for this chat…",
  });
  // Max reply tokens (per chat). Default 2048 — raised from backend defaults so
  // multi-output requests ("write 5 prompts") aren't truncated mid-list.
  const maxTokensInput = el("input", {
    class: "muse-input muse-maxtokens", type: "number", value: "2048",
    title: "Maximum tokens in each reply. Raise for long / multi-item outputs.",
    attrs: { min: "64", step: "256" },
  });
  const maxTokensRow = el("div", { class: "muse-setting-row" }, [
    el("label", { class: "muse-setting-label", text: "Max reply tokens" }),
    maxTokensInput,
  ]);
  const sysDrawer = el("div", { class: "muse-sys-drawer muse-collapsed" }, [sysTextarea, maxTokensRow]);

  // Guide Materials — standing per-chat reference files from ComfyUI/input/.
  const guidesToggle = el("button", {
    class: "muse-sys-toggle",
    html: "<span class='muse-chevron'>&#9656;</span> Guide materials",
    title: "Reference files (style guides, conventions) that influence every message in this chat",
  });
  const guidesRefresh = el("button", { class: "muse-icon-btn muse-guides-refresh", title: "Rescan input/ folder", html: "&#x21bb;" });
  const guidesList = el("div", { class: "muse-guides-list" });
  const guidesDrawer = el("div", { class: "muse-sys-drawer muse-guides-drawer muse-collapsed" }, [
    el("div", { class: "muse-guides-head" }, [
      el("span", { class: "muse-guides-hint", text: "Files in ComfyUI/input/ (.txt .md .json)" }),
      guidesRefresh,
    ]),
    guidesList,
  ]);

  // messages
  const messagesEl = el("div", { class: "muse-messages" });

  // input
  const inputArea = el("textarea", {
    class: "muse-input-textarea",
    placeholder: "Message your model…  (Enter to send, Shift+Enter for newline)\nTip: for multiple outputs, ask for “a numbered list of exactly 5 …”.",
  });
  const attachBtn = el("button", { class: "muse-btn muse-attach-btn", title: "Attach image from ComfyUI/input/", html: "&#43;" });
  const sendBtn = el("button", { class: "muse-btn muse-send-btn", title: "Send", html: "&#10148;" });
  const stopBtn = el("button", { class: "muse-btn muse-stop-btn muse-hidden", title: "Stop", html: "&#9632;" });
  const tokenInfo = el("div", { class: "muse-token-info", text: "" });
  const pendingBar = el("div", { class: "muse-pending-images muse-hidden" });
  const imagePicker = el("div", { class: "muse-image-picker muse-hidden" });
  const inputBar = el("div", { class: "muse-inputbar" }, [
    pendingBar,
    imagePicker,
    el("div", { class: "muse-input-row" }, [attachBtn, inputArea, sendBtn, stopBtn]),
    tokenInfo,
  ]);

  const mainCol = el("div", { class: "muse-main" }, [
    el("div", { class: "muse-sys-area" }, [sysToggle, sysDrawer, guidesToggle, guidesDrawer]),
    messagesEl,
    inputBar,
  ]);

  const body = el("div", { class: "muse-body" }, [sidebar, mainCol]);

  // Draggable title strip. The DOM widget covers the whole node and eats the
  // canvas title bar's drag events, so this strip is what the user grabs to move
  // the node (wired up in muse_chat.js, which has the node + canvas handles).
  const dragbar = el("div", { class: "muse-dragbar" }, [
    el("span", { class: "muse-grip", html: "&#x2059;&#x2059;" }),
    el("span", { class: "muse-dragbar-label", text: "Muse Chat" }),
  ]);

  // header row holds the sidebar toggle + topbar
  const header = el("div", { class: "muse-header" }, [sidebarToggle, topBar]);

  root.appendChild(dragbar);
  root.appendChild(header);
  root.appendChild(body);

  // ---- status helpers ----
  function setStatus(s, label) {
    state.statusState = s;
    statusDot.className = `muse-status-dot muse-status-${s}`;
    statusDot.title = label || s;
    updateLoadBtn();
  }

  function updateLoadBtn() {
    const loaded = state.statusState === "loaded";
    unloadBtn.textContent = "Unload";
    unloadBtn.classList.toggle("muse-loaded", loaded);
    // Only actionable when a model is actually loaded.
    unloadBtn.disabled = !loaded;
  }

  function systemMessage(text, kind = "info") {
    const m = el("div", { class: `muse-sysmsg muse-sysmsg-${kind}`, text });
    // Click to dismiss immediately…
    m.addEventListener("click", () => m.remove());
    messagesEl.appendChild(m);
    autoScroll();
    // …and auto-dismiss so errors don't linger in the chat forever.
    const ttl = kind === "error" ? 9000 : 5000;
    setTimeout(() => {
      m.classList.add("muse-fadeout");
      setTimeout(() => m.remove(), 400);
    }, ttl);
    return m;
  }

  // ---- accent color picker ----
  function applyAccent(preset) {
    root.style.setProperty("--muse-accent", preset.accent);
    root.style.setProperty("--muse-accent-bright", preset.bright);
    root.style.setProperty("--muse-accent-rgb", preset.rgb);
    for (const sw of swatchEls) {
      sw.classList.toggle("muse-swatch-active", sw.dataset.name === preset.name);
    }
    try {
      localStorage.setItem(ACCENT_STORAGE_KEY, preset.name);
    } catch (e) {
      /* storage may be unavailable */
    }
  }

  function initAccent() {
    let name = null;
    try {
      name = localStorage.getItem(ACCENT_STORAGE_KEY);
    } catch (e) {
      /* ignore */
    }
    const preset = ACCENTS.find((p) => p.name === name) || ACCENTS[0];
    applyAccent(preset);
  }

  // ---- Guide Materials ----
  async function refreshGuides() {
    try {
      state.availableGuides = await api.listGuides();
    } catch (e) {
      state.availableGuides = [];
    }
    renderGuidesList();
  }

  function activeGuides() {
    return (state.chat && state.chat.guides) || [];
  }

  function renderGuidesList() {
    guidesList.innerHTML = "";
    const active = activeGuides();
    const all = new Set(state.availableGuides.map((g) => g.name));
    // Active-but-missing guides still show, flagged, so the user notices.
    const names = [...new Set([...state.availableGuides.map((g) => g.name), ...active])].sort();
    if (!names.length) {
      guidesList.appendChild(el("div", { class: "muse-guides-empty", text: "No guide files in input/" }));
      updateGuidesToggleCount();
      return;
    }
    for (const name of names) {
      const missing = !all.has(name);
      const cb = el("input", { class: "muse-guide-cb", type: "checkbox" });
      cb.checked = active.includes(name);
      cb.addEventListener("change", () => toggleGuide(name, cb.checked));
      const label = el("label", { class: `muse-guide-row${missing ? " muse-guide-missing" : ""}` }, [
        cb,
        el("span", { class: "muse-guide-name", text: name }),
        missing ? el("span", { class: "muse-guide-warn", text: "⚠ not found", title: "File missing from input/" }) : null,
      ]);
      guidesList.appendChild(label);
    }
    updateGuidesToggleCount();
  }

  function updateGuidesToggleCount() {
    const n = activeGuides().length;
    const chevron = "<span class='muse-chevron'>&#9656;</span>";
    guidesToggle.innerHTML = `${chevron} Guide materials${n ? ` <span class='muse-guides-count'>${n}</span>` : ""}`;
  }

  function toggleGuide(name, on) {
    if (!state.chat) return;
    if (!Array.isArray(state.chat.guides)) state.chat.guides = [];
    const i = state.chat.guides.indexOf(name);
    if (on && i === -1) state.chat.guides.push(name);
    else if (!on && i !== -1) state.chat.guides.splice(i, 1);
    persistChat();
    updateGuidesToggleCount();
    updateTokenInfo();
  }

  // ---- image attachments ----
  async function refreshInputImages() {
    try {
      state.availableImages = await api.listInputImages();
    } catch (e) {
      state.availableImages = [];
    }
  }

  function modelSupportsVision() {
    const sel = state.models.find((m) => m.id === state.model);
    // Only treat as unsupported when we explicitly know it's not vision.
    return !(sel && sel.vision === false);
  }

  function renderPendingImages() {
    pendingBar.innerHTML = "";
    if (!state.pendingImages.length) {
      pendingBar.classList.add("muse-hidden");
      return;
    }
    pendingBar.classList.remove("muse-hidden");
    for (const name of state.pendingImages) {
      const thumb = el("div", { class: "muse-pending-thumb" });
      const img = el("img", { attrs: { src: api.inputFileUrl(name), alt: name } });
      const rm = el("button", {
        class: "muse-pending-rm", title: "Remove", html: "&#10005;",
        on: { click: () => { state.pendingImages = state.pendingImages.filter((n) => n !== name); renderPendingImages(); } },
      });
      thumb.appendChild(img);
      thumb.appendChild(rm);
      pendingBar.appendChild(thumb);
    }
  }

  function addPendingImage(name) {
    if (!state.pendingImages.includes(name)) state.pendingImages.push(name);
    renderPendingImages();
  }

  async function toggleImagePicker() {
    if (!imagePicker.classList.contains("muse-hidden")) {
      imagePicker.classList.add("muse-hidden");
      return;
    }
    await refreshInputImages();
    imagePicker.innerHTML = "";
    const head = el("div", { class: "muse-picker-head", text: "Images in ComfyUI/input/ — click to attach, or drag files onto the panel" });
    imagePicker.appendChild(head);
    const grid = el("div", { class: "muse-picker-grid" });
    if (!state.availableImages.length) {
      grid.appendChild(el("div", { class: "muse-guides-empty", text: "No images in input/" }));
    }
    for (const name of state.availableImages) {
      const cell = el("div", { class: "muse-picker-cell", title: name });
      cell.appendChild(el("img", { attrs: { src: api.inputFileUrl(name), alt: name } }));
      cell.addEventListener("click", () => {
        addPendingImage(name);
        imagePicker.classList.add("muse-hidden");
      });
      grid.appendChild(cell);
    }
    imagePicker.appendChild(grid);
    imagePicker.classList.remove("muse-hidden");
  }

  async function handleDroppedFiles(files) {
    const images = [...files].filter((f) => /^image\//.test(f.type) || /\.(png|jpe?g|webp|gif|bmp)$/i.test(f.name));
    if (!images.length) return;
    for (const file of images) {
      try {
        const name = await api.saveInputImage(file);
        addPendingImage(name);
      } catch (e) {
        systemMessage(`Could not save ${file.name}: ${e.message}`, "error");
      }
    }
    refreshInputImages();
  }

  function estimateGuideTokens() {
    // Rough heuristic: bytes / 4, consistent with the no-local-tokenizer policy.
    const active = activeGuides();
    let bytes = 0;
    for (const g of state.availableGuides) {
      if (active.includes(g.name)) bytes += g.bytes || 0;
    }
    return Math.round(bytes / 4);
  }

  // ---- scroll handling: don't yank if user scrolled up ----
  let pinnedToBottom = true;
  messagesEl.addEventListener("scroll", () => {
    const gap = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight;
    pinnedToBottom = gap < 40;
  });
  function autoScroll(force) {
    if (force || pinnedToBottom) messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // ---- message rendering ----
  function makeToolbar(msg) {
    const copyBtn = el("button", {
      class: "muse-msg-act",
      title: "Copy",
      html: "&#128203;",
      on: {
        click: async () => {
          try {
            await navigator.clipboard.writeText(msg.content || "");
            copyBtn.innerHTML = "&#10003;";
            setTimeout(() => (copyBtn.innerHTML = "&#128203;"), 1200);
          } catch (e) {
            /* clipboard may be blocked */
          }
        },
      },
    });
    const buttons = [copyBtn];

    // Regenerate is only meaningful for assistant replies.
    if (msg.role === "assistant") {
      buttons.push(
        el("button", {
          class: "muse-msg-act",
          title: "Regenerate",
          html: "&#x21bb;",
          on: { click: () => regenerateMessage(msg) },
        })
      );
    }

    buttons.push(
      el("button", {
        class: "muse-msg-act muse-msg-del",
        title: "Delete",
        html: "&#128465;",
        on: { click: () => deleteMessage(msg) },
      })
    );

    return el("div", { class: "muse-msg-toolbar" }, buttons);
  }

  function renderMessage(msg) {
    const isUser = msg.role === "user";
    const wrap = el("div", { class: `muse-msg muse-msg-${isUser ? "user" : "assistant"}` });

    const bubble = el("div", { class: "muse-bubble" });

    const thinkingEl = el("div", { class: "muse-thinking muse-collapsed" });
    const thinkingToggle = el("button", {
      class: "muse-thinking-toggle",
      html: "&#128173; Thinking <span class='muse-thinking-hint'>(click to expand)</span>",
      on: {
        click: () => thinkingEl.classList.toggle("muse-collapsed"),
      },
    });
    const thinkingBody = el("div", { class: "muse-thinking-body" });
    thinkingEl.appendChild(thinkingToggle);
    thinkingEl.appendChild(thinkingBody);

    const answerEl = el("div", { class: "muse-answer" });

    bubble.appendChild(thinkingEl);
    // Inline image attachments (read live from input/, placeholder if missing).
    if (msg.images && msg.images.length) {
      const imgRow = el("div", { class: "muse-msg-images" });
      for (const name of msg.images) {
        const im = el("img", { class: "muse-msg-img", attrs: { src: api.inputFileUrl(name), alt: name, title: name } });
        im.addEventListener("error", () => {
          const ph = el("div", { class: "muse-msg-img-missing", text: "⚠ image not found" });
          im.replaceWith(ph);
        });
        imgRow.appendChild(im);
      }
      bubble.appendChild(imgRow);
    }
    bubble.appendChild(answerEl);
    wrap.appendChild(bubble);
    wrap.appendChild(makeToolbar(msg));

    messagesEl.appendChild(wrap);

    function update() {
      const { thinking, answer } = splitThinking(msg.content || "");
      const reasoning = (msg.thinking || "") + thinking;
      if (reasoning.trim()) {
        thinkingEl.style.display = "";
        thinkingBody.innerHTML = renderMarkdown(reasoning);
      } else {
        thinkingEl.style.display = "none";
      }
      answerEl.innerHTML = renderMarkdown(answer);
    }
    update();
    return { wrap, update, msg };
  }

  function renderAllMessages() {
    messagesEl.innerHTML = "";
    if (!state.chat || !state.chat.messages.length) {
      messagesEl.appendChild(
        el("div", { class: "muse-empty", html: "Start a conversation.<br>Your prompts stay local." })
      );
      return;
    }
    for (const m of state.chat.messages) renderMessage(m);
    autoScroll(true);
  }

  // ---- token info ----
  function updateTokenInfo() {
    const usage = state.chat && state.chat.token_usage;
    const guideTokens = estimateGuideTokens();
    const total = (usage ? usage.total_tokens || 0 : 0) + guideTokens;
    const suffix = guideTokens ? ` (+${guideTokens} guides)` : "";
    if (state.contextLength) {
      tokenInfo.textContent = `${total} / ${state.contextLength} tokens${suffix}`;
      const ratio = total / state.contextLength;
      tokenInfo.classList.toggle("muse-token-warn", ratio > 0.85);
    } else if (total) {
      tokenInfo.textContent = `${total} tokens used${suffix}`;
      tokenInfo.classList.remove("muse-token-warn");
    } else {
      tokenInfo.textContent = "";
    }
  }

  // ---- backend / model wiring ----
  async function refreshModels() {
    modelSelect.innerHTML = "";
    modelSelect.appendChild(el("option", { value: "", text: "Loading…" }));
    try {
      const models = await api.fetchModels(state.backend, state.baseUrl);
      state.models = models;
      modelSelect.innerHTML = "";
      if (!models.length) {
        modelSelect.appendChild(el("option", { value: "", text: "No models found" }));
        setStatus("not-loaded", "Connected, no models");
        return;
      }
      for (const m of models) {
        const label = m.loaded ? `● ${m.id}` : m.id;
        modelSelect.appendChild(el("option", { value: m.id, text: label }));
      }
      // Restore prior selection if still present.
      const want = state.model || (state.chat && state.chat.model);
      if (want && models.some((m) => m.id === want)) {
        modelSelect.value = want;
        state.model = want;
      } else {
        state.model = modelSelect.value;
      }
      syncSelectedModelMeta();
      const sel = models.find((m) => m.id === state.model);
      setStatus(sel && sel.loaded ? "loaded" : "not-loaded", "Connected");
    } catch (e) {
      modelSelect.innerHTML = "";
      modelSelect.appendChild(el("option", { value: "", text: "— unreachable —" }));
      setStatus("offline", `Offline: ${e.message}`);
      systemMessage(`Could not reach ${state.backend} at ${state.baseUrl}: ${e.message}`, "error");
    }
    updateLoadBtn();
  }

  function syncSelectedModelMeta() {
    const sel = state.models.find((m) => m.id === state.model);
    state.contextLength = (sel && sel.context_length) || null;
    updateTokenInfo();
  }

  async function refreshStatus() {
    if (!state.model) return;
    try {
      const st = await api.getStatus(state.backend, state.baseUrl, state.model);
      if (st.context_length) {
        state.contextLength = st.context_length;
        updateTokenInfo();
      }
      if (st.state === "loaded") setStatus("loaded", "Model loaded");
      else if (st.state === "not-loaded") setStatus("not-loaded", "Model not loaded");
      else if (st.state === "offline") setStatus("offline", st.error || "Offline");
    } catch (e) {
      /* leave last known status */
    }
  }

  async function doUnload() {
    if (state.statusState !== "loaded" || !state.model) return;
    unloadBtn.disabled = true;
    unloadBtn.textContent = "Unloading…";
    try {
      // confirm=true polls until VRAM is actually released.
      await api.unloadModel(state.backend, state.baseUrl, state.model, state.modelInstanceId, true);
      state.modelInstanceId = null;
      if (state.chat) {
        state.chat.model_instance_id = null;
        persistChat();
      }
      setStatus("not-loaded", "Model unloaded");
    } catch (e) {
      systemMessage(`Unload failed: ${e.message}`, "error");
      setStatus("loaded", e.message);
    }
    updateLoadBtn();
  }

  // ---- chat persistence ----
  let persistTimer = null;
  function persistChat() {
    if (!state.chat) return;
    clearTimeout(persistTimer);
    persistTimer = setTimeout(async () => {
      try {
        await api.updateChat(state.chat.id, state.chat);
        refreshChatList();
      } catch (e) {
        /* swallow; will retry on next change */
      }
    }, 400);
  }

  async function refreshChatList() {
    try {
      state.chats = await api.listChats();
    } catch (e) {
      return;
    }
    chatList.innerHTML = "";
    if (!state.chats.length) {
      chatList.appendChild(el("div", { class: "muse-chatlist-empty", text: "No chats yet" }));
      return;
    }
    for (const c of state.chats) {
      const active = state.chat && state.chat.id === c.id;
      const item = el("div", { class: `muse-chat-item${active ? " muse-active" : ""}` });

      const title = el("div", { class: "muse-chat-title", text: c.title || "Untitled" });
      const meta = el("div", { class: "muse-chat-meta", text: relativeTime(c.updated_at) });
      const main = el("div", { class: "muse-chat-itemmain" }, [title, meta]);

      const actions = el("div", { class: "muse-chat-actions" });
      const renameBtn = el("button", {
        class: "muse-chat-action", title: "Rename", html: "&#9998;",
        on: { click: (e) => { e.stopPropagation(); startInlineRename(c, title); } },
      });
      const delBtn = el("button", {
        class: "muse-chat-action muse-chat-del", title: "Delete", html: "&#128465;",
        on: { click: (e) => { e.stopPropagation(); showDeleteConfirm(c, actions); } },
      });
      actions.appendChild(renameBtn);
      actions.appendChild(delBtn);

      item.appendChild(main);
      item.appendChild(actions);
      item.addEventListener("click", () => switchChat(c.id));
      chatList.appendChild(item);
    }
  }

  // Inline rename — native prompt() is blocked inside ComfyUI's DOM widget, so we
  // swap the title for an editable input instead.
  function startInlineRename(c, titleEl) {
    const input = el("input", { class: "muse-rename-input", value: c.title || "" });
    titleEl.replaceWith(input);
    input.focus();
    input.select();
    let settled = false;
    const finish = (save) => {
      if (settled) return;
      settled = true;
      const val = input.value.trim();
      if (save && val) titleEl.textContent = val;
      input.replaceWith(titleEl);
      if (save && val && val !== (c.title || "")) commitRename(c, val);
    };
    input.addEventListener("click", (e) => e.stopPropagation());
    input.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter") { e.preventDefault(); finish(true); }
      else if (e.key === "Escape") { e.preventDefault(); finish(false); }
    });
    input.addEventListener("blur", () => finish(true));
  }

  async function commitRename(c, title) {
    c.title = title;
    if (state.chat && state.chat.id === c.id) {
      state.chat.title = title;
      persistChat();
    } else {
      try {
        const data = await api.getChat(c.id);
        data.title = title;
        await api.updateChat(c.id, data);
      } catch (e) {
        systemMessage(`Rename failed: ${e.message}`, "error");
      }
      refreshChatList();
    }
  }

  // Inline delete confirm — replaces the row's actions with ✓ / ✕ (confirm() is
  // also blocked in the widget context).
  function showDeleteConfirm(c, actionsEl) {
    actionsEl.innerHTML = "";
    actionsEl.style.display = "flex";
    const yes = el("button", {
      class: "muse-chat-action muse-chat-del", title: "Confirm delete", html: "&#10003;",
      on: { click: (e) => { e.stopPropagation(); doDelete(c); } },
    });
    const no = el("button", {
      class: "muse-chat-action", title: "Cancel", html: "&#10005;",
      on: { click: (e) => { e.stopPropagation(); refreshChatList(); } },
    });
    actionsEl.appendChild(yes);
    actionsEl.appendChild(no);
  }

  async function doDelete(c) {
    try {
      await api.deleteChat(c.id);
    } catch (e) {
      systemMessage(`Delete failed: ${e.message}`, "error");
      return;
    }
    if (state.chat && state.chat.id === c.id) {
      state.chat = null;
      const remaining = state.chats.filter((x) => x.id !== c.id);
      if (remaining.length) await switchChat(remaining[0].id);
      else await newChat();
    }
    refreshChatList();
  }

  async function newChat() {
    try {
      const data = await api.createChat({
        backend: state.backend,
        base_url: state.baseUrl,
        model: state.model,
        system_prompt: state.chat ? "" : "",
      });
      await switchChat(data.id);
      refreshChatList();
    } catch (e) {
      systemMessage(`Could not create chat: ${e.message}`, "error");
    }
  }

  async function switchChat(id) {
    if (state.streaming) stopGeneration();
    let data;
    try {
      data = await api.getChat(id);
      state.chat = data;
    } catch (e) {
      systemMessage(`Could not open chat: ${e.message}`, "error");
      return;
    }
    // Apply the chat's saved backend/model/system prompt to the UI.
    if (data.backend && data.backend !== state.backend) {
      state.backend = data.backend;
      backendSelect.value = data.backend;
    }
    if (data.base_url) {
      state.baseUrl = data.base_url;
      baseUrlInput.value = data.base_url;
    }
    state.model = data.model || state.model;
    state.modelInstanceId = data.model_instance_id || null;
    sysTextarea.value = data.system_prompt || "";
    if (data.system_prompt) sysDrawer.classList.remove("muse-collapsed");
    else sysDrawer.classList.add("muse-collapsed");

    // Per-chat generation settings + standing guides.
    if (!Array.isArray(state.chat.guides)) state.chat.guides = [];
    maxTokensInput.value = data.max_tokens || 2048;
    state.pendingImages = [];
    renderPendingImages();
    imagePicker.classList.add("muse-hidden");
    renderGuidesList();

    renderAllMessages();
    updateTokenInfo();
    refreshChatList();
    if (state.model) modelSelect.value = state.model;
    syncSelectedModelMeta();
  }

  // ---- sending / streaming ----
  function setComposerStreaming(on) {
    state.streaming = on;
    sendBtn.classList.toggle("muse-hidden", on);
    stopBtn.classList.toggle("muse-hidden", !on);
    inputArea.disabled = on;
    if (!on) inputArea.focus();
  }

  // Stream one assistant reply. Assumes state.chat.messages currently ends with
  // a user message (no trailing assistant). Shared by send() and regenerate.
  async function runGeneration() {
    if (state.streaming || !state.chat) return;
    if (!state.model) {
      systemMessage("Select a model first.", "error");
      return;
    }

    const assistantMsg = { role: "assistant", content: "", thinking: null, timestamp: new Date().toISOString() };
    state.chat.messages.push(assistantMsg);
    const rendered = renderMessage(assistantMsg);
    autoScroll(true);

    state.abortCtl = new AbortController();
    setComposerStreaming(true);
    let aborted = false;
    let streamError = null;
    let finishReason = null;

    try {
      await api.streamChat(
        {
          backend: state.backend,
          baseUrl: state.baseUrl,
          model: state.model,
          systemPrompt: sysTextarea.value,
          messages: state.chat.messages.slice(0, -1).map((m) => ({
            role: m.role,
            content: m.content,
            images: m.images && m.images.length ? m.images : undefined,
          })),
          freeComfyVram: state.freeComfyVram,
          maxTokens: state.chat.max_tokens || 2048,
          guides: activeGuides(),
        },
        (chunk) => {
          if (chunk.error) {
            streamError = chunk.error;
            return;
          }
          if (chunk.status === "freeing-vram") {
            tokenInfo.textContent = "Freeing ComfyUI VRAM…";
            return;
          }
          if (chunk.finish_reason) finishReason = chunk.finish_reason;
          if (chunk.thinking) {
            assistantMsg.thinking = (assistantMsg.thinking || "") + chunk.thinking;
          }
          if (chunk.delta) {
            assistantMsg.content += chunk.delta;
          }
          if (chunk.usage) {
            state.chat.token_usage = chunk.usage;
            updateTokenInfo();
          }
          rendered.update();
          autoScroll();
        },
        state.abortCtl.signal
      );
    } catch (e) {
      if (e.name === "AbortError") aborted = true;
      else streamError = e.message;
    }
    rendered.update();

    // Reply was truncated by the token limit — nudge the user to raise it.
    if (finishReason === "length") {
      systemMessage("Reply hit the max-token limit — raise “Max reply tokens” for longer output.", "info");
    }

    state.abortCtl = null;
    setComposerStreaming(false);

    if (streamError) {
      // Show the error transiently instead of baking it into the chat history.
      systemMessage(streamError, "error");
      setStatus("error", streamError);
      // No partial output? Drop the empty assistant bubble so nothing lingers.
      if (!assistantMsg.content.trim()) {
        const idx = state.chat.messages.indexOf(assistantMsg);
        if (idx !== -1) state.chat.messages.splice(idx, 1);
        rendered.wrap.remove();
      }
    } else if (!aborted) {
      // The model JIT-loads on send, so a successful stream means it's in VRAM —
      // reflect that so the Unload button becomes available.
      setStatus("loaded", "Model loaded");
      refreshStatus();
    }

    persistChat();
  }

  async function send() {
    const text = inputArea.value.trim();
    if ((!text && !state.pendingImages.length) || state.streaming) return;
    if (!state.model) {
      systemMessage("Select a model first.", "error");
      return;
    }
    if (!state.chat) await newChat();

    // Warn (don't block) if attaching images to a model that isn't vision-capable.
    if (state.pendingImages.length && !modelSupportsVision()) {
      systemMessage("This model may not support image input — the image could be ignored.", "info");
    }

    const userMsg = {
      role: "user",
      content: text,
      thinking: null,
      timestamp: new Date().toISOString(),
    };
    if (state.pendingImages.length) userMsg.images = state.pendingImages.slice();
    state.chat.messages.push(userMsg);
    state.pendingImages = [];
    renderPendingImages();
    imagePicker.classList.add("muse-hidden");

    // Auto-title from first user message.
    if (state.chat.messages.filter((m) => m.role === "user").length === 1) {
      state.chat.title = (text || "Image").slice(0, 48);
    }
    state.chat.model = state.model;
    state.chat.backend = state.backend;
    state.chat.base_url = state.baseUrl;
    state.chat.system_prompt = sysTextarea.value;

    inputArea.value = "";
    autoGrow();
    const emptyHint = messagesEl.querySelector(".muse-empty");
    if (emptyHint) emptyHint.remove();
    renderMessage(userMsg);
    autoScroll(true);

    await runGeneration();
  }

  // Regenerate an assistant reply: drop it (and anything after it) and re-stream
  // from the preceding user message.
  async function regenerateMessage(msg) {
    if (state.streaming || !state.chat) return;
    const idx = state.chat.messages.indexOf(msg);
    if (idx < 0) return;
    state.chat.messages.splice(idx);
    // Must end on a user message to regenerate from.
    const last = state.chat.messages[state.chat.messages.length - 1];
    renderAllMessages();
    if (!last || last.role !== "user") {
      persistChat();
      return;
    }
    await runGeneration();
  }

  function deleteMessage(msg) {
    if (state.streaming || !state.chat) return;
    const idx = state.chat.messages.indexOf(msg);
    if (idx < 0) return;
    state.chat.messages.splice(idx, 1);
    renderAllMessages();
    persistChat();
  }

  function stopGeneration() {
    if (state.abortCtl) state.abortCtl.abort();
  }

  // ---- input auto-grow ----
  function autoGrow() {
    inputArea.style.height = "auto";
    inputArea.style.height = Math.min(inputArea.scrollHeight, 160) + "px";
  }

  // ---- event wiring ----
  backendSelect.addEventListener("change", () => {
    state.backend = backendSelect.value;
    state.baseUrl = DEFAULT_URLS[state.backend];
    baseUrlInput.value = state.baseUrl;
    state.model = null;
    refreshModels();
  });
  baseUrlInput.addEventListener("change", () => {
    state.baseUrl = baseUrlInput.value.trim() || DEFAULT_URLS[state.backend];
    refreshModels();
  });
  modelSelect.addEventListener("change", () => {
    state.model = modelSelect.value;
    if (state.chat) {
      state.chat.model = state.model;
      persistChat();
    }
    syncSelectedModelMeta();
    refreshStatus();
    updateLoadBtn();
  });
  refreshBtn.addEventListener("click", refreshModels);
  unloadBtn.addEventListener("click", doUnload);
  vramToggle.addEventListener("change", () => {
    state.freeComfyVram = vramToggle.checked;
    try {
      localStorage.setItem(VRAM_STORAGE_KEY, vramToggle.checked ? "1" : "0");
    } catch (e) {
      /* storage may be unavailable */
    }
  });
  runToggle.addEventListener("change", () => {
    try {
      localStorage.setItem("museChatUnloadOnRun", runToggle.checked ? "1" : "0");
    } catch (e) {
      /* storage may be unavailable */
    }
  });
  newChatBtn.addEventListener("click", newChat);
  sendBtn.addEventListener("click", send);
  stopBtn.addEventListener("click", stopGeneration);

  sysToggle.addEventListener("click", () => {
    sysDrawer.classList.toggle("muse-collapsed");
    sysToggle.classList.toggle("muse-open", !sysDrawer.classList.contains("muse-collapsed"));
  });
  sysTextarea.addEventListener("change", () => {
    if (state.chat) {
      state.chat.system_prompt = sysTextarea.value;
      persistChat();
    }
  });
  maxTokensInput.addEventListener("change", () => {
    const v = parseInt(maxTokensInput.value, 10);
    const val = isNaN(v) || v < 64 ? 2048 : v;
    maxTokensInput.value = val;
    if (state.chat) {
      state.chat.max_tokens = val;
      persistChat();
    }
  });

  guidesToggle.addEventListener("click", () => {
    guidesDrawer.classList.toggle("muse-collapsed");
    guidesToggle.classList.toggle("muse-open", !guidesDrawer.classList.contains("muse-collapsed"));
  });
  guidesRefresh.addEventListener("click", (e) => { e.stopPropagation(); refreshGuides(); });

  attachBtn.addEventListener("click", toggleImagePicker);

  sidebarToggle.addEventListener("click", () => {
    state.sidebarCollapsed = !state.sidebarCollapsed;
    sidebar.classList.toggle("muse-sidebar-collapsed", state.sidebarCollapsed);
  });

  // Drag-and-drop images anywhere on the panel: save to input/, then attach.
  // stopPropagation so ComfyUI's canvas doesn't also spawn a LoadImage node.
  root.addEventListener("dragover", (e) => {
    if (e.dataTransfer && [...e.dataTransfer.types].includes("Files")) {
      e.preventDefault();
      e.stopPropagation();
      root.classList.add("muse-dragover");
    }
  });
  root.addEventListener("dragleave", (e) => {
    if (e.target === root) root.classList.remove("muse-dragover");
  });
  root.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
      e.preventDefault();
      e.stopPropagation();
      root.classList.remove("muse-dragover");
      handleDroppedFiles(e.dataTransfer.files);
    }
  });

  inputArea.addEventListener("input", autoGrow);
  inputArea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  // ---- init ----
  async function init() {
    initAccent();
    await refreshChatList();
    if (state.chats.length) {
      await switchChat(state.chats[0].id);
    } else {
      await newChat();
    }
    await refreshModels();
    await refreshGuides();
    refreshInputImages();
    updateLoadBtn();
  }
  init();

  // Used by the Run-button hook in muse_chat.js to free VRAM before a render.
  async function unloadForRun() {
    if (state.statusState !== "loaded" || !state.model) return;
    setStatus("not-loaded", "Freeing VRAM for ComfyUI…");
    try {
      await api.unloadModel(state.backend, state.baseUrl, state.model, state.modelInstanceId, true);
      state.modelInstanceId = null;
    } catch (e) {
      /* best effort — don't block the render on a failed unload */
    }
  }

  return {
    root,
    unloadForRun,
    isLoaded: () => state.statusState === "loaded",
    destroy() {
      stopGeneration();
      clearTimeout(persistTimer);
    },
  };
}

window.MuseChatUI = { create };
export default { create };
