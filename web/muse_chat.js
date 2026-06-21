// muse_chat.js — registers the ComfyUI frontend extension and injects the Muse
// chat panel as a DOM widget into the MuseChat node body.

import { app } from "../../scripts/app.js";
import MuseChatUI from "./muse_chat_ui.js";

const DEFAULT_W = 440;
const DEFAULT_H = 640;
const MIN_W = 320;
const MIN_H = 300;

// Live Muse panels, so the Run-button gate can ask each to free its VRAM.
const museControllers = new Set();
let queueGateInstalled = false;

// Wrap app.queuePrompt ONCE so that, before ComfyUI's queued generation actually
// runs, every loaded Muse model is unloaded and confirmed freed first — avoiding
// the LLM and ComfyUI contending for VRAM at the same moment. Frontend-side
// interception (delaying the queuePrompt call) per v2 spec §2.3.
function installQueueGate() {
  if (queueGateInstalled || typeof app.queuePrompt !== "function") return;
  queueGateInstalled = true;
  const original = app.queuePrompt.bind(app);
  app.queuePrompt = async function (...args) {
    let enabled = true;
    try {
      enabled = localStorage.getItem("museChatUnloadOnRun") !== "0";
    } catch (e) {
      /* default on */
    }
    if (enabled) {
      const pending = [...museControllers].filter((c) => c.isLoaded && c.isLoaded());
      if (pending.length) {
        try {
          await Promise.all(pending.map((c) => c.unloadForRun && c.unloadForRun()));
        } catch (e) {
          /* best effort — never block the render on an unload hiccup */
        }
      }
    }
    return original(...args);
  };
}

// Drag a node by a DOM handle. Translates screen-pixel movement into graph
// coordinates (dividing by the canvas zoom) and updates node.pos. Move/up are
// bound to window so the drag survives the pointer leaving the small strip.
function makeNodeDraggable(handle, node) {
  function redraw() {
    if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
    else if (app.graph) app.graph.setDirtyCanvas(true, true);
    else if (app.canvas) app.canvas.setDirty(true, true);
  }

  handle.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    const startX = e.clientX;
    const startY = e.clientY;
    const startPos = [node.pos[0], node.pos[1]];
    handle.classList.add("muse-dragging");

    // The DOM widget swallows the canvas mousedown, so the node never gets
    // selected (no white border) the normal way — do it explicitly.
    if (app.canvas) {
      if (typeof app.canvas.selectNode === "function") app.canvas.selectNode(node);
      else if (typeof app.canvas.selectNodes === "function") app.canvas.selectNodes([node]);
    }

    const onMove = (ev) => {
      const scale = (app.canvas && app.canvas.ds && app.canvas.ds.scale) || 1;
      // Reassign (don't mutate) — node.pos may be a getter returning a copy.
      node.pos = [
        startPos[0] + (ev.clientX - startX) / scale,
        startPos[1] + (ev.clientY - startY) / scale,
      ];
      redraw();
      ev.preventDefault();
    };
    const onUp = () => {
      handle.classList.remove("muse-dragging");
      window.removeEventListener("mousemove", onMove, true);
      window.removeEventListener("mouseup", onUp, true);
    };
    window.addEventListener("mousemove", onMove, true);
    window.addEventListener("mouseup", onUp, true);

    e.preventDefault();
    e.stopPropagation();
  });
}

// Inject the stylesheet once. WEB_DIRECTORY serves it at this path.
function ensureStyles() {
  if (document.getElementById("muse-chat-style")) return;
  const link = document.createElement("link");
  link.id = "muse-chat-style";
  link.rel = "stylesheet";
  link.href = new URL("./muse_chat.css", import.meta.url).href;
  document.head.appendChild(link);
}

app.registerExtension({
  name: "Muse.Chat",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "MuseChat") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      ensureStyles();

      const node = this;
      const container = document.createElement("div");
      container.className = "muse-widget-root";

      // addDOMWidget is the standard mechanism to embed arbitrary HTML in a node.
      const widget = this.addDOMWidget("muse_chat", "muse_chat", container, {
        serialize: false,
        hideOnZoom: false,
      });

      // The DOM element is stretched to fill the node body. Its height tracks
      // the node height (minus the widget's own top offset) so the panel fills
      // on resize. This value is used ONLY to lay out the element — node growth
      // is decoupled via the node.computeSize override below, so this can't
      // feed back into the node's size.
      widget.computeSize = function (width) {
        const top = this.y || 0;
        const h = Math.max(MIN_H - 60, node.size[1] - top - 6);
        return [width, h];
      };

      const controller = MuseChatUI.create(container);
      this.museController = controller;
      museControllers.add(controller);
      installQueueGate();

      // Make the panel's title strip drag the node. The DOM widget swallows the
      // canvas title-bar drag, so without this the node can't be moved.
      const dragbar = container.querySelector(".muse-dragbar");
      if (dragbar) makeNodeDraggable(dragbar, node);

      if (!this.size || this.size[0] < DEFAULT_W) {
        this.setSize([DEFAULT_W, DEFAULT_H]);
      }

      const onRemoved = this.onRemoved;
      this.onRemoved = function () {
        museControllers.delete(controller);
        try {
          controller.destroy();
        } catch (e) {
          /* ignore */
        }
        return onRemoved ? onRemoved.apply(this, arguments) : undefined;
      };

      return r;
    };

    // Decouple node sizing from widget content. ComfyUI calls node.computeSize()
    // during a drag to get the *minimum* size, and it does so BEFORE applying the
    // newly dragged size — so anything size-derived clamps you to the pre-drag
    // (larger) size and you can only ever grow. Returning a fixed floor lets the
    // node freely shrink down to MIN_W x MIN_H and grow as large as you drag,
    // with no self-growth (the panel fill is handled by widget.computeSize).
    nodeType.prototype.computeSize = function (out) {
      const res = out || new Float32Array(2);
      res[0] = MIN_W;
      res[1] = MIN_H;
      return res;
    };

    // Hard floor so the glass panels never get crushed to nothing.
    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      size[0] = Math.max(size[0], MIN_W);
      size[1] = Math.max(size[1], MIN_H);
      return onResize ? onResize.apply(this, arguments) : undefined;
    };
  },
});
