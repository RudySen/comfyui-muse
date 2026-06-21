# ComfyUI-Muse

**A real LLM chat panel that lives inside ComfyUI — not a one-shot prompt box.**

![Muse Chat generating prompts](docs/screenshot-5.png)

## Why this exists

Running an LLM inside ComfyUI itself is slow and limited. So I ran mine in LM Studio / Ollama instead — but that meant constant alt-tabbing to copy prompts back and forth, and manually unloading whichever app I wasn't using so the other one had VRAM. And it was never just about prompts — sometimes I just wanted to ask a question, which meant opening a whole separate app for a five-second chat.

There wasn't a node that solved this properly, so I built one. Muse talks to the LLM server you're already running locally — no duplicate inference engine, no extra VRAM tax — so you get a real chat experience without ever leaving your workflow.

## What it does

- **Two backends** — LM Studio and Ollama, switchable on the fly.
- **Multi-session chats** — create, switch, rename, delete. Each chat keeps its own history, system prompt, model, and settings, saved to disk and restored across restarts.
- **Streaming replies** with a Stop button, and a configurable **max reply length** so multi-item outputs ("write 5 prompts") don't get cut short.
- **Hands-free VRAM coordination** — Muse frees ComfyUI's VRAM before loading its own model, and confirms its own model is fully unloaded *before* your Run/Queue actually starts. No more remembering to unload anything yourself.
- **Guide Materials** — point Muse at `.txt`/`.md`/`.json` files in `ComfyUI/input/` and pick which ones shape *every* reply in a chat — your own style rules, official references, whatever you want the model to always follow.
- **Image attachments** — drag-and-drop or pick images from `input/` for vision-capable models, with no artificial per-chat cap.
- **Reasoning model support** — `<think>` output collapses into its own block, separate from the answer.
- **Per-message Copy / Regenerate / Delete**, live token usage against the model's real context window, and a frosted-glass UI with an accent color picker.

It's a **true drop-in**: no `pip install`, no build step, no downloads. The only dependency is `aiohttp`, which already ships with ComfyUI.

## Install

1. Copy this folder into your ComfyUI custom nodes directory:
   ```
   ComfyUI/custom_nodes/comfyui-muse/
   ```
2. Restart ComfyUI.
3. Add the node: right-click the canvas → **Add Node → utils → muse → Muse Chat**.

![Adding the Muse Chat node](docs/screenshot-1.png)

## Quick start

1. Pick a backend, confirm the base URL, hit **⟳** to list models.
2. Type a message and press **Enter**. The model loads automatically and streams a reply.
3. Drop reference files into `ComfyUI/input/` and toggle them on under **Guide Materials** if you want standing context for that chat.
4. Copy whatever you like into your graph. That's the whole loop.

> This node has **no graph inputs or outputs** by design. It's safely ignored by Queue Prompt — drop it into any workflow as a pure tool, not a dependency.

![Panel overview](docs/screenshot-3.png)

## Full docs

- [VRAM coordination](#vram-coordination)
- [Guide Materials](#guide-materials)
- [Image attachments](#image-attachments)

### VRAM coordination

Two toggles in the top bar, both on by default:
- **Free ComfyUI VRAM** — unloads ComfyUI's models before the chat model loads.
- **Unload on Run** — unloads the chat model and *confirms* it's actually freed before your Run/Queue proceeds. Doesn't auto-reload — just send a new message when you want to chat again.

A manual **Unload** button is always available too.

### Guide Materials

Standing, per-chat reference material that shapes every reply — not a one-off upload.

- Drop `.txt` / `.md` / `.json` files into `ComfyUI/input/`.
- Check the ones you want active for the current chat.
- Muse reads them fresh from disk on every message, so edits apply automatically. Missing files are flagged, not silently dropped. Their token cost is included in the usage meter.

### Image attachments

For vision-capable models — click **+** to pick from `ComfyUI/input/`, or drag-and-drop a file onto the panel (it gets saved into `input/` first, then attached). Thumbnails persist in chat history; the actual image bytes are read live from disk each time, so chat files stay small.

![Image attachment with a vision model](docs/screenshot-7.png)

## Where it fits

- Iterating on prompts right next to the workflow that's going to use them.
- Cleaning up or expanding tag lists with a local model.
- Per-project chat threads with their own guide files — a house style that shapes every prompt automatically.
- Describing a reference image and asking for a prompt "in this style."
- Chatting freely, then just hitting Run — Muse gets out of the GPU's way on its own.

## Requirements

- A recent ComfyUI build (custom DOM widget support).
- A running **LM Studio** and/or **Ollama** locally, with at least one model.
- A vision-capable model for image attachments.

## Roadmap

This is v1. If people find it useful, next up: broader local API support, MCP support, and whatever workflow gaps people actually report.

## Feedback

First public project — bug reports, feature requests, and PRs are all genuinely welcome.

## Support

Free, and staying free. If Muse saves you the back-and-forth it used to cost me, a coffee is appreciated: **[☕ Support me on Ko-fi](https://ko-fi.com/rudysen)**. A star or a share helps just as much.

## License

MIT.
