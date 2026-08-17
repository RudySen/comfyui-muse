# Muse V2

Hey everyone — Rudy here.

I honestly didn't expect the reaction V1 got. Seeing people actually use this in their day-to-day ComfyUI workflow, star it, and tell me what was annoying about it — that's what pushed me to sit down and do a proper V2 instead of leaving it as a one-off tool I built for myself. This update is basically a direct response to that feedback, plus a few things I ran into myself while using it for my own renders.

Here's what changed and why.

## "Just let me load a GGUF, I don't want to open LM Studio"

This was the biggest ask, and it's the headline feature of V2: a **Direct model loader**. You point it at a folder of GGUF models — your LM Studio models folder, wherever you keep them — and Muse loads them straight from disk. No LM Studio, no Ollama, nothing else running in the background.

I was pretty stubborn about one thing here: it had to be as fast as LM Studio, not one of those "technically works but takes forever" solutions. Turns out LM Studio's own GGUF engine *is* llama.cpp under the hood, so instead of writing my own inference code (please, no), Muse spawns the real `llama-server` binary and talks to it the same way it talks to LM Studio. Same engine, same speed, no compromise.

The part I'm actually proud of: you don't have to go hunting for `llama-server` yourself. Click **Download llama-server** in the settings and Muse figures out your OS/GPU, grabs the right build straight from the official llama.cpp releases, and installs it — one click. Already have an install somewhere? **Use existing install** finds it on PATH for you. Git clone the node, click one button, you're chatting with a local GGUF model. That was the whole point.

**How to use it:** switch the backend dropdown to **Direct (GGUF)**, open Direct Loader settings, hit **Download llama-server**, add a folder of models, pick one, chat.

## Things I broke on myself first (and then fixed)

Once Direct loading was working, I threw a 31B model at it and immediately learned some things the hard way — which is exactly the kind of feedback loop I want, even when it's just me:

- **It ran out of VRAM and crashed.** Turns out defaulting to "offload every layer, use the model's full context window" is a great way to blow past your GPU's memory on anything big. Fixed the context-length default (down to a much saner 8192 from "whatever the model claims it supports," which can be 128k+ tokens) and added an actual **Fit to GPU** button that checks your free VRAM and suggests a layer count that'll actually fit — same idea as LM Studio's GPU offload slider, just automatic.
- **When it didn't crash, it was suspiciously slow.** This one was sneakier — on some setups, overshooting your VRAM doesn't error out, it just quietly spills into system RAM and tanks your speed without telling you why. Now if a load fails from running out of GPU memory, Muse automatically retries at lower layer counts instead of just giving up, so you get a working, fast setup without babysitting a slider yourself.
- **I had no idea what was happening while a model loaded.** No progress bar, no logs, just... waiting. Added a live elapsed-time indicator so you know it's actually working, plus a retractable **Log** panel so you can see exactly what `llama-server` is doing (or why it failed) instead of guessing.

## The rest of the V2 list

A grab-bag of things people asked for or that I felt were missing:

- **Edit & resend** — fix a message instead of deleting and retyping the whole thing.
- **Chat branching** — fork a new conversation from any earlier message, not just the last one, so you can try a different direction without losing the original thread.
- **Video attachments** — attach a video and a vision model like Qwen3-VL can reason about it frame-by-frame (sampled and timestamped automatically).
- **Audio attachments** — same idea for audio-capable models like Qwen2.5-Omni via the Direct loader.
- **Broader image format support** — basically anything Pillow can open now gets converted and sent, instead of quietly failing on formats a vision API doesn't recognize.

Full details on all of it, and how to use each one, are in the [README](README.md).

## What's next

If something's broken, slow, or just annoying, tell me — that's genuinely how V1 became this. Open an issue, or find me:

**Ko-fi:** [ko-fi.com/rudysen](https://ko-fi.com/rudysen) — if Muse saved you some time and you want to buy me a coffee, it's appreciated but never expected.

**Instagram:** [@rudysen_official](https://www.instagram.com/rudysen_official/)

Thanks for using this thing. Onward to V3, probably whenever the next thing annoys me enough.

— Rudy
