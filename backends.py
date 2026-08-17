"""
Backend connectors for ComfyUI-Muse.

Backend-agnostic async functions for talking to LM Studio, Ollama, and our own
"direct" (llama.cpp-server-backed) loader. Everything returns *normalized* data
structures so server.py never has to special-case which backend is active
beyond passing the backend name through.

Normalized shapes
-----------------
model:  {"id": str, "loaded": bool|None, "context_length": int|None, "state": str|None,
         "vision": bool|None, "audio": bool|None, "display_name": str (optional)}
chunk:  {"delta": str, "thinking": str, "done": bool,
         "usage": {"prompt_tokens", "completion_tokens", "total_tokens"} | None,
         "stats": {"tokens_per_second", "time_to_first_token"} | None,
         "error": str | None}

The "direct" backend doesn't talk to an already-running app the way LM Studio/
Ollama do — it spawns and manages its own llama-server process (see
direct_backend.py) and, once that's up, is OpenAI-compatible, so it reuses the
same _lmstudio_* HTTP plumbing for the actual chat traffic.

Zero extra hard dependencies: only aiohttp (already a ComfyUI dependency) + stdlib.
"""

import asyncio
import json
import time

import aiohttp

from . import direct_backend
from . import settings_store

LMSTUDIO_DEFAULT_URL = "http://localhost:1234"
OLLAMA_DEFAULT_URL = "http://localhost:11434"

DEFAULT_MAX_TOKENS = 2048

DEFAULT_URLS = {
    "lmstudio": LMSTUDIO_DEFAULT_URL,
    "ollama": OLLAMA_DEFAULT_URL,
    "direct": "",  # managed internally — no user-facing base URL
}

# Generous timeout: model load + first token on a cold model can take a while.
_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)


class BackendError(Exception):
    """Raised for any backend communication failure with a user-readable message."""


def default_url(backend):
    return DEFAULT_URLS.get(backend, LMSTUDIO_DEFAULT_URL)


def _base(base_url, backend):
    return (base_url or default_url(backend)).strip().rstrip("/")


def _fmt_ts(seconds):
    seconds = seconds or 0
    m = int(seconds // 60)
    s = seconds - m * 60
    return "%d:%05.2f" % (m, s) if m else "%.2fs" % s


def _build_openai_messages(system_prompt, messages):
    """OpenAI/LM Studio/llama.cpp-style messages. Messages may carry resolved
    `images` (list of {"data": b64, "mime": str}) -> image_url content parts,
    `video_frames` (list of {"name", "duration", "fps", "frames": [{"data",
    "mime", "timestamp"}]}) -> an interleaved "t=0:03.00" text part + image_url
    part per sampled frame, and `audio` (list of {"name", "data", "format"})
    -> input_audio content parts (the OpenAI/llama.cpp-server shape: {"type":
    "input_audio", "input_audio": {"data": b64, "format": "wav"}}). LM Studio
    doesn't understand input_audio as of this writing, but the shape is
    forward-compatible and is exactly what llama-server (our Direct backend)
    expects, so it's built here rather than duplicated per-backend."""
    out = []
    if system_prompt and system_prompt.strip():
        out.append({"role": "system", "content": system_prompt})
    for m in messages:
        role = m.get("role", "user")
        if role not in ("user", "assistant"):
            continue
        text = m.get("content", "") or ""
        images = m.get("images") or []
        videos = m.get("video_frames") or []
        audio = m.get("audio") or []
        if (images or videos or audio) and role == "user":
            parts = [{"type": "text", "text": text}]
            for img in images:
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": "data:%s;base64,%s" % (img["mime"], img["data"])},
                })
            for vid in videos:
                frames = vid.get("frames") or []
                parts.append({
                    "type": "text",
                    "text": "--- Video: %s (%d sampled frames, ~%.1fs) ---" % (
                        vid.get("name", "video"), len(frames), vid.get("duration") or 0,
                    ),
                })
                for fr in frames:
                    parts.append({"type": "text", "text": "t=%s" % _fmt_ts(fr.get("timestamp"))})
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": "data:%s;base64,%s" % (fr["mime"], fr["data"])},
                    })
            for a in audio:
                parts.append({
                    "type": "input_audio",
                    "input_audio": {"data": a["data"], "format": a.get("format") or "wav"},
                })
            out.append({"role": role, "content": parts})
        else:
            out.append({"role": role, "content": text})
    return out


def _build_ollama_messages(system_prompt, messages):
    """Ollama-style messages. Images (and sampled video frames) go in a
    separate `images` array of base64 — Ollama's protocol has no per-image
    text interleaving, so frame timestamps are folded into the text content
    instead, in the same order the frames are attached. Ollama has no audio
    input mechanism at all (as of this writing), so audio attachments are
    noted in text only — nothing to actually send."""
    out = []
    if system_prompt and system_prompt.strip():
        out.append({"role": "system", "content": system_prompt})
    for m in messages:
        role = m.get("role", "user")
        if role not in ("user", "assistant"):
            continue
        text = m.get("content", "") or ""
        images = list(m.get("images") or [])
        videos = m.get("video_frames") or []
        audio = m.get("audio") or []
        for vid in videos:
            frames = vid.get("frames") or []
            ts_list = ", ".join("t=%s" % _fmt_ts(fr.get("timestamp")) for fr in frames)
            text += "\n\n[Video: %s — %d sampled frames attached in order below, timestamps: %s]" % (
                vid.get("name", "video"), len(frames), ts_list,
            )
            images.extend(frames)
        for a in audio:
            text += "\n\n[Audio attached: %s — Ollama's API has no audio input, so this could not be sent]" % a.get("name", "audio")
        msg = {"role": role, "content": text}
        if images and role == "user":
            msg["images"] = [img["data"] for img in images]
        out.append(msg)
    return out


# ---------------------------------------------------------------------------
# LM Studio
# ---------------------------------------------------------------------------

def _normalize_lmstudio_models(data):
    if isinstance(data, dict):
        items = data.get("data")
        if items is None:
            items = data.get("models", [])
    else:
        items = data
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        mid = it.get("id") or it.get("key") or it.get("model")
        if not mid:
            continue
        state = it.get("state")
        loaded = (state == "loaded") if state is not None else None
        ctx = (
            it.get("max_context_length")
            or it.get("loaded_context_length")
            or it.get("context_length")
        )
        out.append({
            "id": mid, "loaded": loaded, "context_length": ctx, "state": state,
            "vision": _detect_vision(it, mid),
            "audio": _detect_audio(it, mid),
        })
    return out


def _detect_vision(it, mid):
    """Best-effort vision-capability detection. Returns True/False/None(unknown).
    LM Studio model objects vary across versions; check the likely fields and
    fall back to a name heuristic, but never hard-block on a guess (None=unknown)."""
    caps = it.get("capabilities")
    if isinstance(caps, (list, tuple)):
        low = [str(c).lower() for c in caps]
        if any("vision" in c or "image" in c for c in low):
            return True
    if it.get("vision") is True or it.get("type") in ("vlm", "vision"):
        return True
    name = str(mid).lower()
    if any(tok in name for tok in ("vl", "vision", "llava", "-vl-", "gemma-3", "qwen2.5-vl", "qwen2-vl")):
        return True
    return None


def _detect_audio(it, mid):
    """Best-effort audio-input-capability detection, same shape/caveats as
    _detect_vision. Nothing exposes this reliably via an API field yet, so
    it's name-heuristic only — used purely to give the user a heads-up before
    they attach an audio file to a model that likely won't use it."""
    caps = it.get("capabilities")
    if isinstance(caps, (list, tuple)):
        low = [str(c).lower() for c in caps]
        if any("audio" in c or "speech" in c for c in low):
            return True
    name = str(mid).lower()
    if any(tok in name for tok in ("audio", "omni", "ultravox", "voxtral", "moshi")):
        return True
    return None


async def _lmstudio_list_models(session, base_url):
    last_err = None
    for path in ("/api/v1/models", "/api/v0/models", "/v1/models"):
        try:
            async with session.get(base_url + path) as resp:
                if resp.status != 200:
                    last_err = "%s -> HTTP %s" % (path, resp.status)
                    continue
                data = await resp.json(content_type=None)
                return _normalize_lmstudio_models(data)
        except aiohttp.ClientError as e:
            last_err = str(e)
    raise BackendError("Could not reach LM Studio at %s (%s)" % (base_url, last_err))


async def _lmstudio_chat_stream(session, base_url, model, messages, system_prompt, max_tokens):
    payload = {
        "model": model,
        "messages": _build_openai_messages(system_prompt, messages),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if max_tokens and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    paths = ("/api/v0/chat/completions", "/v1/chat/completions", "/api/v1/chat/completions")
    last_err = None
    for path in paths:
        try:
            async with session.post(base_url + path, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    last_err = "%s -> HTTP %s: %s" % (path, resp.status, body[:300])
                    continue
                async for chunk in _parse_openai_sse(resp):
                    yield chunk
                return
        except aiohttp.ClientError as e:
            last_err = str(e)
    raise BackendError("LM Studio chat failed: %s" % last_err)


async def _parse_openai_sse(resp):
    """Parse an OpenAI-compatible SSE stream into normalized chunks."""
    usage = None
    stats = None
    finish_reason = None
    async for raw in resp.content:
        line = raw.decode("utf-8", "replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if obj.get("usage"):
            u = obj["usage"]
            usage = {
                "prompt_tokens": u.get("prompt_tokens", 0),
                "completion_tokens": u.get("completion_tokens", 0),
                "total_tokens": u.get("total_tokens", 0),
            }
        if obj.get("stats"):
            s = obj["stats"]
            stats = {
                "tokens_per_second": s.get("tokens_per_second"),
                "time_to_first_token": s.get("time_to_first_token"),
            }
        choices = obj.get("choices") or []
        if choices:
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            thinking = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if content or thinking:
                yield {
                    "delta": content,
                    "thinking": thinking,
                    "done": False,
                    "usage": None,
                    "stats": None,
                    "error": None,
                }
    yield {"delta": "", "thinking": "", "done": True, "usage": usage, "stats": stats,
           "error": None, "finish_reason": finish_reason}


async def _lmstudio_load(session, base_url, model):
    payload = {"model": model, "ttl": -1}
    try:
        async with session.post(base_url + "/api/v1/models/load", json=payload) as resp:
            if resp.status == 200:
                data = await resp.json(content_type=None)
                return {
                    "loaded": True,
                    "instance_id": data.get("instance_id") or data.get("id"),
                    "status": data.get("status", "loaded"),
                }
            body = await resp.text()
    except aiohttp.ClientError as e:
        raise BackendError("LM Studio load failed: %s" % e)
    # Fallback: JIT load via a tiny chat completion warms the model into VRAM.
    try:
        async with session.post(
            base_url + "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
        ) as resp:
            if resp.status == 200:
                return {"loaded": True, "instance_id": None, "status": "loaded"}
            body = await resp.text()
    except aiohttp.ClientError as e:
        raise BackendError("LM Studio load failed: %s" % e)
    raise BackendError("LM Studio load failed: %s" % body[:300])


async def _lmstudio_loaded_instance_ids(session, base_url, model):
    """Best-effort discovery of candidate instance ids for a loaded model.

    JIT-loaded models (loaded implicitly by a chat request) never gave us an
    instance_id, but /models/unload requires one. Scan the model list for any
    id-like field on the matching loaded entry.
    """
    candidates = []
    for path in ("/api/v1/models", "/api/v0/models"):
        try:
            async with session.get(base_url + path) as resp:
                if resp.status != 200:
                    continue
                data = await resp.json(content_type=None)
        except aiohttp.ClientError:
            continue
        if isinstance(data, dict):
            items = data.get("data")
            if items is None:
                items = data.get("models", [])
        else:
            items = data
        for it in items or []:
            if not isinstance(it, dict):
                continue
            mid = it.get("id") or it.get("key") or it.get("model")
            if mid != model:
                continue
            if it.get("state") not in (None, "loaded"):
                continue
            for key in ("instance_id", "instance_reference", "identifier", "instance"):
                v = it.get(key)
                if v and v not in candidates:
                    candidates.append(v)
        if candidates:
            break
    return candidates


async def _lmstudio_unload(session, base_url, model, instance_id):
    candidates = []
    if instance_id:
        candidates.append(instance_id)
    # Single JIT-loaded instances are usually keyed by the model id itself.
    if model and model not in candidates:
        candidates.append(model)
    for iid in await _lmstudio_loaded_instance_ids(session, base_url, model):
        if iid not in candidates:
            candidates.append(iid)

    last_err = None
    for iid in candidates:
        try:
            async with session.post(
                base_url + "/api/v1/models/unload", json={"instance_id": iid}
            ) as resp:
                if resp.status in (200, 204):
                    return {"loaded": False}
                body = await resp.text()
                # "not loaded" / model_not_found means it's already unloaded —
                # treat that as success so a redundant unload isn't an error.
                low = body.lower()
                if resp.status == 404 and ("not loaded" in low or "not_found" in low):
                    return {"loaded": False, "note": "already unloaded"}
                last_err = "HTTP %s: %s" % (resp.status, body[:200])
        except aiohttp.ClientError as e:
            last_err = str(e)
    raise BackendError("LM Studio unload failed: %s" % (last_err or "no loaded instance found"))


async def _lmstudio_status(session, base_url, model):
    models = await _lmstudio_list_models(session, base_url)
    for m in models:
        if m["id"] == model:
            if m["loaded"] is True:
                return {"state": "loaded", "context_length": m.get("context_length")}
            if m["loaded"] is False:
                return {"state": "not-loaded", "context_length": m.get("context_length")}
    return {"state": "unknown", "context_length": None}


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

async def _ollama_list_models(session, base_url):
    try:
        async with session.get(base_url + "/api/tags") as resp:
            if resp.status != 200:
                raise BackendError("Ollama /api/tags -> HTTP %s" % resp.status)
            data = await resp.json(content_type=None)
    except aiohttp.ClientError as e:
        raise BackendError("Could not reach Ollama at %s (%s)" % (base_url, e))
    out = []
    for it in data.get("models", []):
        name = it.get("name") or it.get("model")
        if name:
            vision = None
            fam = (it.get("details") or {}).get("families") or []
            if any("clip" in str(f).lower() or "vision" in str(f).lower() for f in fam):
                vision = True
            out.append({"id": name, "loaded": None, "context_length": None,
                        "state": None, "vision": vision, "audio": _detect_audio({}, name)})
    return out


async def _ollama_context_length(session, base_url, model):
    try:
        async with session.post(base_url + "/api/show", json={"model": model}) as resp:
            if resp.status != 200:
                return None
            data = await resp.json(content_type=None)
    except aiohttp.ClientError:
        return None
    info = data.get("model_info") or {}
    for k, v in info.items():
        if k.endswith(".context_length"):
            return v
    return None


async def _ollama_chat_stream(session, base_url, model, messages, system_prompt, max_tokens):
    payload = {
        "model": model,
        "messages": _build_ollama_messages(system_prompt, messages),
        "stream": True,
    }
    if max_tokens and max_tokens > 0:
        payload["options"] = {"num_predict": max_tokens}
    try:
        async with session.post(base_url + "/api/chat", json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise BackendError("Ollama chat -> HTTP %s: %s" % (resp.status, body[:300]))
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("error"):
                    yield {"delta": "", "thinking": "", "done": True, "usage": None,
                           "stats": None, "error": obj["error"]}
                    return
                msg = obj.get("message") or {}
                content = msg.get("content") or ""
                thinking = msg.get("thinking") or ""
                done = bool(obj.get("done"))
                usage = None
                stats = None
                if done:
                    prompt_tokens = obj.get("prompt_eval_count", 0)
                    completion_tokens = obj.get("eval_count", 0)
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                    eval_dur = obj.get("eval_duration")
                    if eval_dur and completion_tokens:
                        stats = {
                            "tokens_per_second": completion_tokens / (eval_dur / 1e9),
                            "time_to_first_token": (obj.get("prompt_eval_duration") or 0) / 1e9,
                        }
                if content or thinking or done:
                    yield {"delta": content, "thinking": thinking, "done": done,
                           "usage": usage, "stats": stats, "error": None,
                           "finish_reason": obj.get("done_reason") if done else None}
                if done:
                    return
    except aiohttp.ClientError as e:
        raise BackendError("Ollama chat failed: %s" % e)


async def _ollama_load(session, base_url, model):
    try:
        async with session.post(
            base_url + "/api/generate",
            json={"model": model, "keep_alive": "30m"},
        ) as resp:
            if resp.status == 200:
                await resp.read()
                return {"loaded": True, "instance_id": None, "status": "loaded"}
            body = await resp.text()
            raise BackendError("Ollama load -> HTTP %s: %s" % (resp.status, body[:300]))
    except aiohttp.ClientError as e:
        raise BackendError("Ollama load failed: %s" % e)


async def _ollama_unload(session, base_url, model, instance_id):
    try:
        async with session.post(
            base_url + "/api/generate",
            json={"model": model, "keep_alive": 0},
        ) as resp:
            if resp.status == 200:
                await resp.read()
                return {"loaded": False}
            body = await resp.text()
            raise BackendError("Ollama unload -> HTTP %s: %s" % (resp.status, body[:300]))
    except aiohttp.ClientError as e:
        raise BackendError("Ollama unload failed: %s" % e)


async def _ollama_status(session, base_url, model):
    """Ollama reports running models via /api/ps."""
    try:
        async with session.get(base_url + "/api/ps") as resp:
            if resp.status != 200:
                return {"state": "unknown", "context_length": None}
            data = await resp.json(content_type=None)
    except aiohttp.ClientError:
        return {"state": "unknown", "context_length": None}
    running = {m.get("name") or m.get("model") for m in data.get("models", [])}
    state = "loaded" if model in running else "not-loaded"
    return {"state": state, "context_length": None}


# ---------------------------------------------------------------------------
# Public dispatch interface
# ---------------------------------------------------------------------------

def _new_session():
    return aiohttp.ClientSession(timeout=_TIMEOUT)


async def list_models(backend, base_url):
    if backend == "direct":
        settings = settings_store.load()
        loop = asyncio.get_event_loop()
        # scan_models() walks the filesystem — keep it off the event loop.
        models = await loop.run_in_executor(None, direct_backend.scan_models, settings.get("direct_folders") or [])
        loaded_id = direct_backend.current_model_id()
        return [
            {
                "id": m["id"], "display_name": m["display_name"],
                "loaded": (m["id"] == loaded_id), "state": "loaded" if m["id"] == loaded_id else "not-loaded",
                "context_length": m.get("context_length"),
                "vision": m.get("vision"), "audio": m.get("audio"), "bytes": m.get("bytes"),
                "layer_count": m.get("layer_count"),
            }
            for m in models
        ]
    base_url = _base(base_url, backend)
    async with _new_session() as session:
        if backend == "ollama":
            models = await _ollama_list_models(session, base_url)
            # Best-effort enrich with context length for loaded/known models.
            return models
        return await _lmstudio_list_models(session, base_url)


async def chat_stream(backend, base_url, model, messages, system_prompt, max_tokens=DEFAULT_MAX_TOKENS):
    """Async generator yielding normalized chunks. Manages its own session."""
    if backend == "direct":
        start = time.monotonic()
        yield {"delta": "", "thinking": "", "done": False, "usage": None, "stats": None,
               "error": None, "status": "loading-model", "elapsed": 0}
        # llama-server exposes no real load-percentage — only an honest elapsed-time
        # readout is possible. Run the (potentially slow, first-load) ensure_loaded()
        # as a background task and poll it so the UI can show a live counter instead
        # of one static message for however long loading takes.
        load_task = asyncio.ensure_future(direct_backend.ensure_loaded(model))
        try:
            while not load_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(load_task), timeout=1.0)
                except asyncio.TimeoutError:
                    yield {"delta": "", "thinking": "", "done": False, "usage": None, "stats": None,
                           "error": None, "status": "loading-model",
                           "elapsed": round(time.monotonic() - start, 1)}
            direct_url = load_task.result()
        except direct_backend.DirectBackendError as e:
            yield {"delta": "", "thinking": "", "done": True, "usage": None, "stats": None, "error": str(e)}
            return
        session = _new_session()
        try:
            async for chunk in _lmstudio_chat_stream(session, direct_url, model, messages, system_prompt, max_tokens):
                yield chunk
        finally:
            await session.close()
        return

    base_url = _base(base_url, backend)
    session = _new_session()
    try:
        if backend == "ollama":
            gen = _ollama_chat_stream(session, base_url, model, messages, system_prompt, max_tokens)
        else:
            gen = _lmstudio_chat_stream(session, base_url, model, messages, system_prompt, max_tokens)
        async for chunk in gen:
            yield chunk
    finally:
        await session.close()


async def load_model(backend, base_url, model):
    if backend == "direct":
        try:
            url = await direct_backend.ensure_loaded(model)
        except direct_backend.DirectBackendError as e:
            raise BackendError(str(e))
        return {"loaded": True, "instance_id": None, "status": "loaded", "base_url": url}
    base_url = _base(base_url, backend)
    async with _new_session() as session:
        if backend == "ollama":
            return await _ollama_load(session, base_url, model)
        return await _lmstudio_load(session, base_url, model)


async def unload_model(backend, base_url, model, instance_id=None):
    if backend == "direct":
        await direct_backend.unload()
        return {"loaded": False}
    base_url = _base(base_url, backend)
    async with _new_session() as session:
        if backend == "ollama":
            return await _ollama_unload(session, base_url, model, instance_id)
        return await _lmstudio_unload(session, base_url, model, instance_id)


async def unload_and_confirm(backend, base_url, model, instance_id=None,
                             timeout=10.0, interval=0.4):
    """Unload then poll the model's state until it actually reports not-loaded,
    so callers can sequence a VRAM handoff rather than fire-and-forget. Returns
    {"loaded": False, "confirmed": bool, "warning"?: str}."""
    if backend == "direct":
        # We wait out the process's own termination inside unload() itself, so
        # by the time it returns VRAM release is already confirmed — no polling.
        await direct_backend.unload()
        return {"loaded": False, "confirmed": True}

    base_url = _base(base_url, backend)
    loop = asyncio.get_event_loop()
    async with _new_session() as session:
        if backend == "ollama":
            await _ollama_unload(session, base_url, model, instance_id)
            check = _ollama_status
        else:
            await _lmstudio_unload(session, base_url, model, instance_id)
            check = _lmstudio_status

        deadline = loop.time() + timeout
        while loop.time() < deadline:
            try:
                st = await check(session, base_url, model)
            except BackendError:
                # Server unreachable while polling — can't confirm, but the
                # unload call itself returned, so don't hang.
                return {"loaded": False, "confirmed": False,
                        "warning": "could not confirm unload (status unreachable)"}
            if st.get("state") in ("not-loaded", "unknown", "offline"):
                return {"loaded": False, "confirmed": True}
            await asyncio.sleep(interval)
    return {"loaded": False, "confirmed": False,
            "warning": "unload not confirmed within %.0fs" % timeout}


async def get_status(backend, base_url, model):
    if backend == "direct":
        return direct_backend.status(model)
    base_url = _base(base_url, backend)
    async with _new_session() as session:
        if backend == "ollama":
            return await _ollama_status(session, base_url, model)
        return await _lmstudio_status(session, base_url, model)


async def context_length(backend, base_url, model):
    if backend == "direct":
        m = direct_backend.lookup(model)
        return m.get("context_length") if m else None
    base_url = _base(base_url, backend)
    async with _new_session() as session:
        if backend == "ollama":
            return await _ollama_context_length(session, base_url, model)
        status = await _lmstudio_status(session, base_url, model)
        return status.get("context_length")
