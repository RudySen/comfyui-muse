"""
Backend connectors for ComfyUI-Muse.

Backend-agnostic async functions for talking to LM Studio and Ollama. Everything
returns *normalized* data structures so server.py never has to special-case which
backend is active beyond passing the backend name through.

Normalized shapes
-----------------
model:  {"id": str, "loaded": bool|None, "context_length": int|None, "state": str|None}
chunk:  {"delta": str, "thinking": str, "done": bool,
         "usage": {"prompt_tokens", "completion_tokens", "total_tokens"} | None,
         "stats": {"tokens_per_second", "time_to_first_token"} | None,
         "error": str | None}

Zero extra dependencies: only aiohttp (already a ComfyUI dependency) + stdlib.
"""

import asyncio
import json

import aiohttp

LMSTUDIO_DEFAULT_URL = "http://localhost:1234"
OLLAMA_DEFAULT_URL = "http://localhost:11434"

DEFAULT_MAX_TOKENS = 2048

DEFAULT_URLS = {
    "lmstudio": LMSTUDIO_DEFAULT_URL,
    "ollama": OLLAMA_DEFAULT_URL,
}

# Generous timeout: model load + first token on a cold model can take a while.
_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)


class BackendError(Exception):
    """Raised for any backend communication failure with a user-readable message."""


def default_url(backend):
    return DEFAULT_URLS.get(backend, LMSTUDIO_DEFAULT_URL)


def _base(base_url, backend):
    return (base_url or default_url(backend)).strip().rstrip("/")


def _build_openai_messages(system_prompt, messages):
    """OpenAI/LM Studio-style messages. Messages may carry resolved `images`
    (list of {"data": b64, "mime": str}) -> rendered as image_url content parts."""
    out = []
    if system_prompt and system_prompt.strip():
        out.append({"role": "system", "content": system_prompt})
    for m in messages:
        role = m.get("role", "user")
        if role not in ("user", "assistant"):
            continue
        text = m.get("content", "") or ""
        images = m.get("images") or []
        if images and role == "user":
            parts = [{"type": "text", "text": text}]
            for img in images:
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": "data:%s;base64,%s" % (img["mime"], img["data"])},
                })
            out.append({"role": role, "content": parts})
        else:
            out.append({"role": role, "content": text})
    return out


def _build_ollama_messages(system_prompt, messages):
    """Ollama-style messages. Images go in a separate `images` array of base64."""
    out = []
    if system_prompt and system_prompt.strip():
        out.append({"role": "system", "content": system_prompt})
    for m in messages:
        role = m.get("role", "user")
        if role not in ("user", "assistant"):
            continue
        msg = {"role": role, "content": m.get("content", "") or ""}
        images = m.get("images") or []
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
                        "state": None, "vision": vision})
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
    base_url = _base(base_url, backend)
    async with _new_session() as session:
        if backend == "ollama":
            models = await _ollama_list_models(session, base_url)
            # Best-effort enrich with context length for loaded/known models.
            return models
        return await _lmstudio_list_models(session, base_url)


async def chat_stream(backend, base_url, model, messages, system_prompt, max_tokens=DEFAULT_MAX_TOKENS):
    """Async generator yielding normalized chunks. Manages its own session."""
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
    base_url = _base(base_url, backend)
    async with _new_session() as session:
        if backend == "ollama":
            return await _ollama_load(session, base_url, model)
        return await _lmstudio_load(session, base_url, model)


async def unload_model(backend, base_url, model, instance_id=None):
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
    base_url = _base(base_url, backend)
    async with _new_session() as session:
        if backend == "ollama":
            return await _ollama_status(session, base_url, model)
        return await _lmstudio_status(session, base_url, model)


async def context_length(backend, base_url, model):
    base_url = _base(base_url, backend)
    async with _new_session() as session:
        if backend == "ollama":
            return await _ollama_context_length(session, base_url, model)
        status = await _lmstudio_status(session, base_url, model)
        return status.get("context_length")
