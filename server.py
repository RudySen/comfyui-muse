"""
aiohttp route handlers for ComfyUI-Muse, registered on PromptServer.instance.routes
under the /muse/ prefix. Imported at node load time so registration runs at startup.
"""

import asyncio
import json

from aiohttp import web

from . import backends, chat_store, input_store, direct_backend, settings_store

try:
    from server import PromptServer
    routes = PromptServer.instance.routes
except Exception:  # pragma: no cover - only happens outside a running ComfyUI
    PromptServer = None
    routes = web.RouteTableDef()


async def _on_app_cleanup(app):
    # Make sure a Muse-spawned llama-server never outlives this ComfyUI process.
    try:
        await direct_backend.shutdown_all()
    except Exception:
        pass


try:
    if PromptServer is not None and getattr(PromptServer.instance, "app", None) is not None:
        PromptServer.instance.app.on_cleanup.append(_on_app_cleanup)
except Exception:
    pass


def _err(message, status=502):
    return web.json_response({"error": str(message)}, status=status)


async def free_comfy_vram():
    """Unload ComfyUI's own models and clear its cache to hand VRAM to the LLM.

    Runs the (blocking, GPU-touching) torch calls in a thread so the event loop
    isn't stalled. Degrades silently if not running inside ComfyUI. Returns an
    error string on failure, else None.
    """
    loop = asyncio.get_event_loop()

    def _do():
        try:
            import comfy.model_management as mm
        except Exception:
            return "comfy.model_management unavailable"
        try:
            mm.unload_all_models()
            mm.soft_empty_cache(True)
        except Exception as e:
            return str(e)
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        return None

    try:
        return await loop.run_in_executor(None, _do)
    except Exception as e:
        return str(e)


async def _body(request):
    try:
        return await request.json()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

@routes.get("/muse/models")
async def muse_models(request):
    backend = request.query.get("backend", "lmstudio")
    base_url = request.query.get("base_url", "")
    try:
        models = await backends.list_models(backend, base_url)
        return web.json_response({"models": models})
    except backends.BackendError as e:
        return _err(e)
    except Exception as e:
        return _err("Unexpected error listing models: %s" % e, status=500)


@routes.post("/muse/load")
async def muse_load(request):
    body = await _body(request)
    backend = body.get("backend", "lmstudio")
    base_url = body.get("base_url", "")
    model = body.get("model")
    if not model:
        return _err("No model specified", status=400)
    try:
        result = await backends.load_model(backend, base_url, model)
        return web.json_response(result)
    except backends.BackendError as e:
        return _err(e)
    except Exception as e:
        return _err("Unexpected error loading model: %s" % e, status=500)


@routes.post("/muse/unload")
async def muse_unload(request):
    body = await _body(request)
    backend = body.get("backend", "lmstudio")
    base_url = body.get("base_url", "")
    model = body.get("model")
    instance_id = body.get("model_instance_id")
    # confirm=true polls until the model actually reports not-loaded (VRAM freed).
    confirm = body.get("confirm", False)
    if not model:
        return _err("No model specified", status=400)
    try:
        if confirm:
            result = await backends.unload_and_confirm(backend, base_url, model, instance_id)
        else:
            result = await backends.unload_model(backend, base_url, model, instance_id)
        return web.json_response(result)
    except backends.BackendError as e:
        return _err(e)
    except Exception as e:
        return _err("Unexpected error unloading model: %s" % e, status=500)


@routes.get("/muse/status")
async def muse_status(request):
    backend = request.query.get("backend", "lmstudio")
    base_url = request.query.get("base_url", "")
    model = request.query.get("model", "")
    try:
        result = await backends.get_status(backend, base_url, model)
        return web.json_response(result)
    except backends.BackendError as e:
        return web.json_response({"state": "offline", "error": str(e)})
    except Exception as e:
        return web.json_response({"state": "unknown", "error": str(e)})


# ---------------------------------------------------------------------------
# Chat streaming (SSE)
# ---------------------------------------------------------------------------

@routes.post("/muse/chat/stream")
async def muse_chat_stream(request):
    body = await _body(request)
    backend = body.get("backend", "lmstudio")
    base_url = body.get("base_url", "")
    model = body.get("model")
    system_prompt = body.get("system_prompt", "")
    messages = body.get("messages", [])
    free_vram = body.get("free_comfy_vram", True)
    max_tokens = body.get("max_tokens", backends.DEFAULT_MAX_TOKENS)
    guides = body.get("guides", [])
    video_fps = body.get("video_fps") or input_store.DEFAULT_VIDEO_FPS
    video_max_frames = body.get("video_max_frames") or input_store.DEFAULT_VIDEO_MAX_FRAMES

    if not model:
        return _err("No model specified", status=400)

    # Append active Guide Materials (read fresh from input/) to the system prompt.
    guide_text = input_store.assemble_guides(guides)
    if guide_text:
        system_prompt = (system_prompt + "\n\n" + guide_text).strip() if system_prompt else guide_text

    # Resolve per-message image/video references (filenames) into base64 for
    # the backend. Video frames are re-sampled fresh from disk on every send
    # (no cache), same as guides/images — keeps chat JSON small and picks up
    # edits to the source file automatically.
    resolved = []
    attachment_warnings = []
    for m in messages:
        msg = {"role": m.get("role", "user"), "content": m.get("content", "")}
        imgs = m.get("images") or []
        if imgs:
            loaded_imgs = []
            for name in imgs:
                got = input_store.read_image_b64(name)
                if got:
                    loaded_imgs.append({"data": got[0], "mime": got[1]})
            if loaded_imgs:
                msg["images"] = loaded_imgs
        vids = m.get("videos") or []
        if vids:
            loaded_vids = []
            for name in vids:
                result, err = input_store.extract_frames(name, video_fps, video_max_frames)
                if result:
                    loaded_vids.append({
                        "name": name,
                        "duration": result["duration"],
                        "fps": result["fps"],
                        "frames": result["frames"],
                    })
                else:
                    attachment_warnings.append("%s: %s" % (name, err))
            if loaded_vids:
                msg["video_frames"] = loaded_vids
        auds = m.get("audio") or []
        if auds:
            loaded_auds = []
            for name in auds:
                got = input_store.read_audio_b64(name)
                if got:
                    loaded_auds.append({"name": name, "data": got[0], "format": got[1]})
                else:
                    attachment_warnings.append("%s: audio file not found" % name)
            if loaded_auds:
                msg["audio"] = loaded_auds
        resolved.append(msg)
    messages = resolved

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    async def send(obj):
        await response.write(("data: " + json.dumps(obj) + "\n\n").encode("utf-8"))

    if attachment_warnings:
        await send({"delta": "", "thinking": "", "done": False, "usage": None,
                    "stats": None, "error": None,
                    "attachment_warning": "Attachment problem — " + "; ".join(attachment_warnings)})

    # Free ComfyUI's VRAM BEFORE the request reaches LM Studio/Ollama, so the LLM
    # loads into the freed memory rather than fighting ComfyUI for it.
    if free_vram:
        await send({"delta": "", "thinking": "", "done": False, "usage": None,
                    "stats": None, "error": None, "status": "freeing-vram"})
        await free_comfy_vram()

    try:
        async for chunk in backends.chat_stream(
            backend, base_url, model, messages, system_prompt, max_tokens
        ):
            await send(chunk)
    except backends.BackendError as e:
        await send({"delta": "", "thinking": "", "done": True,
                    "usage": None, "stats": None, "error": str(e)})
    except ConnectionResetError:
        # Client aborted (Stop button) — nothing to report.
        return response
    except Exception as e:
        await send({"delta": "", "thinking": "", "done": True,
                    "usage": None, "stats": None, "error": "Server error: %s" % e})
    try:
        await response.write(b"data: [DONE]\n\n")
    except (ConnectionResetError, RuntimeError):
        pass
    return response


# ---------------------------------------------------------------------------
# Chat persistence CRUD
# ---------------------------------------------------------------------------

@routes.get("/muse/chats")
async def muse_list_chats(request):
    return web.json_response({"chats": chat_store.list_chats()})


@routes.get("/muse/chats/{chat_id}")
async def muse_get_chat(request):
    chat_id = request.match_info["chat_id"]
    data = chat_store.load_chat(chat_id)
    if data is None:
        return _err("Chat not found", status=404)
    return web.json_response(data)


@routes.post("/muse/chats")
async def muse_create_chat(request):
    body = await _body(request)
    data = chat_store.create_chat(
        backend=body.get("backend", "lmstudio"),
        base_url=body.get("base_url", ""),
        model=body.get("model"),
        system_prompt=body.get("system_prompt", ""),
        title=body.get("title", "New chat"),
    )
    return web.json_response(data)


@routes.put("/muse/chats/{chat_id}")
async def muse_update_chat(request):
    chat_id = request.match_info["chat_id"]
    body = await _body(request)
    try:
        data = chat_store.save_chat(chat_id, body)
        return web.json_response(data)
    except ValueError as e:
        return _err(e, status=400)


@routes.delete("/muse/chats/{chat_id}")
async def muse_delete_chat(request):
    chat_id = request.match_info["chat_id"]
    ok = chat_store.delete_chat(chat_id)
    if not ok:
        return _err("Chat not found", status=404)
    return web.json_response({"deleted": True})


# ---------------------------------------------------------------------------
# ComfyUI input/ access (Guide Materials + image attachments)
# ---------------------------------------------------------------------------

@routes.get("/muse/input/guides")
async def muse_input_guides(request):
    return web.json_response({"guides": input_store.list_guides()})


@routes.get("/muse/input/images")
async def muse_input_images(request):
    return web.json_response({"images": input_store.list_images()})


@routes.post("/muse/input/save-image")
async def muse_save_image(request):
    """Save a dropped image into ComfyUI/input/ with collision-safe naming.
    Filename comes from the ?filename= query param; body is the raw bytes."""
    filename = request.query.get("filename", "image.png")
    try:
        data = await request.read()
        if not data:
            return _err("Empty file", status=400)
        name = input_store.save_image(filename, data)
        return web.json_response({"name": name})
    except Exception as e:
        return _err("Could not save image: %s" % e, status=500)


@routes.get("/muse/input/videos")
async def muse_input_videos(request):
    return web.json_response({"videos": input_store.list_videos()})


@routes.post("/muse/input/save-video")
async def muse_save_video(request):
    """Save a dropped video into ComfyUI/input/ with collision-safe naming.
    Filename comes from the ?filename= query param; body is the raw bytes."""
    filename = request.query.get("filename", "video.mp4")
    try:
        data = await request.read()
        if not data:
            return _err("Empty file", status=400)
        name = input_store.save_video(filename, data)
        return web.json_response({"name": name})
    except Exception as e:
        return _err("Could not save video: %s" % e, status=500)


@routes.get("/muse/input/audio")
async def muse_input_audio(request):
    return web.json_response({"audio": input_store.list_audio()})


@routes.post("/muse/input/save-audio")
async def muse_save_audio(request):
    """Save a dropped audio file into ComfyUI/input/ with collision-safe naming.
    Filename comes from the ?filename= query param; body is the raw bytes."""
    filename = request.query.get("filename", "audio.wav")
    try:
        data = await request.read()
        if not data:
            return _err("Empty file", status=400)
        name = input_store.save_audio(filename, data)
        return web.json_response({"name": name})
    except Exception as e:
        return _err("Could not save audio: %s" % e, status=500)


# ---------------------------------------------------------------------------
# Direct model loader (GGUF folders + llama-server settings)
# ---------------------------------------------------------------------------

@routes.get("/muse/direct/settings")
async def muse_direct_get_settings(request):
    return web.json_response(settings_store.load())


@routes.post("/muse/direct/settings")
async def muse_direct_save_settings(request):
    body = await _body(request)
    return web.json_response(settings_store.save(body))


@routes.get("/muse/direct/detect-binary")
async def muse_direct_detect_binary(request):
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(None, direct_backend.detect_binary)
    return web.json_response({"path": path})


@routes.get("/muse/direct/suggest-folders")
async def muse_direct_suggest_folders(request):
    loop = asyncio.get_event_loop()
    folders = await loop.run_in_executor(None, direct_backend.suggest_folders)
    return web.json_response({"folders": folders})


@routes.get("/muse/direct/platform-info")
async def muse_direct_platform_info(request):
    os_key, arch = direct_backend.detect_platform()
    return web.json_response({"os": os_key, "arch": arch})


@routes.post("/muse/direct/download-binary")
async def muse_direct_download_binary(request):
    """Download+extract a matching llama.cpp release build and point Direct
    Loader settings at it. Blocks for the duration of the download (typically
    tens of seconds to a few minutes) rather than streaming progress — kept
    simple since this only runs on an explicit user click."""
    body = await _body(request)
    variant = body.get("variant") or "vulkan"
    try:
        binary = await direct_backend.download_binary(variant)
    except direct_backend.DirectBackendError as e:
        return _err(e, status=502)
    except Exception as e:
        return _err("Download failed: %s" % e, status=500)
    settings = settings_store.save({"direct_binary": binary})
    return web.json_response({"path": binary, "settings": settings})


@routes.get("/muse/direct/gpu-info")
async def muse_direct_gpu_info(request):
    gpus = await direct_backend.detect_gpu_vram()
    return web.json_response({"gpus": gpus})


@routes.get("/muse/direct/log")
async def muse_direct_log(request):
    return web.json_response(direct_backend.get_log())
