"""
The "Direct model loader" backend — load GGUF models straight from disk,
without LM Studio or Ollama running as a middleman.

The trick to matching LM Studio's speed is that we're not writing our own
inference code: LM Studio's GGUF runtime *is* llama.cpp under the hood, so we
spawn the same engine ourselves — llama.cpp's own `llama-server`, in
single-model mode, with the same techniques that make it fast (mmap'd
weights, full GPU layer offload, flash attention). Once it's up, it speaks
the same OpenAI-compatible /v1/chat/completions API LM Studio does, so
backends.py's existing LM-Studio-style HTTP code handles the actual chat
traffic unchanged — this module's job is only: find models on disk, and
manage the llama-server process's lifecycle.

Three responsibilities, kept in three sections of this file:
  1. GGUF discovery — a tiny pure-python GGUF header parser (reads only the
     metadata section, seeking past tensor data and big arrays like tokenizer
     vocabularies) used to find chat-capable models in user-configured
     folders, pair them with their mmproj (vision/audio projector) file, and
     skip non-chat GGUFs (e.g. ComfyUI's own diffusion-model GGUF checkpoints,
     which live in the same file format but aren't llama.cpp chat models).
  2. Process management — spawn/health-check/terminate a single llama-server
     instance at a time (mirrors the "one model loaded at a time" model this
     whole add-on already assumes for LM Studio/Ollama), tracking enough state
     that repeated loads of the same model/settings are a no-op.
  3. Binary acquisition — download a matching official llama.cpp release build
     straight from GitHub so "git clone the node, click a button" is enough;
     no manual hunting through the releases page for the right zip.
"""

import asyncio
import collections
import os
import platform
import re
import shlex
import shutil
import struct
import tarfile
import tempfile
import time
import zipfile

import aiohttp

from . import settings_store

class DirectBackendError(Exception):
    """Raised for any Direct-loader failure with a user-readable message."""

# ---------------------------------------------------------------------------
# 1. GGUF metadata parsing (header-only — never reads tensor data)
# ---------------------------------------------------------------------------

_GGUF_MAGIC = b"GGUF"
_TYPE_STRING = 8
_TYPE_ARRAY = 9
_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _read_gguf_string(f):
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", "replace")


def _skip_gguf_value(f, vtype):
    if vtype == _TYPE_STRING:
        (length,) = struct.unpack("<Q", f.read(8))
        f.seek(length, 1)
    elif vtype == _TYPE_ARRAY:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if elem_type == _TYPE_STRING:
            for _ in range(count):
                (length,) = struct.unpack("<Q", f.read(8))
                f.seek(length, 1)
        elif elem_type in _SCALAR_SIZES:
            f.seek(_SCALAR_SIZES[elem_type] * count, 1)  # one seek for the whole array
        elif elem_type == _TYPE_ARRAY:
            for _ in range(count):
                _skip_gguf_value(f, _TYPE_ARRAY)  # nested arrays: rare, recurse
        else:
            raise ValueError("unknown gguf array elem type %d" % elem_type)
    elif vtype in _SCALAR_SIZES:
        f.seek(_SCALAR_SIZES[vtype], 1)
    else:
        raise ValueError("unknown gguf value type %d" % vtype)


def _read_gguf_value(f, vtype):
    if vtype == _TYPE_STRING:
        return _read_gguf_string(f)
    if vtype in _SCALAR_FMT:
        (val,) = struct.unpack(_SCALAR_FMT[vtype], f.read(_SCALAR_SIZES[vtype]))
        return val
    if vtype == _TYPE_ARRAY:
        _skip_gguf_value(f, vtype)  # we never need array values, only scalars/strings
        return None
    raise ValueError("unknown gguf value type %d" % vtype)


def read_gguf_meta(path):
    """Parse a GGUF file's metadata header only. Big values we don't care
    about (tokenizer vocab arrays, etc.) are skipped via seek() rather than
    read, so this stays fast even on multi-GB files — we never touch tensor
    data at all. Returns a dict of every 'general.*' / '*.context_length' key
    found, or None if the file isn't a readable GGUF."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != _GGUF_MAGIC:
                return None
            (version,) = struct.unpack("<I", f.read(4))
            if version == 1:
                _tensor_count, kv_count = struct.unpack("<II", f.read(8))
            else:
                _tensor_count, kv_count = struct.unpack("<QQ", f.read(16))
            out = {}
            for _ in range(kv_count):
                key = _read_gguf_string(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                if key.startswith("general.") or key.endswith(".context_length") or key.endswith(".block_count"):
                    out[key] = _read_gguf_value(f, vtype)
                else:
                    _skip_gguf_value(f, vtype)
            return out
    except (OSError, struct.error, ValueError, UnicodeDecodeError, MemoryError):
        return None


# Architectures llama-server can't run as a chat model even though they're
# valid GGUF files — mainly ComfyUI-GGUF's own diffusion-model checkpoints,
# which share the container format but are UNets/VAEs/text-encoders, not LLMs.
# Deliberately a denylist (not an allowlist): llama.cpp's set of supported chat
# architectures is large and keeps growing, so anything NOT in this short,
# stable list of known non-chat archs is assumed usable.
_NON_CHAT_ARCHS = {
    "flux", "flux1", "sd1", "sd2", "sd3", "sdxl", "sdxl_refiner", "chroma",
    "hunyuan", "hunyuandit", "hunyuanvideo", "wan", "wan2", "cogvideo",
    "cogvideox", "ltxv", "auraflow", "pixart", "pixartalpha", "pixartsigma",
    "kolors", "playground", "playgroundv2", "stablecascade", "t5", "t5encoder",
    "clip", "clip_vision", "vae", "controlnet", "unet",
}

_VISION_NAME_HINTS = ("vl", "vision", "llava", "clip", "siglip")
_AUDIO_NAME_HINTS = ("audio", "omni", "ultravox", "voxtral", "moshi", "whisper")
_SHARD_RE = re.compile(r"^(.+)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_MMPROJ_RE = re.compile(r"^mmproj", re.IGNORECASE)


def _classify_modality(*names):
    """Vision/audio guess from filenames (model + its mmproj, if any) — same
    best-effort spirit as backends._detect_vision. Presence of an mmproj file
    at all is a strong signal *something* multimodal is going on even when the
    name gives no hint, so that alone defaults to vision=True (vision mmprojs
    are far more common than audio ones in the wild)."""
    blob = " ".join(str(n) for n in names if n).lower()
    vision = any(h in blob for h in _VISION_NAME_HINTS)
    audio = any(h in blob for h in _AUDIO_NAME_HINTS)
    return vision, audio


def scan_models(folders, max_depth=4):
    """Recursively scan `folders` for chat-usable GGUF models. Returns a list
    of dicts: {id, display_name, path, mmproj, vision, audio, bytes,
    context_length, architecture, folder}, sorted by display name.

    Pairing convention (matches llama.cpp's own): an mmproj-*.gguf file pairs
    with every non-mmproj .gguf model in the *same directory*. Split models
    (name-00001-of-00003.gguf) are represented by their first shard only —
    llama.cpp finds the rest automatically when given that path.
    """
    dirs = {}
    for root_folder in folders or []:
        root_folder = os.path.expanduser(str(root_folder or "").strip())
        if not root_folder or not os.path.isdir(root_folder):
            continue
        root_folder = os.path.normpath(root_folder)
        base_depth = root_folder.count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root_folder):
            depth = dirpath.count(os.sep) - base_depth
            if depth >= max_depth:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            gguf_files = [fn for fn in filenames if fn.lower().endswith(".gguf")]
            if gguf_files:
                dirs.setdefault(dirpath, {"folder": root_folder, "files": []})
                dirs[dirpath]["files"] = gguf_files

    models = []
    for dirpath, info in dirs.items():
        gguf_files = info["files"]
        mmproj_files = sorted(fn for fn in gguf_files if _MMPROJ_RE.match(fn))
        mmproj_path = os.path.join(dirpath, mmproj_files[0]) if mmproj_files else None
        model_files = [fn for fn in gguf_files if fn not in mmproj_files]

        shown = []
        for fn in model_files:
            m = _SHARD_RE.match(fn)
            if m and int(m.group(2)) != 1:
                continue  # secondary shard — represented by the first shard only
            shown.append(fn)

        for fn in shown:
            full_path = os.path.join(dirpath, fn)
            meta = read_gguf_meta(full_path) or {}
            arch = str(meta.get("general.architecture") or "").lower()
            if arch and arch in _NON_CHAT_ARCHS:
                continue
            ctx = None
            layer_count = None
            for k, v in meta.items():
                if k.endswith(".context_length") and isinstance(v, int) and ctx is None:
                    ctx = v
                if k.endswith(".block_count") and isinstance(v, int) and layer_count is None:
                    layer_count = v
            name_vision, name_audio = _classify_modality(fn, mmproj_files[0] if mmproj_files else None)
            vision = name_vision or (bool(mmproj_path) and not name_audio)
            audio = name_audio
            try:
                size = os.path.getsize(full_path)
            except OSError:
                size = 0
            models.append({
                "id": full_path,
                "display_name": meta.get("general.name") or os.path.splitext(fn)[0],
                "path": full_path,
                "mmproj": mmproj_path,
                "vision": vision,
                "audio": audio,
                "bytes": size,
                "context_length": ctx,
                "layer_count": layer_count,
                "architecture": meta.get("general.architecture"),
                "folder": info["folder"],
            })
    models.sort(key=lambda m: m["display_name"].lower())

    global _model_cache
    _model_cache = {m["id"]: m for m in models}
    return models


# Populated by the most recent scan_models() call. load/unload/status only
# receive a model id (its path) from the frontend, so this is how they look
# up the matching mmproj + display info without rescanning the whole tree.
_model_cache = {}


def lookup(model_id):
    return _model_cache.get(model_id)


# ---------------------------------------------------------------------------
# 2. Process management — one llama-server instance at a time
# ---------------------------------------------------------------------------

def _find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def detect_binary():
    """Best-effort search for a llama-server executable: PATH first, then a
    handful of common manual-install locations. There's no reliable way to
    reuse LM Studio's own bundled runtime (its layout varies by version/OS and
    isn't a standalone llama-server binary), so this is intentionally modest —
    most users will need to download a llama.cpp release once and either put
    it on PATH or set the path explicitly in Direct Loader settings."""
    found = shutil.which("llama-server")
    if found:
        return found
    home = os.path.expanduser("~")
    candidates = []
    if os.name == "nt":
        candidates += [
            os.path.join(home, "llama.cpp", "llama-server.exe"),
            os.path.join(home, "llama-server", "llama-server.exe"),
            r"C:\llama.cpp\llama-server.exe",
        ]
    else:
        candidates += [
            os.path.join(home, "llama.cpp", "llama-server"),
            os.path.join(home, ".local", "bin", "llama-server"),
            "/usr/local/bin/llama-server",
            "/opt/homebrew/bin/llama-server",
        ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _build_launch_args(binary, model_path, mmproj_path, port, settings, ngl_override=None):
    args = [binary, "-m", model_path, "--host", "127.0.0.1", "--port", str(port)]
    if mmproj_path:
        args += ["--mmproj", mmproj_path]
    if ngl_override is not None:
        ngl = ngl_override
    else:
        try:
            ngl = int(settings.get("direct_ngl", -1))
        except (TypeError, ValueError):
            ngl = -1
    # -1 ("offload everything") isn't a valid llama.cpp value — 999 is the
    # conventional stand-in ("more layers than any model has").
    args += ["-ngl", "999" if ngl < 0 else str(ngl)]
    try:
        ctx = int(settings.get("direct_context") or 0)
    except (TypeError, ValueError):
        ctx = 0
    args += ["-c", str(ctx)]  # 0 = pull the model's own trained context length
    args += ["--flash-attn", settings.get("direct_flash_attn") or "auto"]
    args += ["--jinja"]  # correct chat-template rendering for modern models
    extra = (settings.get("direct_extra_args") or "").strip()
    if extra:
        args += shlex.split(extra)
    return args


def _signature(model_path, mmproj_path, settings):
    return (
        model_path, mmproj_path, settings.get("direct_binary"), settings.get("direct_ngl"),
        settings.get("direct_context"), settings.get("direct_flash_attn"), settings.get("direct_extra_args"),
    )


def _ngl_ladder(configured_ngl, layer_count):
    """Sequence of -ngl values to try in order. A user-pinned specific layer
    count (not -1 = "auto/max") is respected exactly — we only second-guess
    the ambiguous "just offload everything" default, which is also the one
    case an out-of-memory failure is actually informative about."""
    try:
        configured_ngl = int(configured_ngl)
    except (TypeError, ValueError):
        configured_ngl = -1
    if configured_ngl >= 0:
        return [configured_ngl]
    if layer_count and layer_count > 0:
        ladder = []
        for frac in (1.0, 0.75, 0.5, 0.25, 0.0):
            n = int(round(layer_count * frac))
            if n not in ladder:
                ladder.append(n)
        return ladder
    return [999, 0]  # layer count unknown — try "all", then CPU-only as a last resort


_OOM_MARKERS = (
    "out of memory", "outofdevicememory", "out_of_device_memory", "cuda_error_out_of_memory",
    "failed to allocate", "insufficient memory", "not enough memory", "allocation of size",
)


def _looks_like_oom(text):
    low = (text or "").lower()
    return any(marker in low for marker in _OOM_MARKERS)


_lock = asyncio.Lock()
_current = {
    "proc": None, "path": None, "mmproj": None, "signature": None, "port": None,
    "stderr": collections.deque(maxlen=80), "reader_task": None,
}
_event_log = collections.deque(maxlen=300)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _log(msg):
    """Append to the persistent event log (survives process restarts, unlike
    the per-process stderr buffer) — surfaced in the frontend's Log panel."""
    _event_log.append("[%s] %s" % (time.strftime("%H:%M:%S"), msg))


def get_log():
    proc = _current.get("proc")
    return {
        "events": list(_event_log),
        "stderr": list(_current.get("stderr") or []),
        "running": proc is not None and proc.returncode is None,
    }


async def _drain_stderr(proc, bucket):
    try:
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            text = _ANSI_RE.sub("", line.decode("utf-8", "replace")).rstrip()
            if text:
                bucket.append(text)
    except Exception:
        pass


async def _terminate_current():
    proc = _current.get("proc")
    task = _current.get("reader_task")
    if task:
        task.cancel()
    if proc is not None and proc.returncode is None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=6)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=4)
            except asyncio.TimeoutError:
                pass
    _current.update({"proc": None, "path": None, "mmproj": None, "signature": None,
                      "port": None, "reader_task": None})


async def _health_check(port, timeout):
    deadline = time.monotonic() + timeout
    url = "http://127.0.0.1:%d/health" % port
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
        while time.monotonic() < deadline:
            if _current.get("proc") is not None and _current["proc"].returncode is not None:
                return False  # process died during startup — no point continuing to poll
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        return True
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.35)
    return False


async def _try_spawn(binary, model, port, settings, ngl_value, timeout):
    """One spawn+health-check attempt at a specific -ngl. Returns (ok, stderr_tail)."""
    args = _build_launch_args(binary, model["path"], model["mmproj"], port, settings, ngl_override=ngl_value)
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        raise DirectBackendError("Could not start llama-server (%s): %s" % (binary, e))

    stderr_bucket = collections.deque(maxlen=80)
    _current.update({"proc": proc, "path": model["path"], "mmproj": model["mmproj"],
                      "signature": None, "port": port, "stderr": stderr_bucket})
    _current["reader_task"] = asyncio.ensure_future(_drain_stderr(proc, stderr_bucket))

    ok = await _health_check(port, timeout)
    return ok, "\n".join(stderr_bucket)


async def ensure_loaded(model_id, timeout=180.0):
    """Make sure the requested model is the one currently running behind
    llama-server, spawning/replacing the process if needed (a different model
    already running is stopped first — one model loaded at a time, same as
    the LM Studio/Ollama backends here). Returns the localhost base_url to
    send chat requests to.

    When GPU layers is left at "auto" (-1, the default), a failure that looks
    like a GPU-memory allocation error triggers an automatic retry with fewer
    layers offloaded (100% -> 75% -> 50% -> 25% -> 0%, using the model's known
    layer count from its GGUF metadata), instead of just failing once loaded
    "too big" for the GPU. An explicitly-chosen layer count is never
    second-guessed this way. Raises DirectBackendError with a diagnostic
    message (including a tail of llama-server's stderr) if every attempt fails.
    """
    model = lookup(model_id)
    if not model:
        raise DirectBackendError("Unknown model — rescan (refresh models) and try again: %s" % model_id)

    settings = settings_store.load()
    binary = (settings.get("direct_binary") or "").strip() or detect_binary()
    if not binary:
        raise DirectBackendError(
            "No llama-server binary configured. Click “Download llama-server” in Direct "
            "Loader settings (or set an existing install's path manually)."
        )
    if not os.path.isfile(binary):
        raise DirectBackendError("llama-server binary not found at: %s" % binary)

    sig = _signature(model["path"], model["mmproj"], settings)
    async with _lock:
        proc = _current.get("proc")
        if proc is not None and proc.returncode is None and _current.get("signature") == sig:
            return "http://127.0.0.1:%d" % _current["port"]  # already the right model+settings

        await _terminate_current()

        ladder = _ngl_ladder(settings.get("direct_ngl", -1), model.get("layer_count"))
        _log("Loading %s (trying GPU layers: %s)" % (model["display_name"], ", ".join(str(n) for n in ladder)))
        attempts = []
        for i, ngl_value in enumerate(ladder):
            port = _find_free_port()
            ok, tail = await _try_spawn(binary, model, port, settings, ngl_value, timeout)
            if ok:
                _current["signature"] = sig
                if i > 0:
                    _log("Loaded at -ngl %d after falling back from a larger GPU-layer count that didn't fit." % ngl_value)
                else:
                    _log("Loaded %s." % model["display_name"])
                return "http://127.0.0.1:%d" % port

            exited = _current.get("proc") is not None and _current["proc"].returncode is not None
            await _terminate_current()
            attempts.append((ngl_value, tail))
            is_oom = _looks_like_oom(tail)
            _log("Attempt at -ngl %d failed (%s)%s" % (
                ngl_value, "exited" if exited else "timed out",
                " — looks like a GPU memory shortfall, retrying with fewer layers offloaded." if (is_oom and i < len(ladder) - 1) else "",
            ))
            if not is_oom or i == len(ladder) - 1:
                break  # not an OOM signature, or we're out of rungs — stop retrying

        last_ngl, last_tail = attempts[-1]
        if len(attempts) > 1:
            summary = "Ran out of GPU memory at every offload level tried (%s layers)." % ", ".join(str(a[0]) for a in attempts)
        else:
            reason = "ran out of GPU memory" if _looks_like_oom(last_tail) else "exited before it finished loading"
            summary = "llama-server %s." % reason
        detail = ("\n" + last_tail[-1500:]) if last_tail else ""
        raise DirectBackendError("%s%s" % (summary, detail))


async def unload():
    async with _lock:
        was_running = _current.get("proc") is not None
        await _terminate_current()
        if was_running:
            _log("Unloaded.")


def status(model_id):
    model = lookup(model_id)
    if not model:
        return {"state": "unknown", "context_length": None}
    settings = settings_store.load()
    sig = _signature(model["path"], model["mmproj"], settings)
    proc = _current.get("proc")
    if proc is not None and proc.returncode is None and _current.get("signature") == sig:
        return {"state": "loaded", "context_length": model.get("context_length")}
    return {"state": "not-loaded", "context_length": model.get("context_length")}


def current_model_id():
    """The model id (path) currently running, or None."""
    proc = _current.get("proc")
    if proc is not None and proc.returncode is None:
        return _current.get("path")
    return None


async def shutdown_all():
    """Call on ComfyUI/aiohttp app shutdown so a Muse-spawned llama-server
    never outlives the ComfyUI process it was started for."""
    async with _lock:
        await _terminate_current()


def suggest_folders():
    """A handful of well-known model-folder locations, filtered to ones that
    actually exist on this machine — just a convenience for the settings UI,
    not exhaustive. Deliberately skips Ollama's model store: it keeps models
    as content-addressed blobs, not plain .gguf files, so there's nothing
    scannable there."""
    candidates = []
    home = os.path.expanduser("~")
    candidates.append(os.path.join(home, ".lmstudio", "models"))
    candidates.append(os.path.join(home, ".cache", "lm-studio", "models"))
    try:
        import folder_paths
        models_dir = folder_paths.models_dir
        candidates.append(os.path.join(models_dir, "LLM"))
        candidates.append(os.path.join(models_dir, "gguf"))
    except Exception:
        pass
    out = []
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.isdir(c) and c not in out:
            out.append(c)
    return out


async def detect_gpu_vram():
    """Best-effort VRAM detection via nvidia-smi — covers the large majority
    of ComfyUI users, since it's the one universally-installed inspection
    tool on any machine with a working NVIDIA driver. Returns
    [{"name", "total_mb", "free_mb"}, ...], or [] if nvidia-smi isn't present
    (AMD/Intel/Apple Silicon, or no NVIDIA driver) or the query fails — treat
    an empty list as "unknown", not "no GPU". Used only to suggest a GPU-layer
    count (see "Fit to GPU" in the UI); llama-server's own allocator is the
    actual source of truth, so a rough estimate here is good enough."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        proc = await asyncio.create_subprocess_exec(
            exe, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _err = await asyncio.wait_for(proc.communicate(), timeout=5)
    except Exception:
        return []
    gpus = []
    for line in out.decode("utf-8", "replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        name, total, free = parts
        try:
            gpus.append({"name": name, "total_mb": int(float(total)), "free_mb": int(float(free))})
        except ValueError:
            continue
    return gpus


# ---------------------------------------------------------------------------
# 3. Binary acquisition — download a matching official llama.cpp release
# ---------------------------------------------------------------------------
#
# "git clone and go" means the node has to be able to fetch its own copy of
# llama-server rather than sending everyone to a releases page to guess which
# of ~25 zip files matches their machine. Default choice is the Vulkan build
# on Windows/Linux: it runs on NVIDIA/AMD/Intel GPUs alike with no extra
# runtime package to pair up, unlike the CUDA builds (which need a separately
# downloaded, version-matched `cudart-*` zip extracted alongside them) — so
# it's the one most likely to "just work" for the most people. CUDA/CPU-only
# are offered as explicit opt-in alternatives for anyone who wants them.

_GITHUB_API_LATEST = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
_GITHUB_HEADERS = {"User-Agent": "comfyui-muse", "Accept": "application/vnd.github+json"}
_INSTALL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")


def detect_platform():
    """Returns (os_key, arch_key), e.g. ("win", "x64"), ("macos", "arm64")."""
    sysname = platform.system()
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    if sysname == "Windows":
        return "win", arch
    if sysname == "Darwin":
        return "macos", arch
    return "ubuntu", arch  # llama.cpp only publishes one Linux flavor; works on most glibc distros


def _pick_asset(assets, os_key, arch, variant):
    """Match release assets by filename suffix (tag numbers live in the
    middle of the name, e.g. llama-b10448-bin-win-vulkan-x64.zip, so suffix
    matching is what stays stable release to release). Returns
    (main_asset, cudart_asset_or_None); main_asset is None if nothing matched.
    """
    by_name = {a.get("name", ""): a for a in assets}

    def find(suffix):
        for name, a in by_name.items():
            # cudart-*.zip is a runtime companion package, never the main
            # binary — its name happens to share the same suffix pattern
            # (e.g. both "llama-bNNNN-bin-win-cuda-13.3-x64.zip" and
            # "cudart-llama-bin-win-cuda-13.3-x64.zip" end the same way).
            if name.startswith("cudart"):
                continue
            if name.endswith(suffix):
                return a
        return None

    if os_key == "macos":
        return find("-bin-macos-%s.tar.gz" % arch), None

    if os_key == "win":
        if variant == "cpu":
            return find("-bin-win-cpu-%s.zip" % arch), None
        if variant == "cuda":
            # Prefer the newer CUDA toolkit version; each needs its own
            # matching cudart runtime package extracted into the same folder.
            for ver in (("13.4",) if arch == "arm64" else ("13.3", "12.4")):
                main = find("-bin-win-cuda-%s-%s.zip" % (ver, arch))
                if main:
                    cudart = by_name.get("cudart-llama-bin-win-cuda-%s-%s.zip" % (ver, arch))
                    return main, cudart
            return None, None
        if arch == "arm64":
            return None, None  # no win-vulkan-arm64 build published; caller falls back to manual install
        return find("-bin-win-vulkan-x64.zip"), None

    if os_key == "ubuntu":
        if variant == "cpu":
            return find("-bin-ubuntu-%s.tar.gz" % arch), None
        return find("-bin-ubuntu-vulkan-%s.tar.gz" % arch), None

    return None, None


async def _fetch_json(url):
    async with aiohttp.ClientSession(headers=_GITHUB_HEADERS, timeout=aiohttp.ClientTimeout(total=20)) as s:
        async with s.get(url) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise DirectBackendError("GitHub API error (HTTP %s) fetching the latest llama.cpp release: %s"
                                          % (resp.status, body[:200]))
            return await resp.json()


async def _download_to(url, dest_path):
    async with aiohttp.ClientSession(headers=_GITHUB_HEADERS,
                                      timeout=aiohttp.ClientTimeout(total=None, sock_connect=15)) as s:
        async with s.get(url) as resp:
            if resp.status != 200:
                raise DirectBackendError("Download failed (HTTP %s): %s" % (resp.status, url))
            with open(dest_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(1 << 16):
                    f.write(chunk)


def _extract(archive_path, dest_dir):
    if archive_path.lower().endswith(".zip"):
        with zipfile.ZipFile(archive_path) as z:
            z.extractall(dest_dir)
    else:
        with tarfile.open(archive_path) as t:
            t.extractall(dest_dir)


async def _extract_with_retry(archive_path, dest_dir, attempts=4, delay=1.5):
    """Windows commonly locks a just-downloaded .exe/.dll for a moment while
    antivirus scans it, which surfaces as PermissionError on extract — retry
    a few times before giving up rather than failing on the first race."""
    for i in range(attempts):
        try:
            _extract(archive_path, dest_dir)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            await asyncio.sleep(delay)


def _find_server_binary(root_dir):
    target = "llama-server.exe" if os.name == "nt" else "llama-server"
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if target in filenames:
            return os.path.join(dirpath, target)
    return None


async def download_binary(variant="vulkan"):
    """Download and extract an official llama.cpp release build matching this
    machine (blocking-ish — this awaits the full download, typically tens to
    a couple hundred MB). Returns the path to the resulting llama-server
    executable. `variant` is "vulkan" (default), "cpu", or "cuda" (Windows
    NVIDIA only; also pulls the matching cudart runtime package). Raises
    DirectBackendError with a specific reason on any failure — including "no
    matching asset for your platform", in which case the release page is the
    fallback."""
    os_key, arch = detect_platform()
    _log("Looking up latest llama.cpp release for %s/%s (%s)…" % (os_key, arch, variant))
    release = await _fetch_json(_GITHUB_API_LATEST)
    assets = release.get("assets") or []
    main_asset, cudart_asset = _pick_asset(assets, os_key, arch, variant)
    if not main_asset:
        _log("No matching release asset for %s/%s (%s)." % (os_key, arch, variant))
        raise DirectBackendError(
            "No matching llama.cpp release build found for %s/%s (variant=%s). Grab one "
            "manually from https://github.com/ggml-org/llama.cpp/releases and set its path "
            "below instead." % (os_key, arch, variant)
        )

    # A currently-running llama-server we manage may hold its own executable
    # open (Windows won't let us overwrite a locked file) — a fresh download
    # supersedes whatever's loaded anyway, so release it first.
    await unload()

    tag = release.get("tag_name", "latest")
    os.makedirs(_INSTALL_ROOT, exist_ok=True)
    dest_dir = os.path.join(_INSTALL_ROOT, "%s-%s-%s-%s" % (tag, os_key, arch, variant))

    # Extract into a scratch directory first, then swap it into place at the
    # end. A failed/partial attempt then never leaves a half-extracted copy
    # sitting at the deterministic dest_dir for a retry to collide with.
    tmp_dir = tempfile.mkdtemp(dir=_INSTALL_ROOT, prefix=".dl-")
    try:
        for asset in (main_asset, cudart_asset):
            if not asset:
                continue
            _log("Downloading %s…" % asset["name"])
            archive_path = os.path.join(tmp_dir, asset["name"])
            await _download_to(asset["browser_download_url"], archive_path)
            try:
                await _extract_with_retry(archive_path, tmp_dir)
            except PermissionError as e:
                raise DirectBackendError(
                    "Permission denied extracting %s (%s). This is usually antivirus briefly "
                    "locking a freshly-downloaded file, or another Muse-managed llama-server "
                    "still running — try again in a few seconds; unload any loaded model first "
                    "if it keeps happening." % (asset["name"], e)
                )
            try:
                os.remove(archive_path)
            except OSError:
                pass

        binary = _find_server_binary(tmp_dir)
        if not binary:
            _log("Downloaded %s but couldn't find llama-server inside it." % main_asset["name"])
            raise DirectBackendError("Downloaded %s but couldn't find llama-server inside it." % main_asset["name"])
        if os.name != "nt":
            try:
                os.chmod(binary, os.stat(binary).st_mode | 0o111)
            except OSError:
                pass

        binary_rel = os.path.relpath(binary, tmp_dir)
        if os.path.exists(dest_dir):
            try:
                shutil.rmtree(dest_dir)
            except OSError as e:
                raise DirectBackendError(
                    "Couldn't replace the existing install at %s (%s). Unload any loaded "
                    "model in Muse and try again, or delete that folder manually." % (dest_dir, e)
                )
        try:
            os.replace(tmp_dir, dest_dir)
        except OSError as e:
            raise DirectBackendError("Couldn't finish installing to %s: %s" % (dest_dir, e))
        binary = os.path.join(dest_dir, binary_rel)
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    _log("Installed llama-server (%s, %s) at %s" % (tag, variant, binary))
    return binary
