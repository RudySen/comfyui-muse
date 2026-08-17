"""
Persisted settings for the Muse "Direct model loader" backend: which folders
to scan for GGUF models, the llama-server binary to launch, and default
launch options (GPU offload, context size, flash attention, extra args).

Stored as a single JSON file next to chats/ — same atomic-write pattern as
chat_store.py, just one file instead of one-per-chat since this is global
config, not per-conversation.
"""

import json
import os
import tempfile

_DIR = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(_DIR, "muse_settings.json")

_DEFAULTS = {
    "direct_folders": [],       # absolute paths to scan for .gguf models
    "direct_binary": "",        # path to a llama-server (or llama-server.exe) executable
    "direct_ngl": -1,           # GPU layers to offload; -1 = every layer, auto-fallback if that OOMs
    "direct_context": 8192,     # KV cache size. NOT 0/"model max" by default — modern models often
                                 # advertise 128k-262k context, and a KV cache sized for that is what
                                 # actually pushes VRAM past what fits, causing silent (slow!) GPU
                                 # memory oversubscription rather than a clean failure. 8192 is a much
                                 # safer default for ordinary chat; raise it deliberately if you need more.
    "direct_flash_attn": "auto",  # "auto" | "on" | "off"
    "direct_extra_args": "",    # advanced: extra raw CLI args appended verbatim
}


def _atomic_write(data):
    os.makedirs(_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _PATH)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def load():
    """Always returns every key in _DEFAULTS, filling in anything missing or
    stale from an older version of this file."""
    if not os.path.exists(_PATH):
        return dict(_DEFAULTS)
    try:
        with open(_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return dict(_DEFAULTS)
    if not isinstance(data, dict):
        return dict(_DEFAULTS)
    out = dict(_DEFAULTS)
    out.update({k: v for k, v in data.items() if k in _DEFAULTS})
    return out


def save(partial):
    """Merge `partial` (any subset of _DEFAULTS keys) into the persisted
    settings and write it out. Returns the full resulting settings dict."""
    data = load()
    data.update({k: v for k, v in (partial or {}).items() if k in _DEFAULTS})
    _atomic_write(data)
    return data
