"""
Filesystem-backed chat persistence for ComfyUI-Muse.

One JSON file per chat session under ./chats/<uuid>.json. Writes are atomic
(temp file + os.replace) so a crash mid-write can't corrupt an existing chat.
"""

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone

_CHATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chats")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir():
    os.makedirs(_CHATS_DIR, exist_ok=True)


def _path(chat_id):
    # Guard against path traversal: only allow the bare uuid hex/hyphen charset.
    safe = "".join(c for c in str(chat_id) if c.isalnum() or c in "-_")
    if not safe:
        raise ValueError("invalid chat id")
    return os.path.join(_CHATS_DIR, safe + ".json")


def _atomic_write(path, data):
    _ensure_dir()
    fd, tmp = tempfile.mkstemp(dir=_CHATS_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_chats():
    """Lightweight listing: id, title, model, backend, updated_at — no messages."""
    _ensure_dir()
    out = []
    for name in os.listdir(_CHATS_DIR):
        if not name.endswith(".json"):
            continue
        try:
            data = _read(os.path.join(_CHATS_DIR, name))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({
            "id": data.get("id", name[:-5]),
            "title": data.get("title", "Untitled chat"),
            "model": data.get("model"),
            "backend": data.get("backend"),
            "updated_at": data.get("updated_at"),
            "message_count": len(data.get("messages", [])),
        })
    out.sort(key=lambda c: c.get("updated_at") or "", reverse=True)
    return out


def load_chat(chat_id):
    path = _path(chat_id)
    if not os.path.exists(path):
        return None
    return _read(path)


def create_chat(backend="lmstudio", base_url="", model=None, system_prompt="", title="New chat"):
    chat_id = uuid.uuid4().hex
    now = _now()
    data = {
        "id": chat_id,
        "title": title,
        "backend": backend,
        "base_url": base_url,
        "model": model,
        "model_instance_id": None,
        "system_prompt": system_prompt,
        "guides": [],
        "max_tokens": 2048,
        "video_fps": 1.0,
        "video_max_frames": 24,
        "messages": [],
        "created_at": now,
        "updated_at": now,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    _atomic_write(_path(chat_id), data)
    return data


def save_chat(chat_id, data):
    """Full overwrite. Forces id and bumps updated_at."""
    existing = load_chat(chat_id)
    data = dict(data)
    data["id"] = chat_id
    if existing and existing.get("created_at"):
        data.setdefault("created_at", existing["created_at"])
    data.setdefault("created_at", _now())
    data["updated_at"] = _now()
    data.setdefault("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    data.setdefault("messages", [])
    _atomic_write(_path(chat_id), data)
    return data


def rename_chat(chat_id, new_title):
    data = load_chat(chat_id)
    if not data:
        return None
    data["title"] = new_title
    data["updated_at"] = _now()
    _atomic_write(_path(chat_id), data)
    return data


def delete_chat(chat_id):
    path = _path(chat_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
