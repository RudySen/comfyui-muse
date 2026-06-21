"""
Read/write helpers scoped to ComfyUI's `input/` directory, for Guide Materials
(text references) and image attachments. Everything is reference-based: we store
filenames in chat JSON and read content live from input/ when needed.

All access is locked to the top level of input/ (no traversal, no subfolders for
v1) via basename sanitisation.
"""

import base64
import os

GUIDE_EXTS = (".txt", ".md", ".json")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")

_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def input_dir():
    """Resolve ComfyUI's input directory, with a relative fallback."""
    try:
        import folder_paths
        return folder_paths.get_input_directory()
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        # custom_nodes/comfyui-muse/ -> ComfyUI/input
        return os.path.normpath(os.path.join(here, "..", "..", "input"))


def _safe_path(name):
    """Return an absolute path inside input/ for a bare filename, or None if the
    name tries to escape the directory."""
    base = os.path.basename(str(name))
    if not base or base != str(name):
        return None
    return os.path.join(input_dir(), base)


def _list(exts):
    d = input_dir()
    out = []
    try:
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(exts) and os.path.isfile(os.path.join(d, fn)):
                out.append(fn)
    except OSError:
        pass
    return out


# ---- guides -------------------------------------------------------------

def list_guides():
    """Return [{"name", "bytes"}] for guide files, so the UI can estimate token
    cost (~bytes/4) without reading full contents."""
    d = input_dir()
    out = []
    try:
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if fn.lower().endswith(GUIDE_EXTS) and os.path.isfile(p):
                try:
                    size = os.path.getsize(p)
                except OSError:
                    size = 0
                out.append({"name": fn, "bytes": size})
    except OSError:
        pass
    return out


def read_guide(name):
    path = _safe_path(name)
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def assemble_guides(names):
    """Concatenate active guide files into a delimited system-context string.
    Missing files are skipped. Returns "" if nothing usable."""
    parts = []
    for name in names or []:
        text = read_guide(name)
        if text is None:
            continue
        parts.append("--- Guide: %s ---\n%s" % (os.path.basename(str(name)), text.strip()))
    if not parts:
        return ""
    return "\n\n".join(parts)


# ---- images -------------------------------------------------------------

def list_images():
    return _list(IMAGE_EXTS)


def image_exists(name):
    path = _safe_path(name)
    return bool(path and os.path.isfile(path))


def read_image_b64(name):
    """Return (base64_str, mime) for an input image, or None if missing."""
    path = _safe_path(name)
    if not path or not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    mime = _MIME.get(ext, "image/png")
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return data, mime
    except OSError:
        return None


def save_image(filename, data_bytes):
    """Write dropped image bytes into input/ with collision-safe naming.
    Returns the final filename actually written."""
    d = input_dir()
    os.makedirs(d, exist_ok=True)
    base = os.path.basename(str(filename)) or "image.png"
    stem, ext = os.path.splitext(base)
    if ext.lower() not in IMAGE_EXTS:
        ext = ".png"
    candidate = stem + ext
    i = 1
    while os.path.exists(os.path.join(d, candidate)):
        candidate = "%s_%d%s" % (stem, i, ext)
        i += 1
    with open(os.path.join(d, candidate), "wb") as f:
        f.write(data_bytes)
    return candidate
