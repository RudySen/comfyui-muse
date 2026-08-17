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
# Broad set of stills-container formats. Anything Pillow can open gets
# normalized to PNG before it's sent to a backend (see read_image_b64), so this
# list is mostly about what shows up in the attach picker / drag-drop filter —
# it's fine to be generous here.
IMAGE_EXTS = (
    ".png", ".jpg", ".jpeg", ".jfif", ".webp", ".gif", ".bmp",
    ".tif", ".tiff", ".ico", ".ppm", ".pgm", ".pbm", ".heic", ".heif", ".avif",
)
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v")
# Passed through as raw bytes (no server-side transcoding) — llama.cpp's mtmd
# audio path (and OpenAI's input_audio) decode common containers themselves.
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus", ".wma")

_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jfif": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".ico": "image/x-icon",
    ".ppm": "image/x-portable-pixmap",
    ".pgm": "image/x-portable-graymap",
    ".pbm": "image/x-portable-bitmap",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".avif": "image/avif",
}
# Formats most vision-model APIs actually accept over the wire. Anything else
# gets transcoded to PNG in read_image_b64 rather than sent as-is.
_WIRE_SAFE_MIME = {"image/png", "image/jpeg", "image/webp", "image/gif"}


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
    """Return (base64_str, mime) for an input image, or None if missing.

    Any format Pillow can open (tiff, bmp, ico, heic/avif if a plugin is
    present, etc.) is transcoded to PNG so it's always something the backend's
    vision API will actually accept — most only document png/jpeg/webp/gif.
    Formats already in that wire-safe set are passed through untouched (no
    quality loss from a round-trip). Falls back to raw bytes if Pillow is
    unavailable or fails to open the file, so already-common formats still
    work even without Pillow.
    """
    path = _safe_path(name)
    if not path or not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    mime = _MIME.get(ext, "image/png")

    if mime not in _WIRE_SAFE_MIME:
        try:
            from PIL import Image
            import io
            with Image.open(path) as im:
                im.load()
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("ascii"), "image/png"
        except Exception:
            pass  # fall through to raw bytes below

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


# ---- videos ---------------------------------------------------------------
#
# Videos are attached by reference (filename), like images and guides. There's
# no persisted frame cache: every time a video is actually sent to a model,
# extract_frames() re-samples it fresh from disk at the chat's configured
# fps/max-frames, the same "read live from input/" philosophy as everything
# else in this module.

DEFAULT_VIDEO_FPS = 1.0
DEFAULT_VIDEO_MAX_FRAMES = 24
# Long side a sampled frame gets downscaled to before JPEG-encoding — keeps
# per-frame payload small since a single video message can carry dozens of
# these plus the previous chat history.
_FRAME_MAX_DIM = 768


def list_videos():
    """Return [{"name", "bytes"}] for video files in input/."""
    d = input_dir()
    out = []
    try:
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if fn.lower().endswith(VIDEO_EXTS) and os.path.isfile(p):
                try:
                    size = os.path.getsize(p)
                except OSError:
                    size = 0
                out.append({"name": fn, "bytes": size})
    except OSError:
        pass
    return out


def video_exists(name):
    path = _safe_path(name)
    return bool(path and os.path.isfile(path))


def save_video(filename, data_bytes):
    """Write a dropped video into input/ with collision-safe naming."""
    d = input_dir()
    os.makedirs(d, exist_ok=True)
    base = os.path.basename(str(filename)) or "video.mp4"
    stem, ext = os.path.splitext(base)
    if ext.lower() not in VIDEO_EXTS:
        ext = ".mp4"
    candidate = stem + ext
    i = 1
    while os.path.exists(os.path.join(d, candidate)):
        candidate = "%s_%d%s" % (stem, i, ext)
        i += 1
    with open(os.path.join(d, candidate), "wb") as f:
        f.write(data_bytes)
    return candidate


def extract_frames(name, fps=DEFAULT_VIDEO_FPS, max_frames=DEFAULT_VIDEO_MAX_FRAMES):
    """Sample frames from a video at roughly `fps`, capped at `max_frames`
    (evenly subsampled across the full duration if there'd be more than that,
    so long videos still get temporal coverage instead of just the first N
    seconds). Each frame is JPEG-encoded and base64'd, tagged with its
    timestamp in seconds — the shape vision models expect for "here's what
    happens over time" prompting.

    Returns (result, error). `result` is None on failure, else
    {"frames": [{"data", "mime", "timestamp"}], "duration": float, "fps": float}.
    `error` is a user-readable string on failure, else None.
    """
    path = _safe_path(name)
    if not path or not os.path.isfile(path):
        return None, "video not found: %s" % name

    try:
        import cv2
    except ImportError:
        return None, (
            "Reading video frames needs OpenCV, which isn't installed in this "
            "ComfyUI environment. Run `pip install opencv-python` (or "
            "opencv-python-headless) in ComfyUI's Python environment and "
            "restart."
        )

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None, "could not open video: %s" % name

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if video_fps <= 0:
            video_fps = 30.0  # sane fallback for odd containers that don't report fps
        duration = (total_frames / video_fps) if total_frames else 0.0

        fps = fps if fps and fps > 0 else DEFAULT_VIDEO_FPS
        step = max(1, round(video_fps / fps))
        indices = list(range(0, max(total_frames, 1), step)) or [0]

        if max_frames and len(indices) > max_frames:
            if max_frames <= 1:
                indices = [indices[0]]
            else:
                n = len(indices)
                indices = sorted({
                    indices[round(i * (n - 1) / (max_frames - 1))]
                    for i in range(max_frames)
                })

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if max(h, w) > _FRAME_MAX_DIM:
                scale = _FRAME_MAX_DIM / float(max(h, w))
                frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok2:
                continue
            frames.append({
                "data": base64.b64encode(buf.tobytes()).decode("ascii"),
                "mime": "image/jpeg",
                "timestamp": round(idx / video_fps, 2),
            })

        if not frames:
            return None, "could not read any frames from %s" % name
        return {"frames": frames, "duration": round(duration, 2), "fps": video_fps}, None
    finally:
        cap.release()


# ---- audio ------------------------------------------------------------
#
# Unlike video, no server-side decoding happens here: the raw file bytes are
# base64'd as-is. llama.cpp's mtmd audio path (and OpenAI's input_audio spec,
# which llama-server mirrors) decode common containers themselves, so there's
# nothing for us to transcode — same "read live from input/, ship bytes"
# philosophy as images.

def list_audio():
    """Return [{"name", "bytes"}] for audio files in input/."""
    d = input_dir()
    out = []
    try:
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if fn.lower().endswith(AUDIO_EXTS) and os.path.isfile(p):
                try:
                    size = os.path.getsize(p)
                except OSError:
                    size = 0
                out.append({"name": fn, "bytes": size})
    except OSError:
        pass
    return out


def audio_exists(name):
    path = _safe_path(name)
    return bool(path and os.path.isfile(path))


def save_audio(filename, data_bytes):
    """Write a dropped audio file into input/ with collision-safe naming."""
    d = input_dir()
    os.makedirs(d, exist_ok=True)
    base = os.path.basename(str(filename)) or "audio.wav"
    stem, ext = os.path.splitext(base)
    if ext.lower() not in AUDIO_EXTS:
        ext = ".wav"
    candidate = stem + ext
    i = 1
    while os.path.exists(os.path.join(d, candidate)):
        candidate = "%s_%d%s" % (stem, i, ext)
        i += 1
    with open(os.path.join(d, candidate), "wb") as f:
        f.write(data_bytes)
    return candidate


def read_audio_b64(name):
    """Return (base64_str, format_str) for an input audio file, or None if
    missing. `format_str` is the bare extension (e.g. "wav", "mp3") — the
    field name the OpenAI/llama.cpp input_audio content part expects."""
    path = _safe_path(name)
    if not path or not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return data, ext
    except OSError:
        return None
