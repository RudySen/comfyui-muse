// muse_chat_api.js — thin fetch() wrappers around our own /muse/* backend routes.
// All same-origin (ComfyUI's own server), so no CORS concerns. Exposed as
// window.museChatApi for the UI module to consume.

const museChatApi = (() => {
  async function jsonOrThrow(resp) {
    let data = null;
    try {
      data = await resp.json();
    } catch (e) {
      /* non-JSON body */
    }
    if (!resp.ok) {
      const msg = (data && data.error) || `HTTP ${resp.status}`;
      throw new Error(msg);
    }
    return data;
  }

  async function fetchModels(backend, baseUrl) {
    const q = new URLSearchParams({ backend, base_url: baseUrl || "" });
    const resp = await fetch(`/muse/models?${q.toString()}`);
    const data = await jsonOrThrow(resp);
    return data.models || [];
  }

  async function loadModel(backend, baseUrl, model) {
    const resp = await fetch("/muse/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend, base_url: baseUrl, model }),
    });
    return jsonOrThrow(resp);
  }

  async function unloadModel(backend, baseUrl, model, modelInstanceId, confirm) {
    const resp = await fetch("/muse/unload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        backend,
        base_url: baseUrl,
        model,
        model_instance_id: modelInstanceId || null,
        confirm: !!confirm,
      }),
    });
    return jsonOrThrow(resp);
  }

  async function getStatus(backend, baseUrl, model) {
    const q = new URLSearchParams({ backend, base_url: baseUrl || "", model: model || "" });
    const resp = await fetch(`/muse/status?${q.toString()}`);
    return jsonOrThrow(resp);
  }

  // Streams a chat completion. Calls onChunk(normalizedChunk) for each SSE event.
  // Returns a promise that resolves when the stream ends. Pass an AbortSignal to
  // support the Stop button.
  async function streamChat(
    { backend, baseUrl, model, systemPrompt, messages, freeComfyVram, maxTokens, guides, videoFps, videoMaxFrames },
    onChunk,
    signal
  ) {
    const resp = await fetch("/muse/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        backend,
        base_url: baseUrl,
        model,
        system_prompt: systemPrompt,
        messages,
        free_comfy_vram: freeComfyVram !== false,
        max_tokens: maxTokens || undefined,
        guides: guides || [],
        video_fps: videoFps || undefined,
        video_max_frames: videoMaxFrames || undefined,
      }),
      signal,
    });
    if (!resp.ok) {
      let msg = `HTTP ${resp.status}`;
      try {
        const data = await resp.json();
        if (data && data.error) msg = data.error;
      } catch (e) {
        /* ignore */
      }
      throw new Error(msg);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const line = rawEvent.trim();
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") return;
        try {
          onChunk(JSON.parse(payload));
        } catch (e) {
          /* skip malformed event */
        }
      }
    }
  }

  async function listChats() {
    const resp = await fetch("/muse/chats");
    const data = await jsonOrThrow(resp);
    return data.chats || [];
  }

  async function getChat(id) {
    const resp = await fetch(`/muse/chats/${id}`);
    return jsonOrThrow(resp);
  }

  async function createChat(payload) {
    const resp = await fetch("/muse/chats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    });
    return jsonOrThrow(resp);
  }

  async function updateChat(id, data) {
    const resp = await fetch(`/muse/chats/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return jsonOrThrow(resp);
  }

  async function deleteChat(id) {
    const resp = await fetch(`/muse/chats/${id}`, { method: "DELETE" });
    return jsonOrThrow(resp);
  }

  // ---- ComfyUI input/ (guides + images) ----
  async function listGuides() {
    const resp = await fetch("/muse/input/guides");
    const data = await jsonOrThrow(resp);
    return data.guides || [];
  }

  async function listInputImages() {
    const resp = await fetch("/muse/input/images");
    const data = await jsonOrThrow(resp);
    return data.images || [];
  }

  async function listInputVideos() {
    const resp = await fetch("/muse/input/videos");
    const data = await jsonOrThrow(resp);
    return data.videos || [];
  }

  // Save a dropped video File into ComfyUI/input/. Returns the (collision-safe) name.
  async function saveInputVideo(file) {
    const q = new URLSearchParams({ filename: file.name || "video.mp4" });
    const resp = await fetch(`/muse/input/save-video?${q.toString()}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    const data = await jsonOrThrow(resp);
    return data.name;
  }

  // Save a dropped File into ComfyUI/input/. Returns the (collision-safe) name.
  async function saveInputImage(file) {
    const q = new URLSearchParams({ filename: file.name || "image.png" });
    const resp = await fetch(`/muse/input/save-image?${q.toString()}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    const data = await jsonOrThrow(resp);
    return data.name;
  }

  async function listInputAudio() {
    const resp = await fetch("/muse/input/audio");
    const data = await jsonOrThrow(resp);
    return data.audio || [];
  }

  // Save a dropped audio File into ComfyUI/input/. Returns the (collision-safe) name.
  async function saveInputAudio(file) {
    const q = new URLSearchParams({ filename: file.name || "audio.wav" });
    const resp = await fetch(`/muse/input/save-audio?${q.toString()}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    const data = await jsonOrThrow(resp);
    return data.name;
  }

  // URL ComfyUI serves input files from — used for inline thumbnails.
  function inputFileUrl(name) {
    const q = new URLSearchParams({ filename: name, type: "input", subfolder: "" });
    return `/view?${q.toString()}`;
  }

  // ---- Direct model loader settings ----
  async function getDirectSettings() {
    const resp = await fetch("/muse/direct/settings");
    return jsonOrThrow(resp);
  }

  async function saveDirectSettings(settings) {
    const resp = await fetch("/muse/direct/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings || {}),
    });
    return jsonOrThrow(resp);
  }

  async function detectDirectBinary() {
    const resp = await fetch("/muse/direct/detect-binary");
    return jsonOrThrow(resp);
  }

  async function suggestDirectFolders() {
    const resp = await fetch("/muse/direct/suggest-folders");
    return jsonOrThrow(resp);
  }

  async function getDirectPlatformInfo() {
    const resp = await fetch("/muse/direct/platform-info");
    return jsonOrThrow(resp);
  }

  // Downloads + extracts a matching llama.cpp release build server-side and
  // points Direct Loader settings at it. No request timeout is set here —
  // the download itself can take a while on a slow connection.
  async function downloadDirectBinary(variant) {
    const resp = await fetch("/muse/direct/download-binary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ variant: variant || "vulkan" }),
    });
    return jsonOrThrow(resp);
  }

  async function getDirectGpuInfo() {
    const resp = await fetch("/muse/direct/gpu-info");
    return jsonOrThrow(resp);
  }

  async function getDirectLog() {
    const resp = await fetch("/muse/direct/log");
    return jsonOrThrow(resp);
  }

  return {
    fetchModels,
    loadModel,
    unloadModel,
    getStatus,
    streamChat,
    listChats,
    getChat,
    createChat,
    updateChat,
    deleteChat,
    listGuides,
    listInputImages,
    saveInputImage,
    listInputVideos,
    saveInputVideo,
    listInputAudio,
    saveInputAudio,
    inputFileUrl,
    getDirectSettings,
    saveDirectSettings,
    detectDirectBinary,
    suggestDirectFolders,
    getDirectPlatformInfo,
    downloadDirectBinary,
    getDirectGpuInfo,
    getDirectLog,
  };
})();

window.museChatApi = museChatApi;
export default museChatApi;
