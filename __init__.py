"""
ComfyUI-Muse — a local LLM chat panel for ComfyUI.

Drop-in custom node: registers the MuseChat node, serves the web/ frontend, and
imports server.py so its /muse/* routes register against the running PromptServer.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Importing server.py registers the aiohttp routes at startup. Wrapped so a route
# registration failure can't take down the whole custom_nodes import.
try:
    from . import server  # noqa: F401
except Exception as e:  # pragma: no cover
    print("[ComfyUI-Muse] Failed to register server routes: %s" % e)

WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
