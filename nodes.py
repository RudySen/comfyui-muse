"""
The Muse Chat node. This is intentionally a UI-only node: all real work happens in
the custom JS widget hitting the /muse/* backend routes directly, bypassing graph
execution. The node exists only as a host for that DOM widget.
"""


class MuseChatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "noop"
    OUTPUT_NODE = True
    CATEGORY = "utils/muse"

    def noop(self):
        # Never expected to do meaningful work; returns instantly so a queue run
        # that happens to include this node doesn't error.
        return {}


NODE_CLASS_MAPPINGS = {"MuseChat": MuseChatNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MuseChat": "Muse Chat"}
