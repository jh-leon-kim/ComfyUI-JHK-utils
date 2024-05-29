import torch

class JHK_Utils_LoadEmbed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": '', "multiline": False}),
            }
        }

    RETURN_TYPES = ("EMBEDS", )
    FUNCTION = "load"
    CATEGORY = "JHK_utils/embed"

    def load(self, path):
        return (torch.load(path).cpu(), )
    

NODE_CLASS_MAPPINGS = {
    # Main Apply Nodes
    "JHK_Utils_LoadEmbed": JHK_Utils_LoadEmbed,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    # Main Apply Nodes
    "JHK_Utils_LoadEmbed": "JHK_Utils_LoadEmbed",
}