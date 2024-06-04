import torch
from rembg import remove
from PIL import Image
import torch
import numpy as np

from icecream import ic


# Tensor to PIL
def tensor2pil(image):
    print('1----',image.shape)
    image = image[0]
    print('2----',image.shape)
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)



# def tensor2pil(image):
#     # Convert the tensor to a numpy array and move it to CPU memory
#     image_np = image.cpu().numpy()
    
#     # Handle batch dimension
#     if image_np.ndim == 4 and image_np.shape[0] == 1:
#         image_np = image_np.squeeze(0)  # Remove the batch dimension if it's 1
#     elif image_np.ndim == 4 and image_np.shape[0] > 1:
#         raise ValueError("Batch size greater than 1 is not supported for conversion to single PIL Image")

#     # Convert to PIL image, ensuring that the data type and dimensions are correct
#     if image_np.ndim == 3 and image_np.shape[2] == 3:  # Check for color image
#         return Image.fromarray(np.clip(image_np * 255, 0, 255).astype(np.uint8))
#     else:
#         raise ValueError("Image format not supported. Ensure it is a color image with 3 channels.")


def pil2tensor(image):
    # Convert PIL image to numpy array, normalize, convert to tensor
    tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    # Add a batch dimension if needed (based on your processing pipeline)
    return tensor.unsqueeze(0) if tensor.ndim == 3 else tensor

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

class JHK_Utils_string_merge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": '', "multiline": False}),
            },
            "optional": {
                "string2": ("STRING", {"default": '', "multiline": False}),
                "string3": ("STRING", {"default": '', "multiline": False}),
                "string4": ("STRING", {"default": '', "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", )
    FUNCTION = "merge"
    CATEGORY = "JHK_utils/string"
    
    def merge(self, string1, string2='', string3='', string4=''):
        # Split each string into a set of tags
        set1 = set(string1.split(", "))
        set2 = set(string2.split(", ")) if string2 else set()
        set3 = set(string3.split(", ")) if string3 else set()
        set4 = set(string4.split(", ")) if string4 else set()
        
        # Combine all sets into one to remove duplicates
        combined_set = set1.union(set2).union(set3).union(set4)
        
        # Convert the set back to a sorted list and then to a single string
        merged_string = ', '.join(sorted(combined_set))
        
        # Print combined set and merged string for debugging
        ic(combined_set)  # Use ic for introspection if ic is imported, else use print
        ic(merged_string)
        
        # Return the result in the specified format
        return {"ui": {"text": merged_string}, "result": (merged_string,)}



class JHK_Utils_ImageRemoveBackground:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image):
        image = pil2tensor(remove(tensor2pil(image)))
        return (image,)


NODE_CLASS_MAPPINGS = {
    # Main Apply Nodes
    "JHK_Utils_LoadEmbed": JHK_Utils_LoadEmbed,
    "JHK_Utils_string_merge":JHK_Utils_string_merge,
    "JHK_Utils_ImageRemoveBackground": JHK_Utils_ImageRemoveBackground,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    # Main Apply Nodes
    "JHK_Utils_LoadEmbed": "JHK_Utils_LoadEmbed",
    "JHK_Utils_string_merge": "JHK_Utils_string_merge",
    "JHK_Utils_ImageRemoveBackground": "JHK_Utils_ImageRemoveBackground",
}


# pip install rembg[gpu]
# https://github.com/cubiq/ComfyUI_FaceAnalysis.git