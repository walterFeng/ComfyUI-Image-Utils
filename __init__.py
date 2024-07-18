from .nodes.brightness import CalculateImageBrightness
from .nodes.load_image_by_url import LoadImageByUrlOrPath

NODE_CLASS_MAPPINGS = {
    "Calculate Image Brightness": CalculateImageBrightness,
    "Load Image (By Url)": LoadImageByUrlOrPath,
}
