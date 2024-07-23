from .nodes.brightness import CalculateImageBrightness
from .nodes.saturation import CalculateImageSaturation
from .nodes.load_image_by_url import LoadImageByUrlOrPath
from .nodes.color_similarity_checker import ColorSimilarityChecker

NODE_CLASS_MAPPINGS = {
    "Calculate Image Brightness": CalculateImageBrightness,
    "Load Image (By Url)": LoadImageByUrlOrPath,
    "Color Similarity Checker": ColorSimilarityChecker,
    "Calculate Image Saturation": CalculateImageSaturation,
}
