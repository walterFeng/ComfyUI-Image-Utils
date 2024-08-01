from .nodes.brightness import CalculateImageBrightness
from .nodes.color_similarity_checker import ColorSimilarityChecker
from .nodes.contrast import CalculateImageContrast
from .nodes.load_image_by_url import LoadImageByUrlOrPath
from .nodes.saturation import CalculateImageSaturation
from .nodes.displacement import DisplaceFilter

NODE_CLASS_MAPPINGS = {
    "Load Image (By Url)": LoadImageByUrlOrPath,
    "Color Similarity Checker": ColorSimilarityChecker,
    "Calculate Image Brightness": CalculateImageBrightness,
    "Calculate Image Saturation": CalculateImageSaturation,
    "Calculate Image Contrast": CalculateImageContrast,
    "Displace Filter": DisplaceFilter,
}
