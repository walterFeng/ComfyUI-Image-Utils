from .nodes.brightness import CalculateImageBrightness
from .nodes.color_similarity_checker import ColorSimilarityChecker
from .nodes.contrast import CalculateImageContrast
from .nodes.mask_crop import CropMask
from .nodes.load_image_by_url import LoadImageByUrlOrPath
from .nodes.saturation import CalculateImageSaturation
from .nodes.displacement import DisplaceFilter
from .nodes.aliyun_mask_refine import RefineMask
from .nodes.image_shape_fix import ImageShapeFix

NODE_CLASS_MAPPINGS = {
    "Load Image (By Url)": LoadImageByUrlOrPath,
    "Color Similarity Checker": ColorSimilarityChecker,
    "Calculate Image Brightness": CalculateImageBrightness,
    "Calculate Image Saturation": CalculateImageSaturation,
    "Calculate Image Contrast": CalculateImageContrast,
    "Crop Mask Util": CropMask,
    "Mask Refine (Aliyun)": RefineMask,
    "Image Fix (tensor shape convert)": ImageShapeFix,
    "Displace Filter": DisplaceFilter,
}
