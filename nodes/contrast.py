import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from skimage import exposure

from nodes.common_utils import check_shape


def is_low_contrast(tensor):
    tensor = check_shape(tensor)

    image_numpy = tensor.numpy()

    image = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2GRAY)

    isContrast = exposure.is_low_contrast(image)
    print("Is low contrast:", isContrast)
    return isContrast


def calculate_rms_contrast(tensor):
    """
    ## Root Mean Square Contrast
    """

    tensor = check_shape(tensor)

    image_numpy = tensor.numpy()

    image = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2GRAY)

    mean = np.mean(image)

    rms_contrast = np.sqrt(np.mean((image - mean) ** 2))
    print("RMS Contrast:", rms_contrast)

    return rms_contrast


def calculate_intensity_range_contrast(tensor):
    """
    ## Intensity range contrast
    """

    tensor = check_shape(tensor)

    image_numpy = tensor.numpy()

    image = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2GRAY)

    min_intensity = np.min(image)
    max_intensity = np.max(image)

    contrast = max_intensity - min_intensity

    print("Contrast:", contrast)
    return contrast


class CalculateImageContrast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "FLOAT", "FLOAT")
    RETURN_NAMES = ("IMAGE", "Is Low Contrast", "RMS Contrast", "Contrast")
    FUNCTION = "load"
    CATEGORY = "image"

    def load(self, image):
        if isinstance(image, Image.Image):
            transform = transforms.ToTensor()
            image = transform(image)

        isLowContrast = is_low_contrast(image)
        rmsContrast = calculate_rms_contrast(image)
        contrast = calculate_intensity_range_contrast(image)
        return (image, isLowContrast, rmsContrast, contrast)


if __name__ == "__main__":
    print("main")
    image = Image.open('./test.png')
    calc = CalculateImageContrast()
    (image, isLowContrast, rmsContrast, contrast) = calc.load(image)
    print(f"Contrast->: {isLowContrast}: {rmsContrast}: {isLowContrast}")
