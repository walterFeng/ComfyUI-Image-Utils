import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from .common_utils import check_shape


def calculate_contrast_pil(image):
    # Convert to grayscale
    grayscale_image = image.convert('L')

    # Convert to numpy array
    image_array = np.array(grayscale_image)

    print("image_array", image_array)

    # Calculate mean luminance
    mean_luminance = np.mean(image_array)

    # Calculate RMS contrast
    rms_contrast = np.sqrt(np.mean((image_array - mean_luminance) ** 2))

    return rms_contrast


def calculate_contrast_tensor(tensor_image):
    # Ensure the image is in grayscale
    if tensor_image.ndim == 3 and tensor_image.size(0) == 3:
        tensor_image = tensor_image.mean(dim=0, keepdim=True)

    # Convert to PIL Image for easier handling
    pil_image = to_pil_image(tensor_image)

    # Use the PIL method
    # Convert to grayscale
    grayscale_image = pil_image.convert('L')

    # Convert to numpy array
    image_array = np.array(grayscale_image)

    print("image_array", image_array)

    # Calculate mean luminance
    mean_luminance = np.mean(image_array)

    # Calculate RMS contrast
    rms_contrast = np.sqrt(np.mean((image_array - mean_luminance) ** 2))
    return rms_contrast

def calculate_intensity_range_contrast(tensor):
    """
    ## Intensity range contrast
    """

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

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT")
    RETURN_NAMES = ("IMAGE", "RMS Contrast", "Contrast")
    FUNCTION = "load"
    CATEGORY = "image"

    def load(self, image):
        if isinstance(image, Image.Image):
            transform = transforms.ToTensor()
            image = transform(image)

        tensor_image = check_shape(image, "CHW", True)
        rmsContrast = calculate_contrast_tensor(tensor_image)
        contrast = calculate_intensity_range_contrast(tensor_image)
        return (image, rmsContrast, contrast)


if __name__ == "__main__":
    print("main")
    image = Image.open('./test.png')
    calc = CalculateImageContrast()
    (image, rmsContrast, contrast) = calc.load(image)
    print(f"Contrast->: {rmsContrast}: {contrast}")

    # Load the new image
    image_path_black_dress = './test.png'
    image_black_dress = cv2.imread(image_path_black_dress, cv2.IMREAD_GRAYSCALE)

    cropped_image_black_dress = image_black_dress#[50:900, 100:400]

    print("image_array right:", cropped_image_black_dress)
    # Calculate the mean luminance of the cropped image
    mean_luminance_black_dress = np.mean(cropped_image_black_dress)

    # Calculate RMS contrast
    rms_contrast_black_dress = np.sqrt(np.mean((cropped_image_black_dress - mean_luminance_black_dress) ** 2))
    print(f"RMS Contrast 1 : {rms_contrast_black_dress}")

    image = Image.open('./test.png').convert("RGBA")
    transform = transforms.ToTensor()
    tensor_image = transform(image)
    tensor_image = check_shape(tensor_image, "CHW", True)
    contrast = calculate_contrast_tensor(tensor_image)
    print(f"RMS Contrast 2 : {contrast}")