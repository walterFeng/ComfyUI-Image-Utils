import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from .common_utils import check_shape


def calculate_contrast_tensor(tensor_image):
    # Convert to PIL Image for easier handling
    pil_image = to_pil_image(tensor_image)

    # Use the PIL method
    # Convert to grayscale
    grayscale_image = pil_image.convert('L')

    # Convert to numpy array
    image_array = np.array(grayscale_image)

    # Calculate mean luminance
    mean_luminance = np.mean(image_array)

    # Calculate RMS contrast
    rms_contrast = np.sqrt(np.mean((image_array - mean_luminance) ** 2))
    return rms_contrast


class CalculateImageContrast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT")
    RETURN_NAMES = ("IMAGE", "Contrast", "Contrast Percentage")
    FUNCTION = "load"
    CATEGORY = "image"

    def load(self, image):
        if isinstance(image, Image.Image):
            transform = transforms.ToTensor()
            image = transform(image)

        tensor_image = check_shape(image, "CHW", True)
        rmsContrast = calculate_contrast_tensor(tensor_image)
        contrast_percentage = rmsContrast / 255.0
        return (image, rmsContrast, contrast_percentage)


if __name__ == "__main__":
    print("main")
    image_path = './test.png'
    image = Image.open(image_path)
    calc = CalculateImageContrast()
    (image, rmsContrast, contrast_percentage) = calc.load(image)
    print(f"Contrast->: {rmsContrast}: {contrast_percentage}")

    # Load the new image
    image_path_black_dress = image_path
    image_black_dress = cv2.imread(image_path_black_dress, cv2.IMREAD_GRAYSCALE)

    cropped_image_black_dress = image_black_dress  # [50:900, 100:400]

    # print("image_array right:", cropped_image_black_dress)
    # Calculate the mean luminance of the cropped image
    mean_luminance_black_dress = np.mean(cropped_image_black_dress)

    # Calculate RMS contrast
    rms_contrast_black_dress = np.sqrt(np.mean((cropped_image_black_dress - mean_luminance_black_dress) ** 2))
    print(f"RMS Contrast 1 : {rms_contrast_black_dress / 255.0}")

    image = Image.open(image_path)  # .convert("RGBA")
    transform = transforms.ToTensor()
    tensor_image = transform(image)
    tensor_image = check_shape(tensor_image, "CHW", True)
    contrast = calculate_contrast_tensor(tensor_image)
    print(f"RMS Contrast 2 : {contrast / 255.0}")

    img = Image.open(image_path).convert('L')  # 转换为灰度图像

    img_array = np.array(img)

    contrast = np.std(img_array)
    print(f"RMS Contrast 3 : {contrast / 255.0}")

    img = Image.open(image_path)

    if img.mode == 'RGBA':
        r, g, b, a = img.split()
        img = Image.merge('RGB', (r, g, b)).convert('L')
    elif img.mode == 'RGB':
        img = img.convert('L')

    img_array = np.array(img)

    contrast = np.std(img_array)
    print(f"RMS Contrast 4 : {contrast / 255.0}")
