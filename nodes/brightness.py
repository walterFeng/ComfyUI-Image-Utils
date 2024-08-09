import os

import torchvision.transforms as transforms
from PIL import Image

from .common_utils import check_shape


def load_image(image_source):
    img = Image.open(image_source)
    file_name = os.path.basename(image_source)
    return img, file_name


def calculate_brightness(tensor):
    tensor = check_shape(tensor, 'CHW', False)

    if tensor.shape[0] == 4:  # RGBA
        rgb_tensor = tensor[:3]  # Extract RGB channels
        alpha_channel = tensor[3]  # Extract alpha channel
        valid_mask = alpha_channel > 0  # Create a mask for valid pixels
        rgb_tensor = rgb_tensor[:, valid_mask]  # Apply mask to RGB channels
    elif tensor.shape[0] == 3:  # RGB
        rgb_tensor = tensor
    else:
        raise ValueError(
            "Unsupported tensor shape. Expected 3D tensor with shape (3, H, W) or 4D tensor with shape (4, H, W).")

    # Calculate brightness
    brightness = (0.299 * rgb_tensor[0] + 0.587 * rgb_tensor[1] + 0.114 * rgb_tensor[2]).mean()
    return brightness.item()


class CalculateImageBrightness:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT")
    RETURN_NAMES = ("IMAGE", "brightness", "inverted brightness")
    FUNCTION = "load"
    CATEGORY = "image"

    def load(self, image):
        if isinstance(image, Image.Image):
            transform = transforms.ToTensor()
            image = transform(image)

        brightness = calculate_brightness(image)
        inverted_brightness = 0.5 / brightness if brightness > 0 else 255
        return (image, round(brightness, 3), round(inverted_brightness, 3))


if __name__ == "__main__":
    print("main")
    # loader = LoadImageByUrlOrPath()
    # img_hwc, img_chw = loader.load("../test.png")
    #
    # calc = CalculateImageBrightness()
    # image, brightness, average_multiple = calc.load(img_hwc)
    # print(f"Brightness: {brightness}")
    # print(f"Average Multiple: {average_multiple}")
