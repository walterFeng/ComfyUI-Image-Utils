import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def load_image(image_source):
    img = Image.open(image_source)
    file_name = os.path.basename(image_source)
    return img, file_name


def calculate_brightness(tensor):
    if tensor.ndim == 4 and tensor.shape[1] in [1, 2, 3, 4]:
        tensor = tensor.squeeze(0)  # Remove batch dimension if present

    if tensor.ndim == 2:  # Handle grayscale images
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        tensor = torch.cat([tensor] * 3, dim=0)  # Convert to RGB by replicating the single channel

    if tensor.shape[0] == 1:
        tensor = torch.cat([tensor] * 3, dim=0)

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

    RETURN_TYPES = ("image", "brightness", "brightness_percent", "average_multiple")
    FUNCTION = "load"
    CATEGORY = "image"

    def load(self, image):
        if isinstance(image, Image.Image):
            transform = transforms.ToTensor()
            image = transform(image)

        if image.ndim >= 3 and image.shape[2] in [1, 2, 3, 4]:
            img_first_frame = image[0]  # (H, W, C)
            image = np.transpose(img_first_frame, (2, 0, 1))  # (C, H, W)

        brightness = calculate_brightness(image)
        brightness_percent = brightness / 255.0
        average_multiple = 0.5 / brightness_percent
        return (image, round(brightness, 3), round(brightness_percent, 3), round(average_multiple, 3))


if __name__ == "__main__":
    image_tensor = load_image("test.png")
    calc = CalculateImageBrightness()
    image, brightness, brightness_percent, average_multiple = calc.load(image_tensor)
    print(f"Brightness: {brightness}")
    print(f"Brightness Percent: {brightness_percent}")
    print(f"Average Multiple: {average_multiple}")
