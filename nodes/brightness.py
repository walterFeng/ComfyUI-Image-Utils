import torch
import torchvision.transforms as transforms
from PIL import Image


def calculate_brightness(tensor):
    # Ensure the tensor is at least 3D
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    # Ensure the tensor is RGB or RGBA
    if tensor.shape[0] == 1:
        tensor = torch.cat([tensor] * 3, dim=0)

    # Check if the tensor is in a valid shape
    if tensor.shape[0] not in [3, 4]:
        raise ValueError("Unsupported tensor shape. Only 3D tensors with shape (3, H, W) or (4, H, W) are supported.")

    if tensor.shape[0] == 4:  # RGBA
        rgb_tensor = tensor[:3]
        alpha_channel = tensor[3]
        valid_mask = alpha_channel > 0
        rgb_tensor = rgb_tensor[:, valid_mask]
    else:
        rgb_tensor = tensor  # RGB

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
            image = image.convert("RGBA")
            transform = transforms.ToTensor()
            image = transform(image)

        brightness = calculate_brightness(image)
        brightness_percent = brightness / 255.0
        average_multiple = 0.5 / brightness_percent
        return (image, round(brightness, 3), round(brightness_percent, 3), round(average_multiple, 3))


if __name__ == "__main__":
    image = Image.open("../4.png").convert("RGBA")
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    print(calculate_brightness(image_tensor))

    rgb_tensor = torch.rand(3, 100, 100)
    rgba_tensor = torch.rand(4, 100, 100)
    gray_tensor = torch.rand(1, 100, 100)

    print(calculate_brightness(rgb_tensor))
    print(calculate_brightness(rgba_tensor))
    print(calculate_brightness(gray_tensor))
