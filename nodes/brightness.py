import torch
from PIL import Image
import torchvision.transforms as transforms


def calculate_brightness(tensor):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] == 1:
        tensor = torch.cat([tensor] * 3, dim=0)

    print('shape2', tensor.ndim, tensor.shape[0])
    if tensor.shape[0] not in [3, 4]:
        raise ValueError("Unsupported tensor shape. Only 3D tensors with shape (3, H, W) or (4, H, W) are supported.")

    if tensor.shape[0] == 4:  # RGBA
        rgb_tensor = tensor[:3]
        alpha_channel = tensor[3]
        valid_mask = alpha_channel > 0
        rgb_tensor = rgb_tensor[:, valid_mask]
    else:
        rgb_tensor = tensor  # RGB

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
        print('shape', image.shape[2])

        if isinstance(image, Image.Image):
            transform = transforms.ToTensor()
            image = transform(image)

        if image.ndim == 3 and image.shape[2] in [3, 4]:
            image = image.permute(2, 0, 1)

        print('shape1', image.shape[2])
        brightness = calculate_brightness(image)
        brightness_percent = brightness / 255.0
        average_multiple = 0.5 / brightness_percent
        return (image, round(brightness, 3), round(brightness_percent, 3), round(average_multiple, 3))


if __name__ == "__main__":
    image = Image.open("path/to/image.jpg").convert("RGB")
    transform = transforms.ToTensor()
    image_tensor = transform(image)

    brightness = calculate_brightness(image_tensor)
    print(f"Brightness: {brightness}")

    calc = CalculateImageBrightness()
    image, brightness, brightness_percent, average_multiple = calc.load(image_tensor)
    print(f"Brightness: {brightness}")
    print(f"Brightness Percent: {brightness_percent}")
    print(f"Average Multiple: {average_multiple}")