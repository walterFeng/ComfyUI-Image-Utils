import torch
from PIL import Image


def calculate_brightness(tensor):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) == 3 and tensor.shape[0] == 4:  # RGBA
        rgb_tensor = tensor[:3]
        alpha_channel = tensor[3]
        valid_mask = alpha_channel != 0
        rgb_tensor = rgb_tensor[:, valid_mask]
    elif len(tensor.shape) == 3 and tensor.shape[0] == 3:  # RGB
        rgb_tensor = tensor
    else:
        raise ValueError("Unsupported tensor shape. Only 3D tensors with shape (3, H, W) or (4, H, W) are supported.")

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
        brightness = calculate_brightness(image)
        brightness_percent = brightness / 255.0
        average_multiple = 0.5 / brightness_percent
        return (image, round(brightness, 3), round(brightness_percent, 3), round(average_multiple, 3))


if __name__ == "__main__":
    img = Image.open("path_to_your_image.png").convert("RGBA")
    tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(img.size[1], img.size[0], 4).permute(2, 0,
                                                                                                                 1)
    brightness = calculate_brightness(tensor)
    brightness_percent = brightness / 255.0
    brightness_rate = 0.5 / brightness_percent
    print(round(brightness, 3), round(brightness_percent, 3), round(brightness_rate, 3))
