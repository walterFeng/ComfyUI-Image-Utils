import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def calculate_saturation(tensor):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension if present

    if tensor.ndim >= 3 and tensor.shape[0] in [1, 2, 3, 4]:
        tensor = np.transpose(tensor, (1, 2, 0))  # (H, W, C)

    if tensor.ndim == 2:  # Handle grayscale images
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        tensor = torch.cat([tensor] * 3, dim=0)  # Convert to RGB by replicating the single channel

    # Convert to grayscale, ensuring correct handling of PNG with alpha channel
    if tensor.shape[2] == 4:  # Check if image has an alpha channel
        tensor = tensor[:, :, :3]  # to RGB

    image_numpy = tensor.numpy()

    # to HSV
    hsv_image = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv_image)

    average_saturation = np.mean(s)
    print(f'Average Saturation: {average_saturation}')

    return average_saturation


class CalculateImageSaturation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("IMAGE", "saturation")
    FUNCTION = "load"
    CATEGORY = "image"

    def load(self, image):
        if isinstance(image, Image.Image):
            transform = transforms.ToTensor()
            image = transform(image)

        saturation = calculate_saturation(image)
        return image, saturation


if __name__ == "__main__":
    print("main")
    #loader = LoadImageByUrlOrPath()
    #image, img_chw = loader.load("../../Desktop/test.png")
    image = Image.open('../../Desktop/test.png')
    calc = CalculateImageSaturation()
    image, Saturation = calc.load(image)
    print(f"Saturation->: {Saturation}")
    # print(f"Average Multiple: {average_multiple}")
