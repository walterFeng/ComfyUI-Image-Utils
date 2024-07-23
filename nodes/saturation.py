import os

import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def calculate_saturation(tensor):
    if tensor.ndim >= 3 and tensor.shape[0] in [1, 2, 3, 4]:
        tensor = np.transpose(tensor, (1, 2, 0))  # (H, W, C)

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
        return image, round(saturation, 3)


if __name__ == "__main__":
    print("main")
    # loader = LoadImageByUrlOrPath()
    # img_hwc, img_chw = loader.load("../test.png")
    image = Image.open('./test.png')
    calc = CalculateImageSaturation()
    image, Saturation = calc.load(image)
    print(f"Saturation->: {Saturation}")
    # print(f"Average Multiple: {average_multiple}")
