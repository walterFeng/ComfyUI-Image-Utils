import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from .common_utils import check_shape


def calculate_saturation(tensor):
    tensor = check_shape(tensor, True)

    image_numpy = tensor.numpy()

    # to HSV
    hsv_image = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2HSV)

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
    image = Image.open('./test.png')
    calc = CalculateImageSaturation()
    image, Saturation = calc.load(image)
    print(f"Saturation->: {Saturation}")
