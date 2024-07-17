import os
from io import BytesIO

import requests
from PIL import Image


def calculate_brightness(img):
    pixels = img.getdata()
    brightness = 0
    count = 0
    for pixel in pixels:
        if len(pixel) == 3:
            brightness += (0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
            count += 1
        elif len(pixel) == 4 and pixel[3] != 0:
            brightness += (0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
            count += 1

    if count != 0:
        average_brightness = brightness / count
    else:
        average_brightness = 0

    print(f'Average brightness (excluding transparent areas): {average_brightness}')
    return average_brightness

def load_image(image_source):
    if image_source.startswith('http'):
        print(image_source)
        response = requests.get(image_source)
        img = Image.open(BytesIO(response.content))
        file_name = image_source.split('/')[-1]
    else:
        img = Image.open(image_source)
        file_name = os.path.basename(image_source)
    return img, file_name

class ComputeImageBrightness:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "FLOAT")
    FUNCTION = "load"
    CATEGORY = "image"

    def load(self, image):
        brightness = calculate_brightness(image)
        brightness_percent = brightness / 255.0
        brightness_rate = 0.5 / brightness_percent
        return (image, round(brightness, 3), round(brightness_percent, 3), round(brightness_rate, 3))


if __name__ == "__main__":
    img, name = load_image("https://pic35.photophoto.cn/20150511/0034034892281415_b.jpg")
    brightness = calculate_brightness(img)
    brightness_percent = brightness / 255.0
    brightness_rate = 0.5 / brightness_percent
    print(round(brightness, 3), round(brightness_percent, 3), round(brightness_rate, 3))
