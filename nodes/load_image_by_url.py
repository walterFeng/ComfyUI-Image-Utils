import os
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image


def pil2tensor(img):
    np_img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(np_img)
    return (img, img.permute(2, 0, 1))


def load_image(image_source):
    if image_source.startswith('http'):
        print(image_source)
        response = requests.get(image_source)
        img = Image.open(BytesIO(response.content)).convert("RGBA")
        file_name = image_source.split('/')[-1]
    else:
        img = Image.open(image_source).convert("RGBA")
        file_name = os.path.basename(image_source)
    return img, file_name


class LoadImageByUrlOrPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_or_path": ("STRING", {"multiline": True, "dynamicPrompts": False})
            }
        }

    RETURN_TYPES = ("IMAGE", "image")
    RETURN_NAMES = ("IMAGE (H,W,C)", "image (C,H,W)")
    FUNCTION = "load"
    CATEGORY = "image"

    def load(self, url_or_path):
        print(url_or_path)
        img, name = load_image(url_or_path)
        img_hwc, img_chw = pil2tensor(img)
        return (img_hwc, img_chw)


if __name__ == "__main__":
    img, name = load_image(
        "https://ts1.cn.mm.bing.net/th/id/R-C.26fa5434823e0afae3f9b576b61b3df0?rik=1ki5rrqJXLS00w&riu=http%3a%2f%2fpic.52112.com%2f180420%2f180420_32%2fJ9xjxe1jIg_small.jpg&ehk=a8hQQlllEncpFeXgnFZ1a7fIII7lcz2ph6WLdtzS51k%3d&risl=&pid=ImgRaw&r=0")
    img_hwc, img_chw = pil2tensor(img)
    print(img_chw.shape)
    print(img_hwc.shape)
