import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from .common_utils import check_shape


def mask_crop(input_tensor, padding):
    input_tensor = check_shape(input_tensor)

    # 将Tensor转换为NumPy格式，以便进行OpenCV处理
    image = (input_tensor.numpy() * 255).astype(np.uint8)

    # 将图像从BGR转换为灰度图（CV_8UC1）
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化处理，假设黑白图像
    _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 获取图像的大小
    h1, w1, _ = image.shape

    # 初始化矩形区域的边界
    x1, y1, r1, b1 = [sys.maxsize, sys.maxsize, 0, 0]

    # 遍历每个轮廓
    for contour in contours:
        # 获取轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        r = x + w
        b = y + h

        # 更新矩形的边界
        x1 = np.max([np.min([x, x1]) - padding, 0])
        y1 = np.max([np.min([y, y1]) - padding, 0])
        r1 = np.min([np.max([r, r1]) + padding, w1])
        b1 = np.min([np.max([b, b1]) + padding, h1])

    # 创建一个全零的图像
    result = np.zeros_like(binary)

    # 在结果图像上绘制矩形
    cv2.rectangle(result, (x1, y1), (r1, b1), 255, -1)

    # 将结果转换回Tensor
    result_tensor = torch.from_numpy(result).float() / 255.0

    return result_tensor


class CropMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("IMAGE",),
                "padding": ("INT", {"default": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "load"
    CATEGORY = "image"

    def load(self, mask, padding):
        if isinstance(mask, Image.Image):
            transform = transforms.ToTensor()
            mask = transform(mask)

        return mask_crop(mask, padding)


if __name__ == "__main__":
    print("main")
    testImage = Image.open('./test.png')
    calc = CropMask()
    testImage = calc.load(testImage, 10)
    # 将张量转换为PIL图像的函数
    toPIL = transforms.ToPILImage()
    # 将张量转换为PIL图像
    pic = toPIL(testImage)
    # 将PIL图像保存为JPEG文件
    pic.save('output.png')
    print(f"testImage->: {testImage.shape}")
