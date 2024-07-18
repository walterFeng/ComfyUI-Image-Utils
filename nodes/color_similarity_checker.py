import numpy as np
import tensorflow as tf
import torchvision.transforms as transforms
from PIL import Image
from skimage import color


def color_similarity_checker(tensor_image, threshold=20):
    # Convert Tensor to a NumPy array
    image_np = tensor_image.numpy()

    # Calculate the average color of the picture
    average_color = tf.reduce_mean(image_np, axis=(0, 1)).numpy()

    # Convert images to Lab color space
    image_lab = color.rgb2lab(image_np / 255.0)
    average_color_lab = color.rgb2lab(np.array([[average_color / 255.0]]))[0][0]

    # Calculate the difference between each pixel and the average color (CIEDE2000)
    color_differences = color.deltaE_ciede2000(image_lab, average_color_lab)

    # Determine how many pixels of color difference are within the threshold range
    similar_pixels = tf.reduce_sum(tf.cast(color_differences < threshold, tf.int32)).numpy()
    total_pixels = image_np.shape[0] * image_np.shape[1]

    # Judgment result
    return similar_pixels / total_pixels > 0.95  # Set a percentage, 95%

class ColorSimilarityChecker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("INT", {"multiline": False, "dynamicPrompts": False, "default": 20})
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("IMAGE", "is similarity")
    FUNCTION = "load"
    CATEGORY = "image"

    def load(self, image, threshold):
        if isinstance(image, Image.Image):
            transform = transforms.ToTensor()
            image = transform(image)

        return (image, color_similarity_checker(image, threshold))


if __name__ == "__main__":
    print("main")
    image = Image.open('../test.png')
    transform = transforms.ToTensor()
    image = transform(image)
    calc = ColorSimilarityChecker()
    image, is_similarity = calc.load(image, 30)
