import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from common_utils import check_shape


class DisplaceFilter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "displacement_map": ("IMAGE",),
                "scale": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 1.0})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_displace_filter"
    CATEGORY = "Filter"

    def apply_displace_filter(self, src_image, displacement_map, scale):
        src_image = check_shape(src_image)
        src_image = src_image.numpy()

        displacement_map = check_shape(displacement_map)
        displacement_map = displacement_map.numpy()

        # Ensure displacement map is grayscale
        displacement_map_gray = cv2.cvtColor(displacement_map, cv2.COLOR_RGB2GRAY)

        # Normalize displacement map to range [0, 1]
        displacement_map_normalized = displacement_map_gray / 255.0

        # Scale displacement map
        displacement_map_scaled = (displacement_map_normalized * 2 - 1) * scale

        # Create displacement vectors
        displacement_x = displacement_map_scaled
        displacement_y = displacement_map_scaled

        # Get image dimensions
        height, width = src_image.shape[:2]

        # Create mesh grid
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Add displacement vectors to grid
        map_x = (x + displacement_x).astype(np.float32)
        map_y = (y + displacement_y).astype(np.float32)

        # Apply remap
        displaced_image = cv2.remap(src_image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        return (displaced_image,)


if __name__ == "__main__":
    print('main')
    image = Image.open('./test.png')
    transform = transforms.ToTensor()
    image = transform(image)
    image1 = Image.open('./test1.png')
    transform = transforms.ToTensor()
    image1 = transform(image1)
    d = DisplaceFilter()
    d.apply_displace_filter(check_shape(image), image1, 10)