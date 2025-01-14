from PIL.Image import Image
import torchvision.transforms as transforms

from .common_utils import check_shape

class ImageShapeFix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE (B,H,W,C)", "IMAGE (B,C,H,W)", "image (H,W,C)", "image (C,H,W)")
    FUNCTION = "load"
    CATEGORY = "IMAGE"

    def load(self, image):
        if isinstance(image, Image.Image):
            transform = transforms.ToTensor()
            image = transform(image)
        image1 = check_shape(image, "HWC")
        image2 = check_shape(image, "CHW")
        return (image1.unsqueeze(0), image2.unsqueeze(0), image1, image2)


if __name__ == "__main__":
    print("main")
