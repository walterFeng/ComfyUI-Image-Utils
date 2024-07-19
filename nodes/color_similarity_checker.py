import torchvision.transforms as transforms
from PIL import Image


def color_similarity_checker(tensor, threshold):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension if present

    if tensor.ndim >= 3 and tensor.shape[0] in [1, 2, 3, 4]:
        tensor = tensor.permute(1, 2, 0)  # shape to (H, W, C)

    # Delete alpha channel
    tensor = tensor[:, :, :3]

    # Convert Tensor to a NumPy array
    np_image = (tensor * 255).numpy()

    # Calculate the mean and variance of each channel
    std_dev = np_image.std(axis=(0, 1))
    # mean = np_image.mean(axis=(0, 1))
    # print(f'Mean: {mean}')
    # print(f'Standard Deviation: {std_dev}')

    # check colors are similar
    print(std_dev.max())
    if std_dev.max() < threshold:
        return True
    return False


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
    print(is_similarity)
