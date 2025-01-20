import numpy as np
import torch
from PIL import ImageSequence, ImageOps

def pil2tensor(img):
    output_images = []
    output_masks = []
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return (output_image, output_mask)

def check_channel_order(tensor):
    if tensor.shape[0] == 3:
        red_channel = tensor[0]
        blue_channel = tensor[2]
        if red_channel.mean() > blue_channel.mean():
            return "RGB"
        else:
            return "BGR"
    return "RGB"


def check_shape(tensor, to_type="HWC", remove_alpha=True):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension if present

    c_indexed = 2 if to_type == "HWC" else 0
    check_index = 0 if to_type == "HWC" else 2
    indexed = (1, 2, 0) if to_type == "HWC" else (2, 0, 1)
    if tensor.ndim >= 3 and tensor.shape[check_index] in [1, 2, 3, 4]:
        tensor = np.transpose(tensor, indexed)  # (H, W, C)

    if tensor.ndim == 2:  # Handle grayscale images
        tensor = tensor.unsqueeze(2)  # Add batch dimension
        tensor = torch.cat([tensor] * 3, dim=2)  # Convert to RGB by replicating the single channel

    if tensor.shape[c_indexed] == 1:
        tensor = torch.cat([tensor] * 3, dim=0)

    # Convert to grayscale, ensuring correct handling of PNG with alpha channel
    if remove_alpha and tensor.shape[c_indexed] == 4:  # Check if image has an alpha channel
        tensor = tensor[:, :, :3] if to_type == "HWC" else tensor[:3, :, :]  # to RGB
    return tensor

def image_to_mask(image, channel):
    channels = ["red", "green", "blue", "alpha"]
    mask = image[:, :, :, channels.index(channel)]
    return (mask,)