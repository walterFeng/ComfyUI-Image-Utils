import numpy as np
import torch


def check_channel_order(tensor):
    if tensor.shape[0] == 3:
        red_channel = tensor[0]
        blue_channel = tensor[2]
        if red_channel.mean() > blue_channel.mean():
            return "RGB"
        else:
            return "BGR"
    return "RGB"


def check_shape(tensor, toType="HWC", removeAlpha=True):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension if present

    cIndexed = 2 if toType == "HWC" else 0
    checkIndex = 0 if toType == "HWC" else 2
    indexed = (1, 2, 0) if toType == "HWC" else (2, 0, 1)
    if tensor.ndim >= 3 and tensor.shape[checkIndex] in [1, 2, 3, 4]:
        tensor = np.transpose(tensor, indexed)  # (H, W, C)

    if tensor.ndim == 2:  # Handle grayscale images
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        tensor = torch.cat([tensor] * 3, dim=0)  # Convert to RGB by replicating the single channel

    if tensor.shape[cIndexed] == 1:
        tensor = torch.cat([tensor] * 3, dim=0)

    # Convert to grayscale, ensuring correct handling of PNG with alpha channel
    if removeAlpha and tensor.shape[cIndexed] == 4:  # Check if image has an alpha channel
        tensor = tensor[:, :, :3] if toType == "HWC" else tensor[:3, :, :]  # to RGB
    return tensor, cIndexed
