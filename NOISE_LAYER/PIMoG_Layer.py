"""PIMoG screen-shooting degradation exposed through the unified interface."""

import kornia
import math
import numpy as np
import random
import torch
import torch.nn as nn
from kornia.geometry.transform import (
    get_perspective_transform,
    get_rotation_matrix2d,
    warp_affine,
    warp_perspective,
)


# The vendored official implementation uses Kornia's former top-level API.
_KORNIA_COMPAT_ALIASES = {
    "get_perspective_transform": get_perspective_transform,
    "get_rotation_matrix2d": get_rotation_matrix2d,
    "warp_affine": warp_affine,
    "warp_perspective": warp_perspective,
}
for _name, _function in _KORNIA_COMPAT_ALIASES.items():
    if not hasattr(kornia, _name):
        setattr(kornia, _name, _function)

def perspective(image, device, d=8):
    """Original PIMoG random four-corner perspective transform."""
    batch, _, height, width = image.shape
    image_size = height
    points_src = torch.ones(batch, 4, 2)
    points_dst = torch.ones(batch, 4, 2)
    for i in range(batch):
        points_src[i] = torch.tensor([
            [0.0, 0.0], [width - 1.0, 0.0],
            [width - 1.0, height - 1.0], [0.0, height - 1.0],
        ])
        offsets = [random.uniform(-d, d) for _ in range(8)]
        tl_x, tl_y, bl_x, bl_y, tr_x, tr_y, br_x, br_y = offsets
        points_dst[i] = torch.tensor([
            [tl_x, tl_y],
            [tr_x + image_size, tr_y],
            [br_x + image_size, br_y + image_size],
            [bl_x, bl_y + image_size],
        ])
    matrix = kornia.get_perspective_transform(points_src, points_dst).to(device)
    return kornia.warp_perspective(image.float(), matrix, dsize=(height, width)).to(device)


def _moire_pattern(size, theta, center_x, center_y):
    pattern = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            radial = 0.5 + 0.5 * math.cos(
                2 * math.pi * np.sqrt((i + 1 - center_x) ** 2 + (j + 1 - center_y) ** 2)
            )
            directional = 0.5 + 0.5 * math.cos(
                math.cos(theta / 180 * math.pi) * (j + 1)
                + math.sin(theta / 180 * math.pi) * (i + 1)
            )
            pattern[i, j] = np.min([radial, directional])
    return (pattern + 1) / 2


def _light_distortion(kind, image):
    mask = np.zeros(image.shape)
    mask_2d = np.zeros((image.shape[2], image.shape[3]))
    a = 0.7 + np.random.rand(1) * 0.2
    b = 1.1 + np.random.rand(1) * 0.2
    if kind == 0:
        direction = np.random.randint(1, 5)
        for i in range(image.shape[2]):
            mask_2d[i, :] = -((b - a) / (mask.shape[2] - 1)) * (i - mask.shape[3]) + a
        if direction in (2, 3, 4):
            mask_2d = np.rot90(mask_2d, 1)
        for batch in range(image.shape[0]):
            for channel in range(image.shape[1]):
                mask[batch, channel] = mask_2d
    else:
        x = np.random.randint(0, mask.shape[2])
        y = np.random.randint(0, mask.shape[3])
        # Preserve the official PIMoG reference geometry.
        max_len = np.max([
            np.sqrt(x ** 2 + y ** 2), np.sqrt((x - 255) ** 2 + y ** 2),
            np.sqrt(x ** 2 + (y - 255) ** 2),
            np.sqrt((x - 255) ** 2 + (y - 255) ** 2),
        ])
        for i in range(mask.shape[2]):
            for j in range(mask.shape[3]):
                mask[:, :, i, j] = np.sqrt((i - x) ** 2 + (j - y) ** 2) / max_len * (a - b) + b
    return mask


def _moire_distortion(image):
    texture = np.zeros(image.shape)
    for channel in range(3):
        theta = np.random.randint(0, 180)
        center_x = np.random.rand(1) * image.shape[2]
        center_y = np.random.rand(1) * image.shape[3]
        texture[:, channel] = _moire_pattern(image.shape[2], theta, center_x, center_y)
    return texture


class ScreenShooting(nn.Module):
    """Original PIMoG perspective, illumination, moire and Gaussian pipeline."""

    def forward(self, embed_image):
        device = embed_image.device
        noised_image = perspective(embed_image, device, 2)
        light = _light_distortion(np.random.randint(0, 2), embed_image)
        moire = _moire_distortion(embed_image) * 2 - 1
        noised_image = (
            noised_image * torch.from_numpy(light.copy()).to(device) * 0.85
            + torch.from_numpy(moire.copy()).to(device) * 0.15
        )
        return noised_image + 0.001 ** 0.5 * torch.randn(noised_image.size()).to(device)


class PIMoGLayer(nn.Module):
    """Adapter for the original PIMoG pipeline with a ``[0, 1]`` contract.

    The vendored PIMoG implementation itself remains unchanged. Inputs are
    mapped to its historical ``[-1, 1]`` range and its output is mapped back
    to the unified degradation-layer range.
    """

    def __init__(self, p: float = 1.0):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p = float(p)
        self.screen_shooting = ScreenShooting()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("PIMoGLayer expects x with shape [B, 3, H, W]")
        clean = x.clamp(0.0, 1.0)
        if self.p == 0.0 or (self.p < 1.0 and torch.rand((), device=x.device) >= self.p):
            return clean
        degraded = self.screen_shooting(clean.mul(2.0).sub(1.0)).float()
        return degraded.add(1.0).mul(0.5).clamp(0.0, 1.0).to(dtype=x.dtype)
