"""Reusable differentiable tensor operations for degradation layers."""

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


def validate_range(name: str, value: Sequence[float]) -> Tuple[float, float]:
    """Return a validated numeric ``(low, high)`` pair."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise ValueError(f"{name} lower bound must not exceed upper bound")
    return low, high


def sample_uniform(x: torch.Tensor, value_range, shape):
    """Sample on the same device and with the same dtype as ``x``."""
    low, high = value_range
    return torch.empty(shape, device=x.device, dtype=x.dtype).uniform_(low, high)


def gaussian_kernel2d(kernel_size: int, sigma: torch.Tensor) -> torch.Tensor:
    """Create one normalized 2-D Gaussian kernel per batch item."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    coords = torch.arange(
        kernel_size, device=sigma.device, dtype=sigma.dtype
    ) - (kernel_size - 1) / 2
    sigma = sigma.reshape(-1, 1).clamp_min(torch.finfo(sigma.dtype).eps)
    kernel_1d = torch.exp(-(coords.reshape(1, -1) ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum(dim=1, keepdim=True)
    kernel_2d = kernel_1d[:, :, None] * kernel_1d[:, None, :]
    return kernel_2d


def gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: torch.Tensor) -> torch.Tensor:
    """Apply per-sample depthwise Gaussian blur without PIL or OpenCV."""
    if x.ndim != 4:
        raise ValueError("x must have shape [B, C, H, W]")
    batch, channels, height, width = x.shape
    kernels = gaussian_kernel2d(kernel_size, sigma).to(dtype=x.dtype)
    kernels = kernels[:, None].expand(-1, channels, -1, -1)
    kernels = kernels.reshape(batch * channels, 1, kernel_size, kernel_size)

    pad = kernel_size // 2
    padding_mode = "reflect" if height > pad and width > pad else "replicate"
    padded = F.pad(x, (pad, pad, pad, pad), mode=padding_mode)
    grouped = padded.reshape(1, batch * channels, height + 2 * pad, width + 2 * pad)
    blurred = F.conv2d(grouped, kernels, groups=batch * channels)
    return blurred.reshape(batch, channels, height, width)
