"""Differentiable projector-camera degradation for robust watermark training."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import gaussian_blur, sample_uniform, validate_range


class ProjectorSimulator(nn.Module):
    """
    Lightweight differentiable projector-camera degradation simulator.

    This module is inspired by the image-based relighting / projector-to-camera
    forward mapping task in DeProCams. Instead of learning a scene-specific
    ProCam neural renderer, this simulator approximates the major photometric
    and geometric degradations of projector-camera capture using randomized
    differentiable PyTorch operations.

    It is designed for robust watermark training:
        watermarked image -> projector-like degradation -> watermark decoder.

    Main simulated effects:
        - projector gamma nonlinearity
        - brightness falloff / vignetting
        - soft projector hotspot
        - surface reflectance texture
        - projector pixel grid / camera resampling interference
        - spatially varying defocus blur
        - keystone / perspective warp
        - lens radial falloff
        - ambient light and contrast reduction
        - color shift / white balance variation
        - camera sensor noise
    """

    def __init__(
        self,
        p=0.8,
        gamma_range=(1.1, 1.8),
        brightness_gain=(1.00, 1.15),
        falloff_strength=(0.06, 0.20),
        hotspot_strength=(0.00, 0.10),
        hotspot_sigma=(0.5, 1.2),
        texture_strength=(0.015, 0.06),
        pixel_grid_strength=(0.00, 0.02),
        moire_strength=(0.00, 0.02),
        blur_sigma=(0.15, 0.8),
        blur_kernel_choices=(3, 5),
        blur_mix_range=(0.20, 0.70),
        perspective_distortion=(0.015, 0.06),
        lens_distortion=(0.00, 0.03),
        contrast_range=(0.78, 1.00),
        ambient_light_range=(0.00, 0.10),
        color_gain_range=(0.92, 1.08),
        color_bias_range=(-0.015, 0.015),
        noise_std_range=(0.000, 0.012),
        enable_gamma=True,
        enable_falloff=True,
        enable_hotspot=True,
        enable_texture=True,
        enable_pixel_grid=False,
        enable_moire=False,
        enable_blur=True,
        enable_perspective=True,
        enable_lens=False,
        enable_ambient=True,
        enable_color=True,
        enable_noise=True,
    ):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p = float(p)
        self.gamma_range = validate_range("gamma_range", gamma_range)
        self.brightness_gain = validate_range("brightness_gain", brightness_gain)
        self.falloff_strength = validate_range("falloff_strength", falloff_strength)
        self.hotspot_strength = validate_range("hotspot_strength", hotspot_strength)
        self.hotspot_sigma = validate_range("hotspot_sigma", hotspot_sigma)
        self.texture_strength = validate_range("texture_strength", texture_strength)
        self.pixel_grid_strength = validate_range(
            "pixel_grid_strength", pixel_grid_strength
        )
        self.moire_strength = validate_range("moire_strength", moire_strength)
        self.blur_sigma = validate_range("blur_sigma", blur_sigma)
        self.blur_mix_range = validate_range("blur_mix_range", blur_mix_range)
        self.perspective_distortion = validate_range(
            "perspective_distortion", perspective_distortion
        )
        self.lens_distortion = validate_range("lens_distortion", lens_distortion)
        self.contrast_range = validate_range("contrast_range", contrast_range)
        self.ambient_light_range = validate_range(
            "ambient_light_range", ambient_light_range
        )
        self.color_gain_range = validate_range("color_gain_range", color_gain_range)
        self.color_bias_range = validate_range("color_bias_range", color_bias_range)
        self.noise_std_range = validate_range("noise_std_range", noise_std_range)

        kernels = tuple(int(k) for k in blur_kernel_choices)
        if not kernels or any(k <= 0 or k % 2 == 0 for k in kernels):
            raise ValueError("blur_kernel_choices must contain positive odd integers")
        self.blur_kernel_choices = kernels

        self.enable_gamma = bool(enable_gamma)
        self.enable_falloff = bool(enable_falloff)
        self.enable_hotspot = bool(enable_hotspot)
        self.enable_texture = bool(enable_texture)
        self.enable_pixel_grid = bool(enable_pixel_grid)
        self.enable_moire = bool(enable_moire)
        self.enable_blur = bool(enable_blur)
        self.enable_perspective = bool(enable_perspective)
        self.enable_lens = bool(enable_lens)
        self.enable_ambient = bool(enable_ambient)
        self.enable_color = bool(enable_color)
        self.enable_noise = bool(enable_noise)

    def _coordinate_grid(self, x):
        _, _, height, width = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        return yy, xx

    def apply_projector_gamma(self, x):
        gamma = sample_uniform(x, self.gamma_range, (x.shape[0], 1, 1, 1))
        gain = sample_uniform(x, self.brightness_gain, (x.shape[0], 1, 1, 1))
        return x.clamp(0.0, 1.0).pow(gamma) * gain

    def apply_brightness_falloff(self, x):
        batch = x.shape[0]
        yy, xx = self._coordinate_grid(x)
        center_x = sample_uniform(x, (-0.20, 0.20), (batch, 1, 1, 1))
        center_y = sample_uniform(x, (-0.20, 0.20), (batch, 1, 1, 1))
        radius2 = ((xx[None, None] - center_x) ** 2 + (yy[None, None] - center_y) ** 2) / 2.0
        strength = sample_uniform(x, self.falloff_strength, (batch, 1, 1, 1))
        mask = (1.0 - strength * radius2).clamp(0.75, 1.0)
        return x * mask

    def apply_hotspot(self, x):
        batch = x.shape[0]
        yy, xx = self._coordinate_grid(x)
        center_x = sample_uniform(x, (-0.20, 0.20), (batch, 1, 1, 1))
        center_y = sample_uniform(x, (-0.20, 0.20), (batch, 1, 1, 1))
        radius2 = (xx[None, None] - center_x).square() + (yy[None, None] - center_y).square()
        strength = sample_uniform(x, self.hotspot_strength, (batch, 1, 1, 1))
        sigma = sample_uniform(x, self.hotspot_sigma, (batch, 1, 1, 1))
        hotspot = 1.0 + strength * torch.exp(-radius2 / sigma.square().clamp_min(1e-6))
        return x * hotspot

    def apply_surface_texture(self, x):
        batch, _, height, width = x.shape
        small_h, small_w = max(4, height // 24), max(4, width // 24)
        texture = torch.randn(
            batch, 1, small_h, small_w, device=x.device, dtype=x.dtype
        )
        texture = F.interpolate(
            texture, size=(height, width), mode="bilinear", align_corners=False
        )
        minimum = texture.amin(dim=(2, 3), keepdim=True)
        maximum = texture.amax(dim=(2, 3), keepdim=True)
        texture = 2.0 * (texture - minimum) / (maximum - minimum).clamp_min(1e-6) - 1.0
        strength = sample_uniform(x, self.texture_strength, (batch, 1, 1, 1))
        return x * (1.0 + strength * texture)

    def apply_projector_pixel_grid(self, x):
        batch = x.shape[0]
        yy, xx = self._coordinate_grid(x)
        phase_x = sample_uniform(x, (0.0, 6.283185307), (batch, 1, 1, 1))
        phase_y = sample_uniform(x, (0.0, 6.283185307), (batch, 1, 1, 1))
        freq_x = sample_uniform(x, (22.0, 58.0), (batch, 1, 1, 1))
        freq_y = sample_uniform(x, (22.0, 58.0), (batch, 1, 1, 1))
        vertical = torch.sin(freq_x * xx[None, None] + phase_x)
        horizontal = torch.sin(freq_y * yy[None, None] + phase_y)
        grid = 0.5 * (vertical + horizontal)
        strength = sample_uniform(x, self.pixel_grid_strength, (batch, 1, 1, 1))
        return x * (1.0 + strength * grid)

    def apply_moire(self, x):
        batch = x.shape[0]
        yy, xx = self._coordinate_grid(x)
        theta = sample_uniform(x, (0.0, 3.141592654), (batch, 1, 1, 1))
        freq = sample_uniform(x, (7.0, 20.0), (batch, 1, 1, 1))
        phase = sample_uniform(x, (0.0, 6.283185307), (batch, 1, 1, 1))
        directional = torch.sin(
            freq * (torch.cos(theta) * xx[None, None] + torch.sin(theta) * yy[None, None])
            + phase
        )
        radius = torch.sqrt(xx.square() + yy.square()).clamp_min(1e-6)
        radial = torch.sin(freq * 1.7 * radius[None, None] + phase)
        pattern = 0.5 * directional + 0.5 * radial
        color = sample_uniform(x, (0.7, 1.3), (batch, 3, 1, 1))
        strength = sample_uniform(x, self.moire_strength, (batch, 1, 1, 1))
        return x + strength * pattern * color

    def apply_defocus_blur(self, x):
        sigma = sample_uniform(x, self.blur_sigma, (x.shape[0],))
        index = torch.randint(
            len(self.blur_kernel_choices), (), device=x.device
        ).item()
        blurred = gaussian_blur(x, self.blur_kernel_choices[index], sigma)

        batch = x.shape[0]
        yy, xx = self._coordinate_grid(x)
        center_x = sample_uniform(x, (-0.7, 0.7), (batch, 1, 1, 1))
        center_y = sample_uniform(x, (-0.7, 0.7), (batch, 1, 1, 1))
        radius = torch.sqrt(
            (xx[None, None] - center_x).square()
            + (yy[None, None] - center_y).square()
        )
        radius = radius / radius.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        blur_mix = sample_uniform(x, self.blur_mix_range, (batch, 1, 1, 1))
        mask = (0.25 + 0.75 * radius) * blur_mix
        return x * (1.0 - mask).clamp_min(0.0) + blurred * mask.clamp(0.0, 1.0)

    def apply_lens_distortion(self, x):
        batch, _, _, _ = x.shape
        yy, xx = self._coordinate_grid(x)
        xx = xx[None].expand(batch, -1, -1)
        yy = yy[None].expand(batch, -1, -1)
        radius2 = xx.square() + yy.square()
        strength = sample_uniform(x, self.lens_distortion, (batch, 1, 1))
        sign = torch.where(
            torch.rand(batch, 1, 1, device=x.device) < 0.5,
            -torch.ones(batch, 1, 1, device=x.device, dtype=x.dtype),
            torch.ones(batch, 1, 1, device=x.device, dtype=x.dtype),
        )
        scale = 1.0 + sign * strength * radius2
        grid = torch.stack((xx * scale, yy * scale), dim=-1)
        return F.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    def apply_perspective_or_keystone_warp(self, x):
        batch, _, height, width = x.shape
        yy, xx = self._coordinate_grid(x)
        xx = xx[None].expand(batch, -1, -1)
        yy = yy[None].expand(batch, -1, -1)
        magnitude = sample_uniform(x, self.perspective_distortion, (batch, 1, 1))
        signs = torch.where(
            torch.rand(batch, 3, 1, device=x.device) < 0.5,
            -torch.ones(batch, 3, 1, device=x.device, dtype=x.dtype),
            torch.ones(batch, 3, 1, device=x.device, dtype=x.dtype),
        )
        kx = magnitude * signs[:, 0:1]
        ky = magnitude * signs[:, 1:2]
        qx = magnitude * signs[:, 2:3]
        grid_x = xx + kx * yy + qx * yy.square()
        grid_y = yy + ky * xx
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return F.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    def apply_ambient_light_and_contrast(self, x):
        shape = (x.shape[0], 1, 1, 1)
        contrast = sample_uniform(x, self.contrast_range, shape)
        ambient = sample_uniform(x, self.ambient_light_range, shape)
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * contrast + mean
        return x * (1.0 - ambient) + ambient

    def apply_color_shift(self, x):
        shape = (x.shape[0], 3, 1, 1)
        gain = sample_uniform(x, self.color_gain_range, shape)
        bias = sample_uniform(x, self.color_bias_range, shape)
        return x * gain + bias

    def apply_camera_noise(self, x):
        std = sample_uniform(x, self.noise_std_range, (x.shape[0], 1, 1, 1))
        return x + torch.randn_like(x) * std

    def forward(self, x):
        """Degrade ``x`` of shape ``[B, 3, H, W]`` in range ``[0, 1]``."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("ProjectorSimulator expects x with shape [B, 3, H, W]")
        clean = x.clamp(0.0, 1.0)
        if self.p == 0.0 or (self.p < 1.0 and torch.rand((), device=x.device) >= self.p):
            return clean

        x = clean
        if self.enable_gamma:
            x = self.apply_projector_gamma(x)
        if self.enable_falloff:
            x = self.apply_brightness_falloff(x)
        if self.enable_hotspot:
            x = self.apply_hotspot(x)
        if self.enable_texture:
            x = self.apply_surface_texture(x)
        if self.enable_pixel_grid:
            x = self.apply_projector_pixel_grid(x)
        if self.enable_moire:
            x = self.apply_moire(x)
        if self.enable_blur:
            x = self.apply_defocus_blur(x)
        if self.enable_perspective:
            x = self.apply_perspective_or_keystone_warp(x)
        if self.enable_lens:
            x = self.apply_lens_distortion(x)
        if self.enable_ambient:
            x = self.apply_ambient_light_and_contrast(x)
        if self.enable_color:
            x = self.apply_color_shift(x)
        if self.enable_noise:
            x = self.apply_camera_noise(x)
        return x.clamp(0.0, 1.0)
