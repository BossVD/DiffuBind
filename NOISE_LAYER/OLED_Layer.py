"""Differentiable mobile OLED screen-shooting degradation layer."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import gaussian_blur, sample_uniform, validate_range


class OLED_Layer(nn.Module):
    """
    OLED display-camera degradation simulator.

    The forward path follows a physical capture order: OLED tone response,
    PenTile-like sub-pixels, display emission spread, camera blur / PWM
    rolling-shutter bands, viewing-angle color shift, sensor noise, glass
    reflection / haze, optional resampling and an optional differentiable JPEG
    proxy. Inputs and outputs use the project-wide ``[0, 1]`` tensor contract.
    """

    def __init__(
        self,
        config=None,
        p=1.0,
        enable_tone=True,
        enable_subpixel=True,
        enable_display_blur=True,
        enable_perspective=True,
        enable_camera_blur=True,
        enable_banding=True,
        enable_view_color_shift=True,
        enable_noise=True,
        enable_reflection=True,
        enable_resample=False,
        use_jpeg=False,
        subpixel_mode="pentile",
        gamma_range=(0.85, 1.25),
        contrast_range=(1.05, 1.35),
        saturation_range=(1.05, 1.45),
        black_crush_range=(0.00, 0.08),
        highlight_clip_prob=0.3,
        highlight_clip_range=(0.92, 1.00),
        brightness_jitter=0.03,
        color_gain_range=(0.96, 1.04),
        subpixel_prob=0.7,
        subpixel_strength_range=(0.03, 0.15),
        subpixel_blur_sigma_range=(0.2, 0.8),
        display_blur_sigma_range=(0.2, 0.8),
        perspective_strength_range=(0.0, 0.018),
        camera_blur_sigma_range=(0.4, 1.5),
        banding_prob=0.5,
        banding_strength_range=(0.015, 0.08),
        banding_frequency_range=(4.0, 32.0),
        banding_tilt_range=(-0.15, 0.15),
        view_color_shift_prob=0.5,
        view_color_shift_strength_range=(0.01, 0.08),
        reflection_prob=0.35,
        reflection_strength_range=(0.02, 0.12),
        haze_strength_range=(0.00, 0.06),
        noise_std_range=(0.002, 0.02),
        motion_blur_prob=0.2,
        motion_blur_kernel_range=(3, 9),
        resample_scale_range=(0.92, 1.08),
        jpeg_quality_range=(55.0, 95.0),
        **kwargs,
    ):
        super().__init__()
        params = dict(
            p=p,
            enable_tone=enable_tone,
            enable_subpixel=enable_subpixel,
            enable_display_blur=enable_display_blur,
            enable_perspective=enable_perspective,
            enable_camera_blur=enable_camera_blur,
            enable_banding=enable_banding,
            enable_view_color_shift=enable_view_color_shift,
            enable_noise=enable_noise,
            enable_reflection=enable_reflection,
            enable_resample=enable_resample,
            use_jpeg=use_jpeg,
            subpixel_mode=subpixel_mode,
            gamma_range=gamma_range,
            contrast_range=contrast_range,
            saturation_range=saturation_range,
            black_crush_range=black_crush_range,
            highlight_clip_prob=highlight_clip_prob,
            highlight_clip_range=highlight_clip_range,
            brightness_jitter=brightness_jitter,
            color_gain_range=color_gain_range,
            subpixel_prob=subpixel_prob,
            subpixel_strength_range=subpixel_strength_range,
            subpixel_blur_sigma_range=subpixel_blur_sigma_range,
            display_blur_sigma_range=display_blur_sigma_range,
            perspective_strength_range=perspective_strength_range,
            camera_blur_sigma_range=camera_blur_sigma_range,
            banding_prob=banding_prob,
            banding_strength_range=banding_strength_range,
            banding_frequency_range=banding_frequency_range,
            banding_tilt_range=banding_tilt_range,
            view_color_shift_prob=view_color_shift_prob,
            view_color_shift_strength_range=view_color_shift_strength_range,
            reflection_prob=reflection_prob,
            reflection_strength_range=reflection_strength_range,
            haze_strength_range=haze_strength_range,
            noise_std_range=noise_std_range,
            motion_blur_prob=motion_blur_prob,
            motion_blur_kernel_range=motion_blur_kernel_range,
            resample_scale_range=resample_scale_range,
            jpeg_quality_range=jpeg_quality_range,
        )
        if config is not None:
            params.update(dict(config))
        params.update(kwargs)

        # Backward-compatible aliases from the first OLED implementation.
        if "gamma_min" in params or "gamma_max" in params:
            params["gamma_range"] = (
                params.pop("gamma_min", params["gamma_range"][0]),
                params.pop("gamma_max", params["gamma_range"][1]),
            )
        if "contrast_jitter" in params:
            jitter = float(params.pop("contrast_jitter"))
            params["contrast_range"] = (max(0.0, 1.0 - jitter), 1.0 + jitter)
        if "saturation_jitter" in params:
            jitter = float(params.pop("saturation_jitter"))
            params["saturation_range"] = (max(0.0, 1.0 - jitter), 1.0 + jitter)
        if "color_shift" in params:
            shift = float(params.pop("color_shift"))
            params["color_gain_range"] = (1.0 - shift, 1.0 + shift)
        if "grid_strength" in params:
            strength = float(params.pop("grid_strength"))
            params["subpixel_strength_range"] = (strength, max(strength, 0.12))
        if "glare_strength" in params:
            strength = float(params.pop("glare_strength"))
            params["reflection_strength_range"] = (0.0, max(strength, 0.0))
        if "noise_std" in params:
            std = float(params.pop("noise_std"))
            params["noise_std_range"] = (0.0, max(std, 0.0))
        if "blur_sigma" in params:
            params["camera_blur_sigma_range"] = params.pop("blur_sigma")
        if "blur_prob" in params:
            params["enable_camera_blur"] = float(params.pop("blur_prob")) > 0.0
        params.pop("moire_strength", None)
        params.pop("falloff_strength", None)

        self.p = float(params.pop("p"))
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("p must be in [0, 1]")

        for key in (
            "enable_tone",
            "enable_subpixel",
            "enable_display_blur",
            "enable_perspective",
            "enable_camera_blur",
            "enable_banding",
            "enable_view_color_shift",
            "enable_noise",
            "enable_reflection",
            "enable_resample",
            "use_jpeg",
        ):
            setattr(self, key, bool(params.pop(key)))

        self.subpixel_mode = str(params.pop("subpixel_mode")).lower()
        if self.subpixel_mode not in {"stripe", "pentile"}:
            raise ValueError("subpixel_mode must be 'stripe' or 'pentile'")

        self.gamma_range = validate_range("gamma_range", params.pop("gamma_range"))
        self.contrast_range = validate_range("contrast_range", params.pop("contrast_range"))
        self.saturation_range = validate_range("saturation_range", params.pop("saturation_range"))
        self.black_crush_range = validate_range("black_crush_range", params.pop("black_crush_range"))
        self.highlight_clip_range = validate_range("highlight_clip_range", params.pop("highlight_clip_range"))
        self.color_gain_range = validate_range("color_gain_range", params.pop("color_gain_range"))
        self.subpixel_strength_range = validate_range("subpixel_strength_range", params.pop("subpixel_strength_range"))
        self.subpixel_blur_sigma_range = validate_range("subpixel_blur_sigma_range", params.pop("subpixel_blur_sigma_range"))
        self.display_blur_sigma_range = validate_range("display_blur_sigma_range", params.pop("display_blur_sigma_range"))
        self.perspective_strength_range = validate_range("perspective_strength_range", params.pop("perspective_strength_range"))
        self.camera_blur_sigma_range = validate_range("camera_blur_sigma_range", params.pop("camera_blur_sigma_range"))
        self.banding_strength_range = validate_range("banding_strength_range", params.pop("banding_strength_range"))
        self.banding_frequency_range = validate_range("banding_frequency_range", params.pop("banding_frequency_range"))
        self.banding_tilt_range = validate_range("banding_tilt_range", params.pop("banding_tilt_range"))
        self.view_color_shift_strength_range = validate_range(
            "view_color_shift_strength_range", params.pop("view_color_shift_strength_range")
        )
        self.reflection_strength_range = validate_range("reflection_strength_range", params.pop("reflection_strength_range"))
        self.haze_strength_range = validate_range("haze_strength_range", params.pop("haze_strength_range"))
        self.noise_std_range = validate_range("noise_std_range", params.pop("noise_std_range"))
        self.motion_blur_kernel_range = validate_range("motion_blur_kernel_range", params.pop("motion_blur_kernel_range"))
        self.resample_scale_range = validate_range("resample_scale_range", params.pop("resample_scale_range"))
        self.jpeg_quality_range = validate_range("jpeg_quality_range", params.pop("jpeg_quality_range"))

        self.highlight_clip_prob = self._validate_prob("highlight_clip_prob", params.pop("highlight_clip_prob"))
        self.subpixel_prob = self._validate_prob("subpixel_prob", params.pop("subpixel_prob"))
        self.banding_prob = self._validate_prob("banding_prob", params.pop("banding_prob"))
        self.view_color_shift_prob = self._validate_prob(
            "view_color_shift_prob", params.pop("view_color_shift_prob")
        )
        self.reflection_prob = self._validate_prob("reflection_prob", params.pop("reflection_prob"))
        self.motion_blur_prob = self._validate_prob("motion_blur_prob", params.pop("motion_blur_prob"))
        self.brightness_jitter = float(params.pop("brightness_jitter"))

    @staticmethod
    def _validate_prob(name, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
        return value

    def _coordinate_grid(self, x):
        _, _, height, width = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        return yy, xx

    def _apply_oled_tone(self, x):
        # OLED display response: high saturation / contrast, deep blacks and
        # mild highlight roll-off or clipping before camera capture.
        batch = x.shape[0]
        gamma = sample_uniform(x, self.gamma_range, (batch, 1, 1, 1))
        contrast = sample_uniform(x, self.contrast_range, (batch, 1, 1, 1))
        saturation = sample_uniform(x, self.saturation_range, (batch, 1, 1, 1))
        black = sample_uniform(x, self.black_crush_range, (batch, 1, 1, 1))
        brightness = sample_uniform(
            x, (-self.brightness_jitter, self.brightness_jitter), (batch, 1, 1, 1)
        )
        gain = sample_uniform(x, self.color_gain_range, (batch, 3, 1, 1))

        x = x.clamp(1e-6, 1.0).pow(gamma)
        x = ((x - black).clamp_min(0.0) / (1.0 - black).clamp_min(1e-4)).pow(1.04)
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * contrast + mean + brightness
        luma = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x = (luma + (x - luma) * saturation) * gain

        if torch.rand((), device=x.device) < self.highlight_clip_prob:
            clip = sample_uniform(x, self.highlight_clip_range, (batch, 1, 1, 1))
            x = clip * torch.tanh(x / clip.clamp_min(1e-4))
        else:
            x = x - 0.10 * (x - 1.0).clamp_min(0.0)
        return x.clamp(0.0, 1.0)

    def _apply_subpixel_pentile(self, x):
        # PenTile / stripe masks make high-contrast edges show subtle colored
        # serration without destroying the image content.
        if not self.enable_subpixel or torch.rand((), device=x.device) >= self.subpixel_prob:
            return x
        batch, channels, height, width = x.shape
        if channels != 3:
            return x
        rows = torch.arange(height, device=x.device).reshape(height, 1)
        cols = torch.arange(width, device=x.device).reshape(1, width)
        parity = torch.remainder(rows + cols, 2).to(dtype=x.dtype)
        col3 = torch.remainder(cols, 3)

        if self.subpixel_mode == "stripe":
            mask = torch.stack(
                ((col3 == 0).to(x.dtype), (col3 == 1).to(x.dtype), (col3 == 2).to(x.dtype)),
                dim=0,
            )
            mask = 0.82 + 0.36 * mask
        else:
            red = (1.0 - parity) * 1.20 + parity * 0.84
            green = torch.ones_like(red) * 1.08
            blue = parity * 1.20 + (1.0 - parity) * 0.84
            mask = torch.stack((red, green, blue), dim=0)

        strength = sample_uniform(x, self.subpixel_strength_range, (batch, 1, 1, 1))
        mask = mask.unsqueeze(0)
        patterned = x * (1.0 + strength * (mask - 1.0))

        shifted = torch.cat(
            (
                torch.roll(patterned[:, 0:1], shifts=1, dims=3),
                patterned[:, 1:2],
                torch.roll(patterned[:, 2:3], shifts=-1, dims=3),
            ),
            dim=1,
        )
        sigma = sample_uniform(x, self.subpixel_blur_sigma_range, (batch,))
        edge = (x - gaussian_blur(x, 3, sigma)).abs().mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        return x * (1.0 - strength) + (0.78 * patterned + 0.22 * shifted) * strength + edge * (patterned - x)

    def _apply_display_blur(self, x):
        # Screen emission diffuses slightly through OLED stack and cover glass.
        if not self.enable_display_blur:
            return x
        sigma = sample_uniform(x, self.display_blur_sigma_range, (x.shape[0],))
        blurred = gaussian_blur(x, 3, sigma)
        return 0.82 * x + 0.18 * blurred

    def _apply_perspective(self, x):
        if not self.enable_perspective:
            return x
        batch, _, _, _ = x.shape
        yy, xx = self._coordinate_grid(x)
        xx = xx[None].expand(batch, -1, -1)
        yy = yy[None].expand(batch, -1, -1)
        strength = sample_uniform(x, self.perspective_strength_range, (batch, 1, 1))
        sign = torch.where(
            torch.rand(batch, 2, 1, device=x.device) < 0.5,
            -torch.ones(batch, 2, 1, device=x.device, dtype=x.dtype),
            torch.ones(batch, 2, 1, device=x.device, dtype=x.dtype),
        )
        grid_x = xx + strength * sign[:, 0:1] * yy + 0.5 * strength * yy.square()
        grid_y = yy + strength * sign[:, 1:2] * xx
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def _apply_camera_blur(self, x):
        # Camera defocus / lens PSF after the displayed image is photographed.
        if not self.enable_camera_blur:
            return x
        sigma = sample_uniform(x, self.camera_blur_sigma_range, (x.shape[0],))
        blurred = gaussian_blur(x, 5, sigma)
        return 0.35 * x + 0.65 * blurred

    def _apply_pwm_banding(self, x):
        # OLED PWM plus rolling shutter produces horizontal brightness bands;
        # tilt approximates phone/camera misalignment.
        if not self.enable_banding or torch.rand((), device=x.device) >= self.banding_prob:
            return x
        batch, _, height, width = x.shape
        yy, xx = self._coordinate_grid(x)
        rows = yy[:, :1].reshape(1, 1, height, 1)
        cols = xx[:1, :].reshape(1, 1, 1, width)
        strength = sample_uniform(x, self.banding_strength_range, (batch, 1, 1, 1))
        freq = sample_uniform(x, self.banding_frequency_range, (batch, 1, 1, 1))
        phase = sample_uniform(x, (0.0, 2.0 * math.pi), (batch, 1, 1, 1))
        tilt = sample_uniform(x, self.banding_tilt_range, (batch, 1, 1, 1))
        coord = rows + tilt * cols
        high = torch.sin(2.0 * math.pi * freq * (coord + 1.0) * 0.5 + phase)
        low = torch.sin(2.0 * math.pi * (freq * 0.25) * (coord + 1.0) * 0.5 + 0.37 * phase)
        band = 1.0 + strength * (0.65 * high + 0.35 * low)
        return x * band

    def _apply_view_color_shift(self, x):
        # Viewing-angle color shift is low frequency and channel dependent.
        if not self.enable_view_color_shift or torch.rand((), device=x.device) >= self.view_color_shift_prob:
            return x
        batch, channels, height, width = x.shape
        if channels != 3:
            return x
        yy, xx = self._coordinate_grid(x)
        rows = yy[:, :1].reshape(1, 1, height, 1)
        cols = xx[:1, :].reshape(1, 1, 1, width)
        ax = sample_uniform(x, (-1.0, 1.0), (batch, 1, 1, 1))
        ay = sample_uniform(x, (-1.0, 1.0), (batch, 1, 1, 1))
        field = ax * cols + ay * rows
        field = field / field.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        strength = sample_uniform(x, self.view_color_shift_strength_range, (batch, 1, 1, 1))
        channel_vec = sample_uniform(x, (-1.0, 1.0), (batch, 3, 1, 1))
        channel_vec = channel_vec - channel_vec.mean(dim=1, keepdim=True)
        return x * (1.0 + strength * field * channel_vec)

    def _apply_sensor_noise(self, x):
        if not self.enable_noise:
            return x
        std = sample_uniform(x, self.noise_std_range, (x.shape[0], 1, 1, 1))
        # sqrt has an infinite derivative at zero. OLED tone mapping creates
        # exact black pixels, so sqrt(clamp_min(0)) produced finite images but
        # Inf/NaN gradients back to the watermark U-Net. A small positive floor
        # preserves signal-dependent shot noise while bounding its Jacobian.
        shot_scale = x.clamp_min(1e-4).sqrt()
        shot = torch.randn_like(x) * std * shot_scale
        read = torch.randn_like(x) * (0.45 * std)
        return x + shot + read

    def _apply_motion_blur(self, x):
        if self.motion_blur_prob == 0.0 or torch.rand((), device=x.device) >= self.motion_blur_prob:
            return x
        low, high = int(round(self.motion_blur_kernel_range[0])), int(round(self.motion_blur_kernel_range[1]))
        if low % 2 == 0:
            low += 1
        if high % 2 == 0:
            high -= 1
        kernel_size = max(3, low if high < low else int(torch.randint(low, high + 1, (), device=x.device).item()) | 1)
        pad = kernel_size // 2
        kernel = torch.zeros((x.shape[1], 1, kernel_size, kernel_size), device=x.device, dtype=x.dtype)
        if torch.rand((), device=x.device) < 0.5:
            kernel[:, 0, pad, :] = 1.0 / kernel_size
        else:
            kernel[:, 0, :, pad] = 1.0 / kernel_size
        padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(padded, kernel, groups=x.shape[1])

    def _apply_reflection_haze(self, x):
        # Glass cover adds weak haze and localized white/gray reflection.
        if not self.enable_reflection or torch.rand((), device=x.device) >= self.reflection_prob:
            return x
        batch, _, height, width = x.shape
        yy, xx = self._coordinate_grid(x)
        rows = yy[:, :1].reshape(1, 1, height, 1)
        cols = xx[:1, :].reshape(1, 1, 1, width)
        haze = sample_uniform(x, self.haze_strength_range, (batch, 1, 1, 1))
        haze_color = sample_uniform(x, (0.82, 1.0), (batch, 3, 1, 1))
        x = x * (1.0 - haze) + haze_color * haze

        cx = sample_uniform(x, (-0.8, 0.8), (batch, 1, 1, 1))
        cy = sample_uniform(x, (-0.8, 0.8), (batch, 1, 1, 1))
        sx = sample_uniform(x, (0.18, 0.75), (batch, 1, 1, 1))
        sy = sample_uniform(x, (0.08, 0.42), (batch, 1, 1, 1))
        angle = sample_uniform(x, (-0.9, 0.9), (batch, 1, 1, 1))
        x0 = cols - cx
        y0 = rows - cy
        xr = torch.cos(angle) * x0 + torch.sin(angle) * y0
        yr = -torch.sin(angle) * x0 + torch.cos(angle) * y0
        mask = torch.exp(-(xr.square() / sx.square().clamp_min(1e-4) + yr.square() / sy.square().clamp_min(1e-4)))
        strength = sample_uniform(x, self.reflection_strength_range, (batch, 1, 1, 1))
        return x + mask * strength

    def _apply_resample(self, x):
        if not self.enable_resample:
            return x
        _, _, height, width = x.shape
        scale = float(sample_uniform(x, self.resample_scale_range, ()).item())
        mid_h = max(2, int(round(height * scale)))
        mid_w = max(2, int(round(width * scale)))
        if mid_h == height and mid_w == width:
            return x
        x = F.interpolate(x, size=(mid_h, mid_w), mode="bilinear", align_corners=False)
        return F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)

    def _apply_jpeg_proxy(self, x):
        # Differentiable JPEG proxy: block-average mixing and soft quantization.
        # It is off by default for training stability.
        if not self.use_jpeg:
            return x
        batch, channels, height, width = x.shape
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        pooled = F.avg_pool2d(padded, kernel_size=8, stride=8)
        blocky = F.interpolate(pooled, size=padded.shape[-2:], mode="nearest")[..., :height, :width]
        quality = sample_uniform(x, self.jpeg_quality_range, (batch, 1, 1, 1))
        mix = ((100.0 - quality) / 100.0).clamp(0.0, 0.45)
        quant = torch.round(x * 255.0) / 255.0
        quant = x + (quant - x).detach()
        return x * (1.0 - mix) + (0.75 * quant + 0.25 * blocky) * mix

    def forward(self, x):
        """Degrade ``x`` of shape ``[B, 3, H, W]`` in range ``[0, 1]``."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("OLED_Layer expects x with shape [B, 3, H, W]")
        clean = x.clamp(0.0, 1.0)
        if self.p == 0.0 or (self.p < 1.0 and torch.rand((), device=x.device) >= self.p):
            return clean.to(dtype=x.dtype)

        y = clean
        if self.enable_tone:
            y = self._apply_oled_tone(y)
        y = self._apply_subpixel_pentile(y)
        y = self._apply_display_blur(y)
        y = self._apply_perspective(y)
        y = self._apply_camera_blur(y)
        y = self._apply_pwm_banding(y)
        y = self._apply_view_color_shift(y)
        y = self._apply_sensor_noise(y)
        y = self._apply_motion_blur(y)
        y = self._apply_reflection_haze(y)
        y = self._apply_resample(y)
        y = self._apply_jpeg_proxy(y)
        return y.clamp(0.0, 1.0).to(dtype=x.dtype)


OLEDLayer = OLED_Layer
OLEDNoiseLayer = OLED_Layer
