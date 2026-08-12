"""Differentiable LED display-camera degradation layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import gaussian_blur, sample_uniform, validate_range


_SEVERITY_PRESETS = {
    "mild": {
        "downsample_scale": (0.68, 0.90),
        "led_pitch": (4, 7),
        "dot_radius": (0.34, 0.44),
        "gap_strength": (0.10, 0.22),
        "dot_strength": (0.90, 1.05),
        "bloom_strength": (0.02, 0.07),
        "brightness_gain": (0.98, 1.08),
        "contrast_gain": (0.94, 1.06),
        "gamma": (0.90, 1.12),
        "noise_std": (0.000, 0.006),
        "blur_sigma": (0.10, 0.35),
        "perspective": (0.000, 0.010),
        "resample_jitter": (0.96, 1.04),
        "scanline_strength": (0.01, 0.035),
        "moire_strength": (0.01, 0.035),
    },
    "medium": {
        "downsample_scale": (0.45, 0.75),
        "led_pitch": (5, 10),
        "dot_radius": (0.28, 0.40),
        "gap_strength": (0.18, 0.36),
        "dot_strength": (0.95, 1.16),
        "bloom_strength": (0.05, 0.14),
        "brightness_gain": (0.98, 1.18),
        "contrast_gain": (0.90, 1.14),
        "gamma": (0.82, 1.22),
        "noise_std": (0.002, 0.014),
        "blur_sigma": (0.20, 0.65),
        "perspective": (0.004, 0.025),
        "resample_jitter": (0.92, 1.08),
        "scanline_strength": (0.02, 0.070),
        "moire_strength": (0.025, 0.080),
    },
    "strong": {
        "downsample_scale": (0.30, 0.55),
        "led_pitch": (7, 14),
        "dot_radius": (0.22, 0.34),
        "gap_strength": (0.30, 0.55),
        "dot_strength": (1.02, 1.32),
        "bloom_strength": (0.10, 0.24),
        "brightness_gain": (1.02, 1.30),
        "contrast_gain": (0.84, 1.22),
        "gamma": (0.72, 1.35),
        "noise_std": (0.006, 0.026),
        "blur_sigma": (0.35, 1.10),
        "perspective": (0.012, 0.050),
        "resample_jitter": (0.86, 1.14),
        "scanline_strength": (0.04, 0.120),
        "moire_strength": (0.050, 0.140),
    },
}


class LEDLayer(nn.Module):
    """
    LED display-camera degradation simulator.

    The modeled path is:
        image -> low-resolution LED display -> LED bead / sub-pixel emission
        -> active light response -> camera capture -> degraded image.

    The project uses ``[0, 1]`` tensors at the degradation-layer boundary. For
    standalone use, ``input_range="auto"`` also accepts ``[-1, 1]`` tensors and
    maps the output back to the original range.
    """

    def __init__(
        self,
        config=None,
        p=1.0,
        severity="medium",
        input_range="auto",
        subpixel_mode="rgb_triplet",
        dot_shape="gaussian",
        enable_scanline=True,
        scanline_prob=0.35,
        enable_moire=True,
        enable_perspective=True,
        highlight_clip=True,
        **kwargs,
    ):
        super().__init__()
        params = dict(
            p=p,
            severity=severity,
            input_range=input_range,
            subpixel_mode=subpixel_mode,
            dot_shape=dot_shape,
            enable_scanline=enable_scanline,
            scanline_prob=scanline_prob,
            enable_moire=enable_moire,
            enable_perspective=enable_perspective,
            highlight_clip=highlight_clip,
        )
        if config is not None:
            params.update(dict(config))
        params.update(kwargs)

        self.p = float(params.pop("p"))
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("p must be in [0, 1]")

        self.severity = str(params.pop("severity")).lower()
        if self.severity not in {"mild", "medium", "strong", "random"}:
            raise ValueError("severity must be one of: mild, medium, strong, random")

        self.input_range = str(params.pop("input_range")).lower()
        if self.input_range not in {"auto", "0_1", "-1_1"}:
            raise ValueError("input_range must be 'auto', '0_1', or '-1_1'")

        self.subpixel_mode = str(params.pop("subpixel_mode")).lower()
        if self.subpixel_mode not in {"mono_dot", "rgb_triplet"}:
            raise ValueError("subpixel_mode must be 'mono_dot' or 'rgb_triplet'")

        self.dot_shape = str(params.pop("dot_shape")).lower()
        if self.dot_shape not in {"gaussian", "soft_disk"}:
            raise ValueError("dot_shape must be 'gaussian' or 'soft_disk'")

        self.enable_scanline = bool(params.pop("enable_scanline"))
        self.scanline_prob = float(params.pop("scanline_prob"))
        if not 0.0 <= self.scanline_prob <= 1.0:
            raise ValueError("scanline_prob must be in [0, 1]")
        self.enable_moire = bool(params.pop("enable_moire"))
        self.enable_perspective = bool(params.pop("enable_perspective"))
        self.highlight_clip = bool(params.pop("highlight_clip"))

        # Backward-compatible aliases from the first LED implementation.
        self.led_pitch = self._range_from_params(
            params, "led_pitch", "pixel_pitch_min", "pixel_pitch_max", None
        )
        self.downsample_scale = self._range_from_params(
            params, "downsample_scale", "downsample_min", "downsample_max", None
        )
        self.dot_radius = self._range_from_params(
            params, "dot_radius", None, None, None
        )
        self.gap_strength = self._range_from_params(
            params, "gap_strength", None, None, params.pop("grid_strength", None)
        )
        self.dot_strength = self._range_from_params(
            params, "dot_strength", None, None, None
        )
        self.bloom_strength = self._range_from_params(
            params, "bloom_strength", None, None, None
        )
        self.brightness_gain = self._range_from_params(
            params,
            "brightness_gain",
            None,
            None,
            self._jitter_to_gain(params.pop("brightness_jitter", None)),
        )
        self.contrast_gain = self._range_from_params(
            params, "contrast_gain", None, None, None
        )
        self.gamma = self._range_from_params(params, "gamma", None, None, None)
        self.color_shift = float(params.pop("color_shift", 0.08))
        self.noise_std = self._range_from_params(
            params, "noise_std", None, None, params.pop("noise_std_max", None)
        )
        self.blur_sigma = self._range_from_params(
            params, "blur_sigma", None, None, None
        )
        self.perspective = self._range_from_params(
            params, "perspective", None, None, None
        )
        self.resample_jitter = self._range_from_params(
            params, "resample_jitter", None, None, None
        )
        self.scanline_strength = self._range_from_params(
            params, "scanline_strength", None, None, None
        )
        self.scanline_frequency = self._range_from_params(
            params, "scanline_frequency", None, None, (4.0, 18.0)
        )
        self.moire_strength = self._range_from_params(
            params, "moire_strength", None, None, None
        )
        self.view_falloff_strength = float(params.pop("view_falloff_strength", 0.10))

    @staticmethod
    def _jitter_to_gain(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return value
        value = float(value)
        return (1.0, 1.0 + max(value, 0.0))

    @staticmethod
    def _as_range(name, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError(f"{name} must contain exactly two values")
            return validate_range(name, value)
        value = float(value)
        return (value, value)

    def _range_from_params(self, params, key, min_key, max_key, default):
        if key in params:
            return self._as_range(key, params.pop(key))
        if min_key is not None and max_key is not None and min_key in params and max_key in params:
            return validate_range(key, (params.pop(min_key), params.pop(max_key)))
        return self._as_range(key, default)

    def _select_severity(self):
        if self.severity != "random":
            return self.severity
        choices = ("mild", "medium", "strong")
        return choices[int(torch.randint(0, 3, ()).item())]

    def _range_for(self, key, severity):
        override = getattr(self, key)
        return override if override is not None else _SEVERITY_PRESETS[severity][key]

    def _sample_params(self, x):
        severity = self._select_severity()
        pitch_range = self._range_for("led_pitch", severity)
        pitch_min, pitch_max = int(round(pitch_range[0])), int(round(pitch_range[1]))
        pitch = int(torch.randint(pitch_min, pitch_max + 1, (), device=x.device).item())
        return {
            "severity": severity,
            "pitch": max(2, pitch),
            "downsample_scale": sample_uniform(x, self._range_for("downsample_scale", severity), ()).item(),
            "dot_radius": sample_uniform(x, self._range_for("dot_radius", severity), ()).item(),
            "gap_strength": sample_uniform(x, self._range_for("gap_strength", severity), ()).item(),
            "dot_strength": sample_uniform(x, self._range_for("dot_strength", severity), ()).item(),
            "bloom_strength": sample_uniform(x, self._range_for("bloom_strength", severity), (x.shape[0], 1, 1, 1)),
            "brightness_gain": sample_uniform(x, self._range_for("brightness_gain", severity), (x.shape[0], 1, 1, 1)),
            "contrast_gain": sample_uniform(x, self._range_for("contrast_gain", severity), (x.shape[0], 1, 1, 1)),
            "gamma": sample_uniform(x, self._range_for("gamma", severity), (x.shape[0], 1, 1, 1)),
            "noise_std": sample_uniform(x, self._range_for("noise_std", severity), (x.shape[0], 1, 1, 1)),
            "blur_sigma": sample_uniform(x, self._range_for("blur_sigma", severity), (x.shape[0],)),
            "perspective": sample_uniform(x, self._range_for("perspective", severity), (x.shape[0], 1, 1)),
            "resample_jitter": sample_uniform(x, self._range_for("resample_jitter", severity), ()).item(),
            "scanline_strength": sample_uniform(x, self._range_for("scanline_strength", severity), (x.shape[0], 1, 1, 1)),
            "scanline_frequency": sample_uniform(x, self.scanline_frequency, (x.shape[0], 1, 1, 1)),
            "moire_strength": sample_uniform(x, self._range_for("moire_strength", severity), (x.shape[0], 1, 1, 1)),
        }

    def _to_unit_range(self, x):
        if self.input_range == "-1_1" or (
            self.input_range == "auto" and x.detach().amin() < -0.05
        ):
            return x.add(1.0).mul(0.5).clamp(0.0, 1.0), True
        return x.clamp(0.0, 1.0), False

    def _from_unit_range(self, x, was_minus_one_to_one):
        if was_minus_one_to_one:
            return x.mul(2.0).sub(1.0).clamp(-1.0, 1.0)
        return x.clamp(0.0, 1.0)

    def _coordinate_grid(self, x):
        _, _, height, width = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        return yy, xx

    def _simulate_lowres_display(self, x, params):
        # LED walls have a lower effective spatial resolution than the camera
        # output. Rendering low-res first creates true detail loss and blocky
        # edges before bead structure is applied.
        _, _, height, width = x.shape
        scale = float(params["downsample_scale"])
        pitch = int(params["pitch"])
        low_h = max(2, min(height, int(round(height * scale))))
        low_w = max(2, min(width, int(round(width * scale))))
        low_h = max(2, min(low_h, max(2, height // max(1, pitch // 2))))
        low_w = max(2, min(low_w, max(2, width // max(1, pitch // 2))))
        low = F.interpolate(x, size=(low_h, low_w), mode="area")
        blocky = F.interpolate(low, size=(height, width), mode="nearest")
        smooth = F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)
        mix = sample_uniform(x, (0.25, 0.65), (x.shape[0], 1, 1, 1))
        return blocky * mix + smooth * (1.0 - mix)

    def _build_led_pattern(self, x, params):
        # The mask represents light emitted by LED beads and darker gaps
        # between beads. It is built in display coordinates, not overlaid as an
        # independent texture after camera capture.
        _, channels, height, width = x.shape
        pitch = int(params["pitch"])
        radius = float(params["dot_radius"])
        yy = torch.arange(height, device=x.device, dtype=x.dtype).reshape(height, 1)
        xx = torch.arange(width, device=x.device, dtype=x.dtype).reshape(1, width)
        phase_x = torch.randint(0, pitch, (), device=x.device).to(dtype=x.dtype)
        phase_y = torch.randint(0, pitch, (), device=x.device).to(dtype=x.dtype)
        jitter = 0.08 * pitch * torch.sin(0.19 * yy + 0.13 * xx)
        cell_x = torch.remainder(xx + phase_x + jitter, pitch) / max(pitch - 1, 1)
        cell_y = torch.remainder(yy + phase_y - jitter, pitch) / max(pitch - 1, 1)

        if self.subpixel_mode == "rgb_triplet" and channels == 3:
            centers = x.new_tensor([0.28, 0.50, 0.72]).reshape(3, 1, 1)
            dx = cell_x[None] - centers
            dy = cell_y[None] - 0.50
            distance2 = dx.square() + dy.square()
        else:
            dx = cell_x - 0.50
            dy = cell_y - 0.50
            distance2 = (dx.square() + dy.square())[None].expand(channels, -1, -1)

        if self.dot_shape == "soft_disk":
            bead = (1.0 - torch.sqrt(distance2.clamp_min(1e-6)) / radius).clamp(0.0, 1.0)
        else:
            bead = torch.exp(-distance2 / max(2.0 * radius * radius, 1e-6))
        bead = bead / bead.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return bead[None]

    def _apply_subpixel_pattern(self, x, pattern, params):
        # Dark gaps reduce emitted light while bead centers keep or amplify it.
        gap = float(params["gap_strength"])
        dot = float(params["dot_strength"])
        emission = (1.0 - gap) + gap * pattern
        return x * emission * dot

    def _apply_bloom(self, x, params):
        # Bright LED content bleeds into neighboring pixels in camera capture.
        gamma = params["gamma"]
        contrast = params["contrast_gain"]
        gain = params["brightness_gain"]
        x = x.clamp(1e-6, 1.0).pow(gamma)
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * contrast + mean
        x = x * gain
        highlight = (x - 0.62).clamp_min(0.0)
        bloom = gaussian_blur(highlight, 5, sample_uniform(x, (0.8, 1.7), (x.shape[0],)))
        x = x + params["bloom_strength"] * bloom
        if self.highlight_clip:
            x = x - 0.08 * (x - 1.0).clamp_min(0.0)
        return x

    def _apply_color_response(self, x):
        if x.shape[1] != 3:
            return x
        gain = sample_uniform(
            x, (1.0 - self.color_shift, 1.0 + self.color_shift), (x.shape[0], 3, 1, 1)
        )
        shifted = torch.cat(
            (
                torch.roll(x[:, 0:1], shifts=1, dims=3),
                x[:, 1:2],
                torch.roll(x[:, 2:3], shifts=-1, dims=2),
            ),
            dim=1,
        )
        return 0.86 * x * gain + 0.14 * shifted

    def _apply_scanline_artifact(self, x, params):
        # Scanlines approximate rolling shutter / LED refresh mismatch.
        if not self.enable_scanline or torch.rand((), device=x.device) >= self.scanline_prob:
            return x
        batch, _, height, _ = x.shape
        rows = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
        phase = sample_uniform(x, (0.0, 6.283185307), (batch, 1, 1, 1))
        band_center = sample_uniform(x, (-1.0, 1.0), (batch, 1, 1, 1))
        freq = params["scanline_frequency"]
        stripe = torch.sin(freq * rows.reshape(1, 1, height, 1) + phase)
        rolling_band = torch.exp(
            -((rows.reshape(1, 1, height, 1) - band_center).square()) / 0.08
        )
        modulation = 1.0 + params["scanline_strength"] * (0.6 * stripe - 0.4 * rolling_band)
        return x * modulation

    def _apply_moire_artifact(self, x, params):
        # A tiny resample plus directional waves creates sampling interference
        # between the LED grid and the camera sensor.
        if not self.enable_moire:
            return x
        batch, _, height, width = x.shape
        scale = float(params["resample_jitter"])
        mid_h = max(2, int(round(height * scale)))
        mid_w = max(2, int(round(width * scale)))
        if mid_h != height or mid_w != width:
            x = F.interpolate(x, size=(mid_h, mid_w), mode="bilinear", align_corners=False)
            x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)

        yy, xx = self._coordinate_grid(x)
        theta = sample_uniform(x, (0.0, 3.141592654), (batch, 1, 1, 1))
        freq = sample_uniform(x, (10.0, 36.0), (batch, 1, 1, 1))
        phase = sample_uniform(x, (0.0, 6.283185307), (batch, 1, 1, 1))
        wave = torch.sin(
            freq * (torch.cos(theta) * xx[None, None] + torch.sin(theta) * yy[None, None])
            + phase
        )
        tilt = sample_uniform(x, (-1.0, 1.0), (batch, 1, 1, 1))
        view = 1.0 - self.view_falloff_strength * (tilt * xx[None, None]).abs()
        return x * view + params["moire_strength"] * wave

    def _perspective_warp_if_available(self, x, params):
        if not self.enable_perspective:
            return x
        batch, _, _, _ = x.shape
        yy, xx = self._coordinate_grid(x)
        xx = xx[None].expand(batch, -1, -1)
        yy = yy[None].expand(batch, -1, -1)
        strength = params["perspective"]
        signs = torch.where(
            torch.rand(batch, 3, 1, device=x.device) < 0.5,
            -torch.ones(batch, 3, 1, device=x.device, dtype=x.dtype),
            torch.ones(batch, 3, 1, device=x.device, dtype=x.dtype),
        )
        grid_x = xx + strength * signs[:, 0:1] * yy + strength * signs[:, 1:2] * yy.square()
        grid_y = yy + strength * signs[:, 2:3] * xx
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def _apply_camera_degradation(self, x, params):
        # Camera capture adds optical blur, small geometric disturbance,
        # resampling loss and sensor noise after the LED image is emitted.
        x = self._perspective_warp_if_available(x, params)
        x = gaussian_blur(x, 3, params["blur_sigma"])
        if torch.rand((), device=x.device) < 0.5:
            _, _, height, width = x.shape
            scale = sample_uniform(x, (0.94, 1.06), ()).item()
            mid_h = max(2, int(round(height * scale)))
            mid_w = max(2, int(round(width * scale)))
            x = F.interpolate(x, size=(mid_h, mid_w), mode="bilinear", align_corners=False)
            x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        x = x + torch.randn_like(x) * params["noise_std"]
        return x

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("LEDLayer expects x with shape [B, C, H, W]")
        unit, was_minus_one_to_one = self._to_unit_range(x)
        if self.p == 0.0 or (self.p < 1.0 and torch.rand((), device=x.device) >= self.p):
            return self._from_unit_range(unit, was_minus_one_to_one).to(dtype=x.dtype)

        params = self._sample_params(unit)
        y = self._simulate_lowres_display(unit, params)
        pattern = self._build_led_pattern(y, params)
        y = self._apply_subpixel_pattern(y, pattern, params)
        y = self._apply_bloom(y, params)
        y = self._apply_color_response(y)
        y = self._apply_scanline_artifact(y, params)
        y = self._apply_moire_artifact(y, params)
        y = self._apply_camera_degradation(y, params)
        y = y.clamp(0.0, 1.0)
        return self._from_unit_range(y, was_minus_one_to_one).to(dtype=x.dtype)


# Backward-compatible name used by build_noise_layer.py and existing configs.
LEDNoiseLayer = LEDLayer
