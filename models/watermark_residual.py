"""Shared cover-dependent guidance and watermark residual constraints."""

import math

import torch
import torch.nn.functional as F


def _normalize_activity_map(activity, eps=1e-6):
    """Normalize a non-negative activity map without max-outlier scaling."""
    flat = activity.flatten(1)
    scale = (
        flat.mean(dim=1, keepdim=True)
        + 2.0 * flat.std(dim=1, keepdim=True, unbiased=False)
    )
    scale = scale.view(-1, 1, 1, 1)
    return (activity / (scale + eps)).clamp(0.0, 1.0)


def _validate_odd_scales(name, values):
    if not isinstance(values, (list, tuple)) or not values:
        raise TypeError(f"{name} must be a non-empty list of odd integers")
    scales = tuple(int(value) for value in values)
    if any(scale < 1 or scale % 2 == 0 for scale in scales):
        raise ValueError(f"{name} must contain positive odd integers, got {scales}")
    return scales


def _box_blur(image, kernel_size):
    if kernel_size == 1:
        return image
    padding = kernel_size // 2
    return F.avg_pool2d(
        F.pad(image, (padding, padding, padding, padding), mode="reflect"),
        kernel_size=kernel_size,
        stride=1,
    )


def _sobel_activity(luminance, smoothing_kernel=1):
    source = _box_blur(luminance, smoothing_kernel)
    sobel_x = luminance.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    padded = F.pad(source, (1, 1, 1, 1), mode="reflect")
    grad_x = F.conv2d(padded, sobel_x)
    grad_y = F.conv2d(padded, sobel_y)
    return torch.sqrt((grad_x.square() + grad_y.square()).clamp_min(0.0))


def _local_std_activity(luminance, kernel_size, eps=1e-6):
    local_mean = _box_blur(luminance, kernel_size)
    local_mean_sq = _box_blur(luminance.square(), kernel_size)
    variance = (local_mean_sq - local_mean.square()).clamp_min(0.0)
    return torch.where(
        variance > eps,
        torch.sqrt(variance),
        torch.zeros_like(variance),
    )


def _build_strict_multiscale_allowance(luminance, config, eps):
    edge_scales = _validate_odd_scales(
        "edge_scales", config.get("edge_scales", [3, 5, 9])
    )
    texture_scales = _validate_odd_scales(
        "texture_scales", config.get("texture_scales", [3, 5, 9])
    )
    target_mask_area = float(config.get("target_mask_area", 0.35))
    mask_temperature = float(config.get("mask_temperature", 0.08))
    if not 0.0 < target_mask_area < 1.0:
        raise ValueError("target_mask_area must be in (0, 1)")
    if not math.isfinite(mask_temperature) or mask_temperature <= 0.0:
        raise ValueError("mask_temperature must be finite and positive")

    edge = torch.stack(
        [
            _normalize_activity_map(
                _sobel_activity(luminance, smoothing_kernel=scale), eps=eps
            )
            for scale in edge_scales
        ],
        dim=0,
    ).mean(dim=0)
    texture = torch.stack(
        [
            _normalize_activity_map(
                _local_std_activity(luminance, scale, eps=eps), eps=eps
            )
            for scale in texture_scales
        ],
        dim=0,
    ).mean(dim=0)

    edge_weight = float(config.get("edge_weight", 0.4))
    texture_weight = float(config.get("texture_weight", 0.6))
    if edge_weight < 0.0 or texture_weight < 0.0:
        raise ValueError("edge/texture guidance weights must be non-negative")
    weight_sum = edge_weight + texture_weight
    if weight_sum <= 0.0:
        raise ValueError("at least one edge/texture guidance weight must be positive")
    activity = (edge_weight * edge + texture_weight * texture) / weight_sum

    # A detached per-image quantile keeps the preferred support comparable
    # across covers without turning the mask into a discontinuous hard top-k.
    threshold = torch.quantile(
        activity.flatten(1),
        1.0 - target_mask_area,
        dim=1,
        keepdim=True,
    ).view(-1, 1, 1, 1)
    return torch.sigmoid((activity - threshold) / mask_temperature)


def build_edge_texture_guidance(cover_01, config=None):
    """Build the fixed Sobel/local-texture allowance and penalty maps."""
    config = config or {}
    mode = str(config.get("mode", "legacy")).strip().lower()
    if mode not in {"legacy", "strict_multiscale"}:
        raise ValueError(
            "region guidance mode must be 'legacy' or 'strict_multiscale', "
            f"got {mode!r}"
        )
    edge_weight = float(config.get("edge_weight", 0.4))
    texture_weight = float(config.get("texture_weight", 0.6))
    texture_kernel = int(config.get("texture_kernel", 7))
    dilation_kernel = int(config.get("dilation_kernel", 5))
    blur_kernel = int(config.get("blur_kernel", 5))
    gamma = float(config.get("gamma", 1.5))
    min_penalty = float(config.get("min_penalty", 0.1))
    eps = float(config.get("eps", 1e-6))

    for name, kernel in (
        ("texture_kernel", texture_kernel),
        ("dilation_kernel", dilation_kernel),
        ("blur_kernel", blur_kernel),
    ):
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer, got {kernel}")
    if edge_weight < 0.0 or texture_weight < 0.0:
        raise ValueError("edge/texture guidance weights must be non-negative")
    weight_sum = edge_weight + texture_weight
    if weight_sum <= 0.0:
        raise ValueError("at least one edge/texture guidance weight must be positive")
    if gamma <= 0.0:
        raise ValueError("region guidance gamma must be positive")
    if min_penalty < 0.0:
        raise ValueError("region guidance min_penalty must be non-negative")

    image = cover_01.detach().float().clamp(0.0, 1.0)
    luminance = (
        0.299 * image[:, 0:1]
        + 0.587 * image[:, 1:2]
        + 0.114 * image[:, 2:3]
    )

    if mode == "strict_multiscale":
        allowance = _build_strict_multiscale_allowance(luminance, config, eps)
    else:
        edge = _normalize_activity_map(_sobel_activity(luminance), eps=eps)
        texture = _normalize_activity_map(
            _local_std_activity(luminance, texture_kernel, eps=eps), eps=eps
        )
        allowance = (edge_weight * edge + texture_weight * texture) / weight_sum

    if dilation_kernel > 1:
        allowance = F.max_pool2d(
            allowance,
            kernel_size=dilation_kernel,
            stride=1,
            padding=dilation_kernel // 2,
        )
    if blur_kernel > 1:
        blur_padding = blur_kernel // 2
        allowance = F.avg_pool2d(
            F.pad(
                allowance,
                (blur_padding, blur_padding, blur_padding, blur_padding),
                mode="reflect",
            ),
            kernel_size=blur_kernel,
            stride=1,
        )
    allowance = allowance.clamp(0.0, 1.0)
    penalty = (1.0 - allowance).pow(gamma) + min_penalty
    return allowance, penalty


def get_residual_constraint_settings(config=None):
    """Validate and normalize the optional residual constraint settings."""
    config = config or {}
    if not isinstance(config, dict):
        raise TypeError("residual_constraint must be a mapping")

    enabled = config.get("enabled", False)
    if type(enabled) is not bool:
        raise TypeError("residual_constraint.enabled must be true or false")
    uses_spatial_budget = any(
        key in config
        for key in ("flat_max_abs_delta_01", "texture_max_abs_delta_01", "mask_power")
    )
    mode = "spatial_budget" if uses_spatial_budget else "legacy"
    max_abs_delta_01 = float(config.get("max_abs_delta_01", 0.03))
    flat_floor = float(config.get("flat_floor", 0.2))
    flat_max_abs_delta_01 = float(
        config.get("flat_max_abs_delta_01", max_abs_delta_01 * flat_floor)
    )
    texture_max_abs_delta_01 = float(
        config.get("texture_max_abs_delta_01", max_abs_delta_01)
    )
    mask_power = float(config.get("mask_power", 1.0))
    eps = float(config.get("eps", 1e-6))
    if not math.isfinite(max_abs_delta_01) or max_abs_delta_01 <= 0.0:
        raise ValueError("max_abs_delta_01 must be finite and positive")
    if not math.isfinite(flat_floor) or not 0.0 <= flat_floor <= 1.0:
        raise ValueError("flat_floor must be in [0, 1]")
    if (
        not math.isfinite(flat_max_abs_delta_01)
        or flat_max_abs_delta_01 <= 0.0
    ):
        raise ValueError("flat_max_abs_delta_01 must be finite and positive")
    if (
        not math.isfinite(texture_max_abs_delta_01)
        or texture_max_abs_delta_01 <= 0.0
    ):
        raise ValueError("texture_max_abs_delta_01 must be finite and positive")
    if flat_max_abs_delta_01 > texture_max_abs_delta_01:
        raise ValueError(
            "flat_max_abs_delta_01 cannot exceed texture_max_abs_delta_01"
        )
    if not math.isfinite(mask_power) or mask_power <= 0.0:
        raise ValueError("mask_power must be finite and positive")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("residual_constraint.eps must be finite and positive")
    return {
        "enabled": enabled,
        "mode": mode,
        "max_abs_delta_01": max_abs_delta_01,
        "flat_floor": flat_floor,
        "flat_max_abs_delta_01": flat_max_abs_delta_01,
        "texture_max_abs_delta_01": texture_max_abs_delta_01,
        "mask_power": mask_power,
        "eps": eps,
    }


def constrain_watermark_residual(pred_x0, cover_img, allowance, config=None):
    """Bound and content-gate a residual while preserving ``[-1, 1]`` I/O."""
    settings = get_residual_constraint_settings(config)
    if not settings["enabled"]:
        return pred_x0.clamp(-1.0, 1.0)
    if pred_x0.shape != cover_img.shape:
        raise ValueError(
            "pred_x0 and cover_img must have the same shape: "
            f"{tuple(pred_x0.shape)} vs {tuple(cover_img.shape)}"
        )
    if (
        allowance.ndim != 4
        or allowance.shape[0] != pred_x0.shape[0]
        or allowance.shape[1] not in (1, pred_x0.shape[1])
        or allowance.shape[2:] != pred_x0.shape[2:]
    ):
        raise ValueError(
            "allowance must be [B,1,H,W] or match pred_x0 channels, got "
            f"{tuple(allowance.shape)}"
        )

    raw_delta_01 = (pred_x0.float() - cover_img.float()) / 2.0
    allowance_float = allowance.detach().float().clamp(0.0, 1.0)
    if settings["mode"] == "spatial_budget":
        flat_max = settings["flat_max_abs_delta_01"]
        texture_max = settings["texture_max_abs_delta_01"]
        budget = flat_max + (texture_max - flat_max) * allowance_float.pow(
            settings["mask_power"]
        )
        constrained_delta_01 = budget * torch.tanh(
            raw_delta_01 / budget.clamp_min(settings["eps"])
        )
    else:
        max_abs = settings["max_abs_delta_01"]
        bounded_delta_01 = max_abs * torch.tanh(raw_delta_01 / max_abs)
        gate = settings["flat_floor"] + (
            1.0 - settings["flat_floor"]
        ) * allowance_float
        constrained_delta_01 = bounded_delta_01 * gate
    constrained = cover_img.float() + 2.0 * constrained_delta_01
    return constrained.clamp(-1.0, 1.0).to(dtype=pred_x0.dtype)
