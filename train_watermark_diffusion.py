"""
Train Watermark-Conditioned Image-to-Image Diffusion Model.

Core training logic:
  1. Two timestep ranges: t_diff (full) for noise prediction, t_wm (small) for watermark loss
  2. Clean watermark loss always reaches the U-Net; degraded loss is curriculum-controlled
  3. Image range discipline: [-1,1] for diffusion/decoder, [0,1] for degradations
  4. Unified none/PIMoG/projector/mixed degradation construction

KEY DEBUG POINTS if bit_acc ~ 0.5:
  1. Check wm_bits are actually fed into U-Net (watermark_mlp)
  2. Check watermark_mlp parameters have requires_grad=True
  3. Check loss_wm backprop reaches diffusion_model (no .detach() on pred_x0)
  4. Check decoder input range is [-1, 1]
  5. Check lambda_wm is not too small
  6. Check wm_t_max is not too large

Usage:
    D:/Anaconda_envs/envs/wadiff/python.exe train_watermark_diffusion.py --config configs/watermark_stage1.yaml
"""
import os
import sys
import argparse
import csv
import glob
import hashlib
import math
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.utils import save_image
from kornia.metrics import ssim as kornia_ssim

from guided_diffusion.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.nn import mean_flat

from dataset.watermark_image_dataset import WatermarkImageDataset
from models.watermark_unet import WatermarkConditionedUNet
from models.watermark_decoder import (
    build_watermark_decoder,
    load_watermark_decoder_state,
)
from models.watermark_residual import (
    build_edge_texture_guidance,
    constrain_watermark_residual,
    get_residual_constraint_settings,
)
from NOISE_LAYER import build_noise_layer, get_noise_layer_type

# ============================================================
# Helper: image-quality metrics
# ============================================================
def compute_psnr(pred, target, max_val=1.0):
    """Compute PSNR in [0, max_val] range."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    return (20 * math.log10(max_val) - 10 * math.log10(mse.item()))


def compute_ssim(pred, target, max_val=1.0, window_size=11):
    """Compute mean RGB SSIM for a batch in [0, max_val]."""
    if pred.shape != target.shape:
        raise ValueError(
            f"SSIM inputs must have the same shape, got {pred.shape} and "
            f"{target.shape}"
        )
    if pred.ndim != 4:
        raise ValueError("SSIM inputs must be 4-D BCHW tensors")
    ssim_map = kornia_ssim(
        pred.float(),
        target.float(),
        window_size=window_size,
        max_val=max_val,
        padding='same',
    )
    return ssim_map.mean().item()


def residual_tv_loss(delta):
    """Penalize short, high-frequency residual streaks."""
    loss_h = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
    loss_w = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
    return loss_h + loss_w


def residual_topk_loss(delta, fraction=0.01):
    """Penalize sparse, visually obvious residual spikes."""
    flat = delta.abs().flatten(1)
    k = max(1, int(flat.size(1) * fraction))
    return flat.topk(k, dim=1).values.mean()


def residual_channel_balance_loss(delta):
    """Discourage hiding most residual energy in one color channel."""
    channel_energy = delta.abs().mean(dim=(0, 2, 3))
    return channel_energy.std(unbiased=False)


def residual_region_loss(delta, penalty, eps=1e-6):
    """Penalize watermark residuals more strongly in smooth image regions."""
    pixel_energy = delta.float().abs().mean(dim=1, keepdim=True)
    return (pixel_energy * penalty).sum() / (penalty.sum() + eps)


def residual_region_enrichment_loss(
    delta,
    allowance,
    target_enrichment=0.03,
    eps=1e-6,
):
    """
    Require residual energy to be enriched in edge/texture regions.

    The residual-weighted allowance is compared with the cover's mean
    allowance. Subtracting this per-image baseline prevents naturally
    high-texture images from receiving an artificially better score.
    """
    target_enrichment = float(target_enrichment)
    if not 0.0 <= target_enrichment <= 1.0:
        raise ValueError(
            "region guidance target_enrichment must be in [0, 1], "
            f"got {target_enrichment}"
        )

    pixel_energy = delta.float().abs().mean(dim=1, keepdim=True)
    allowance = allowance.detach().float().clamp(0.0, 1.0)
    total_energy = pixel_energy.sum(dim=(1, 2, 3))
    texture_energy_ratio = (
        pixel_energy * allowance
    ).sum(dim=(1, 2, 3)) / total_energy.clamp_min(eps)
    allowance_baseline = allowance.mean(dim=(1, 2, 3))
    enrichment = texture_energy_ratio - allowance_baseline
    per_image_loss = F.relu(target_enrichment - enrichment)

    # Localization is undefined for a truly zero residual. The watermark BCE
    # supplies the embedding gradient in that corner case without amplifying
    # numerical noise through a nearly zero ratio denominator.
    per_image_loss = torch.where(
        total_energy > eps,
        per_image_loss,
        torch.zeros_like(per_image_loss),
    )
    return per_image_loss.mean()


def get_active_localization_targets(config=None, global_step=0):
    """Linearly tighten strict inside/outside residual-energy targets."""
    config = config or {}
    final_inside = float(config.get('target_inside_ratio', 0.85))
    start_inside = float(config.get('start_inside_ratio', final_inside))
    final_outside = float(config.get('max_outside_ratio', 1.0 - final_inside))
    start_outside = float(
        config.get('start_max_outside_ratio', 1.0 - start_inside)
    )
    warmup_steps = int(config.get('ratio_warmup_steps', 0))
    for name, value in (
        ('start_inside_ratio', start_inside),
        ('target_inside_ratio', final_inside),
        ('start_max_outside_ratio', start_outside),
        ('max_outside_ratio', final_outside),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"region_guidance.{name} must be in [0, 1]")
    if start_inside > final_inside:
        raise ValueError("start_inside_ratio cannot exceed target_inside_ratio")
    if start_outside < final_outside:
        raise ValueError(
            "start_max_outside_ratio cannot be smaller than max_outside_ratio"
        )
    if warmup_steps <= 0:
        progress = 1.0
    else:
        progress = min(1.0, max(0.0, (int(global_step) + 1) / warmup_steps))
    return {
        'inside': start_inside + progress * (final_inside - start_inside),
        'outside': start_outside + progress * (final_outside - start_outside),
        'progress': progress,
    }


def residual_energy_ratio_loss(
    delta,
    allowance,
    target_inside_ratio=0.85,
    max_outside_ratio=0.15,
    min_active_area=0.08,
    eps=1e-6,
):
    """Constrain the residual-energy fraction inside a soft structure mask."""
    if delta.ndim != 4 or allowance.ndim != 4:
        raise ValueError("delta and allowance must be 4-D image tensors")
    if allowance.shape[0] != delta.shape[0] or allowance.shape[2:] != delta.shape[2:]:
        raise ValueError("allowance batch/spatial shape must match delta")
    if allowance.shape[1] not in (1, delta.shape[1]):
        raise ValueError("allowance channels must be 1 or match delta channels")
    target_inside_ratio = float(target_inside_ratio)
    max_outside_ratio = float(max_outside_ratio)
    min_active_area = float(min_active_area)
    if not 0.0 <= target_inside_ratio <= 1.0:
        raise ValueError("target_inside_ratio must be in [0, 1]")
    if not 0.0 <= max_outside_ratio <= 1.0:
        raise ValueError("max_outside_ratio must be in [0, 1]")
    if not 0.0 <= min_active_area <= 1.0:
        raise ValueError("min_active_area must be in [0, 1]")

    pixel_energy = delta.float().abs().mean(dim=1, keepdim=True)
    mask = allowance.detach().float().clamp(0.0, 1.0)
    total = pixel_energy.sum(dim=(1, 2, 3))
    inside = (pixel_energy * mask).sum(dim=(1, 2, 3)) / total.clamp_min(eps)
    outside = (pixel_energy * (1.0 - mask)).sum(
        dim=(1, 2, 3)
    ) / total.clamp_min(eps)
    mask_area = mask.mean(dim=(1, 2, 3))
    loss = F.relu(target_inside_ratio - inside) + F.relu(
        outside - max_outside_ratio
    )
    valid = (total > eps) & (mask_area >= min_active_area)
    return torch.where(valid, loss, torch.zeros_like(loss)).mean()


def residual_outside_region_loss(delta, allowance, eps=1e-6):
    """
    Continuously penalize watermark residual energy outside soft content regions.

    ``allowance`` is the existing fixed Sobel/local-texture soft mask. It must
    have shape [B, 1, H, W] or [B, C, H, W]. The denominator explicitly
    accounts for RGB broadcasting so the loss is a weighted per-channel mean.
    """
    if delta.ndim != 4:
        raise ValueError(
            f"outside-region delta must have shape [B, C, H, W], got {delta.shape}"
        )
    if allowance.ndim != 4:
        raise ValueError(
            "outside-region allowance must have shape [B, 1, H, W] or "
            f"[B, C, H, W], got {allowance.shape}"
        )
    if allowance.shape[0] != delta.shape[0] or allowance.shape[2:] != delta.shape[2:]:
        raise ValueError(
            "outside-region allowance batch/spatial shape must match delta: "
            f"delta={tuple(delta.shape)}, allowance={tuple(allowance.shape)}"
        )
    if allowance.shape[1] not in (1, delta.shape[1]):
        raise ValueError(
            "outside-region allowance channels must be 1 or match delta: "
            f"delta={delta.shape[1]}, allowance={allowance.shape[1]}"
        )
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"outside-region eps must be finite and positive, got {eps}")

    delta_float = delta.float()
    allowance_float = allowance.to(
        device=delta.device,
        dtype=delta_float.dtype,
    ).clamp(0.0, 1.0)
    outside = 1.0 - allowance_float
    numerator = (outside * delta_float.abs()).sum()
    channel_factor = (
        1 if allowance_float.shape[1] == delta.shape[1] else delta.shape[1]
    )
    denominator = outside.sum() * channel_factor + eps
    return numerator / denominator


def get_region_outside_settings(config=None):
    """Strictly parse the optional, default-off outside-region loss settings."""
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise TypeError("region_guidance must be a mapping")

    outside_enabled = config.get('outside_enabled', False)
    if type(outside_enabled) is not bool:
        raise TypeError(
            "region_guidance.outside_enabled must be true or false, "
            f"got {outside_enabled!r}"
        )

    raw_weight = config.get('lambda_outside', 0.0)
    if isinstance(raw_weight, bool):
        raise TypeError("region_guidance.lambda_outside must be a non-negative number")
    try:
        lambda_outside = float(raw_weight)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "region_guidance.lambda_outside must be a non-negative number, "
            f"got {raw_weight!r}"
        ) from exc
    if not math.isfinite(lambda_outside) or lambda_outside < 0.0:
        raise ValueError(
            "region_guidance.lambda_outside must be finite and non-negative, "
            f"got {lambda_outside}"
        )
    if outside_enabled and lambda_outside <= 0.0:
        raise ValueError(
            "region_guidance.outside_enabled=true requires lambda_outside > 0"
        )
    return outside_enabled, lambda_outside


def combine_region_guidance_losses(
    enrichment_loss,
    outside_loss,
    lambda_enrichment,
    lambda_outside,
    outside_enabled,
):
    """Combine each region term exactly once while preserving the old baseline."""
    total = float(lambda_enrichment) * enrichment_loss
    if outside_enabled:
        total = total + float(lambda_outside) * outside_loss
    return total


def compute_region_guidance_loss(
    delta,
    allowance,
    penalty,
    config=None,
    global_step=0,
):
    """Dispatch the configured cover-dependent residual localization loss."""
    config = config or {}
    loss_mode = str(
        config.get('loss_mode', 'weighted_amplitude')
    ).strip().lower()
    if loss_mode == 'weighted_amplitude':
        return residual_region_loss(delta, penalty)
    if loss_mode == 'enrichment_margin':
        return residual_region_enrichment_loss(
            delta,
            allowance,
            target_enrichment=config.get('target_enrichment', 0.03),
            eps=float(config.get('eps', 1e-6)),
        )
    if loss_mode == 'energy_ratio':
        targets = get_active_localization_targets(config, global_step)
        return residual_energy_ratio_loss(
            delta,
            allowance,
            target_inside_ratio=targets['inside'],
            max_outside_ratio=targets['outside'],
            min_active_area=float(config.get('min_active_area', 0.08)),
            eps=float(config.get('eps', 1e-6)),
        )
    raise ValueError(
        "region_guidance.loss_mode must be 'weighted_amplitude', "
        f"'enrichment_margin', or 'energy_ratio', got {loss_mode!r}"
    )


def residual_region_energy_ratios(delta, allowance, eps=1e-6):
    """Return mean per-image residual energy fractions in flat/texture regions."""
    pixel_energy = delta.float().abs().mean(dim=1, keepdim=True)
    total = pixel_energy.sum(dim=(1, 2, 3)).clamp_min(eps)
    texture = (pixel_energy * allowance).sum(dim=(1, 2, 3)) / total
    flat = (pixel_energy * (1.0 - allowance)).sum(dim=(1, 2, 3)) / total
    return flat.mean(), texture.mean()


def residual_structure_metrics(
    delta,
    allowance=None,
    eps=1e-8,
    include_cross_image=True,
):
    """Return direction and spectrum diagnostics for signed RGB residuals."""
    residual = delta.float()
    dx_energy = (residual[:, :, :, 1:] - residual[:, :, :, :-1]).square().mean()
    dy_energy = (residual[:, :, 1:, :] - residual[:, :, :-1, :]).square().mean()
    directional_ratio = torch.maximum(dx_energy, dy_energy) / torch.minimum(
        dx_energy, dy_energy
    ).clamp_min(eps)

    power = torch.fft.fftshift(
        torch.fft.fft2(residual, norm='ortho'), dim=(-2, -1)
    ).abs().square().mean(dim=(0, 1))
    height, width = power.shape
    cy, cx = height // 2, width // 2
    power = power.clone()
    power[cy, cx] = 0.0
    flat_power = power.flatten()
    top_count = max(1, int(math.ceil(flat_power.numel() * 0.01)))
    fft_peak_ratio = flat_power.topk(top_count).values.sum() / flat_power.sum().clamp_min(
        eps
    )

    yy = torch.arange(height, device=power.device, dtype=power.dtype) - cy
    xx = torch.arange(width, device=power.device, dtype=power.dtype) - cx
    radius = torch.sqrt(yy[:, None].square() + xx[None, :].square())
    radius = radius / radius.max().clamp_min(1.0)
    midband = (radius >= 0.10) & (radius <= 0.45)
    fft_midband_ratio = power[midband].sum() / power.sum().clamp_min(eps)

    metrics = {
        'dx_energy': dx_energy,
        'dy_energy': dy_energy,
        'directional_ratio': directional_ratio,
        'fft_peak_ratio': fft_peak_ratio,
        'fft_midband_ratio': fft_midband_ratio,
    }
    if include_cross_image:
        flattened = residual.flatten(1)
        flattened = flattened - flattened.mean(dim=1, keepdim=True)
        flattened = flattened / flattened.norm(
            dim=1, keepdim=True
        ).clamp_min(eps)
        if flattened.shape[0] > 1:
            correlation = flattened @ flattened.transpose(0, 1)
            off_diagonal = ~torch.eye(
                flattened.shape[0], device=correlation.device, dtype=torch.bool
            )
            cross_image_correlation = correlation[off_diagonal].abs().mean()
        else:
            cross_image_correlation = residual.new_zeros(())
        metrics['cross_image_correlation'] = cross_image_correlation
    if allowance is not None:
        mask = allowance.detach().float().clamp(0.0, 1.0)
        flat_ratio, inside_ratio = residual_region_energy_ratios(
            residual, mask, eps=eps
        )
        metrics.update({
            'mask_area_ratio': mask.mean(),
            'inside_energy_ratio': inside_ratio,
            'outside_energy_ratio': flat_ratio,
        })
    return metrics


def get_residual_spectral_settings(config=None):
    config = config or {}
    if not isinstance(config, dict):
        raise TypeError("train.residual_spectral must be a mapping")
    enabled = config.get('enabled', False)
    if type(enabled) is not bool:
        raise TypeError("residual_spectral.enabled must be true or false")
    settings = {
        'enabled': enabled,
        'start_step': int(config.get('start_step', 2000)),
        'warmup_steps': int(config.get('warmup_steps', 1000)),
        'lambda_peak': float(config.get('lambda_peak', 0.0)),
        'lambda_anisotropy': float(config.get('lambda_anisotropy', 0.0)),
        'max_peak_ratio': float(config.get('max_peak_ratio', 0.20)),
        'max_directional_ratio': float(config.get('max_directional_ratio', 2.0)),
    }
    if settings['start_step'] < 0 or settings['warmup_steps'] < 0:
        raise ValueError("residual_spectral step values must be non-negative")
    if settings['lambda_peak'] < 0.0 or settings['lambda_anisotropy'] < 0.0:
        raise ValueError("residual_spectral loss weights must be non-negative")
    if not 0.0 <= settings['max_peak_ratio'] <= 1.0:
        raise ValueError("residual_spectral.max_peak_ratio must be in [0, 1]")
    if settings['max_directional_ratio'] < 1.0:
        raise ValueError("residual_spectral.max_directional_ratio must be >= 1")
    return settings


def residual_spectral_regularization_loss(delta, settings, global_step, eps=1e-8):
    zero = delta.new_zeros(())
    if not settings['enabled'] or int(global_step) < settings['start_step']:
        return zero, zero, zero, 0.0
    metrics = residual_structure_metrics(
        delta,
        allowance=None,
        eps=eps,
        include_cross_image=False,
    )
    peak_loss = F.relu(
        metrics['fft_peak_ratio'] - settings['max_peak_ratio']
    )
    anisotropy_loss = F.relu(
        torch.log(metrics['directional_ratio'].clamp_min(1.0))
        - math.log(settings['max_directional_ratio'])
    )
    elapsed = int(global_step) - settings['start_step'] + 1
    warmup_steps = settings['warmup_steps']
    scale = 1.0 if warmup_steps <= 0 else min(1.0, elapsed / warmup_steps)
    total = scale * (
        settings['lambda_peak'] * peak_loss
        + settings['lambda_anisotropy'] * anisotropy_loss
    )
    return total, peak_loss, anisotropy_loss, scale


def get_active_region_weight(base_weight, global_step, config=None):
    """Linearly warm up the texture-guided loss weight."""
    config = config or {}
    warmup_steps = int(config.get('warmup_steps', 0))
    if warmup_steps <= 0:
        return float(base_weight)
    warmup_scale = min(1.0, max(0.0, (global_step + 1) / warmup_steps))
    return float(base_weight) * warmup_scale


def get_loss_weights(cfg, global_step):
    train_cfg = cfg.get('train', {})
    stage = str(train_cfg.get('stage', '')).lower()
    stages_cfg = train_cfg.get('stages', {})
    if stages_cfg:
        if not stage:
            raise ValueError(
                "train.stage is required when train.stages is configured; "
                f"choose one of {sorted(stages_cfg)}"
            )
        if stage not in stages_cfg:
            raise ValueError(
                f"Unknown train.stage={stage!r}; choose one of "
                f"{sorted(stages_cfg)}"
            )
    if stage and stage in stages_cfg:
        stage_cfg = stages_cfg[stage] or {}
        schedule = stage_cfg.get('loss_schedule')
        if schedule:
            for item in schedule:
                until_step = int(item.get('until_step', -1))
                if until_step < 0 or global_step < until_step:
                    return _loss_weight_dict(item, stage_cfg)
        return _loss_weight_dict(stage_cfg, train_cfg)

    if train_cfg.get('use_loss_schedule', False):
        for item in train_cfg.get('loss_schedule', []):
            until_step = int(item.get('until_step', -1))
            if until_step < 0 or global_step < until_step:
                return {
                    'lambda_diff': float(item.get('lambda_diff', train_cfg['lambda_diff'])),
                    'lambda_img': float(item.get('lambda_img', train_cfg['lambda_img'])),
                    'lambda_wm': float(item.get('lambda_wm', train_cfg['lambda_wm'])),
                    'lambda_delta': float(item.get('lambda_delta', train_cfg.get('lambda_delta', 0.0))),
                    'lambda_tv': float(item.get('lambda_tv', train_cfg.get('lambda_tv', 0.0))),
                    'lambda_topk': float(item.get('lambda_topk', train_cfg.get('lambda_topk', 0.0))),
                    'lambda_channel': float(item.get('lambda_channel', train_cfg.get('lambda_channel', 0.0))),
                    'lambda_region': float(item.get('lambda_region', train_cfg.get('lambda_region', 0.0))),
                }
    return {
        'lambda_diff': float(train_cfg['lambda_diff']),
        'lambda_img': float(train_cfg['lambda_img']),
        'lambda_wm': float(train_cfg['lambda_wm']),
        'lambda_delta': float(train_cfg.get('lambda_delta', 0.0)),
        'lambda_tv': float(train_cfg.get('lambda_tv', 0.0)),
        'lambda_topk': float(train_cfg.get('lambda_topk', 0.0)),
        'lambda_channel': float(train_cfg.get('lambda_channel', 0.0)),
        'lambda_region': float(train_cfg.get('lambda_region', 0.0)),
    }


def get_stage_training_setting(train_cfg, key, default=None):
    """Return a stage-specific setting with a top-level training fallback."""
    stage = str(train_cfg.get('stage', '')).strip().lower()
    stages_cfg = train_cfg.get('stages', {})
    if stages_cfg and stage in stages_cfg:
        stage_cfg = stages_cfg[stage] or {}
        if key in stage_cfg:
            return stage_cfg[key]
    return train_cfg.get(key, default)


def _loss_weight_dict(source, fallback):
    return {
        'lambda_diff': float(source.get('lambda_diff', fallback.get('lambda_diff', 0.0))),
        'lambda_img': float(source.get('lambda_img', fallback.get('lambda_img', 0.0))),
        'lambda_wm': float(source.get('lambda_wm', fallback.get('lambda_wm', 0.0))),
        'lambda_delta': float(source.get('lambda_delta', fallback.get('lambda_delta', 0.0))),
        'lambda_tv': float(source.get('lambda_tv', fallback.get('lambda_tv', 0.0))),
        'lambda_topk': float(source.get('lambda_topk', fallback.get('lambda_topk', 0.0))),
        'lambda_channel': float(source.get('lambda_channel', fallback.get('lambda_channel', 0.0))),
        'lambda_region': float(source.get('lambda_region', fallback.get('lambda_region', 0.0))),
    }


def get_noise_curriculum_state(cfg, global_step, lambda_wm, use_noise_layer):
    """Return the active Stage 2 degradation curriculum settings."""
    train_cfg = cfg.get('train', {})
    if not use_noise_layer:
        return {
            'phase': 0,
            'apply_prob': 0.0,
            'strength': 0.0,
            'lambda_wm_clean': float(lambda_wm),
            'lambda_wm_degraded': 0.0,
            'detach_degraded_from_model': False,
            'candidates': None,
            'probs': None,
            'lr_scale': 1.0,
        }

    schedule = train_cfg.get('noise_curriculum', [])
    selected = None
    selected_phase = 1
    for phase_index, item in enumerate(schedule, start=1):
        until_step = int(item.get('until_step', -1))
        if until_step < 0 or global_step < until_step:
            selected = item
            selected_phase = phase_index
            break
    if selected is None and schedule:
        selected_phase = len(schedule) + 1
    source = selected or {}

    state = {
        'phase': selected_phase,
        'apply_prob': float(source.get('apply_prob', 1.0)),
        'strength': float(source.get('strength', 1.0)),
        'lambda_wm_clean': float(source.get(
            'lambda_wm_clean', train_cfg.get('lambda_wm_clean', 0.0)
        )),
        'lambda_wm_degraded': float(source.get(
            'lambda_wm_degraded', lambda_wm
        )),
        'detach_degraded_from_model': bool(source.get(
            'detach_degraded_from_model', False
        )),
        'candidates': source.get('candidates'),
        'probs': source.get('probs'),
        'lr_scale': float(source.get('lr_scale', 1.0)),
    }

    if not 0.0 <= state['apply_prob'] <= 1.0:
        raise ValueError("noise curriculum apply_prob must be in [0, 1]")
    if not 0.0 <= state['strength'] <= 1.0:
        raise ValueError("noise curriculum strength must be in [0, 1]")
    if state['lambda_wm_clean'] < 0.0 or state['lambda_wm_degraded'] < 0.0:
        raise ValueError("noise curriculum watermark weights must be non-negative")
    if state['lr_scale'] <= 0.0:
        raise ValueError("noise curriculum lr_scale must be positive")
    if state['candidates'] is not None:
        state['candidates'] = [str(name).lower() for name in state['candidates']]
        if not state['candidates']:
            raise ValueError("noise curriculum candidates must not be empty")
        if state['probs'] is not None:
            state['probs'] = [float(prob) for prob in state['probs']]
            if len(state['probs']) != len(state['candidates']):
                raise ValueError("noise curriculum probs must match candidates")
            if any(prob < 0.0 for prob in state['probs']):
                raise ValueError("noise curriculum probs must be non-negative")
            if sum(state['probs']) <= 0.0:
                raise ValueError("noise curriculum probs must contain a positive value")
    return state


def get_noise_curriculum_phase(cfg, global_step, use_noise_layer=True):
    """Return the one-based active curriculum phase (zero when noise is disabled)."""
    if not use_noise_layer:
        return 0
    schedule = cfg.get('train', {}).get('noise_curriculum', [])
    for phase_index, item in enumerate(schedule, start=1):
        until_step = int(item.get('until_step', -1))
        if until_step < 0 or global_step < until_step:
            return phase_index
    return len(schedule) + 1 if schedule else 1


def get_degradation_stage(cfg, global_step, use_noise_layer=True):
    """Group curriculum phases by the set of degradation types introduced."""
    if not use_noise_layer:
        return 0

    schedule = cfg.get('train', {}).get('noise_curriculum', [])
    if not schedule:
        return 1

    curriculum_phase = get_noise_curriculum_phase(
        cfg, global_step, use_noise_layer=True
    )
    active_index = min(max(curriculum_phase - 1, 0), len(schedule) - 1)
    active_candidates = schedule[active_index].get('candidates')
    if active_candidates is None:
        return 1
    active_signature = tuple(sorted(
        str(name).lower() for name in active_candidates
    ))

    unique_signatures = []
    for item in schedule:
        candidates = item.get('candidates')
        if candidates is None:
            signature = ('__default__',)
        else:
            signature = tuple(sorted(
                str(name).lower() for name in candidates
            ))
        if signature not in unique_signatures:
            unique_signatures.append(signature)
        if signature == active_signature:
            return unique_signatures.index(signature) + 1

    raise RuntimeError("Active degradation candidate set is absent from curriculum")


def apply_degradation_with_strength(
    source_01,
    noise_layer,
    noise_type,
    strength,
    candidates=None,
    probs=None,
):
    """Apply one full degradation, then blend it with the source by ``strength``."""
    noise_type = str(noise_type).lower()
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("degradation strength must be in [0, 1]")

    source_01 = source_01.float()
    if noise_type == 'none':
        return source_01.clamp(0.0, 1.0), 'clean'

    if noise_type == 'mixed':
        degraded_full_01 = noise_layer(
            source_01,
            candidates=candidates,
            probs=probs,
        ).float()
        active_noise_type = noise_layer.get_last_name()
    else:
        degraded_full_01 = noise_layer(source_01).float()
        active_noise_type = noise_type

    degraded_01 = source_01 + strength * (degraded_full_01 - source_01)
    return degraded_01.clamp(0.0, 1.0), active_noise_type


def get_multi_attack_settings(config=None):
    """Validate the optional same-image multi-attack objective."""
    config = config or {}
    if not isinstance(config, dict):
        raise TypeError("train.multi_attack must be a mapping")
    enabled = config.get("enabled", False)
    if type(enabled) is not bool:
        raise TypeError("multi_attack.enabled must be true or false")
    attacks_per_batch = int(config.get("attacks_per_batch", 2))
    lambda_mean = float(config.get("lambda_mean", 0.5))
    lambda_worst = float(config.get("lambda_worst", 0.5))
    if attacks_per_batch < 1:
        raise ValueError("multi_attack.attacks_per_batch must be positive")
    if lambda_mean < 0.0 or lambda_worst < 0.0:
        raise ValueError("multi_attack loss weights must be non-negative")
    if lambda_mean + lambda_worst <= 0.0:
        raise ValueError("at least one multi_attack loss weight must be positive")
    return {
        "enabled": enabled,
        "attacks_per_batch": attacks_per_batch,
        "lambda_mean": lambda_mean,
        "lambda_worst": lambda_worst,
    }


def select_multi_attack_candidates(candidates, probs, count, device):
    """Sample distinct curriculum attacks without changing their base weights."""
    candidates = list(candidates or [])
    if not candidates:
        raise ValueError("multi-attack training requires curriculum candidates")
    count = min(int(count), len(candidates))
    if probs is None:
        weights = torch.ones(len(candidates), device=device)
    else:
        weights = torch.tensor(probs, device=device, dtype=torch.float32)
    indices = torch.multinomial(weights, count, replacement=False).tolist()
    return [str(candidates[index]).lower() for index in indices]


def normalize_psnr_score(psnr, low=30.0, high=45.0):
    """Map PSNR linearly to [0, 1] for checkpoint scoring."""
    if high <= low:
        raise ValueError("PSNR score upper bound must be greater than lower bound")
    return max(0.0, min(1.0, (float(psnr) - low) / (high - low)))


def compute_residual_quality_score(topk_delta, tv_delta, channel_delta_std):
    """Return normalized residual-fidelity score and its components."""
    topk_score = 1.0 - max(0.0, min(1.0, float(topk_delta) / 0.10))
    tv_score = 1.0 - max(0.0, min(1.0, float(tv_delta) / 0.05))
    channel_score = 1.0 - max(
        0.0, min(1.0, float(channel_delta_std) / 0.02)
    )
    residual_score = (topk_score + tv_score + channel_score) / 3.0
    return residual_score, {
        'topk_score': topk_score,
        'tv_score': tv_score,
        'channel_score': channel_score,
    }


def compute_balanced_checkpoint_score(
    degraded_macro_acc,
    degraded_worst_acc,
    clean_acc,
    degraded_macro_bce,
    psnr,
    topk_delta,
    tv_delta,
    channel_delta_std,
):
    """Compute the gate-free normalized checkpoint score defined for Stage 2 V3."""
    raw_values = (
        degraded_macro_acc,
        degraded_worst_acc,
        clean_acc,
        degraded_macro_bce,
        psnr,
        topk_delta,
        tv_delta,
        channel_delta_std,
    )
    if not all(math.isfinite(float(value)) for value in raw_values):
        return float('-inf'), {}

    loss_score = math.exp(-max(0.0, float(degraded_macro_bce)))
    psnr_score = normalize_psnr_score(psnr)
    residual_score, residual_components = compute_residual_quality_score(
        topk_delta, tv_delta, channel_delta_std
    )
    balanced_score = (
        0.40 * float(degraded_macro_acc)
        + 0.25 * float(degraded_worst_acc)
        + 0.10 * float(clean_acc)
        + 0.05 * loss_score
        + 0.10 * psnr_score
        + 0.10 * residual_score
    )
    components = {
        'loss_score': loss_score,
        'psnr_score': psnr_score,
        'residual_score': residual_score,
        **residual_components,
    }
    return balanced_score, components


def tensor_is_finite(value):
    return bool(torch.isfinite(value.detach()).all().item())


def gradients_are_finite(parameters):
    for param in parameters:
        if (
            param.grad is not None
            and not bool(torch.isfinite(param.grad.detach()).all().item())
        ):
            return False
    return True


def generate_train_watermark(batch_size, length, device):
    return torch.randint(0, 2, (batch_size, length), device=device).float()


def generate_val_watermark(batch_size, length, seed, device, offset=0):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed) + int(offset))
    bits = torch.randint(0, 2, (batch_size, length), generator=generator).float()
    return bits.to(device)


def grad_norm(module):
    total = 0.0
    has_grad = False
    for param in module.parameters():
        if param.grad is None:
            continue
        param_norm = param.grad.detach().float().norm(2).item()
        total += param_norm * param_norm
        has_grad = True
    return math.sqrt(total) if has_grad else float('nan')


def parameter_grad_norm(parameters):
    """Return an L2 gradient norm, or ``None`` when no gradient exists."""
    total = 0.0
    has_grad = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().norm(2).item()
        total += value * value
        has_grad = True
    return math.sqrt(total) if has_grad else None


def count_parameters(parameters):
    """Count scalar parameters without changing an iterator's training state."""
    return sum(parameter.numel() for parameter in parameters)


def configure_encoder_training(encoder, freeze_encoder):
    """Apply the ablation's real freeze while preserving the default path."""
    if not freeze_encoder:
        return
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()


def get_encoder_train_mode(train_config):
    """Resolve the new mode while retaining the legacy freeze flag."""
    mode = train_config.get("encoder_train_mode")
    if mode is None:
        mode = "frozen" if train_config.get("freeze_encoder", False) else "full"
    mode = str(mode).strip().lower()
    if mode not in {"full", "frozen", "partial"}:
        raise ValueError(
            "train.encoder_train_mode must be full, frozen, or partial"
        )
    return mode


def configure_encoder_train_mode(encoder, mode, partial_output_blocks=2,
                                 freeze_watermark_map_mlp=True):
    """Configure full, frozen, or narrowly constrained encoder adaptation."""
    mode = str(mode).lower()
    if mode == "full":
        for parameter in encoder.parameters():
            parameter.requires_grad = True
        encoder.train()
        return
    if mode == "frozen":
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        encoder.eval()
        return
    if mode != "partial":
        raise ValueError(f"Unsupported encoder train mode: {mode}")

    partial_output_blocks = int(partial_output_blocks)
    if partial_output_blocks < 0:
        raise ValueError("partial_output_blocks must be non-negative")
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    for parameter in encoder.watermark_mlp.parameters():
        parameter.requires_grad = True
    if not freeze_watermark_map_mlp:
        for parameter in encoder.watermark_map_mlp.parameters():
            parameter.requires_grad = True
    if partial_output_blocks > 0:
        for block in encoder.inner_unet.output_blocks[-partial_output_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    for parameter in encoder.inner_unet.out.parameters():
        parameter.requires_grad = True
    set_encoder_training_mode(encoder, mode, partial_output_blocks,
                              freeze_watermark_map_mlp)


def set_encoder_training_mode(encoder, mode, partial_output_blocks=2,
                              freeze_watermark_map_mlp=True):
    """Restore train/eval modes after an outer ``model.train()`` call."""
    mode = str(mode).lower()
    if mode == "full":
        encoder.train()
        return
    encoder.eval()
    if mode == "frozen":
        return
    encoder.watermark_mlp.train()
    if not freeze_watermark_map_mlp:
        encoder.watermark_map_mlp.train()
    if partial_output_blocks > 0:
        for block in encoder.inner_unet.output_blocks[-partial_output_blocks:]:
            block.train()
    encoder.inner_unet.out.train()


def build_training_optimizer(encoder, decoder, lr, encoder_lr=None,
                             decoder_lr=None):
    """Build AdamW while preserving the old single-LR path by default."""
    encoder_parameters = [
        parameter for parameter in encoder.parameters() if parameter.requires_grad
    ]
    decoder_parameters = [
        parameter for parameter in decoder.parameters() if parameter.requires_grad
    ]
    if not encoder_parameters and not decoder_parameters:
        raise ValueError("No trainable parameters remain for the optimizer.")
    if encoder_lr is None and decoder_lr is None:
        optimizer = AdamW(encoder_parameters + decoder_parameters, lr=lr)
        optimizer.param_groups[0]["base_lr"] = float(lr)
        optimizer.param_groups[0]["name"] = "joint"
        return optimizer

    encoder_lr = float(lr if encoder_lr is None else encoder_lr)
    decoder_lr = float(lr if decoder_lr is None else decoder_lr)
    groups = []
    if encoder_parameters:
        groups.append({
            "params": encoder_parameters,
            "lr": encoder_lr,
            "base_lr": encoder_lr,
            "name": "encoder",
        })
    if decoder_parameters:
        groups.append({
            "params": decoder_parameters,
            "lr": decoder_lr,
            "base_lr": decoder_lr,
            "name": "decoder",
        })
    return AdamW(groups)


def optimizer_parameter_ids(optimizer):
    """Return optimizer ownership by object identity, not by parameter count."""
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group['params']
    }


def validate_optimizer_parameter_ownership(optimizer, encoder, decoder,
                                           freeze_encoder):
    """Fail fast if the optimizer violates the requested module boundary."""
    optimizer_ids = optimizer_parameter_ids(optimizer)
    encoder_ids = {id(parameter) for parameter in encoder.parameters()}
    decoder_trainable_ids = {
        id(parameter) for parameter in decoder.parameters() if parameter.requires_grad
    }
    encoder_overlap = optimizer_ids & encoder_ids
    missing_decoder = decoder_trainable_ids - optimizer_ids
    if freeze_encoder and encoder_overlap:
        raise RuntimeError(
            "Frozen encoder parameters are present in the optimizer."
        )
    if missing_decoder:
        raise RuntimeError(
            "Trainable decoder parameters are missing from the optimizer."
        )
    return optimizer_ids


def parameter_digest(module):
    """Compute an exact, temporary SHA-256 digest of a module's parameters."""
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode('utf-8'))
            digest.update(str(tuple(value.shape)).encode('ascii'))
            digest.update(str(value.dtype).encode('ascii'))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def clone_parameter_values(module):
    """Keep a small module snapshot for a startup-only parameter delta check."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
    }


def parameter_max_abs_delta(module, snapshot):
    """Return the largest element-wise parameter change from ``snapshot``."""
    maximum = 0.0
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name not in snapshot:
                raise KeyError(f"Missing parameter snapshot entry: {name}")
            delta = (
                parameter.detach().cpu().float()
                - snapshot[name].float()
            ).abs().max().item()
            maximum = max(maximum, delta)
    return maximum

# ============================================================
# Helper: predict x0 from noise prediction
# ============================================================
def predict_start_from_noise(diffusion, x_t, t, noise_pred):
    """Wrapper around GaussianDiffusion._predict_xstart_from_eps."""
    return diffusion._predict_xstart_from_eps(x_t, t, noise_pred)


def set_random_seed(seed, deterministic=True):
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """Give every DataLoader worker a reproducible Python/NumPy RNG state."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_random_state(train_generator):
    numpy_state = np.random.get_state()
    state = {
        'python': random.getstate(),
        'numpy': {
            'bit_generator': numpy_state[0],
            'state': torch.from_numpy(numpy_state[1].copy()),
            'pos': numpy_state[2],
            'has_gauss': numpy_state[3],
            'cached_gaussian': numpy_state[4],
        },
        'torch': torch.get_rng_state(),
        'train_generator': train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state, train_generator):
    if not state:
        return
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((
        numpy_state['bit_generator'],
        numpy_state['state'].cpu().numpy(),
        numpy_state['pos'],
        numpy_state['has_gauss'],
        numpy_state['cached_gaussian'],
    ))
    # Checkpoints are loaded with map_location=device, which can move these
    # CPU generator states onto CUDA. PyTorch RNG restore APIs require CPU
    # ByteTensors even when restoring CUDA generator states.
    torch.set_rng_state(state['torch'].detach().cpu().to(torch.uint8))
    train_generator.set_state(
        state['train_generator'].detach().cpu().to(torch.uint8)
    )
    if torch.cuda.is_available() and 'cuda' in state:
        cuda_states = [
            rng_state.detach().cpu().to(torch.uint8)
            for rng_state in state['cuda']
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def capture_eval_rng_state():
    """Capture process RNG state so deterministic validation cannot perturb training."""
    numpy_state = np.random.get_state()
    state = {
        'python': random.getstate(),
        'numpy': (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_eval_rng_state(state):
    """Restore a state produced by :func:`capture_eval_rng_state`."""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].detach().cpu().to(torch.uint8))
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all([
            rng_state.detach().cpu().to(torch.uint8)
            for rng_state in state['cuda']
        ])


# ============================================================
# Configuration loading
# ============================================================
def load_config(config_path):
    """Load a required YAML config file."""
    if not config_path:
        raise ValueError("A YAML config path must be provided via --config.")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load the training config.") from exc

    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    return config


def validate_initialization_policy(config, resume_path=None, init_from_path=None):
    """Enforce optional config-level guards for controlled ablation runs."""
    encoder_mode = get_encoder_train_mode(config.get('train', {}))
    controlled_encoder_stage = encoder_mode in {"frozen", "partial"}
    allow_encoder_mode_resume = bool(
        config.get('train', {}).get('allow_encoder_mode_resume', False)
    )
    if resume_path and init_from_path:
        raise ValueError("Use either --resume or --init_from, not both.")
    if controlled_encoder_stage and resume_path and not allow_encoder_mode_resume:
        raise ValueError(
            f"encoder_train_mode={encoder_mode} forbids --resume. Start from "
            "the Stage 1 checkpoint with --init_from."
        )
    if controlled_encoder_stage and not (resume_path or init_from_path):
        raise ValueError(
            f"encoder_train_mode={encoder_mode} requires --init_from with the "
            "Stage 1 checkpoint"
            + (
                " or --resume with this experiment's own checkpoint."
                if allow_encoder_mode_resume else "."
            )
        )

    policy = config.get('initialization', {})
    if not policy:
        return
    if not isinstance(policy, dict):
        raise TypeError("initialization must be a mapping")

    if bool(policy.get('forbid_resume', False)) and resume_path:
        raise ValueError(
            "This config forbids --resume. Start the ablation from its required "
            "Stage 1 checkpoint with --init_from."
        )
    if (
        bool(policy.get('require_init_from_on_new_run', False))
        and not (resume_path or init_from_path)
    ):
        raise ValueError(
            "This config requires --init_from for a new run, or --resume with "
            "this experiment's own checkpoint."
        )
    if bool(policy.get('require_init_from', False)) and not init_from_path:
        raise ValueError(
            "This config requires --init_from with the designated Stage 1 checkpoint."
        )

    checkpoint_stages = {
        str(stage).strip().lower()
        for stage in policy.get('require_checkpoint_for_stages', [])
    }
    active_stage = str(config.get('train', {}).get('stage', '')).strip().lower()
    if active_stage in checkpoint_stages and not (resume_path or init_from_path):
        raise ValueError(
            f"train.stage={active_stage!r} must start with --init_from from the "
            "previous manual phase, or --resume its own interrupted checkpoint."
        )

    expected_path = policy.get('expected_init_from')
    if expected_path and init_from_path:
        expected_normalized = os.path.normcase(os.path.abspath(str(expected_path)))
        actual_normalized = os.path.normcase(os.path.abspath(str(init_from_path)))
        if actual_normalized != expected_normalized:
            raise ValueError(
                "This config requires --init_from "
                f"{expected_path!r}, got {init_from_path!r}."
            )


def _get_checkpoint_model_state(ckpt):
    if 'diffusion_model' in ckpt:
        return ckpt['diffusion_model']
    if 'model' in ckpt:
        return ckpt['model']
    return ckpt


def load_decoder_checkpoint(decoder, decoder_state, log_prefix,
                            require_full_match=False):
    try:
        decoder.load_state_dict(decoder_state, strict=True)
        print(f"{log_prefix} Loaded watermark decoder weights.")
        return
    except RuntimeError as exc:
        print(f"{log_prefix} Decoder strict=True load failed: {exc}")
        if require_full_match:
            raise RuntimeError(
                f"{log_prefix} Full decoder checkpoint loading is required."
            ) from exc

    missing, unexpected, mismatched = load_watermark_decoder_state(
        decoder, decoder_state
    )
    print(f"{log_prefix} Decoder weights are partially loaded with strict=False.")
    print(f"{log_prefix} Missing keys: {missing}")
    print(f"{log_prefix} Unexpected keys: {unexpected}")
    print(f"{log_prefix} Mismatched keys: {mismatched}")


def load_model_state_for_init(model, checkpoint_state, require_full_match=False):
    current_state = model.state_dict()
    load_state = dict(current_state)
    missing_keys = []
    unexpected_keys = []
    shape_mismatch_keys = []
    copied_first_conv = False

    for key, value in checkpoint_state.items():
        if key not in current_state:
            unexpected_keys.append(key)
            continue

        current_value = current_state[key]
        if current_value.shape == value.shape:
            load_state[key] = value
            continue

        shape_mismatch_keys.append(
            f"{key}: checkpoint={tuple(value.shape)}, current={tuple(current_value.shape)}"
        )
        if (
            key.endswith('input_blocks.0.0.weight')
            and value.ndim == 4
            and current_value.ndim == 4
            and value.shape[0] == current_value.shape[0]
            and value.shape[2:] == current_value.shape[2:]
            and value.shape[1] < current_value.shape[1]
        ):
            new_weight = current_value.clone()
            new_weight[:, :value.shape[1], :, :] = value
            new_weight[:, value.shape[1]:, :, :] = 0.0
            load_state[key] = new_weight
            copied_first_conv = True
            print(
                "[Init] Detected input channel mismatch in first conv: "
                f"checkpoint={value.shape[1]}, current={current_value.shape[1]}."
            )
            print("[Init] Copied old x_t and cover_img channels.")
            print("[Init] Initialized new wm_map channels.")

    for key in current_state:
        if key not in checkpoint_state:
            missing_keys.append(key)

    print(f"[Init] Missing model keys: {missing_keys}")
    print(f"[Init] Unexpected model keys: {unexpected_keys}")
    print(f"[Init] Shape mismatch model keys: {shape_mismatch_keys}")
    if require_full_match and (
        missing_keys or unexpected_keys or shape_mismatch_keys
    ):
        raise RuntimeError(
            "[Init] Full encoder checkpoint loading is required for the "
            "freeze-encoder ablation."
        )

    model.load_state_dict(load_state, strict=True)
    print("[Init] Loaded compatible diffusion model weights.")
    if not copied_first_conv and shape_mismatch_keys:
        print("[Init] Shape-mismatched tensors kept at current initialization.")


def resume_training(checkpoint_path, model, decoder, optimizer, scaler,
                    train_generator, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    print(f"[Resume] Resume training from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(_get_checkpoint_model_state(ckpt), strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "[Resume Error] Checkpoint model structure does not match current model.\n"
            "Use --init_from if you want to initialize a new stage with changed architecture."
        ) from exc
    print("[Resume] Loaded diffusion model.")

    if 'decoder' in ckpt:
        load_decoder_checkpoint(
            decoder,
            ckpt['decoder'],
            "[Resume]",
            require_full_match=True,
        )
    else:
        print("[Resume] No decoder weights found in checkpoint.")

    if 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
        print("[Resume] Loaded optimizer.")
    else:
        print("[Resume] No optimizer state found in checkpoint.")

    if scaler is not None and ckpt.get('scaler') is not None:
        scaler.load_state_dict(ckpt['scaler'])
        print("[Resume] Loaded AMP scaler.")

    start_epoch = ckpt.get('epoch', 0) + 1
    global_step = ckpt.get('global_step', 0)
    restore_random_state(ckpt.get('random_state'), train_generator)
    print(f"[Resume] start_epoch={start_epoch}, global_step={global_step}")
    if 'best_metric_value' in ckpt or 'best_bit_acc' in ckpt:
        best_name = ckpt.get('best_metric_name', 'bit_acc_clean')
        best_value = ckpt.get('best_metric_value', ckpt.get('best_bit_acc'))
        print(f"[Resume] Previous best {best_name}={best_value:.4f}")
    return start_epoch, global_step


def init_from_checkpoint(checkpoint_path, model, decoder, reset_decoder, device,
                         require_full_match=False):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Init checkpoint not found: {checkpoint_path}")

    print(f"[Init] Initialize new training stage from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    load_model_state_for_init(
        model,
        _get_checkpoint_model_state(ckpt),
        require_full_match=require_full_match,
    )
    print("[Init] Loaded diffusion model weights.")

    if reset_decoder:
        if require_full_match:
            raise RuntimeError(
                "[Init] reset_decoder=true is incompatible with the "
                "freeze-encoder ablation."
            )
        print("[Init] reset_decoder=true, skip loading old decoder weights.")
    elif 'decoder' in ckpt:
        load_decoder_checkpoint(
            decoder,
            ckpt['decoder'],
            "[Init]",
            require_full_match=require_full_match,
        )
    else:
        if require_full_match:
            raise RuntimeError(
                "[Init] The Stage 1 checkpoint has no decoder weights."
            )
        print("[Init] No decoder weights found in checkpoint.")

    print("[Init] Skip optimizer state.")
    print("[Init] Reset start_epoch=1, global_step=0.")
    return 1, 0

# ============================================================
# embed_watermark: Full DDPM reverse sampling for image-to-image
# ============================================================
@torch.no_grad()
def embed_watermark(diffusion, model, cover_img, wm_bits, t_start=300,
                    region_guidance_config=None,
                    residual_constraint_config=None):
    """
    Image-to-image watermark embedding via partial DDPM reverse sampling.

    Strategy:
      1. Add noise to cover_img up to t_start
      2. Reverse-denoise with cover_img + wm_bits as conditions
      3. Return watermarked image in [-1, 1]

    Args:
        diffusion: GaussianDiffusion instance (for schedules)
        model: WatermarkConditionedUNet
        cover_img: [B, 3, H, W] in [-1, 1]
        wm_bits:  [B, wm_len] 0/1 float
        t_start:  timestep to start reverse from (controls edit strength)

    Returns:
        watermarked: [B, 3, H, W] in [-1, 1]
    """
    device = cover_img.device
    B = cover_img.size(0)
    residual_settings = get_residual_constraint_settings(
        residual_constraint_config
    )
    needs_content_guidance = (
        residual_settings['enabled']
        or bool(getattr(model, 'use_content_gated_wm_map', False))
    )
    if needs_content_guidance:
        allowance, _ = build_edge_texture_guidance(
            (cover_img + 1.0) / 2.0,
            region_guidance_config,
        )
    else:
        allowance = None

    def constrain_xstart(x_start):
        return constrain_watermark_residual(
            x_start,
            cover_img,
            allowance,
            residual_constraint_config,
        )

    # 1. Forward diffuse to t_start
    t = torch.full((B,), t_start - 1, device=device, dtype=torch.long)
    noise = torch.randn_like(cover_img)
    x_t = diffusion.q_sample(cover_img, t, noise=noise)

    # 2. Reverse denoise step by step
    for step in reversed(range(t_start)):
        t_batch = torch.full((B,), step, device=device, dtype=torch.long)

        # Scale timesteps for model
        t_scaled = t_batch.float() * (1000.0 / diffusion.num_timesteps)

        pred_noise = model(
            x_t=x_t,
            t=t_scaled,
            cover_img=cover_img,
            wm_bits=wm_bits,
            content_mask=allowance,
        )

        # Use DDPM sampling step
        out = diffusion.p_mean_variance(
            model=lambda *a, **kw: pred_noise,
            x=x_t,
            t=t_batch,
            clip_denoised=True,
            denoised_fn=constrain_xstart if residual_settings['enabled'] else None,
            model_kwargs={},
        )
        mean = out['mean']
        log_variance = out['log_variance']

        # Sample x_{t-1}
        noise_term = torch.randn_like(x_t) if step > 0 else torch.zeros_like(x_t)
        x_t = mean + torch.exp(0.5 * log_variance) * noise_term

    watermarked = constrain_xstart(x_t)
    return watermarked

# ============================================================
# Main training function
# ============================================================
def train(config):
    cfg = config
    validate_initialization_policy(
        cfg,
        resume_path=cfg.get('_resume_path'),
        init_from_path=cfg.get('_init_from_path'),
    )

    seed = cfg['train'].get('seed', 42)
    deterministic = cfg['train'].get('deterministic', True)
    set_random_seed(seed, deterministic=deterministic)
    print(f"[Train] Random seed: {seed}, deterministic={deterministic}")

    # --- Device ---
    device_str = cfg['train'].get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Using device: {device}")

    # --- Create output directories ---
    checkpoint_dir = cfg['output']['checkpoint_dir']
    sample_dir = cfg['output']['sample_dir']
    log_dir = cfg['output']['log_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --- Dataset ---
    image_size = cfg['data']['image_size']
    watermark_length = cfg['data']['watermark_length']
    watermark_seed = cfg['data'].get('watermark_seed', seed)
    batch_size = cfg['train']['batch_size']
    num_workers = cfg['train']['num_workers']

    train_dataset = WatermarkImageDataset(
        data_dir=cfg['data']['train_dir'],
        image_size=image_size,
        watermark_length=watermark_length,
        watermark_seed=watermark_seed,
        watermark_mode=cfg['data'].get('train_watermark_mode', 'per_epoch'),
        is_train=True,
        max_images=cfg['data'].get('max_train_images', None),
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )

    val_dataset = WatermarkImageDataset(
        data_dir=cfg['data']['val_dir'],
        image_size=image_size,
        watermark_length=watermark_length,
        watermark_seed=watermark_seed,
        watermark_mode=cfg['data'].get('val_watermark_mode', 'fixed'),
        is_train=False,
        max_images=cfg['data'].get('max_val_images', None),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed + 1),
    )

    print(f"[Train] Train set: {len(train_dataset)} images, Val set: {len(val_dataset)} images")

    # --- Training scale summary ---
    total_train_available = len(glob.glob(os.path.join(cfg['data']['train_dir'], '*.jpg'))) + \
                           len(glob.glob(os.path.join(cfg['data']['train_dir'], '*.png'))) + \
                           len(glob.glob(os.path.join(cfg['data']['train_dir'], '*.jpeg')))
    total_val_available = len(glob.glob(os.path.join(cfg['data']['val_dir'], '*.jpg'))) + \
                         len(glob.glob(os.path.join(cfg['data']['val_dir'], '*.png'))) + \
                         len(glob.glob(os.path.join(cfg['data']['val_dir'], '*.jpeg')))
    steps_per_epoch = len(train_dataset) // batch_size
    total_steps = steps_per_epoch * cfg['train']['epochs']
    noise_type = get_noise_layer_type(cfg)
    print(f"[Scale] Train images: {len(train_dataset)} / {total_train_available}")
    print(f"[Scale] Val images:   {len(val_dataset)} / {total_val_available}")
    print(f"[Scale] Batch size: {batch_size}")
    print(f"[Scale] Steps per epoch: {steps_per_epoch}")
    print(f"[Scale] Epochs: {cfg['train']['epochs']}")
    print(f"[Scale] Total training steps: {total_steps}")
    print(f"[Scale] Noise layer: {noise_type}")

    # --- Diffusion ---
    timesteps = cfg['diffusion']['timesteps']
    betas = get_named_beta_schedule(cfg['diffusion']['beta_schedule'], timesteps)
    diffusion = GaussianDiffusion(
        betas=torch.tensor(betas, dtype=torch.float32),
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )

    # --- Model ---
    model_cfg = cfg['model']
    active_wm_map_flat_floor = float(
        get_stage_training_setting(
            cfg['train'],
            'wm_map_flat_floor',
            model_cfg.get('wm_map_flat_floor', 0.2),
        )
    )
    if not 0.0 <= active_wm_map_flat_floor <= 1.0:
        raise ValueError("active wm_map_flat_floor must be in [0, 1]")
    model = WatermarkConditionedUNet(
        image_size=image_size,
        base_channels=model_cfg['base_channels'],
        cond_dim=model_cfg['cond_dim'],
        watermark_length=watermark_length,
        use_pretrained_unet=model_cfg['use_pretrained_unet'],
        pretrained_path=model_cfg['pretrained_path'],
        use_watermark_time_emb=model_cfg.get('use_watermark_time_emb', True),
        use_watermark_spatial_map=model_cfg.get('use_watermark_spatial_map', True),
        wm_map_channels=model_cfg.get('wm_map_channels', 4),
        wm_map_size=model_cfg.get('wm_map_size', 16),
        wm_time_scale=model_cfg.get('wm_time_scale', 1.0),
        wm_map_scale=model_cfg.get('wm_map_scale', 1.0),
        use_content_gated_wm_map=model_cfg.get(
            'use_content_gated_wm_map', False
        ),
        wm_map_flat_floor=active_wm_map_flat_floor,
    ).to(device)

    # --- Watermark Decoder ---
    decoder = build_watermark_decoder(
        cfg,
        watermark_length=watermark_length,
    ).to(device)
    decoder_cfg = cfg.get('decoder', {})
    print(
        "[Decoder] type={}, base_channels={}, hidden_dim={}, multiscale={}".format(
            decoder_cfg.get('type', 'residual_multiscale'),
            decoder_cfg.get('base_channels', 32),
            decoder_cfg.get('hidden_dim', 512),
            decoder_cfg.get('use_multiscale', True),
        )
    )

    # Unified layers consume and return [0, 1]. The diffusion model and
    # decoder retain their existing [-1, 1] contracts around this boundary.
    noise_layer = build_noise_layer(cfg).to(device)
    use_noise_layer = noise_type != 'none'
    print(f"[NoiseLayer] type: {noise_type}")
    if noise_type == 'mixed':
        noise_cfg = cfg.get('noise_layer', {})
        mixed_cfg = noise_cfg.get('mixed', {})
        mixed_candidates = mixed_cfg.get('candidates', ['pimog', 'projector'])
        mixed_probs = mixed_cfg.get('probs', noise_cfg.get('mixed_probs', [0.5, 0.5]))
        print(f"[NoiseLayer] layers: {', '.join(mixed_candidates)}")
        print(f"[NoiseLayer] probs: {mixed_probs}")
    elif use_noise_layer:
        print(f"[NoiseLayer] {noise_layer.__class__.__name__} enabled")

    experiment_cfg = cfg.get('experiment', {})
    if experiment_cfg:
        if not isinstance(experiment_cfg, dict):
            raise TypeError("experiment must be a mapping")
        experiment_label = str(experiment_cfg.get('label', '')).strip()
        configured_candidates = (
            list(noise_layer.names)
            if noise_type == 'mixed'
            else ([noise_type] if use_noise_layer else [])
        )
        expected_candidates = [
            str(name).lower()
            for name in experiment_cfg.get('expected_noise_candidates', [])
        ]
        excluded_candidates = {
            str(name).lower()
            for name in experiment_cfg.get('excluded_noise_candidates', [])
        }
        curriculum_candidates = []
        for phase in cfg.get('train', {}).get('noise_curriculum', []):
            phase_candidates = phase.get('candidates', configured_candidates)
            curriculum_candidates.extend(
                str(name).lower() for name in phase_candidates
            )
        all_active_candidates = set(configured_candidates + curriculum_candidates)
        if expected_candidates and set(expected_candidates) != all_active_candidates:
            raise ValueError(
                "Experiment degradation guard failed: expected "
                f"{expected_candidates}, configured {sorted(all_active_candidates)}."
            )
        forbidden_active = sorted(all_active_candidates & excluded_candidates)
        if forbidden_active:
            raise ValueError(
                "Experiment degradation guard failed; excluded candidates are active: "
                f"{forbidden_active}"
            )
        if experiment_label:
            print(f"[Experiment] {experiment_label}")
        print(
            "[Experiment] active_degradations="
            f"{','.join(sorted(all_active_candidates)) or 'clean'}"
        )
        if excluded_candidates:
            print(
                "[Experiment] excluded_degradations="
                f"{','.join(sorted(excluded_candidates))}"
            )

    # --- Encoder adaptation mode and optimizer ---
    encoder_train_mode = get_encoder_train_mode(cfg['train'])
    freeze_encoder = encoder_train_mode == 'frozen'
    partial_output_blocks = int(
        cfg['train'].get('partial_output_blocks', 2)
    )
    freeze_watermark_map_mlp = bool(
        cfg['train'].get('freeze_watermark_map_mlp', True)
    )
    configure_encoder_train_mode(
        model,
        encoder_train_mode,
        partial_output_blocks=partial_output_blocks,
        freeze_watermark_map_mlp=freeze_watermark_map_mlp,
    )
    optimizer = build_training_optimizer(
        model,
        decoder,
        cfg['train']['lr'],
        encoder_lr=cfg['train'].get('encoder_lr'),
        decoder_lr=cfg['train'].get('decoder_lr'),
    )
    optimizer_ids = validate_optimizer_parameter_ownership(
        optimizer,
        model,
        decoder,
        freeze_encoder,
    )

    encoder_parameters = list(model.parameters())
    decoder_parameters = list(decoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_ids = {id(parameter) for parameter in decoder_parameters}
    other_optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group['params']
        if id(parameter) not in encoder_ids and id(parameter) not in decoder_ids
    ]
    print(
        f"[Encoder] train_mode={encoder_train_mode} "
        f"partial_output_blocks={partial_output_blocks} "
        f"freeze_watermark_map_mlp={freeze_watermark_map_mlp}"
    )
    print(f"[Parameters] encoder_total={count_parameters(encoder_parameters)}")
    print(
        "[Parameters] encoder_trainable="
        f"{count_parameters(p for p in encoder_parameters if p.requires_grad)}"
    )
    print(
        "[Parameters] decoder_trainable="
        f"{count_parameters(p for p in decoder_parameters if p.requires_grad)}"
    )
    print(
        "[Parameters] other_trainable="
        f"{count_parameters(other_optimizer_parameters)}"
    )
    print(
        "[Parameters] optimizer_total="
        f"{count_parameters(parameter for group in optimizer.param_groups for parameter in group['params'])}"
    )
    print(
        "[Parameters] optimizer_contains_encoder="
        f"{bool(optimizer_ids & encoder_ids)}"
    )
    for group in optimizer.param_groups:
        print(
            f"[Optimizer] group={group.get('name', 'unnamed')} "
            f"base_lr={group.get('base_lr', group['lr']):g} "
            f"parameters={count_parameters(group['params'])}"
        )

    # --- AMP ---
    use_amp = cfg['train'].get('use_amp', False)
    amp_enabled = use_amp and device.type == 'cuda'
    amp_init_scale = float(cfg['train'].get('amp_init_scale', 1024.0))
    amp_growth_interval = int(cfg['train'].get('amp_growth_interval', 2000))
    scaler = (
        torch.amp.GradScaler(
            'cuda',
            init_scale=amp_init_scale,
            growth_interval=amp_growth_interval,
        )
        if amp_enabled else None
    )
    print(f"[Train] AMP autocast: {'enabled' if amp_enabled else 'disabled'}")
    if scaler is not None:
        print(
            f"[Train] AMP init_scale={amp_init_scale:g}, "
            f"growth_interval={amp_growth_interval}"
        )

    # --- Checkpoint loading ---
    resume_path = cfg.get('_resume_path', None)
    init_from_path = cfg.get('_init_from_path', None)
    start_epoch = 1
    global_step = 0
    if resume_path:
        start_epoch, global_step = resume_training(
            resume_path, model, decoder, optimizer, scaler,
            train_generator, device,
        )
        if scaler is not None:
            resume_scale = float(scaler.get_scale())
            reset_below = float(
                cfg['train'].get('amp_reset_scale_on_resume_below', 0.0)
            )
            if reset_below > 0.0 and resume_scale < reset_below:
                scaler_state = scaler.state_dict()
                scaler_state['scale'] = amp_init_scale
                scaler_state['_growth_tracker'] = 0
                scaler.load_state_dict(scaler_state)
                print(
                    f"[Resume] Reset collapsed AMP scale "
                    f"{resume_scale:g}->{amp_init_scale:g}."
                )
    elif init_from_path:
        start_epoch, global_step = init_from_checkpoint(
            init_from_path,
            model,
            decoder,
            cfg['train'].get('reset_decoder', False),
            device,
            require_full_match=encoder_train_mode in {'frozen', 'partial'},
        )

    encoder_start_digest = parameter_digest(model) if freeze_encoder else None
    decoder_start_snapshot = (
        clone_parameter_values(decoder) if freeze_encoder else None
    )
    freeze_gradient_checked = False
    freeze_successful_steps = 0
    freeze_parameter_checked = False

    # --- Loss weights ---
    initial_loss_weights = get_loss_weights(cfg, 0)
    region_guidance_cfg = get_stage_training_setting(
        cfg['train'],
        'region_guidance',
        {},
    )
    region_guidance_enabled = bool(
        region_guidance_cfg.get(
            'enabled',
            initial_loss_weights['lambda_region'] > 0.0,
        )
    )
    outside_region_enabled, base_lambda_region_outside = (
        get_region_outside_settings(region_guidance_cfg)
    )
    if outside_region_enabled and not region_guidance_enabled:
        raise ValueError(
            "region_guidance.outside_enabled=true requires "
            "region_guidance.enabled=true"
        )
    residual_constraint_cfg = get_stage_training_setting(
        cfg['train'],
        'residual_constraint',
        {},
    )
    residual_constraint_settings = get_residual_constraint_settings(
        residual_constraint_cfg
    )
    content_gated_wm_map = bool(
        model_cfg.get('use_content_gated_wm_map', False)
    )
    needs_content_guidance = (
        region_guidance_enabled
        or residual_constraint_settings['enabled']
        or content_gated_wm_map
    )
    multi_attack_settings = get_multi_attack_settings(
        cfg['train'].get('multi_attack', {})
    )
    if multi_attack_settings['enabled'] and noise_type != 'mixed':
        raise ValueError("multi_attack training requires noise_layer.type=mixed")
    residual_spectral_settings = get_residual_spectral_settings(
        get_stage_training_setting(
            cfg['train'],
            'residual_spectral',
            {},
        )
    )

    # --- Timestep config ---
    wm_t_min = cfg['diffusion']['wm_t_min']
    wm_t_max = cfg['diffusion']['wm_t_max']
    train_t_start = cfg['diffusion']['train_t_start']

    # --- Training state ---
    epochs = cfg['train']['epochs']
    save_interval = cfg['train']['save_interval']
    sample_interval = cfg['train']['sample_interval']
    log_interval = cfg['train']['log_interval']
    debug_interval = cfg['train'].get('debug_interval', log_interval * 5)
    max_grad_norm = float(cfg['train'].get('max_grad_norm', 1.0))
    max_consecutive_nonfinite = int(
        cfg['train'].get('max_consecutive_nonfinite', 5)
    )
    base_lr = float(cfg['train']['lr'])
    validation_cfg = cfg.get('validation', {})
    validation_seed = int(validation_cfg.get('seed', seed + 4200))
    sync_validation_strength = bool(
        validation_cfg.get('sync_curriculum_strength', True)
    )
    evaluate_validation_per_type = bool(
        validation_cfg.get('evaluate_per_type', True)
    )
    validation_noise_names = ('pimog', 'oled', 'led', 'projector')
    fixed_validation_strengths = [
        float(value) for value in validation_cfg.get('fixed_strengths', [])
    ]
    fixed_validation_candidates = list(
        validation_cfg.get('fixed_candidates', validation_noise_names)
    )
    fixed_validation_interval = int(
        validation_cfg.get('fixed_matrix_interval', 1)
    )
    fixed_validation_max_batches = int(
        validation_cfg.get('fixed_matrix_max_batches', 0)
    )
    fixed_matrix_for_checkpoint = bool(
        validation_cfg.get('use_fixed_matrix_for_checkpoint', False)
    )
    if any(not 0.0 <= value <= 1.0 for value in fixed_validation_strengths):
        raise ValueError("validation.fixed_strengths values must be in [0, 1]")
    unknown_fixed_candidates = set(fixed_validation_candidates).difference(
        validation_noise_names
    )
    if unknown_fixed_candidates:
        raise ValueError(
            "validation.fixed_candidates contains unsupported names: "
            f"{sorted(unknown_fixed_candidates)}"
        )
    if fixed_validation_interval <= 0:
        raise ValueError("validation.fixed_matrix_interval must be positive")
    if fixed_validation_max_batches < 0:
        raise ValueError("validation.fixed_matrix_max_batches must be >= 0")
    if fixed_matrix_for_checkpoint and not fixed_validation_strengths:
        raise ValueError(
            "validation.use_fixed_matrix_for_checkpoint requires "
            "validation.fixed_strengths"
        )
    if fixed_matrix_for_checkpoint and fixed_validation_interval != 1:
        raise ValueError(
            "validation.fixed_matrix_interval must be 1 when the fixed "
            "matrix is used for checkpoint selection"
        )
    if (
        fixed_validation_strengths
        and use_noise_layer
        and noise_type != 'mixed'
        and fixed_validation_candidates != [noise_type]
    ):
        raise ValueError(
            "For a non-mixed noise layer, validation.fixed_candidates must "
            f"be exactly [{noise_type!r}]"
        )
    all_trainable_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group['params']
    ]
    consecutive_nonfinite = 0
    last_validation_metadata = None
    # --- CSV loggers ---
    train_log_path = os.path.join(log_dir, 'train_log.csv')
    val_log_path = os.path.join(log_dir, 'val_log.csv')
    fixed_val_log_path = os.path.join(log_dir, 'val_fixed_matrix.csv')
    sample_log_path = os.path.join(log_dir, 'sample_log.csv')

    csv_mode = 'a' if resume_path else 'w'
    train_fieldnames = [
        'epoch', 'global_step', 'loss_total', 'loss_diff', 'loss_img', 'loss_wm',
        'loss_wm_clean', 'loss_wm_degraded',
        'loss_wm_degraded_mean', 'loss_wm_degraded_worst',
        'loss_delta', 'loss_tv', 'loss_topk', 'loss_channel', 'loss_region',
        'loss_region_enrichment', 'loss_region_outside', 'loss_region_total',
        'loss_spectral_total', 'loss_spectral_peak', 'loss_spectral_anisotropy',
        'spectral_loss_scale', 'target_inside_ratio', 'target_outside_ratio',
        'mask_area_ratio', 'inside_energy_ratio', 'outside_energy_ratio',
        'flat_energy_ratio', 'texture_energy_ratio',
        'residual_dx_energy', 'residual_dy_energy', 'directional_ratio',
        'fft_peak_ratio', 'fft_midband_ratio', 'cross_image_correlation',
        'bit_acc', 'bit_acc_clean', 'bit_acc_degraded',
        'psnr', 'ssim', 'logits_std', 'sigmoid_mean',
        'bit_flip_image_delta', 'bit_flip_logit_delta', 'lr', 'noise_layer_type',
        'noise_apply_prob', 'noise_strength', 'lambda_wm_clean',
        'lambda_wm_degraded', 'lambda_region_enrichment',
        'lambda_region_outside', 'encoder_lr', 'decoder_lr',
        'grad_norm', 'amp_scale', 'step_skipped',
    ]
    train_log_exists = (
        os.path.exists(train_log_path) and os.path.getsize(train_log_path) > 0
    )
    if csv_mode == 'a' and train_log_exists:
        # Preserve the existing schema when resuming a pre-change run. New
        # fields are ignored for that legacy CSV instead of shifting columns
        # under an older header; fresh A/B runs receive the full schema.
        with open(train_log_path, newline='') as existing_train_csv:
            existing_header = next(csv.reader(existing_train_csv), None)
        if existing_header:
            train_fieldnames = existing_header

    train_csv = open(train_log_path, csv_mode, newline='')
    train_writer = csv.DictWriter(
        train_csv,
        fieldnames=train_fieldnames,
        extrasaction='ignore',
    )
    if csv_mode == 'w' or not train_log_exists:
        train_writer.writeheader()

    val_fieldnames = [
        'epoch', 'global_step', 'curriculum_phase', 'degradation_stage',
        'noise_strength', 'active_candidates',
        'bit_acc_clean', 'bit_acc_degraded',
        'bit_acc_pimog', 'bit_acc_oled', 'bit_acc_led',
        'bit_acc_projector',
        'loss_wm_clean', 'loss_wm_degraded',
        'loss_pimog', 'loss_oled', 'loss_led', 'loss_projector',
        'degraded_macro_acc', 'degraded_worst_acc',
        'degraded_macro_loss',
        'score_eval_source', 'fixed_macro_acc', 'fixed_worst_acc',
        'fixed_macro_loss', 'fixed_macro_ssim',
        'psnr', 'ssim_watermarked', 'ssim_degraded',
        'ssim_pimog', 'ssim_oled', 'ssim_led', 'ssim_projector',
        'mae_watermarked', 'topk_watermarked',
        'tv_watermarked', 'channel_delta_std',
        'mask_area_ratio', 'inside_energy_ratio', 'outside_energy_ratio',
        'flat_energy_ratio', 'texture_energy_ratio',
        'residual_dx_energy', 'residual_dy_energy', 'directional_ratio',
        'fft_peak_ratio', 'fft_midband_ratio', 'cross_image_correlation',
        'balanced_score',
    ]
    val_log_exists = (
        os.path.exists(val_log_path) and os.path.getsize(val_log_path) > 0
    )
    if csv_mode == 'a' and val_log_exists:
        with open(val_log_path, newline='') as existing_val_csv:
            existing_header = next(csv.reader(existing_val_csv), None)
        if existing_header:
            val_fieldnames = existing_header
    val_csv = open(val_log_path, csv_mode, newline='')
    val_writer = csv.DictWriter(
        val_csv,
        fieldnames=val_fieldnames,
        extrasaction='ignore',
    )
    if csv_mode == 'w' or not val_log_exists:
        val_writer.writeheader()

    fixed_val_fieldnames = [
        'epoch', 'global_step', 'strength', 'noise_type',
        'bit_acc', 'loss_wm', 'ssim_end2end', 'num_bits', 'num_batches',
    ]
    fixed_val_log_exists = (
        os.path.exists(fixed_val_log_path)
        and os.path.getsize(fixed_val_log_path) > 0
    )
    if csv_mode == 'a' and fixed_val_log_exists:
        with open(fixed_val_log_path, newline='') as existing_fixed_val_csv:
            existing_header = next(csv.reader(existing_fixed_val_csv), None)
        if existing_header:
            fixed_val_fieldnames = existing_header
    fixed_val_csv = open(fixed_val_log_path, csv_mode, newline='')
    fixed_val_writer = csv.DictWriter(
        fixed_val_csv,
        fieldnames=fixed_val_fieldnames,
        extrasaction='ignore',
    )
    if (
        not os.path.exists(fixed_val_log_path)
        or os.path.getsize(fixed_val_log_path) == 0
    ):
        fixed_val_writer.writeheader()

    sample_fieldnames = [
        'epoch', 'global_step', 'noise_layer_type', 'noise_strength', 'bit_acc',
        'psnr_watermarked', 'psnr_degraded',
        'ssim_watermarked', 'ssim_degraded',
        'mae_watermarked', 'mae_degraded',
        'max_abs_delta_watermarked', 'max_abs_delta_degraded',
        'topk_abs_delta_watermarked', 'tv_watermarked',
        'channel_delta_r', 'channel_delta_g', 'channel_delta_b',
        'channel_delta_std',
        'mask_area_ratio', 'inside_energy_ratio', 'outside_energy_ratio',
        'flat_energy_ratio', 'texture_energy_ratio',
        'residual_dx_energy', 'residual_dy_energy', 'directional_ratio',
        'fft_peak_ratio', 'fft_midband_ratio', 'cross_image_correlation',
        'logits_std', 'sigmoid_mean',
    ]
    sample_log_exists = (
        os.path.exists(sample_log_path) and os.path.getsize(sample_log_path) > 0
    )
    if csv_mode == 'a' and sample_log_exists:
        with open(sample_log_path, newline='') as existing_sample_csv:
            existing_header = next(csv.reader(existing_sample_csv), None)
        if existing_header:
            sample_fieldnames = existing_header
    sample_csv = open(sample_log_path, csv_mode, newline='')
    sample_writer = csv.DictWriter(
        sample_csv,
        fieldnames=sample_fieldnames,
        extrasaction='ignore',
    )
    if not os.path.exists(sample_log_path) or os.path.getsize(sample_log_path) == 0:
        sample_writer.writeheader()

    # --- Training loop ---
    print(
        f"[Train] Starting training: {epochs} epochs, "
        f"log_interval={log_interval}, debug_interval={debug_interval}"
    )
    print(
        f"[Train] stage={cfg['train'].get('stage', 'legacy')} "
        f"initial lambda_diff={initial_loss_weights['lambda_diff']}, "
        f"lambda_img={initial_loss_weights['lambda_img']}, "
        f"lambda_wm={initial_loss_weights['lambda_wm']}, "
        f"lambda_delta={initial_loss_weights['lambda_delta']}, "
        f"lambda_tv={initial_loss_weights['lambda_tv']}, "
        f"lambda_topk={initial_loss_weights['lambda_topk']}, "
        f"lambda_channel={initial_loss_weights['lambda_channel']}, "
        f"lambda_region={initial_loss_weights['lambda_region']}"
    )
    print(
        f"[Train] region_guidance="
        f"{'enabled' if region_guidance_enabled else 'disabled'} "
        f"config={region_guidance_cfg}"
    )
    print(
        f"[Train] outside_region_loss="
        f"{'enabled' if outside_region_enabled else 'disabled'} "
        f"lambda_outside={base_lambda_region_outside}"
    )
    print(
        f"[Train] residual_constraint={residual_constraint_settings} "
        f"content_gated_wm_map={content_gated_wm_map} "
        f"wm_map_flat_floor={model.wm_map_flat_floor}"
    )
    print(f"[Train] multi_attack={multi_attack_settings}")
    print(f"[Train] residual_spectral={residual_spectral_settings}")
    if cfg['train'].get('use_loss_schedule', False):
        print(f"[Train] loss schedule enabled: {cfg['train'].get('loss_schedule', [])}")
    print(f"[Train] wm_t range: [{wm_t_min}, {wm_t_max}), noise_layer={noise_type}")
    print(
        f"[Train] max_grad_norm={max_grad_norm}, "
        f"max_consecutive_nonfinite={max_consecutive_nonfinite}"
    )
    print(
        f"[Validation] seed={validation_seed}, "
        f"sync_curriculum_strength={sync_validation_strength}, "
        f"evaluate_per_type={evaluate_validation_per_type}"
    )
    if fixed_validation_strengths and use_noise_layer:
        print(
            "[Validation] fixed_matrix="
            f"{fixed_validation_candidates} x {fixed_validation_strengths}, "
            f"interval={fixed_validation_interval}, "
            f"max_batches={fixed_validation_max_batches or 'all'}, "
            f"checkpoint_score={fixed_matrix_for_checkpoint}"
        )

    for epoch in range(start_epoch, epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        set_encoder_training_mode(
            model,
            encoder_train_mode,
            partial_output_blocks,
            freeze_watermark_map_mlp,
        )
        decoder.train()
        noise_layer.train()

        for batch in train_loader:
            cover_img = batch['image'].to(device)    # [B, 3, H, W], [-1, 1]
            B = cover_img.size(0)
            wm_bits = generate_train_watermark(B, watermark_length, device)
            cover_01_for_loss = (cover_img + 1.0) / 2.0
            if needs_content_guidance:
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=False,
                ):
                    region_allowance, region_penalty = (
                        build_edge_texture_guidance(
                            cover_01_for_loss,
                            region_guidance_cfg,
                        )
                    )
            else:
                region_allowance = None
                region_penalty = None
            optimizer.zero_grad(set_to_none=True)
            loss_weights = get_loss_weights(cfg, global_step)
            lambda_diff = loss_weights['lambda_diff']
            lambda_img = loss_weights['lambda_img']
            lambda_wm = loss_weights['lambda_wm']
            lambda_delta = loss_weights['lambda_delta']
            lambda_tv = loss_weights['lambda_tv']
            lambda_topk = loss_weights['lambda_topk']
            lambda_channel = loss_weights['lambda_channel']
            lambda_region = get_active_region_weight(
                loss_weights['lambda_region'],
                global_step,
                region_guidance_cfg,
            )
            if str(region_guidance_cfg.get('loss_mode', '')).lower() == 'energy_ratio':
                localization_targets = get_active_localization_targets(
                    region_guidance_cfg,
                    global_step,
                )
            else:
                localization_targets = {
                    'inside': float('nan'),
                    'outside': float('nan'),
                    'progress': float('nan'),
                }
            lambda_region_outside = (
                get_active_region_weight(
                    base_lambda_region_outside,
                    global_step,
                    region_guidance_cfg,
                )
                if outside_region_enabled
                else 0.0
            )
            curriculum = get_noise_curriculum_state(
                cfg, global_step, lambda_wm, use_noise_layer
            )
            for param_group in optimizer.param_groups:
                group_base_lr = float(param_group.get('base_lr', base_lr))
                param_group['lr'] = group_base_lr * curriculum['lr_scale']

            # ========================================================
            # Official PIMoG is either fully enabled or disabled.
            # ========================================================
            active_noise_type = noise_type

            # ========================================================
            # 1. Diffusion noise prediction loss (full timestep range)
            # ========================================================
            # Do not execute the full-timestep U-Net pass when a stage
            # explicitly disables it with lambda_diff=0. An unused FP16
            # diagnostic can overflow after the model specializes to the
            # low-timestep watermark objective, even though every active loss
            # remains finite. Active diffusion stages retain the strict
            # finite-value guard below.
            if lambda_diff > 0.0:
                t_diff = torch.randint(
                    0, timesteps, (B,), device=device
                ).long()
                noise = torch.randn_like(cover_img)
                x_t_diff = diffusion.q_sample(
                    cover_img, t_diff, noise=noise
                )
                t_diff_scaled = t_diff.float() * (1000.0 / timesteps)

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=amp_enabled,
                ):
                    pred_noise = model(
                        x_t=x_t_diff,
                        t=t_diff_scaled,
                        cover_img=cover_img,
                        wm_bits=wm_bits,
                        content_mask=region_allowance,
                    )
                    loss_diff = F.mse_loss(pred_noise, noise)

                if not tensor_is_finite(loss_diff):
                    consecutive_nonfinite += 1
                    optimizer.zero_grad(set_to_none=True)
                    print(
                        f"[SKIP S{global_step:06d}] non-finite active "
                        f"loss_diff before noise layer; "
                        f"consecutive={consecutive_nonfinite}"
                    )
                    if consecutive_nonfinite >= max_consecutive_nonfinite:
                        raise FloatingPointError(
                            "Repeated non-finite active diffusion forward "
                            "values; stop before saving a contaminated "
                            "checkpoint."
                        )
                    global_step += 1
                    continue

                # Backpropagate this branch immediately so its large U-Net
                # activation graph is released before the watermark branch.
                diffusion_objective = lambda_diff * loss_diff
                if diffusion_objective.requires_grad:
                    if scaler is not None:
                        scaler.scale(diffusion_objective).backward()
                    else:
                        diffusion_objective.backward()
                loss_diff = loss_diff.detach()
                del pred_noise, diffusion_objective, x_t_diff
            else:
                loss_diff = cover_img.new_zeros(())

            # ========================================================
            # 2. Watermark + image fidelity loss (small timestep range)
            # ========================================================
            # KEY: t_wm from a SMALL range so pred_x0 is meaningful
            t_wm = torch.randint(wm_t_min, wm_t_max, (B,), device=device).long()
            noise_wm = torch.randn_like(cover_img)
            x_t_wm = diffusion.q_sample(cover_img, t_wm, noise=noise_wm)

            t_wm_scaled = t_wm.float() * (1000.0 / timesteps)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred_noise_wm = model(
                    x_t=x_t_wm,
                    t=t_wm_scaled,
                    cover_img=cover_img,
                    wm_bits=wm_bits,
                    content_mask=region_allowance,
                )

                # KEY: NO .detach() — loss_wm backpropagates through the U-Net.
                pred_x0_raw = predict_start_from_noise(
                    diffusion, x_t_wm, t_wm, pred_noise_wm
                ).clamp(-1, 1)
                pred_x0 = constrain_watermark_residual(
                    pred_x0_raw,
                    cover_img,
                    region_allowance,
                    residual_constraint_cfg,
                )

                # Image fidelity loss in [-1, 1]
                loss_img = F.l1_loss(pred_x0, cover_img)
                loss_delta = (pred_x0 - cover_img).abs().mean()

                # Unified degradation layers use [0, 1].
                pred_x0_01 = (pred_x0 + 1.0) / 2.0
                residual_01 = pred_x0_01 - cover_01_for_loss
                loss_tv = residual_tv_loss(residual_01)
                loss_topk = residual_topk_loss(
                    residual_01,
                    cfg['train'].get('topk_delta_fraction', 0.01),
                )
                loss_channel = residual_channel_balance_loss(residual_01)
                if region_guidance_enabled:
                    with torch.amp.autocast(
                        device_type=device.type,
                        enabled=False,
                    ):
                        loss_region_enrichment = compute_region_guidance_loss(
                            residual_01,
                            region_allowance,
                            region_penalty,
                            region_guidance_cfg,
                            global_step=global_step,
                        )
                        loss_region_outside = (
                            residual_outside_region_loss(
                                residual_01,
                                region_allowance,
                                eps=float(region_guidance_cfg.get('eps', 1e-6)),
                            )
                            if outside_region_enabled
                            else residual_01.new_zeros(())
                        )
                else:
                    loss_region_enrichment = residual_01.new_zeros(())
                    loss_region_outside = residual_01.new_zeros(())
                loss_region_total = combine_region_guidance_losses(
                    loss_region_enrichment,
                    loss_region_outside,
                    lambda_region,
                    lambda_region_outside,
                    outside_region_enabled,
                )
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=False,
                ):
                    (
                        loss_spectral_total,
                        loss_spectral_peak,
                        loss_spectral_anisotropy,
                        spectral_loss_scale,
                    ) = residual_spectral_regularization_loss(
                        residual_01,
                        residual_spectral_settings,
                        global_step,
                    )

            # Keep geometric warps, blur kernels and power functions in FP32.
            noise_applied = (
                use_noise_layer
                and torch.rand((), device=device).item() < curriculum['apply_prob']
            )
            if noise_applied:
                attack_source = (
                    pred_x0_01.detach()
                    if curriculum['detach_degraded_from_model']
                    else pred_x0_01
                )
                attacked_variants = []
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    if multi_attack_settings['enabled']:
                        selected_candidates = select_multi_attack_candidates(
                            curriculum['candidates'],
                            curriculum['probs'],
                            multi_attack_settings['attacks_per_batch'],
                            device,
                        )
                        for candidate in selected_candidates:
                            attacked_candidate, selected_name = (
                                apply_degradation_with_strength(
                                    attack_source,
                                    noise_layer,
                                    noise_type,
                                    curriculum['strength'],
                                    candidates=[candidate],
                                    probs=[1.0],
                                )
                            )
                            attacked_variants.append(
                                (selected_name, attacked_candidate)
                            )
                        active_noise_type = "multi:" + "+".join(
                            name for name, _ in attacked_variants
                        )
                    else:
                        attacked_01, selected_noise_type = (
                            apply_degradation_with_strength(
                                attack_source,
                                noise_layer,
                                noise_type,
                                curriculum['strength'],
                                candidates=curriculum['candidates'],
                                probs=curriculum['probs'],
                            )
                        )
                        attacked_variants.append(
                            (selected_noise_type, attacked_01)
                        )
                        active_noise_type = (
                            f"mixed:{selected_noise_type}"
                            if noise_type == 'mixed'
                            else selected_noise_type
                        )
                lambda_wm_degraded_active = curriculum['lambda_wm_degraded']
            else:
                attacked_01 = pred_x0_01
                attacked_variants = [('clean', attacked_01)]
                active_noise_type = 'clean'
                lambda_wm_degraded_active = 0.0

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred_logits_clean = decoder(pred_x0)
                loss_wm_clean = F.binary_cross_entropy_with_logits(
                    pred_logits_clean, wm_bits.float()
                )
                if noise_applied:
                    degraded_logits = []
                    degraded_losses = []
                    for _, attacked_variant in attacked_variants:
                        logits_variant = decoder(
                            attacked_variant.mul(2.0).sub(1.0)
                        )
                        degraded_logits.append(logits_variant)
                        degraded_losses.append(
                            F.binary_cross_entropy_with_logits(
                                logits_variant,
                                wm_bits.float(),
                            )
                        )
                    degraded_loss_values = torch.stack(degraded_losses)
                    loss_wm_degraded_mean = degraded_loss_values.mean()
                    loss_wm_degraded_worst = degraded_loss_values.max()
                    loss_wm_degraded = (
                        multi_attack_settings['lambda_mean']
                        * loss_wm_degraded_mean
                        + multi_attack_settings['lambda_worst']
                        * loss_wm_degraded_worst
                        if multi_attack_settings['enabled']
                        else loss_wm_degraded_mean
                    )
                    worst_attack_index = int(
                        degraded_loss_values.detach().argmax().item()
                    )
                    pred_logits_degraded = degraded_logits[worst_attack_index]
                    worst_attack_name, attacked_01 = attacked_variants[
                        worst_attack_index
                    ]
                    if multi_attack_settings['enabled']:
                        active_noise_type += f":worst={worst_attack_name}"
                else:
                    pred_logits_degraded = pred_logits_clean
                    loss_wm_degraded = loss_wm_clean
                    loss_wm_degraded_mean = loss_wm_clean
                    loss_wm_degraded_worst = loss_wm_clean

            pred_logits = pred_logits_degraded
            loss_wm = loss_wm_degraded if noise_applied else loss_wm_clean
            logits_mean = pred_logits.detach().mean().item()
            logits_std = pred_logits.detach().std().item()
            sigmoid_mean = torch.sigmoid(pred_logits.detach()).mean().item()

            # ========================================================
            # 3. Total loss
            # ========================================================
            watermark_objective = (
                lambda_img * loss_img
                + curriculum['lambda_wm_clean'] * loss_wm_clean
                + lambda_wm_degraded_active * loss_wm_degraded
                + lambda_delta * loss_delta
                + lambda_tv * loss_tv
                + lambda_topk * loss_topk
                + lambda_channel * loss_channel
                + loss_region_total
                + loss_spectral_total
            )

            forward_values = (
                pred_x0, attacked_01, pred_logits_clean, pred_logits_degraded,
                loss_img, loss_wm_clean, loss_wm_degraded,
                loss_region_enrichment, loss_region_outside, loss_region_total,
                loss_spectral_total, loss_spectral_peak,
                loss_spectral_anisotropy,
                watermark_objective,
            )
            if not all(tensor_is_finite(value) for value in forward_values):
                consecutive_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[SKIP S{global_step:06d}] non-finite watermark forward "
                    f"(noise={active_noise_type}, strength={curriculum['strength']:.2f}, "
                    f"amp_scale={scaler.get_scale() if scaler is not None else 1.0:g}); "
                    f"consecutive={consecutive_nonfinite}"
                )
                if consecutive_nonfinite >= max_consecutive_nonfinite:
                    raise FloatingPointError(
                        "Repeated non-finite watermark forward values; stop before "
                        "saving a contaminated checkpoint."
                    )
                global_step += 1
                continue

            # ========================================================
            # 4. Backward
            # ========================================================
            amp_scale_before = scaler.get_scale() if scaler is not None else 1.0
            if scaler is not None:
                scaler.scale(watermark_objective).backward()
                scaler.unscale_(optimizer)
            else:
                watermark_objective.backward()

            if freeze_encoder and not freeze_gradient_checked:
                encoder_gradient = parameter_grad_norm(model.parameters())
                decoder_gradient = parameter_grad_norm(decoder.parameters())
                other_gradient = parameter_grad_norm(other_optimizer_parameters)
                print(
                    "[FreezeCheck] encoder_grad_norm="
                    f"{encoder_gradient} decoder_grad_norm={decoder_gradient} "
                    f"other_trainable_grad_norm={other_gradient}"
                )
                if encoder_gradient not in (None, 0.0):
                    raise RuntimeError(
                        "Frozen encoder received a non-zero gradient."
                    )
                if decoder_gradient is None or decoder_gradient <= 0.0:
                    raise RuntimeError(
                        "Decoder did not receive a non-zero gradient."
                    )
                freeze_gradient_checked = True

            model_gn = grad_norm(model)
            decoder_gn = grad_norm(decoder)
            wm_mlp_gn = grad_norm(model.watermark_mlp)
            wm_map_mlp_gn = (
                grad_norm(model.watermark_map_mlp)
                if hasattr(model, 'watermark_map_mlp')
                else float('nan')
            )
            grads_finite = gradients_are_finite(all_trainable_parameters)
            total_grad_norm = float('nan')
            if grads_finite:
                total_grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    all_trainable_parameters,
                    max_norm=max_grad_norm if max_grad_norm > 0 else float('inf'),
                )
                total_grad_norm = float(total_grad_norm_tensor.detach().item())
            step_skipped = not (grads_finite and math.isfinite(total_grad_norm))

            if step_skipped:
                consecutive_nonfinite += 1
                if scaler is not None:
                    reduced_scale = max(
                        float(cfg['train'].get('amp_min_scale', 1.0)),
                        float(amp_scale_before) * 0.5,
                    )
                    scaler.update(new_scale=reduced_scale)
                print(
                    f"[SKIP S{global_step:06d}] non-finite gradients "
                    f"(noise={active_noise_type}, model_gn={model_gn}, "
                    f"decoder_gn={decoder_gn}, amp_scale={amp_scale_before:g}); "
                    f"consecutive={consecutive_nonfinite}"
                )
                optimizer.zero_grad(set_to_none=True)
                if consecutive_nonfinite >= max_consecutive_nonfinite:
                    raise FloatingPointError(
                        "Repeated non-finite gradients; stop before saving a "
                        "contaminated checkpoint."
                    )
                global_step += 1
                continue

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            consecutive_nonfinite = 0
            amp_scale = scaler.get_scale() if scaler is not None else 1.0

            if freeze_encoder and not freeze_parameter_checked:
                freeze_successful_steps += 1
                if freeze_successful_steps >= 3:
                    encoder_end_digest = parameter_digest(model)
                    encoder_parameter_delta = (
                        0.0
                        if encoder_end_digest == encoder_start_digest
                        else float('inf')
                    )
                    decoder_parameter_delta = parameter_max_abs_delta(
                        decoder,
                        decoder_start_snapshot,
                    )
                    print(
                        "[FreezeCheck] encoder_parameter_delta="
                        f"{encoder_parameter_delta} decoder_parameter_delta="
                        f"{decoder_parameter_delta:.9g} checked_after_steps="
                        f"{freeze_successful_steps}"
                    )
                    if encoder_end_digest != encoder_start_digest:
                        raise RuntimeError(
                            "Frozen encoder parameters changed during startup."
                        )
                    if decoder_parameter_delta <= 0.0:
                        raise RuntimeError(
                            "Decoder parameters did not change during startup."
                        )
                    freeze_parameter_checked = True

            loss_total = (
                lambda_diff * loss_diff
                + lambda_img * loss_img.detach()
                + curriculum['lambda_wm_clean'] * loss_wm_clean.detach()
                + lambda_wm_degraded_active * loss_wm_degraded.detach()
                + lambda_delta * loss_delta.detach()
                + lambda_tv * loss_tv.detach()
                + lambda_topk * loss_topk.detach()
                + lambda_channel * loss_channel.detach()
                + loss_region_total.detach()
                + loss_spectral_total.detach()
            )

            # ========================================================
            # 5. Metrics (no_grad for logging)
            # ========================================================
            with torch.no_grad():
                pred_bits_clean = (torch.sigmoid(pred_logits_clean) > 0.5).float()
                bit_acc_clean = (
                    pred_bits_clean == wm_bits
                ).float().mean().item()
                if noise_applied:
                    pred_bits_degraded = (
                        torch.sigmoid(pred_logits_degraded) > 0.5
                    ).float()
                    bit_acc_degraded = (
                        pred_bits_degraded == wm_bits
                    ).float().mean().item()
                    loss_wm_degraded_log = loss_wm_degraded.item()
                    bit_acc_degraded_text = f"{bit_acc_degraded:.3f}"
                    loss_wm_degraded_text = f"{loss_wm_degraded.item():.4f}"
                else:
                    # No degradation was evaluated on a curriculum clean batch.
                    # Reusing clean logits is useful for the loss graph but must
                    # not be reported as a measured degraded accuracy.
                    bit_acc_degraded = float('nan')
                    loss_wm_degraded_log = float('nan')
                    bit_acc_degraded_text = 'n/a'
                    loss_wm_degraded_text = 'n/a'
                bit_acc = bit_acc_degraded if noise_applied else bit_acc_clean

                cover_01 = (cover_img + 1.0) / 2.0
                psnr_val = compute_psnr(pred_x0_01, cover_01, max_val=1.0)
                ssim_val = (
                    compute_ssim(pred_x0_01, cover_01, max_val=1.0)
                    if global_step % log_interval == 0
                    else float('nan')
                )
                if region_allowance is not None:
                    flat_energy_ratio, texture_energy_ratio = (
                        residual_region_energy_ratios(
                            residual_01,
                            region_allowance,
                        )
                    )
                    flat_energy_ratio = flat_energy_ratio.item()
                    texture_energy_ratio = texture_energy_ratio.item()
                else:
                    flat_energy_ratio = float('nan')
                    texture_energy_ratio = float('nan')
                if global_step % log_interval == 0:
                    structure_metrics = residual_structure_metrics(
                        residual_01,
                        region_allowance,
                    )
                    structure_metrics_log = {
                        key: value.item()
                        for key, value in structure_metrics.items()
                    }
                else:
                    structure_metrics_log = {}

            # ========================================================
            # 6. Logging
            # ========================================================
            if global_step % log_interval == 0:
                group_lrs = {
                    group.get('name', 'joint'): float(group['lr'])
                    for group in optimizer.param_groups
                }
                encoder_lr_log = group_lrs.get(
                    'encoder', group_lrs.get('joint', float('nan'))
                )
                decoder_lr_log = group_lrs.get(
                    'decoder', group_lrs.get('joint', float('nan'))
                )
                log_data = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'loss_total': loss_total.item(),
                    'loss_diff': loss_diff.item(),
                    'loss_img': loss_img.item(),
                    'loss_wm': loss_wm.item(),
                    'loss_wm_clean': loss_wm_clean.item(),
                    'loss_wm_degraded': loss_wm_degraded_log,
                    'loss_wm_degraded_mean': (
                        loss_wm_degraded_mean.item()
                        if noise_applied else float('nan')
                    ),
                    'loss_wm_degraded_worst': (
                        loss_wm_degraded_worst.item()
                        if noise_applied else float('nan')
                    ),
                    'loss_delta': loss_delta.item(),
                    'loss_tv': loss_tv.item(),
                    'loss_topk': loss_topk.item(),
                    'loss_channel': loss_channel.item(),
                    # Retain the legacy column as the raw active region term.
                    'loss_region': loss_region_enrichment.item(),
                    'loss_region_enrichment': loss_region_enrichment.item(),
                    'loss_region_outside': loss_region_outside.item(),
                    'loss_region_total': loss_region_total.item(),
                    'loss_spectral_total': loss_spectral_total.item(),
                    'loss_spectral_peak': loss_spectral_peak.item(),
                    'loss_spectral_anisotropy': loss_spectral_anisotropy.item(),
                    'spectral_loss_scale': spectral_loss_scale,
                    'target_inside_ratio': localization_targets['inside'],
                    'target_outside_ratio': localization_targets['outside'],
                    'mask_area_ratio': structure_metrics_log.get(
                        'mask_area_ratio', float('nan')
                    ),
                    'inside_energy_ratio': structure_metrics_log.get(
                        'inside_energy_ratio', float('nan')
                    ),
                    'outside_energy_ratio': structure_metrics_log.get(
                        'outside_energy_ratio', float('nan')
                    ),
                    'flat_energy_ratio': flat_energy_ratio,
                    'texture_energy_ratio': texture_energy_ratio,
                    'residual_dx_energy': structure_metrics_log.get(
                        'dx_energy', float('nan')
                    ),
                    'residual_dy_energy': structure_metrics_log.get(
                        'dy_energy', float('nan')
                    ),
                    'directional_ratio': structure_metrics_log.get(
                        'directional_ratio', float('nan')
                    ),
                    'fft_peak_ratio': structure_metrics_log.get(
                        'fft_peak_ratio', float('nan')
                    ),
                    'fft_midband_ratio': structure_metrics_log.get(
                        'fft_midband_ratio', float('nan')
                    ),
                    'cross_image_correlation': structure_metrics_log.get(
                        'cross_image_correlation', float('nan')
                    ),
                    'bit_acc': bit_acc,
                    'bit_acc_clean': bit_acc_clean,
                    'bit_acc_degraded': bit_acc_degraded,
                    'psnr': psnr_val,
                    'ssim': ssim_val,
                    'logits_std': logits_std,
                    'sigmoid_mean': sigmoid_mean,
                    'bit_flip_image_delta': float('nan'),
                    'bit_flip_logit_delta': float('nan'),
                    'lr': optimizer.param_groups[0]['lr'],
                    'noise_layer_type': active_noise_type,
                    'noise_apply_prob': curriculum['apply_prob'],
                    'noise_strength': curriculum['strength'],
                    'lambda_wm_clean': curriculum['lambda_wm_clean'],
                    'lambda_wm_degraded': lambda_wm_degraded_active,
                    'lambda_region_enrichment': lambda_region,
                    'lambda_region_outside': lambda_region_outside,
                    'encoder_lr': encoder_lr_log,
                    'decoder_lr': decoder_lr_log,
                    'grad_norm': total_grad_norm,
                    'amp_scale': amp_scale,
                    'step_skipped': False,
                }

                is_debug_log = (
                    debug_interval > 0 and global_step % debug_interval == 0
                )
                if is_debug_log:
                    with torch.no_grad():
                        debug_count = min(2, B)
                        debug_x_t = x_t_wm[:debug_count]
                        debug_t = t_wm[:debug_count]
                        debug_t_scaled = t_wm_scaled[:debug_count]
                        debug_cover = cover_img[:debug_count]
                        debug_wm_a = wm_bits[:debug_count]
                        debug_wm_b = 1.0 - debug_wm_a
                        pred_noise_a = model(
                            x_t=debug_x_t,
                            t=debug_t_scaled,
                            cover_img=debug_cover,
                            wm_bits=debug_wm_a,
                            content_mask=(
                                region_allowance[:debug_count]
                                if region_allowance is not None else None
                            ),
                        )
                        pred_noise_b = model(
                            x_t=debug_x_t,
                            t=debug_t_scaled,
                            cover_img=debug_cover,
                            wm_bits=debug_wm_b,
                            content_mask=(
                                region_allowance[:debug_count]
                                if region_allowance is not None else None
                            ),
                        )
                        pred_x0_a = predict_start_from_noise(
                            diffusion, debug_x_t, debug_t, pred_noise_a
                        )
                        pred_x0_b = predict_start_from_noise(
                            diffusion, debug_x_t, debug_t, pred_noise_b
                        )
                        debug_allowance = (
                            region_allowance[:debug_count]
                            if region_allowance is not None else None
                        )
                        pred_x0_a = constrain_watermark_residual(
                            pred_x0_a.clamp(-1, 1),
                            debug_cover,
                            debug_allowance,
                            residual_constraint_cfg,
                        )
                        pred_x0_b = constrain_watermark_residual(
                            pred_x0_b.clamp(-1, 1),
                            debug_cover,
                            debug_allowance,
                            residual_constraint_cfg,
                        )
                        pred_x0_delta = (pred_x0_a - pred_x0_b).abs().mean().item()
                        logits_a = decoder(pred_x0_a.clamp(-1, 1))
                        logits_b = decoder(pred_x0_b.clamp(-1, 1))
                        bit_flip_logit_delta = (logits_a - logits_b).abs().mean().item()
                        log_data['bit_flip_image_delta'] = pred_x0_delta
                        log_data['bit_flip_logit_delta'] = bit_flip_logit_delta

                train_writer.writerow(log_data)
                train_csv.flush()

                train_log_lines = [
                    f"[TRAIN E{epoch:03d} S{global_step:06d}]"
                    f"{' [DEBUG]' if is_debug_log else ''}",
                    f"  Loss     total={loss_total.item():.4f}  "
                    f"diff={loss_diff.item():.4f}  image={loss_img.item():.4f}",
                    f"           wm_clean={loss_wm_clean.item():.4f}  "
                    f"wm_degraded={loss_wm_degraded_text}",
                    f"  Visual   delta={loss_delta.item():.4f}  "
                    f"tv={loss_tv.item():.4f}  topk={loss_topk.item():.4f}  "
                    f"channel={loss_channel.item():.4f}",
                    f"           region_enrichment="
                    f"{loss_region_enrichment.item():.4f}  "
                    f"region_outside={loss_region_outside.item():.4f}  "
                    f"region_total={loss_region_total.item():.4f}",
                    f"           spectral={loss_spectral_total.item():.4f}  "
                    f"peak={loss_spectral_peak.item():.4f}  "
                    f"anisotropy={loss_spectral_anisotropy.item():.4f}",
                    f"           flat_energy={flat_energy_ratio:.3f}  "
                    f"texture_energy={texture_energy_ratio:.3f}",
                    f"           mask_area={structure_metrics_log.get('mask_area_ratio', float('nan')):.3f}  "
                    f"direction={structure_metrics_log.get('directional_ratio', float('nan')):.3f}  "
                    f"fft_peak={structure_metrics_log.get('fft_peak_ratio', float('nan')):.3f}  "
                    f"cross_corr={structure_metrics_log.get('cross_image_correlation', float('nan')):.3f}",
                    f"  Accuracy clean={bit_acc_clean:.3f}  "
                    f"degraded={bit_acc_degraded_text}",
                    f"  Quality  pred_x0_psnr={psnr_val:.1f}dB  "
                    f"ssim={ssim_val:.4f}",
                    f"  Decoder  logits_std={logits_std:.4f}  "
                    f"sigmoid_mean={sigmoid_mean:.4f}",
                    f"  Noise    type={active_noise_type}  "
                    f"probability={curriculum['apply_prob']:.2f}  "
                    f"strength={curriculum['strength']:.2f}",
                    f"  Train    grad_norm={total_grad_norm:.3f}  "
                    f"amp_scale={amp_scale:g}",
                ]
                if is_debug_log:
                    train_log_lines.extend([
                        f"  Debug    image_delta={pred_x0_delta:.6f}  "
                        f"logit_delta={bit_flip_logit_delta:.6f}",
                        f"           grad(model={model_gn:.3f}  "
                        f"decoder={decoder_gn:.3f}  wm={wm_mlp_gn:.3f}  "
                        f"map={wm_map_mlp_gn:.3f})",
                    ])
                train_log_lines.extend([
                    f"  Weight   diff={lambda_diff:.2f}  image={lambda_img:.2f}  "
                    f"wm_clean={curriculum['lambda_wm_clean']:.2f}  "
                    f"wm_degraded={lambda_wm_degraded_active:.2f}",
                    f"           delta={lambda_delta:.2f}  tv={lambda_tv:.2f}  "
                    f"topk={lambda_topk:.2f}  channel={lambda_channel:.2f}  "
                    f"region_enrichment={lambda_region:.3f}  "
                    f"region_outside={lambda_region_outside:.3f}",
                ])
                print('\n'.join(train_log_lines), end='\n\n')
                if psnr_val > 45.0 and bit_acc < 0.6:
                    print(
                        f"[WARNING E{epoch:03d} S{global_step:06d}]\n"
                        f"  Metric   pred_x0_psnr={psnr_val:.1f}dB  "
                        f"bit_acc={bit_acc:.3f}\n"
                        "  Message  High PSNR but watermark is not learning; "
                        "the model may be collapsing to near-identity images.\n"
                    )

            # ========================================================
            # 7. Periodic sampling
            # ========================================================
            if global_step % sample_interval == 0 and global_step > 0:
                model.eval()
                with torch.no_grad():
                    # Take a batch for sampling
                    sample_batch = next(iter(train_loader))
                    s_cover = sample_batch['image'][:4].to(device)
                    s_wm = generate_val_watermark(
                        s_cover.size(0),
                        watermark_length,
                        watermark_seed,
                        device,
                        offset=global_step,
                    )

                    # Generate watermarked images via full reverse sampling
                    s_watermarked = embed_watermark(
                        diffusion, model, s_cover, s_wm,
                        t_start=train_t_start,
                        region_guidance_config=region_guidance_cfg,
                        residual_constraint_config=residual_constraint_cfg,
                    )

                    # Convert to [0, 1] for saving
                    s_cover_01 = (s_cover + 1.0) / 2.0
                    s_wm_01 = (s_watermarked + 1.0) / 2.0

                    # Randomly sample one attack from the active curriculum
                    # distribution and evaluate it at the active strength.
                    s_degraded_01, sample_noise_type = (
                        apply_degradation_with_strength(
                            s_wm_01,
                            noise_layer,
                            noise_type,
                            curriculum['strength'],
                            candidates=curriculum['candidates'],
                            probs=curriculum['probs'],
                        )
                    )
                    s_decoder_input = s_degraded_01.mul(2.0).sub(1.0)
                    s_logits = decoder(s_decoder_input)
                    s_bits = (torch.sigmoid(s_logits) > 0.5).float()
                    s_acc = (s_bits == s_wm).float().mean().item()
                    s_sigmoid = torch.sigmoid(s_logits)
                    s_logits_std = s_logits.detach().std().item()
                    s_sigmoid_mean = s_sigmoid.detach().mean().item()

                    s_delta_wm = (s_wm_01 - s_cover_01).abs()
                    s_delta_deg = (s_degraded_01 - s_cover_01).abs()
                    # Signed residual visualization: zero maps to neutral gray,
                    # while RGB direction and magnitude remain interpretable.
                    s_residual_signed_x5 = (
                        (s_wm_01 - s_cover_01) * 5.0 + 0.5
                    ).clamp(0.0, 1.0)
                    s_psnr_wm = compute_psnr(s_wm_01, s_cover_01, max_val=1.0)
                    s_psnr_deg = compute_psnr(s_degraded_01, s_cover_01, max_val=1.0)
                    s_ssim_wm = compute_ssim(
                        s_wm_01, s_cover_01, max_val=1.0
                    )
                    s_ssim_deg = compute_ssim(
                        s_degraded_01, s_cover_01, max_val=1.0
                    )
                    s_mae_wm = s_delta_wm.mean().item()
                    s_mae_deg = s_delta_deg.mean().item()
                    s_max_delta_wm = s_delta_wm.max().item()
                    s_max_delta_deg = s_delta_deg.max().item()
                    s_topk_delta_wm = residual_topk_loss(
                        s_wm_01 - s_cover_01,
                        cfg['train'].get('topk_delta_fraction', 0.01),
                    ).item()
                    s_tv_wm = residual_tv_loss(s_wm_01 - s_cover_01).item()
                    s_channel_delta = s_delta_wm.mean(dim=(0, 2, 3))
                    s_channel_delta_std = s_channel_delta.std(unbiased=False).item()
                    if region_guidance_enabled:
                        s_region_allowance, _ = build_edge_texture_guidance(
                            s_cover_01,
                            region_guidance_cfg,
                        )
                        s_flat_ratio, s_texture_ratio = (
                            residual_region_energy_ratios(
                                s_wm_01 - s_cover_01,
                                s_region_allowance,
                            )
                        )
                        s_flat_ratio = s_flat_ratio.item()
                        s_texture_ratio = s_texture_ratio.item()
                    else:
                        s_region_allowance = None
                        s_flat_ratio = float('nan')
                        s_texture_ratio = float('nan')
                    s_structure = residual_structure_metrics(
                        s_wm_01 - s_cover_01,
                        s_region_allowance,
                    )
                    s_structure = {
                        key: value.item() for key, value in s_structure.items()
                    }

                    sample_writer.writerow({
                        'epoch': epoch,
                        'global_step': global_step,
                        'noise_layer_type': sample_noise_type,
                        'noise_strength': curriculum['strength'],
                        'bit_acc': s_acc,
                        'psnr_watermarked': s_psnr_wm,
                        'psnr_degraded': s_psnr_deg,
                        'ssim_watermarked': s_ssim_wm,
                        'ssim_degraded': s_ssim_deg,
                        'mae_watermarked': s_mae_wm,
                        'mae_degraded': s_mae_deg,
                        'max_abs_delta_watermarked': s_max_delta_wm,
                        'max_abs_delta_degraded': s_max_delta_deg,
                        'topk_abs_delta_watermarked': s_topk_delta_wm,
                        'tv_watermarked': s_tv_wm,
                        'channel_delta_r': s_channel_delta[0].item(),
                        'channel_delta_g': s_channel_delta[1].item(),
                        'channel_delta_b': s_channel_delta[2].item(),
                        'channel_delta_std': s_channel_delta_std,
                        'mask_area_ratio': s_structure.get(
                            'mask_area_ratio', float('nan')
                        ),
                        'inside_energy_ratio': s_structure.get(
                            'inside_energy_ratio', float('nan')
                        ),
                        'outside_energy_ratio': s_structure.get(
                            'outside_energy_ratio', float('nan')
                        ),
                        'flat_energy_ratio': s_flat_ratio,
                        'texture_energy_ratio': s_texture_ratio,
                        'residual_dx_energy': s_structure['dx_energy'],
                        'residual_dy_energy': s_structure['dy_energy'],
                        'directional_ratio': s_structure['directional_ratio'],
                        'fft_peak_ratio': s_structure['fft_peak_ratio'],
                        'fft_midband_ratio': s_structure['fft_midband_ratio'],
                        'cross_image_correlation': s_structure[
                            'cross_image_correlation'
                        ],
                        'logits_std': s_logits_std,
                        'sigmoid_mean': s_sigmoid_mean,
                    })
                    sample_csv.flush()

                    # Save comparison grid
                    comparison = torch.cat(
                        [
                            s_cover_01,
                            s_wm_01,
                            s_degraded_01,
                            s_residual_signed_x5,
                        ],
                        dim=0,
                    )
                    save_path = os.path.join(
                        sample_dir,
                        f'step_{global_step:06d}_{sample_noise_type}'
                        f'_acc_{s_acc:.3f}_psnr_{s_psnr_wm:.2f}.png',
                    )
                    save_image(comparison, save_path, nrow=4)
                    sample_log_lines = [
                        f"[SAMPLE E{epoch:03d} S{global_step:06d}]",
                        f"  Noise    type={sample_noise_type}  "
                        f"strength={curriculum['strength']:.2f}",
                        f"  Accuracy degraded={s_acc:.3f}",
                        f"  Quality  watermarked_psnr={s_psnr_wm:.2f}dB  "
                        f"watermarked_ssim={s_ssim_wm:.4f}",
                        f"           degraded_psnr={s_psnr_deg:.2f}dB  "
                        f"degraded_ssim={s_ssim_deg:.4f}",
                        f"  Residual wm_mae={s_mae_wm:.4f}  "
                        f"wm_max={s_max_delta_wm:.4f}  "
                        f"wm_topk={s_topk_delta_wm:.4f}  wm_tv={s_tv_wm:.4f}",
                        f"           channel=(R={s_channel_delta[0].item():.4f}  "
                        f"G={s_channel_delta[1].item():.4f}  "
                        f"B={s_channel_delta[2].item():.4f})",
                        f"           flat_energy={s_flat_ratio:.3f}  "
                        f"texture_energy={s_texture_ratio:.3f}",
                        f"           mask_area={s_structure.get('mask_area_ratio', float('nan')):.3f}  "
                        f"direction={s_structure['directional_ratio']:.3f}  "
                        f"fft_peak={s_structure['fft_peak_ratio']:.3f}  "
                        f"cross_corr={s_structure['cross_image_correlation']:.3f}",
                        f"  Decoder  logits_std={s_logits_std:.4f}  "
                        f"sigmoid_mean={s_sigmoid_mean:.4f}",
                        f"  File     {save_path}",
                    ]
                    print('\n'.join(sample_log_lines), end='\n\n')

                    # Also save individual images
                    for i in range(min(4, s_cover.size(0))):
                        save_image(s_cover_01[i], os.path.join(
                            sample_dir, f'step_{global_step:06d}_cover_{i}.png'))
                        save_image(s_wm_01[i], os.path.join(
                            sample_dir, f'step_{global_step:06d}_watermarked_{i}.png'))
                        save_image(s_degraded_01[i], os.path.join(
                            sample_dir,
                            f'step_{global_step:06d}_degraded_{sample_noise_type}_{i}.png'))
                        save_image(
                            s_residual_signed_x5[i],
                            os.path.join(
                                sample_dir,
                                f'step_{global_step:06d}_signed_residual_x5_{i}.png',
                            ),
                        )
                        cover_residual_pair = torch.cat(
                            [s_cover_01[i], s_residual_signed_x5[i]],
                            dim=2,
                        )
                        save_image(
                            cover_residual_pair,
                            os.path.join(
                                sample_dir,
                                f'step_{global_step:06d}_cover_residual_x5_{i}.png',
                            ),
                        )
                        if s_region_allowance is not None:
                            save_image(
                                s_region_allowance[i],
                                os.path.join(
                                    sample_dir,
                                    f'step_{global_step:06d}_guidance_{i}.png',
                                ),
                            )

                model.train()
                set_encoder_training_mode(
                    model,
                    encoder_train_mode,
                    partial_output_blocks,
                    freeze_watermark_map_mlp,
                )

            global_step += 1

        # ============================================================
        # End of epoch: Validation
        # ============================================================
        model.eval()
        decoder.eval()
        noise_layer.eval()
        validation_step = max(global_step - 1, 0)
        validation_loss_weights = get_loss_weights(cfg, validation_step)
        validation_curriculum = get_noise_curriculum_state(
            cfg,
            validation_step,
            validation_loss_weights['lambda_wm'],
            use_noise_layer,
        )
        validation_phase = get_noise_curriculum_phase(
            cfg, validation_step, use_noise_layer
        )
        degradation_stage = get_degradation_stage(
            cfg, validation_step, use_noise_layer
        )
        if validation_phase != validation_curriculum['phase']:
            raise RuntimeError("Inconsistent curriculum phase calculation")
        validation_strength = (
            validation_curriculum['strength']
            if sync_validation_strength
            else 1.0
        )

        if not use_noise_layer:
            active_validation_candidates = []
            validation_eval_names = []
        elif noise_type == 'mixed':
            active_validation_candidates = (
                list(validation_curriculum['candidates'])
                if validation_curriculum['candidates'] is not None
                else list(noise_layer.names)
            )
            validation_eval_names = (
                active_validation_candidates
                if evaluate_validation_per_type
                else ['mixed']
            )
        else:
            active_validation_candidates = [noise_type]
            validation_eval_names = [noise_type]

        per_type_correct = {name: 0.0 for name in validation_eval_names}
        per_type_bit_count = {name: 0 for name in validation_eval_names}
        per_type_loss_sum = {name: 0.0 for name in validation_eval_names}
        per_type_ssim_sum = {name: 0.0 for name in validation_eval_names}
        per_type_image_count = {name: 0 for name in validation_eval_names}
        run_fixed_validation = bool(
            use_noise_layer
            and fixed_validation_strengths
            and epoch % fixed_validation_interval == 0
        )
        fixed_validation_keys = [
            (strength, candidate)
            for strength in fixed_validation_strengths
            for candidate in fixed_validation_candidates
        ] if run_fixed_validation else []
        fixed_correct = {key: 0.0 for key in fixed_validation_keys}
        fixed_bit_count = {key: 0 for key in fixed_validation_keys}
        fixed_loss_sum = {key: 0.0 for key in fixed_validation_keys}
        fixed_ssim_sum = {key: 0.0 for key in fixed_validation_keys}
        fixed_image_count = {key: 0 for key in fixed_validation_keys}
        fixed_batch_count = {key: 0 for key in fixed_validation_keys}
        val_clean_correct = 0.0
        val_clean_bit_count = 0
        val_clean_loss_sum = 0.0
        val_squared_error_sum = 0.0
        val_ssim_watermarked_sum = 0.0
        val_abs_error_sum = 0.0
        val_pixel_count = 0
        val_topk_sum = 0.0
        val_tv_sum = 0.0
        val_flat_energy_ratio_sum = 0.0
        val_texture_energy_ratio_sum = 0.0
        val_mask_area_ratio_sum = 0.0
        val_inside_energy_ratio_sum = 0.0
        val_outside_energy_ratio_sum = 0.0
        val_dx_energy_sum = 0.0
        val_dy_energy_sum = 0.0
        val_directional_ratio_sum = 0.0
        val_fft_peak_ratio_sum = 0.0
        val_fft_midband_ratio_sum = 0.0
        val_cross_image_correlation_sum = 0.0
        val_image_count = 0
        val_channel_abs_sum = torch.zeros(3, device=device, dtype=torch.float64)
        val_channel_pixel_count = 0

        eval_rng_state = capture_eval_rng_state()
        set_random_seed(validation_seed, deterministic=deterministic)
        try:
            with torch.no_grad():
                for v_batch_index, v_batch in enumerate(val_loader):
                    v_cover = v_batch['image'].to(device)
                    B_v = v_cover.size(0)
                    # Validation dataset bits are deterministic per image and
                    # no longer vary with global_step.
                    v_wm = v_batch['wm_bits'].to(device).float()
                    v_cover_01 = (v_cover + 1.0) / 2.0
                    if needs_content_guidance:
                        v_region_allowance, _ = build_edge_texture_guidance(
                            v_cover_01,
                            region_guidance_cfg,
                        )
                    else:
                        v_region_allowance = None

                    # ---- Single-step pred_x0 validation ----
                    embedding_rng_state = capture_eval_rng_state()
                    try:
                        set_random_seed(
                            validation_seed + 50000 + v_batch_index,
                            deterministic=deterministic,
                        )
                        t_eval = torch.randint(
                            wm_t_min, wm_t_max, (B_v,), device=device
                        ).long()
                        noise_eval = torch.randn_like(v_cover)
                    finally:
                        restore_eval_rng_state(embedding_rng_state)
                    x_t_eval = diffusion.q_sample(
                        v_cover, t_eval, noise=noise_eval
                    )

                    t_eval_scaled = t_eval.float() * (1000.0 / timesteps)
                    pred_noise_eval = model(
                        x_t=x_t_eval,
                        t=t_eval_scaled,
                        cover_img=v_cover,
                        wm_bits=v_wm,
                        content_mask=v_region_allowance,
                    )

                    v_watermarked_raw = predict_start_from_noise(
                        diffusion, x_t_eval, t_eval, pred_noise_eval
                    ).clamp(-1, 1)
                    v_watermarked = constrain_watermark_residual(
                        v_watermarked_raw,
                        v_cover,
                        v_region_allowance,
                        residual_constraint_cfg,
                    )
                    v_wm_01 = (v_watermarked + 1.0) / 2.0

                    v_logits_clean = decoder(v_watermarked)
                    v_bits_clean = (
                        torch.sigmoid(v_logits_clean) > 0.5
                    ).float()
                    val_clean_correct += (
                        v_bits_clean == v_wm
                    ).float().sum().item()
                    val_clean_bit_count += v_wm.numel()
                    val_clean_loss_sum += F.binary_cross_entropy_with_logits(
                        v_logits_clean,
                        v_wm,
                        reduction='sum',
                    ).item()

                    residual_01 = v_wm_01 - v_cover_01
                    val_squared_error_sum += residual_01.square().sum().item()
                    val_ssim_watermarked_sum += (
                        compute_ssim(v_wm_01, v_cover_01, max_val=1.0) * B_v
                    )
                    val_abs_error_sum += residual_01.abs().sum().item()
                    val_pixel_count += residual_01.numel()
                    val_topk_sum += residual_topk_loss(
                        residual_01,
                        cfg['train'].get('topk_delta_fraction', 0.01),
                    ).item() * B_v
                    val_tv_sum += residual_tv_loss(residual_01).item() * B_v
                    if region_guidance_enabled:
                        v_flat_ratio, v_texture_ratio = (
                            residual_region_energy_ratios(
                                residual_01,
                                v_region_allowance,
                            )
                        )
                        val_flat_energy_ratio_sum += v_flat_ratio.item() * B_v
                        val_texture_energy_ratio_sum += (
                            v_texture_ratio.item() * B_v
                        )
                    v_structure = residual_structure_metrics(
                        residual_01,
                        v_region_allowance,
                    )
                    val_dx_energy_sum += v_structure['dx_energy'].item() * B_v
                    val_dy_energy_sum += v_structure['dy_energy'].item() * B_v
                    val_directional_ratio_sum += (
                        v_structure['directional_ratio'].item() * B_v
                    )
                    val_fft_peak_ratio_sum += (
                        v_structure['fft_peak_ratio'].item() * B_v
                    )
                    val_fft_midband_ratio_sum += (
                        v_structure['fft_midband_ratio'].item() * B_v
                    )
                    val_cross_image_correlation_sum += (
                        v_structure['cross_image_correlation'].item() * B_v
                    )
                    if v_region_allowance is not None:
                        val_mask_area_ratio_sum += (
                            v_structure['mask_area_ratio'].item() * B_v
                        )
                        val_inside_energy_ratio_sum += (
                            v_structure['inside_energy_ratio'].item() * B_v
                        )
                        val_outside_energy_ratio_sum += (
                            v_structure['outside_energy_ratio'].item() * B_v
                        )
                    val_image_count += B_v
                    val_channel_abs_sum += residual_01.abs().sum(
                        dim=(0, 2, 3), dtype=torch.float64
                    )
                    val_channel_pixel_count += (
                        B_v * residual_01.size(2) * residual_01.size(3)
                    )

                    for eval_name in validation_eval_names:
                        if noise_type == 'mixed':
                            if eval_name == 'mixed':
                                eval_candidates = validation_curriculum['candidates']
                                eval_probs = validation_curriculum['probs']
                            else:
                                eval_candidates = [eval_name]
                                eval_probs = [1.0]
                        else:
                            eval_candidates = None
                            eval_probs = None

                        with torch.amp.autocast(
                            device_type=device.type, enabled=False
                        ):
                            v_degraded_01, _ = apply_degradation_with_strength(
                                v_wm_01,
                                noise_layer,
                                noise_type,
                                validation_strength,
                                candidates=eval_candidates,
                                probs=eval_probs,
                            )
                        v_logits_deg = decoder(
                            v_degraded_01.mul(2.0).sub(1.0)
                        )
                        v_bits_deg = (
                            torch.sigmoid(v_logits_deg) > 0.5
                        ).float()
                        per_type_correct[eval_name] += (
                            v_bits_deg == v_wm
                        ).float().sum().item()
                        per_type_bit_count[eval_name] += v_wm.numel()
                        per_type_loss_sum[eval_name] += (
                            F.binary_cross_entropy_with_logits(
                                v_logits_deg,
                                v_wm,
                                reduction='sum',
                            ).item()
                        )
                        per_type_ssim_sum[eval_name] += (
                            compute_ssim(
                                v_degraded_01,
                                v_cover_01,
                                max_val=1.0,
                            )
                            * B_v
                        )
                        per_type_image_count[eval_name] += B_v

                    within_fixed_batch_limit = (
                        fixed_validation_max_batches == 0
                        or v_batch_index < fixed_validation_max_batches
                    )
                    if run_fixed_validation and within_fixed_batch_limit:
                        fixed_rng_state = capture_eval_rng_state()
                        try:
                            for fixed_cell_index, (
                                fixed_strength,
                                fixed_name,
                            ) in enumerate(fixed_validation_keys):
                                # Each matrix cell receives an epoch-independent
                                # RNG stream. Curriculum branch count and order
                                # therefore cannot move the fixed benchmark.
                                fixed_seed = (
                                    validation_seed
                                    + 100000
                                    + v_batch_index * len(fixed_validation_keys)
                                    + fixed_cell_index
                                )
                                set_random_seed(
                                    fixed_seed,
                                    deterministic=deterministic,
                                )
                                with torch.amp.autocast(
                                    device_type=device.type, enabled=False
                                ):
                                    fixed_degraded_01, _ = (
                                        apply_degradation_with_strength(
                                            v_wm_01,
                                            noise_layer,
                                            noise_type,
                                            fixed_strength,
                                            candidates=[fixed_name]
                                            if noise_type == 'mixed' else None,
                                            probs=[1.0]
                                            if noise_type == 'mixed' else None,
                                        )
                                    )
                                fixed_logits = decoder(
                                    fixed_degraded_01.mul(2.0).sub(1.0)
                                )
                                fixed_bits = (
                                    torch.sigmoid(fixed_logits) > 0.5
                                ).float()
                                fixed_key = (fixed_strength, fixed_name)
                                fixed_correct[fixed_key] += (
                                    fixed_bits == v_wm
                                ).float().sum().item()
                                fixed_bit_count[fixed_key] += v_wm.numel()
                                fixed_loss_sum[fixed_key] += (
                                    F.binary_cross_entropy_with_logits(
                                        fixed_logits,
                                        v_wm,
                                        reduction='sum',
                                    ).item()
                                )
                                fixed_ssim_sum[fixed_key] += (
                                    compute_ssim(
                                        fixed_degraded_01,
                                        v_cover_01,
                                        max_val=1.0,
                                    )
                                    * B_v
                                )
                                fixed_image_count[fixed_key] += B_v
                                fixed_batch_count[fixed_key] += 1
                        finally:
                            restore_eval_rng_state(fixed_rng_state)
        finally:
            restore_eval_rng_state(eval_rng_state)

        if val_clean_bit_count == 0 or val_pixel_count == 0:
            raise RuntimeError("Validation loader produced no samples")

        avg_acc_clean = val_clean_correct / val_clean_bit_count
        avg_loss_clean = val_clean_loss_sum / val_clean_bit_count
        val_mse = val_squared_error_sum / val_pixel_count
        avg_psnr = (
            100.0 if val_mse == 0.0 else -10.0 * math.log10(val_mse)
        )
        avg_ssim_watermarked = val_ssim_watermarked_sum / val_image_count
        avg_mae_watermarked = val_abs_error_sum / val_pixel_count
        avg_topk_watermarked = val_topk_sum / val_image_count
        avg_tv_watermarked = val_tv_sum / val_image_count
        if region_guidance_enabled:
            avg_flat_energy_ratio = (
                val_flat_energy_ratio_sum / val_image_count
            )
            avg_texture_energy_ratio = (
                val_texture_energy_ratio_sum / val_image_count
            )
            avg_mask_area_ratio = val_mask_area_ratio_sum / val_image_count
            avg_inside_energy_ratio = (
                val_inside_energy_ratio_sum / val_image_count
            )
            avg_outside_energy_ratio = (
                val_outside_energy_ratio_sum / val_image_count
            )
        else:
            avg_flat_energy_ratio = float('nan')
            avg_texture_energy_ratio = float('nan')
            avg_mask_area_ratio = float('nan')
            avg_inside_energy_ratio = float('nan')
            avg_outside_energy_ratio = float('nan')
        avg_dx_energy = val_dx_energy_sum / val_image_count
        avg_dy_energy = val_dy_energy_sum / val_image_count
        avg_directional_ratio = val_directional_ratio_sum / val_image_count
        avg_fft_peak_ratio = val_fft_peak_ratio_sum / val_image_count
        avg_fft_midband_ratio = val_fft_midband_ratio_sum / val_image_count
        avg_cross_image_correlation = (
            val_cross_image_correlation_sum / val_image_count
        )
        channel_energy = val_channel_abs_sum / val_channel_pixel_count
        avg_channel_delta_std = channel_energy.std(
            unbiased=False
        ).item()

        per_type_acc = {}
        per_type_loss = {}
        per_type_ssim = {}
        for eval_name in validation_eval_names:
            bit_count = per_type_bit_count[eval_name]
            if bit_count <= 0:
                raise RuntimeError(
                    f"Validation degradation {eval_name} produced no samples"
                )
            per_type_acc[eval_name] = (
                per_type_correct[eval_name] / bit_count
            )
            per_type_loss[eval_name] = (
                per_type_loss_sum[eval_name] / bit_count
            )
            image_count = per_type_image_count[eval_name]
            if image_count <= 0:
                raise RuntimeError(
                    f"Validation SSIM for {eval_name} produced no samples"
                )
            per_type_ssim[eval_name] = (
                per_type_ssim_sum[eval_name] / image_count
            )

        if per_type_acc:
            avg_acc_deg = sum(per_type_acc.values()) / len(per_type_acc)
            worst_acc_deg = min(per_type_acc.values())
            avg_loss_deg = sum(per_type_loss.values()) / len(per_type_loss)
            avg_ssim_degraded = sum(per_type_ssim.values()) / len(per_type_ssim)
        else:
            avg_acc_deg = avg_acc_clean
            worst_acc_deg = avg_acc_clean
            avg_loss_deg = avg_loss_clean
            avg_ssim_degraded = float('nan')

        fixed_matrix_acc = {}
        fixed_matrix_loss = {}
        fixed_matrix_ssim = {}
        if run_fixed_validation:
            for fixed_key in fixed_validation_keys:
                num_bits = fixed_bit_count[fixed_key]
                if num_bits <= 0:
                    fixed_strength, fixed_name = fixed_key
                    raise RuntimeError(
                        "Fixed validation matrix produced no samples for "
                        f"{fixed_name}@{fixed_strength:.2f}"
                    )
                fixed_matrix_acc[fixed_key] = fixed_correct[fixed_key] / num_bits
                fixed_matrix_loss[fixed_key] = (
                    fixed_loss_sum[fixed_key] / num_bits
                )
                num_images = fixed_image_count[fixed_key]
                if num_images <= 0:
                    raise RuntimeError(
                        "Fixed validation SSIM produced no samples for "
                        f"{fixed_name}@{fixed_strength:.2f}"
                    )
                fixed_matrix_ssim[fixed_key] = (
                    fixed_ssim_sum[fixed_key] / num_images
                )

        if fixed_matrix_for_checkpoint:
            score_eval_source = 'fixed_matrix'
            score_degraded_macro_acc = (
                sum(fixed_matrix_acc.values()) / len(fixed_matrix_acc)
            )
            score_degraded_worst_acc = min(fixed_matrix_acc.values())
            score_degraded_macro_loss = (
                sum(fixed_matrix_loss.values()) / len(fixed_matrix_loss)
            )
        else:
            score_eval_source = 'curriculum'
            score_degraded_macro_acc = avg_acc_deg
            score_degraded_worst_acc = worst_acc_deg
            score_degraded_macro_loss = avg_loss_deg

        balanced_score, score_components = compute_balanced_checkpoint_score(
            degraded_macro_acc=score_degraded_macro_acc,
            degraded_worst_acc=score_degraded_worst_acc,
            clean_acc=avg_acc_clean,
            degraded_macro_bce=score_degraded_macro_loss,
            psnr=avg_psnr,
            topk_delta=avg_topk_watermarked,
            tv_delta=avg_tv_watermarked,
            channel_delta_std=avg_channel_delta_std,
        )

        val_log_data = {
            'epoch': epoch,
            'global_step': global_step,
            'curriculum_phase': validation_phase,
            'degradation_stage': degradation_stage,
            'noise_strength': validation_strength,
            'active_candidates': ','.join(active_validation_candidates)
            if active_validation_candidates else 'clean',
            'bit_acc_clean': avg_acc_clean,
            'bit_acc_degraded': avg_acc_deg,
            'loss_wm_clean': avg_loss_clean,
            'loss_wm_degraded': avg_loss_deg,
            'degraded_macro_acc': avg_acc_deg,
            'degraded_worst_acc': worst_acc_deg,
            'degraded_macro_loss': avg_loss_deg,
            'score_eval_source': score_eval_source,
            'fixed_macro_acc': (
                sum(fixed_matrix_acc.values()) / len(fixed_matrix_acc)
                if fixed_matrix_acc else float('nan')
            ),
            'fixed_worst_acc': (
                min(fixed_matrix_acc.values())
                if fixed_matrix_acc else float('nan')
            ),
            'fixed_macro_loss': (
                sum(fixed_matrix_loss.values()) / len(fixed_matrix_loss)
                if fixed_matrix_loss else float('nan')
            ),
            'fixed_macro_ssim': (
                sum(fixed_matrix_ssim.values()) / len(fixed_matrix_ssim)
                if fixed_matrix_ssim else float('nan')
            ),
            'psnr': avg_psnr,
            'ssim_watermarked': avg_ssim_watermarked,
            'ssim_degraded': avg_ssim_degraded,
            'mae_watermarked': avg_mae_watermarked,
            'topk_watermarked': avg_topk_watermarked,
            'tv_watermarked': avg_tv_watermarked,
            'channel_delta_std': avg_channel_delta_std,
            'mask_area_ratio': avg_mask_area_ratio,
            'inside_energy_ratio': avg_inside_energy_ratio,
            'outside_energy_ratio': avg_outside_energy_ratio,
            'flat_energy_ratio': avg_flat_energy_ratio,
            'texture_energy_ratio': avg_texture_energy_ratio,
            'residual_dx_energy': avg_dx_energy,
            'residual_dy_energy': avg_dy_energy,
            'directional_ratio': avg_directional_ratio,
            'fft_peak_ratio': avg_fft_peak_ratio,
            'fft_midband_ratio': avg_fft_midband_ratio,
            'cross_image_correlation': avg_cross_image_correlation,
            'balanced_score': balanced_score,
        }
        for metric_name in validation_noise_names:
            val_log_data[f'bit_acc_{metric_name}'] = per_type_acc.get(
                metric_name, float('nan')
            )
            val_log_data[f'loss_{metric_name}'] = per_type_loss.get(
                metric_name, float('nan')
            )
            val_log_data[f'ssim_{metric_name}'] = per_type_ssim.get(
                metric_name, float('nan')
            )
        val_writer.writerow(val_log_data)
        val_csv.flush()

        if run_fixed_validation:
            for fixed_strength, fixed_name in fixed_validation_keys:
                fixed_key = (fixed_strength, fixed_name)
                num_bits = fixed_bit_count[fixed_key]
                fixed_acc = fixed_matrix_acc[fixed_key]
                fixed_loss = fixed_matrix_loss[fixed_key]
                fixed_val_writer.writerow({
                    'epoch': epoch,
                    'global_step': global_step,
                    'strength': fixed_strength,
                    'noise_type': fixed_name,
                    'bit_acc': fixed_acc,
                    'loss_wm': fixed_loss,
                    'ssim_end2end': fixed_matrix_ssim[fixed_key],
                    'num_bits': num_bits,
                    'num_batches': fixed_batch_count[fixed_key],
                })
            fixed_val_csv.flush()

        # ============================================================
        # Save checkpoint using either the curriculum view or the fixed matrix.
        # ============================================================
        best_metric_name = (
            'balanced_score_v1_fixed_matrix'
            if score_eval_source == 'fixed_matrix'
            else 'balanced_score_v1_curriculum'
        )
        current_best_metric = balanced_score
        checkpoint = {
            'diffusion_model': model.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler is not None else None,
            'epoch': epoch,
            'global_step': global_step,
            'config': cfg,
            'random_state': capture_random_state(train_generator),
            'bit_acc_clean': avg_acc_clean,
            'bit_acc_degraded': avg_acc_deg,
            'balanced_score': balanced_score,
            'score_components': score_components,
            'curriculum_phase': validation_phase,
            'degradation_stage': degradation_stage,
            'active_candidates': active_validation_candidates,
            'degraded_macro_acc': avg_acc_deg,
            'degraded_worst_acc': worst_acc_deg,
            'degraded_macro_loss': avg_loss_deg,
            'score_eval_source': score_eval_source,
            'score_degraded_macro_acc': score_degraded_macro_acc,
            'score_degraded_worst_acc': score_degraded_worst_acc,
            'score_degraded_macro_loss': score_degraded_macro_loss,
            'fixed_matrix': [
                {
                    'strength': strength,
                    'noise_type': name,
                    'bit_acc': fixed_matrix_acc[(strength, name)],
                    'loss_wm': fixed_matrix_loss[(strength, name)],
                    'ssim_end2end': fixed_matrix_ssim[(strength, name)],
                }
                for strength, name in fixed_validation_keys
            ],
            'per_type_acc': per_type_acc,
            'per_type_loss': per_type_loss,
            'per_type_ssim': per_type_ssim,
            'psnr': avg_psnr,
            'ssim_watermarked': avg_ssim_watermarked,
            'ssim_degraded': avg_ssim_degraded,
            'mae_watermarked': avg_mae_watermarked,
            'topk_delta': avg_topk_watermarked,
            'tv_delta': avg_tv_watermarked,
            'channel_delta_std': avg_channel_delta_std,
            'mask_area_ratio': avg_mask_area_ratio,
            'inside_energy_ratio': avg_inside_energy_ratio,
            'outside_energy_ratio': avg_outside_energy_ratio,
            'flat_energy_ratio': avg_flat_energy_ratio,
            'texture_energy_ratio': avg_texture_energy_ratio,
            'residual_dx_energy': avg_dx_energy,
            'residual_dy_energy': avg_dy_energy,
            'directional_ratio': avg_directional_ratio,
            'fft_peak_ratio': avg_fft_peak_ratio,
            'fft_midband_ratio': avg_fft_midband_ratio,
            'cross_image_correlation': avg_cross_image_correlation,
            'best_metric_name': best_metric_name,
            'best_metric_value': current_best_metric,
            # Retained for compatibility with older checkpoint readers.
            'best_bit_acc': score_degraded_macro_acc,
        }
        last_validation_metadata = {
            key: value
            for key, value in checkpoint.items()
            if key not in {
                'diffusion_model', 'decoder', 'optimizer', 'scaler',
                'config', 'random_state',
            }
        }

        # latest.pt follows save_interval.
        latest_path = os.path.join(checkpoint_dir, 'latest.pt')
        latest_saved = epoch % save_interval == 0
        if latest_saved:
            torch.save(checkpoint, latest_path)

        # Curriculum phases that share the same degradation candidate set also
        # share one best checkpoint. Strength-only changes do not create a new
        # best file. best.pt mirrors the active degradation-stage winner.
        degradation_stage_for_filename = max(1, degradation_stage)
        degradation_best_path = os.path.join(
            checkpoint_dir,
            f'best_degradation_stage{degradation_stage_for_filename}.pt',
        )
        best_path = os.path.join(checkpoint_dir, 'best.pt')
        previous_degradation_best = float('-inf')
        if os.path.exists(degradation_best_path):
            best_ckpt = torch.load(degradation_best_path, map_location='cpu')
            if (
                best_ckpt.get('best_metric_name') == best_metric_name
                and int(best_ckpt.get('degradation_stage', -1))
                == degradation_stage
            ):
                previous_degradation_best = float(
                    best_ckpt.get('best_metric_value', float('-inf'))
                )

        best_updated = current_best_metric > previous_degradation_best
        if best_updated:
            torch.save(checkpoint, degradation_best_path)
            torch.save(checkpoint, best_path)

        latest_status = 'saved' if latest_saved else 'not_due'
        best_status = 'updated' if best_updated else 'unchanged'
        candidate_acc_text = '  '.join(
            f"{name}={per_type_acc[name]:.3f}"
            for name in validation_eval_names
        )
        candidate_ssim_text = '  '.join(
            f"{name}={per_type_ssim[name]:.4f}"
            for name in validation_eval_names
        )
        validation_log_lines = [
            f"[VALIDATE E{epoch:03d} S{validation_step:06d}]",
            "  Mode       pred_x0 (single-step)",
            f"  Phase      curriculum={validation_phase}  "
            f"degradation={degradation_stage}",
            f"  Strength   {validation_strength:.2f}",
            f"  Candidates "
            f"{','.join(active_validation_candidates) if active_validation_candidates else 'clean'}",
            f"  Accuracy   clean={avg_acc_clean:.3f}"
            + (f"  {candidate_acc_text}" if candidate_acc_text else ""),
            f"             macro={avg_acc_deg:.3f}  worst={worst_acc_deg:.3f}",
            f"  Quality    pred_x0_psnr={avg_psnr:.1f}dB  "
            f"ssim={avg_ssim_watermarked:.4f}  "
            f"mae={avg_mae_watermarked:.4f}",
            f"             degraded_ssim="
            f"{avg_ssim_degraded:.4f}"
            if math.isfinite(avg_ssim_degraded)
            else "             degraded_ssim=n/a",
            f"             topk={avg_topk_watermarked:.4f}  "
            f"tv={avg_tv_watermarked:.4f}  "
            f"channel={avg_channel_delta_std:.4f}",
            f"             flat_energy={avg_flat_energy_ratio:.3f}  "
            f"texture_energy={avg_texture_energy_ratio:.3f}",
            f"             mask_area={avg_mask_area_ratio:.3f}  "
            f"direction={avg_directional_ratio:.3f}  "
            f"fft_peak={avg_fft_peak_ratio:.3f}  "
            f"cross_corr={avg_cross_image_correlation:.3f}",
            f"  Score      balanced={balanced_score:.6f}",
            f"  Checkpoint latest={latest_status}  "
            f"best_degradation_stage{degradation_stage_for_filename}="
            f"{best_status}  best={best_status}",
            f"  Monitor    metric={best_metric_name}  "
            f"current={current_best_metric:.6f}  "
            f"previous_degradation_best={previous_degradation_best:.6f}",
        ]
        if candidate_ssim_text:
            validation_log_lines.insert(
                7,
                f"  SSIM       {candidate_ssim_text}",
            )
        if fixed_matrix_acc:
            validation_log_lines.insert(
                -2,
                "  Fixed      "
                f"macro={sum(fixed_matrix_acc.values()) / len(fixed_matrix_acc):.3f}  "
                f"worst={min(fixed_matrix_acc.values()):.3f}  "
                f"ssim={sum(fixed_matrix_ssim.values()) / len(fixed_matrix_ssim):.4f}  "
                f"score_source={score_eval_source}",
            )
        if latest_saved:
            validation_log_lines.append(f"  File       latest={latest_path}")
        if best_updated:
            validation_log_lines.extend([
                f"             degradation_best={degradation_best_path}",
                f"             best={best_path}",
            ])
        print('\n'.join(validation_log_lines), end='\n\n')

        # ============================================================
        # DIAGNOSTIC: If bit_acc stays near 0.5, print warning
        # ============================================================
        if avg_acc_clean < 0.55 and epoch > 10:
            current_weights = get_loss_weights(cfg, global_step)
            print(
                f"[DIAGNOSTIC E{epoch:03d} S{validation_step:06d}]\n"
                "  Message  bit_acc_clean is near 0.5. Possible causes:\n"
                "  1. wm_bits not actually fed to U-Net? Check watermark_mlp.\n"
                "  2. watermark_mlp requires_grad=True? Check parameters.\n"
                "  3. loss_wm backprop to diffusion_model? Check no .detach().\n"
                "  4. decoder input range correct [-1, 1]?\n"
                "  5. lambda_wm too small? Current: {:.1f}\n"
                "  6. wm_t_max too large? Current: {}\n".format(
                    current_weights['lambda_wm'], wm_t_max
                )
            )

    # --- End training ---
    train_csv.close()
    val_csv.close()
    fixed_val_csv.close()
    sample_csv.close()

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, 'final.pt')
    final_checkpoint = {
        'diffusion_model': model.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epochs,
        'global_step': global_step,
        'config': cfg,
        'random_state': capture_random_state(train_generator),
    }
    if last_validation_metadata is not None:
        final_checkpoint.update(last_validation_metadata)
    torch.save(final_checkpoint, final_path)
    final_step = max(global_step - 1, 0)
    print(
        f"[COMPLETE E{epoch:03d} S{final_step:06d}]\n"
        f"  Logs       {log_dir}\n"
        f"  Checkpoint {final_path}\n"
    )

# ============================================================
# CLI entry point
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Watermark-Conditioned Diffusion')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to continue the same training stage')
    parser.add_argument('--init_from', type=str, default=None,
                        help='Path to checkpoint used to initialize a new training stage')
    args = parser.parse_args()

    if args.resume and args.init_from:
        parser.error(
            "[Error] --resume and --init_from cannot be used at the same time.\n"
            "Use --resume for continuing the same training stage.\n"
            "Use --init_from for initializing a new stage from a previous checkpoint."
        )

    config = load_config(args.config)
    try:
        validate_initialization_policy(
            config,
            resume_path=args.resume,
            init_from_path=args.init_from,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.resume:
        config['_resume_path'] = args.resume
    if args.init_from:
        config['_init_from_path'] = args.init_from
    train(config)


