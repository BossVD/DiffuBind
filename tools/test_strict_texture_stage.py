"""Focused tests for the two-stage strict-texture training workflow."""

import math
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.watermark_residual import (
    build_edge_texture_guidance,
    constrain_watermark_residual,
    get_residual_constraint_settings,
)
from train_watermark_diffusion import (
    compute_ssim,
    get_active_localization_targets,
    get_loss_weights,
    get_residual_spectral_settings,
    get_stage_training_setting,
    load_config,
    residual_energy_ratio_loss,
    residual_spectral_regularization_loss,
    residual_structure_metrics,
    validate_initialization_policy,
)


STRICT_GUIDANCE = {
    "mode": "strict_multiscale",
    "edge_scales": [3, 5, 9],
    "texture_scales": [3, 5, 9],
    "edge_weight": 0.4,
    "texture_weight": 0.6,
    "target_mask_area": 0.35,
    "mask_temperature": 0.08,
    "dilation_kernel": 1,
    "blur_kernel": 3,
}


def test_strict_mask_is_finite_and_content_dependent():
    torch.manual_seed(7)
    cover = torch.rand(3, 3, 64, 64)
    mask, penalty = build_edge_texture_guidance(cover, STRICT_GUIDANCE)
    assert mask.shape == (3, 1, 64, 64)
    assert torch.isfinite(mask).all() and torch.isfinite(penalty).all()
    assert 0.0 <= mask.min() <= mask.max() <= 1.0
    assert 0.15 <= mask.mean().item() <= 0.65
    assert not torch.allclose(mask[0], mask[1])


def test_ssim_metric_identity_and_perturbation():
    torch.manual_seed(11)
    reference = torch.rand(2, 3, 32, 32)
    identical_ssim = compute_ssim(reference, reference)
    perturbed = (reference + 0.10 * torch.randn_like(reference)).clamp(0.0, 1.0)
    perturbed_ssim = compute_ssim(perturbed, reference)
    assert identical_ssim > 0.9999
    assert math.isfinite(perturbed_ssim)
    assert 0.0 <= perturbed_ssim < identical_ssim


def test_spatial_budget_bounds_and_gradient():
    cover = torch.zeros(1, 3, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, :, 8:] = 1.0
    raw = torch.ones_like(cover, requires_grad=True)
    config = {
        "enabled": True,
        "flat_max_abs_delta_01": 0.005,
        "texture_max_abs_delta_01": 0.045,
        "mask_power": 2.0,
    }
    settings = get_residual_constraint_settings(config)
    assert settings["mode"] == "spatial_budget"
    constrained = constrain_watermark_residual(raw, cover, mask, config)
    delta = ((constrained - cover) / 2.0).abs()
    assert delta[:, :, :, :8].max().item() <= 0.005001
    assert delta[:, :, :, 8:].max().item() <= 0.045001
    assert delta[:, :, :, 8:].mean() > delta[:, :, :, :8].mean()
    constrained.mean().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()


def test_energy_ratio_targets_and_gradient():
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, :, :4] = 1.0
    residual = torch.full((1, 3, 8, 8), 0.1, requires_grad=True)
    loss = residual_energy_ratio_loss(
        residual,
        mask,
        target_inside_ratio=0.85,
        max_outside_ratio=0.15,
        min_active_area=0.08,
    )
    assert loss.item() > 0.0
    loss.backward()
    assert residual.grad is not None and torch.isfinite(residual.grad).all()

    start = get_active_localization_targets(
        {
            "start_inside_ratio": 0.65,
            "target_inside_ratio": 0.85,
            "start_max_outside_ratio": 0.35,
            "max_outside_ratio": 0.15,
            "ratio_warmup_steps": 5000,
        },
        global_step=0,
    )
    end = get_active_localization_targets(
        {
            "start_inside_ratio": 0.65,
            "target_inside_ratio": 0.85,
            "start_max_outside_ratio": 0.35,
            "max_outside_ratio": 0.15,
            "ratio_warmup_steps": 5000,
        },
        global_step=6000,
    )
    assert start["inside"] < end["inside"]
    assert start["outside"] > end["outside"]


def test_horizontal_stripe_diagnostics_and_loss():
    y = torch.arange(64, dtype=torch.float32).view(1, 1, 64, 1)
    stripe = torch.sin(2.0 * torch.pi * y / 8.0).expand(4, 3, 64, 64)
    metrics = residual_structure_metrics(stripe)
    assert metrics["directional_ratio"].item() > 2.0
    assert torch.isfinite(metrics["fft_peak_ratio"])
    settings = get_residual_spectral_settings({
        "enabled": True,
        "start_step": 0,
        "warmup_steps": 0,
        "lambda_peak": 0.02,
        "lambda_anisotropy": 0.01,
        "max_peak_ratio": 0.20,
        "max_directional_ratio": 2.0,
    })
    trainable_stripe = stripe.clone().requires_grad_()
    total, _, anisotropy, scale = residual_spectral_regularization_loss(
        trainable_stripe, settings, global_step=0
    )
    assert scale == 1.0 and anisotropy.item() > 0.0 and total.item() > 0.0
    total.backward()
    assert trainable_stripe.grad is not None
    assert torch.isfinite(trainable_stripe.grad).all()


def test_two_stage_configs_and_initialization_guards():
    stage1_path = REPO_ROOT / "configs" / "watermark_stage1.yaml"
    stage2_path = REPO_ROOT / "configs" / "watermark_stage2_mixed_strict_texture_v1.yaml"
    stage1 = load_config(stage1_path)
    stage2 = load_config(stage2_path)
    assert stage1["train"]["stage"] == "full"
    assert set(stage1["train"]["stages"]) == {"warmup", "balance", "full"}
    assert stage1["noise_layer"]["type"] == "none"
    assert stage1["train"]["encoder_train_mode"] == "full"
    assert stage2["train"]["stage"] == "stage2_strict_texture"
    assert stage2["noise_layer"]["type"] == "mixed"
    assert stage1["data"]["watermark_length"] == 30
    assert stage2["data"]["watermark_length"] == 30
    assert "30bit" in stage1["output"]["checkpoint_dir"]
    assert "30bit" in stage2["output"]["checkpoint_dir"]
    assert stage2["initialization"]["expected_init_from"].endswith(
        "checkpoints_stage1_strict_texture_30bit_fine_v2/latest.pt"
    )
    for config in (stage1, stage2):
        guidance = config["train"]["region_guidance"]
        residual = config["train"]["residual_constraint"]
        assert guidance["loss_mode"] == "energy_ratio"
        assert guidance["mode"] == "strict_multiscale"
        assert guidance["blur_kernel"] == 1
        assert get_residual_constraint_settings(
            residual
        )["mode"] == "spatial_budget"
        assert config["model"]["use_content_gated_wm_map"] is True

    final_guidance = stage1["train"]["region_guidance"]
    assert final_guidance["edge_scales"] == [1, 3]
    assert final_guidance["texture_scales"] == [3, 5]
    assert final_guidance["target_mask_area"] == 0.24
    assert final_guidance["mask_temperature"] == 0.03

    stage2_guidance = stage2["train"]["region_guidance"]
    assert stage2_guidance["edge_scales"] == [1, 3]
    assert stage2_guidance["texture_scales"] == [3, 5]
    assert stage2_guidance["target_mask_area"] == 0.24
    assert stage2_guidance["mask_temperature"] == 0.03
    assert stage2_guidance == final_guidance
    assert stage2["train"]["residual_constraint"] == stage1[
        "train"
    ]["stages"]["full"]["residual_constraint"]
    assert stage2["model"]["wm_map_flat_floor"] == stage1[
        "train"
    ]["stages"]["full"]["wm_map_flat_floor"]

    warmup_stage1 = load_config(stage1_path)
    warmup_stage1["train"]["stage"] = "warmup"
    warmup_weights = get_loss_weights(warmup_stage1, global_step=0)
    assert warmup_weights["lambda_wm"] == 20.0
    assert warmup_weights["lambda_diff"] == 0.01
    assert warmup_weights["lambda_region"] == 0.10
    warmup_residual = get_stage_training_setting(
        warmup_stage1["train"], "residual_constraint"
    )
    assert warmup_residual["flat_max_abs_delta_01"] == 0.012
    assert warmup_residual["texture_max_abs_delta_01"] == 0.060
    assert warmup_residual["mask_power"] == 1.0
    assert get_stage_training_setting(
        warmup_stage1["train"], "wm_map_flat_floor"
    ) == 0.20
    assert get_stage_training_setting(
        warmup_stage1["train"], "residual_spectral"
    )["enabled"] is False
    warmup_guidance = get_stage_training_setting(
        warmup_stage1["train"], "region_guidance"
    )
    assert warmup_guidance["target_mask_area"] == 0.30
    assert warmup_guidance["mask_temperature"] == 0.05
    validate_initialization_policy(warmup_stage1)

    balance_stage1 = load_config(stage1_path)
    balance_stage1["train"]["stage"] = "balance"
    balance_residual = get_stage_training_setting(
        balance_stage1["train"], "residual_constraint"
    )
    assert balance_residual["flat_max_abs_delta_01"] == 0.007
    assert balance_residual["texture_max_abs_delta_01"] == 0.050
    assert balance_residual["mask_power"] == 1.6
    assert get_stage_training_setting(
        balance_stage1["train"], "wm_map_flat_floor"
    ) == 0.10
    balance_guidance = get_stage_training_setting(
        balance_stage1["train"], "region_guidance"
    )
    assert balance_guidance["target_mask_area"] == 0.27
    assert balance_guidance["mask_temperature"] == 0.04

    full_stage1 = load_config(stage1_path)
    full_stage1["train"]["stage"] = "full"
    full_weights = get_loss_weights(full_stage1, global_step=0)
    assert full_weights["lambda_diff"] == 0.3
    assert full_weights["lambda_img"] == 3.0
    full_residual = get_stage_training_setting(
        full_stage1["train"], "residual_constraint"
    )
    assert full_residual["flat_max_abs_delta_01"] == 0.003
    assert full_residual["texture_max_abs_delta_01"] == 0.040
    assert full_residual["mask_power"] == 2.8
    assert get_stage_training_setting(
        full_stage1["train"], "wm_map_flat_floor"
    ) == 0.03
    full_guidance = get_stage_training_setting(
        full_stage1["train"], "region_guidance"
    )
    assert full_guidance["target_mask_area"] == 0.24
    assert full_guidance["mask_temperature"] == 0.03
    assert get_stage_training_setting(
        full_stage1["train"], "residual_spectral"
    )["enabled"] is True
    try:
        validate_initialization_policy(full_stage1)
    except ValueError:
        pass
    else:
        raise AssertionError("Stage 1 full must not start without a checkpoint")
    validate_initialization_policy(
        full_stage1,
        init_from_path="checkpoints_stage1_strict_texture_30bit_fine_v2/best.pt",
    )

    invalid_stage1 = load_config(stage1_path)
    invalid_stage1["train"]["stage"] = "blacne"
    try:
        get_loss_weights(invalid_stage1, global_step=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Stage 1 must reject an invalid manual phase name")

    try:
        validate_initialization_policy(stage2)
    except ValueError:
        pass
    else:
        raise AssertionError("Stage 2 must reject a new run without --init_from")

    validate_initialization_policy(
        stage2,
        init_from_path=stage2["initialization"]["expected_init_from"],
    )


def main():
    tests = [
        test_strict_mask_is_finite_and_content_dependent,
        test_ssim_metric_identity_and_perturbation,
        test_spatial_budget_bounds_and_gradient,
        test_energy_ratio_targets_and_gradient,
        test_horizontal_stripe_diagnostics_and_loss,
        test_two_stage_configs_and_initialization_guards,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"[PASS] {len(tests)} strict-texture tests")


if __name__ == "__main__":
    main()
