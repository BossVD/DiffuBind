"""Focused finite-value, broadcasting, AMP, and gradient tests for region loss."""

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_watermark_diffusion import (
    combine_region_guidance_losses,
    get_region_outside_settings,
    residual_outside_region_loss,
    residual_region_enrichment_loss,
)


def assert_close(actual, expected, message, atol=1e-6, rtol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{message}: actual={actual.detach().cpu().item()}, "
            f"expected={expected.detach().cpu().item()}"
        )


def test_shape_and_soft_mask_broadcasting():
    torch.manual_seed(1)
    delta = torch.randn(2, 3, 5, 7)
    mask = torch.rand(2, 1, 5, 7)
    outside = 1.0 - mask
    expected = (
        (outside * delta.abs()).sum()
        / (outside.sum() * delta.shape[1] + 1e-6)
    )
    actual = residual_outside_region_loss(delta, mask)
    assert_close(actual, expected, "single-channel soft-mask broadcasting")

    channel_mask = mask.expand(-1, 3, -1, -1).clone()
    expected_channel = (
        ((1.0 - channel_mask) * delta.abs()).sum()
        / ((1.0 - channel_mask).sum() + 1e-6)
    )
    actual_channel = residual_outside_region_loss(delta, channel_mask)
    assert_close(actual_channel, expected_channel, "per-channel soft mask")

    try:
        residual_outside_region_loss(delta, torch.rand(2, 2, 5, 7))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid allowance channels must raise ValueError")


def test_zero_one_and_random_masks_are_finite():
    torch.manual_seed(2)
    delta = torch.randn(2, 3, 8, 8)
    zero_mask = torch.zeros(2, 1, 8, 8)
    one_mask = torch.ones(2, 1, 8, 8)
    random_mask = torch.rand(2, 1, 8, 8)

    zero_loss = residual_outside_region_loss(delta, zero_mask)
    one_loss = residual_outside_region_loss(delta, one_mask)
    random_loss = residual_outside_region_loss(delta, random_mask)
    if not all(torch.isfinite(value) for value in (zero_loss, one_loss, random_loss)):
        raise AssertionError("all mask edge cases must produce finite losses")
    assert_close(zero_loss, delta.abs().mean(), "all-zero mask")
    assert_close(one_loss, delta.new_zeros(()), "all-one mask")


def test_backward_and_smooth_region_gradient():
    delta = torch.full((1, 3, 4, 4), 0.1, requires_grad=True)
    mask = torch.ones(1, 1, 4, 4)
    mask[:, :, :, 2:] = 0.0
    loss = residual_outside_region_loss(delta, mask)
    loss.backward()
    if delta.grad is None or not torch.isfinite(delta.grad).all():
        raise AssertionError("outside loss must backpropagate finite gradients")
    if not bool((delta.grad[:, :, :, 2:].abs() > 0).all()):
        raise AssertionError("smooth/outside pixels must receive non-zero gradients")
    if not bool((delta.grad[:, :, :, :2] == 0).all()):
        raise AssertionError("fully allowed pixels must not receive outside-loss gradients")


def test_outside_gradient_survives_zero_enrichment_hinge():
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, :, :2] = 1.0
    values = torch.full((1, 3, 4, 4), 0.1)
    values[:, :, :, :2] = 1.0
    delta = values.requires_grad_()

    enrichment = residual_region_enrichment_loss(
        delta,
        mask,
        target_enrichment=0.03,
    )
    outside = residual_outside_region_loss(delta, mask)
    enrichment_grad = torch.autograd.grad(
        enrichment,
        delta,
        retain_graph=True,
    )[0]
    outside_grad = torch.autograd.grad(outside, delta)[0]

    assert_close(enrichment, delta.new_zeros(()), "satisfied enrichment hinge")
    if float(enrichment_grad.abs().max()) != 0.0:
        raise AssertionError("satisfied enrichment hinge should have zero gradient")
    if not bool((outside_grad[:, :, :, 2:].abs() > 0).all()):
        raise AssertionError(
            "outside loss must retain smooth-region gradients after hinge saturation"
        )


def test_amp_path():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    delta = torch.randn(2, 3, 8, 8, device=device, requires_grad=True)
    mask = torch.rand(2, 1, 8, 8, device=device)
    with torch.amp.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
        enabled=True,
    ):
        loss = residual_outside_region_loss(delta, mask)
    loss.backward()
    if not torch.isfinite(loss) or delta.grad is None or not torch.isfinite(delta.grad).all():
        raise AssertionError("AMP path must produce finite loss and gradients")


def test_disabled_baseline_is_exactly_unchanged():
    enrichment = torch.tensor(0.1234567)
    outside = torch.tensor(0.7654321)
    common_objective = torch.tensor(4.25)
    old_region_total = 0.5 * enrichment
    new_region_total = combine_region_guidance_losses(
        enrichment,
        outside,
        lambda_enrichment=0.5,
        lambda_outside=999.0,
        outside_enabled=False,
    )
    old_total = common_objective + old_region_total
    new_total = common_objective + new_region_total
    if not torch.equal(old_region_total, new_region_total):
        raise AssertionError("disabled region term must match the old expression")
    if not torch.equal(old_total, new_total):
        raise AssertionError("outside_enabled=false must exactly match old region total")


def test_config_validation():
    if get_region_outside_settings({}) != (False, 0.0):
        raise AssertionError("missing outside settings must preserve the old baseline")
    if get_region_outside_settings(
        {"outside_enabled": False, "lambda_outside": 0.0}
    ) != (False, 0.0):
        raise AssertionError("explicit baseline settings parsed incorrectly")
    enabled, weight = get_region_outside_settings(
        {"outside_enabled": True, "lambda_outside": 0.25}
    )
    if not enabled or not math.isclose(weight, 0.25):
        raise AssertionError("enabled outside settings parsed incorrectly")

    invalid_configs = (
        {"outside_enabled": "true", "lambda_outside": 0.25},
        {"outside_enabled": True, "lambda_outside": 0.0},
        {"outside_enabled": True, "lambda_outside": -1.0},
        {"outside_enabled": True, "lambda_outside": float("nan")},
    )
    for config in invalid_configs:
        try:
            get_region_outside_settings(config)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"invalid region config must fail: {config}")


def main():
    tests = [
        test_shape_and_soft_mask_broadcasting,
        test_zero_one_and_random_masks_are_finite,
        test_backward_and_smooth_region_gradient,
        test_outside_gradient_survives_zero_enrichment_hinge,
        test_amp_path,
        test_disabled_baseline_is_exactly_unchanged,
        test_config_validation,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"[PASS] {len(tests)} outside-region tests")


if __name__ == "__main__":
    main()
