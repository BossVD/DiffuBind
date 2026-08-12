"""Smoke tests for the first-round constrained Stage-2 training path."""

import os
import sys

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from guided_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from models.watermark_residual import (
    build_edge_texture_guidance,
    constrain_watermark_residual,
)
from models.watermark_unet import WatermarkConditionedUNet
from sample_embed_watermark import embed_watermark_sample
from train_watermark_diffusion import (
    build_training_optimizer,
    configure_encoder_train_mode,
    select_multi_attack_candidates,
)


def build_small_model(content_gated=False):
    return WatermarkConditionedUNet(
        image_size=32,
        base_channels=32,
        cond_dim=128,
        watermark_length=30,
        use_pretrained_unet=False,
        pretrained_path=None,
        use_watermark_time_emb=True,
        use_watermark_spatial_map=True,
        wm_map_channels=4,
        wm_map_size=4,
        wm_time_scale=1.0,
        wm_map_scale=1.0,
        use_content_gated_wm_map=content_gated,
        wm_map_flat_floor=0.2,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    legacy = build_small_model(content_gated=False).to(device).eval()
    gated = build_small_model(content_gated=True).to(device).eval()

    # The gate has no parameters, so Stage-1 state dictionaries remain strict-loadable.
    result = gated.load_state_dict(legacy.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys

    configure_encoder_train_mode(
        gated,
        "partial",
        partial_output_blocks=2,
        freeze_watermark_map_mlp=True,
    )
    assert all(not p.requires_grad for p in gated.watermark_map_mlp.parameters())
    assert all(p.requires_grad for p in gated.watermark_mlp.parameters())
    assert all(
        not p.requires_grad for p in gated.inner_unet.input_blocks[0].parameters()
    )
    assert any(
        p.requires_grad for p in gated.inner_unet.output_blocks[-1].parameters()
    )

    decoder_stub = torch.nn.Linear(8, 4).to(device)
    optimizer = build_training_optimizer(
        gated,
        decoder_stub,
        lr=2e-5,
        encoder_lr=1e-6,
        decoder_lr=2e-5,
    )
    assert [group["name"] for group in optimizer.param_groups] == [
        "encoder",
        "decoder",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-6, 2e-5]

    cover = torch.rand(1, 3, 32, 32, device=device).mul(2.0).sub(1.0)
    bits = torch.randint(0, 2, (1, 30), device=device).float()
    allowance, _ = build_edge_texture_guidance(
        (cover + 1.0) / 2.0,
        {
            "edge_weight": 0.4,
            "texture_weight": 0.6,
            "texture_kernel": 5,
            "dilation_kernel": 3,
            "blur_kernel": 3,
            "gamma": 3.0,
            "min_penalty": 0.03,
        },
    )
    x_t = torch.randn_like(cover)
    timestep = torch.tensor([50.0], device=device)
    with torch.no_grad():
        legacy_output = legacy(x_t, timestep, cover, bits)
        all_texture_output = gated(
            x_t,
            timestep,
            cover,
            bits,
            content_mask=torch.ones_like(allowance),
        )
    assert torch.allclose(legacy_output, all_texture_output, atol=1e-6, rtol=1e-5)

    # Regression for guided_diffusion's custom attention checkpoint: frozen
    # parameters must not be passed to autograd.grad as differentiation targets.
    gated.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        checkpoint_output = gated(
            x_t,
            timestep,
            cover,
            bits,
            content_mask=allowance,
        )
        checkpoint_loss = checkpoint_output.square().mean()
    checkpoint_loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in gated.parameters()
        if parameter.requires_grad
    )
    assert all(
        parameter.grad is None
        for parameter in gated.parameters()
        if not parameter.requires_grad
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in gated.parameters()
        if parameter.requires_grad
    )

    constraint = {
        "enabled": True,
        "max_abs_delta_01": 0.03,
        "flat_floor": 0.2,
    }
    mask = torch.zeros_like(allowance)
    mask[:, :, :, 16:] = 1.0
    raw = (cover + 0.5 * torch.randn_like(cover)).requires_grad_()
    constrained = constrain_watermark_residual(raw, cover, mask, constraint)
    delta_01 = ((constrained - cover) / 2.0).abs()
    assert delta_01.max().item() <= 0.030001
    assert delta_01[:, :, :, :16].max().item() <= 0.006001
    constrained.mean().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum().item() > 0.0

    selected = select_multi_attack_candidates(
        ["pimog", "oled", "led", "projector"],
        [0.25, 0.25, 0.25, 0.25],
        count=2,
        device=device,
    )
    assert len(selected) == 2 and len(set(selected)) == 2

    betas = get_named_beta_schedule("linear", 1000)
    diffusion = GaussianDiffusion(
        betas=torch.tensor(betas, dtype=torch.float32),
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )
    with torch.no_grad():
        sampled = embed_watermark_sample(
            diffusion,
            gated,
            cover,
            bits,
            t_start=2,
            region_guidance_config={
                "edge_weight": 0.4,
                "texture_weight": 0.6,
                "texture_kernel": 5,
                "dilation_kernel": 3,
                "blur_kernel": 3,
                "gamma": 3.0,
                "min_penalty": 0.03,
            },
            residual_constraint_config=constraint,
        )
    sampled_delta = ((sampled - cover) / 2.0).abs().max().item()
    assert sampled_delta <= 0.030001

    print(
        "[PASS] constrained Stage-2 smoke test: "
        f"device={device}, max_delta_01={delta_01.max().item():.6f}, "
        f"sampled_max_delta_01={sampled_delta:.6f}, attacks={selected}"
    )


if __name__ == "__main__":
    main()
