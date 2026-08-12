"""
Watermark-Conditioned UNet for Image-to-Image Diffusion.

Extends the original guided-diffusion UNet to accept:
  - x_t:         noisy image [B, 3, H, W]
  - t:           timestep
  - cover_img:   original cover image [B, 3, H, W] as condition
  - wm_bits:     watermark bits [B, wm_length] as condition

Condition injection:
  - cover_img:   channel-wise concatenation with x_t
  - wm_bits:     MLP embedding + addition to time embedding
  - wm_map:      spatial watermark map concatenated into UNet input channels

*** IMPORTANT: Reinitializes all zero_module convs in ResBlocks
    to Xavier init. The original DDPM uses zero_module which blocks
    gradient flow through FiLM conditioning on the first forward pass.
    With Xavier init, gradient flows immediately through watermark_mlp.

KEY PRINCIPLE (see train_watermark_diffusion.py):
  loss_wm must backprop through diffusion_model — never .detach() pred_x0.
"""
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from guided_diffusion.unet import UNetModel


class WatermarkConditionedUNet(nn.Module):
    """
    UNet wrapper that adds cover image and watermark bit conditioning.

    Forward signature:
        pred_noise = model(x_t, t, cover_img, wm_bits)
    """

    def __init__(
        self,
        image_size=128,
        base_channels=64,
        cond_dim=256,
        watermark_length=30,
        use_pretrained_unet=False,
        pretrained_path=None,
        use_watermark_time_emb=True,
        use_watermark_spatial_map=True,
        wm_map_channels=4,
        wm_map_size=16,
        wm_time_scale=1.0,
        wm_map_scale=1.0,
        use_content_gated_wm_map=False,
        wm_map_flat_floor=0.2,
        **unet_kwargs,
    ):
        super().__init__()

        self.image_size = image_size
        self.watermark_length = watermark_length
        self.cond_dim = cond_dim
        self.use_watermark_time_emb = use_watermark_time_emb
        self.use_watermark_spatial_map = use_watermark_spatial_map
        self.wm_map_channels = wm_map_channels
        self.wm_map_size = wm_map_size
        self.wm_time_scale = wm_time_scale
        self.wm_map_scale = wm_map_scale
        self.use_content_gated_wm_map = bool(use_content_gated_wm_map)
        self.wm_map_flat_floor = float(wm_map_flat_floor)
        if not 0.0 <= self.wm_map_flat_floor <= 1.0:
            raise ValueError("wm_map_flat_floor must be in [0, 1]")

        # --- Watermark MLP: wm_bits -> global embedding ---
        self.watermark_mlp = nn.Sequential(
            nn.Linear(watermark_length, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # --- Watermark map MLP: wm_bits -> spatial condition map ---
        self.watermark_map_mlp = nn.Sequential(
            nn.Linear(watermark_length, 256),
            nn.SiLU(),
            nn.Linear(256, wm_map_channels * wm_map_size * wm_map_size),
        )

        # --- Inner UNet ---
        # in_channels=6 for x_t + cover_img, optionally plus wm_map channels.
        # wm_length=0 disables built-in watermark path
        unet_in_channels = 6
        if use_watermark_spatial_map:
            unet_in_channels += wm_map_channels
        default_unet_kwargs = dict(
            image_size=image_size,
            in_channels=unet_in_channels,
            model_channels=base_channels,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(16, 8) if image_size >= 32 else (8,),
            dropout=0.0,
            channel_mult=(1, 2, 4, 8) if image_size >= 64 else (1, 2, 4),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=False,
            use_new_attention_order=False,
            wm_length=0,
        )
        default_unet_kwargs.update(unet_kwargs)
        self.inner_unet = UNetModel(**default_unet_kwargs)

        # --- Fix ALL zero_module initializations ---
        # zero_module sets conv weights to zero, blocking gradient flow
        # through FiLM conditioning in the first forward pass.
        # We replace with small Xavier init so gradients flow immediately.
        self._reinit_zero_modules()

        # --- Load pre-trained weights if requested ---
        if use_pretrained_unet and pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path)

    def _reinit_zero_modules(self):
        """
        Reinitialize ALL zero_module conv layers with Xavier uniform.

        These include:
        - output conv (inner_unet.out[2])
        - out_layers conv in every ResBlock (input_blocks, middle_block, output_blocks)

        Without this fix, gradient through FiLM conditioning is dead on step 0,
        meaning watermark_mlp gets zero gradient and never trains.
        """
        count = 0

        def _fix_module(module, prefix=""):
            nonlocal count
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Conv2d) and hasattr(child, 'weight'):
                    # Check if weights are all zero (sign of zero_module)
                    if child.weight.abs().sum() == 0:
                        nn.init.xavier_uniform_(child.weight, gain=1e-3)
                        nn.init.zeros_(child.bias)
                        count += 1
                _fix_module(child, full_name)

        _fix_module(self.inner_unet)
        # print(f"[WatermarkConditionedUNet] Reinitialized {count} zero_module convs with Xavier (gain=1e-3).")

    def _load_pretrained_weights(self, pretrained_path):
        """
        Load pre-trained UNet weights, handling the 3->6 channel expansion.
        """
        if not os.path.exists(pretrained_path):
            print(f"[WatermarkConditionedUNet] Pretrained path not found: {pretrained_path}")
            print("[WatermarkConditionedUNet] Training from scratch.")
            return

        print(f"[WatermarkConditionedUNet] Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')

        if 'model' in state_dict:
            state_dict = state_dict['model']

        # Handle first conv layer expansion to the current condition width.
        first_conv_key = 'input_blocks.0.0.weight'
        if first_conv_key in state_dict:
            old_weight = state_dict[first_conv_key]
            current_weight = self.inner_unet.state_dict()[first_conv_key]
            if (
                old_weight.ndim == 4
                and old_weight.shape[1] < current_weight.shape[1]
                and old_weight.shape[0] == current_weight.shape[0]
                and old_weight.shape[2:] == current_weight.shape[2:]
            ):
                new_weight = current_weight.clone()
                new_weight[:, :old_weight.shape[1], :, :] = old_weight
                new_weight[:, old_weight.shape[1]:, :, :] = 0.0
                state_dict[first_conv_key] = new_weight
                print(
                    "[WatermarkConditionedUNet] Expanded first conv from "
                    f"{old_weight.shape[1]}->{current_weight.shape[1]} input channels."
                )

        # Fix key mismatches
        unet_state = self.inner_unet.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            clean_k = k.replace('inner_unet.', '')
            if clean_k in unet_state and v.shape == unet_state[clean_k].shape:
                filtered_dict[clean_k] = v

        if len(filtered_dict) > 0:
            missing, unexpected = self.inner_unet.load_state_dict(filtered_dict, strict=False)
            print(f"[WatermarkConditionedUNet] Loaded {len(filtered_dict)} matching weights.")
        else:
            print("[WatermarkConditionedUNet] No matching weights found — training from scratch.")

    def forward(self, x_t, t, cover_img, wm_bits, content_mask=None):
        """
        Forward pass with cover image and watermark conditions.

        Args:
            x_t:        [B, 3, H, W]  noisy image at timestep t, range [-1, 1]
            t:          [B]           diffusion timestep indices (can be float)
            cover_img:  [B, 3, H, W]  original cover image, range [-1, 1]
            wm_bits:    [B, wm_len]   watermark bits, 0/1 float

        Returns:
            pred_noise: [B, 3, H, W]  predicted noise
        """
        # 1. Compute time embedding
        from guided_diffusion.nn import timestep_embedding
        time_emb = self.inner_unet.time_embed(
            timestep_embedding(t, self.inner_unet.model_channels)
        )

        # 2. Optionally fuse watermark bits into the global time embedding.
        wm_bits = wm_bits.float()
        if self.use_watermark_time_emb:
            wm_emb = self.watermark_mlp(wm_bits)  # [B, cond_dim]
            cond_emb = time_emb + self.wm_time_scale * wm_emb
        else:
            cond_emb = time_emb

        # 3. Optionally add an explicit spatial watermark condition.
        if self.use_watermark_spatial_map:
            batch = x_t.shape[0]
            wm_map = self.watermark_map_mlp(wm_bits)
            wm_map = wm_map.view(
                batch,
                self.wm_map_channels,
                self.wm_map_size,
                self.wm_map_size,
            )
            wm_map = F.interpolate(
                wm_map,
                size=x_t.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            wm_map = self.wm_map_scale * wm_map
            if self.use_content_gated_wm_map:
                if content_mask is None:
                    raise ValueError(
                        "content_mask is required when "
                        "use_content_gated_wm_map=true"
                    )
                if (
                    content_mask.ndim != 4
                    or content_mask.shape[0] != batch
                    or content_mask.shape[1] not in (1, self.wm_map_channels)
                    or content_mask.shape[2:] != x_t.shape[2:]
                ):
                    raise ValueError(
                        "content_mask must be [B,1,H,W] or match wm_map "
                        f"channels, got {tuple(content_mask.shape)}"
                    )
                content_gate = self.wm_map_flat_floor + (
                    1.0 - self.wm_map_flat_floor
                ) * content_mask.detach().to(
                    device=wm_map.device,
                    dtype=wm_map.dtype,
                ).clamp(0.0, 1.0)
                wm_map = wm_map * content_gate
            model_input = torch.cat([x_t, cover_img, wm_map], dim=1)
        else:
            model_input = torch.cat([x_t, cover_img], dim=1)

        # 4. Pass through inner UNet with pre-fused conditions.
        return self._inner_unet_forward(model_input, cond_emb)

    def _inner_unet_forward(self, x, emb):
        """
        Manual forward through inner UNet with pre-computed conditioning embedding.

        Mirrors UNetModel.forward but uses pre-fused cond_emb.
        """
        h = x.type(self.inner_unet.dtype)
        hs = []

        # Downsample
        for module in self.inner_unet.input_blocks:
            h = module(h, emb)
            hs.append(h)

        # Middle
        h = self.inner_unet.middle_block(h, emb)

        # Upsample
        for module in self.inner_unet.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        return self.inner_unet.out(h)



