"""Shape smoke test for watermark spatial-map conditioning."""

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.watermark_unet import WatermarkConditionedUNet


def main():
    model = WatermarkConditionedUNet(
        image_size=128,
        base_channels=64,
        cond_dim=256,
        watermark_length=30,
        use_watermark_time_emb=True,
        use_watermark_spatial_map=True,
        wm_map_channels=4,
        wm_map_size=16,
        wm_time_scale=1.0,
        wm_map_scale=1.0,
    )
    x_t = torch.randn(2, 3, 128, 128)
    cover_img = torch.randn(2, 3, 128, 128)
    wm_bits = torch.randint(0, 2, (2, 30)).float()
    t = torch.randint(0, 1000, (2,))

    with torch.no_grad():
        pred_noise = model(x_t=x_t, t=t, cover_img=cover_img, wm_bits=wm_bits)

    expected_shape = (2, 3, 128, 128)
    assert tuple(pred_noise.shape) == expected_shape, (
        f"Expected {expected_shape}, got {tuple(pred_noise.shape)}"
    )
    print(f"[OK] pred_noise.shape={tuple(pred_noise.shape)}")


if __name__ == "__main__":
    main()
