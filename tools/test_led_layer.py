"""Visual smoke test for the LED display-camera degradation layer."""

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from NOISE_LAYER.LED_Layer import LEDLayer


def _make_default_image(image_size, device):
    axis = torch.linspace(0.0, 1.0, image_size, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy, (xx + yy) * 0.5)).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Render LED degradation samples")
    parser.add_argument("--input", default=None, help="Optional RGB image path")
    parser.add_argument("--output_dir", default="outputs/led_layer_debug")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--subpixel_mode", default="rgb_triplet", choices=["mono_dot", "rgb_triplet"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.input:
        transform = transforms.Compose(
            [
                transforms.Resize(args.image_size, antialias=True),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
            ]
        )
        x = transform(Image.open(args.input).convert("RGB")).unsqueeze(0).to(device)
    else:
        x = _make_default_image(args.image_size, device)

    os.makedirs(args.output_dir, exist_ok=True)
    save_image(x[0], os.path.join(args.output_dir, "original.png"))

    outputs = [x]
    for severity in ("mild", "medium", "strong", "random"):
        layer = LEDLayer(severity=severity, subpixel_mode=args.subpixel_mode).to(device)
        with torch.no_grad():
            y = layer(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert y.min() >= 0.0 and y.max() <= 1.0
        outputs.append(y)
        save_image(y[0], os.path.join(args.output_dir, f"led_{severity}.png"))

    save_image(
        torch.cat(outputs, dim=0),
        os.path.join(args.output_dir, "compare.png"),
        nrow=len(outputs),
    )
    print(f"LED-layer visualization saved to {args.output_dir}")


if __name__ == "__main__":
    main()
