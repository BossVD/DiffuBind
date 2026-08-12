"""Visual smoke test for the OLED display-camera degradation layer."""

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

from NOISE_LAYER.OLED_Layer import OLED_Layer


def _make_default_image(image_size, device):
    axis = torch.linspace(0.0, 1.0, image_size, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    checker = (torch.remainder(torch.arange(image_size, device=device), 16) < 8).float()
    checker = checker.reshape(1, -1).expand(image_size, -1)
    image = torch.stack((xx, yy, 0.35 + 0.65 * checker), dim=0)
    return image.unsqueeze(0).clamp(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Render OLED degradation sample")
    parser.add_argument("--input", default=None, help="Optional RGB image path")
    parser.add_argument("--output", default="outputs/oled_test.png", help="Degraded image path")
    parser.add_argument("--compare_output", default=None, help="Optional side-by-side image path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--subpixel_mode", default="pentile", choices=["pentile", "stripe"])
    parser.add_argument("--use_jpeg", action="store_true", help="Enable differentiable JPEG proxy")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.input:
        transform = transforms.Compose(
            [
                transforms.Resize(args.size, antialias=True),
                transforms.CenterCrop(args.size),
                transforms.ToTensor(),
            ]
        )
        x = transform(Image.open(args.input).convert("RGB")).unsqueeze(0).to(device)
    else:
        x = _make_default_image(args.size, device)

    layer = OLED_Layer(subpixel_mode=args.subpixel_mode, use_jpeg=args.use_jpeg).to(device)
    layer.eval()
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert y.min() >= 0.0 and y.max() <= 1.0

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_image(y[0], args.output)

    compare_output = args.compare_output
    if compare_output is None:
        root, ext = os.path.splitext(args.output)
        compare_output = f"{root}_compare{ext or '.png'}"
    compare_dir = os.path.dirname(os.path.abspath(compare_output))
    if compare_dir:
        os.makedirs(compare_dir, exist_ok=True)
    save_image(torch.cat([x, y], dim=0), compare_output, nrow=2)
    print(f"OLED degraded image saved to {args.output}")
    print(f"OLED comparison image saved to {compare_output}")


if __name__ == "__main__":
    main()
