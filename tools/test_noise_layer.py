"""Visual and numerical smoke test for each concrete degradation layer."""

import argparse
import os
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.utils import save_image
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from NOISE_LAYER.build_noise_layer import build_noise_layer


def resolve_project_input(path):
    """Resolve an input path from either the current directory or project root."""
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.join(PROJECT_ROOT, path)


def save_labeled_comparison(outputs, output_path):
    """Save a horizontal comparison image with each degradation mode labeled."""
    images = [
        transforms.ToPILImage()(tensor[0].detach().cpu().clamp(0.0, 1.0))
        for tensor in outputs.values()
    ]
    labels = list(outputs.keys())
    image_width, image_height = images[0].size
    font_size = max(14, image_width // 10)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    label_height = max(28, font_size + 12)
    measure_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    label_widths = [
        measure_draw.textbbox((0, 0), label, font=font)[2] for label in labels
    ]
    cell_width = max(image_width, max(label_widths) + 12)
    canvas = Image.new(
        "RGB",
        (cell_width * len(images), image_height + label_height),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(labels, images)):
        x_offset = index * cell_width
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = x_offset + (cell_width - text_width) // 2
        text_y = (label_height - text_height) // 2 - text_box[1]
        draw.text((text_x, text_y), label, fill="black", font=font)
        image_x = x_offset + (cell_width - image_width) // 2
        canvas.paste(image, (image_x, label_height))
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Test degradation layers")
    parser.add_argument("--input", default=None, help="Optional sample image path")
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "configs", "watermark_stage2.yaml"),
        help="Noise-layer YAML (relative paths may be based on the project root)",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PROJECT_ROOT, "noise_layer_debug"),
        help="Directory used to save noise-layer comparison images",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--projector_samples", type=int, default=4)
    args = parser.parse_args()

    config_path = resolve_project_input(args.config)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Noise-layer config does not exist: {config_path}. "
            f"Use --config to select a YAML under {os.path.join(PROJECT_ROOT, 'configs')}."
        )
    with open(config_path, "r", encoding="utf-8-sig") as handle:
        cfg = yaml.safe_load(handle)
    image_size = args.image_size or cfg.get("data", {}).get("image_size", 128)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    if args.input:
        input_path = resolve_project_input(args.input)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input image does not exist: {input_path}")
        x = transform(Image.open(input_path).convert("RGB")).unsqueeze(0).to(device)
    else:
        # A deterministic RGB gradient keeps the script usable in a fresh repo.
        axis = torch.linspace(0.0, 1.0, image_size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        x = torch.stack((xx, yy, (xx + yy) / 2.0)).unsqueeze(0).to(device)
    os.makedirs(args.output_dir, exist_ok=True)
    outputs = {"original": x}

    # Mixed only selects one concrete layer per call. Showing it here would
    # duplicate a randomly re-sampled PIMoG or Projector result and make the
    # visual comparison ambiguous.
    for noise_type in ("pimog", "oled", "led", "projector"):
        test_cfg = dict(cfg)
        test_cfg["noise_layer"] = dict(cfg.get("noise_layer", {}), type=noise_type)
        test_cfg["noise_layer"]["pimog"] = dict(
            cfg.get("noise_layer", {}).get("pimog", {}), p=1.0
        )
        test_cfg["noise_layer"]["oled"] = dict(
            cfg.get("noise_layer", {}).get("oled", {}), p=1.0
        )
        test_cfg["noise_layer"]["led"] = dict(
            cfg.get("noise_layer", {}).get("led", {}), p=1.0
        )
        test_cfg["noise_layer"]["projector"] = dict(
            cfg.get("noise_layer", {}).get("projector", {}), p=1.0
        )
        layer = build_noise_layer(test_cfg).to(device)
        x_for_grad = x.detach().clone().requires_grad_(True)
        y = layer(x_for_grad)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert y.min() >= 0 and y.max() <= 1
        probe = torch.randn_like(y)
        (y * probe).mean().backward()
        assert x_for_grad.grad is not None
        assert torch.isfinite(x_for_grad.grad).all(), (
            f"{noise_type} produced non-finite input gradients"
        )
        y = y.detach()
        outputs[noise_type] = y
        save_image(y[0], os.path.join(args.output_dir, f"{noise_type}_deg.png"))
        if noise_type == "projector" and args.projector_samples > 1:
            samples = [y]
            for _ in range(args.projector_samples - 1):
                samples.append(layer(x))
            save_image(
                torch.cat(samples, dim=0),
                os.path.join(args.output_dir, "projector_samples.png"),
                nrow=args.projector_samples,
            )

    save_image(x[0], os.path.join(args.output_dir, "original.png"))
    save_labeled_comparison(
        outputs,
        os.path.join(args.output_dir, "compare.png"),
    )
    print(f"Noise-layer visualization saved to {args.output_dir}")


if __name__ == "__main__":
    main()
