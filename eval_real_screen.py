"""
Decode watermark from real screen-captured photos.

Usage:
    python eval_real_screen.py \
        --checkpoint checkpoints_stage2/best.pt \
        --input_dir ./real_screen_photos/ \
        --watermark "1010101011001010" \
        --image_size 128
"""

import os
import sys
import argparse
import glob

import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.transforms.functional import center_crop, resize

sys.path.insert(0, os.path.dirname(__file__))

from models.watermark_unet import WatermarkConditionedUNet
from models.watermark_decoder import (
    build_watermark_decoder,
    load_watermark_decoder_state,
)


def main():
    parser = argparse.ArgumentParser(description='Decode watermark from real screen-captured photos')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint (.pt)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing real screen-captured photos (.png/.jpg)')
    parser.add_argument('--watermark', type=str, default=None,
                        help='Expected watermark bits (e.g. "101010..."). If not provided, only outputs decoded bits.')
    parser.add_argument('--watermark_length', type=int, default=None,
                        help='Watermark bit length; defaults to the checkpoint config')
    parser.add_argument('--image_size', type=int, default=None,
                        help='Image size; defaults to the checkpoint config')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Eval] Using device: {device}")

    # --- Load checkpoint ---
    print(f"[Eval] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Recover model config
    cfg = ckpt.get('config', {})
    model_cfg = cfg.get('model', {})
    diffusion_cfg = cfg.get('diffusion', {})
    image_size = (
        args.image_size
        if args.image_size is not None
        else cfg.get('data', {}).get('image_size', 128)
    )
    watermark_length = (
        args.watermark_length
        if args.watermark_length is not None
        else cfg.get('data', {}).get('watermark_length', 30)
    )

    # --- Build decoder ---
    decoder = build_watermark_decoder(
        cfg,
        watermark_length=watermark_length,
    ).to(device)
    missing, unexpected, mismatched = load_watermark_decoder_state(
        decoder, ckpt['decoder']
    )
    if missing or unexpected or mismatched:
        print(
            "[Eval] Decoder checkpoint partially loaded "
            "(architecture may have changed)."
        )
    decoder.eval()

    # --- Collect image files ---
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    image_paths = []
    for ext in exts:
        image_paths.extend(sorted(glob.glob(os.path.join(args.input_dir, ext))))
        image_paths.extend(sorted(glob.glob(os.path.join(args.input_dir, ext.upper()))))

    if not image_paths:
        print(f"[Eval] ERROR: No images found in {args.input_dir}")
        sys.exit(1)

    print(f"[Eval] Found {len(image_paths)} image(s)")

    # --- Prepare expected watermark ---
    expected_bits = None
    expected_bits_str = None
    if args.watermark:
        wm_str = args.watermark.strip()
        if len(wm_str) < watermark_length:
            wm_str = wm_str.ljust(watermark_length, '0')
        elif len(wm_str) > watermark_length:
            wm_str = wm_str[:watermark_length]
        expected_bits_str = wm_str
        expected_bits = torch.tensor([int(b) for b in wm_str], device=device).float()

    # --- Decode each photo ---
    print(
        f"{'Image':<40s} {'Decoded bits (first 32)':<35s} "
        f"{'Accuracy':<10s} {'BER'}"
    )
    print("-" * 105)

    accuracies = []
    csv_results = []
    total_bit_errors = 0
    total_bits = 0

    with torch.no_grad():
        for img_path in image_paths:
            # Read image (returns [C, H, W] in [0, 255] uint8)
            img = read_image(img_path).float() / 255.0

            # Preserve aspect ratio, then take the same center crop used by validation.
            if img.shape[1] != image_size or img.shape[2] != image_size:
                img = resize(img, image_size, antialias=True)
                img = center_crop(img, [image_size, image_size])

            # Add batch dim [1, C, H, W]
            img = (img * 2.0 - 1.0).unsqueeze(0).to(device)

            # Decode
            logits = decoder(img)  # [1, L]
            bits = (torch.sigmoid(logits) > 0.5).float()  # [1, L]

            # Display
            bits_str = ''.join(str(int(b)) for b in bits[0].cpu())
            display_bits_str = bits_str[:32]
            fname = os.path.basename(img_path)

            if expected_bits is not None:
                expected_bits_device = expected_bits.to(device)
                bit_errors = (
                    bits[0] != expected_bits_device
                ).sum().item()
                bit_count = expected_bits_device.numel()
                ber = bit_errors / bit_count
                acc = 1.0 - ber
                accuracies.append(acc)
                total_bit_errors += bit_errors
                total_bits += bit_count
                csv_results.append(
                    (fname, expected_bits_str, bits_str, acc, ber)
                )
                print(
                    f"{fname:<40s} {display_bits_str:<35s} "
                    f"{acc:<10.4f} {ber:.4f}"
                )
            else:
                print(f"{fname:<40s} {display_bits_str}")

    # --- Summary ---
    if expected_bits is not None and accuracies:
        avg_acc = sum(accuracies) / len(accuracies)
        avg_ber = total_bit_errors / total_bits
        print("-" * 105)
        print(f"Average accuracy over {len(accuracies)} image(s): {avg_acc:.4f}")
        print(f"Average BER over {len(accuracies)} image(s): {avg_ber:.4f}")

        # Save CSV
        csv_path = os.path.join(args.input_dir, 'real_screen_results.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow([
                'image',
                'original_bits',
                'decoded_bits',
                'bit_accuracy',
                'ber',
            ])
            for fname, original_bits, decoded_bits, acc, ber in csv_results:
                writer.writerow([
                    fname,
                    original_bits,
                    decoded_bits,
                    f'{acc:.4f}',
                    f'{ber:.4f}',
                ])
            writer.writerow([
                'AVERAGE',
                '',
                '',
                f'{avg_acc:.4f}',
                f'{avg_ber:.4f}',
            ])
        print(f"[Eval] Results saved to: {csv_path}")


if __name__ == '__main__':
    main()
