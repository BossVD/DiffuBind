"""
Sample watermarked images using the trained WatermarkConditionedUNet.

Given a cover image and watermark bits, produces a watermarked image
via image-to-image diffusion (partial forward + full reverse).

Usage:
    D:\Anaconda_envs\envs\wadiff\python.exe sample_embed_watermark.py \
        --checkpoint checkpoints/best.pt \
        --input ./test_images/cover.png \
        --watermark "1010101011001010" \
        --output ./outputs/watermarked.png \
        --t_start 300

    # Non-recursive directory mode:
    python sample_embed_watermark.py \
        --checkpoint checkpoints/best.pt \
        --input ./test_images \
        --watermark "1010101011001010" \
        --output ./outputs/watermarked_batch \
        --t_start 300
"""
import os
import sys
import argparse
import csv
import random
import numpy as np
import torch
from PIL import Image
from kornia.metrics import ssim as kornia_ssim
from torchvision import transforms
from torchvision.utils import save_image


from guided_diffusion.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, ModelMeanType, ModelVarType, LossType

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
from NOISE_LAYER.build_noise_layer import build_noise_layer


SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp','.tif', '.tiff'}


def load_yaml_config(config_path):
    if not config_path:
        return None
    import yaml
    with open(config_path, 'r', encoding='utf-8-sig') as handle:
        return yaml.safe_load(handle)


def build_sample_noise_layer(cfg, noise_type, device):
    noise_type = str(noise_type).lower()
    if noise_type == 'none':
        return None
    if noise_type not in {'pimog', 'oled', 'led', 'projector', 'mixed'}:
        raise ValueError(f'Unsupported noise layer for sampling: {noise_type}')
    sample_cfg = dict(cfg)
    sample_cfg['noise_layer'] = dict(cfg.get('noise_layer', {}), type=noise_type)
    if noise_type in {'pimog', 'oled', 'led', 'projector'}:
        sample_cfg['noise_layer'][noise_type] = dict(
            cfg.get('noise_layer', {}).get(noise_type, {}),
            p=1.0,
        )
    layer = build_noise_layer(sample_cfg).to(device)
    layer.eval()
    return layer


def is_supported_image(path):
    return os.path.splitext(path)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS


def per_image_psnr(pred_01, target_01):
    """Compute RGB PSNR per image for tensors in [0, 1]."""
    mse = (pred_01 - target_01).pow(2).flatten(1).mean(dim=1)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
    return torch.where(mse <= 1e-10, torch.full_like(psnr, 100.0), psnr)


def per_image_ssim(pred_01, target_01):
    """Compute mean RGB SSIM per image for tensors in [0, 1]."""
    ssim_map = kornia_ssim(
        pred_01,
        target_01,
        window_size=11,
        max_val=1.0,
        padding='same',
    )
    return ssim_map.flatten(1).mean(dim=1)


def collect_input_images(input_path):
    """Return non-recursive input image paths and whether input is a directory."""
    if os.path.isfile(input_path):
        if not is_supported_image(input_path):
            supported = ', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
            raise ValueError(
                f'Unsupported input image extension: {input_path}. '
                f'Supported extensions: {supported}'
            )
        return [input_path], False

    if os.path.isdir(input_path):
        image_paths = [
            os.path.join(input_path, name)
            for name in sorted(os.listdir(input_path), key=str.lower)
            if os.path.isfile(os.path.join(input_path, name))
            and is_supported_image(name)
        ]
        if not image_paths:
            supported = ', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
            raise ValueError(
                f'No supported images found directly in directory: {input_path}. '
                f'Supported extensions: {supported}'
            )
        return image_paths, True

    raise FileNotFoundError(f'Input path does not exist: {input_path}')


def make_unique_batch_output_path(output_dir, input_path, used_names):
    """Create a stable, lossless output name without overwriting another input."""
    stem = os.path.splitext(os.path.basename(input_path))[0]
    candidate = f'{stem}_watermarked.png'
    candidate_key = candidate.lower()

    if candidate_key in used_names:
        source_ext = os.path.splitext(input_path)[1].lower().lstrip('.') or 'image'
        candidate = f'{stem}_{source_ext}_watermarked.png'
        candidate_key = candidate.lower()

    suffix = 2
    base_candidate = candidate
    while candidate_key in used_names:
        base_stem, base_ext = os.path.splitext(base_candidate)
        candidate = f'{base_stem}_{suffix}{base_ext}'
        candidate_key = candidate.lower()
        suffix += 1

    used_names.add(candidate_key)
    return os.path.join(output_dir, candidate)


def resolve_input_output_pairs(input_path, output_path):
    """Resolve single-image or non-recursive directory processing paths."""
    input_paths, input_is_directory = collect_input_images(input_path)

    if input_is_directory:
        if is_supported_image(output_path):
            raise ValueError(
                'When --input is a directory, --output must be an output '
                'directory, not an image file.'
            )
        if os.path.normcase(os.path.abspath(input_path)) == os.path.normcase(
            os.path.abspath(output_path)
        ):
            raise ValueError(
                'Input and output directories must be different in directory mode.'
            )

        used_names = set()
        pairs = [
            (
                path,
                make_unique_batch_output_path(output_path, path, used_names),
            )
            for path in input_paths
        ]
        return pairs, True, output_path

    if os.path.isdir(output_path) or not is_supported_image(output_path):
        output_file = make_unique_batch_output_path(output_path, input_paths[0], set())
        return [(input_paths[0], output_file)], False, output_path

    output_dir = os.path.dirname(output_path) or '.'
    return [(input_paths[0], output_path)], False, output_dir


def embed_watermark_sample(diffusion, model, cover_img, wm_bits, t_start=300,
                           region_guidance_config=None,
                           residual_constraint_config=None):
    """
    Image-to-image watermark embedding via partial DDPM reverse sampling.

    Args:
        diffusion: GaussianDiffusion instance
        model: WatermarkConditionedUNet
        cover_img: [1, 3, H, W] in [-1, 1]
        wm_bits:  [1, wm_len] 0/1 float
        t_start:  timestep to start reverse from

    Returns:
        watermarked: [1, 3, H, W] in [-1, 1]
    """
    device = cover_img.device
    B = cover_img.size(0)
    residual_settings = get_residual_constraint_settings(
        residual_constraint_config
    )
    needs_guidance = (
        residual_settings['enabled']
        or bool(getattr(model, 'use_content_gated_wm_map', False))
    )
    allowance = (
        build_edge_texture_guidance(
            (cover_img + 1.0) / 2.0,
            region_guidance_config,
        )[0]
        if needs_guidance else None
    )

    def constrain_xstart(x_start):
        return constrain_watermark_residual(
            x_start,
            cover_img,
            allowance,
            residual_constraint_config,
        )

    # Forward diffuse to t_start
    t = torch.full((B,), t_start - 1, device=device, dtype=torch.long)
    noise = torch.randn_like(cover_img)
    x_t = diffusion.q_sample(cover_img, t, noise=noise)

    # Reverse denoise
    for step in reversed(range(t_start)):
        t_batch = torch.full((B,), step, device=device, dtype=torch.long)
        t_scaled = t_batch.float() * (1000.0 / diffusion.num_timesteps)

        pred_noise = model(
            x_t=x_t,
            t=t_scaled,
            cover_img=cover_img,
            wm_bits=wm_bits,
            content_mask=allowance,
        )

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
        noise_term = torch.randn_like(x_t) if step > 0 else torch.zeros_like(x_t)
        x_t = mean + torch.exp(0.5 * log_variance) * noise_term

    return constrain_xstart(x_t)


def process_cover_image(
    input_path,
    output_path,
    transform,
    diffusion,
    model,
    decoder,
    wm_bits,
    t_start,
    requested_noise_type,
    noise_layer,
    save_degraded,
    degradation_layers,
    device,
    config,
):
    """Embed and verify one image, then save its requested artifacts."""
    print(f"[Sample] Loading cover image: {input_path}")
    with Image.open(input_path) as cover_img:
        cover_tensor = (
            transform(cover_img.convert("RGB")).unsqueeze(0).to(device)
        )

    print(f"[Sample] Embedding watermark (t_start={t_start})...")
    with torch.no_grad():
        watermarked = embed_watermark_sample(
            diffusion,
            model,
            cover_tensor,
            wm_bits,
            t_start=t_start,
            region_guidance_config=config.get('train', {}).get(
                'region_guidance', {}
            ),
            residual_constraint_config=config.get('train', {}).get(
                'residual_constraint', {}
            ),
        )

        watermarked_01 = (watermarked + 1.0) / 2.0
        cover_01 = (cover_tensor + 1.0) / 2.0
        psnr_db = per_image_psnr(watermarked_01, cover_01)[0].item()
        ssim = per_image_ssim(watermarked_01, cover_01)[0].item()
        logits_clean = decoder(watermarked)
        pred_bits_clean = (torch.sigmoid(logits_clean) > 0.5).float()
        bit_acc_clean = (pred_bits_clean == wm_bits).float().mean().item()

        selected_noise_label = requested_noise_type
        if noise_layer is not None:
            degraded_01 = noise_layer(watermarked_01).float().clamp(0.0, 1.0)
            if requested_noise_type == 'mixed' and hasattr(
                noise_layer, 'get_last_name'
            ):
                selected_noise_label = f"mixed:{noise_layer.get_last_name()}"
            degraded = degraded_01.mul(2.0).sub(1.0)
            logits_degraded = decoder(degraded)
            pred_bits_degraded = (
                torch.sigmoid(logits_degraded) > 0.5
            ).float()
            bit_acc_degraded = (
                (pred_bits_degraded == wm_bits).float().mean().item()
            )
        else:
            degraded_01 = watermarked_01
            pred_bits_degraded = None
            bit_acc_degraded = None

    print(f"[Sample] bit_acc_clean={bit_acc_clean:.4f}")
    print(f"[Sample] PSNR={psnr_db:.4f} dB SSIM={ssim:.6f}")
    if bit_acc_degraded is None:
        print(
            "[Sample] bit_acc_degraded=N/A "
            "noise_layer=none (not evaluated)"
        )
    else:
        print(
            f"[Sample] bit_acc_degraded={bit_acc_degraded:.4f} "
            f"noise_layer={requested_noise_type}"
        )
    if selected_noise_label != requested_noise_type:
        print(f"[Sample] mixed selected layer: {selected_noise_label}")
    print(
        "[Sample] Recovered bits clean: "
        f"{''.join(str(int(b)) for b in pred_bits_clean[0].tolist())}"
    )
    if pred_bits_degraded is not None:
        print(
            "[Sample] Recovered bits degraded: "
            f"{''.join(str(int(b)) for b in pred_bits_degraded[0].tolist())}"
        )

    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    save_image(watermarked_01[0], output_path)

    comparison_dir = os.path.join(output_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    comparison = torch.cat([cover_01, watermarked_01], dim=0)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    comparison_path = os.path.join(
        comparison_dir, f'{base_name}_comparison.png'
    )
    save_image(comparison, comparison_path, nrow=1)

    if save_degraded:
        degraded_dir = os.path.join(output_dir, 'degraded')
        os.makedirs(degraded_dir, exist_ok=True)
        file_noise_label = selected_noise_label.replace(':', '_')
        degraded_path = os.path.join(
            degraded_dir, f'{base_name}_degraded_{file_noise_label}.png'
        )
        save_image(degraded_01[0], degraded_path)

        residual_01 = (watermarked_01 - cover_01).abs().clamp(0.0, 1.0)
        grid = torch.cat(
            [cover_01, watermarked_01, degraded_01, residual_01], dim=0
        )
        grid_path = os.path.join(degraded_dir, f'{base_name}_grid.png')
        save_image(grid, grid_path, nrow=4)
        print(f"[Sample] Degraded image saved to: {degraded_path}")
        print(f"[Sample] Grid saved to: {grid_path}")

        with torch.no_grad():
            for noise_type, layer in degradation_layers.items():
                fixed_degraded_01 = (
                    layer(watermarked_01).float().clamp(0.0, 1.0)
                )
                fixed_degraded_path = os.path.join(
                    degraded_dir, f'{base_name}_{noise_type}.png'
                )
                save_image(fixed_degraded_01[0], fixed_degraded_path)
                print(
                    f"[Sample] {noise_type} degraded image saved to: "
                    f"{fixed_degraded_path}"
                )

    print(f"[Sample] Watermarked image saved to: {output_path}")
    print(f"[Sample] Comparison saved to: {comparison_path}")
    return {
        'input_image': input_path,
        'output_image': output_path,
        'bit_acc_clean': bit_acc_clean,
        'bit_acc_degraded': bit_acc_degraded,
        'psnr_db': psnr_db,
        'ssim': ssim,
        'status': 'ok',
        'error': '',
    }


def main():
    parser = argparse.ArgumentParser(
        description='Embed watermark into one cover image or a directory of images'
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Optional YAML config path; overrides checkpoint config')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (.pt)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to one cover image or a directory (non-recursive)')
    parser.add_argument('--watermark', type=str, default=None,
                        help='Watermark bits as binary string, e.g. "10101010". If None, random.')
    parser.add_argument('--watermark_length', type=int, default=30,
                        help='Number of watermark bits (used if --watermark not provided)')
    parser.add_argument('--output', type=str, default='./outputs/watermarked.png',
                        help='Output image path for single input, or output directory for directory input')
    parser.add_argument('--t_start', type=int, default=300,
                        help='Timestep to start reverse from (controls edit strength)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed; defaults to the checkpoint training seed')
    parser.add_argument('--noise_layer', type=str, default='none',
                        help='Noise layer for degraded verification: none, pimog, oled, led, projector, mixed')
    parser.add_argument('--save_degraded', action='store_true',
                        help='Also save fixed degradation variants of the watermarked image')
    parser.add_argument('--degradation_types', type=str, default='pimog,oled,led,projector',
                        help='Comma-separated degradation types used with --save_degraded')
    args = parser.parse_args()

    input_output_pairs, batch_mode, batch_output_dir = (
        resolve_input_output_pairs(args.input, args.output)
    )
    if batch_mode:
        print(
            f"[Sample] Directory mode: found {len(input_output_pairs)} image(s) "
            "(non-recursive)"
        )
        print(f"[Sample] Batch output directory: {batch_output_dir}")
    else:
        print("[Sample] Single-image mode")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Sample] Using device: {device}")

    # --- Load checkpoint ---
    print(f"[Sample] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    cfg = checkpoint.get('config', {})
    config_override = load_yaml_config(args.config)
    if config_override is not None:
        cfg = config_override

    seed = args.seed if args.seed is not None else cfg.get('train', {}).get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Sample] Random seed: {seed}")

    # --- Determine watermark length (ALWAYS from checkpoint config) ---
    if cfg and 'data' in cfg and 'watermark_length' in cfg['data']:
        watermark_length = cfg['data']['watermark_length']
    elif args.watermark is not None:
        watermark_length = len(args.watermark)
    else:
        watermark_length = args.watermark_length
    print(f"[Sample] Model trained with watermark_length={watermark_length}")

    image_size = cfg.get('data', {}).get('image_size', 128)
    base_channels = cfg.get('model', {}).get('base_channels', 64)
    cond_dim = cfg.get('model', {}).get('cond_dim', 256)
    timesteps = cfg.get('diffusion', {}).get('timesteps', 1000)
    beta_schedule = cfg.get('diffusion', {}).get('beta_schedule', 'linear')

    print(f"[Sample] image_size={image_size}, watermark_length={watermark_length}")

    # --- Create diffusion ---
    betas = get_named_beta_schedule(beta_schedule, timesteps)
    diffusion = GaussianDiffusion(
        betas=torch.tensor(betas, dtype=torch.float32),
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )

    # --- Create model ---
    model = WatermarkConditionedUNet(
        image_size=image_size,
        base_channels=base_channels,
        cond_dim=cond_dim,
        watermark_length=watermark_length,
        use_pretrained_unet=False,
        pretrained_path=None,
        use_watermark_time_emb=cfg.get('model', {}).get('use_watermark_time_emb', True),
        use_watermark_spatial_map=cfg.get('model', {}).get('use_watermark_spatial_map', True),
        wm_map_channels=cfg.get('model', {}).get('wm_map_channels', 4),
        wm_map_size=cfg.get('model', {}).get('wm_map_size', 16),
        wm_time_scale=cfg.get('model', {}).get('wm_time_scale', 1.0),
        wm_map_scale=cfg.get('model', {}).get('wm_map_scale', 1.0),
        use_content_gated_wm_map=cfg.get('model', {}).get(
            'use_content_gated_wm_map', False
        ),
        wm_map_flat_floor=cfg.get('model', {}).get('wm_map_flat_floor', 0.2),
    ).to(device)

    # Load weights
    if 'diffusion_model' in checkpoint:
        model.load_state_dict(checkpoint['diffusion_model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("[Sample] Model loaded.")

    # --- Create decoder for verification ---
    decoder = build_watermark_decoder(
        cfg,
        watermark_length=watermark_length,
    ).to(device)
    if 'decoder' in checkpoint:
        missing, unexpected, mismatched = load_watermark_decoder_state(
            decoder, checkpoint['decoder']
        )
        if missing or unexpected or mismatched:
            print(
                "[Sample] Decoder checkpoint partially loaded "
                "(architecture may have changed)."
            )
    decoder.eval()

    # --- Build preprocessing and optional degradation layers once ---
    transform = transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # --- Create watermark bits ---
    if args.watermark is not None:
        if any(bit not in {'0', '1'} for bit in args.watermark):
            raise ValueError('--watermark must contain only 0 and 1')
        wm_bits = torch.tensor([float(b) for b in args.watermark], device=device).unsqueeze(0)
        # Pad or truncate to match expected length
        if wm_bits.size(1) < watermark_length:
            pad = torch.zeros(1, watermark_length - wm_bits.size(1), device=device)
            wm_bits = torch.cat([wm_bits, pad], dim=1)
        elif wm_bits.size(1) > watermark_length:
            wm_bits = wm_bits[:, :watermark_length]
        print(f"[Sample] Watermark bits: {args.watermark}")
    else:
        wm_bits = torch.randint(0, 2, (1, watermark_length), device=device, dtype=torch.float32)
        print(f"[Sample] Random watermark: {''.join(str(int(b)) for b in wm_bits[0].tolist())}")

    requested_noise_type = str(args.noise_layer).lower()
    noise_layer = build_sample_noise_layer(cfg, requested_noise_type, device)
    degradation_layers = {}
    if args.save_degraded:
        degradation_types = [
            item.strip().lower()
            for item in args.degradation_types.split(',')
            if item.strip()
        ]
        for noise_type in degradation_types:
            if noise_type not in {'pimog', 'oled', 'led', 'projector'}:
                raise ValueError(f'Unsupported degradation type for sampling: {noise_type}')
            degradation_layers[noise_type] = build_sample_noise_layer(
                cfg, noise_type, device
            )

    # --- Process one image or a non-recursive directory ---
    results = []
    total_images = len(input_output_pairs)
    for index, (input_path, output_path) in enumerate(input_output_pairs, start=1):
        print("=" * 80)
        print(f"[Sample] Processing image {index}/{total_images}: {input_path}")
        try:
            result = process_cover_image(
                input_path=input_path,
                output_path=output_path,
                transform=transform,
                diffusion=diffusion,
                model=model,
                decoder=decoder,
                wm_bits=wm_bits,
                t_start=args.t_start,
                requested_noise_type=requested_noise_type,
                noise_layer=noise_layer,
                save_degraded=args.save_degraded,
                degradation_layers=degradation_layers,
                device=device,
                config=cfg,
            )
        except Exception as exc:
            if not batch_mode:
                raise
            print(f"[Sample] ERROR processing {input_path}: {exc}")
            result = {
                'input_image': input_path,
                'output_image': output_path,
                'bit_acc_clean': '',
                'bit_acc_degraded': '',
                'psnr_db': '',
                'ssim': '',
                'status': 'error',
                'error': str(exc),
            }
        results.append(result)

    if batch_mode:
        os.makedirs(batch_output_dir, exist_ok=True)
        csv_path = os.path.join(batch_output_dir, 'batch_embed_results.csv')
        fieldnames = [
            'input_image',
            'output_image',
            'bit_acc_clean',
            'bit_acc_degraded',
            'psnr_db',
            'ssim',
            'status',
            'error',
        ]
        successful = [item for item in results if item['status'] == 'ok']
        failed = [item for item in results if item['status'] != 'ok']
        mean_clean_acc = (
            sum(item['bit_acc_clean'] for item in successful) / len(successful)
            if successful
            else 0.0
        )
        degraded_scores = [
            item['bit_acc_degraded']
            for item in successful
            if item['bit_acc_degraded'] is not None
        ]
        mean_degraded_acc = (
            sum(degraded_scores) / len(degraded_scores)
            if degraded_scores
            else None
        )
        mean_psnr_db = (
            sum(item['psnr_db'] for item in successful) / len(successful)
            if successful
            else 0.0
        )
        mean_ssim = (
            sum(item['ssim'] for item in successful) / len(successful)
            if successful
            else 0.0
        )

        average_row = {
            'input_image': 'AVERAGE',
            'output_image': '',
            'bit_acc_clean': (
                f'{mean_clean_acc:.6f}' if successful else ''
            ),
            'bit_acc_degraded': (
                f'{mean_degraded_acc:.6f}'
                if mean_degraded_acc is not None
                else ''
            ),
            'psnr_db': f'{mean_psnr_db:.6f}' if successful else '',
            'ssim': f'{mean_ssim:.6f}' if successful else '',
            'status': 'summary',
            'error': '',
        }
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            writer.writerow(average_row)

        print("=" * 80)
        print(
            f"[Sample] Batch complete: total={len(results)} "
            f"success={len(successful)} failed={len(failed)}"
        )
        print(f"[Sample] Mean clean bit accuracy: {mean_clean_acc:.4f}")
        print(f"[Sample] Mean PSNR: {mean_psnr_db:.4f} dB")
        print(f"[Sample] Mean SSIM: {mean_ssim:.6f}")
        if mean_degraded_acc is None:
            print(
                "[Sample] Mean degraded bit accuracy: "
                "N/A (noise_layer=none)"
            )
        else:
            print(
                f"[Sample] Mean degraded bit accuracy: "
                f"{mean_degraded_acc:.4f}"
            )
        print(f"[Sample] Batch results saved to: {csv_path}")
        if failed:
            sys.exit(1)


if __name__ == '__main__':
    main()
