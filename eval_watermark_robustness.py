r"""
Evaluate watermark robustness on a fixed validation subset.

The diffusion model and decoder use [-1, 1]. Degradation layers and image
quality metrics use [0, 1].

Linux example:
    /root/miniconda3/envs/wadiff/bin/python eval_watermark_robustness.py \
        --checkpoint checkpoints_stage2_mixed_v3/final.pt \
        --config configs/watermark_stage2_mixed_v3.yaml \
        --noise_layers clean,pimog,oled,led,projector,mixed \
        --t_start 200 \
        --batch_size 8 \
        --seed 42 \
        --num_eval_images 500 \
        --subset_seed 42 \
        --num_visual_samples 8 \
        --noise_strength 1.0 \
        --output ./outputs_stage2_mixed_v3/eval_results_500.csv
"""

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from kornia.metrics import ssim as kornia_ssim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image

from guided_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)

from dataset.watermark_image_dataset import WatermarkImageDataset
from models.watermark_decoder import (
    build_watermark_decoder,
    load_watermark_decoder_state,
)
from models.watermark_unet import WatermarkConditionedUNet
from models.watermark_residual import (
    build_edge_texture_guidance,
    constrain_watermark_residual,
    get_residual_constraint_settings,
)
from NOISE_LAYER import build_noise_layer


VALID_NOISE_LAYERS = ('clean', 'pimog', 'oled', 'led', 'projector', 'mixed')
VISUAL_NOISE_ORDER = ('pimog', 'oled', 'led', 'projector', 'mixed')
PER_IMAGE_FIELDS = (
    'dataset_index',
    'noise_type',
    'repeat_index',
    'mixed_active_type',
    'noise_strength',
    'bit_acc',
    'ber',
    'message_success',
    'watermark_psnr',
    'watermark_ssim',
    'watermark_l1',
    'watermark_lpips',
    'attack_psnr',
    'attack_ssim',
    'attack_l1',
    'end2end_psnr',
    'end2end_ssim',
    'end2end_l1',
)


class IndexedDataset(Dataset):
    """Expose stable source indices while optionally evaluating a subset."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        dataset_index = self.indices[item]
        sample = dict(self.dataset[dataset_index])
        sample['_dataset_index'] = dataset_index
        return sample


def load_yaml_config(config_path):
    if not config_path:
        return None
    import yaml

    with open(config_path, 'r', encoding='utf-8-sig') as handle:
        return yaml.safe_load(handle)


def parse_noise_layers(value):
    layers = [item.strip().lower() for item in value.split(',') if item.strip()]
    invalid = [layer for layer in layers if layer not in VALID_NOISE_LAYERS]
    if invalid:
        raise ValueError(f'Unsupported noise layer(s): {invalid}')
    if not layers:
        raise ValueError('At least one noise layer must be requested')
    if len(set(layers)) != len(layers):
        raise ValueError(f'Noise layers must not contain duplicates: {layers}')
    return layers


def build_eval_noise_layer(cfg, noise_type, device):
    eval_cfg = dict(cfg)
    eval_cfg['noise_layer'] = dict(cfg.get('noise_layer', {}), type=noise_type)
    if noise_type in {'pimog', 'oled', 'led', 'projector'}:
        eval_cfg['noise_layer'][noise_type] = dict(
            cfg.get('noise_layer', {}).get(noise_type, {}),
            p=1.0,
        )
    simulator = build_noise_layer(eval_cfg).to(device)
    simulator.eval()
    return simulator


def embed_watermark_eval(diffusion, model, cover_img, wm_bits, t_start=300,
                         region_guidance_config=None,
                         residual_constraint_config=None):
    """Generate a watermarked image with the full t_start-step DDPM trajectory."""
    device = cover_img.device
    batch_size = cover_img.size(0)
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

    t = torch.full(
        (batch_size,),
        t_start - 1,
        device=device,
        dtype=torch.long,
    )
    noise = torch.randn_like(cover_img)
    x_t = diffusion.q_sample(cover_img, t, noise=noise)

    for step in reversed(range(t_start)):
        t_batch = torch.full(
            (batch_size,),
            step,
            device=device,
            dtype=torch.long,
        )
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
        noise_term = (
            torch.randn_like(x_t)
            if step > 0
            else torch.zeros_like(x_t)
        )
        x_t = (
            out['mean']
            + torch.exp(0.5 * out['log_variance']) * noise_term
        )

    return constrain_xstart(x_t)


def to_01(image_m11):
    return image_m11.add(1.0).mul(0.5).clamp(0.0, 1.0)


def to_m11(image_01):
    return image_01.mul(2.0).sub(1.0)


def apply_degradation_with_strength(source_01, simulator, strength):
    """Apply a full degradation and blend it with source_01 in [0, 1]."""
    full_degraded_01 = simulator(source_01).float()
    degraded_01 = source_01 + float(strength) * (
        full_degraded_01 - source_01
    )
    return degraded_01.clamp(0.0, 1.0)


def per_image_l1(pred_01, target_01):
    return (pred_01 - target_01).abs().flatten(1).mean(dim=1)


def per_image_psnr(pred_01, target_01):
    mse = (pred_01 - target_01).pow(2).flatten(1).mean(dim=1)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
    return torch.where(mse <= 1e-10, torch.full_like(psnr, 100.0), psnr)


def per_image_ssim(pred_01, target_01):
    ssim_map = kornia_ssim(
        pred_01,
        target_01,
        window_size=11,
        max_val=1.0,
        padding='same',
    )
    return ssim_map.flatten(1).mean(dim=1)


def select_eval_indices(dataset_size, num_eval_images, subset_seed):
    if num_eval_images < 0:
        raise ValueError('--num_eval_images must be >= 0')
    if num_eval_images == 0 or num_eval_images >= dataset_size:
        return list(range(dataset_size))
    generator = torch.Generator()
    generator.manual_seed(int(subset_seed))
    return torch.randperm(dataset_size, generator=generator)[:num_eval_images].tolist()


def percentile(values, q):
    if not values:
        return float('nan')
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def safe_mean(values):
    if not values:
        return float('nan')
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def format_duration(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        return '--:--:--'
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def output_paths(output_path):
    root, extension = os.path.splitext(output_path)
    if extension.lower() != '.csv':
        root = output_path
        output_path = f'{output_path}.csv'
    output_dir = os.path.dirname(output_path) or '.'
    stem = os.path.basename(root)
    return {
        'summary': output_path,
        'by_noise': f'{root}_by_noise.csv',
        'per_image': f'{root}_per_image.csv',
        'metadata': f'{root}_metadata.json',
        'indices': f'{root}_indices.txt',
        'sample_dir': os.path.join(output_dir, f'{stem}_samples'),
    }


def write_indices(path, indices):
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        for index in indices:
            handle.write(f'{int(index)}\n')


def aggregate_results(per_image_rows, watermark_rows, requested_layers):
    by_noise_rows = []
    summary = {}

    watermark_psnr = [row['watermark_psnr'] for row in watermark_rows]
    watermark_ssim = [row['watermark_ssim'] for row in watermark_rows]
    watermark_l1 = [row['watermark_l1'] for row in watermark_rows]
    watermark_lpips = [
        row['watermark_lpips']
        for row in watermark_rows
        if row['watermark_lpips'] != ''
    ]

    summary.update({
        'watermark_psnr_mean': safe_mean(watermark_psnr),
        'watermark_psnr_p5': percentile(watermark_psnr, 5),
        'watermark_psnr_min': min(watermark_psnr, default=float('nan')),
        'watermark_ssim_mean': safe_mean(watermark_ssim),
        'watermark_ssim_p5': percentile(watermark_ssim, 5),
        'watermark_ssim_min': min(watermark_ssim, default=float('nan')),
        'watermark_l1_mean': safe_mean(watermark_l1),
    })
    if watermark_lpips:
        summary.update({
            'watermark_lpips_mean': safe_mean(watermark_lpips),
            'watermark_lpips_p95': percentile(watermark_lpips, 95),
            'watermark_lpips_max': max(watermark_lpips),
        })

    for noise_type in requested_layers:
        rows = [
            row for row in per_image_rows
            if row['noise_type'] == noise_type
        ]
        if not rows:
            continue

        bit_acc = [row['bit_acc'] for row in rows]
        ber = [row['ber'] for row in rows]
        message_success = [row['message_success'] for row in rows]
        attack_psnr = [row['attack_psnr'] for row in rows]
        attack_ssim = [row['attack_ssim'] for row in rows]
        attack_l1 = [row['attack_l1'] for row in rows]
        end2end_psnr = [row['end2end_psnr'] for row in rows]
        end2end_ssim = [row['end2end_ssim'] for row in rows]
        end2end_l1 = [row['end2end_l1'] for row in rows]

        aggregate = {
            'noise_type': noise_type,
            'num_images': len({row['dataset_index'] for row in rows}),
            'num_observations': len(rows),
            'noise_strength': rows[0]['noise_strength'],
            'bit_acc_mean': safe_mean(bit_acc),
            'bit_acc_p5': percentile(bit_acc, 5),
            'bit_acc_min': min(bit_acc),
            'ber_mean': safe_mean(ber),
            'message_success_rate': safe_mean(message_success),
            'attack_psnr_mean': safe_mean(attack_psnr),
            'attack_ssim_mean': safe_mean(attack_ssim),
            'attack_l1_mean': safe_mean(attack_l1),
            'end2end_psnr_mean': safe_mean(end2end_psnr),
            'end2end_ssim_mean': safe_mean(end2end_ssim),
            'end2end_l1_mean': safe_mean(end2end_l1),
        }
        by_noise_rows.append(aggregate)

        for key, value in aggregate.items():
            if key in {'noise_type', 'num_images', 'num_observations'}:
                continue
            summary[f'{key}_{noise_type}'] = value
        summary[f'num_images_{noise_type}'] = aggregate['num_images']
        summary[f'num_observations_{noise_type}'] = aggregate['num_observations']

    return summary, by_noise_rows


def write_result_files(
    paths,
    per_image_rows,
    watermark_rows,
    requested_layers,
    metadata,
):
    summary, by_noise_rows = aggregate_results(
        per_image_rows,
        watermark_rows,
        requested_layers,
    )

    summary_prefix = {
        'status': metadata['status'],
        'dataset_size': metadata['dataset_size'],
        'evaluated_images': metadata['processed_images'],
        'requested_eval_images': metadata['requested_eval_images'],
        't_start': metadata['t_start'],
        'batch_size': metadata['batch_size'],
        'seed': metadata['seed'],
        'subset_seed': metadata['subset_seed'],
        'noise_strength': metadata['noise_strength'],
        'attack_repeats': metadata['attack_repeats'],
    }

    with open(paths['summary'], 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'value'])
        for key, value in {**summary_prefix, **summary}.items():
            writer.writerow([key, value])

    by_noise_fields = (
        'noise_type',
        'num_images',
        'num_observations',
        'noise_strength',
        'bit_acc_mean',
        'bit_acc_p5',
        'bit_acc_min',
        'ber_mean',
        'message_success_rate',
        'attack_psnr_mean',
        'attack_ssim_mean',
        'attack_l1_mean',
        'end2end_psnr_mean',
        'end2end_ssim_mean',
        'end2end_l1_mean',
    )
    with open(paths['by_noise'], 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=by_noise_fields)
        writer.writeheader()
        writer.writerows(by_noise_rows)

    with open(paths['per_image'], 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_IMAGE_FIELDS)
        writer.writeheader()
        writer.writerows(per_image_rows)

    with open(paths['metadata'], 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return summary, by_noise_rows


def tensor_to_pil(image_01):
    return to_pil_image(image_01.detach().cpu().clamp(0.0, 1.0))


def make_text_lines(title, metrics=None):
    lines = [title]
    if metrics:
        lines.extend(metrics)
    return lines


def save_comparison_grid(
    path,
    record,
    requested_layers,
    save_individual_samples=False,
):
    cover = record['cover']
    watermarked = record['watermarked']
    residual = ((watermarked - cover) * 5.0 + 0.5).clamp(0.0, 1.0)
    wm_metrics = record['watermark_metrics']

    panels = [
        (
            'Cover',
            cover,
            [f"Index={record['dataset_index']}"],
        ),
        (
            'Watermarked',
            watermarked,
            [
                f"PSNR={wm_metrics['psnr']:.2f} SSIM={wm_metrics['ssim']:.4f}",
                (
                    f"ACC={record['clean_acc']:.4f}"
                    if record.get('clean_acc') is not None
                    else 'ACC=not evaluated'
                ),
            ],
        ),
    ]

    for noise_type in VISUAL_NOISE_ORDER:
        if noise_type not in requested_layers:
            continue
        panel = record['degradations'].get(noise_type)
        if panel is None:
            continue
        title = noise_type.upper()
        if noise_type == 'mixed' and panel['mixed_active_type']:
            title = f"MIXED: {panel['mixed_active_type'].upper()}"
        panels.append((
            title,
            panel['image'],
            [
                f"ACC={panel['bit_acc']:.4f} BER={panel['ber']:.4f}",
                f"Attack PSNR={panel['attack_psnr']:.2f}",
            ],
        ))

    panels.append(('Signed residual x5', residual, []))

    tile_image = tensor_to_pil(cover)
    image_width, image_height = tile_image.size
    label_height = 42
    tile_height = label_height + image_height
    columns = min(4, max(1, len(panels)))
    rows = int(math.ceil(len(panels) / columns))
    canvas = Image.new(
        'RGB',
        (columns * image_width, rows * tile_height),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for panel_index, (title, image_tensor, metrics) in enumerate(panels):
        row, column = divmod(panel_index, columns)
        x = column * image_width
        y = row * tile_height
        draw.rectangle(
            (x, y, x + image_width - 1, y + tile_height - 1),
            outline=(180, 180, 180),
        )
        for line_index, line in enumerate(make_text_lines(title, metrics)):
            draw.text(
                (x + 4, y + 2 + line_index * 12),
                line,
                fill=(20, 20, 20),
                font=font,
            )
        canvas.paste(tensor_to_pil(image_tensor), (x, y + label_height))

    canvas.save(path)

    if not save_individual_samples:
        return

    root, _ = os.path.splitext(path)
    tensor_to_pil(cover).save(f'{root}_cover.png')
    tensor_to_pil(watermarked).save(f'{root}_watermarked.png')
    tensor_to_pil(residual).save(f'{root}_residual_x5.png')
    for noise_type, panel in record['degradations'].items():
        tensor_to_pil(panel['image']).save(
            f'{root}_degraded_{noise_type}.png'
        )


def print_progress(
    processed_images,
    total_images,
    batch_index,
    total_batches,
    start_time,
    running_accuracy,
):
    elapsed = max(time.time() - start_time, 1e-6)
    rate = processed_images / elapsed
    eta = (
        (total_images - processed_images) / rate
        if rate > 0
        else float('inf')
    )
    print(
        f'[Eval] {processed_images}/{total_images} images | '
        f'batches={batch_index}/{total_batches} | '
        f'elapsed={format_duration(elapsed)} | '
        f'ETA={format_duration(eta)}'
    )
    if running_accuracy:
        values = []
        for noise_type, stats in running_accuracy.items():
            if stats['count'] > 0:
                values.append(
                    f"{noise_type}={stats['sum'] / stats['count']:.4f}"
                )
        if values:
            print(f"[Running] {'  '.join(values)}")


def validate_args(args, timesteps):
    if args.batch_size <= 0:
        raise ValueError('--batch_size must be positive')
    if args.t_start <= 0 or args.t_start > timesteps:
        raise ValueError(
            f'--t_start must be in [1, {timesteps}], got {args.t_start}'
        )
    if args.num_visual_samples < 0:
        raise ValueError('--num_visual_samples must be >= 0')
    if args.attack_repeats <= 0:
        raise ValueError('--attack_repeats must be positive')
    if not 0.0 <= args.noise_strength <= 1.0:
        raise ValueError('--noise_strength must be in [0, 1]')
    if args.progress_interval <= 0:
        raise ValueError('--progress_interval must be positive')


def build_lpips_model(enabled, device):
    if not enabled:
        return None
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError(
            'LPIPS was requested but the optional "lpips" package is not '
            'installed. Install it in the WaDiff environment or omit '
            '--enable_lpips.'
        ) from exc
    model = lpips.LPIPS(net='alex').to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate watermark robustness on a fixed validation subset'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (.pt)',
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Optional YAML config path; replaces the checkpoint config',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Validation directory; defaults to config data.val_dir',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./outputs/eval_results.csv',
        help='Summary CSV path',
    )
    parser.add_argument(
        '--noise_layers',
        type=str,
        default='clean,pimog,oled,led,projector,mixed',
        help='Comma-separated layers to evaluate',
    )
    parser.add_argument(
        '--t_start',
        type=int,
        default=300,
        help='DDPM start timestep and number of reverse steps',
    )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Sampling/attack seed; defaults to checkpoint training seed',
    )
    parser.add_argument(
        '--num_eval_images',
        type=int,
        default=0,
        help='Fixed subset size; 0 evaluates the full validation set',
    )
    parser.add_argument(
        '--subset_seed',
        type=int,
        default=42,
        help='Seed used only to select the validation subset',
    )
    parser.add_argument(
        '--num_visual_samples',
        type=int,
        default=8,
        help='Number of combined comparison grids to save; 0 disables images',
    )
    parser.add_argument(
        '--save_individual_samples',
        action='store_true',
        help='Also save the individual images used in each comparison grid',
    )
    parser.add_argument(
        '--noise_strength',
        type=float,
        default=1.0,
        help='Blend strength in [0,1]; 1.0 applies the full degradation',
    )
    parser.add_argument(
        '--attack_repeats',
        type=int,
        default=1,
        help='Random degradation repeats per watermarked image and noise type',
    )
    parser.add_argument(
        '--enable_lpips',
        action='store_true',
        help='Compute optional cover-watermarked LPIPS',
    )
    parser.add_argument(
        '--progress_interval',
        type=int,
        default=10,
        help='Print progress every N batches',
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu'
    )
    print(f'[Eval] Using device: {device}')

    print(f'[Eval] Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    cfg = checkpoint.get('config', {})
    config_override = load_yaml_config(args.config)
    if config_override is not None:
        cfg = config_override

    seed = (
        args.seed
        if args.seed is not None
        else cfg.get('train', {}).get('seed', 42)
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f'[Eval] Random seed: {seed}')

    image_size = cfg.get('data', {}).get('image_size', 128)
    watermark_length = cfg.get('data', {}).get('watermark_length', 30)
    base_channels = cfg.get('model', {}).get('base_channels', 64)
    cond_dim = cfg.get('model', {}).get('cond_dim', 256)
    timesteps = cfg.get('diffusion', {}).get('timesteps', 1000)
    beta_schedule = cfg.get('diffusion', {}).get(
        'beta_schedule',
        'linear',
    )
    validate_args(args, timesteps)

    print(
        f'[Eval] image_size={image_size}, '
        f'watermark_length={watermark_length}'
    )

    betas = get_named_beta_schedule(beta_schedule, timesteps)
    diffusion = GaussianDiffusion(
        betas=torch.tensor(betas, dtype=torch.float32),
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )

    model = WatermarkConditionedUNet(
        image_size=image_size,
        base_channels=base_channels,
        cond_dim=cond_dim,
        watermark_length=watermark_length,
        use_watermark_time_emb=cfg.get(
            'model', {}
        ).get('use_watermark_time_emb', True),
        use_watermark_spatial_map=cfg.get(
            'model', {}
        ).get('use_watermark_spatial_map', True),
        wm_map_channels=cfg.get('model', {}).get('wm_map_channels', 4),
        wm_map_size=cfg.get('model', {}).get('wm_map_size', 16),
        wm_time_scale=cfg.get('model', {}).get('wm_time_scale', 1.0),
        wm_map_scale=cfg.get('model', {}).get('wm_map_scale', 1.0),
        use_content_gated_wm_map=cfg.get('model', {}).get(
            'use_content_gated_wm_map', False
        ),
        wm_map_flat_floor=cfg.get('model', {}).get('wm_map_flat_floor', 0.2),
    ).to(device)

    if 'diffusion_model' in checkpoint:
        model.load_state_dict(checkpoint['diffusion_model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    decoder = build_watermark_decoder(
        cfg,
        watermark_length=watermark_length,
    ).to(device)
    if 'decoder' in checkpoint:
        missing, unexpected, mismatched = load_watermark_decoder_state(
            decoder,
            checkpoint['decoder'],
        )
        if missing or unexpected or mismatched:
            print(
                '[Eval] Decoder checkpoint partially loaded '
                '(architecture may have changed).'
            )
    decoder.eval()
    lpips_model = build_lpips_model(args.enable_lpips, device)

    data_dir = args.data_dir or cfg.get('data', {}).get('val_dir')
    if not data_dir:
        raise ValueError(
            'Validation data directory is required via --data_dir or '
            'config data.val_dir'
        )
    base_dataset = WatermarkImageDataset(
        data_dir=data_dir,
        image_size=image_size,
        watermark_length=watermark_length,
        watermark_seed=cfg.get('data', {}).get('watermark_seed', 42),
        watermark_mode='fixed',
        is_train=False,
    )
    eval_indices = select_eval_indices(
        len(base_dataset),
        args.num_eval_images,
        args.subset_seed,
    )
    dataset = IndexedDataset(base_dataset, eval_indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=(device.type == 'cuda'),
    )
    print(
        f'[Eval] Dataset size: {len(base_dataset)} | '
        f'evaluating: {len(dataset)}'
    )

    requested_layers = parse_noise_layers(args.noise_layers)
    simulators = {}
    for noise_type in requested_layers:
        simulators[noise_type] = (
            None
            if noise_type == 'clean'
            else build_eval_noise_layer(cfg, noise_type, device)
        )
    print(f"[Eval] Noise layers: {', '.join(requested_layers)}")
    print(
        f'[Eval] t_start={args.t_start}, '
        f'noise_strength={args.noise_strength:.3f}, '
        f'attack_repeats={args.attack_repeats}'
    )

    paths = output_paths(args.output)
    os.makedirs(os.path.dirname(paths['summary']) or '.', exist_ok=True)
    if args.num_visual_samples > 0:
        os.makedirs(paths['sample_dir'], exist_ok=True)
    write_indices(paths['indices'], eval_indices)

    per_image_rows = []
    watermark_rows = []
    running_accuracy = defaultdict(lambda: {'sum': 0.0, 'count': 0})
    processed_images = 0
    visual_count = 0
    interrupted = False
    start_time = time.time()
    total_batches = len(loader)

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                cover_m11 = batch['image'].to(device)
                wm_bits = batch['wm_bits'].to(device)
                dataset_indices = batch['_dataset_index'].tolist()
                batch_size = cover_m11.size(0)
                batch_watermark_rows = []
                batch_per_image_rows = []

                watermarked_m11 = embed_watermark_eval(
                    diffusion,
                    model,
                    cover_m11,
                    wm_bits,
                    t_start=args.t_start,
                    region_guidance_config=cfg.get('train', {}).get(
                        'region_guidance', {}
                    ),
                    residual_constraint_config=cfg.get('train', {}).get(
                        'residual_constraint', {}
                    ),
                )
                cover_01 = to_01(cover_m11)
                watermarked_01 = to_01(watermarked_m11)

                wm_psnr = per_image_psnr(watermarked_01, cover_01)
                wm_ssim = per_image_ssim(watermarked_01, cover_01)
                wm_l1 = per_image_l1(watermarked_01, cover_01)
                wm_lpips = None
                if lpips_model is not None:
                    wm_lpips = lpips_model(
                        cover_m11,
                        watermarked_m11,
                    ).flatten(1).mean(dim=1)

                for image_index in range(batch_size):
                    batch_watermark_rows.append({
                        'dataset_index': int(dataset_indices[image_index]),
                        'watermark_psnr': float(wm_psnr[image_index].item()),
                        'watermark_ssim': float(wm_ssim[image_index].item()),
                        'watermark_l1': float(wm_l1[image_index].item()),
                        'watermark_lpips': (
                            float(wm_lpips[image_index].item())
                            if wm_lpips is not None
                            else ''
                        ),
                    })

                visual_positions = []
                remaining_visuals = args.num_visual_samples - visual_count
                if remaining_visuals > 0:
                    visual_positions = list(
                        range(min(batch_size, remaining_visuals))
                    )
                visual_records = {}
                for image_index in visual_positions:
                    visual_records[image_index] = {
                        'dataset_index': int(dataset_indices[image_index]),
                        'cover': cover_01[image_index].detach().cpu(),
                        'watermarked': (
                            watermarked_01[image_index].detach().cpu()
                        ),
                        'watermark_metrics': {
                            'psnr': float(wm_psnr[image_index].item()),
                            'ssim': float(wm_ssim[image_index].item()),
                        },
                        'clean_acc': None,
                        'degradations': {},
                    }

                for noise_type, simulator in simulators.items():
                    repeat_count = (
                        1 if noise_type == 'clean' else args.attack_repeats
                    )
                    for repeat_index in range(repeat_count):
                        mixed_active_type = ''
                        if simulator is None:
                            degraded_01 = watermarked_01
                        else:
                            degraded_01 = apply_degradation_with_strength(
                                watermarked_01,
                                simulator,
                                args.noise_strength,
                            )
                            if noise_type == 'mixed':
                                mixed_active_type = (
                                    simulator.get_last_name() or ''
                                )
                        degraded_m11 = to_m11(degraded_01)

                        logits = decoder(degraded_m11)
                        pred_bits = torch.sigmoid(logits).gt(0.5)
                        target_bits = wm_bits.gt(0.5)
                        bit_acc = (
                            pred_bits.eq(target_bits)
                            .float()
                            .mean(dim=1)
                        )
                        message_success = (
                            pred_bits.eq(target_bits)
                            .all(dim=1)
                            .float()
                        )

                        attack_psnr = per_image_psnr(
                            degraded_01,
                            watermarked_01,
                        )
                        attack_ssim = per_image_ssim(
                            degraded_01,
                            watermarked_01,
                        )
                        attack_l1 = per_image_l1(
                            degraded_01,
                            watermarked_01,
                        )
                        end2end_psnr = per_image_psnr(
                            degraded_01,
                            cover_01,
                        )
                        end2end_ssim = per_image_ssim(
                            degraded_01,
                            cover_01,
                        )
                        end2end_l1 = per_image_l1(
                            degraded_01,
                            cover_01,
                        )

                        for image_index in range(batch_size):
                            acc_value = float(bit_acc[image_index].item())
                            row = {
                                'dataset_index': int(
                                    dataset_indices[image_index]
                                ),
                                'noise_type': noise_type,
                                'repeat_index': repeat_index,
                                'mixed_active_type': mixed_active_type,
                                'noise_strength': (
                                    0.0
                                    if noise_type == 'clean'
                                    else float(args.noise_strength)
                                ),
                                'bit_acc': acc_value,
                                'ber': 1.0 - acc_value,
                                'message_success': float(
                                    message_success[image_index].item()
                                ),
                                'watermark_psnr': float(
                                    wm_psnr[image_index].item()
                                ),
                                'watermark_ssim': float(
                                    wm_ssim[image_index].item()
                                ),
                                'watermark_l1': float(
                                    wm_l1[image_index].item()
                                ),
                                'watermark_lpips': (
                                    float(wm_lpips[image_index].item())
                                    if wm_lpips is not None
                                    else ''
                                ),
                                'attack_psnr': float(
                                    attack_psnr[image_index].item()
                                ),
                                'attack_ssim': float(
                                    attack_ssim[image_index].item()
                                ),
                                'attack_l1': float(
                                    attack_l1[image_index].item()
                                ),
                                'end2end_psnr': float(
                                    end2end_psnr[image_index].item()
                                ),
                                'end2end_ssim': float(
                                    end2end_ssim[image_index].item()
                                ),
                                'end2end_l1': float(
                                    end2end_l1[image_index].item()
                                ),
                            }
                            batch_per_image_rows.append(row)

                            if (
                                image_index in visual_records
                                and repeat_index == 0
                            ):
                                if noise_type == 'clean':
                                    visual_records[image_index][
                                        'clean_acc'
                                    ] = acc_value
                                else:
                                    visual_records[image_index][
                                        'degradations'
                                    ][noise_type] = {
                                        'image': degraded_01[
                                            image_index
                                        ].detach().cpu(),
                                        'mixed_active_type': (
                                            mixed_active_type
                                        ),
                                        'bit_acc': acc_value,
                                        'ber': 1.0 - acc_value,
                                        'attack_psnr': float(
                                            attack_psnr[
                                                image_index
                                            ].item()
                                        ),
                                    }

                for image_index in visual_positions:
                    output_index = visual_count
                    comparison_path = os.path.join(
                        paths['sample_dir'],
                        f'{output_index:04d}_comparison.png',
                    )
                    save_comparison_grid(
                        comparison_path,
                        visual_records[image_index],
                        requested_layers,
                        save_individual_samples=(
                            args.save_individual_samples
                        ),
                    )
                    visual_count += 1

                # Commit a batch only after sampling, all requested attacks,
                # decoding, metrics, and visualizations have completed. This
                # keeps partial results consistent if evaluation is interrupted.
                watermark_rows.extend(batch_watermark_rows)
                per_image_rows.extend(batch_per_image_rows)
                for row in batch_per_image_rows:
                    stats = running_accuracy[row['noise_type']]
                    stats['sum'] += row['bit_acc']
                    stats['count'] += 1
                processed_images += batch_size
                if (
                    batch_idx % args.progress_interval == 0
                    or batch_idx == total_batches
                ):
                    print_progress(
                        processed_images,
                        len(dataset),
                        batch_idx,
                        total_batches,
                        start_time,
                        running_accuracy,
                    )
    except KeyboardInterrupt:
        interrupted = True
        print(
            '\n[Eval] Interrupted by user. '
            'Saving metrics completed so far...'
        )

    metadata = {
        'status': 'interrupted' if interrupted else 'completed',
        'checkpoint': os.path.abspath(args.checkpoint),
        'config': os.path.abspath(args.config) if args.config else None,
        'data_dir': os.path.abspath(data_dir),
        'dataset_size': len(base_dataset),
        'requested_eval_images': len(eval_indices),
        'processed_images': processed_images,
        'subset_seed': int(args.subset_seed),
        'seed': int(seed),
        't_start': int(args.t_start),
        'batch_size': int(args.batch_size),
        'noise_strength': float(args.noise_strength),
        'attack_repeats': int(args.attack_repeats),
        'noise_layers': requested_layers,
        'num_visual_samples': int(args.num_visual_samples),
        'lpips_enabled': bool(args.enable_lpips),
        'elapsed_seconds': float(time.time() - start_time),
    }

    summary, by_noise_rows = write_result_files(
        paths,
        per_image_rows,
        watermark_rows,
        requested_layers,
        metadata,
    )

    print('\n' + '=' * 72)
    print('EVALUATION RESULTS')
    print('=' * 72)
    print(
        f"[Eval] status={metadata['status']} "
        f"images={processed_images}/{len(dataset)} "
        f"strength={args.noise_strength:.3f}"
    )
    if watermark_rows:
        print(
            '[Eval] watermarked: '
            f"PSNR={summary['watermark_psnr_mean']:.2f}, "
            f"SSIM={summary['watermark_ssim_mean']:.4f}"
        )
    for row in by_noise_rows:
        print(
            f"[Eval] {row['noise_type']:9s}: "
            f"bit_acc={row['bit_acc_mean']:.4f}, "
            f"BER={row['ber_mean']:.4f}, "
            f"message_success={row['message_success_rate']:.4f}, "
            f"end2end_PSNR={row['end2end_psnr_mean']:.2f}, "
            f"end2end_SSIM={row['end2end_ssim_mean']:.4f}"
        )

    print(f"\n[Eval] Summary: {paths['summary']}")
    print(f"[Eval] By noise: {paths['by_noise']}")
    print(f"[Eval] Per image: {paths['per_image']}")
    print(f"[Eval] Metadata: {paths['metadata']}")
    print(f"[Eval] Indices: {paths['indices']}")
    if args.num_visual_samples > 0:
        print(f"[Eval] Comparison images: {paths['sample_dir']}")


if __name__ == '__main__':
    main()
