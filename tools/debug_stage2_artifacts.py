"""Diagnose Stage-2 watermark artifacts without changing the training path.

The script embeds one fixed batch once, then sends the identical watermarked
images through clean/PIMoG/OLED/mixed degradation branches. It writes images,
FFT and edge/texture metrics, tensor range statistics, decoder metrics, and the
actual random values sampled by the degradation implementations.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guided_diffusion.gaussian_diffusion import (  # noqa: E402
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from models.watermark_decoder import build_watermark_decoder  # noqa: E402
from models.watermark_residual import (  # noqa: E402
    build_edge_texture_guidance,
    constrain_watermark_residual,
    get_residual_constraint_settings,
)
from models.watermark_unet import WatermarkConditionedUNet  # noqa: E402
from NOISE_LAYER.build_noise_layer import MixedNoiseLayer, build_noise_layer  # noqa: E402
from NOISE_LAYER.OLED_Layer import OLED_Layer  # noqa: E402
from NOISE_LAYER.PIMoG_Layer import PIMoGLayer  # noqa: E402

led_module = importlib.import_module("NOISE_LAYER.LED_Layer")
oled_module = importlib.import_module("NOISE_LAYER.OLED_Layer")
pimog_module = importlib.import_module("NOISE_LAYER.PIMoG_Layer")
projector_module = importlib.import_module("NOISE_LAYER.Projector_Layer")


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PERIODS = (2, 4, 8, 16)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose periodic/color artifacts in Stage-2 checkpoints."
    )
    parser.add_argument("--config", required=True, help="YAML config used to build the model.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint containing model and decoder.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="New output directory. The script refuses to overwrite an existing path.",
    )
    parser.add_argument(
        "--noise_layer",
        default="clean,pimog,oled,mixed",
        help="Comma-separated subset of clean,pimog,oled,led,projector,mixed.",
    )
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--random_trials",
        type=int,
        default=4,
        help="Repeated forwards per non-clean layer for RNG/output-variation checks.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Optional image directory/file; defaults to config data.val_dir.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Requested device; falls back to CPU when CUDA is unavailable.",
    )
    parser.add_argument(
        "--t_start",
        type=int,
        default=None,
        help="Reverse start timestep; defaults to diffusion.train_t_start.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Blend strength between watermarked and fully degraded images.",
    )
    parser.add_argument(
        "--gradient_diagnostics",
        action="store_true",
        help="Also run one FP32 single-image autograd.grad loss competition audit.",
    )
    parser.add_argument(
        "--diagnostic_step",
        type=int,
        default=None,
        help="Step used to resolve loss/curriculum weights; defaults to checkpoint global_step.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config is empty or invalid: {path}")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dictionary, got {type(checkpoint).__name__}")
    return checkpoint


def json_value(value: Any, max_values: int = 32) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().cpu()
        flat = tensor.flatten()
        result: Dict[str, Any] = {
            "shape": list(tensor.shape),
            "min": float(flat.min()) if flat.numel() else None,
            "max": float(flat.max()) if flat.numel() else None,
            "mean": float(flat.mean()) if flat.numel() else None,
            "std": float(flat.std(unbiased=False)) if flat.numel() else None,
        }
        if flat.numel() <= max_values:
            result["values"] = flat.tolist()
        else:
            result["first_values"] = flat[:max_values].tolist()
        return result
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        flat = array.reshape(-1)
        result = {
            "shape": list(array.shape),
            "min": float(flat.min()) if flat.size else None,
            "max": float(flat.max()) if flat.size else None,
            "mean": float(flat.mean()) if flat.size else None,
            "std": float(flat.std()) if flat.size else None,
        }
        if flat.size <= max_values:
            result["values"] = flat.tolist()
        else:
            result["first_values"] = flat[:max_values].tolist()
        return result
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [json_value(item, max_values=max_values) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item, max_values=max_values) for key, item in value.items()}
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


class RandomParameterTrace:
    """Temporarily record the exact RNG outputs used by degradation forwards."""

    def __init__(self) -> None:
        self.records: List[dict] = []
        self._restores: List[Tuple[Any, str, Any]] = []

    def _record(self, source: str, value: Any, arguments: Any = None) -> None:
        caller = inspect.currentframe().f_back.f_back.f_code.co_name
        self.records.append(
            {
                "index": len(self.records),
                "source": source,
                "caller": caller,
                "arguments": json_value(arguments),
                "sample": json_value(value),
            }
        )

    def _replace(self, owner: Any, name: str, replacement: Any) -> None:
        original = getattr(owner, name)
        self._restores.append((owner, name, original))
        setattr(owner, name, replacement)

    def __enter__(self) -> "RandomParameterTrace":
        for module, module_name in (
            (oled_module, "oled"),
            (led_module, "led"),
            (projector_module, "projector"),
        ):
            original = module.sample_uniform

            def traced_uniform(
                x: torch.Tensor,
                value_range: Sequence[float],
                shape: Sequence[int],
                _original=original,
                _module_name=module_name,
            ) -> torch.Tensor:
                value = _original(x, value_range, shape)
                self._record(
                    f"{_module_name}.sample_uniform",
                    value,
                    {"range": list(value_range), "shape": list(shape) if isinstance(shape, tuple) else shape},
                )
                return value

            self._replace(module, "sample_uniform", traced_uniform)

        original_torch_rand = torch.rand

        def traced_torch_rand(*args: Any, **kwargs: Any) -> torch.Tensor:
            value = original_torch_rand(*args, **kwargs)
            self._record("torch.rand", value, {"args": args, "kwargs": kwargs})
            return value

        self._replace(torch, "rand", traced_torch_rand)

        original_torch_randint = torch.randint

        def traced_torch_randint(*args: Any, **kwargs: Any) -> torch.Tensor:
            value = original_torch_randint(*args, **kwargs)
            self._record("torch.randint", value, {"args": args, "kwargs": kwargs})
            return value

        self._replace(torch, "randint", traced_torch_randint)

        original_np_rand = pimog_module.np.random.rand

        def traced_np_rand(*args: Any) -> Any:
            value = original_np_rand(*args)
            self._record("pimog.numpy.rand", value, {"args": args})
            return value

        self._replace(pimog_module.np.random, "rand", traced_np_rand)

        original_np_randint = pimog_module.np.random.randint

        def traced_np_randint(*args: Any, **kwargs: Any) -> Any:
            value = original_np_randint(*args, **kwargs)
            self._record("pimog.numpy.randint", value, {"args": args, "kwargs": kwargs})
            return value

        self._replace(pimog_module.np.random, "randint", traced_np_randint)

        original_py_uniform = pimog_module.random.uniform

        def traced_py_uniform(a: float, b: float) -> float:
            value = original_py_uniform(a, b)
            self._record("pimog.python.uniform", value, {"low": a, "high": b})
            return value

        self._replace(pimog_module.random, "uniform", traced_py_uniform)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for owner, name, original in reversed(self._restores):
            setattr(owner, name, original)
        self._restores.clear()


def compare_state_dicts(current: dict, candidate: dict) -> dict:
    current_keys = set(current)
    candidate_keys = set(candidate)
    mismatched = sorted(
        key
        for key in current_keys & candidate_keys
        if tuple(current[key].shape) != tuple(candidate[key].shape)
    )
    return {
        "missing_keys": sorted(current_keys - candidate_keys),
        "unexpected_keys": sorted(candidate_keys - current_keys),
        "shape_mismatches": [
            {
                "key": key,
                "checkpoint": list(candidate[key].shape),
                "current": list(current[key].shape),
            }
            for key in mismatched
        ],
    }


def normalize_decoder_state(decoder: torch.nn.Module, state: dict) -> dict:
    current = decoder.state_dict()
    if any(key.startswith("decoder.") for key in state):
        return state
    prefixed = {f"decoder.{key}": value for key, value in state.items()}
    if set(prefixed) & set(current):
        return prefixed
    return state


def build_components(config: dict, device: torch.device) -> Tuple[Any, Any, Any]:
    image_size = int(config["data"]["image_size"])
    watermark_length = int(config["data"]["watermark_length"])
    model_cfg = config["model"]
    diffusion_cfg = config["diffusion"]

    betas = get_named_beta_schedule(
        diffusion_cfg.get("beta_schedule", "linear"),
        int(diffusion_cfg.get("timesteps", 1000)),
    )
    diffusion = GaussianDiffusion(
        betas=torch.tensor(betas, dtype=torch.float32),
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )
    model = WatermarkConditionedUNet(
        image_size=image_size,
        base_channels=int(model_cfg.get("base_channels", 64)),
        cond_dim=int(model_cfg.get("cond_dim", 256)),
        watermark_length=watermark_length,
        use_pretrained_unet=False,
        pretrained_path=None,
        use_watermark_time_emb=bool(model_cfg.get("use_watermark_time_emb", True)),
        use_watermark_spatial_map=bool(model_cfg.get("use_watermark_spatial_map", True)),
        wm_map_channels=int(model_cfg.get("wm_map_channels", 4)),
        wm_map_size=int(model_cfg.get("wm_map_size", 16)),
        wm_time_scale=float(model_cfg.get("wm_time_scale", 1.0)),
        wm_map_scale=float(model_cfg.get("wm_map_scale", 1.0)),
        use_content_gated_wm_map=bool(
            model_cfg.get("use_content_gated_wm_map", False)
        ),
        wm_map_flat_floor=float(model_cfg.get("wm_map_flat_floor", 0.2)),
    ).to(device)
    decoder = build_watermark_decoder(config, watermark_length=watermark_length).to(device)
    return diffusion, model, decoder


def load_weights_strict(
    checkpoint: dict,
    model: torch.nn.Module,
    decoder: torch.nn.Module,
) -> dict:
    model_state = checkpoint.get("diffusion_model", checkpoint.get("model", checkpoint))
    if not isinstance(model_state, dict):
        raise TypeError("Checkpoint diffusion model state is not a dictionary")
    decoder_state = checkpoint.get("decoder")
    if not isinstance(decoder_state, dict):
        raise KeyError("Checkpoint has no decoder state; decoder diagnostics would be invalid")
    decoder_state = normalize_decoder_state(decoder, decoder_state)

    report = {
        "model": compare_state_dicts(model.state_dict(), model_state),
        "decoder": compare_state_dicts(decoder.state_dict(), decoder_state),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "checkpoint_curriculum_phase": checkpoint.get("curriculum_phase"),
        "checkpoint_degradation_stage": checkpoint.get("degradation_stage"),
    }
    has_problem = any(
        report[part][field]
        for part in ("model", "decoder")
        for field in ("missing_keys", "unexpected_keys", "shape_mismatches")
    )
    if has_problem:
        raise RuntimeError(
            "Checkpoint/config architecture mismatch. See checkpoint_load.json; "
            "partial loading is intentionally refused for diagnostics."
        )
    model.load_state_dict(model_state, strict=True)
    decoder.load_state_dict(decoder_state, strict=True)
    model.eval()
    decoder.eval()
    return report


def discover_images(path: Path, count: int) -> List[Path]:
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(
            item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
    else:
        raise FileNotFoundError(f"Image input does not exist: {path}")
    if not paths:
        raise RuntimeError(f"No supported images found under: {path}")
    return paths[:count]


def load_cover_batch(paths: Sequence[Path], image_size: int, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            tensors.append(transform(image.convert("RGB")))
    return torch.stack(tensors, dim=0).to(device)


@torch.no_grad()
def embed_watermark_with_trace(
    diffusion: GaussianDiffusion,
    model: torch.nn.Module,
    cover: torch.Tensor,
    bits: torch.Tensor,
    t_start: int,
    config: dict,
) -> Tuple[torch.Tensor, dict]:
    if not 1 <= t_start <= diffusion.num_timesteps:
        raise ValueError(f"t_start must be in [1, {diffusion.num_timesteps}], got {t_start}")
    batch = cover.shape[0]
    train_config = config.get("train", {})
    guidance_config = train_config.get("region_guidance", {})
    constraint_config = train_config.get("residual_constraint", {})
    constraint_settings = get_residual_constraint_settings(constraint_config)
    needs_guidance = (
        constraint_settings["enabled"]
        or bool(getattr(model, "use_content_gated_wm_map", False))
    )
    allowance = (
        build_edge_texture_guidance((cover + 1.0) * 0.5, guidance_config)[0]
        if needs_guidance else None
    )

    def constrain_xstart(x_start: torch.Tensor) -> torch.Tensor:
        return constrain_watermark_residual(
            x_start, cover, allowance, constraint_config
        )

    timestep = torch.full((batch,), t_start - 1, device=cover.device, dtype=torch.long)
    x_t = diffusion.q_sample(cover, timestep, noise=torch.randn_like(cover))
    initial_stats = tensor_stats(x_t, expected_range=(-1.0, 1.0))

    for step in reversed(range(t_start)):
        t_batch = torch.full((batch,), step, device=cover.device, dtype=torch.long)
        t_scaled = t_batch.float() * (1000.0 / diffusion.num_timesteps)
        predicted_noise = model(
            x_t=x_t,
            t=t_scaled,
            cover_img=cover,
            wm_bits=bits,
            content_mask=allowance,
        )
        output = diffusion.p_mean_variance(
            model=lambda *args, **kwargs: predicted_noise,
            x=x_t,
            t=t_batch,
            clip_denoised=True,
            denoised_fn=constrain_xstart
            if constraint_settings["enabled"] else None,
            model_kwargs={},
        )
        noise_term = torch.randn_like(x_t) if step > 0 else torch.zeros_like(x_t)
        x_t = output["mean"] + torch.exp(0.5 * output["log_variance"]) * noise_term

    raw = x_t
    clipped = constrain_xstart(raw)
    trace = {
        "initial_q_sample": initial_stats,
        "final_before_clamp": tensor_stats(raw, expected_range=(-1.0, 1.0)),
        "final_after_clamp": tensor_stats(clipped, expected_range=(-1.0, 1.0)),
        "final_clamp": clamp_report(raw, -1.0, 1.0),
    }
    return clipped, trace


def tensor_stats(tensor: torch.Tensor, expected_range: Tuple[float, float] = (0.0, 1.0)) -> dict:
    value = tensor.detach().float()
    finite = torch.isfinite(value)
    finite_values = value[finite]
    low, high = expected_range
    result = {
        "shape": list(value.shape),
        "dtype": str(tensor.dtype),
        "min": float(finite_values.min()) if finite_values.numel() else None,
        "max": float(finite_values.max()) if finite_values.numel() else None,
        "mean": float(finite_values.mean()) if finite_values.numel() else None,
        "std": float(finite_values.std(unbiased=False)) if finite_values.numel() else None,
        "nan_count": int(torch.isnan(value).sum()),
        "inf_count": int(torch.isinf(value).sum()),
        "below_expected_ratio": float((value < low).float().mean()),
        "above_expected_ratio": float((value > high).float().mean()),
        "at_lower_bound_ratio": float((value == low).float().mean()),
        "at_upper_bound_ratio": float((value == high).float().mean()),
    }
    if value.ndim == 4 and value.shape[1] == 3:
        result["rgb_mean"] = value.mean(dim=(0, 2, 3)).cpu().tolist()
        result["rgb_std"] = value.std(dim=(0, 2, 3), unbiased=False).cpu().tolist()
    return result


def clamp_report(before: torch.Tensor, low: float, high: float) -> dict:
    value = before.detach().float()
    return {
        "low": low,
        "high": high,
        "below_ratio": float((value < low).float().mean()),
        "above_ratio": float((value > high).float().mean()),
        "truncated_ratio": float(((value < low) | (value > high)).float().mean()),
    }


def run_oled_tone_exact(
    layer: OLED_Layer,
    source: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """Mirror OLED tone mapping while exposing its hidden clamp boundaries."""
    batch = source.shape[0]
    gamma = oled_module.sample_uniform(
        source, layer.gamma_range, (batch, 1, 1, 1)
    )
    contrast = oled_module.sample_uniform(
        source, layer.contrast_range, (batch, 1, 1, 1)
    )
    saturation = oled_module.sample_uniform(
        source, layer.saturation_range, (batch, 1, 1, 1)
    )
    black = oled_module.sample_uniform(
        source, layer.black_crush_range, (batch, 1, 1, 1)
    )
    brightness = oled_module.sample_uniform(
        source,
        (-layer.brightness_jitter, layer.brightness_jitter),
        (batch, 1, 1, 1),
    )
    gain = oled_module.sample_uniform(
        source, layer.color_gain_range, (batch, 3, 1, 1)
    )

    gamma_input = source.clamp(1e-6, 1.0)
    value = gamma_input.pow(gamma)
    black_raw = (value - black) / (1.0 - black).clamp_min(1e-4)
    value = black_raw.clamp_min(0.0).pow(1.04)
    mean = value.mean(dim=(2, 3), keepdim=True)
    value = (value - mean) * contrast + mean + brightness
    luma = (
        0.299 * value[:, 0:1]
        + 0.587 * value[:, 1:2]
        + 0.114 * value[:, 2:3]
    )
    value = (luma + (value - luma) * saturation) * gain

    highlight_gate = bool(
        (torch.rand((), device=value.device) < layer.highlight_clip_prob).item()
    )
    if highlight_gate:
        clip = oled_module.sample_uniform(
            value, layer.highlight_clip_range, (batch, 1, 1, 1)
        )
        tone_raw = clip * torch.tanh(value / clip.clamp_min(1e-4))
        highlight_mode = "soft_tanh_clip"
    else:
        tone_raw = value - 0.10 * (value - 1.0).clamp_min(0.0)
        highlight_mode = "rolloff"
    output = tone_raw.clamp(0.0, 1.0)
    return output, {
        "gamma_input_clamp": clamp_report(source, 1e-6, 1.0),
        "after_gamma": tensor_stats(gamma_input.pow(gamma)),
        "black_crush_below_zero_ratio": float(
            (black_raw < 0.0).float().mean()
        ),
        "before_highlight": tensor_stats(value),
        "highlight_mode": highlight_mode,
        "tone_before_final_clamp": tensor_stats(tone_raw),
        "tone_final_clamp": clamp_report(tone_raw, 0.0, 1.0),
        "tone_output": tensor_stats(output),
    }


def run_oled_exact(layer: OLED_Layer, source: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    stages: Dict[str, dict] = {}
    clean = source.clamp(0.0, 1.0)
    stages["input_clamp"] = {
        "stats": tensor_stats(clean),
        "clamp": clamp_report(source, 0.0, 1.0),
    }
    if layer.p == 0.0 or (layer.p < 1.0 and torch.rand((), device=source.device) >= layer.p):
        return clean.to(dtype=source.dtype), {"skipped_by_probability": True, "stages": stages}

    value = clean
    if layer.enable_tone:
        value, tone_details = run_oled_tone_exact(layer, value)
        stages["tone"] = tone_details
    operations = [
        ("subpixel", layer._apply_subpixel_pentile),
        ("display_blur", layer._apply_display_blur),
        ("perspective", layer._apply_perspective),
        ("camera_blur", layer._apply_camera_blur),
        ("pwm_banding", layer._apply_pwm_banding),
        ("view_color_shift", layer._apply_view_color_shift),
        ("sensor_noise", layer._apply_sensor_noise),
        ("motion_blur", layer._apply_motion_blur),
        ("reflection_haze", layer._apply_reflection_haze),
        ("resample", layer._apply_resample),
        ("jpeg_proxy", layer._apply_jpeg_proxy),
    ]
    for name, operation in operations:
        if operation is None:
            continue
        value = operation(value)
        stages[name] = tensor_stats(value)
    output = value.clamp(0.0, 1.0).to(dtype=source.dtype)
    stages["final_clamp"] = {
        "stats": tensor_stats(output),
        "clamp": clamp_report(value, 0.0, 1.0),
    }
    return output, {"skipped_by_probability": False, "stages": stages}


def run_pimog_exact(layer: PIMoGLayer, source: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    clean = source.clamp(0.0, 1.0)
    stages: Dict[str, Any] = {
        "input_clamp": {
            "stats": tensor_stats(clean),
            "clamp": clamp_report(source, 0.0, 1.0),
        }
    }
    if layer.p == 0.0 or (layer.p < 1.0 and torch.rand((), device=source.device) >= layer.p):
        return clean, {"skipped_by_probability": True, "stages": stages}
    internal_input = clean.mul(2.0).sub(1.0)
    degraded_internal = layer.screen_shooting(internal_input).float()
    mapped = degraded_internal.add(1.0).mul(0.5)
    output = mapped.clamp(0.0, 1.0).to(dtype=source.dtype)
    stages["internal_input_minus1_1"] = tensor_stats(internal_input, (-1.0, 1.0))
    stages["internal_output_minus1_1"] = tensor_stats(degraded_internal, (-1.0, 1.0))
    stages["mapped_before_clamp"] = tensor_stats(mapped)
    stages["final_clamp"] = {
        "stats": tensor_stats(output),
        "clamp": clamp_report(mapped, 0.0, 1.0),
    }
    return output, {"skipped_by_probability": False, "stages": stages}


def run_concrete_exact(layer: torch.nn.Module, source: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    if isinstance(layer, OLED_Layer):
        return run_oled_exact(layer, source)
    if isinstance(layer, PIMoGLayer):
        return run_pimog_exact(layer, source)
    output = layer(source).float()
    return output, {
        "stages": {
            "input": tensor_stats(source),
            "output": tensor_stats(output),
        }
    }


def build_diagnostic_layer(config: dict, noise_name: str, device: torch.device) -> Any:
    if noise_name == "clean":
        return None
    layer_config = dict(config)
    noise_config = dict(config.get("noise_layer", {}))
    noise_config["type"] = noise_name
    if noise_name in {"pimog", "oled", "led", "projector"}:
        noise_config[noise_name] = dict(noise_config.get(noise_name, {}), p=1.0)
    layer_config["noise_layer"] = noise_config
    layer = build_noise_layer(layer_config).to(device)
    layer.eval()
    return layer


@torch.no_grad()
def apply_degradation_traced(
    layer: Any,
    name: str,
    source: torch.Tensor,
    strength: float,
) -> Tuple[torch.Tensor, dict]:
    if name == "clean":
        return source, {
            "selected_noise_layer": "clean",
            "random_parameters": [],
            "stages": {"identity": tensor_stats(source)},
            "blend_clamp": clamp_report(source, 0.0, 1.0),
        }

    selection: Dict[str, Any] = {"requested_noise_layer": name}
    with RandomParameterTrace() as random_trace:
        if isinstance(layer, MixedNoiseLayer):
            indices, probabilities = layer._distribution()
            local_index = int(torch.multinomial(probabilities, 1).item())
            index = indices[local_index]
            layer.last_index = index
            layer.last_name = layer.names[index]
            selection.update(
                {
                    "candidates": [layer.names[item] for item in indices],
                    "normalized_probabilities": probabilities.detach().cpu().tolist(),
                    "sampled_local_index": local_index,
                    "selected_noise_layer": layer.last_name,
                }
            )
            full, details = run_concrete_exact(layer.layers[index], source)
        else:
            selection["selected_noise_layer"] = name
            full, details = run_concrete_exact(layer, source)

    blended_before_clamp = source.float() + strength * (full.float() - source.float())
    output = blended_before_clamp.clamp(0.0, 1.0).to(dtype=source.dtype)
    return output, {
        **selection,
        **details,
        "strength": strength,
        "full_degradation_stats": tensor_stats(full),
        "blend_before_clamp": tensor_stats(blended_before_clamp),
        "blend_clamp": clamp_report(blended_before_clamp, 0.0, 1.0),
        "output_stats": tensor_stats(output),
        "random_parameters": random_trace.records,
    }


def normalized_activity_map(activity: torch.Tensor) -> torch.Tensor:
    flat = activity.flatten(1)
    scale = flat.mean(dim=1, keepdim=True) + 2.0 * flat.std(
        dim=1, keepdim=True, unbiased=False
    )
    return (activity / (scale.view(-1, 1, 1, 1) + 1e-6)).clamp(0.0, 1.0)


def edge_texture_maps(cover_01: torch.Tensor, texture_kernel: int = 5) -> Tuple[Any, Any]:
    image = cover_01.detach().float().clamp(0.0, 1.0)
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    sobel_x = gray.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    padded = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    grad_x = F.conv2d(padded, sobel_x)
    grad_y = F.conv2d(padded, sobel_x.transpose(-1, -2))
    edge = normalized_activity_map(torch.sqrt(grad_x.square() + grad_y.square()))

    pad = texture_kernel // 2
    texture_source = F.pad(gray, (pad, pad, pad, pad), mode="reflect")
    local_mean = F.avg_pool2d(texture_source, texture_kernel, stride=1)
    local_square_mean = F.avg_pool2d(texture_source.square(), texture_kernel, stride=1)
    texture = normalized_activity_map(
        (local_square_mean - local_mean.square()).clamp_min(0.0).sqrt()
    )
    return edge, texture


def spatial_energy_metrics(
    delta: torch.Tensor,
    edge_map: torch.Tensor,
    texture_map: torch.Tensor,
) -> List[dict]:
    energy = delta.detach().float().abs().mean(dim=1, keepdim=True)
    total = energy.sum(dim=(1, 2, 3)).clamp_min(EPS)
    edge = (energy * edge_map).sum(dim=(1, 2, 3)) / total
    flat = (energy * (1.0 - edge_map)).sum(dim=(1, 2, 3)) / total
    texture = (energy * texture_map).sum(dim=(1, 2, 3)) / total
    smooth = (energy * (1.0 - texture_map)).sum(dim=(1, 2, 3)) / total
    return [
        {
            "edge_energy_ratio": float(edge[index]),
            "flat_energy_ratio": float(flat[index]),
            "texture_energy_ratio": float(texture[index]),
            "smooth_energy_ratio": float(smooth[index]),
        }
        for index in range(delta.shape[0])
    ]


def _period_mask(height: int, width: int, period: int, device: torch.device) -> torch.Tensor:
    center_y, center_x = height // 2, width // 2
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    target_y = height / period
    target_x = width / period

    def mark_neighborhood(y: int, x: int) -> None:
        for offset_y in (-1, 0, 1):
            for offset_x in (-1, 0, 1):
                mask[(y + offset_y) % height, (x + offset_x) % width] = True

    for y_sign in (-1, 1):
        for x_sign in (-1, 0, 1):
            if x_sign == 0:
                y = int(round(center_y + y_sign * target_y)) % height
                x = center_x
            else:
                y = int(round(center_y + y_sign * target_y)) % height
                x = int(round(center_x + x_sign * target_x)) % width
            mark_neighborhood(y, x)
    for x_sign in (-1, 1):
        x = int(round(center_x + x_sign * target_x)) % width
        y = center_y
        mark_neighborhood(y, x)
    return mask


def fft_metrics_single(delta_chw: torch.Tensor) -> Tuple[dict, torch.Tensor, torch.Tensor]:
    delta = delta_chw.detach().float()
    gray = 0.299 * delta[0] + 0.587 * delta[1] + 0.114 * delta[2]
    fft_rgb = torch.fft.fftshift(torch.fft.fft2(delta, norm="ortho"), dim=(-2, -1))
    fft_gray = torch.fft.fftshift(torch.fft.fft2(gray, norm="ortho"), dim=(-2, -1))
    power = fft_gray.abs().square()
    height, width = power.shape
    center_y, center_x = height // 2, width // 2

    non_dc = torch.ones_like(power, dtype=torch.bool)
    non_dc[
        max(0, center_y - 1) : min(height, center_y + 2),
        max(0, center_x - 1) : min(width, center_x + 2),
    ] = False
    total = power[non_dc].sum().clamp_min(EPS)
    count = max(1, int(non_dc.sum().item() * 0.001))
    peak_energy = power[non_dc].topk(count).values.sum()

    axis = torch.zeros_like(non_dc)
    axis[max(0, center_y - 1) : min(height, center_y + 2), :] = True
    axis[:, max(0, center_x - 1) : min(width, center_x + 2)] = True
    axis &= non_dc

    period_ratios = {}
    combined_period_mask = torch.zeros_like(non_dc)
    for period in PERIODS:
        mask = _period_mask(height, width, period, power.device) & non_dc
        combined_period_mask |= mask
        period_ratios[str(period)] = float(power[mask].sum() / total)

    peak_count = min(12, int(non_dc.sum()))
    candidate_power = power.clone()
    candidate_power[~non_dc] = -1.0
    peak_indices = candidate_power.flatten().topk(peak_count).indices
    peaks = []
    for flat_index in peak_indices.tolist():
        y, x = divmod(flat_index, width)
        dy, dx = y - center_y, x - center_x
        peaks.append(
            {
                "offset_y": dy,
                "offset_x": dx,
                "power": float(power[y, x]),
                "period_y": (height / abs(dy)) if dy else None,
                "period_x": (width / abs(dx)) if dx else None,
            }
        )

    metrics = {
        "fft_peak_ratio": float(peak_energy / total),
        "axis_energy_ratio": float(power[axis].sum() / total),
        "periodic_energy_ratio": float(power[combined_period_mask].sum() / total),
        "period_energy_ratio_by_pixels": period_ratios,
        "gray_fft_peaks": peaks,
    }
    spectrum_rgb = torch.log1p(fft_rgb.abs())
    spectrum_gray = torch.log1p(fft_gray.abs()).unsqueeze(0)
    return metrics, spectrum_rgb, spectrum_gray


def normalize_for_display(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float()
    if value.ndim == 3:
        flat = value.flatten(1)
        low = flat.min(dim=1).values.view(-1, 1, 1)
        high = flat.max(dim=1).values.view(-1, 1, 1)
    else:
        low, high = value.min(), value.max()
    return ((value - low) / (high - low).clamp_min(1e-12)).clamp(0.0, 1.0)


def channel_metrics_single(delta: torch.Tensor) -> dict:
    value = delta.detach().float()
    red, green, blue = value[0], value[1], value[2]
    chroma = {
        "g_minus_rb_half": green - 0.5 * (red + blue),
        "rb_half_minus_g": 0.5 * (red + blue) - green,
        "r_minus_g": red - green,
        "b_minus_g": blue - green,
    }
    return {
        "signed_mean_rgb": value.mean(dim=(1, 2)).cpu().tolist(),
        "absolute_mean_rgb": value.abs().mean(dim=(1, 2)).cpu().tolist(),
        "std_rgb": value.std(dim=(1, 2), unbiased=False).cpu().tolist(),
        "chroma": {
            key: {
                "mean": float(item.mean()),
                "absolute_mean": float(item.abs().mean()),
                "std": float(item.std(unbiased=False)),
            }
            for key, item in chroma.items()
        },
    }


def save_residual_bundle(directory: Path, stem: str, delta: torch.Tensor) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    signed_x5 = (0.5 + 5.0 * delta).clamp(0.0, 1.0)
    abs_x5 = (5.0 * delta.abs()).clamp(0.0, 1.0)
    abs_x10 = (10.0 * delta.abs()).clamp(0.0, 1.0)
    minmax = normalize_for_display(delta)
    save_image(signed_x5, directory / f"{stem}_signed_x5.png")
    save_image(abs_x5, directory / f"{stem}_abs_x5.png")
    save_image(abs_x10, directory / f"{stem}_abs_x10.png")
    save_image(minmax, directory / f"{stem}_per_image_minmax.png")

    for channel, label in enumerate(("r", "g", "b")):
        channel_view = (0.5 + 5.0 * delta[channel : channel + 1]).clamp(0.0, 1.0)
        save_image(channel_view, directory / f"{stem}_{label}_signed_x5.png")

    signed_gray = delta.mean(dim=0)
    limit = 0.05
    normalized = (signed_gray / limit).clamp(-1.0, 1.0)
    positive = normalized.clamp_min(0.0)
    negative = (-normalized).clamp_min(0.0)
    neutral = 1.0 - normalized.abs()
    heatmap = torch.stack((positive + neutral, neutral, negative + neutral), dim=0)
    save_image(heatmap.clamp(0.0, 1.0), directory / f"{stem}_fixed_signed_heatmap_0p05.png")


def mean_dicts(rows: Sequence[dict], keys: Iterable[str]) -> dict:
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in keys
    }


def _gradient_group_norms(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    group_indices: Dict[str, Sequence[int]],
    retain_graph: bool,
) -> dict:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    result = {}
    for name, indices in group_indices.items():
        squared_norm = 0.0
        used = 0
        for index in indices:
            gradient = gradients[index]
            if gradient is None:
                continue
            norm = float(gradient.detach().float().norm())
            squared_norm += norm * norm
            used += 1
        result[name] = {
            "norm": math.sqrt(squared_norm),
            "parameter_tensors_with_gradient": used,
        }
    del gradients
    return result


def _resolve_training_weights(config: dict, global_step: int) -> Tuple[dict, dict]:
    from train_watermark_diffusion import (
        get_active_region_weight,
        get_loss_weights,
        get_noise_curriculum_state,
    )

    weights = get_loss_weights(config, global_step)
    curriculum = get_noise_curriculum_state(
        config,
        global_step,
        weights["lambda_wm"],
        str(config.get("noise_layer", {}).get("type", "none")).lower() != "none",
    )
    weights = dict(weights)
    weights["lambda_region"] = get_active_region_weight(
        weights["lambda_region"],
        global_step,
        config.get("train", {}).get("region_guidance", {}),
    )
    return weights, curriculum


def run_gradient_diagnostics(
    config: dict,
    checkpoint: dict,
    diffusion: GaussianDiffusion,
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    cover: torch.Tensor,
    bits: torch.Tensor,
    noise_name: str,
    strength: float,
    seed: int,
    device: torch.device,
    diagnostic_step: int | None,
) -> dict:
    """Measure unweighted and configured-weight gradient norms on one image."""
    from train_watermark_diffusion import (
        compute_region_guidance_loss,
        get_active_region_weight,
        get_region_outside_settings,
        residual_channel_balance_loss,
        residual_outside_region_loss,
        residual_topk_loss,
        residual_tv_loss,
    )
    from guided_diffusion.unet import AttentionBlock

    seed_everything(seed)
    # AttentionBlock.forward hard-codes the repository's destructive custom
    # checkpoint wrapper even when use_checkpoint=False. That wrapper deletes
    # saved tensors after one backward and therefore cannot support several
    # autograd.grad calls on one graph. Bypass it only inside this diagnostic.
    attention_forwards = []
    for module in model.modules():
        if isinstance(module, AttentionBlock):
            attention_forwards.append((module, module.forward))
            module.forward = module._forward
    model.train()
    decoder.train()
    model.zero_grad(set_to_none=True)
    decoder.zero_grad(set_to_none=True)

    sample_cover = cover[:1].detach()
    sample_bits = bits[:1].detach()
    wm_min = int(config["diffusion"].get("wm_t_min", 0))
    wm_max = int(config["diffusion"].get("wm_t_max", 200))
    timestep_value = max(wm_min, min(wm_max - 1, (wm_min + wm_max) // 2))
    timestep = torch.full((1,), timestep_value, device=device, dtype=torch.long)
    x_t = diffusion.q_sample(
        sample_cover,
        timestep,
        noise=torch.randn_like(sample_cover),
    )
    t_scaled = timestep.float() * (1000.0 / diffusion.num_timesteps)
    cover_01 = (sample_cover + 1.0) * 0.5
    guidance_config = config.get("train", {}).get("region_guidance", {})
    allowance, penalty = build_edge_texture_guidance(cover_01, guidance_config)
    predicted_noise = model(
        x_t=x_t,
        t=t_scaled,
        cover_img=sample_cover,
        wm_bits=sample_bits,
        content_mask=allowance,
    )
    raw_x0 = diffusion._predict_xstart_from_eps(x_t, timestep, predicted_noise)
    pred_x0 = constrain_watermark_residual(
        raw_x0.clamp(-1.0, 1.0),
        sample_cover,
        allowance,
        config.get("train", {}).get("residual_constraint", {}),
    )
    pred_01 = (pred_x0 + 1.0) * 0.5
    residual_01 = pred_01 - cover_01

    layer = build_diagnostic_layer(config, noise_name, device)
    if layer is None:
        attacked_01 = pred_01
        selected_noise = "clean"
    else:
        full_attack = layer(pred_01.float()).float()
        attacked_01 = (
            pred_01.float() + strength * (full_attack - pred_01.float())
        ).clamp(0.0, 1.0)
        selected_noise = (
            layer.get_last_name() if isinstance(layer, MixedNoiseLayer) else noise_name
        )

    clean_logits = decoder(pred_x0)
    degraded_logits = (
        clean_logits
        if layer is None
        else decoder(attacked_01.mul(2.0).sub(1.0))
    )
    losses = {
        "image": F.l1_loss(pred_x0, sample_cover),
        "delta": residual_01.abs().mean() * 2.0,
        "tv": residual_tv_loss(residual_01),
        "topk": residual_topk_loss(
            residual_01,
            config.get("train", {}).get("topk_delta_fraction", 0.01),
        ),
        "channel": residual_channel_balance_loss(residual_01),
        "region": compute_region_guidance_loss(
            residual_01,
            allowance,
            penalty,
            guidance_config,
        ),
        "region_outside": residual_outside_region_loss(
            residual_01,
            allowance,
            eps=float(guidance_config.get("eps", 1e-6)),
        ),
        "wm_clean": F.binary_cross_entropy_with_logits(clean_logits, sample_bits),
        "wm_degraded": F.binary_cross_entropy_with_logits(
            degraded_logits, sample_bits
        ),
    }

    inner_parameters = [
        parameter for parameter in model.inner_unet.parameters() if parameter.requires_grad
    ]
    time_mapper_parameters = [
        parameter for parameter in model.watermark_mlp.parameters() if parameter.requires_grad
    ]
    spatial_mapper_parameters = [
        parameter
        for parameter in model.watermark_map_mlp.parameters()
        if parameter.requires_grad
    ]
    decoder_parameters = [
        parameter for parameter in decoder.parameters() if parameter.requires_grad
    ]
    parameters = (
        inner_parameters
        + time_mapper_parameters
        + spatial_mapper_parameters
        + decoder_parameters
    )
    start_time = len(inner_parameters)
    start_spatial = start_time + len(time_mapper_parameters)
    start_decoder = start_spatial + len(spatial_mapper_parameters)
    group_indices = {
        "inner_unet": range(0, start_time),
        "watermark_mlp": range(start_time, start_spatial),
        "watermark_map_mlp": range(start_spatial, start_decoder),
        "decoder": range(start_decoder, len(parameters)),
    }

    global_step = (
        int(diagnostic_step)
        if diagnostic_step is not None
        else int(checkpoint.get("global_step", 0))
    )
    weights, curriculum = _resolve_training_weights(config, global_step)
    outside_enabled, base_lambda_outside = get_region_outside_settings(
        guidance_config
    )
    lambda_outside = (
        get_active_region_weight(
            base_lambda_outside,
            global_step,
            guidance_config,
        )
        if outside_enabled
        else 0.0
    )
    configured_weights = {
        "image": weights["lambda_img"],
        "delta": weights["lambda_delta"],
        "tv": weights["lambda_tv"],
        "topk": weights["lambda_topk"],
        "channel": weights["lambda_channel"],
        "region": weights["lambda_region"],
        "region_outside": lambda_outside,
        "wm_clean": curriculum["lambda_wm_clean"],
        "wm_degraded": curriculum["lambda_wm_degraded"],
    }

    results = {}
    names = list(losses)
    for loss_index, name in enumerate(names):
        group_norms = _gradient_group_norms(
            losses[name],
            parameters,
            group_indices,
            retain_graph=loss_index < len(names) - 1,
        )
        weight = float(configured_weights[name])
        for group in group_norms.values():
            group["configured_weighted_norm"] = abs(weight) * group["norm"]
        results[name] = {
            "value": float(losses[name].detach()),
            "configured_weight": weight,
            "gradient_groups": group_norms,
        }

    for module, original_forward in attention_forwards:
        module.forward = original_forward
    model.eval()
    decoder.eval()
    return {
        "global_step_used_for_weights": global_step,
        "timestep": timestep_value,
        "requested_noise_layer": noise_name,
        "selected_noise_layer": selected_noise,
        "strength": strength,
        "pred_x0_before_clamp": tensor_stats(raw_x0, (-1.0, 1.0)),
        "pred_x0_clamp": clamp_report(raw_x0, -1.0, 1.0),
        "attacked_stats": tensor_stats(attacked_01),
        "losses": results,
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_value(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def main() -> None:
    args = parse_args()
    if args.num_images < 1:
        raise ValueError("--num_images must be positive")
    if args.random_trials < 1:
        raise ValueError("--random_trials must be positive")
    if args.diagnostic_step is not None and args.diagnostic_step < 0:
        raise ValueError("--diagnostic_step must be non-negative")
    if not 0.0 <= args.strength <= 1.0:
        raise ValueError("--strength must be in [0, 1]")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output path: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config = load_yaml(config_path)
    checkpoint = torch_load_checkpoint(checkpoint_path)
    seed_everything(args.seed)

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    diffusion, model, decoder = build_components(config, device)
    try:
        checkpoint_report = load_weights_strict(checkpoint, model, decoder)
    except Exception:
        model_state = checkpoint.get("diffusion_model", checkpoint.get("model", checkpoint))
        decoder_state = checkpoint.get("decoder", {})
        if isinstance(decoder_state, dict):
            decoder_state = normalize_decoder_state(decoder, decoder_state)
        mismatch_report = {
            "model": compare_state_dicts(model.state_dict(), model_state)
            if isinstance(model_state, dict)
            else {"error": "invalid model state"},
            "decoder": compare_state_dicts(decoder.state_dict(), decoder_state)
            if isinstance(decoder_state, dict)
            else {"error": "invalid decoder state"},
        }
        write_json(output_dir / "checkpoint_load.json", mismatch_report)
        raise
    write_json(output_dir / "checkpoint_load.json", checkpoint_report)

    checkpoint_config = checkpoint.get("config", {})
    provenance = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_has_config": isinstance(checkpoint_config, dict) and bool(checkpoint_config),
        "checkpoint_noise_type": checkpoint_config.get("noise_layer", {}).get("type")
        if isinstance(checkpoint_config, dict)
        else None,
        "diagnostic_seed": args.seed,
        "device": str(device),
    }

    source_path = Path(args.data_dir or config["data"]["val_dir"]).expanduser()
    image_paths = discover_images(source_path, args.num_images)
    cover = load_cover_batch(image_paths, int(config["data"]["image_size"]), device)
    watermark_length = int(config["data"]["watermark_length"])
    bits_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    bits = torch.randint(
        0,
        2,
        (len(image_paths), watermark_length),
        generator=bits_generator,
    ).float().to(device)

    t_start = args.t_start
    if t_start is None:
        t_start = int(config["diffusion"].get("train_t_start", 200))
    seed_everything(args.seed)
    watermarked, embed_trace = embed_watermark_with_trace(
        diffusion, model, cover, bits, t_start, config
    )
    cover_01 = (cover + 1.0) * 0.5
    watermarked_01 = (watermarked + 1.0) * 0.5
    edge_map, texture_map = edge_texture_maps(cover_01)

    requested_layers = [
        name.strip().lower() for name in args.noise_layer.split(",") if name.strip()
    ]
    valid_layers = {"clean", "pimog", "oled", "led", "projector", "mixed"}
    unknown = sorted(set(requested_layers) - valid_layers)
    if unknown:
        raise ValueError(f"Unknown --noise_layer values: {unknown}")
    if not requested_layers:
        raise ValueError("--noise_layer must select at least one branch")

    base_dir = output_dir / "base"
    base_dir.mkdir()
    for index, path in enumerate(image_paths):
        stem = f"{index:03d}"
        save_image(cover_01[index], base_dir / f"{stem}_cover.png")
        save_image(watermarked_01[index], base_dir / f"{stem}_watermarked.png")
        save_image(edge_map[index], base_dir / f"{stem}_edge_map.png")
        save_image(texture_map[index], base_dir / f"{stem}_texture_map.png")
        save_residual_bundle(
            base_dir,
            f"{stem}_embedding_residual",
            watermarked_01[index] - cover_01[index],
        )

    report: Dict[str, Any] = {
        "provenance": provenance,
        "image_paths": [str(path.resolve()) for path in image_paths],
        "watermark_bits": bits.detach().cpu().int().tolist(),
        "t_start": t_start,
        "strength": args.strength,
        "embedding_trace": embed_trace,
        "cover_stats": tensor_stats(cover_01),
        "watermarked_stats": tensor_stats(watermarked_01),
        "variants": {},
    }
    csv_rows: List[dict] = []

    for variant_index, noise_name in enumerate(requested_layers):
        # Each branch gets a deterministic but distinct stream. The embedded
        # batch is reused exactly; only degradation randomness changes.
        seed_everything(args.seed + 1000 + variant_index)
        layer = build_diagnostic_layer(config, noise_name, device)
        degraded_01, degradation_trace = apply_degradation_traced(
            layer, noise_name, watermarked_01, args.strength
        )
        decoder_input = degraded_01.mul(2.0).sub(1.0)
        with torch.no_grad():
            logits = decoder(decoder_input)
            probabilities = torch.sigmoid(logits)
            predicted_bits = (probabilities > 0.5).float()
            bit_acc_per_image = (predicted_bits == bits).float().mean(dim=1)

        embedding_delta = watermarked_01 - cover_01
        observed_delta = degraded_01 - cover_01
        embedding_spatial = spatial_energy_metrics(embedding_delta, edge_map, texture_map)
        observed_spatial = spatial_energy_metrics(observed_delta, edge_map, texture_map)
        variant_dir = output_dir / noise_name
        variant_dir.mkdir()

        per_image = []
        for index in range(len(image_paths)):
            stem = f"{index:03d}"
            save_image(degraded_01[index], variant_dir / f"{stem}_degraded.png")
            save_residual_bundle(
                variant_dir,
                f"{stem}_observed_residual",
                observed_delta[index],
            )
            fft_embedding, spectrum_rgb, spectrum_gray = fft_metrics_single(
                embedding_delta[index]
            )
            fft_observed, observed_spectrum_rgb, observed_spectrum_gray = fft_metrics_single(
                observed_delta[index]
            )
            save_image(
                normalize_for_display(spectrum_rgb),
                variant_dir / f"{stem}_embedding_fft_rgb.png",
            )
            save_image(
                normalize_for_display(spectrum_gray),
                variant_dir / f"{stem}_embedding_fft_gray.png",
            )
            save_image(
                normalize_for_display(observed_spectrum_rgb),
                variant_dir / f"{stem}_observed_fft_rgb.png",
            )
            save_image(
                normalize_for_display(observed_spectrum_gray),
                variant_dir / f"{stem}_observed_fft_gray.png",
            )

            item = {
                "index": index,
                "path": str(image_paths[index].resolve()),
                "bit_accuracy": float(bit_acc_per_image[index]),
                "logits_mean": float(logits[index].mean()),
                "logits_std": float(logits[index].std(unbiased=False)),
                "embedding_spatial": embedding_spatial[index],
                "observed_spatial": observed_spatial[index],
                "embedding_fft": fft_embedding,
                "observed_fft": fft_observed,
                "embedding_channels": channel_metrics_single(embedding_delta[index]),
                "observed_channels": channel_metrics_single(observed_delta[index]),
            }
            per_image.append(item)
            csv_rows.append(
                {
                    "variant": noise_name,
                    "selected_noise_layer": degradation_trace.get(
                        "selected_noise_layer", noise_name
                    ),
                    "index": index,
                    "bit_accuracy": item["bit_accuracy"],
                    "logits_std": item["logits_std"],
                    **{f"embedding_{key}": value for key, value in embedding_spatial[index].items()},
                    **{f"observed_{key}": value for key, value in observed_spatial[index].items()},
                    "embedding_fft_peak_ratio": fft_embedding["fft_peak_ratio"],
                    "embedding_axis_energy_ratio": fft_embedding["axis_energy_ratio"],
                    "embedding_periodic_energy_ratio": fft_embedding[
                        "periodic_energy_ratio"
                    ],
                    "observed_fft_peak_ratio": fft_observed["fft_peak_ratio"],
                    "observed_axis_energy_ratio": fft_observed["axis_energy_ratio"],
                    "observed_periodic_energy_ratio": fft_observed[
                        "periodic_energy_ratio"
                    ],
                }
            )

        report["variants"][noise_name] = {
            "selected_noise_layer": degradation_trace.get("selected_noise_layer", noise_name),
            "bit_accuracy": float(bit_acc_per_image.mean()),
            "logits_mean": float(logits.mean()),
            "logits_std": float(logits.std(unbiased=False)),
            "decoder_input_stats": tensor_stats(decoder_input, (-1.0, 1.0)),
            "degraded_output_stats": tensor_stats(degraded_01),
            "degradation_trace": degradation_trace,
            "per_image": per_image,
            "mean_embedding_spatial": mean_dicts(
                embedding_spatial,
                (
                    "edge_energy_ratio",
                    "flat_energy_ratio",
                    "texture_energy_ratio",
                    "smooth_energy_ratio",
                ),
            ),
            "mean_observed_spatial": mean_dicts(
                observed_spatial,
                (
                    "edge_energy_ratio",
                    "flat_energy_ratio",
                    "texture_energy_ratio",
                    "smooth_energy_ratio",
                ),
            ),
        }
        if noise_name != "clean":
            seed_everything(args.seed + 5000 + variant_index)
            trial_layer = build_diagnostic_layer(config, noise_name, device)
            trial_outputs = []
            trial_records = []
            for trial_index in range(args.random_trials):
                trial_output, trial_trace = apply_degradation_traced(
                    trial_layer,
                    noise_name,
                    watermarked_01,
                    args.strength,
                )
                trial_outputs.append(trial_output)
                trial_records.append(
                    {
                        "trial": trial_index,
                        "selected_noise_layer": trial_trace.get(
                            "selected_noise_layer", noise_name
                        ),
                        "mean_abs_difference_from_trial_0": (
                            0.0
                            if trial_index == 0
                            else float((trial_output - trial_outputs[0]).abs().mean())
                        ),
                        "output_stats": tensor_stats(trial_output),
                        "random_parameters": trial_trace.get("random_parameters", []),
                    }
                )
            report["variants"][noise_name]["randomness_trials"] = trial_records

    if args.gradient_diagnostics:
        gradient_noise_name = next(
            (name for name in requested_layers if name != "clean"),
            "clean",
        )
        report["gradient_diagnostics"] = run_gradient_diagnostics(
            config=config,
            checkpoint=checkpoint,
            diffusion=diffusion,
            model=model,
            decoder=decoder,
            cover=cover,
            bits=bits,
            noise_name=gradient_noise_name,
            strength=args.strength,
            seed=args.seed + 9000,
            device=device,
            diagnostic_step=args.diagnostic_step,
        )

    write_json(output_dir / "diagnostics.json", report)
    fieldnames = list(csv_rows[0].keys())
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    write_json(
        output_dir / "run_config.json",
        {
            "arguments": vars(args),
            "resolved_output_dir": str(output_dir),
            "resolved_data_source": str(source_path.resolve()),
            "checkpoint_load": checkpoint_report,
        },
    )
    print(f"[Diagnosis] Wrote new diagnostic directory: {output_dir}")
    print(f"[Diagnosis] Images: {len(image_paths)}, variants: {requested_layers}")
    print(f"[Diagnosis] Metrics: {output_dir / 'metrics.csv'}")
    print(f"[Diagnosis] Full trace: {output_dir / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
