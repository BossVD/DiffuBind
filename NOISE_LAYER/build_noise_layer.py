"""Factory and composition utilities for degradation layers."""

import torch
import torch.nn as nn

from .PIMoG_Layer import PIMoGLayer
from .Projector_Layer import ProjectorSimulator
from .OLED_Layer import OLEDNoiseLayer
from .LED_Layer import LEDNoiseLayer


class MixedNoiseLayer(nn.Module):
    """Select one degradation layer for the whole batch on each forward call."""

    def __init__(self, layers, probs=None, names=None):
        super().__init__()
        if not layers:
            raise ValueError("layers must not be empty")
        self.layers = nn.ModuleList(layers)
        if names is None:
            names = [layer.__class__.__name__ for layer in layers]
        if len(names) != len(layers):
            raise ValueError("names must match layers")
        self.names = [str(name).lower() for name in names]
        self.last_index = None
        self.last_name = None
        if probs is None:
            probs = [1.0 / len(layers)] * len(layers)
        if len(probs) != len(layers) or any(float(p) < 0 for p in probs):
            raise ValueError("probs must be non-negative and match layers")
        probs_tensor = torch.tensor(probs, dtype=torch.float32)
        if probs_tensor.sum() <= 0:
            raise ValueError("at least one probability must be positive")
        self.register_buffer("probs", probs_tensor / probs_tensor.sum())

    def _distribution(self, candidates=None, probs=None):
        if candidates is None:
            indices = list(range(len(self.layers)))
        else:
            requested = [str(name).lower() for name in candidates]
            if not requested:
                raise ValueError("candidates must not be empty")
            unknown = [name for name in requested if name not in self.names]
            if unknown:
                raise ValueError(
                    f"Unknown mixed noise candidates: {unknown}. Available: {self.names}"
                )
            indices = [self.names.index(name) for name in requested]

        if probs is None:
            selected_probs = self.probs[indices]
        else:
            if len(probs) != len(indices) or any(float(p) < 0 for p in probs):
                raise ValueError(
                    "probs must be non-negative and match the selected candidates"
                )
            selected_probs = self.probs.new_tensor(probs)
        if selected_probs.sum() <= 0:
            raise ValueError("at least one selected probability must be positive")
        return indices, selected_probs / selected_probs.sum()

    def forward(self, x, candidates=None, probs=None):
        """Apply one layer, optionally using a curriculum-specific distribution."""
        indices, selected_probs = self._distribution(candidates, probs)
        local_index = torch.multinomial(selected_probs, 1).item()
        index = indices[local_index]
        self.last_index = index
        self.last_name = self.names[index]
        return self.layers[index](x)

    def get_last_name(self):
        return self.last_name


def get_noise_layer_type(config):
    """Return the degradation type selected by ``noise_layer.type``."""
    return str(config.get("noise_layer", {}).get("type", "none")).lower()


def _build_single_noise_layer(noise_type, noise_cfg):
    """Build one concrete degradation layer from the ``noise_layer`` section."""
    if noise_type == "pimog":
        return PIMoGLayer(**noise_cfg.get("pimog", {}))
    if noise_type == "oled":
        return OLEDNoiseLayer(**noise_cfg.get("oled", {}))
    if noise_type == "led":
        return LEDNoiseLayer(**noise_cfg.get("led", {}))
    if noise_type == "projector":
        return ProjectorSimulator(**noise_cfg.get("projector", {}))
    raise ValueError(f"Unsupported mixed noise layer candidate: {noise_type}")


def build_noise_layer(config):
    """Build ``none``, concrete screen layers, or ``mixed`` from a config dict."""
    noise_cfg = config.get("noise_layer", {})
    noise_type = get_noise_layer_type(config)

    if noise_type == "none":
        return nn.Identity()
    if noise_type in {"pimog", "oled", "led", "projector"}:
        return _build_single_noise_layer(noise_type, noise_cfg)
    if noise_type == "mixed":
        mixed_cfg = noise_cfg.get("mixed", {})
        candidates = mixed_cfg.get("candidates", None)
        if candidates is None:
            # Backward compatible default for existing configs with
            # mixed_probs: [pimog_prob, projector_prob].
            candidates = ["pimog", "projector"]
            probs = noise_cfg.get("mixed_probs", [0.5, 0.5])
        else:
            candidates = [str(candidate).lower() for candidate in candidates]
            probs = mixed_cfg.get("probs", noise_cfg.get("mixed_probs", None))
        return MixedNoiseLayer(
            layers=[_build_single_noise_layer(candidate, noise_cfg) for candidate in candidates],
            probs=probs,
            names=candidates,
        )
    raise ValueError(
        f"Unsupported noise layer type: {noise_type}. "
        "Expected one of: none, pimog, oled, led, projector, mixed"
    )
