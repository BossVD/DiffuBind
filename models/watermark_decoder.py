"""
Watermark decoder (extractor).

Input:  image in [-1, 1] range, shape [B, 3, H, W]
Output: raw logits in [B, watermark_length]

The decoder does not apply sigmoid or image normalization internally. Training
should keep using BCEWithLogitsLoss and compute bit accuracy from sigmoid(logits).
"""
import torch
import torch.nn as nn


def get_num_groups(channels, max_groups=8):
    """Pick the largest group count up to max_groups that divides channels."""
    max_groups = min(int(max_groups), int(channels))
    for groups in range(max_groups, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    """Conv2d -> GroupNorm -> SiLU."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        norm_groups=8,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.GroupNorm(get_num_groups(out_channels, norm_groups), out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block that preserves spatial resolution."""

    def __init__(self, in_channels, out_channels, norm_groups=8):
        super().__init__()
        self.conv1 = ConvGNAct(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_groups=norm_groups,
        )
        self.conv2 = ConvGNAct(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_groups=norm_groups,
        )
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.shortcut(x)


class DownsampleBlock(ConvGNAct):
    """Stride-2 downsampling block."""

    def __init__(self, channels, norm_groups=8):
        super().__init__(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_groups=norm_groups,
        )


class SimpleConvBlock(nn.Module):
    """Original Conv2d -> BatchNorm2d -> SiLU block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class SimpleWatermarkDecoder(nn.Module):
    """
    Original lightweight CNN decoder kept for ablation experiments.

    Architecture: 4 downsampling conv blocks -> AdaptiveAvgPool2d(1) -> Linear.
    """

    def __init__(self, watermark_length=30):
        super().__init__()
        self.watermark_length = watermark_length
        self.encoder = nn.Sequential(
            SimpleConvBlock(3, 32),
            SimpleConvBlock(32, 64),
            SimpleConvBlock(64, 128),
            SimpleConvBlock(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, watermark_length)

    def forward(self, x):
        feat = self.encoder(x).flatten(1)
        return self.fc(feat)


class ResidualMultiScaleWatermarkDecoder(nn.Module):
    """
    Residual multi-scale decoder for robust watermark extraction.

    The 32x32, 16x16, and 8x8 feature maps are globally pooled and concatenated
    before the MLP head. This gives the decoder access to both mid-level local
    traces and lower-resolution robust features.
    """

    def __init__(
        self,
        watermark_length=30,
        base_channels=32,
        hidden_dim=512,
        dropout=0.1,
        norm_groups=8,
        use_multiscale=True,
    ):
        super().__init__()
        self.watermark_length = watermark_length
        self.base_channels = base_channels
        self.hidden_dim = hidden_dim
        self.use_multiscale = use_multiscale

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = ConvGNAct(3, c1, stride=1, norm_groups=norm_groups)

        self.stage1 = nn.Sequential(
            ResidualBlock(c1, c1, norm_groups=norm_groups),
            DownsampleBlock(c1, norm_groups=norm_groups),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(c1, c2, norm_groups=norm_groups),
            ResidualBlock(c2, c2, norm_groups=norm_groups),
            DownsampleBlock(c2, norm_groups=norm_groups),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(c2, c3, norm_groups=norm_groups),
            ResidualBlock(c3, c3, norm_groups=norm_groups),
            DownsampleBlock(c3, norm_groups=norm_groups),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(c3, c4, norm_groups=norm_groups),
            ResidualBlock(c4, c4, norm_groups=norm_groups),
            DownsampleBlock(c4, norm_groups=norm_groups),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        head_in_dim = c2 + c3 + c4 if use_multiscale else c4
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, watermark_length),
        )

    def forward(self, x):
        h = self.stem(x)
        h64 = self.stage1(h)
        h32 = self.stage2(h64)
        h16 = self.stage3(h32)
        h8 = self.stage4(h16)

        if self.use_multiscale:
            feat32 = self.pool(h32).flatten(1)
            feat16 = self.pool(h16).flatten(1)
            feat8 = self.pool(h8).flatten(1)
            feat = torch.cat([feat32, feat16, feat8], dim=1)
        else:
            feat = self.pool(h8).flatten(1)
        return self.head(feat)


class WatermarkDecoder(nn.Module):
    """
    Backward-compatible decoder entry point.

    type="residual_multiscale" is the default. type="simple" keeps the original
    baseline available for ablation without changing external imports.
    """

    def __init__(
        self,
        watermark_length=30,
        base_channels=32,
        hidden_dim=512,
        dropout=0.1,
        norm_groups=8,
        use_multiscale=True,
        type="residual_multiscale",
    ):
        super().__init__()
        decoder_type = str(type).lower()
        self.decoder_type = decoder_type

        if decoder_type == "simple":
            self.decoder = SimpleWatermarkDecoder(
                watermark_length=watermark_length,
            )
        elif decoder_type == "residual_multiscale":
            self.decoder = ResidualMultiScaleWatermarkDecoder(
                watermark_length=watermark_length,
                base_channels=base_channels,
                hidden_dim=hidden_dim,
                dropout=dropout,
                norm_groups=norm_groups,
                use_multiscale=use_multiscale,
            )
        else:
            raise ValueError(
                "Unsupported decoder type: "
                f"{type}. Expected 'simple' or 'residual_multiscale'."
            )

    def forward(self, x):
        return self.decoder(x)


def build_watermark_decoder(config=None, watermark_length=30):
    """Build a WatermarkDecoder from an optional project config dict."""
    decoder_cfg = {}
    if config is not None:
        decoder_cfg = dict(config.get("decoder", {}))
    return WatermarkDecoder(
        watermark_length=watermark_length,
        type=decoder_cfg.get("type", "residual_multiscale"),
        base_channels=decoder_cfg.get("base_channels", 32),
        hidden_dim=decoder_cfg.get("hidden_dim", 512),
        dropout=decoder_cfg.get("dropout", 0.1),
        norm_groups=decoder_cfg.get("norm_groups", 8),
        use_multiscale=decoder_cfg.get("use_multiscale", True),
    )


def load_watermark_decoder_state(decoder, checkpoint_state):
    """
    Load compatible decoder tensors and report incompatible keys.

    This avoids crashes when resuming or evaluating checkpoints created with a
    previous decoder architecture.
    """
    current_state = decoder.state_dict()
    candidate_state = checkpoint_state
    if not any(key.startswith("decoder.") for key in checkpoint_state):
        candidate_state = {
            f"decoder.{key}": value for key, value in checkpoint_state.items()
        }

    compatible_state = {
        key: value
        for key, value in candidate_state.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    missing_keys = sorted(set(current_state) - set(compatible_state))
    unexpected_keys = sorted(set(candidate_state) - set(current_state))
    mismatched_keys = sorted(
        key for key, value in candidate_state.items()
        if key in current_state and current_state[key].shape != value.shape
    )

    decoder.load_state_dict(compatible_state, strict=False)
    return missing_keys, unexpected_keys, mismatched_keys
