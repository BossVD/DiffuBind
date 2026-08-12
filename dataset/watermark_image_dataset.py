"""
Watermark Image Dataset.

Reads cover images from a directory tree (train/val), resizes to a fixed size,
normalizes to [-1, 1], and generates deterministic watermark bits for each
image from its relative path, a configurable seed, and optionally the epoch.

Returns:
    dict with "image": [3, H, W] in [-1, 1], "wm_bits": [wm_length] 0/1 float
"""
import os
import glob
import hashlib
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class WatermarkImageDataset(Dataset):
    """Dataset that loads cover images and assigns random watermark bits."""

    def __init__(
        self,
        data_dir: str,
        image_size: int = 128,
        watermark_length: int = 30,
        watermark_seed: int = 42,
        watermark_mode: str = 'fixed',
        is_train: bool = False,
        max_images: int = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.watermark_length = watermark_length
        self.watermark_seed = watermark_seed
        if watermark_mode not in {'fixed', 'per_epoch', 'random', 'deterministic_random'}:
            raise ValueError(
                "watermark_mode must be 'fixed', 'per_epoch', 'random', "
                "or 'deterministic_random'"
            )
        self.watermark_mode = watermark_mode
        self.epoch = 0
        self.is_train = is_train

        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        if len(self.image_paths) == 0:
            self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        if len(self.image_paths) == 0:
            self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpeg")))
        if len(self.image_paths) == 0:
            self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*/*.png")))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {data_dir}")

        # Limit dataset size for fast debugging
        if max_images is not None and max_images > 0 and max_images < len(self.image_paths):
            import random
            random.seed(42)
            random.shuffle(self.image_paths)
            self.image_paths = self.image_paths[:max_images]
            self.image_paths.sort()

        crop = transforms.RandomCrop(image_size) if is_train else transforms.CenterCrop(image_size)
        self.transform = transforms.Compose([
            # Integer Resize preserves aspect ratio and scales the shorter edge.
            transforms.Resize(image_size, antialias=True),
            crop,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def set_epoch(self, epoch):
        """Select the deterministic watermark sequence for this epoch."""
        self.epoch = int(epoch)

    def _watermark_for_path(self, img_path):
        """Return stable bits for an image without relying on process RNG state."""
        relative_path = os.path.relpath(img_path, self.data_dir).replace('\\', '/')
        bits = []
        counter = 0
        if self.watermark_mode == 'per_epoch':
            epoch_key = self.epoch
        elif self.watermark_mode == 'random':
            epoch_key = f"random:{self.epoch}"
        else:
            epoch_key = 'deterministic_random'
        while len(bits) < self.watermark_length:
            payload = (
                f"{self.watermark_seed}:{epoch_key}:{relative_path}:{counter}"
            ).encode('utf-8')
            digest = hashlib.sha256(payload).digest()
            for byte in digest:
                bits.extend((byte >> shift) & 1 for shift in range(7, -1, -1))
            counter += 1
        return torch.tensor(bits[:self.watermark_length], dtype=torch.float32)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: failed to load {img_path}: {e}. Returning a different image.")
            return self.__getitem__((idx + 1) % len(self.image_paths))

        image = self.transform(img)  # [3, H, W], range [-1, 1]

        # Stable across worker counts, machines, and dataset ordering. In
        # per_epoch mode it changes reproducibly once per training epoch.
        wm_bits = self._watermark_for_path(img_path)

        return {
            "image": image,
            "wm_bits": wm_bits,
        }
