"""
Thin loader for the byte-identical official PIMoG PIMoG_Layer.py.

This wrapper does not resize, clamp, interpolate, or otherwise alter the
official ScreenShooting output. The only compatibility work is exposing old
Kornia top-level API names that were moved in newer Kornia releases.
"""

import os
import importlib.util
import kornia
import torch.nn as nn

from kornia.geometry.transform import (
    get_perspective_transform,
    get_rotation_matrix2d,
    warp_affine,
    warp_perspective,
)

# Official PIMoG calls these through the old ``kornia.<name>`` API. Aliasing
# moved symbols preserves its source and numerical implementation unchanged.
_KORNIA_COMPAT_ALIASES = {
    'get_perspective_transform': get_perspective_transform,
    'get_rotation_matrix2d': get_rotation_matrix2d,
    'warp_affine': warp_affine,
    'warp_perspective': warp_perspective,
}
for _name, _function in _KORNIA_COMPAT_ALIASES.items():
    if not hasattr(kornia, _name):
        setattr(kornia, _name, _function)

# Load the original PIMoG ScreenShooting class directly from disk
_pimog_path = os.path.join(os.path.dirname(__file__), '..', 'NOISE_LAYER', 'PIMoG_Layer.py')
_pimog_path = os.path.normpath(_pimog_path)
_spec = importlib.util.spec_from_file_location("_pimog_noise_layer", _pimog_path)
_pimog_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pimog_module)
_PIMoG_ScreenShooting = _pimog_module.ScreenShooting


class ScreenShootingSimulator(nn.Module):
    """
    Wraps the original PIMoG ``ScreenShooting`` noise layer.

    Calls the official ``ScreenShooting`` forward path directly.
    """

    def __init__(self):
        super().__init__()
        self._pimog = _PIMoG_ScreenShooting()

    def forward(self, x):
        """
        Args:
            x: tensor passed unchanged to official PIMoG
        Returns:
            exact output of official PIMoG ScreenShooting
        """
        return self._pimog(x)
