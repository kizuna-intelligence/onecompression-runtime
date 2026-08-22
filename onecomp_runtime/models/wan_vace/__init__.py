"""Wan2.1 VACE 14B int4 transformer runtime."""
from __future__ import annotations

from .loader import load_int4_wan_vace_transformer
from .multicontrol import enable_shared_vace_multicontrol, expand_layer_scale

__all__ = [
    "enable_shared_vace_multicontrol",
    "expand_layer_scale",
    "load_int4_wan_vace_transformer",
]
