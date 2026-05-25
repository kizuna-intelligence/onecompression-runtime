"""FireRed-Image-Edit int4 runtime — packed-int4 ``QwenImageTransformer2DModel``.

Loads the 20B dual-stream Qwen-Image backbone (OneCompression GPTQ-packed,
~12GB int4) and runs each quantized Linear with a fused int4 GEMM (GemLite or
the bundled Triton kernel). See ``example/firered/generate.py``.

Copyright 2025-2026 Kizuna Intelligence.
"""
from __future__ import annotations

from .loader import load_int4_transformer

__all__ = ["load_int4_transformer"]
