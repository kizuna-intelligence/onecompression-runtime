"""Qwen3.6-27B int4 text-decoder runtime — packed-int4 ``Qwen3_5ForCausalLM``.

Loads the 64-layer GatedDeltaNet/attention hybrid decoder (OneCompression
GPTQ-packed, {4,8}-bit AutoBit floor, ~13GB int4 weights) and runs each
quantized Linear with a GemLite Triton GEMM. See ``example/qwen36/generate.py``.

Copyright 2025-2026 Kizuna Intelligence.
"""
from __future__ import annotations

from .loader import load_int4_qwen36

__all__ = ["load_int4_qwen36"]
