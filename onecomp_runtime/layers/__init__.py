"""Int4 inference layers shared across OneCompression runtimes.

- :class:`FusedInt4Linear` / :func:`fused_int4_gemm` — Triton dequant+GEMM for
  AutoGPTQ-v1 packed int4 weights (groupsize 32, fp16/bf16/fp32 in, fp16 out).
- :class:`GemLiteInt4Linear` / :func:`gemlite_available` — GemLite kernel wrapper
  (fp16 I/O) for the same packed format.
- :class:`PackedRTNLinear` / :class:`PackedEmbedding` — RTN uint8-nibble layers
  that dequantize on the fly (encoders, AdaLN extras, embedding tables).
- :class:`Dsv4PackedLinear` / :func:`dequant_dsv4_packed` — int4/int2 packed Linear
  for DeepSeek-V4-Flash QEP weights (offloadable MoE experts, on-the-fly dequant).
- :class:`PackedInt4Conv1d` / :class:`PackedInt4ConvTranspose1d` /
  :func:`replace_conv_with_packed` — int4 conv layers (DAC-VAE style codecs).

Copyright 2025-2026 Kizuna Intelligence / Fujitsu Ltd. MIT License.
"""
from __future__ import annotations

from .fused_int4_linear import FusedInt4Linear, fused_int4_gemm
from .gemlite_int4_linear import GemLiteInt4Linear, gemlite_available
from .packed_conv import (
    PackedInt4Conv1d,
    PackedInt4ConvTranspose1d,
    quantize_conv_module,
    quantize_conv_weight,
    replace_conv_with_packed,
)
from .packed_linear import PackedEmbedding, PackedRTNLinear
from .dsv4_packed_linear import Dsv4PackedLinear, dequant_dsv4_packed

__all__ = [
    "FusedInt4Linear",
    "fused_int4_gemm",
    "GemLiteInt4Linear",
    "gemlite_available",
    "PackedRTNLinear",
    "PackedEmbedding",
    "Dsv4PackedLinear",
    "dequant_dsv4_packed",
    "PackedInt4Conv1d",
    "PackedInt4ConvTranspose1d",
    "replace_conv_with_packed",
    "quantize_conv_module",
    "quantize_conv_weight",
]
