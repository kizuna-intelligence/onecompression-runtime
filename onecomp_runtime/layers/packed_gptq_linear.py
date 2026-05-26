"""Packed AutoGPTQ Linear with pure-torch on-the-fly dequant.

Unlike :class:`GemLiteInt4Linear` (which needs the GemLite Triton kernels) and
:class:`FusedInt4Linear` (a custom Triton GEMM limited to wbits=4/groupsize=32),
this layer keeps the packed buffers in VRAM and reconstructs the fp weight with
:func:`dequant_gptq_to_fp` — *plain torch* — inside ``forward``, then runs
``F.linear``. Only one layer's fp weight is ever materialised at a time, so a
27B model stays at its packed footprint (~13-18 GiB) instead of the ~54 GiB a
full eager dequant needs.

It is slower than a fused kernel (a dequant runs every call), but it has no
backend dependency: it runs anywhere torch runs, including ROCm / AMD RDNA4
where neither GemLite nor the CUDA-tuned fused kernel are available. It also
handles the {4,8}-bit mixed, groupsize-128 GPTQ checkpoints (e.g. Qwen3.6-27B)
that ``FusedInt4Linear`` cannot.

Copyright 2025-2026 Kizuna Intelligence.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..quant_utils import dequant_gptq_to_fp


class PackedGPTQLinear(nn.Module):
    """nn.Linear-compatible packed GPTQ layer (4- or 8-bit, any groupsize).

    Holds ``qweight``/``scales``/``qzeros``/``g_idx`` verbatim and dequantises
    to ``x.dtype`` per forward call. Backend-free (pure torch)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        g_idx: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        groupsize: int = 128,
        wbits: int = 4,
        v1: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.groupsize = int(groupsize)
        self.wbits = int(wbits)
        self.v1 = bool(v1)
        if g_idx is None:
            g_idx = (torch.arange(in_features) // groupsize).to(torch.long)
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("qzeros", qzeros.contiguous())
        self.register_buffer("g_idx", g_idx.to(torch.long).contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.contiguous())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequant_gptq_to_fp(
            qweight=self.qweight, scales=self.scales, qzeros=self.qzeros,
            g_idx=self.g_idx, in_features=self.in_features,
            out_features=self.out_features, dtype=x.dtype,
            v1=self.v1, wbits=self.wbits,
        )
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def _apply(self, fn, recurse=True):
        # Keep packed integer buffers int32 across .to(dtype=...) on the parent.
        qw, qz = self.qweight, self.qzeros
        result = super()._apply(fn, recurse=recurse)
        if self.qweight.dtype != torch.int32:
            self.qweight = qw.to(self.qweight.device)
        if self.qzeros.dtype != torch.int32:
            self.qzeros = qz.to(self.qzeros.device)
        return result

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"wbits={self.wbits}, groupsize={self.groupsize}, packed=True")
