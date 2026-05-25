"""Packed int4 / int2 Linear for DeepSeek-V4-Flash QEP weights.

Loads the per-layer ``layer_{L}.pt`` produced by ``Dsv4StreamingQEP._pack``:
each module dict carries ``qweight_packed`` (uint8, ``8//wbits`` codes per byte
along the in-dim), ``scales``/``qzeros`` (fp16, groupwise), ``wbits``, ``groupsize``,
``in_features``.  Dense attn + shared experts are int4, routed experts int2.

The forward dequantizes on the fly and runs ``F.linear`` — for the MoE experts the
cost is dominated by the CPU->GPU weight transfer of the (few) routed experts, not
the matmul, so an on-the-fly dequant matches a fused kernel here (see the cmda
expert-offload finding).  Keeping the packed buffers on CPU and materializing only
the routed experts per token is what makes the 284B model fit 24GB.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def dequant_dsv4_packed(p: dict, dtype: torch.dtype = torch.bfloat16,
                        device=None) -> torch.Tensor:
    """Inverse of ``Dsv4StreamingQEP._pack`` → dense (out, in) weight.

    GPTQ asymmetric groupwise: ``w = (q - qzero) * scale`` with one scale/zero per
    ``groupsize`` columns (group index = column // groupsize)."""
    wbits = int(p["wbits"])
    gs = int(p["groupsize"])
    per = 8 // wbits
    packed = p["qweight_packed"]
    if device is not None:
        packed = packed.to(device)
    packed = packed.to(torch.int32)                       # [out, ncol]
    out_f, ncol = packed.shape
    shift = torch.arange(per, device=packed.device, dtype=torch.int32) * wbits
    q = (packed.unsqueeze(-1) >> shift) & ((1 << wbits) - 1)  # [out, ncol, per]
    q = q.reshape(out_f, ncol * per)[:, : int(p["in_features"])].to(torch.int32)
    in_f = q.shape[1]

    scales = p["scales"].to(device=q.device, dtype=torch.float32)
    qzeros = p["qzeros"].to(device=q.device, dtype=torch.float32)
    # normalize to (num_groups, out_features)
    num_groups = (in_f + gs - 1) // gs if gs > 0 else 1
    if gs <= 0:
        s = scales.reshape(out_f, 1)
        z = qzeros.reshape(out_f, 1)
        w = (q.float() - z) * s
        return w.to(dtype)
    if scales.shape == (out_f, num_groups):
        scales, qzeros = scales.T, qzeros.T               # -> (groups, out)
    g_idx = (torch.arange(in_f, device=q.device) // gs)
    s_exp = scales[g_idx, :].T                             # (out, in)
    z_exp = qzeros[g_idx, :].T
    w = (q.float() - z_exp) * s_exp
    return w.to(dtype)


class Dsv4PackedLinear(nn.Module):
    """nn.Linear-compatible packed int4/int2 Linear for DSV4 QEP weights.

    ``materialize_device``: where to dequant the weight for the matmul (defaults to
    the input device).  Buffers stay where they were registered — keep them on CPU
    for offloaded experts, on CUDA for the hot dense path."""

    def __init__(self, packed: dict, bias: torch.Tensor | None = None,
                 buffer_device="cpu") -> None:
        super().__init__()
        self.in_features = int(packed["in_features"])
        self.out_features = int(packed["qweight_packed"].shape[0])
        self.wbits = int(packed["wbits"])
        self.groupsize = int(packed["groupsize"])
        self.register_buffer("qweight_packed",
                             packed["qweight_packed"].to(buffer_device).contiguous())
        self.register_buffer("scales", packed["scales"].to(buffer_device).contiguous())
        self.register_buffer("qzeros", packed["qzeros"].to(buffer_device).contiguous())
        if bias is not None:
            self.register_parameter("bias", nn.Parameter(bias.detach(), requires_grad=False))
        else:
            self.bias = None

    def _packed_dict(self) -> dict:
        return {"qweight_packed": self.qweight_packed, "scales": self.scales,
                "qzeros": self.qzeros, "wbits": self.wbits,
                "groupsize": self.groupsize, "in_features": self.in_features}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = dequant_dsv4_packed(self._packed_dict(), dtype=x.dtype, device=x.device)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"packed=int{self.wbits}, groupsize={self.groupsize}")
