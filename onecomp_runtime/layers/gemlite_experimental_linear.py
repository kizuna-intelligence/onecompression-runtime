"""Experimental GemLite activation-quantized Linear layers.

These layers are intentionally opt-in. They are used to test activation
quantization on OneCompression GPTQ checkpoints without changing the default
W4A16 runtime path.
"""
from __future__ import annotations

import torch
from torch import nn

from ..quant_utils import dequant_gptq_to_fp, unpack_int_weights, unpack_zeros


class GemLiteA8W4HQQLinear(nn.Module):
    """W4A8-style Linear using GPTQ int weights and dynamic 8-bit activations.

    The GPTQ integer weights, scales, and zero points are preserved and handed
    to GemLite's HQQ-compatible dynamic activation path. Activations are
    quantized dynamically by GemLite.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        bias: torch.Tensor | None = None,
        groupsize: int = 32,
        v1: bool = True,
        wbits: int = 4,
        device: torch.device | str = "cuda:0",
    ):
        super().__init__()
        if wbits != 4:
            raise ValueError(f"GemLiteA8W4HQQLinear requires wbits=4, got {wbits}")
        if groupsize != 32:
            raise ValueError(
                f"GemLiteA8W4HQQLinear requires groupsize=32, got {groupsize}"
            )
        dev = torch.device(device)
        if dev.type != "cuda":
            raise RuntimeError("GemLiteA8W4HQQLinear requires a CUDA device")

        from gemlite.helper import A8W4_HQQ_INT_dynamic

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self._device = dev

        w_q = unpack_int_weights(qweight, wbits, (out_features, in_features))
        w_q = w_q.to(torch.uint8).contiguous()
        zeros = unpack_zeros(qzeros, wbits, out_features)
        if v1:
            zeros = (zeros + 1) & ((1 << wbits) - 1)

        # GemLite's HQQ path expects [out, in] weights and [out, groups] meta.
        scales_hqq = scales.t().contiguous()
        zeros_hqq = zeros.t().contiguous()
        self._gl = A8W4_HQQ_INT_dynamic(
            device=str(dev),
            dtype=torch.float16,
            fp32_scale=False,
        ).from_weights(w_q, scales_hqq, zeros_hqq, bias=bias)

    @property
    def bias(self):
        return getattr(self._gl, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        return self._gl(x.to(torch.float16)).to(orig_dtype)

    @torch.no_grad()
    def warmup(self, m_values=(64, 128, 256, 1024, 4096)) -> None:
        for m in m_values:
            if m <= 0:
                continue
            x = torch.zeros(
                (int(m), self.in_features),
                dtype=torch.float16,
                device=self._device,
            )
            _ = self.forward(x)


class GemLiteNVFP4Linear(nn.Module):
    """Experimental W4A4 Linear using GemLite NVFP4 dynamic activations.

    This path re-quantizes the dequantized GPTQ weight into GemLite NVFP4. It is
    intended only for targeted layer ablations, not as a default full-model
    backend.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        bias: torch.Tensor | None = None,
        groupsize: int = 32,
        v1: bool = True,
        wbits: int = 4,
        device: torch.device | str = "cuda:0",
    ):
        super().__init__()
        if wbits != 4:
            raise ValueError(f"GemLiteNVFP4Linear requires wbits=4, got {wbits}")
        if groupsize != 32:
            raise ValueError(
                f"GemLiteNVFP4Linear requires groupsize=32, got {groupsize}"
            )
        dev = torch.device(device)
        if dev.type != "cuda":
            raise RuntimeError("GemLiteNVFP4Linear requires a CUDA device")

        from gemlite.helper import A4W4_NVFP_dynamic

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self._device = dev

        g_idx = torch.arange(in_features, dtype=torch.long) // groupsize
        weight = dequant_gptq_to_fp(
            qweight=qweight,
            scales=scales,
            qzeros=qzeros,
            g_idx=g_idx,
            in_features=in_features,
            out_features=out_features,
            dtype=torch.float16,
            v1=v1,
            wbits=wbits,
        ).to(dev)

        lin = nn.Linear(
            in_features,
            out_features,
            bias=bias is not None,
            device=dev,
            dtype=torch.float16,
        )
        with torch.no_grad():
            lin.weight.copy_(weight)
            if bias is not None:
                lin.bias.copy_(bias.to(device=dev, dtype=torch.float16))
        del weight

        self._gl = A4W4_NVFP_dynamic(device=str(dev), dtype=torch.float16).from_linear(
            lin,
            del_orig=True,
        )

    @property
    def bias(self):
        return getattr(self._gl, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        return self._gl(x.to(torch.float16)).to(orig_dtype)

    @torch.no_grad()
    def warmup(self, m_values=(64, 128, 256, 1024, 4096)) -> None:
        for m in m_values:
            if m <= 0:
                continue
            x = torch.zeros(
                (int(m), self.in_features),
                dtype=torch.float16,
                device=self._device,
            )
            _ = self.forward(x)


class GemLiteMXFP4A8Linear(nn.Module):
    """Experimental MXFP8 activation + MXFP4 weight Linear.

    This re-quantizes the GPTQ weight into GemLite's MXFP4 format, then uses
    dynamic MXFP8 activations. It is faster for some FFN-up shapes but has much
    larger numeric error than the HQQ-preserving A8W4 path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        bias: torch.Tensor | None = None,
        groupsize: int = 32,
        v1: bool = True,
        wbits: int = 4,
        device: torch.device | str = "cuda:0",
    ):
        super().__init__()
        if wbits != 4:
            raise ValueError(f"GemLiteMXFP4A8Linear requires wbits=4, got {wbits}")
        if groupsize != 32:
            raise ValueError(
                f"GemLiteMXFP4A8Linear requires groupsize=32, got {groupsize}"
            )
        dev = torch.device(device)
        if dev.type != "cuda":
            raise RuntimeError("GemLiteMXFP4A8Linear requires a CUDA device")

        from gemlite.helper import A8W4_MXFP_dynamic

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self._device = dev

        g_idx = torch.arange(in_features, dtype=torch.long) // groupsize
        weight = dequant_gptq_to_fp(
            qweight=qweight,
            scales=scales,
            qzeros=qzeros,
            g_idx=g_idx,
            in_features=in_features,
            out_features=out_features,
            dtype=torch.float16,
            v1=v1,
            wbits=wbits,
        ).to(dev)

        lin = nn.Linear(
            in_features,
            out_features,
            bias=bias is not None,
            device=dev,
            dtype=torch.float16,
        )
        with torch.no_grad():
            lin.weight.copy_(weight)
            if bias is not None:
                lin.bias.copy_(bias.to(device=dev, dtype=torch.float16))
        del weight

        self._gl = A8W4_MXFP_dynamic(device=str(dev), dtype=torch.float16).from_linear(
            lin,
            del_orig=True,
        )

    @property
    def bias(self):
        return getattr(self._gl, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        return self._gl(x.to(torch.float16)).to(orig_dtype)

    @torch.no_grad()
    def warmup(self, m_values=(64, 128, 256, 1024, 4096)) -> None:
        for m in m_values:
            if m <= 0:
                continue
            x = torch.zeros(
                (int(m), self.in_features),
                dtype=torch.float16,
                device=self._device,
            )
            _ = self.forward(x)
