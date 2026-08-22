"""Backend selection + per-layer int4 module construction.

These helpers are model-agnostic: they take a ``quant_layers`` entry (the dict
OneCompression writes into ``quant_layers_json``) plus the layer's loaded
tensors, and build the right int4 ``nn.Module`` for the chosen backend.

Backends:
    - ``gemlite`` — GemLite Triton int4 GEMM (fp16 I/O); fastest at large M.
    - ``a8w4``    — experimental GemLite dynamic 8-bit activation + GPTQ W4.
    - ``a8w4_ffn_up`` — experimental A8W4 only for FFN up-projection layers.
    - ``mxfp4_ffn`` — experimental MXFP4/MXFP8 only for FFN layers.
    - ``mxfp4_ffn_up`` — experimental MXFP4/MXFP8 only for FFN up-projection.
    - ``nvfp4_ffn`` — experimental NVFP4 only for FFN Linear layers.
    - ``fused``   — bundled fused Triton dequant+GEMM kernel.
    - ``packed``  — keep GPTQ buffers packed and dequantize one layer per call.
    - ``eager``   — dequantize once to a plain ``nn.Linear`` (fallback for
      explicit A/B checks).

Copyright 2025-2026 Kizuna Intelligence / Fujitsu Ltd. MIT License.
"""
from __future__ import annotations

from typing import Any

import torch

from .layers.fused_int4_linear import (
    FusedInt4Linear,
    FusedInt4LinearAnyGroup,
    FusedInt4LinearPaddedGroups,
)
from .layers.gemlite_experimental_linear import (
    GemLiteA8W4HQQLinear,
    GemLiteMXFP4A8Linear,
    GemLiteNVFP4Linear,
)
from .layers.gemlite_int4_linear import GemLiteInt4Linear, gemlite_available
from .layers.packed_gptq_linear import PackedGPTQLinear
from .quant_utils import dequant_gptq_to_fp

_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _DTYPES[str(dtype).lower()]
    except KeyError:
        raise ValueError(f"unsupported dtype: {dtype!r}")


def can_use_fused(entry: dict[str, Any]) -> bool:
    """True if a layer meets the fused/gemlite kernel constraints."""
    return (
        not bool(entry.get("actorder", False))
        and int(entry["wbits"]) == 4
        and int(entry["groupsize"]) == 32
        and int(entry["in_features"]) % 32 == 0
        and int(entry["out_features"]) % 8 == 0
    )


def can_use_fused_any_group(entry: dict[str, Any]) -> bool:
    """True if the generic fused kernel can run this GPTQ int4 layer."""
    return (
        not bool(entry.get("actorder", False))
        and int(entry["wbits"]) == 4
        and int(entry["in_features"]) % 8 == 0
        and int(entry["out_features"]) % 8 == 0
    )


def can_use_fused_padded_groups(entry: dict[str, Any]) -> bool:
    """True if odd groups can be re-laid out onto the groupsize-32 kernel."""
    gs = int(entry["groupsize"])
    in_features = int(entry["in_features"])
    return (
        not bool(entry.get("actorder", False))
        and int(entry["wbits"]) == 4
        and gs != 32
        and 0 < gs < 32
        and in_features % gs == 0
        and int(entry["out_features"]) % 8 == 0
    )


def is_ffn_layer(entry: dict[str, Any]) -> bool:
    return ".ffn." in str(entry.get("name", ""))


def is_ffn_up_layer(entry: dict[str, Any]) -> bool:
    return ".ffn.net.0.proj" in str(entry.get("name", ""))


def resolve_backend(backend: str | None, use_fused: bool = True) -> str:
    """Pick the int4 GEMM backend.

    ``backend`` takes precedence; ``use_fused=False`` (legacy) forces eager.
    ``"auto"`` prefers GemLite when importable (within ~15% of bf16 at large-M
    shapes), then the fused Triton kernel. Layers that miss the fused shape
    constraints fall back to packed on-the-fly dequant instead of eager bf16.
    """
    if not use_fused:
        return "eager"
    b = (backend or "auto").lower()
    if b == "auto":
        return "gemlite" if gemlite_available() else "fused"
    if b == "gemlite" and not gemlite_available():
        raise RuntimeError(
            "backend='gemlite' requested but the 'gemlite' package is not "
            "importable; pip install gemlite or use backend='fused'"
        )
    if b in (
        "a8w4",
        "a8w4_ffn_up",
        "mxfp4_ffn",
        "mxfp4_ffn_up",
        "nvfp4_ffn",
    ) and not gemlite_available():
        raise RuntimeError(
            f"backend={b!r} requested but the 'gemlite' package is not "
            "importable; pip install gemlite or use backend='fused'"
        )
    if b not in (
        "gemlite",
        "a8w4",
        "a8w4_ffn_up",
        "mxfp4_ffn",
        "mxfp4_ffn_up",
        "nvfp4_ffn",
        "fused",
        "packed",
        "eager",
    ):
        raise ValueError(f"unknown backend: {backend!r}")
    return b


def build_a8w4(entry: dict, st: dict, device: torch.device) -> GemLiteA8W4HQQLinear:
    return GemLiteA8W4HQQLinear(
        in_features=int(entry["in_features"]),
        out_features=int(entry["out_features"]),
        qweight=st["qweight"],
        scales=st["scales"],
        qzeros=st["qzeros"],
        bias=(st["bias"] if "bias" in st else None),
        groupsize=int(entry["groupsize"]),
        v1=entry.get("checkpoint_format", "gptq") != "gptq_v2",
        wbits=int(entry["wbits"]),
        device=device,
    )


def build_nvfp4(entry: dict, st: dict, device: torch.device) -> GemLiteNVFP4Linear:
    return GemLiteNVFP4Linear(
        in_features=int(entry["in_features"]),
        out_features=int(entry["out_features"]),
        qweight=st["qweight"],
        scales=st["scales"],
        qzeros=st["qzeros"],
        bias=(st["bias"] if "bias" in st else None),
        groupsize=int(entry["groupsize"]),
        v1=entry.get("checkpoint_format", "gptq") != "gptq_v2",
        wbits=int(entry["wbits"]),
        device=device,
    )


def build_mxfp4_a8(entry: dict, st: dict, device: torch.device) -> GemLiteMXFP4A8Linear:
    return GemLiteMXFP4A8Linear(
        in_features=int(entry["in_features"]),
        out_features=int(entry["out_features"]),
        qweight=st["qweight"],
        scales=st["scales"],
        qzeros=st["qzeros"],
        bias=(st["bias"] if "bias" in st else None),
        groupsize=int(entry["groupsize"]),
        v1=entry.get("checkpoint_format", "gptq") != "gptq_v2",
        wbits=int(entry["wbits"]),
        device=device,
    )


def build_gemlite(entry: dict, st: dict, device: torch.device) -> GemLiteInt4Linear:
    return GemLiteInt4Linear(
        in_features=int(entry["in_features"]),
        out_features=int(entry["out_features"]),
        qweight=st["qweight"],
        scales=st["scales"],
        qzeros=st["qzeros"],
        bias=(st["bias"] if "bias" in st else None),
        groupsize=int(entry["groupsize"]),
        v1=entry.get("checkpoint_format", "gptq") != "gptq_v2",
        device=device,
    )


def build_fused(entry: dict, st: dict, device: torch.device) -> FusedInt4Linear:
    return FusedInt4Linear(
        in_features=int(entry["in_features"]),
        out_features=int(entry["out_features"]),
        qweight=st["qweight"].to(device),
        scales=st["scales"].to(device=device, dtype=torch.float16),
        qzeros=st["qzeros"].to(device),
        bias=(st["bias"].to(device) if "bias" in st else None),
        groupsize=32,
    )


def build_fused_any_group(
    entry: dict, st: dict, device: torch.device
) -> FusedInt4LinearAnyGroup:
    return FusedInt4LinearAnyGroup(
        in_features=int(entry["in_features"]),
        out_features=int(entry["out_features"]),
        qweight=st["qweight"].to(device),
        scales=st["scales"].to(device=device, dtype=torch.float16),
        qzeros=st["qzeros"].to(device),
        bias=(st["bias"].to(device) if "bias" in st else None),
        groupsize=int(entry["groupsize"]),
    )


def build_fused_padded_groups(
    entry: dict, st: dict, device: torch.device
) -> FusedInt4LinearPaddedGroups:
    return FusedInt4LinearPaddedGroups(
        in_features=int(entry["in_features"]),
        out_features=int(entry["out_features"]),
        qweight=st["qweight"].to(device),
        scales=st["scales"].to(device=device, dtype=torch.float16),
        qzeros=st["qzeros"].to(device),
        bias=(st["bias"].to(device) if "bias" in st else None),
        groupsize=int(entry["groupsize"]),
    )


def build_eager(entry: dict, st: dict, dtype: torch.dtype,
                device: torch.device) -> torch.nn.Linear:
    in_f, out_f = int(entry["in_features"]), int(entry["out_features"])
    g_idx = st.get("g_idx")
    if g_idx is None:
        gs = int(entry["groupsize"])
        g_idx = (torch.arange(in_f) // gs).to(torch.long)
    weight = dequant_gptq_to_fp(
        qweight=st["qweight"], scales=st["scales"], qzeros=st["qzeros"],
        g_idx=g_idx, in_features=in_f, out_features=out_f, dtype=dtype,
        v1=entry.get("checkpoint_format", "gptq") != "gptq_v2",
    ).to(device)
    has_bias = "bias" in st
    lin = torch.nn.Linear(in_f, out_f, bias=has_bias, device=device, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(weight)
        if has_bias:
            lin.bias.copy_(st["bias"].to(dtype=dtype, device=device))
    return lin


def build_packed(entry: dict, st: dict, dtype: torch.dtype,
                 device: torch.device) -> PackedGPTQLinear:
    in_f, out_f = int(entry["in_features"]), int(entry["out_features"])
    g_idx = st.get("g_idx")
    return PackedGPTQLinear(
        in_features=in_f,
        out_features=out_f,
        qweight=st["qweight"].to(device),
        scales=st["scales"].to(device=device, dtype=dtype),
        qzeros=st["qzeros"].to(device),
        g_idx=g_idx.to(device) if g_idx is not None else None,
        bias=(st["bias"].to(device) if "bias" in st else None),
        groupsize=int(entry["groupsize"]),
        wbits=int(entry["wbits"]),
        v1=entry.get("checkpoint_format", "gptq") != "gptq_v2",
    )


def build_quant_layer(entry: dict, st: dict, backend: str, dtype: torch.dtype,
                      device: torch.device):
    """Dispatch a single quant-layer build, returning ``(module, kind)``.

    ``kind`` is one of ``"gemlite"``, ``"a8w4"``, ``"a8w4_ffn_up"``,
    ``"mxfp4_ffn"``, ``"mxfp4_ffn_up"``, ``"nvfp4_ffn"``, ``"fused"``,
    ``"packed"``, ``"eager"`` — useful for load-summary counters.
    """
    if backend == "a8w4" and can_use_fused(entry):
        return build_a8w4(entry, st, device), "a8w4"
    if backend == "a8w4_ffn_up" and is_ffn_up_layer(entry) and can_use_fused(entry):
        return build_a8w4(entry, st, device), "a8w4_ffn_up"
    if backend == "a8w4_ffn_up" and can_use_fused(entry):
        return build_gemlite(entry, st, device), "gemlite"
    if backend == "mxfp4_ffn" and is_ffn_layer(entry) and can_use_fused(entry):
        return build_mxfp4_a8(entry, st, device), "mxfp4_ffn"
    if backend == "mxfp4_ffn" and can_use_fused(entry):
        return build_gemlite(entry, st, device), "gemlite"
    if backend == "mxfp4_ffn_up" and is_ffn_up_layer(entry) and can_use_fused(entry):
        return build_mxfp4_a8(entry, st, device), "mxfp4_ffn_up"
    if backend == "mxfp4_ffn_up" and can_use_fused(entry):
        return build_gemlite(entry, st, device), "gemlite"
    if backend == "nvfp4_ffn" and is_ffn_layer(entry) and can_use_fused(entry):
        return build_nvfp4(entry, st, device), "nvfp4_ffn"
    if backend == "nvfp4_ffn" and can_use_fused(entry):
        return build_gemlite(entry, st, device), "gemlite"
    if backend == "gemlite" and can_use_fused(entry):
        return build_gemlite(entry, st, device), "gemlite"
    if backend in ("gemlite", "fused") and can_use_fused(entry):
        return build_fused(entry, st, device), "fused"
    if backend in ("gemlite", "fused") and can_use_fused_padded_groups(entry):
        return build_fused_padded_groups(entry, st, device), "fused"
    if backend in ("gemlite", "fused") and can_use_fused_any_group(entry):
        return build_fused_any_group(entry, st, device), "fused"
    if backend in ("gemlite", "fused", "packed"):
        return build_packed(entry, st, dtype, device), "packed"
    return build_eager(entry, st, dtype, device), "eager"
