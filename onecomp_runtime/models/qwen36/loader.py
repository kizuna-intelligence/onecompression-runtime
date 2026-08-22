"""Fast int4 loader for the GPTQ-quantized Qwen3.6-27B text decoder.

Reads a standard HuggingFace checkpoint produced by OneCompression's
``QuantizedModelLoader``-compatible save path (``config.json`` carrying
``quantization_config`` + a ``model.safetensors`` with AutoGPTQ-v1 packed
``qweight/scales/qzeros/g_idx`` per quantized Linear) and rebuilds a standalone
``Qwen3_5ForCausalLM`` whose quantized ``nn.Linear`` modules are replaced by
:class:`GemLiteInt4Linear` — int4/int8 dequant fused into a GemLite Triton GEMM,
so weights stay packed in VRAM and no per-call Python dequant runs.

Qwen3.6-27B is the ``Qwen3_5`` hybrid: 64 decoder layers mixing GatedDeltaNet
``linear_attn`` (fla Triton kernel, GPU-only) with full ``self_attn`` every 4th
layer, plus a SwiGLU ``mlp``. AutoBit assigned a {4,8}-bit floor (408 layers @
4-bit incl. all MLP, 88 @ 8-bit), group_size 128, desc_act=False — all
GemLite-compatible. Only the 496 block Linears are quantized; the 248k-vocab
``embed_tokens`` / ``lm_head``, RMSNorms, conv1d, ``A_log`` / ``dt_bias`` stay in
their loaded dtype (bf16) and are materialised on-device.

The model runs in bf16 (the fla GatedDeltaNet kernel path was validated in bf16);
GemLite layers cast their I/O to fp16 internally per the GemLite requirement.

Usage::

    from onecomp_runtime.models.qwen36 import load_int4_qwen36
    model, tok = load_int4_qwen36("/path/to/qwen36-27b-int4-qep", device="cuda:0")

Copyright 2025-2026 Kizuna Intelligence.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Any

import torch
from safetensors.torch import load_file

from ...layers.gemlite_int4_linear import GemLiteInt4Linear, gemlite_available
from ...layers.packed_gptq_linear import PackedGPTQLinear
from ...quant_utils import dequant_gptq_to_fp


def _iter_quant_layers(quant_config: dict):
    """Yield (name, wbits, groupsize) for every quantized Linear, reading the
    per-layer ``quantization_bits`` table (keys are suffixes after ``layers.X.``).
    """
    global_gs = int(quant_config.get("group_size", 128))
    qbits = quant_config.get("quantization_bits") or []
    for layer_idx, layer_cfg in enumerate(qbits):
        if not isinstance(layer_cfg, dict):
            continue
        for suffix, mod_cfg in layer_cfg.items():
            bits = mod_cfg["bits"] if isinstance(mod_cfg, dict) else int(mod_cfg)
            gs = global_gs
            if isinstance(mod_cfg, dict):
                params = mod_cfg.get("params") or {}
                gs = int(params.get("group_size", global_gs))
            yield f"model.layers.{layer_idx}.{suffix}", int(bits), gs


def _normalize_text_decoder_keys(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int]:
    """Map VLM-wrapped text-decoder keys to standalone CausalLM keys.

    Qwen3.5 conditional-generation checkpoints store the decoder below
    ``model.language_model``.  Quantization may subsequently save a text-only
    config while retaining that state-dict prefix.  The standalone CausalLM
    built by this loader expects the same tensors below ``model``.
    """
    source_prefix = "model.language_model."
    target_prefix = "model."
    if not any(key.startswith(source_prefix) for key in state_dict):
        return state_dict, 0

    normalized: dict[str, torch.Tensor] = {}
    renamed = 0
    for key, tensor in state_dict.items():
        target = (
            target_prefix + key[len(source_prefix):]
            if key.startswith(source_prefix)
            else key
        )
        if target in normalized:
            raise ValueError(f"state-dict key collision while normalizing {key!r}")
        normalized[target] = tensor
        renamed += target != key
    return normalized, renamed


def _build_eager(st: dict, in_f: int, out_f: int, wbits: int, gs: int,
                 dtype: torch.dtype, dev: torch.device, v1: bool) -> torch.nn.Linear:
    g_idx = st.get("g_idx")
    if g_idx is None:
        g_idx = (torch.arange(in_f) // gs).to(torch.long)
    weight = dequant_gptq_to_fp(
        qweight=st["qweight"], scales=st["scales"], qzeros=st["qzeros"],
        g_idx=g_idx, in_features=in_f, out_features=out_f, dtype=dtype, v1=v1,
        wbits=wbits,
    ).to(dev)
    lin = torch.nn.Linear(in_f, out_f, bias=False, device=dev, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(weight)
    return lin


def load_int4_qwen36(
    checkpoint_dir: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    backend: str = "auto",
    warmup: bool = False,
    warmup_m_values: tuple[int, ...] = (1, 8, 64, 256),
    attn_implementation: str | None = None,
):
    """Rebuild a packed-int4 ``Qwen3_5ForCausalLM`` for fast decode.

    Args:
        checkpoint_dir: directory with ``config.json`` (``quantization_config``),
            ``model.safetensors`` and tokenizer files.
        device: target CUDA device (GatedDeltaNet's fla kernel is GPU-only).
        dtype: compute dtype for non-quant tensors (bf16 recommended).
        backend: ``"auto"``/``"gemlite"`` use the GemLite GEMM; ``"eager"`` forces
            the dequant-to-bf16 fallback (for A/B speed comparison).
        warmup: pre-compile GemLite kernels for the given token counts.
        warmup_m_values: M-buckets to warm up (decode hits small M).
        attn_implementation: Transformers attention implementation. ``None`` keeps
            the checkpoint/config default; ``"sdpa"`` uses PyTorch's fused SDPA
            path for Qwen's full-attention layers.

    Returns:
        ``(model, tokenizer)`` — model in ``eval`` mode on ``device``.
    """
    from transformers import CONFIG_MAPPING, AutoModelForCausalLM, AutoTokenizer

    dev = torch.device(device)
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg_dict = json.load(fh)
    quant_config = cfg_dict.get("quantization_config")
    if quant_config is None:
        raise ValueError(f"{cfg_path} has no quantization_config")
    v1 = quant_config.get("checkpoint_format", "gptq") != "gptq_v2"

    if backend not in ("auto", "gemlite", "eager", "packed"):
        raise ValueError(f"unknown backend {backend!r}")
    if backend == "gemlite" and not gemlite_available():
        raise RuntimeError("backend='gemlite' but gemlite is not importable")
    # Resolve to a concrete mode. ``auto`` prefers GemLite (fast, packed) and
    # falls back to ``packed`` (pure-torch on-the-fly dequant) when GemLite is
    # unavailable — e.g. on ROCm/AMD — so the model stays at its int4 footprint
    # instead of the OOM-prone full eager dequant.
    if backend in ("auto", "gemlite"):
        mode = "gemlite" if gemlite_available() else "packed"
    else:
        mode = backend
    use_gemlite = mode == "gemlite"

    # Build the bare standalone CausalLM on meta (no full bf16 alloc).
    clean = {k: v for k, v in cfg_dict.items() if k != "quantization_config"}
    model_type = clean.get("model_type")
    if not model_type or model_type not in CONFIG_MAPPING:
        raise ValueError(f"model_type={model_type!r} not in CONFIG_MAPPING")
    model_config = CONFIG_MAPPING[model_type].from_dict(clean)
    if attn_implementation:
        # ``from_config`` does not reliably override this private field across
        # Transformers releases. Set it on the concrete config before building
        # the model so Qwen3_5Attention reads the requested implementation.
        model_config._attn_implementation = attn_implementation
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config, dtype=dtype)

    state_dict = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
    state_dict, normalized_n = _normalize_text_decoder_keys(state_dict)
    if normalized_n:
        print(f"[qwen36_int4] normalized {normalized_n} "
              "model.language_model.* checkpoint keys")

    modules = dict(model.named_modules())
    quant_keys: set[str] = set()
    gemlite_n = eager_n = packed_n = 0
    t0 = time.perf_counter()
    for name, wbits, gs in _iter_quant_layers(quant_config):
        parent_name, _, child = name.rpartition(".")
        parent = modules.get(parent_name)
        if parent is None:
            raise KeyError(f"quant layer parent not found: {parent_name!r}")
        lin = getattr(parent, child)
        in_f, out_f = lin.in_features, lin.out_features
        st = {}
        for s in ("qweight", "scales", "qzeros", "g_idx"):
            k = f"{name}.{s}"
            if k in state_dict:
                st[s] = state_dict[k]
                quant_keys.add(k)
        quant_keys.add(f"{name}.weight")
        if mode == "gemlite":
            layer = GemLiteInt4Linear(
                in_features=in_f, out_features=out_f,
                qweight=st["qweight"], scales=st["scales"], qzeros=st["qzeros"],
                bias=None, groupsize=gs, v1=v1, wbits=wbits, device=dev,
            )
            gemlite_n += 1
        elif mode == "packed":
            g_idx = st.get("g_idx")
            layer = PackedGPTQLinear(
                in_features=in_f, out_features=out_f,
                qweight=st["qweight"].to(dev),
                scales=st["scales"].to(device=dev, dtype=dtype),
                qzeros=st["qzeros"].to(dev),
                g_idx=g_idx.to(dev) if g_idx is not None else None,
                bias=None, groupsize=gs, wbits=wbits, v1=v1,
            )
            packed_n += 1
        else:
            layer = _build_eager(st, in_f, out_f, wbits, gs, dtype, dev, v1)
            eager_n += 1
        setattr(parent, child, layer)

    # Materialise the remaining (non-quant) tensors on-device.
    non_quant = {
        k: v.to(device=dev, dtype=dtype if v.dtype.is_floating_point else v.dtype)
        for k, v in state_dict.items()
        if k not in quant_keys
    }
    missing, unexpected = model.load_state_dict(non_quant, strict=False, assign=True)
    quant_prefixes = tuple(
        f"{name}." for name, _, _ in _iter_quant_layers(quant_config)
    )
    real_missing = [
        m for m in missing
        if m not in quant_keys and not m.startswith(quant_prefixes)
    ]
    if real_missing:
        raise RuntimeError(f"missing non-quant keys: {real_missing[:8]} ...")
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected[:8]} ...")

    # Non-persistent rope buffers (``inv_freq`` / ``original_inv_freq``) are
    # recomputed in __init__ and absent from the state_dict, so they were created
    # on meta. Rebuild the rotary module fresh on the real device via its own
    # constructor (it derives inv_freq from config).
    named = dict(model.named_modules())
    for mod_name, module in list(named.items()):
        inv = getattr(module, "inv_freq", None)
        if isinstance(inv, torch.Tensor) and inv.is_meta and hasattr(module, "config"):
            fresh = type(module)(config=module.config, device=dev)
            parent_name, _, child = mod_name.rpartition(".")
            parent = named.get(parent_name, model) if parent_name else model
            setattr(parent, child, fresh)

    for n, p in model.named_parameters():
        if p.is_meta:
            raise RuntimeError(f"parameter left on meta device: {n}")
    for n, b in model.named_buffers():
        if b.is_meta:
            raise RuntimeError(f"buffer left on meta device: {n}")

    model.eval()
    parameter_devices = Counter(str(p.device) for p in model.parameters())
    parameter_dtypes = Counter(str(p.dtype) for p in model.parameters())
    buffer_devices = Counter(str(b.device) for b in model.buffers())
    buffer_dtypes = Counter(str(b.dtype) for b in model.buffers())
    quant_module_counts = Counter(
        type(module).__name__
        for module in model.modules()
        if type(module).__name__ in {
            "GemLiteInt4Linear", "PackedGPTQLinear", "Linear",
        }
    )
    quant_bits = Counter(
        str(getattr(module, "wbits"))
        for module in model.modules()
        if type(module).__name__ == "GemLiteInt4Linear"
    )
    runtime_summary = {
        "device": str(dev),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "quant_backend": mode,
        "quant_module_counts": dict(quant_module_counts),
        "gemlite_wbits": dict(quant_bits),
        "parameter_devices": dict(parameter_devices),
        "parameter_dtypes": dict(parameter_dtypes),
        "buffer_devices": dict(buffer_devices),
        "buffer_dtypes": dict(buffer_dtypes),
        "hf_device_map_present": hasattr(model, "hf_device_map"),
    }
    model._onecomp_runtime_summary = runtime_summary
    print(f"[qwen36_int4] loaded {gemlite_n + eager_n + packed_n} quant layers "
          f"(gemlite={gemlite_n}, packed={packed_n}, eager={eager_n}) in "
          f"{time.perf_counter() - t0:.1f}s on {dev}")
    print("[qwen36_runtime] " + json.dumps(runtime_summary, sort_keys=True))

    if warmup and use_gemlite and dev.type == "cuda":
        seen: dict[tuple, Any] = {}
        for m in model.modules():
            if isinstance(m, GemLiteInt4Linear):
                seen.setdefault((m.in_features, m.out_features, m.wbits), m)
        for layer in seen.values():
            layer.warmup(m_values=warmup_m_values)
        torch.cuda.synchronize(dev)

    tok = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    return model, tok
