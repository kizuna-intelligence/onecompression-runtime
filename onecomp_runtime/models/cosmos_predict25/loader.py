"""Runtime loader for official Cosmos-Predict2.5 int4 checkpoints.

The official Cosmos Predict inference stack should still build the model, text
encoder and tokenizer.  This module replaces packed ``nn.Linear`` modules
inside the already-built official ``model.net`` with OneCompression runtime
int4 layers from a packed safetensors checkpoint.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from ...backend import build_quant_layer, resolve_backend, resolve_dtype
from ...layers.fused_int4_linear import (
    FusedInt4Linear,
    FusedInt4LinearAnyGroup,
    FusedInt4LinearPaddedGroups,
)
from ...layers.gemlite_int4_linear import GemLiteInt4Linear


def cuda_memory_summary(label: str | None = None) -> dict[str, float]:
    """Return CUDA memory stats in GiB, printing them when ``label`` is set."""
    if not torch.cuda.is_available():
        stats = {"allocated_gib": 0.0, "reserved_gib": 0.0, "peak_gib": 0.0}
        if label:
            print(f"[mem] {label}: cuda unavailable", flush=True)
        return stats

    torch.cuda.synchronize()
    stats = {
        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    if label:
        print(
            f"[mem] {label}: "
            f"allocated={stats['allocated_gib']:.2f}GiB "
            f"reserved={stats['reserved_gib']:.2f}GiB "
            f"peak={stats['peak_gib']:.2f}GiB",
            flush=True,
        )
    return stats


def _canonical_checkpoint_name(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "")


def _resolve_tensor_prefix(name: str, tensor_keys: set[str]) -> str:
    for prefix in (name, _canonical_checkpoint_name(name)):
        if f"{prefix}.qweight" in tensor_keys:
            return prefix
    raise KeyError(f"packed tensors not found for quantized layer {name!r}")


def _module_name_candidates(name: str) -> tuple[str, ...]:
    canonical = _canonical_checkpoint_name(name)
    candidates = [name, canonical]
    candidates.extend(f"base_model.model.{candidate}" for candidate in tuple(candidates))
    return tuple(dict.fromkeys(candidates))


def _resolve_module(modules: dict[str, nn.Module], name: str) -> tuple[str, nn.Module, str]:
    for candidate in _module_name_candidates(name):
        parent_name, _, child = candidate.rpartition(".")
        parent = modules.get(parent_name) if parent_name else None
        if parent is not None and child in parent._modules:
            return candidate, parent, child
    parent_name, _, child = name.rpartition(".")
    raise KeyError(f"quant layer parent not found in official Predict net: {parent_name!r}")


def load_int4_predict_into_net(
    net: nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cuda:0",
    dtype: str | torch.dtype = "bfloat16",
    backend: str | None = "auto",
    use_fused: bool = True,
    warmup: bool = False,
    warmup_m_values: tuple[int, ...] = (64, 128, 256, 512, 1024, 4096, 21600),
    strict: bool = True,
    progress: bool = True,
) -> dict[str, int]:
    """Replace official Cosmos Predict Linear modules with int4 layers."""
    checkpoint_path = Path(checkpoint_path)
    dev = torch.device(device)
    resolved_dtype = resolve_dtype(dtype)
    resolved_backend = resolve_backend(backend, use_fused=use_fused)
    counts = {"gemlite": 0, "fused": 0, "packed": 0, "eager": 0, "skipped": 0}

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
        md = f.metadata() or {}
        if md.get("official_predict") != "true":
            raise ValueError(f"{checkpoint_path} is not an official Cosmos Predict checkpoint")
        quant_layers: list[dict[str, Any]] = json.loads(md["quant_layers_json"])
        tensor_keys = set(f.keys())
        modules = dict(net.named_modules())
        t0 = time.perf_counter()

        for idx, raw_entry in enumerate(quant_layers, start=1):
            entry: dict[str, Any] = dict(raw_entry)
            entry["checkpoint_format"] = md.get("checkpoint_format", "gptq")
            name = entry["name"]

            try:
                prefix = _resolve_tensor_prefix(name, tensor_keys)
                module_name, parent, child = _resolve_module(modules, name)
            except KeyError:
                if strict:
                    raise
                counts["skipped"] += 1
                continue

            st: dict[str, torch.Tensor] = {}
            for suffix in ("qweight", "scales", "qzeros", "g_idx", "bias"):
                key = f"{prefix}.{suffix}"
                if key in tensor_keys:
                    st[suffix] = f.get_tensor(key)

            layer, kind = build_quant_layer(entry, st, resolved_backend, resolved_dtype, dev)
            current = parent._modules[child]
            if hasattr(current, "base_layer"):
                current.base_layer = layer
                modules[f"{module_name}.base_layer"] = layer
            else:
                parent._modules[child] = layer
                modules[module_name] = layer
            counts[kind] += 1

            del st, layer
            if progress and (idx == 1 or idx % 50 == 0 or idx == len(quant_layers)):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elapsed = time.perf_counter() - t0
                print(
                    f"[int4] loaded {idx}/{len(quant_layers)} Predict layers "
                    f"(gemlite={counts['gemlite']}, fused={counts['fused']}, "
                    f"packed={counts['packed']}, eager={counts['eager']}, "
                    f"skipped={counts['skipped']}) elapsed={elapsed:.1f}s",
                    flush=True,
                )
                cuda_memory_summary(f"after predict int4 {idx}")

    if warmup and dev.type == "cuda":
        seen: dict[tuple[int, int, bool], nn.Module] = {}
        for module in net.modules():
            if isinstance(
                module,
                (FusedInt4Linear, FusedInt4LinearAnyGroup, FusedInt4LinearPaddedGroups, GemLiteInt4Linear),
            ):
                seen.setdefault((module.in_features, module.out_features, module.bias is not None), module)
        for module in seen.values():
            module.warmup(m_values=warmup_m_values)

    return counts


def load_qep_int4_into_predict_net(*args: Any, **kwargs: Any) -> dict[str, int]:
    """Backward-compatible alias for experiment scripts."""
    return load_int4_predict_into_net(*args, **kwargs)
