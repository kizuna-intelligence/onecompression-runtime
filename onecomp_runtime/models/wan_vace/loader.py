"""Wan2.1 VACE 14B int4 runtime loader.

Loads OneCompression checkpoints produced by
``WanVACE14BDiTAdapter.save_quantized_model``.  Quantized Linear weights stay in
packed int4 form and are executed by GemLite or the bundled fused Triton kernel.
"""
from __future__ import annotations

import torch

from ...diffusion import load_int4_model


def _post_load_wan_vace(model: torch.nn.Module, device: str | torch.device) -> None:
    """Rebuild non-persistent RoPE buffers that are absent from safetensors."""
    from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

    rope = getattr(model, "rope", None)
    freqs = getattr(rope, "freqs_cos", None)
    if isinstance(freqs, torch.Tensor) and freqs.is_meta:
        cfg = model.config
        model.rope = WanRotaryPosEmbed(
            attention_head_dim=int(cfg.attention_head_dim),
            patch_size=tuple(cfg.patch_size),
            max_seq_len=int(cfg.rope_max_seq_len),
        ).to(device=device)


def load_int4_wan_vace_transformer(
    checkpoint_path: str,
    *,
    device: str = "cuda:0",
    dtype: str | torch.dtype = "bfloat16",
    backend: str | None = "auto",
    use_fused: bool = True,
    warmup: bool = False,
    warmup_m_values: tuple[int, ...] = (256, 1024, 4096, 8192),
):
    """Rebuild a packed-int4 ``WanVACETransformer3DModel``."""

    from diffusers import WanVACETransformer3DModel

    return load_int4_model(
        checkpoint_path,
        lambda cfg: WanVACETransformer3DModel.from_config(cfg),
        device=device,
        dtype=dtype,
        backend=backend,
        use_fused=use_fused,
        post_load=lambda model: _post_load_wan_vace(model, device),
        warmup=warmup,
        warmup_m_values=warmup_m_values,
        label="wan-vace14b-int4",
    )
