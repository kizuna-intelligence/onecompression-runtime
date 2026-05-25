"""Load a QEP-quantized DeepSeek-V4-Flash for inference on a single 24GB GPU.

The model is assembled block-by-block so host/GPU memory never holds the full
bf16 model (~280GB).  For each layer we:

1. build the shipped ``model.Block`` in **bf16** mode (all custom ``Linear`` become
   plain ``F.linear``; no fp8/fp4 kernels) and load its weights + aux params
   (norms, gate, hc mixing, compressor ``ape``/norm, ``attn_sink``) from the
   original checkpoint shards via :class:`Dsv4ShardReader`;
2. swap every quantized ``Linear`` (dense attn + compressor/indexer + shared expert
   + 256 routed experts) for a :class:`Dsv4PackedLinear` carrying the packed int4/int2
   weight from ``qep_out/layer_{L}.pt`` — **dense path on CUDA, routed experts kept on
   CPU** and dequantized on the fly only for the ~6 experts a token actually routes to;
3. drop the transient bf16 weights.

embed / final norm / lm-head / top-level hc-head params are loaded once from shards.
This mirrors the quantizer's ``build_layer`` exactly (same arch globals, same
state-dict mapping) so the runtime sees identical weights to what was quantized.
"""
from __future__ import annotations

import gc
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

from ...layers import Dsv4PackedLinear


def _set_globals(M):
    M.world_size, M.rank = 1, 0
    M.default_dtype = torch.bfloat16
    M.scale_fmt, M.scale_dtype, M.block_size = None, torch.float32, 128


class Dsv4QuantizedModel(nn.Module):
    """Assembled QEP-int4 DeepSeek-V4-Flash.  ``forward(input_ids, start_pos)`` ->
    logits, matching the shipped ``Transformer.forward`` math (embed -> hc-expand ->
    blocks -> hc-head -> logits)."""

    def __init__(self, inference_dir: str, ckpt_dir: str, qep_dir: str, *,
                 device: str = "cuda", expert_device: str = "cpu",
                 n_layers: int | None = None, max_seq_len: int = 4096,
                 max_batch_size: int = 1):
        super().__init__()
        if inference_dir not in sys.path:
            sys.path.insert(0, inference_dir)
        import model as M  # shipped reference arch (mHC/CSA/HCA/MoE)
        self.M = M
        _set_globals(M)

        # reuse the quantizer's shard reader + arg builder so weights match exactly
        from onecomp.adapters.dsv4_weights import Dsv4ShardReader
        from onecomp.adapters.dsv4_qep import _build_args, _swap_linears, Dsv4StreamingQEP

        self.reader = Dsv4ShardReader(ckpt_dir)
        self.cfg = json.load(open(f"{ckpt_dir}/config.json"))
        self.args = _build_args(self.cfg, max_seq_len, max_batch_size)
        self.dev = self.device = torch.device(device)
        self.expert_dev = torch.device(expert_device)
        self._swap_linears = _swap_linears
        self._build_layer = Dsv4StreamingQEP.build_layer.__get__(self)  # bound helper
        self.hc_mult = self.args.hc_mult

        total = self.args.n_layers if n_layers is None else n_layers
        self.layers = nn.ModuleList()
        for L in range(total):
            self.layers.append(self._load_block(L, qep_dir))
            gc.collect(); torch.cuda.empty_cache()

        # top-level (non-block) params, bf16 from shards
        self.embed_w = self.reader.plain("embed.weight", torch.bfloat16).to(self.dev)
        self.norm_w = self.reader.plain("norm.weight", torch.float32).to(self.dev)
        self.head_w = self.reader.plain("head.weight", torch.float32).to(self.dev)
        self.hc_head_fn = self.reader.plain("hc_head_fn", torch.float32).to(self.dev)
        self.hc_head_base = self.reader.plain("hc_head_base", torch.float32).to(self.dev)
        self.hc_head_scale = self.reader.plain("hc_head_scale", torch.float32).to(self.dev)
        self.norm_eps = self.args.norm_eps
        self.hc_eps = self.args.hc_eps

    # -- aux-only block build (skip quantized-weight dequant) --------------
    def _build_block_aux(self, L: int) -> nn.Module:
        """Construct the bf16 ``Block`` and load ONLY non-quantized params
        (norms, gate, hc mixing, attn_sink, compressor ape/norm). Every Linear
        with a ``.scale`` sibling is replaced by a packed int4/int2 module right
        after, so dequanting its fp4/fp8 weight from disk is pure waste — that
        per-layer fp4 expert dequant was the ~130s/layer load bottleneck."""
        block = self.M.Block(L, self.args)
        prefix = f"layers.{L}."
        sd = {}
        for name in self.reader.weight_map:
            if not name.startswith(prefix) or name.endswith(".scale"):
                continue
            short = name[len(prefix):]
            if short.endswith(".weight") and self.reader.has(name.replace(".weight", ".scale")):
                continue                            # quantized → packed swap covers it
            sd[short] = self.reader.plain(name)
        block.load_state_dict(sd, strict=False)
        self._swap_linears(block, self.M)
        return block.to(self.device).eval()

    # -- per-layer build + packed swap -------------------------------------
    def _load_block(self, L: int, qep_dir: str) -> nn.Module:
        block = self._build_block_aux(L)           # bf16 block, aux only (no expert dequant)
        saved = torch.load(f"{qep_dir}/layer_{L}.pt", map_location="cpu")
        modules = saved["modules"]
        named = dict(block.named_modules())
        for path, packed in modules.items():
            parent = named[path.rsplit(".", 1)[0]] if "." in path else block
            attr = path.rsplit(".", 1)[-1]
            on_cpu = ".experts." in path           # routed experts offloaded
            buf_dev = self.expert_dev if on_cpu else self.dev
            old = getattr(parent, attr)
            bias = old.bias.detach() if getattr(old, "bias", None) is not None else None
            setattr(parent, attr, Dsv4PackedLinear(packed, bias=bias, buffer_device=buf_dev))
        del saved, modules
        return block.eval()

    # -- forward (mirrors Transformer/ParallelHead math) -------------------
    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        with torch.device(self.dev):
            ids = input_ids.to(self.dev)
            h = F.embedding(ids, self.embed_w)
            h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
            for blk in self.layers:
                h = blk(h, start_pos, ids).to(torch.bfloat16)
            return self._head(h)

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        # hc_head: reduce hc copies -> 1, then final RMSNorm + logits (fp32)
        shape, dtype = x.size(), x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(xf, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2).to(dtype)
        # final RMSNorm (norm.weight fp32) over last token then lm-head
        yf = y[:, -1].float()
        var = yf.square().mean(-1, keepdim=True)
        yf = yf * torch.rsqrt(var + self.norm_eps) * self.norm_w
        return F.linear(yf, self.head_w)
