"""Generate text with the int4 Qwen3.6-27B decoder on a single 24GB GPU.

Loads the GPTQ-packed ``Qwen3_5ForCausalLM`` ({4,8}-bit AutoBit floor, ~13GB
int4 + bf16 embed/lm_head) with this runtime; every quantized Linear runs on a
GemLite Triton GEMM so weights stay packed in VRAM (~19GB peak).

Run::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    python example/qwen36/generate.py \\
        --ckpt /path/to/qwen36-27b-int4-qep \\
        --prompt "Explain quantization in one sentence."

``--backend eager`` forces the dequant fallback for A/B comparison. Requires a
transformers build with the ``qwen3_5`` architecture (>= 5.8).

Copyright 2025-2026 Kizuna Intelligence.
"""
from __future__ import annotations

import argparse
import time

import torch

from onecomp_runtime.models.qwen36 import load_int4_qwen36


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="int4 checkpoint directory")
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one word.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--backend", default="auto", choices=["auto", "gemlite", "eager"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--chat", action="store_true", help="wrap prompt in chat template")
    args = ap.parse_args()

    model, tok = load_int4_qwen36(
        args.ckpt, device=args.device, backend=args.backend, warmup=True
    )
    dev = next(model.parameters()).device

    if args.chat:
        text = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        text = args.prompt
    ids = tok(text, return_tensors="pt").input_ids.to(dev)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    gen = out.shape[1] - ids.shape[1]

    print("=" * 60)
    print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
    print("=" * 60)
    print(f"{gen} tokens in {dt:.2f}s = {gen / dt:.1f} tok/s")
    if dev.type == "cuda":
        print(f"peak VRAM {torch.cuda.max_memory_allocated(dev) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
