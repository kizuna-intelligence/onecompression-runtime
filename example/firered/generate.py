"""Edit images with the int4 FireRed-Image-Edit transformer.

Loads the packed-int4 ``QwenImageTransformer2DModel`` with this runtime and plugs
it into the diffusers ``QwenImageEditPlusPipeline``. The text encoder
(Qwen2.5-VL 7B) and VAE default to the upstream bf16 weights; ``--offload``
keeps whole-pipeline VRAM low by streaming components on/off the GPU.

Run::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
    python example/generate.py \\
        --repo /path/to/FireRed-Image-Edit-1.0 \\
        --dit /path/to/model.safetensors \\
        --image input.png --prompt "make it night time" \\
        --offload --outdir ./outputs

If ``--image`` is omitted a synthetic test scene is generated so the pipeline can
be smoke-tested without any external asset.

Copyright 2025-2026 Kizuna Intelligence.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from onecomp_runtime.models.firered import load_int4_transformer


def _make_test_image(size: int = 1024):
    """A simple, recognizable synthetic scene for edit smoke tests."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (135, 206, 235))  # sky blue
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(size * 0.68), size, size], fill=(96, 160, 72))  # grass
    d.ellipse([int(size * 0.70), int(size * 0.08),
               int(size * 0.90), int(size * 0.28)], fill=(255, 221, 64))  # sun
    # a small house
    hx, hy, hw = int(size * 0.30), int(size * 0.45), int(size * 0.22)
    d.rectangle([hx, hy, hx + hw, hy + hw], fill=(210, 180, 140))
    d.polygon([(hx - 12, hy), (hx + hw + 12, hy), (hx + hw // 2, hy - hw // 2)],
              fill=(150, 60, 50))  # roof
    d.rectangle([hx + hw // 3, hy + hw // 2, hx + 2 * hw // 3, hy + hw],
                fill=(90, 60, 40))  # door
    return img


def _sway_sigmas(num_steps: int, coef: float):
    """F5-TTS sway-sampling sigma schedule for a flow-matching scheduler.

    The default Qwen-Image schedule is ``linspace(1.0, 1/N, N)`` (uniform in
    flow-match time).  Sway remaps the progress variable ``u in [0,1]`` with
    ``s = u + coef*(cos(pi/2*u) - 1 + u)`` — fixed endpoints ``s(0)=0``,
    ``s(1)=1``, so the sigma range is unchanged but the sample density shifts:
    ``coef<0`` slows growth near ``u=0`` -> more steps at high sigma (early,
    structure-forming) which is where few-step flow matching benefits most.
    The pipeline still computes ``mu`` from these sigmas and applies its
    resolution-dependent time shift on top.
    """
    import numpy as np

    u = np.linspace(0.0, 1.0, num_steps)
    s = u + coef * (np.cos(np.pi / 2.0 * u) - 1.0 + u)
    lo = 1.0 / num_steps
    sigmas = 1.0 - s * (1.0 - lo)          # descending 1.0 -> 1/N
    return [float(x) for x in sigmas]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="HF repo (or local dir) for tokenizer / processor / "
                         "text encoder / VAE / scheduler")
    ap.add_argument("--dit", required=True,
                    help="packed int4 DiT safetensors (local)")
    ap.add_argument("--image", default=None,
                    help="input image to edit; if omitted a synthetic scene is used")
    ap.add_argument("--prompt", default="Make it a snowy winter night with a "
                                        "glowing moon and warm light in the window.")
    ap.add_argument("--negative-prompt", default=" ")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "gemlite", "fused", "eager"])
    ap.add_argument("--outdir", default="./outputs")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--true-cfg", type=float, default=4.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sway", type=float, default=None,
                    help="sway-sampling coefficient (F5-TTS). Remaps the "
                         "flow-match sigma schedule t'=t+c*(cos(pi/2*t)-1+t); "
                         "c=0 reproduces the default linspace, c<0 clusters "
                         "steps at high sigma (structure) — helps low step "
                         "counts. The pipeline's resolution-dependent mu shift "
                         "still applies on top. Try -1.0 for 20 or fewer steps.")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"],
                    help="pipeline compute dtype. GemLite int4 requires float16; "
                         "the fused/eager backends also support bfloat16, which "
                         "is far more numerically stable for Qwen-Image (fp16 can "
                         "overflow to NaN -> black image).")
    ap.add_argument("--offload", action="store_true",
                    help="enable model CPU offload (whole component at a time)")
    ap.add_argument("--sequential", action="store_true",
                    help="enable sequential (submodule) CPU offload — lowest "
                         "VRAM, fits a heavily-shared card; slower")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)

    from diffusers import QwenImageEditPlusPipeline
    from PIL import Image

    if args.image:
        image = Image.open(args.image).convert("RGB")
    else:
        print("[generate] no --image given; using a synthetic test scene")
        image = _make_test_image(args.size)
    image.save(os.path.join(args.outdir, "input.png"))

    # Run the whole pipeline in fp16: GemLite int4 requires fp16 I/O, and mixing
    # bf16 components triggers Half/BFloat16 mismatches at norm boundaries.
    #
    # Under --offload the int4 DiT must start on CPU so accelerate's hooks can
    # stream it onto the GPU only for the denoising loop (and off again for the
    # text-encode / VAE stages).  If it were pre-placed on the GPU the hooks
    # never offload it, so it stays resident while the 7B text encoder also
    # loads -> OOM on a contended card.  GemLite can't live on CPU buffers, so
    # the offload path uses the fused Triton kernel (its int4 tensors are plain
    # buffers that migrate cleanly with ``module.to(device)``).
    offload = args.offload or args.sequential
    backend = args.backend
    dit_device = "cuda:0"
    if offload:
        dit_device = "cpu"
        if backend in ("auto", "gemlite"):
            backend = "fused"
            print("[generate] offload: loading DiT on CPU with the fused "
                  "backend (GemLite cannot be offloaded)", flush=True)
    if backend == "gemlite" and args.dtype != "float16":
        print("[generate] GemLite requires float16 I/O; forcing dtype=float16",
              flush=True)
        args.dtype = "float16"
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"[generate] loading int4 DiT (backend={backend}, device={dit_device}, "
          f"dtype={args.dtype}) ...", flush=True)
    dit = load_int4_transformer(args.dit, device=dit_device, dtype=args.dtype,
                                backend=backend, warmup=False)

    print("[generate] assembling QwenImageEditPlusPipeline ...", flush=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.repo, transformer=dit, torch_dtype=torch_dtype)

    if args.sequential:
        pipe.enable_sequential_cpu_offload(device="cuda:0")
    elif args.offload:
        pipe.enable_model_cpu_offload(device="cuda:0")
    else:
        pipe.to("cuda:0")

    pipe_kwargs = {}
    if args.sway is not None:
        pipe_kwargs["sigmas"] = _sway_sigmas(args.steps, args.sway)
        print(f"[generate] sway sampling coef={args.sway} "
              f"(sigmas[:3]={[round(s, 4) for s in pipe_kwargs['sigmas'][:3]]} "
              f"...)", flush=True)

    print(f"[generate] editing at {args.size}px, {args.steps} steps, "
          f"true_cfg={args.true_cfg} ...", flush=True)
    # Under model-cpu-offload, accelerate moves whole components to CPU after
    # use but their freed blocks linger in the allocator, inflating reserved to
    # the SUM of the largest few components. Release the cache each step so the
    # reserve high-water tracks the largest single resident component, not the sum.
    def _free(pipe, i, t, kw):
        torch.cuda.empty_cache()
        return kw
    t0 = time.perf_counter()
    out = pipe(
        image=image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.size,
        width=args.size,
        true_cfg_scale=args.true_cfg,
        num_inference_steps=args.steps,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        callback_on_step_end=_free,
        **pipe_kwargs,
    ).images[0]
    dt = time.perf_counter() - t0
    torch.cuda.empty_cache()
    path = os.path.join(args.outdir, "edited.png")
    out.save(path)

    peak = torch.cuda.max_memory_reserved(dev) / (1024 ** 3)
    alloc = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
    print(f"[generate] saved {path}  ({dt:.1f}s)", flush=True)
    print(f"[generate] done. peak reserved {peak:.2f} GB, peak allocated {alloc:.2f} GB. "
          f"images in {os.path.abspath(args.outdir)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
