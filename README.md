# onecomp-runtime

Shared **int4 inference runtime** for [OneCompression](../OneCompression) packed
checkpoints. OneCompression *produces* GPTQ/RTN-packed `safetensors`; this is the
*consumer* side — the int4 GEMM layers, GPTQ unpack/dequant helpers, backend
selection, and a generic diffusion loader that every per-model runtime builds a
thin adapter on top of.

```
pip install onecomp-runtime              # import as onecomp_runtime
pip install onecomp-runtime[gemlite]     # + GemLite int4 kernels
pip install onecomp-runtime[diffusion]   # + diffusers (generic loader builds diffusers classes)
```

## Supported models

| Model | Modality | Backbone | Quant | Adapter | Weights |
|---|---|---|---|---|---|
| Irodori-TTS-500M-v3 | TTS | RF DiT + DAC-VAE | int4 GPTQ + RTN | Irodori-TTS-Lite | `kizuna-intelligence/Irodori-TTS-500M-v3-int4` |
| FLUX.2-klein | text→image | DiT + Qwen3 TE + VAE | int4 GPTQ (QEP) | Flux2-klein-Lite | — |
| LTX-2.3-22b | text→video | dual-stream DiT | int4 GPTQ | ltx2 | — |
| Command-A-Plus | text MoE | Cohere2 MoE | int1/2/4 RTN | cmda offload | — |
| FireRed-Image-Edit | image edit | 20B Qwen-Image | int4 GPTQ (QEP) | `models/firered` | `kizuna-intelligence/FireRed-Image-Edit-int4` |
| DeepSeek-V4-Flash | text MoE 284B | mHC/CSA + 256-expert | int4 dense + int2 experts (QEP) | `models/dsv4` | `kizuna-intelligence/DeepSeek-V4-Flash-int4-qep` |
| Qwen3.6-27B | text LLM | Qwen3_5 GatedDeltaNet/attn hybrid | int4/int8 GPTQ (QEP) | `models/qwen36` | — |
| Wan2.1-VACE-14B | video edit | Wan VACE 3D DiT | int4 GPTQ (QEP main) + RTN VACE | `models/wan_vace` | — |

QEP = Fujitsu Quantization Error Propagation. New models plug in via a
`build_meta_model` + `post_load` adapter; experimental backbones land here once
verified end-to-end on a 24GB GPU.

## Why

The int4 leaf machinery was copy-pasted across the FLUX.2 / LTX-2.3 / FireRed /
Irodori runtimes — `fused_int4_linear.py` was byte-identical in three of them.
A fix to the kernel (K-padding, warmup buckets, dtype safety) had to be hand-
propagated to every repo. This package is the single source of truth.

## Layout

```
onecomp_runtime/
  layers/
    fused_int4_linear.py   # Triton dequant+GEMM (AutoGPTQ-v1, gs=32)
    gemlite_int4_linear.py # GemLite kernel wrapper (fp16 I/O)
    packed_linear.py       # PackedRTNLinear, PackedEmbedding (RTN uint8-nibble)
    packed_conv.py         # int4 Conv1d / ConvTranspose1d (DAC-VAE codecs)
  quant_utils.py           # GPTQ + RTN unpack/dequant helpers
  backend.py               # resolve_backend / can_use_fused / build_{gemlite,fused,eager}
  diffusion.py             # load_int4_model(build_meta_model, ...) — generic GPTQ loader
```

## Usage — a per-model runtime adapter

```python
from onecomp_runtime.diffusion import load_int4_model
from diffusers import Flux2Transformer2DModel

def load_int4_transformer(path, **kw):
    return load_int4_model(
        path,
        lambda cfg: Flux2Transformer2DModel.from_config(cfg),
        label="flux2-klein-lite",
        **kw,
    )
```

The only per-model code is `build_meta_model` (construct the bare module from the
checkpoint's `config_json`) and an optional `post_load(model)` hook for buffer
fixups (e.g. FireRed/Qwen-Image RoPE tables that meta-init leaves uninitialised).

Wan2.1 VACE runtime support lives in `models/wan_vace`. The generation
orchestration and experiment notes are kept in
`~/gitrepos/ad-data-pipeline/scripts/wan_vace_generate.py` and
`~/gitrepos/ad-data-pipeline/docs/wan_vace_experiments.md`.

## Backends

| backend | kernel | I/O dtype | when |
|---|---|---|---|
| `gemlite` | GemLite Triton int4 | fp16 only | large M (FLUX), if installed |
| `a8w4` | GemLite dynamic 8-bit activation + GPTQ W4 | fp16 internal | experimental quality check |
| `a8w4_ffn_up` | `a8w4` only for FFN up-projection | fp16 internal | experimental layer ablation |
| `mxfp4_ffn` | GemLite MXFP4/MXFP8 only for FFN layers | fp16 internal | experimental speed/quality ablation |
| `mxfp4_ffn_up` | GemLite MXFP4/MXFP8 only for FFN up-projection | fp16 internal | experimental speed/quality ablation |
| `nvfp4_ffn` | GemLite NVFP4 only for FFN Linear layers | fp16 internal | experimental layer ablation |
| `fused` | bundled Triton dequant+GEMM | fp16/bf16/fp32 | default; bf16-safe; includes padded-groups path for some odd groupsizes |
| `packed` | packed GPTQ + per-call dequant | fp16/bf16/fp32 | fallback for unsupported fused shapes |
| `eager` | dequant once to `nn.Linear` | any | A/B checks or unsupported packed formats |

`backend="auto"` → gemlite if importable, else fused. **bf16 is the safe default**
for Qwen-Image / LTX (fp16 overflows to NaN); fp16 is only required on the
GemLite path. `a8w4`, `a8w4_ffn_up`, `mxfp4_ffn`, `mxfp4_ffn_up`, and
`nvfp4_ffn` are explicit experimental backends and are never selected by `auto`.

Cosmos Transfer2.5 helpers in `models/cosmos_transfer25` add runtime-only
optimizations on top of layer replacement: odd `groupsize=26` control embedders
use the padded-groups fused kernel, and multicontrol branches whose resolved
control weight is zero are zeroed before the official forward so Cosmos skips
their control blocks.

Cosmos Predict2.5 helpers in `models/cosmos_predict25` use the same
official-pipeline split: build the official Predict inference stack first, then
replace packed int4 layers inside `model.net`.

## Checkpoint contract

A single `safetensors` with metadata keys `config_json`, `quant_layers_json`
(per-layer manifest: `name`, `wbits`, `groupsize`, `actorder`, `in_features`,
`out_features`), and `checkpoint_format` (`gptq` v1 / `gptq_v2`). The RTN tier
(`packed_linear`, `packed_conv`) consumes the encoder/embedding/conv extras that
Irodori-style checkpoints add — those runtimes drive the layers directly rather
than through `load_int4_model`.
