"""Runtime-only optimizations for the official Cosmos Transfer2.5 net."""
from __future__ import annotations

import atexit
import os
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from types import MethodType
import torch
from torch import nn


class ZeroControlPatchEmbed(nn.Module):
    """Skip PatchEmbed for inactive multibranch control inputs.

    The official multibranch forward still calls each control PatchEmbed before
    zeroing inactive branches. At this point the tensor already contains the
    condition mask, so an all-zero branch is detected by checking only the real
    control channels and ignoring trailing mask channels.
    """

    def __init__(self, inner: nn.Module, *, tail_mask_channels: int) -> None:
        super().__init__()
        self.inner = inner
        self.tail_mask_channels = tail_mask_channels
        self.skipped_calls = 0

    def init_weights(self) -> None:
        if hasattr(self.inner, "init_weights"):
            self.inner.init_weights()

    @property
    def spatial_patch_size(self) -> int:
        return self.inner.spatial_patch_size

    @property
    def temporal_patch_size(self) -> int:
        return self.inner.temporal_patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tail_mask_channels > 0:
            control_x = x[:, :-self.tail_mask_channels]
        else:
            control_x = x
        if not control_x.any():
            self.skipped_calls += 1
            b, _, t, h, w = x.shape
            out_features = self.inner.proj[1].out_features
            return x.new_zeros(
                (
                    b,
                    t // self.inner.temporal_patch_size,
                    h // self.inner.spatial_patch_size,
                    w // self.inner.spatial_patch_size,
                    out_features,
                )
            )
        return self.inner(x)


def install_zero_control_patch_embed_skip(net: nn.Module) -> int:
    """Wrap multibranch control embedders so zero-weight branches skip Linear."""
    if getattr(net, "num_control_branches", 1) <= 1:
        return 0
    control_embedder = getattr(net, "control_embedder", None)
    if not isinstance(control_embedder, nn.ModuleList):
        return 0

    tail_mask_channels = 1 + int(bool(getattr(net, "concat_padding_mask", False)))
    installed = 0
    for idx, embedder in enumerate(control_embedder):
        if isinstance(embedder, ZeroControlPatchEmbed):
            continue
        control_embedder[idx] = ZeroControlPatchEmbed(embedder, tail_mask_channels=tail_mask_channels)
        installed += 1
    if installed:
        print(
            f"[cosmos-opt] installed zero-control PatchEmbed skip on {installed} branches "
            f"(tail_mask_channels={tail_mask_channels})",
            flush=True,
        )
    return installed


def _zero_weight_branch_mask(control_context_scale, num_control_branches: int) -> list[bool] | None:
    if isinstance(control_context_scale, torch.Tensor):
        if control_context_scale.ndim == 0:
            return [abs(float(control_context_scale)) <= 1e-8] * num_control_branches
        if control_context_scale.ndim == 1 and control_context_scale.numel() == num_control_branches:
            return [abs(float(w)) <= 1e-8 for w in control_context_scale.detach().cpu()]
        if control_context_scale.ndim >= 2 and control_context_scale.shape[0] == num_control_branches:
            return [not bool(w.detach().abs().gt(1e-8).any()) for w in control_context_scale]
        return None
    if isinstance(control_context_scale, (float, int)):
        return [abs(float(control_context_scale)) <= 1e-8] * num_control_branches
    if isinstance(control_context_scale, str):
        parts = [p.strip() for p in control_context_scale.split(",")]
        if len(parts) == num_control_branches:
            return [abs(float(p)) <= 1e-8 for p in parts]
    if isinstance(control_context_scale, (list, tuple)) and len(control_context_scale) == num_control_branches:
        if all(isinstance(w, (float, int, str)) for w in control_context_scale):
            return [abs(float(w)) <= 1e-8 for w in control_context_scale]
    return None


def install_zero_weight_control_input_skip(net: nn.Module) -> bool:
    """Zero control input chunks whose multicontrol weight is exactly zero.

    Cosmos' stock forward decides whether to execute a control branch from the
    control tensor content, not from the branch weight. A zero-weight branch with
    a provided control video therefore still runs all control blocks. Zeroing the
    chunk before the stock forward preserves the weighted sum while enabling the
    existing has-hint skip path.
    """
    num_control_branches = int(getattr(net, "num_control_branches", 1))
    if num_control_branches <= 1 or hasattr(net, "_onecomp_original_forward"):
        return False

    original_forward = net.forward

    def forward_with_zero_weight_skip(*args, **kwargs):
        latent = kwargs.get("latent_control_input")
        latent_arg_index = 3
        if latent is None and len(args) > latent_arg_index:
            latent = args[latent_arg_index]
        scale = kwargs.get("control_context_scale", 1.0)
        zero_branches = _zero_weight_branch_mask(scale, num_control_branches)
        if latent is not None and zero_branches is not None and any(zero_branches):
            chunks = list(latent.chunk(num_control_branches, dim=1))
            changed = False
            for idx, is_zero_weight in enumerate(zero_branches):
                if is_zero_weight and chunks[idx].any():
                    chunks[idx] = torch.zeros_like(chunks[idx])
                    changed = True
            if changed:
                latent = torch.cat(chunks, dim=1)
                if "latent_control_input" in kwargs:
                    kwargs["latent_control_input"] = latent
                else:
                    args = list(args)
                    args[latent_arg_index] = latent
                    args = tuple(args)
        return original_forward(*args, **kwargs)

    net._onecomp_original_forward = original_forward
    net.forward = forward_with_zero_weight_skip
    print("[cosmos-opt] installed zero-weight control input skip", flush=True)
    return True


def _concat_condition_pair(condition, uncondition):
    kwargs = {}
    control_scale = getattr(condition, "control_context_scale", None)
    num_control_branches = control_scale.shape[0] if isinstance(control_scale, torch.Tensor) and control_scale.ndim >= 2 else None
    for field in dataclass_fields(condition):
        name = field.name
        cond_value = getattr(condition, name)
        uncond_value = getattr(uncondition, name)
        if isinstance(cond_value, torch.Tensor) and isinstance(uncond_value, torch.Tensor):
            if name in {"use_video_condition", "_is_broadcasted"} or cond_value.ndim == 0:
                kwargs[name] = cond_value
            elif name == "control_context_scale" and num_control_branches is not None:
                # Multicontrol weight maps are [num_branches, B, T, H, W, 1].
                kwargs[name] = torch.cat([cond_value, uncond_value], dim=1)
            elif cond_value.shape[:1] == uncond_value.shape[:1]:
                kwargs[name] = torch.cat([cond_value, uncond_value], dim=0)
            else:
                kwargs[name] = cond_value
        else:
            kwargs[name] = cond_value
    return type(condition)(**kwargs)


def install_batched_cfg(model) -> bool:
    """Batch cond/uncond CFG denoise calls into a single net forward.

    This is disabled by default because it trades lower launch overhead for
    higher peak activation memory. Enable with ONECOMP_COSMOS_BATCHED_CFG=1.
    """
    flag = os.environ.get("ONECOMP_COSMOS_BATCHED_CFG", "").lower()
    if flag not in {"1", "true", "yes", "on"}:
        return False
    if hasattr(model, "_onecomp_original_get_velocity_fn_from_batch"):
        return False

    from cosmos_transfer2._src.imaginaire.utils import log
    from cosmos_transfer2._src.imaginaire.utils.distributed import get_rank, get_world_size
    from cosmos_transfer2._src.predict2.conditioner import DataType
    from cosmos_transfer2._src.predict2.models.video2world_model_rectified_flow import NUM_CONDITIONAL_FRAMES_KEY
    from megatron.core import parallel_state

    original_get_velocity = model.get_velocity_fn_from_batch

    def get_velocity_fn_from_batch_batched_cfg(self, data_batch, guidance: float = 1.5, is_negative_prompt: bool = False):
        if guidance is None:
            return original_get_velocity(data_batch, guidance, is_negative_prompt=is_negative_prompt)
        if getattr(self.net, "cfg_parallel", False):
            return original_get_velocity(data_batch, guidance, is_negative_prompt=is_negative_prompt)

        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
            log.info(
                f"num_conditional_frames: {num_conditional_frames} is set by data_batch[NUM_CONDITIONAL_FRAMES_KEY]"
            )
        else:
            num_conditional_frames = 0

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        offload_net = os.environ.get("COSMOS_OFFLOAD_NET_DURING_CONDITION", "1") != "0"
        if offload_net:
            log.info("Offloading DiT net to CPU while preparing video/control latents")
            self.net.to("cpu")
            torch.cuda.empty_cache()
        _, x0, control_condition = self.get_data_and_condition(data_batch)
        if offload_net:
            log.info("Moving DiT net back to CUDA after preparing video/control latents")
            self.net.to("cuda")
            torch.cuda.empty_cache()

        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
        )

        latent_control_input = control_condition.latent_control_input
        control_weight = control_condition.control_context_scale
        condition = condition.set_control_condition(latent_control_input=latent_control_input, control_weight=control_weight)
        uncondition = uncondition.set_control_condition(
            latent_control_input=latent_control_input, control_weight=control_weight
        )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        _ = get_world_size()
        _ = get_rank()
        paired_condition = _concat_condition_pair(condition, uncondition)

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            noise_x = noise_x.to(**self.tensor_kwargs)
            batched_noise = torch.cat([noise, noise], dim=0)
            batched_noise_x = torch.cat([noise_x, noise_x], dim=0)
            batched_timestep = torch.cat([timestep, timestep], dim=0)
            pred = self.denoise(batched_noise, batched_noise_x, batched_timestep, paired_condition)
            cond_v, uncond_v = pred.chunk(2, dim=0)
            return cond_v + guidance * (cond_v - uncond_v)

        return velocity_fn

    model._onecomp_original_get_velocity_fn_from_batch = original_get_velocity
    model.get_velocity_fn_from_batch = MethodType(get_velocity_fn_from_batch_batched_cfg, model)
    print("[cosmos-opt] installed batched CFG velocity function", flush=True)
    return True


@dataclass
class _TimerRecord:
    name: str
    kind: str
    start: torch.cuda.Event
    end: torch.cuda.Event


class CudaModuleTimer:
    """Low-overhead CUDA event profiler for selected Cosmos modules."""

    def __init__(self, net: nn.Module, *, max_rows: int = 40) -> None:
        self.records: list[_TimerRecord] = []
        self.handles = []
        self.max_rows = max_rows
        self._install(net)
        atexit.register(self.report)

    @staticmethod
    def _module_kind(name: str) -> str | None:
        if name == "x_embedder":
            return "embed.x"
        if name.startswith("control_embedder"):
            return "embed.control"
        if name.startswith("after_proj"):
            return "after_proj"
        if name.endswith(".self_attn"):
            return "attn.self"
        if name.endswith(".cross_attn"):
            return "attn.cross"
        if name.endswith(".mlp"):
            return "mlp"
        if name == "final_layer":
            return "final"
        return None

    def _install(self, net: nn.Module) -> None:
        for name, module in net.named_modules():
            kind = self._module_kind(name)
            if kind is None:
                continue

            def pre_hook(_module, _args, name=name, kind=kind):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                rec = _TimerRecord(name=name, kind=kind, start=start, end=end)
                stack = getattr(_module, "_onecomp_timer_stack", None)
                if stack is None:
                    stack = []
                    setattr(_module, "_onecomp_timer_stack", stack)
                stack.append(rec)
                self.records.append(rec)

            def post_hook(_module, _args, _output):
                stack = getattr(_module, "_onecomp_timer_stack")
                stack.pop().end.record()

            self.handles.append(module.register_forward_pre_hook(pre_hook))
            self.handles.append(module.register_forward_hook(post_hook))
        print(f"[cosmos-profile] installed CUDA module timer hooks: {len(self.handles)//2}", flush=True)

    def report(self) -> None:
        if not self.records or not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        by_name: dict[str, list[float]] = defaultdict(list)
        by_kind: dict[str, list[float]] = defaultdict(list)
        for rec in self.records:
            ms = rec.start.elapsed_time(rec.end)
            by_name[rec.name].append(ms)
            by_kind[rec.kind].append(ms)

        print("[cosmos-profile] totals by kind:", flush=True)
        for kind, vals in sorted(by_kind.items(), key=lambda item: sum(item[1]), reverse=True):
            print(
                f"[cosmos-profile]   {kind:14s} total={sum(vals):9.3f} ms "
                f"calls={len(vals):5d} avg={sum(vals)/len(vals):7.3f} ms",
                flush=True,
            )

        print("[cosmos-profile] top modules:", flush=True)
        rows = sorted(by_name.items(), key=lambda item: sum(item[1]), reverse=True)[: self.max_rows]
        for name, vals in rows:
            print(
                f"[cosmos-profile]   {name:48s} total={sum(vals):9.3f} ms "
                f"calls={len(vals):4d} avg={sum(vals)/len(vals):7.3f} ms",
                flush=True,
            )


def maybe_install_cuda_module_timer(net: nn.Module) -> CudaModuleTimer | None:
    flag = os.environ.get("ONECOMP_COSMOS_PROFILE_MODULES", "").lower()
    if flag not in {"1", "true", "yes", "on"}:
        return None
    max_rows = int(os.environ.get("ONECOMP_COSMOS_PROFILE_ROWS", "40"))
    return CudaModuleTimer(net, max_rows=max_rows)


def apply_runtime_optimizations(net: nn.Module) -> dict[str, int]:
    """Apply safe inference-only optimizations to an official Cosmos net."""
    return {
        "zero_weight_control_input_skip": int(install_zero_weight_control_input_skip(net)),
        "zero_control_patch_embed_skip": install_zero_control_patch_embed_skip(net),
    }


def apply_pipeline_optimizations(inference_pipeline) -> dict[str, int]:
    """Apply optimizations that need access to the official pipeline/model."""
    return {
        "batched_cfg": int(install_batched_cfg(inference_pipeline.model)),
    }
