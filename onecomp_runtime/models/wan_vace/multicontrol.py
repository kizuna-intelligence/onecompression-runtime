"""Shared-branch multi-control helpers for Wan VACE.

The stock Hugging Face WanVACE pipeline prepares one conditioning latent tensor
and the transformer runs one VACE branch.  These helpers keep the same learned
VACE weights, but run that branch once per control tensor and add the produced
hints with per-control scales.
"""
from __future__ import annotations

import types
from collections.abc import Sequence
from typing import Any

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput


def _as_control_list(control_hidden_states: Any) -> list[torch.Tensor] | None:
    if isinstance(control_hidden_states, (list, tuple)):
        if not control_hidden_states:
            raise ValueError("control_hidden_states list must not be empty")
        if not all(isinstance(x, torch.Tensor) for x in control_hidden_states):
            raise TypeError("all control_hidden_states entries must be tensors")
        return list(control_hidden_states)
    return None


def _normalise_scales(
    control_hidden_states: Sequence[torch.Tensor],
    control_hidden_states_scale: Any,
    num_vace_layers: int,
) -> list[tuple[torch.Tensor, ...]]:
    first = control_hidden_states[0]
    if control_hidden_states_scale is None:
        return [tuple(torch.ones((), device=first.device, dtype=first.dtype) for _ in range(num_vace_layers))
                for _ in control_hidden_states]

    if isinstance(control_hidden_states_scale, torch.Tensor):
        scale = control_hidden_states_scale.to(device=first.device, dtype=first.dtype)
        if scale.dim() == 1:
            if scale.numel() != num_vace_layers:
                raise ValueError(
                    f"control scale length {scale.numel()} must equal {num_vace_layers}"
                )
            return [tuple(torch.unbind(scale)) for _ in control_hidden_states]
        if scale.dim() == 2:
            if scale.shape != (len(control_hidden_states), num_vace_layers):
                raise ValueError(
                    f"control scale shape {tuple(scale.shape)} must be "
                    f"({len(control_hidden_states)}, {num_vace_layers})"
                )
            return [tuple(torch.unbind(row)) for row in scale]
        raise ValueError("control scale tensor must be 1D or 2D")

    if isinstance(control_hidden_states_scale, (list, tuple)):
        if len(control_hidden_states_scale) == num_vace_layers and not isinstance(
            control_hidden_states_scale[0], (list, tuple, torch.Tensor)
        ):
            scale = torch.tensor(control_hidden_states_scale, device=first.device, dtype=first.dtype)
            return [tuple(torch.unbind(scale)) for _ in control_hidden_states]
        if len(control_hidden_states_scale) != len(control_hidden_states):
            raise ValueError(
                f"got {len(control_hidden_states_scale)} control scale entries for "
                f"{len(control_hidden_states)} controls"
            )
        out = []
        for item in control_hidden_states_scale:
            scale = torch.as_tensor(item, device=first.device, dtype=first.dtype)
            if scale.numel() == 1:
                scale = scale.repeat(num_vace_layers)
            if scale.numel() != num_vace_layers:
                raise ValueError(
                    f"per-control scale length {scale.numel()} must equal {num_vace_layers}"
                )
            out.append(tuple(torch.unbind(scale.reshape(num_vace_layers))))
        return out

    scale = torch.as_tensor(control_hidden_states_scale, device=first.device, dtype=first.dtype)
    if scale.numel() != 1:
        raise ValueError("scalar control scale expected")
    scale = scale.repeat(num_vace_layers)
    return [tuple(torch.unbind(scale)) for _ in control_hidden_states]


def _patch_or_trim_control(control: torch.Tensor, target_seq: int, batch_size: int) -> torch.Tensor:
    control = control.flatten(2).transpose(1, 2)
    seq_delta = target_seq - control.size(1)
    if seq_delta > 0:
        pad = control.new_zeros(batch_size, seq_delta, control.size(2))
        control = torch.cat([control, pad], dim=1)
    elif seq_delta < 0:
        control = control[:, :target_seq]
    return control


def enable_shared_vace_multicontrol(model: torch.nn.Module) -> torch.nn.Module:
    """Patch a WanVACE transformer to accept a list of control tensors.

    The original single-control forward path is preserved.  When
    ``control_hidden_states`` is a list/tuple, every tensor is sent through the
    existing VACE branch and the resulting hints are added at each configured
    VACE layer with the corresponding per-control scale.
    """
    if getattr(model, "_onecomp_shared_vace_multicontrol", False):
        return model

    original_forward = model.forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        control_hidden_states: torch.Tensor | Sequence[torch.Tensor] | None = None,
        control_hidden_states_scale: Any = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        controls = _as_control_list(control_hidden_states)
        if controls is None:
            return original_forward(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                control_hidden_states=control_hidden_states,
                control_hidden_states_scale=control_hidden_states_scale,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
            )

        batch_size, _num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        num_vace_layers = len(self.config.vace_layers)
        control_scales = _normalise_scales(controls, control_hidden_states_scale, num_vace_layers)

        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        control_hint_lists: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        for control, scales in zip(controls, control_scales, strict=True):
            control = control.to(device=hidden_states.device, dtype=hidden_states.dtype)
            control = self.vace_patch_embedding(control)
            control = _patch_or_trim_control(control, hidden_states.size(1), batch_size)

            hints = []
            for i, block in enumerate(self.vace_blocks):
                conditioning_states, control = block(
                    hidden_states,
                    encoder_hidden_states,
                    control,
                    timestep_proj,
                    rotary_emb,
                )
                if conditioning_states is None:
                    conditioning_states = torch.zeros_like(hidden_states)
                hints.append((conditioning_states, scales[i].to(device=hidden_states.device)))
            control_hint_lists.append(hints[::-1])

        for i, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            if i in self.config.vace_layers:
                for hints in control_hint_lists:
                    control_hint, scale = hints.pop()
                    hidden_states = hidden_states + control_hint * scale

        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    model.forward = types.MethodType(forward, model)
    model._onecomp_shared_vace_multicontrol = True
    return model


def expand_layer_scale(
    conditioning_scale: float | Sequence[float] | torch.Tensor,
    branch_scales: Sequence[float],
    *,
    num_vace_layers: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a ``[num_controls, num_vace_layers]`` scale matrix."""
    if isinstance(conditioning_scale, torch.Tensor):
        base = conditioning_scale.to(device=device, dtype=dtype)
    elif isinstance(conditioning_scale, (list, tuple)):
        base = torch.tensor(conditioning_scale, device=device, dtype=dtype)
    else:
        base = torch.full((num_vace_layers,), float(conditioning_scale), device=device, dtype=dtype)
    if base.numel() == 1:
        base = base.repeat(num_vace_layers)
    if base.numel() != num_vace_layers:
        raise ValueError(f"conditioning_scale length {base.numel()} must equal {num_vace_layers}")
    branches = torch.tensor(branch_scales, device=device, dtype=dtype).reshape(-1, 1)
    return branches * base.reshape(1, -1)
