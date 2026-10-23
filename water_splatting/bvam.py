"""Bounded View-Adaptive Mixture of Appearance Experts (BVAM)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from water_splatting.fields.gaussian_appearance import _sh_viewdirs
from water_splatting.sh import spherical_harmonics


def _strict_sigmoid(logits: Tensor) -> Tensor:
    """Sigmoid with a representable open-interval guarantee for float tensors.

    Large logits can round a mathematically sub-unit sigmoid to exactly 1.0
    in float32.  This is a numerical parameterization safeguard, not a
    render-time image clamp; normal-range sigmoid values are unchanged.
    """

    value = torch.sigmoid(logits)
    limit = torch.finfo(value.dtype).eps
    return value.clamp(min=limit, max=1.0 - limit)


@dataclass
class BVAMColorOutput:
    """Bounded expert colors and view-adaptive mixture diagnostics."""

    rgb: Tensor
    shared_logits: Tensor
    dc_logits: Tensor
    dc_rgb: Tensor
    delta_logits: Tensor
    deltas: Tensor
    expert_logits: Tensor
    expert_rgb: Tensor
    gate_logits: Tensor
    gate_probs: Tensor


def compute_bvam_gaussian_colors(
    *,
    means: Tensor,
    features_dc: Tensor,
    features_rest: Tensor,
    bvam_delta: Tensor,
    bvam_gate: Tensor,
    camera_position: Tensor,
    sh_degree: int,
    active_sh_degree: int,
    num_experts: int = 2,
    delta_scale: float = 1.0,
    gate_degree: int = 1,
    gate_temperature: float = 1.0,
    valid_mask: Tensor | None = None,
) -> BVAMColorOutput:
    """Compute K bounded appearance experts and their convex mixture.

    The fixed expert signs are anti-symmetric.  They are invisible at the
    zero initialization, but make the first photometric update point the two
    zero residual heads in opposite directions instead of preserving an
    exact permutation symmetry forever.
    """

    if num_experts < 2:
        raise ValueError("BVAM requires at least two experts")
    if bvam_delta.ndim != 4 or bvam_delta.shape[1] != num_experts or bvam_delta.shape[2] < 1:
        raise ValueError(f"BVAM delta must have shape [N, K, D, 3], got {tuple(bvam_delta.shape)}")
    if bvam_gate.ndim != 3 or bvam_gate.shape[1] != num_experts or bvam_gate.shape[2] < 1:
        raise ValueError(f"BVAM gate must have shape [N, K, D], got {tuple(bvam_gate.shape)}")
    if gate_temperature <= 0.0:
        raise ValueError("BVAM gate temperature must be positive")
    if gate_degree < 0 or gate_degree > 1:
        raise ValueError("BVAM gate degree must be 0 or 1")

    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    if sh_degree > 0:
        viewdirs = _sh_viewdirs(means, camera_position)
        shared_logits = spherical_harmonics(active_sh_degree, viewdirs, colors)
        dc_logits = spherical_harmonics(0, viewdirs, colors[:, :1, :])
    else:
        viewdirs = _sh_viewdirs(means, camera_position)
        shared_logits = colors[:, 0, :]
        dc_logits = shared_logits

    count = int(features_dc.shape[0])
    delta_active_degree = min(int(active_sh_degree), 1)
    delta_coeffs = bvam_delta.reshape(count * num_experts, bvam_delta.shape[2], 3)
    delta_dirs = viewdirs[:, None, :].expand(count, num_experts, 3).reshape(count * num_experts, 3)
    if delta_active_degree == 0:
        delta_logits = delta_coeffs[:, 0, :].reshape(count, num_experts, 3)
    else:
        delta_logits = spherical_harmonics(delta_active_degree, delta_dirs, delta_coeffs).reshape(count, num_experts, 3)
    signs = torch.ones(num_experts, device=bvam_delta.device, dtype=bvam_delta.dtype)
    signs[0::2] = -1.0
    deltas = float(delta_scale) * torch.tanh(delta_logits) * signs[None, :, None]
    if valid_mask is not None:
        valid = valid_mask.reshape(-1, 1, 1).to(device=deltas.device, dtype=deltas.dtype)
        deltas = deltas * valid
    expert_logits = shared_logits[:, None, :] + deltas
    expert_rgb = _strict_sigmoid(expert_logits)

    gate_active_degree = min(int(gate_degree), 1)
    gate_coeffs = bvam_gate.reshape(count * num_experts, bvam_gate.shape[2], 1).expand(-1, -1, 3).contiguous()
    if gate_active_degree == 0:
        gate_logits = gate_coeffs[:, 0, 0].reshape(count, num_experts)
    else:
        gate_logits = spherical_harmonics(gate_active_degree, delta_dirs, gate_coeffs)[..., 0].reshape(count, num_experts)
    gate_probs = torch.softmax(gate_logits / float(gate_temperature), dim=1)
    if valid_mask is not None:
        valid = valid_mask.reshape(-1, 1).to(device=gate_probs.device, dtype=gate_probs.dtype)
        uniform = torch.full_like(gate_probs, 1.0 / float(num_experts))
        gate_probs = torch.where(valid.bool(), gate_probs, uniform)
    rgb = (gate_probs[:, :, None] * expert_rgb).sum(dim=1)
    return BVAMColorOutput(
        rgb=rgb,
        shared_logits=shared_logits,
        dc_logits=dc_logits,
        dc_rgb=_strict_sigmoid(dc_logits),
        delta_logits=delta_logits,
        deltas=deltas,
        expert_logits=expert_logits,
        expert_rgb=expert_rgb,
        gate_logits=gate_logits,
        gate_probs=gate_probs,
    )


def bvam_gate_statistics(output: BVAMColorOutput) -> dict[str, Tensor]:
    """Return differentiable gate/expert statistics for logging or losses."""

    probs = output.gate_probs.clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=1)
    delta_norm = output.deltas.float().norm(dim=-1)
    usage = probs.mean(dim=0)
    return {
        "expert_usage": usage,
        "gate_entropy": entropy,
        "delta_norm": delta_norm,
        "delta_p90": torch.quantile(delta_norm.reshape(-1), 0.90),
    }
