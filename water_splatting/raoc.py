"""Pure tensor helpers for ray-adaptive observability control.

The helpers in this module contain no model, dataset, or evaluation logic.  A
calibrated RAOC state is detached by the caller; the modal projection helper
keeps the residual tensor attached so the ordinary medium-network gradient
path remains available.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


LOCAL_SCALE_EPS = 1e-12


def observability_gates(sigma: Tensor) -> Tensor:
    """Return the OCMC global prior using its fixed median rule."""

    sigma = sigma.detach().reshape(-1)
    if sigma.numel() == 0:
        raise ValueError("sigma must contain at least one mode")
    reference = torch.median(sigma).clamp_min(LOCAL_SCALE_EPS)
    gates = sigma.square() / (sigma.square() + reference.square())
    return gates.clamp(0.0, 1.0)


def calibrate_local_scales(evidence: Tensor, eps: float = LOCAL_SCALE_EPS) -> Tuple[Tensor, Tensor, Tensor]:
    """Calibrate train-only median evidence scales with deterministic fallback.

    Args:
        evidence: Tensor with shape ``[population, modes]``.
        eps: Degeneracy threshold required by the preflight protocol.

    Returns:
        ``(q, active, fallback_mean)``.  ``q`` is the median where positive,
        otherwise the mean where positive, and zero for inactive modes.
    """

    values = evidence.detach().float()
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(f"evidence must have shape [N, modes], got {tuple(values.shape)}")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("evidence must be finite")
    median = torch.median(values, dim=0).values
    mean = values.mean(dim=0)
    use_mean = median <= float(eps)
    active = torch.where(use_mean, mean > float(eps), median > float(eps))
    q = torch.where(use_mean, mean, median)
    q = torch.where(active, q.clamp_min(float(eps)), torch.zeros_like(q))
    return q, active, use_mean


def local_keep_gates(evidence: Tensor, q: Tensor, active: Tensor, eps: float = LOCAL_SCALE_EPS) -> Tensor:
    """Map local evidence to the detached monotone local keep gate."""

    evidence = evidence.detach().float()
    q = q.detach().reshape(-1).to(device=evidence.device, dtype=evidence.dtype)
    active = active.detach().reshape(-1).to(device=evidence.device, dtype=torch.bool)
    if evidence.ndim != 2 or evidence.shape[1] != q.numel() or q.numel() != active.numel():
        raise ValueError("evidence, q, and active mode dimensions do not agree")
    safe_q = q.clamp_min(float(eps))
    gates = evidence.square() / (evidence.square() + safe_q.square())
    gates = torch.where(active[None, :], gates, torch.zeros_like(gates))
    return gates.clamp(0.0, 1.0)


def ray_keep_gates(g_obs: Tensor, g_local: Tensor) -> Tensor:
    """Combine the global prior and local support with probabilistic OR."""

    g_local = g_local.detach()
    g_obs = g_obs.detach().reshape(-1).to(device=g_local.device, dtype=g_local.dtype)
    if g_local.ndim != 2 or g_local.shape[1] != g_obs.numel():
        raise ValueError("g_local and g_obs mode dimensions do not agree")
    keep = 1.0 - (1.0 - g_obs[None, :]) * (1.0 - g_local)
    # The maximum only removes sub-ulp downward roundoff at g_local == 0;
    # analytically this is the same probabilistic-OR expression.
    keep = torch.maximum(keep, g_obs[None, :])
    return keep.clamp(0.0, 1.0)


def modal_coefficients(delta_std: Tensor, basis: Tensor) -> Tensor:
    """Project standardized residuals into a column-basis modal coordinate."""

    basis = basis.detach()
    if basis.ndim != 2 or basis.shape[0] != basis.shape[1]:
        raise ValueError(f"basis must be square, got {tuple(basis.shape)}")
    if delta_std.shape[-1] != basis.shape[0]:
        raise ValueError("delta and basis dimensions do not agree")
    accumulator_dtype = torch.float32 if delta_std.dtype in (torch.float16, torch.bfloat16) else delta_std.dtype
    return delta_std.to(dtype=accumulator_dtype) @ basis.to(dtype=accumulator_dtype)


def apply_standardized_projector(delta_raw: Tensor, projector: Tensor, scale: Tensor) -> Tensor:
    """Apply a standardized raw-space projector with stable accumulator dtype."""

    if delta_raw.shape[-1] != 9 or tuple(projector.shape) != (9, 9) or scale.reshape(-1).numel() != 9:
        raise ValueError("delta, projector, and scale must describe the 9-D medium space")
    scale = scale.detach().reshape(1, 9).to(device=delta_raw.device, dtype=torch.float32).clamp_min(1e-6)
    projector = projector.detach().to(device=delta_raw.device, dtype=torch.float32)
    projected_std = (delta_raw.float() / scale) @ projector.T
    return (projected_std * scale).to(dtype=delta_raw.dtype)


def apply_modal_keep_gate(delta_std: Tensor, basis: Tensor, g_keep: Tensor) -> Tensor:
    """Apply a per-ray modal gate while preserving residual gradients."""

    basis = basis.detach().to(device=delta_std.device)
    g_keep = g_keep.detach().to(device=delta_std.device, dtype=delta_std.dtype)
    # Medium MLP outputs can be float16 under the tensor-contract backend.  A
    # float32 accumulator keeps the modal roundtrip consistent with OCMC.
    accumulator_dtype = torch.float32 if delta_std.dtype in (torch.float16, torch.bfloat16) else delta_std.dtype
    delta_acc = delta_std.to(dtype=accumulator_dtype)
    basis_acc = basis.to(dtype=accumulator_dtype)
    gate_acc = g_keep.to(dtype=accumulator_dtype)
    coeff = delta_acc @ basis_acc
    return ((coeff * gate_acc) @ basis_acc.T).to(dtype=delta_std.dtype)
