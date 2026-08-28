"""Pure tensor helpers for ray-adaptive observability control.

The helpers in this module contain no model, dataset, or evaluation logic.  A
calibrated RAOC state is detached by the caller; the modal projection helper
keeps the residual tensor attached so the ordinary medium-network gradient
path remains available.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor
from torch.autograd import Function


LOCAL_SCALE_EPS = 1e-12


def observability_gates(sigma: Tensor) -> Tensor:
    """Return the OCMC global prior using its fixed median rule."""

    sigma = sigma.detach().reshape(-1)
    if sigma.numel() == 0:
        raise ValueError("sigma must contain at least one mode")
    reference = torch.median(sigma).clamp_min(LOCAL_SCALE_EPS)
    gates = sigma.square() / (sigma.square() + reference.square())
    return gates.clamp(0.0, 1.0)


def calibrate_local_scales(
    evidence: Tensor,
    eps: float = LOCAL_SCALE_EPS,
    quantile: float = 0.50,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Calibrate train-only evidence scales with deterministic fallback.

    Args:
        evidence: Tensor with shape ``[population, modes]``.
        eps: Degeneracy threshold required by the preflight protocol.
        quantile: Evidence quantile used for the local scale.  The default
            ``0.50`` preserves the historical ``torch.median`` behavior used
            by existing RAOC configurations.  Other quantiles use PyTorch's
            deterministic linear interpolation semantics.

    Returns:
        ``(q, active, fallback_mean)``.  ``q`` is the median where positive,
        otherwise the mean where positive, and zero for inactive modes.
    """

    if not math.isfinite(float(quantile)) or not 0.0 <= float(quantile) <= 1.0:
        raise ValueError(f"quantile must be finite and in [0, 1], got {quantile}")
    values = evidence.detach().float()
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(f"evidence must have shape [N, modes], got {tuple(values.shape)}")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("evidence must be finite")
    if float(quantile) == 0.50:
        median = torch.median(values, dim=0).values
    else:
        median = torch.quantile(values, float(quantile), dim=0, interpolation="linear")
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


class _FusedRAOCFunction(Function):
    """CUDA RAOC operator with a detached gate and first-order residual VJP."""

    @staticmethod
    def forward(
        ctx,
        delta_std: Tensor,
        basis: Tensor,
        global_gate: Tensor,
        local_scale: Tensor,
        active: Tensor,
        raw_medium: Tensor,
        raw_directions: Tensor,
        medium_rgb: Tensor,
        medium_bs: Tensor,
        medium_attn: Tensor,
        d_rgb: Tensor,
        d_bs: Tensor,
        d_attn: Tensor,
        xys: Tensor,
        depths: Tensor,
        radii: Tensor,
        conics: Tensor,
        colors: Tensor,
        opacities: Tensor,
        gaussian_ids_sorted: Tensor,
        tile_bins: Tensor,
        height: int,
        width: int,
        block_width: int,
        num_intersects: int,
        density_bias: float,
    ):
        from water_splatting import cuda as _cuda

        outputs = _cuda.raoc_fused_forward(
            delta_std.contiguous(),
            basis.contiguous(),
            global_gate.contiguous(),
            local_scale.contiguous(),
            active.contiguous(),
            raw_medium.contiguous(),
            raw_directions.contiguous(),
            medium_rgb.contiguous(),
            medium_bs.contiguous(),
            medium_attn.contiguous(),
            d_rgb.contiguous(),
            d_bs.contiguous(),
            d_attn.contiguous(),
            xys.contiguous(),
            depths.contiguous(),
            radii.contiguous(),
            conics.contiguous(),
            colors.contiguous(),
            opacities.contiguous(),
            gaussian_ids_sorted.contiguous(),
            tile_bins.contiguous(),
            int(height),
            int(width),
            int(block_width),
            int(num_intersects),
            float(density_bias),
        )
        delta_out, evidence, local_gate, keep_gate, sensitivity = outputs
        ctx.save_for_backward(basis.detach(), keep_gate.detach())
        return delta_out, evidence, local_gate, keep_gate, sensitivity

    @staticmethod
    def backward(ctx, grad_delta_out, grad_evidence, grad_local_gate, grad_keep_gate, grad_sensitivity):
        basis, keep_gate = ctx.saved_tensors
        if grad_delta_out is None:
            grad_delta_out = torch.zeros_like(keep_gate)
        # Match the reference FP32 GEMM accumulation order exactly.  The
        # forward compositor remains fused; using the existing 9-D helper for
        # the VJP avoids introducing a different large-batch reduction order
        # in the detached-gate backward path.
        grad_delta = apply_modal_keep_gate(grad_delta_out, basis, keep_gate)
        # All other inputs are detached control/state or renderer geometry.
        return (
            grad_delta,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_modal_control(
    *,
    delta_std: Tensor,
    basis: Tensor,
    global_gate: Tensor,
    local_scale: Tensor,
    active: Tensor,
    raw_medium: Tensor,
    raw_directions: Tensor,
    medium_rgb: Tensor,
    medium_bs: Tensor,
    medium_attn: Tensor,
    d_rgb: Tensor,
    d_bs: Tensor,
    d_attn: Tensor,
    xys: Tensor,
    depths: Tensor,
    radii: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    gaussian_ids_sorted: Tensor,
    tile_bins: Tensor,
    height: int,
    width: int,
    block_width: int,
    num_intersects: int,
    density_bias: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run the all-nine-mode CUDA RAOC path and return fused diagnostics."""

    if delta_std.ndim != 2 or delta_std.shape[1] != 9:
        raise ValueError(f"delta_std must have shape [N, 9], got {tuple(delta_std.shape)}")
    device = delta_std.device
    if not delta_std.is_cuda:
        raise ValueError("cuda_fused RAOC requires CUDA tensors")
    delta_out, evidence, local_gate, keep_gate, sensitivity = _FusedRAOCFunction.apply(
        delta_std.float(),
        basis.detach().to(device=device, dtype=torch.float32),
        global_gate.detach().to(device=device, dtype=torch.float32).reshape(9),
        local_scale.detach().to(device=device, dtype=torch.float32).reshape(9),
        active.detach().to(device=device, dtype=torch.bool).reshape(9),
        raw_medium.detach().to(device=device, dtype=torch.float32),
        raw_directions.detach().to(device=device, dtype=torch.float32),
        medium_rgb.detach().to(device=device, dtype=torch.float32).reshape(-1, 3),
        medium_bs.detach().to(device=device, dtype=torch.float32).reshape(-1, 3),
        medium_attn.detach().to(device=device, dtype=torch.float32).reshape(-1, 3),
        d_rgb.detach().to(device=device, dtype=torch.float32).reshape(-1, 3),
        d_bs.detach().to(device=device, dtype=torch.float32).reshape(-1, 3),
        d_attn.detach().to(device=device, dtype=torch.float32).reshape(-1, 3),
        xys.detach().to(device=device, dtype=torch.float32),
        depths.detach().to(device=device, dtype=torch.float32).reshape(-1),
        radii.detach().to(device=device),
        conics.detach().to(device=device, dtype=torch.float32),
        colors.detach().to(device=device, dtype=torch.float32),
        opacities.detach().to(device=device, dtype=torch.float32).reshape(-1),
        gaussian_ids_sorted.detach().to(device=device, dtype=torch.int32).reshape(-1),
        tile_bins.detach().to(device=device, dtype=torch.int32),
        int(height),
        int(width),
        int(block_width),
        int(num_intersects),
        float(density_bias),
    )
    # Only the gated residual participates in reconstruction autograd.  The
    # evidence and gate diagnostics are detached control signals, matching the
    # reference path and avoiding a diagnostic graph in normal training.
    return delta_out, evidence.detach(), local_gate.detach(), keep_gate.detach(), sensitivity.detach()
