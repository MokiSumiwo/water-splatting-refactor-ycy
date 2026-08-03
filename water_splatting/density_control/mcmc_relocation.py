"""Relocation math for MCMC-style Gaussian density control.

This module ports the official 3DGS-MCMC relocation transform to pure Torch.
For a parent assigned ``N - 1`` relocated/newborn children, all ``N`` coincident
copies receive the same opacity and scale. The opacity split preserves the
center alpha compositing value, while the scale coefficient preserves the
one-dimensional projected alpha mass used by the official CUDA kernel.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def logit_to_alpha(logits: Tensor) -> Tensor:
    """Convert opacity logits to alpha values with a stable float dtype."""

    return torch.sigmoid(logits.float())


def alpha_to_logit(alpha: Tensor, eps: float = 1e-6) -> Tensor:
    """Convert alpha values to logits after clamping away from 0 and 1."""

    alpha = alpha.float().clamp(eps, 1.0 - eps)
    return torch.logit(alpha)


def _relocation_denominator(new_alpha: Tensor, total_copies: Tensor, max_copies: int = 51) -> Tensor:
    """Return the denominator from 3DGS-MCMC Eq. 9.

    This mirrors ``cuda_rasterizer/utils.cu::compute_relocation`` from the
    official implementation:

    ``sum_{i=1}^{N} sum_{k=0}^{i-1} C(i-1,k) (-1)^k a'^(k+1) / sqrt(k+1)``.
    """

    new_alpha = new_alpha.float().reshape(-1)
    total_copies = total_copies.reshape(-1).to(device=new_alpha.device, dtype=torch.long)
    denom = torch.zeros_like(new_alpha)
    max_n = int(min(max(int(total_copies.max().item()) if total_copies.numel() else 1, 1), max_copies - 1))
    for i in range(1, max_n + 1):
        active = total_copies >= i
        if not bool(active.any()):
            continue
        active_alpha = new_alpha[active]
        term_sum = torch.zeros_like(active_alpha)
        for k in range(i):
            coeff = torch.tensor(
                ((-1.0) ** k) * math.comb(i - 1, k) / math.sqrt(k + 1),
                device=new_alpha.device,
                dtype=new_alpha.dtype,
            )
            term_sum = term_sum + coeff * active_alpha.pow(k + 1)
        denom[active] = denom[active] + term_sum
    return denom


def relocation_alpha_and_scale(
    parent_alpha: Tensor,
    parent_scale: Tensor,
    total_copies: Tensor,
    *,
    max_copies: int = 51,
    min_output_alpha: float = 0.005,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Return relocated per-copy alpha and scale in activated space.

    ``parent_scale`` is the activated Gaussian scale, not the log-scale
    parameter. ``total_copies`` includes the parent itself. The output alpha is
    clamped like the official Python wrapper after the scale coefficient has
    been computed from the raw split alpha.
    """

    parent_alpha = parent_alpha.float().reshape(-1).clamp(eps, 1.0 - eps)
    total_copies = total_copies.reshape(-1).to(device=parent_alpha.device, dtype=torch.long)
    total_copies = total_copies.clamp(min=1, max=max_copies - 1)
    parent_scale = parent_scale.float()
    raw_new_alpha = 1.0 - torch.pow((1.0 - parent_alpha).clamp_min(eps), 1.0 / total_copies.float())
    denom = _relocation_denominator(raw_new_alpha, total_copies, max_copies=max_copies).clamp_min(eps)
    scale_coeff = (parent_alpha / denom).reshape(-1, *([1] * (parent_scale.ndim - 1)))
    new_scale = (scale_coeff * parent_scale).clamp_min(eps)
    new_alpha = raw_new_alpha.clamp(min=min_output_alpha, max=1.0 - torch.finfo(torch.float32).eps)
    return new_alpha, new_scale


def relocation_logits_and_log_scales(
    parent_opacity_logits: Tensor,
    parent_log_scales: Tensor,
    child_counts: Tensor,
    *,
    max_copies: int = 51,
    min_output_alpha: float = 0.005,
) -> tuple[Tensor, Tensor]:
    """Return relocated opacity logits and log-scales for unique parents."""

    parent_alpha = logit_to_alpha(parent_opacity_logits).reshape(-1)
    parent_scale = parent_log_scales.float().exp()
    total_copies = child_counts.reshape(-1).to(device=parent_alpha.device, dtype=torch.long) + 1
    new_alpha, new_scale = relocation_alpha_and_scale(
        parent_alpha,
        parent_scale,
        total_copies,
        max_copies=max_copies,
        min_output_alpha=min_output_alpha,
    )
    new_logits = alpha_to_logit(new_alpha).to(device=parent_opacity_logits.device, dtype=parent_opacity_logits.dtype)
    new_log_scales = torch.log(new_scale).to(device=parent_log_scales.device, dtype=parent_log_scales.dtype)
    return new_logits.reshape_as(parent_opacity_logits), new_log_scales.reshape_as(parent_log_scales)


def split_parent_opacity_logits(parent_logits: Tensor, child_counts: Tensor) -> Tensor:
    """Return the shared parent/child logit after splitting opacity mass.

    ``child_counts`` is the number of relocated or newborn children assigned to
    each parent. The parent itself counts as one additional coincident Gaussian.
    For original alpha ``a`` and total coincident copies ``k``, the new per-copy
    alpha ``a'`` satisfies ``1 - (1 - a') ** k == a``.
    """

    parent_alpha = logit_to_alpha(parent_logits).reshape(-1)
    total_copies = child_counts.reshape(-1).to(device=parent_alpha.device, dtype=parent_alpha.dtype) + 1.0
    total_copies = total_copies.clamp_min(1.0)
    split_alpha = 1.0 - torch.pow((1.0 - parent_alpha).clamp_min(1e-12), 1.0 / total_copies)
    return alpha_to_logit(split_alpha).to(device=parent_logits.device, dtype=parent_logits.dtype).reshape_as(parent_logits)


def relocated_child_opacity_logits(parent_logits: Tensor, child_counts: Tensor, parent_inverse: Tensor) -> Tensor:
    """Return child opacity logits aligned to sampled child order."""

    split_logits = split_parent_opacity_logits(parent_logits, child_counts).reshape(-1)
    return split_logits[parent_inverse.reshape(-1)]


def alpha_after_coincident_split(parent_alpha: Tensor, total_copies: int) -> Tensor:
    """Compute combined alpha after exact opacity mass splitting."""

    if total_copies < 1:
        raise ValueError("total_copies must be >= 1")
    parent_alpha = parent_alpha.float().clamp(1e-6, 1.0 - 1e-6)
    split_alpha = 1.0 - torch.pow(1.0 - parent_alpha, 1.0 / float(total_copies))
    return 1.0 - torch.pow(1.0 - split_alpha, float(total_copies))
