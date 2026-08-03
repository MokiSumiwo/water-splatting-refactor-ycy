"""Loss utilities shared by GMVC diagnostics and future training paths."""

from __future__ import annotations

import torch
from torch import Tensor


def charbonnier_loss(value: Tensor, eps: float = 1e-6) -> Tensor:
    """Robust L1-like loss used for GMVC consistency terms."""

    return torch.sqrt(value.square() + float(eps))


def invert_intrinsic_radiance(
    observed_rgb: Tensor,
    depth: Tensor,
    medium_attn: Tensor,
    medium_bs: Tensor,
    b_inf: Tensor,
    eps: float = 1e-4,
) -> Tensor:
    """Invert the simplified underwater image formation equation.

    J_hat = (I - B_inf * (1 - exp(-beta_B * d))) / (exp(-beta_D * d) + eps)
    """

    if depth.ndim == observed_rgb.ndim - 1:
        depth = depth[..., None]
    transmission = torch.exp(-(medium_attn * depth).clamp_min(0.0))
    backscatter = b_inf * (1.0 - torch.exp(-(medium_bs * depth).clamp_min(0.0)))
    return (observed_rgb - backscatter) / (transmission + float(eps))
