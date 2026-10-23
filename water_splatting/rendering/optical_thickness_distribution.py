"""Mean-preserving optical-thickness distributions on fixed Gaussian rays."""

from __future__ import annotations

import torch
from torch import Tensor


def distributed_transmission(tau: Tensor, eta: Tensor, delta: float = 0.5) -> Tensor:
    """Mix thickness scales 1, 1-delta and 1+delta without changing their mean.

    eta is projected into [0, 1] by the optimizer, not clamped in the forward
    pass, so its derivative remains available at the identity endpoint.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    base = torch.exp(-tau)
    spread = 0.5 * (torch.exp(-(1.0 - delta) * tau) + torch.exp(-(1.0 + delta) * tau))
    return base + eta * (spread - base)


def composite_fixed_transport(
    depths: Tensor,
    weights: Tensor,
    colors: Tensor,
    ambient: Tensor,
    backscatter: Tensor,
    attenuation: Tensor,
    *,
    eta: Tensor | None = None,
    optical_scale: Tensor | float = 1.0,
    delta: float = 0.5,
) -> Tensor:
    """Compose direct, finite medium and tail with unchanged native alpha weights.

    For medium constant along a ray, summation by parts reduces segment plus
    tail scattering to A * (1 - sum_i w_i T_B(d_i)). This also applies to a
    convex mixture of complete transport calls sharing A and Gaussian state.
    """
    depth = depths[..., None]
    weight = weights[..., None]
    tau_d = attenuation[:, None] * optical_scale * depth
    tau_b = backscatter[:, None] * optical_scale * depth
    if eta is None:
        trans_d, trans_b = torch.exp(-tau_d), torch.exp(-tau_b)
    else:
        trans_d = distributed_transmission(tau_d, eta, delta)
        trans_b = distributed_transmission(tau_b, eta, delta)
    direct = (weight * colors * trans_d).sum(1)
    medium = ambient * (1.0 - (weight * trans_b).sum(1))
    return direct + medium
