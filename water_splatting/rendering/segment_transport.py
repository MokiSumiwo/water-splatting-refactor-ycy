"""Differentiable medium transport along fixed Gaussian contribution rays."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def compose_segments(t_front: Tensor, b_front: Tensor, t_back: Tensor, b_back: Tensor):
    return t_front * t_back, b_front + t_front * b_back


def integrate_medium(raw: Tensor, lengths: Tensor):
    """Return cumulative direct optical depth and cumulative backscatter.

    raw[..., :3] is asymptotic radiance; channels 3:6 and 6:9 are
    backscatter and direct extinction logits. lengths are physical lengths.
    """
    ambient = raw[..., :3].sigmoid()
    tau_b = F.softplus(raw[..., 3:6]) * lengths[..., None]
    tau_d = F.softplus(raw[..., 6:9]) * lengths[..., None]
    cumulative_b = torch.cumsum(tau_b, dim=-2)
    previous_b = cumulative_b - tau_b
    emission = (-previous_b).exp() * (-torch.expm1(-tau_b)) * ambient
    cumulative_emission = torch.cumsum(emission, dim=-2)
    zeros = torch.zeros_like(tau_d[..., :1, :])
    return (torch.cat((zeros, tau_d.cumsum(-2)), -2),
            torch.cat((zeros, cumulative_emission), -2), cumulative_b[..., -1, :])


def sample_cumulative(values: Tensor, depths: Tensor, far: Tensor):
    steps = values.shape[-2] - 1
    position = (depths / far[:, None]).clamp(0, 1) * steps
    lower = position.long().clamp_max(steps - 1)
    fraction = (position - lower)[..., None]
    a = values.gather(1, lower[..., None].expand(-1, -1, 3))
    b = values.gather(1, (lower + 1)[..., None].expand(-1, -1, 3))
    return a + fraction * (b - a)


def composite_fixed_rays(depths: Tensor, weights: Tensor, colors: Tensor,
                         medium_raw: Tensor, far: Tensor, ray_norm: Tensor):
    """Compose fixed native alpha weights with sampled heterogeneous medium.

    Piecewise-constant coefficients are integrated on a uniform z partition.
    The terminal medium value extends beyond far to infinity. Interpolation
    of cumulative integrals at Gaussian depths is a quadrature approximation.
    """
    steps = medium_raw.shape[-2]
    lengths = (far * ray_norm / steps)[:, None].expand(-1, steps)
    tau, emission, tau_b_far = integrate_medium(medium_raw, lengths)
    optical_depth = sample_cumulative(tau, depths, far)
    emission_at_depth = sample_cumulative(emission, depths, far)
    direct = (weights[..., None] * (-optical_depth).exp() * colors).sum(1)
    # Summation by parts is exact for fixed discrete alpha weights.
    background = emission[:, -1] + (-tau_b_far).exp() * medium_raw[:, -1, :3].sigmoid()
    residual_t = (1 - weights.sum(1)).clamp_min(0)
    scattering = (weights[..., None] * emission_at_depth).sum(1) + residual_t[:, None] * background
    return direct + scattering


class WorldMediumGrid(nn.Module):
    def __init__(self, initial_raw: Tensor, center: Tensor, half_extent: Tensor, resolution: int = 8):
        super().__init__()
        self.base = nn.Parameter(initial_raw.clone())
        self.delta = nn.Parameter(torch.zeros(1, 9, resolution, resolution, resolution, device=initial_raw.device))
        self.register_buffer("center", center.clone())
        self.register_buffer("half_extent", half_extent.clone().clamp_min(1e-4))

    def forward(self, positions: Tensor, context: Tensor):
        normalized = (positions - self.center) / self.half_extent
        grid = normalized.reshape(1, 1, *normalized.shape[:2], 3)
        residual = F.grid_sample(self.delta, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return self.base + residual[0, :, 0].permute(1, 2, 0)


class RayMediumMLP(nn.Module):
    """Camera-conditioned control with approximately the same parameter count."""

    def __init__(self, initial_raw: Tensor, width: int = 59):
        super().__init__()
        self.base = nn.Parameter(initial_raw.clone())
        self.network = nn.Sequential(nn.Linear(9, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 9))
        self.network.to(initial_raw)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, positions: Tensor, context: Tensor):
        return (self.base + self.network(context))[:, None].expand(-1, positions.shape[1], -1)


class ConstantMedium(nn.Module):
    def __init__(self, initial_raw: Tensor):
        super().__init__()
        self.raw = nn.Parameter(initial_raw.clone())

    def forward(self, positions: Tensor, context: Tensor):
        return self.raw.expand(*positions.shape[:2], -1)
