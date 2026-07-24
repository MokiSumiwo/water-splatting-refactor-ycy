"""Gaussian appearance helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from water_splatting.sh import spherical_harmonics


def compute_gaussian_colors(
    *,
    means: Tensor,
    features_dc: Tensor,
    features_rest: Tensor,
    camera_position: Tensor,
    sh_degree: int,
    active_sh_degree: int,
) -> Tensor:
    """Compute the original WaterSplatting Gaussian RGB values."""

    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    if sh_degree > 0:
        viewdirs = means.detach() - camera_position.detach()
        viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        rgbs = spherical_harmonics(active_sh_degree, viewdirs, colors)
        return torch.clamp(rgbs + 0.5, min=0.0)

    return torch.sigmoid(colors[:, 0, :])


def compute_gaussian_sh_residual(
    *,
    means: Tensor,
    features_dc: Tensor,
    features_rest: Tensor,
    camera_position: Tensor,
    sh_degree: int,
    active_sh_degree: int,
) -> Tensor:
    """Compute the non-DC SH color residual for the current view."""

    if sh_degree <= 0 or active_sh_degree <= 0 or features_rest.numel() == 0:
        return torch.zeros_like(features_dc)

    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    viewdirs = means.detach() - camera_position.detach()
    viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    active_color = spherical_harmonics(active_sh_degree, viewdirs, colors)
    dc_color = spherical_harmonics(0, viewdirs, colors[:, :1, :])
    return active_color - dc_color
