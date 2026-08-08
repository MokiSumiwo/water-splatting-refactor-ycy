"""Gaussian appearance helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from water_splatting.sh import spherical_harmonics


@dataclass
class GaussianColorOutput:
    """Current-view Gaussian colors and optional bounded-color diagnostics."""

    rgb: Tensor
    logits: Tensor | None = None
    sigmoid_derivative: Tensor | None = None
    dc_rgb: Tensor | None = None
    dc_logits: Tensor | None = None


def _sh_viewdirs(means: Tensor, camera_position: Tensor) -> Tensor:
    viewdirs = means.detach() - camera_position.detach()
    return viewdirs / viewdirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)


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
        viewdirs = _sh_viewdirs(means, camera_position)
        rgbs = spherical_harmonics(active_sh_degree, viewdirs, colors)
        return torch.clamp(rgbs + 0.5, min=0.0)

    return torch.sigmoid(colors[:, 0, :])


def compute_bounded_gaussian_colors(
    *,
    means: Tensor,
    features_dc: Tensor,
    features_rest: Tensor,
    camera_position: Tensor,
    sh_degree: int,
    active_sh_degree: int,
) -> GaussianColorOutput:
    """Compute bounded full-SH Gaussian RGB values.

    For SH>0, the active SH evaluation is interpreted as RGB logits and then
    passed through sigmoid. The legacy ``+0.5`` offset is not applied.
    """

    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    if sh_degree > 0:
        viewdirs = _sh_viewdirs(means, camera_position)
        logits = spherical_harmonics(active_sh_degree, viewdirs, colors)
        dc_logits = spherical_harmonics(0, viewdirs, colors[:, :1, :])
    else:
        logits = colors[:, 0, :]
        dc_logits = logits
    rgb = torch.sigmoid(logits)
    return GaussianColorOutput(
        rgb=rgb,
        logits=logits,
        sigmoid_derivative=rgb * (1.0 - rgb),
        dc_rgb=torch.sigmoid(dc_logits),
        dc_logits=dc_logits,
    )


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
    viewdirs = _sh_viewdirs(means, camera_position)
    active_color = spherical_harmonics(active_sh_degree, viewdirs, colors)
    dc_color = spherical_harmonics(0, viewdirs, colors[:, :1, :])
    return active_color - dc_color
