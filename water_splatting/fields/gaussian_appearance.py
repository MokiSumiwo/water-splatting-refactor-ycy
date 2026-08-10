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
    sh_residual: Tensor | None = None
    color_residual: Tensor | None = None
    positive_utilization: Tensor | None = None
    negative_utilization: Tensor | None = None


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
        sh_residual=logits - dc_logits,
        color_residual=rgb - torch.sigmoid(dc_logits),
    )


def compute_bounded_headroom_gaussian_colors(
    *,
    means: Tensor,
    features_dc: Tensor,
    features_rest: Tensor,
    camera_position: Tensor,
    sh_degree: int,
    active_sh_degree: int,
) -> GaussianColorOutput:
    """Compute bounded headroom-SH Gaussian RGB values.

    The degree-0 SH contribution is interpreted as the base logit ``s0``.
    The active non-DC SH contribution ``r`` is mapped through asymmetric
    positive/negative RGB headroom while preserving the BND-v1 first-order
    Jacobian at ``r=0``.
    """

    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    if sh_degree > 0:
        viewdirs = _sh_viewdirs(means, camera_position)
        full_logits = spherical_harmonics(active_sh_degree, viewdirs, colors)
        dc_logits = spherical_harmonics(0, viewdirs, colors[:, :1, :])
    else:
        full_logits = colors[:, 0, :]
        dc_logits = full_logits

    c0 = torch.sigmoid(dc_logits)
    residual = full_logits - dc_logits
    positive_rgb = c0 + (1.0 - c0) * torch.tanh(c0 * residual)
    negative_rgb = c0 + c0 * torch.tanh((1.0 - c0) * residual)
    rgb = torch.where(residual >= 0.0, positive_rgb, negative_rgb)
    color_residual = rgb - c0
    positive_utilization = torch.where(
        color_residual > 0.0,
        color_residual / (1.0 - c0).clamp_min(1e-8),
        torch.zeros_like(color_residual),
    )
    negative_utilization = torch.where(
        color_residual < 0.0,
        (-color_residual) / c0.clamp_min(1e-8),
        torch.zeros_like(color_residual),
    )
    return GaussianColorOutput(
        rgb=rgb,
        logits=full_logits,
        dc_rgb=c0,
        dc_logits=dc_logits,
        sh_residual=residual,
        color_residual=color_residual,
        positive_utilization=positive_utilization,
        negative_utilization=negative_utilization,
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
