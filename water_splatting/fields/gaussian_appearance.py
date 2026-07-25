"""Gaussian appearance helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from water_splatting.sh import spherical_harmonics


@dataclass
class DualColorOutput:
    """Gaussian colors split into underwater and intrinsic clear branches."""

    intrinsic_rgb: Tensor
    underwater_rgb: Tensor
    view_residual: Tensor
    luminance_residual: Tensor
    chroma_residual: Tensor


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


def compute_dual_gaussian_colors(
    *,
    means: Tensor,
    features_dc: Tensor,
    features_rest: Tensor,
    camera_position: Tensor,
    sh_degree: int,
    active_sh_degree: int,
    luminance_scale: float,
    chroma_scale: float,
) -> DualColorOutput:
    """Compute underwater RGB and SH-filtered intrinsic clear RGB."""

    if sh_degree <= 0:
        rgb = torch.sigmoid(features_dc)
        zeros = torch.zeros_like(rgb)
        return DualColorOutput(
            intrinsic_rgb=rgb,
            underwater_rgb=rgb,
            view_residual=zeros,
            luminance_residual=zeros,
            chroma_residual=zeros,
        )

    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    viewdirs = means.detach() - camera_position.detach()
    viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    if active_sh_degree <= 0 or features_rest.numel() == 0:
        dc_rgb = torch.clamp(spherical_harmonics(0, viewdirs, colors[:, :1, :]) + 0.5, min=0.0)
        zeros = torch.zeros_like(dc_rgb)
        return DualColorOutput(
            intrinsic_rgb=dc_rgb,
            underwater_rgb=dc_rgb,
            view_residual=zeros,
            luminance_residual=zeros,
            chroma_residual=zeros,
        )

    active_color = spherical_harmonics(active_sh_degree, viewdirs, colors)
    dc_color = spherical_harmonics(0, viewdirs, colors[:, :1, :])
    view_residual = active_color - dc_color
    luminance_residual = view_residual.mean(dim=-1, keepdim=True).expand_as(view_residual)
    chroma_residual = view_residual - luminance_residual

    underwater_rgb = torch.clamp(active_color + 0.5, min=0.0)
    intrinsic_color = dc_color + float(luminance_scale) * luminance_residual + float(chroma_scale) * chroma_residual
    intrinsic_rgb = torch.clamp(intrinsic_color + 0.5, min=0.0)
    return DualColorOutput(
        intrinsic_rgb=intrinsic_rgb,
        underwater_rgb=underwater_rgb,
        view_residual=view_residual,
        luminance_residual=luminance_residual,
        chroma_residual=chroma_residual,
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
    viewdirs = means.detach() - camera_position.detach()
    viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    active_color = spherical_harmonics(active_sh_degree, viewdirs, colors)
    dc_color = spherical_harmonics(0, viewdirs, colors[:, :1, :])
    return active_color - dc_color
