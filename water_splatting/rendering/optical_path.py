"""Optical transport primitives; no changes to the default renderer."""

from __future__ import annotations

import torch
from torch import Tensor


def positive_normal_transmission(mean: Tensor, variance: Tensor, beta: Tensor) -> Tensor:
    """E[exp(-beta*z) | z >= 0] for a normal depth distribution.

    Zero variance is the point-depth limit. Inputs must have nonnegative
    variance and beta. This describes a conditional depth distribution,
    not a continuous-volume treatment of overlapping Gaussians.
    """
    scale = variance.clamp_min(torch.finfo(variance.dtype).tiny).sqrt()
    standardized = mean / scale
    shifted = standardized - beta * scale
    log_value = -beta * mean + 0.5 * beta.square() * variance
    log_value = log_value + torch.special.log_ndtr(shifted) - torch.special.log_ndtr(standardized)
    integrated = log_value.clamp_max(0).exp()
    return torch.where(variance > 1e-16, integrated, (-beta * mean.clamp_min(0)).exp())


def conditional_depth_parameters(
    means: Tensor, cov3d: Tensor, viewmat: Tensor, conics: Tensor,
    fx: float, fy: float, width: int, height: int,
) -> Tensor:
    """Return [pixel-x slope, pixel-y slope, conditional z variance].

    Reuses the rasterizer conic, including its 0.3 pixel covariance blur,
    and its clipped perspective Jacobian. Packed cov3d uses xx,xy,xz,yy,yz,zz.
    """
    rotation = viewmat[:3, :3].to(means)
    points = means @ rotation.T + viewmat[:3, 3].to(means)
    covariance = means.new_empty(means.shape[0], 3, 3)
    covariance[:, 0, 0] = cov3d[:, 0]
    covariance[:, 0, 1] = covariance[:, 1, 0] = cov3d[:, 1]
    covariance[:, 0, 2] = covariance[:, 2, 0] = cov3d[:, 2]
    covariance[:, 1, 1] = cov3d[:, 3]
    covariance[:, 1, 2] = covariance[:, 2, 1] = cov3d[:, 4]
    covariance[:, 2, 2] = cov3d[:, 5]
    covariance = rotation @ covariance @ rotation.T
    inverse_z = points[:, 2].clamp_min(1e-8).reciprocal()
    xz = (points[:, 0] * inverse_z).clamp(-0.65 * width / fx, 0.65 * width / fx)
    yz = (points[:, 1] * inverse_z).clamp(-0.65 * height / fy, 0.65 * height / fy)
    jacobian = means.new_zeros(means.shape[0], 2, 3)
    jacobian[:, 0, 0] = fx * inverse_z
    jacobian[:, 1, 1] = fy * inverse_z
    jacobian[:, 0, 2] = -fx * xz * inverse_z
    jacobian[:, 1, 2] = -fy * yz * inverse_z
    cross = (covariance[:, 2:3, :] @ jacobian.transpose(-1, -2)).squeeze(1)
    slope_x = cross[:, 0] * conics[:, 0] + cross[:, 1] * conics[:, 1]
    slope_y = cross[:, 0] * conics[:, 1] + cross[:, 1] * conics[:, 2]
    variance = covariance[:, 2, 2] - slope_x * cross[:, 0] - slope_y * cross[:, 1]
    return torch.stack((slope_x, slope_y, variance.clamp_min(0)), -1)
