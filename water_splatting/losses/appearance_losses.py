"""M4 constrained appearance losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sh_residual_mean_anchor_loss(sh_residual: Tensor, visible_mask: Tensor) -> Tensor:
    """Penalize non-DC SH residual color offsets for currently visible Gaussians."""

    mask = visible_mask.reshape(-1)
    if sh_residual.numel() == 0 or not mask.any():
        return sh_residual.new_zeros(())
    return sh_residual[mask].abs().mean()


def dc_softclip_loss(
    *,
    dc_rgb: Tensor,
    visible_mask: Tensor,
    low_transmission_weight: Tensor | None,
    threshold: float,
    beta: float,
) -> Tensor:
    """Soft upper-bound intrinsic DC color in low-transmission regions."""

    mask = visible_mask.reshape(-1)
    if dc_rgb.numel() == 0 or not mask.any():
        return dc_rgb.new_zeros(())

    excess = F.softplus((dc_rgb[mask] - threshold) / max(beta, 1e-6)).square()
    if low_transmission_weight is None:
        return excess.mean()

    weights = low_transmission_weight.reshape(-1)[mask].to(device=dc_rgb.device, dtype=dc_rgb.dtype).clamp(0.0, 1.0)
    denom = weights.sum().clamp_min(1e-6) * dc_rgb.shape[-1]
    return (excess * weights[:, None]).sum() / denom


def dc_channel_balance_loss(
    *,
    dc_rgb: Tensor,
    visible_mask: Tensor,
    low_transmission_weight: Tensor | None,
    margin: float,
    beta: float,
) -> Tensor:
    """Suppress strong red/blue DC dominance in low-transmission regions."""

    mask = visible_mask.reshape(-1)
    if dc_rgb.numel() == 0 or not mask.any():
        return dc_rgb.new_zeros(())

    dc_visible = dc_rgb[mask]
    red_dominance = dc_visible[..., 0] - torch.maximum(dc_visible[..., 1], dc_visible[..., 2]) - margin
    blue_dominance = dc_visible[..., 2] - torch.maximum(dc_visible[..., 0], dc_visible[..., 1]) - margin
    violations = torch.stack((red_dominance, blue_dominance), dim=-1)
    smooth_relu = max(beta, 1e-6) * F.softplus(violations / max(beta, 1e-6))
    penalty = smooth_relu.square()

    if low_transmission_weight is None:
        return penalty.mean()

    weights = low_transmission_weight.reshape(-1)[mask].to(device=dc_rgb.device, dtype=dc_rgb.dtype).clamp(0.0, 1.0)
    denom = weights.sum().clamp_min(1e-6) * penalty.shape[-1]
    return (penalty * weights[:, None]).sum() / denom


def medium_attenuation_order_loss(
    *,
    medium_attn: Tensor,
    low_transmission_weight: Tensor | None,
    margin: float,
    beta: float,
) -> Tensor:
    """Encourage underwater attenuation order red >= green >= blue."""

    if medium_attn.numel() == 0:
        return medium_attn.new_zeros(())

    red = medium_attn[..., 0]
    green = medium_attn[..., 1]
    blue = medium_attn[..., 2]
    green_over_red = green - red - margin
    blue_over_green = blue - green - margin
    violations = torch.stack((green_over_red, blue_over_green), dim=-1)
    smooth_relu = max(beta, 1e-6) * F.softplus(violations / max(beta, 1e-6))
    penalty = smooth_relu.square()

    if low_transmission_weight is None:
        return penalty.mean()

    weights = low_transmission_weight.reshape(*penalty.shape[:-1]).to(
        device=medium_attn.device, dtype=medium_attn.dtype
    ).clamp(0.0, 1.0)
    denom = weights.sum().clamp_min(1e-6) * penalty.shape[-1]
    return (penalty * weights[..., None]).sum() / denom


def low_transmission_weights(
    *,
    sampled_attn: Tensor,
    depths: Tensor,
    threshold: float,
    temperature: float,
) -> Tensor:
    """Build a Gaussian weight that is high where estimated transmission is low."""

    optical_depth = (sampled_attn.reshape(-1) * depths.reshape(-1).detach()).clamp_min(0.0)
    transmission = torch.exp(-optical_depth).clamp(0.0, 1.0)
    return torch.sigmoid((threshold - transmission) / max(temperature, 1e-6))
