"""Medium-explainable scene/medium attribution losses.

These helpers build training-only soft supports from image structure, detached
medium color, and detached depth. They intentionally do not use Gaussian
accumulation to construct support, so erroneous high-accumulation water pixels
cannot self-protect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class MediumExplainabilitySupport:
    """Soft support maps used by medium-explainable attribution."""

    flat: Tensor
    medium: Tensor
    far: Tensor
    far_effective: Tensor
    route: Tensor
    capacity: Tensor
    bootstrap: Tensor
    medium_error: Tensor


def _to_nchw(image: Tensor) -> Tensor:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected image with shape [H, W, 3], got {tuple(image.shape)}")
    return image.permute(2, 0, 1).unsqueeze(0)


def _from_nchw_gray(value: Tensor) -> Tensor:
    return value.squeeze(0).permute(1, 2, 0)


def _luma(rgb: Tensor) -> Tensor:
    return 0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2] + 0.0722 * rgb[..., 2:3]


def _chroma(rgb: Tensor) -> Tensor:
    return rgb - rgb.mean(dim=-1, keepdim=True)


def _normalize_depth(depth: Tensor) -> Tensor:
    if depth.ndim == 2:
        depth = depth[..., None]
    depth = depth.detach().float()
    valid = depth[torch.isfinite(depth) & (depth > 0.0)]
    if valid.numel() == 0:
        return torch.zeros_like(depth)
    scale = torch.quantile(valid, 0.95).clamp_min(1e-6)
    return (depth / scale).clamp(0.0, 1.0)


def compute_image_structure_support(
    gt_img: Tensor,
    *,
    gradient_tau: float,
    variance_tau: float,
    variance_kernel: int = 5,
    gradient_dilation: int = 3,
) -> Tensor:
    """Compute low-gradient, low-variance support from the input image."""

    image = gt_img.detach().float().clamp(0.0, 1.0)
    nchw = _to_nchw(image)

    grad_x = torch.zeros_like(image[..., 0:1])
    grad_y = torch.zeros_like(image[..., 0:1])
    grad_x[:, 1:, :] = (image[:, 1:, :] - image[:, :-1, :]).abs().sum(dim=-1, keepdim=True)
    grad_y[1:, :, :] = (image[1:, :, :] - image[:-1, :, :]).abs().sum(dim=-1, keepdim=True)
    grad = grad_x + grad_y
    if gradient_dilation > 1:
        pad = gradient_dilation // 2
        grad = _from_nchw_gray(F.max_pool2d(grad.permute(2, 0, 1).unsqueeze(0), gradient_dilation, 1, pad))

    k = max(int(variance_kernel), 1)
    pad = k // 2
    mean = F.avg_pool2d(nchw, k, stride=1, padding=pad)
    mean_sq = F.avg_pool2d(nchw * nchw, k, stride=1, padding=pad)
    var = (mean_sq - mean * mean).clamp_min(0.0).mean(dim=1, keepdim=True)
    var = _from_nchw_gray(var)

    flat = torch.exp(-grad / max(float(gradient_tau), 1e-6)) * torch.exp(
        -var / max(float(variance_tau), 1e-6)
    )
    return flat.clamp(0.0, 1.0).detach()


def compute_medium_explainability(
    *,
    gt_img: Tensor,
    medium_rgb: Tensor,
    luma_weight: float,
    color_tau: float,
) -> tuple[Tensor, Tensor]:
    """Compute detached medium explainability support and its error."""

    gt = gt_img.detach().float().clamp(0.0, 1.0)
    medium = medium_rgb.detach().float()
    chroma_error = (_chroma(medium) - _chroma(gt)).abs().sum(dim=-1, keepdim=True)
    luma_error = (_luma(medium) - _luma(gt)).abs()
    error = chroma_error + float(luma_weight) * luma_error
    support = torch.exp(-error / max(float(color_tau), 1e-6)).clamp(0.0, 1.0)
    return support.detach(), error.detach()


def compute_far_depth_support(
    depth: Tensor,
    *,
    depth_mid: float,
    depth_temperature: float,
    far_floor: float,
) -> tuple[Tensor, Tensor]:
    """Compute weak detached far-depth support and its floored form."""

    depth_norm = _normalize_depth(depth)
    far = torch.sigmoid((depth_norm - float(depth_mid)) / max(float(depth_temperature), 1e-6)).clamp(0.0, 1.0)
    floor = float(far_floor)
    far_effective = (floor + (1.0 - floor) * far).clamp(0.0, 1.0)
    return far.detach(), far_effective.detach()


def build_route_capacity_support(
    *,
    gt_img: Tensor,
    medium_rgb: Tensor,
    depth: Tensor,
    gradient_tau: float,
    variance_tau: float,
    color_tau: float,
    luma_weight: float,
    far_floor: float,
    depth_mid: float,
    depth_temperature: float,
    variance_kernel: int = 5,
    gradient_dilation: int = 3,
    use_flatness: bool = True,
    use_medium: bool = True,
    use_far: bool = True,
) -> MediumExplainabilitySupport:
    """Build route/capacity supports from detached evidence only."""

    ones = gt_img.new_ones((*gt_img.shape[:2], 1))
    flat = (
        compute_image_structure_support(
            gt_img,
            gradient_tau=gradient_tau,
            variance_tau=variance_tau,
            variance_kernel=variance_kernel,
            gradient_dilation=gradient_dilation,
        )
        if use_flatness
        else ones
    )
    medium, medium_error = (
        compute_medium_explainability(
            gt_img=gt_img,
            medium_rgb=medium_rgb,
            luma_weight=luma_weight,
            color_tau=color_tau,
        )
        if use_medium
        else (ones, ones.new_zeros(ones.shape))
    )
    far, far_effective = (
        compute_far_depth_support(
            depth,
            depth_mid=depth_mid,
            depth_temperature=depth_temperature,
            far_floor=far_floor,
        )
        if use_far
        else (ones, ones)
    )
    route = (flat * medium).clamp(0.0, 1.0).detach()
    capacity = (flat.square() * medium.square() * far_effective).clamp(0.0, 1.0).detach()
    bootstrap = (flat * far_effective).clamp(0.0, 1.0).detach()
    return MediumExplainabilitySupport(
        flat=flat.detach(),
        medium=medium.detach(),
        far=far.detach(),
        far_effective=far_effective.detach(),
        route=route,
        capacity=capacity,
        bootstrap=bootstrap,
        medium_error=medium_error.detach(),
    )


def build_training_routed_prediction(
    *,
    pred_img: Tensor,
    medium_rgb: Tensor,
    route_support: Tensor,
    min_scene_weight: float,
) -> Tensor:
    """Build the training-only routed image while leaving inference untouched."""

    scene_weight = 1.0 - (1.0 - float(min_scene_weight)) * route_support.detach().clamp(0.0, 1.0)
    return scene_weight * pred_img + (1.0 - scene_weight) * medium_rgb


def weighted_rgb_l1(pred: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    """Per-channel L1 with a single-channel soft pixel weight."""

    if weight.ndim == 2:
        weight = weight[..., None]
    denom = weight.sum().clamp_min(1e-6) * float(pred.shape[-1])
    return (weight * torch.abs(pred - target)).sum() / denom


def budgeted_capacity_loss(
    *,
    accumulation: Tensor,
    support: Tensor,
    budget: float,
    temperature: float,
) -> Tensor:
    """Dense soft capacity budget on Gaussian accumulation."""

    support = support.detach().to(device=accumulation.device, dtype=accumulation.dtype).clamp(0.0, 1.0)
    temp = max(float(temperature), 1e-6)
    penalty = temp * F.softplus((accumulation - float(budget)) / temp)
    return (support * penalty).sum() / support.sum().clamp_min(1e-6)


def clear_proxy_chroma_loss(
    *,
    j_proxy: Tensor,
    medium_rgb: Tensor,
    support: Tensor,
    margin: float,
    detach_medium: bool = True,
) -> Tensor:
    """Suppress medium-direction clear-proxy chroma on supported water pixels."""

    support = support.detach().to(device=j_proxy.device, dtype=j_proxy.dtype).clamp(0.0, 1.0)
    medium = medium_rgb.detach() if detach_medium else medium_rgb
    j_chroma = _chroma(j_proxy)
    medium_chroma = _chroma(medium)
    medium_dir = medium_chroma / medium_chroma.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    projection = (j_chroma * medium_dir).sum(dim=-1, keepdim=True)
    penalty = F.relu(projection - float(margin))
    return (support * penalty).sum() / support.sum().clamp_min(1e-6)


def clear_proxy_luma_budget_loss(
    *,
    j_proxy: Tensor,
    support: Tensor,
    budget: float,
    temperature: float,
) -> Tensor:
    """Optional soft luma budget for residual clear-proxy brightness."""

    support = support.detach().to(device=j_proxy.device, dtype=j_proxy.dtype).clamp(0.0, 1.0)
    temp = max(float(temperature), 1e-6)
    penalty = temp * F.softplus((_luma(j_proxy) - float(budget)) / temp)
    return (support * penalty).sum() / support.sum().clamp_min(1e-6)


def support_coverage_stats(supports: MediumExplainabilitySupport) -> Dict[str, Tensor]:
    """Small scalar stats for training logs."""

    return {
        "flat_mean": supports.flat.mean(),
        "medium_mean": supports.medium.mean(),
        "far_mean": supports.far.mean(),
        "route_mean": supports.route.mean(),
        "capacity_mean": supports.capacity.mean(),
        "bootstrap_mean": supports.bootstrap.mean(),
        "medium_error_mean": supports.medium_error.mean(),
    }
