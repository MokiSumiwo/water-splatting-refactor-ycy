"""GIVAR appearance-consistency utilities.

GIVAR is intentionally appearance-only: all evidence maps are detached, and the
auxiliary color path exposes gradients only through gated Gaussian DC features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from water_splatting.cleanup import sample_pixel_map_at_gaussians


_C0 = 0.28209479177387814


@dataclass
class GIVAREvidence:
    """Detached per-Gaussian evidence used for cross-view appearance statistics."""

    weight: Tensor
    detail: Tensor
    reliability: Tensor
    texture: Tensor
    view_direction: Tensor


def _stats(values: Tensor) -> Dict[str, float]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0}
    return {
        "mean": float(flat.mean().item()),
        "p50": float(torch.quantile(flat, 0.50).item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
    }


def _nchw(image: Tensor) -> Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got {tuple(image.shape)}")
    return image.permute(2, 0, 1).unsqueeze(0)


def _sobel_xy(image: Tensor) -> Tuple[Tensor, Tensor]:
    x = _nchw(image.float())
    channels = int(x.shape[1])
    kx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    kx = kx.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    return (
        F.conv2d(x, kx, padding=1, groups=channels),
        F.conv2d(x, ky, padding=1, groups=channels),
    )


def _highpass(image: Tensor) -> Tensor:
    x = _nchw(image.float())
    blur = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
    return x - blur


def _hwc1(value: Tensor) -> Tensor:
    if value.ndim == 4:
        value = value[0].permute(1, 2, 0)
    if value.ndim == 3 and value.shape[-1] != 1:
        value = value.mean(dim=-1, keepdim=True)
    if value.ndim == 2:
        value = value[..., None]
    return value.contiguous()


def build_givar_detail_residual(pred_img: Tensor, gt_img: Tensor, *, highpass_weight: float) -> Tensor:
    """Build detached Sobel + HP5 residual map in HxWx1 format."""

    pred = pred_img.detach().float().clamp(0.0, 1.0)
    gt = gt_img.detach().float().clamp(0.0, 1.0)
    pred_sx, pred_sy = _sobel_xy(pred)
    gt_sx, gt_sy = _sobel_xy(gt)
    sobel = 0.5 * ((pred_sx - gt_sx).abs() + (pred_sy - gt_sy).abs()).mean(dim=1, keepdim=True)
    highpass = (_highpass(pred) - _highpass(gt)).abs().mean(dim=1, keepdim=True)
    return _hwc1(sobel + float(highpass_weight) * highpass).detach()


def _texture_map(gt_img: Tensor) -> Tensor:
    gt = gt_img.detach().float().clamp(0.0, 1.0)
    sx, sy = _sobel_xy(gt)
    texture = 0.5 * (sx.abs() + sy.abs()).mean(dim=1, keepdim=True)
    return _hwc1(texture).detach()


def build_givar_reliability_map(
    *,
    gt_img: Tensor,
    accumulation: Tensor,
    depth_std_relative: Tensor,
    texture_mid: float,
    texture_temp: float,
    accumulation_mid: float,
    accumulation_temp: float,
    depth_std_kappa: float,
    image_mask: Tensor | None = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Build detached object-safe pixel reliability from M1 internal evidence."""

    texture = _texture_map(gt_img).to(device=accumulation.device, dtype=accumulation.dtype)
    q_texture = torch.sigmoid((texture - float(texture_mid)) / max(float(texture_temp), 1e-6))
    q_acc = torch.sigmoid((accumulation.detach().float() - float(accumulation_mid)) / max(float(accumulation_temp), 1e-6))
    q_conc = torch.exp(-depth_std_relative.detach().float() / max(float(depth_std_kappa), 1e-6))
    reliability = (q_texture.float() * q_acc.float() * q_conc.float()).clamp(0.0, 1.0)
    if image_mask is not None:
        reliability = reliability * image_mask.detach().float().clamp(0.0, 1.0)
    return reliability.detach(), {
        "texture": texture.detach(),
        "q_texture": q_texture.detach(),
        "q_acc": q_acc.detach(),
        "q_conc": q_conc.detach(),
    }


def build_givar_gaussian_evidence(
    *,
    detail_map: Tensor,
    reliability_map: Tensor,
    texture_map: Tensor,
    xys: Tensor,
    radii: Tensor,
    means: Tensor,
    camera_position: Tensor,
) -> GIVAREvidence:
    """Sample detached GIVAR pixel evidence at projected Gaussian centers."""

    h, w = int(detail_map.shape[0]), int(detail_map.shape[1])
    sampled_detail = sample_pixel_map_at_gaussians(detail_map.detach(), xys.detach(), radii.detach(), h, w).float()
    sampled_rel = sample_pixel_map_at_gaussians(reliability_map.detach(), xys.detach(), radii.detach(), h, w).float()
    sampled_texture = sample_pixel_map_at_gaussians(texture_map.detach(), xys.detach(), radii.detach(), h, w).float()
    direction = means.detach().float() - camera_position.detach().float().reshape(1, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    weight = (sampled_detail.clamp_min(0.0) * sampled_rel.clamp(0.0, 1.0)).detach()
    return GIVAREvidence(
        weight=weight.reshape(-1).detach(),
        detail=sampled_detail.reshape(-1).detach(),
        reliability=sampled_rel.reshape(-1).detach(),
        texture=sampled_texture.reshape(-1).detach(),
        view_direction=direction.detach(),
    )


def build_givar_dc_aux_colors(*, full_rgb: Tensor, features_dc: Tensor, dc_gate: Tensor, sh_degree: int) -> Tensor:
    """Return full forward RGB while exposing only gated DC gradients."""

    if sh_degree > 0:
        dc_rgb = features_dc * _C0 + 0.5
    else:
        dc_rgb = torch.sigmoid(features_dc)
    gate = dc_gate.detach().to(device=full_rgb.device, dtype=full_rgb.dtype).reshape(-1, 1).clamp(0.0, 1.0)
    return full_rgb.detach() + gate * (dc_rgb - dc_rgb.detach())


def givar_highpass_charbonnier_loss(
    *,
    pred_img: Tensor,
    gt_img: Tensor,
    reliability_map: Tensor,
    epsilon: float,
) -> Tensor:
    """Weighted HP5 Charbonnier loss for the appearance-only auxiliary render."""

    pred_hp = _highpass(pred_img.float())
    gt_hp = _highpass(gt_img.detach().float())
    rho = torch.sqrt((pred_hp - gt_hp).square() + float(epsilon) ** 2).mean(dim=1, keepdim=True)
    rho_hwc = _hwc1(rho)
    weight = reliability_map.detach().float().clamp(0.0, 1.0)
    return (rho_hwc * weight).sum() / weight.sum().clamp_min(1e-6)


def compute_givar_dc_gate(
    *,
    view_count: Tensor,
    grad_direction_sum: Tensor,
    grad_weight_sum: Tensor,
    grad_magnitude_sum: Tensor,
    view_direction_sum: Tensor,
    min_view_count: int,
    coherence_threshold: float,
    min_view_spread: float,
    magnitude_quantile: float,
    dc_enabled: bool,
) -> Tuple[Tensor, Dict[str, Any]]:
    """Compute the detached DC consensus gate and compact diagnostics."""

    n = int(view_count.reshape(-1).numel())
    device = view_count.device
    if n == 0 or not dc_enabled:
        return torch.zeros(n, device=device), {"givar_dc_eligible_count": 0, "givar_dc_eligible_fraction": 0.0}
    vc = view_count.reshape(-1).float()
    gw = grad_weight_sum.reshape(-1).float().clamp_min(1e-6)
    gsum = grad_direction_sum.reshape(n, -1).float()
    coherence = (gsum.norm(dim=-1) / gw).clamp(0.0, 1.0)
    mean_mag = grad_magnitude_sum.reshape(-1).float() / gw
    view_mean = view_direction_sum.reshape(n, -1).float() / vc.clamp_min(1.0)[:, None]
    view_spread = (1.0 - view_mean.norm(dim=-1)).clamp(0.0, 1.0)

    view_ok = vc >= max(int(min_view_count), 1)
    coherent = coherence >= float(coherence_threshold)
    spread_ok = view_spread >= float(min_view_spread)
    pre_eligible = view_ok & coherent & spread_ok & torch.isfinite(mean_mag)
    if bool(pre_eligible.any().item()):
        q = min(max(float(magnitude_quantile), 0.0), 1.0)
        mag_threshold = torch.quantile(mean_mag[pre_eligible].float(), q)
        mag_ok = mean_mag >= mag_threshold
    else:
        mag_threshold = torch.tensor(float("inf"), device=device)
        mag_ok = torch.zeros_like(pre_eligible)
    eligible = pre_eligible & mag_ok
    gate = torch.where(eligible, coherence, torch.zeros_like(coherence)).detach()
    payload: Dict[str, Any] = {
        "givar_dc_supported_count": int((vc > 0).sum().item()),
        "givar_dc_supported_fraction": float((vc > 0).float().mean().item()) if n else 0.0,
        "givar_dc_min_view_count": int(view_ok.sum().item()),
        "givar_dc_coherent_count": int((view_ok & coherent).sum().item()),
        "givar_dc_spread_count": int((view_ok & coherent & spread_ok).sum().item()),
        "givar_dc_eligible_count": int(eligible.sum().item()),
        "givar_dc_eligible_fraction": float(eligible.float().mean().item()) if n else 0.0,
        "givar_dc_magnitude_threshold": float(mag_threshold.detach().item()) if torch.isfinite(mag_threshold) else 0.0,
        "givar_dc_coherence": _stats(coherence[vc > 0]),
        "givar_dc_view_count": _stats(vc[vc > 0]),
        "givar_dc_view_spread": _stats(view_spread[vc > 0]),
        "givar_dc_mean_grad_magnitude": _stats(mean_mag[vc > 0]),
        "givar_dc_gate": _stats(gate[gate > 0]),
    }
    return gate, payload


def pearson_corr(a: Tensor, b: Tensor) -> float:
    """Return finite Pearson correlation for two same-shaped tensors."""

    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    keep = torch.isfinite(x) & torch.isfinite(y)
    if int(keep.sum().item()) < 3:
        return 0.0
    x = x[keep]
    y = y[keep]
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    value = (x * y).sum() / denom
    if not math.isfinite(float(value.item())):
        return 0.0
    return float(value.item())
