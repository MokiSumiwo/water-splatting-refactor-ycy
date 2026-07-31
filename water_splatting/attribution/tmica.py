"""Tail-guided medium-invariant clear appearance helpers.

TMICA uses tail water only to estimate a detached water-color axis. The active
losses act on far-object clear appearance, preferably through the differentiable
clear-proxy path, and leave geometry / opacity routing to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .tacmd import TacmdTailEvidence


@dataclass(frozen=True)
class TmicaState:
    """Detached support state plus live water-axis projection."""

    tail_quality: Tensor
    tail_active: Tensor
    observed_anchor: Tensor
    anchor: Tensor
    anchor_active: Tensor
    water_axis: Tensor
    q_tail: Tensor
    q_object: Tensor
    q_far: Tensor
    q_near: Tensor
    q_low_transmission: Tensor
    q_sensitivity: Tensor
    support: Tensor
    near_support: Tensor
    b_j: Tensor
    b_near: Tensor
    depth_norm: Tensor
    metrics: Dict[str, Tensor]


def _luma(rgb: Tensor) -> Tensor:
    return 0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2] + 0.0722 * rgb[..., 2:3]


def _normalize_channels(rgb: Tensor, eps: float = 1e-8) -> Tensor:
    rgb = rgb.clamp_min(eps)
    return rgb / rgb.sum(dim=-1, keepdim=True).clamp_min(eps)


def _clr(rgb_or_prop: Tensor, eps: float = 1e-8) -> Tensor:
    prop = _normalize_channels(rgb_or_prop, eps=eps)
    logp = torch.log(prop.clamp_min(eps))
    return logp - logp.mean(dim=-1, keepdim=True)


def _normalize_depth(depth: Tensor) -> Tensor:
    if depth.ndim == 2:
        depth = depth[..., None]
    depth = depth.detach().float()
    valid = depth[torch.isfinite(depth) & (depth > 0.0)]
    if valid.numel() == 0:
        return torch.zeros_like(depth)
    scale = torch.quantile(valid, 0.95).clamp_min(1e-6)
    return (depth / scale).clamp(0.0, 1.0)


def _weighted_mean(values: Tensor, weight: Tensor, eps: float = 1e-8) -> Tensor:
    if weight.ndim == values.ndim - 1:
        weight = weight[..., None]
    return (values * weight).sum(dim=(0, 1)) / weight.sum().clamp_min(eps)


def _weighted_scalar_mean(values: Tensor, weight: Tensor, eps: float = 1e-8) -> Tensor:
    if values.ndim == 2:
        values = values[..., None]
    if weight.ndim == 2:
        weight = weight[..., None]
    return (values * weight).sum() / weight.sum().clamp_min(eps)


def _robust_weighted_scalar_mean(values: Tensor, weight: Tensor, *, detach_result: bool) -> Tensor:
    if values.ndim == 2:
        values = values[..., None]
    if weight.ndim == 2:
        weight = weight[..., None]
    weight = weight.detach().to(device=values.device, dtype=values.dtype).clamp(0.0, 1.0)
    if float(weight.sum().detach().cpu().item()) <= 1e-8:
        out = values.sum() * 0.0
        return out.detach() if detach_result else out

    with torch.no_grad():
        selected = values.detach().reshape(-1)
        selected_weight = weight.reshape(-1)
        valid = selected_weight > 1e-4
        if int(valid.sum().item()) >= 32:
            vals = selected[valid]
            lo = torch.quantile(vals, 0.10)
            hi = torch.quantile(vals, 0.90)
            trim = ((selected >= lo) & (selected <= hi)).reshape_as(weight).to(weight.dtype)
            robust_weight = weight * trim
            if float(robust_weight.sum().item()) > 1e-8:
                weight = robust_weight
    out = _weighted_scalar_mean(values, weight)
    return out.detach() if detach_result else out


def _weighted_variance(values: Tensor, weight: Tensor, eps: float = 1e-8) -> Tensor:
    mean = _weighted_mean(values, weight, eps=eps).view(1, 1, -1)
    var = (((values - mean) ** 2) * weight).sum() / weight.sum().clamp_min(eps)
    return var.detach()


def _border_band_like(weight: Tensor, border: int) -> Tensor:
    border = max(int(border), 1)
    band = torch.zeros_like(weight)
    band[:border, :, :] = 1.0
    band[-border:, :, :] = 1.0
    band[:, :border, :] = 1.0
    band[:, -border:, :] = 1.0
    return band


def _chroma_distance_prop(a: Tensor, b: Tensor) -> Tensor:
    return (_clr(a.view(1, 1, 3)) - _clr(b.view(1, 1, 3))).abs().sum()


def compute_tmica_tail_quality(
    *,
    gt_img: Tensor,
    tail: TacmdTailEvidence,
    scene_anchor: Tensor,
    scene_anchor_weight: Tensor,
    coverage_mid: float,
    coverage_temp: float,
    variance_tau: float,
    border_width: int,
    border_mid: float,
    border_temp: float,
    ema_tau: float,
) -> Dict[str, Tensor]:
    """Return detached quality gates for the current view tail anchor."""

    gt = gt_img.detach().float().clamp(0.0, 1.0)
    q = tail.q_infty.detach().to(device=gt.device, dtype=gt.dtype).clamp(0.0, 1.0)
    support_mean = q.mean()
    q_cov = torch.sigmoid((support_mean - float(coverage_mid)) / max(float(coverage_temp), 1e-6))

    if float(q.sum().detach().cpu().item()) <= 1e-8:
        q_var = support_mean.new_tensor(0.0)
    else:
        q_var = torch.exp(-_weighted_variance(_clr(gt), q) / max(float(variance_tau), 1e-6)).clamp(0.0, 1.0)

    border = _border_band_like(q, border_width)
    border_mean = (q * border).sum() / border.sum().clamp_min(1e-6)
    q_border = torch.sigmoid((border_mean - float(border_mid)) / max(float(border_temp), 1e-6))

    scene_weight = scene_anchor_weight.detach().reshape(()).to(device=gt.device, dtype=gt.dtype).clamp(0.0, 1.0)
    scene_valid = (scene_weight > 1e-4).to(dtype=gt.dtype)
    dist = _chroma_distance_prop(tail.observed_anchor.to(gt), scene_anchor.detach().to(gt))
    q_ema = scene_valid * torch.exp(-dist / max(float(ema_tau), 1e-6)).clamp(0.0, 1.0) + (1.0 - scene_valid)

    quality = (q_cov * q_var * q_border * q_ema).clamp(0.0, 1.0).detach()
    return {
        "quality": quality,
        "coverage_gate": q_cov.detach(),
        "variance_gate": q_var.detach(),
        "border_gate": q_border.detach(),
        "ema_gate": q_ema.detach(),
        "border_mean": border_mean.detach(),
        "support_mean": support_mean.detach(),
        "chroma_variance": _weighted_variance(_clr(gt), q) if float(q.sum().detach().cpu().item()) > 1e-8 else support_mean.new_tensor(0.0),
        "ema_chroma_distance": dist.detach(),
    }


def _combine_axis_anchor(
    *,
    observed_anchor: Tensor,
    scene_anchor: Tensor,
    scene_anchor_weight: Tensor,
    quality: Tensor,
    quality_threshold: float,
    scene_fallback: float,
) -> tuple[Tensor, Tensor, Tensor]:
    q = quality.detach().reshape(()).to(dtype=observed_anchor.dtype, device=observed_anchor.device)
    use_observed = (q > float(quality_threshold)).to(dtype=observed_anchor.dtype)
    scene_weight = scene_anchor_weight.detach().reshape(()).to(dtype=observed_anchor.dtype, device=observed_anchor.device)
    use_scene = (1.0 - use_observed) * (scene_weight > 1e-4).to(dtype=observed_anchor.dtype)
    fallback_weight = float(scene_fallback) * scene_weight * use_scene
    active = (q * use_observed + fallback_weight).clamp(0.0, 1.0)
    anchor = q * use_observed * observed_anchor.detach() + fallback_weight * scene_anchor.detach()
    if float(active.detach().cpu().item()) <= 1e-8:
        anchor = observed_anchor.new_tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    else:
        anchor = _normalize_channels(anchor.view(1, 3)).reshape(3)
    tail_active = (q > float(quality_threshold)).to(dtype=observed_anchor.dtype)
    return anchor.detach(), active.detach(), tail_active.detach()


def _normalized_sensitivity(
    *,
    medium_rgb: Tensor,
    medium_bs: Tensor,
    depth: Tensor,
) -> Tensor:
    d = depth.detach().float()
    if d.ndim == 2:
        d = d[..., None]
    bs = medium_bs.detach().float().clamp_min(0.0)
    rgb = medium_rgb.detach().float().abs()
    raw = (rgb * d * torch.exp(-(bs * d).clamp_min(0.0))).sum(dim=-1, keepdim=True)
    valid = raw[torch.isfinite(raw)]
    if valid.numel() == 0:
        return torch.zeros_like(raw)
    scale = torch.quantile(valid, 0.95).clamp_min(1e-6)
    return (raw / scale).clamp(0.0, 1.0).detach()


def build_tmica_state(
    *,
    gt_img: Tensor,
    j_clear: Tensor,
    tail: TacmdTailEvidence,
    scene_anchor: Tensor,
    scene_anchor_weight: Tensor,
    accumulation: Tensor,
    depth: Tensor,
    depth_std_relative: Tensor,
    medium_attn: Tensor,
    medium_bs: Tensor,
    medium_rgb: Tensor,
    image_mask: Optional[Tensor],
    quality_threshold: float,
    scene_fallback: float,
    coverage_mid: float,
    coverage_temp: float,
    variance_tau: float,
    border_width: int,
    border_mid: float,
    border_temp: float,
    ema_tau: float,
    object_accum_mid: float,
    object_accum_temp: float,
    object_concentration_kappa: float,
    far_depth_mid: float,
    far_depth_temp: float,
    near_depth_mid: float,
    near_depth_temp: float,
    use_low_transmission: bool,
    use_sensitivity: bool,
) -> TmicaState:
    """Build TMICA supports and water-axis projections."""

    target = gt_img.detach().float()
    quality = compute_tmica_tail_quality(
        gt_img=target,
        tail=tail,
        scene_anchor=scene_anchor.detach().to(target),
        scene_anchor_weight=scene_anchor_weight.detach().to(target),
        coverage_mid=coverage_mid,
        coverage_temp=coverage_temp,
        variance_tau=variance_tau,
        border_width=border_width,
        border_mid=border_mid,
        border_temp=border_temp,
        ema_tau=ema_tau,
    )
    observed = tail.observed_anchor.detach().to(target)
    anchor, anchor_active, tail_active = _combine_axis_anchor(
        observed_anchor=observed,
        scene_anchor=scene_anchor.detach().to(target),
        scene_anchor_weight=scene_anchor_weight.detach().to(target),
        quality=quality["quality"],
        quality_threshold=quality_threshold,
        scene_fallback=scene_fallback,
    )
    axis = _clr(anchor.view(1, 1, 3)).reshape(3)
    axis = axis / axis.norm().clamp_min(1e-8)
    axis = axis.detach()

    accum = accumulation.detach().float()
    if accum.ndim == 2:
        accum = accum[..., None]
    q_acc = torch.sigmoid((accum - float(object_accum_mid)) / max(float(object_accum_temp), 1e-6)).clamp(0.0, 1.0)
    q_conc = torch.exp(
        -depth_std_relative.detach().float() / max(float(object_concentration_kappa), 1e-6)
    ).clamp(0.0, 1.0)
    q_tail = tail.q_infty.detach().to(target).clamp(0.0, 1.0)
    q_obj = (q_acc * q_conc * (1.0 - q_tail)).clamp(0.0, 1.0).detach()

    depth_norm = _normalize_depth(depth)
    q_far = torch.sigmoid((depth_norm - float(far_depth_mid)) / max(float(far_depth_temp), 1e-6)).clamp(0.0, 1.0)
    q_near = torch.sigmoid((float(near_depth_mid) - depth_norm) / max(float(near_depth_temp), 1e-6)).clamp(0.0, 1.0)
    if use_low_transmission:
        mean_attn = medium_attn.detach().float().mean(dim=-1, keepdim=True).clamp_min(0.0)
        d = depth.detach().float()
        if d.ndim == 2:
            d = d[..., None]
        q_low_t = (1.0 - torch.exp(-(mean_attn * d).clamp_min(0.0))).clamp(0.0, 1.0)
    else:
        q_low_t = torch.ones_like(q_obj)
    q_sens = (
        _normalized_sensitivity(medium_rgb=medium_rgb, medium_bs=medium_bs, depth=depth)
        if use_sensitivity
        else torch.ones_like(q_obj)
    )

    support = (anchor_active * q_obj * q_far * q_low_t * q_sens).clamp(0.0, 1.0).detach()
    near_support = (anchor_active * q_obj * q_near).clamp(0.0, 1.0).detach()
    if image_mask is not None:
        mask = image_mask.detach().to(target).clamp(0.0, 1.0)
        support = support * mask
        near_support = near_support * mask

    z_j = _clr(j_clear.clamp_min(1e-6))
    b_j = (z_j * axis.view(1, 1, 3).to(z_j)).sum(dim=-1, keepdim=True)
    b_near = _robust_weighted_scalar_mean(b_j, near_support, detach_result=True)

    far_mean = _robust_weighted_scalar_mean(b_j.detach(), support, detach_result=True)
    far_near_gap = (far_mean - b_near).abs().detach()
    overcorr = (b_j.detach() < (b_near - 0.15)).to(dtype=target.dtype)
    overcorr_rate = (support * overcorr).sum() / support.sum().clamp_min(1e-6)
    corr = _weighted_depth_corr(b_j.detach(), depth_norm, (support + near_support).clamp(0.0, 1.0))
    metrics = {
        **quality,
        "anchor_active": anchor_active.detach(),
        "tail_active": tail_active.detach(),
        "axis_norm": axis.norm().detach(),
        "object_support_mean": q_obj.mean().detach(),
        "support_mean": support.mean().detach(),
        "support_sum": support.sum().detach(),
        "near_support_mean": near_support.mean().detach(),
        "b_near": b_near.detach(),
        "b_far": far_mean.detach(),
        "far_near_gap": far_near_gap.detach(),
        "overcorrection_rate": overcorr_rate.detach(),
        "water_depth_corr": corr.detach(),
    }
    for i in range(3):
        metrics[f"water_axis_{i}"] = axis[i].detach()
        metrics[f"anchor_{i}"] = anchor[i].detach()
    return TmicaState(
        tail_quality=quality["quality"],
        tail_active=tail_active,
        observed_anchor=observed,
        anchor=anchor,
        anchor_active=anchor_active,
        water_axis=axis,
        q_tail=q_tail,
        q_object=q_obj,
        q_far=q_far.detach(),
        q_near=q_near.detach(),
        q_low_transmission=q_low_t.detach(),
        q_sensitivity=q_sens.detach(),
        support=support,
        near_support=near_support,
        b_j=b_j,
        b_near=b_near,
        depth_norm=depth_norm.detach(),
        metrics=metrics,
    )


def _weighted_depth_corr(values: Tensor, depth: Tensor, weight: Tensor) -> Tensor:
    if values.ndim == 2:
        values = values[..., None]
    if depth.ndim == 2:
        depth = depth[..., None]
    if weight.ndim == 2:
        weight = weight[..., None]
    weight = weight.detach().to(values).clamp(0.0, 1.0)
    if float(weight.sum().detach().cpu().item()) <= 1e-8:
        return values.sum().detach() * 0.0
    wsum = weight.sum().clamp_min(1e-6)
    vx = values.detach() - (values.detach() * weight).sum() / wsum
    dx = depth.detach() - (depth.detach() * weight).sum() / wsum
    cov = (weight * vx * dx).sum() / wsum
    var_v = (weight * vx * vx).sum() / wsum
    var_d = (weight * dx * dx).sum() / wsum
    return (cov / torch.sqrt(var_v * var_d).clamp_min(1e-8)).detach()


def register_tmica_axis_gradient_hook(j_clear: Tensor, water_axis: Tensor) -> None:
    """Project clear-appearance gradients to the detached water-color axis."""

    if not j_clear.requires_grad:
        return
    axis = water_axis.detach().to(device=j_clear.device, dtype=j_clear.dtype).view(1, 1, 3)

    def _hook(grad: Tensor, axis: Tensor = axis) -> Tensor:
        projected = (grad * axis).sum(dim=-1, keepdim=True) * axis
        return projected - projected.mean(dim=-1, keepdim=True)

    j_clear.register_hook(_hook)


def tmica_axis_losses(
    *,
    state: TmicaState,
    positive_margin: float,
    negative_margin: float,
    trend_margin_step: float,
) -> Dict[str, Tensor]:
    """Far-axis, overcorrection, and depth-trend losses."""

    support = state.support.detach().to(state.b_j).clamp(0.0, 1.0)
    if float(support.sum().detach().cpu().item()) <= 1e-8 or float(state.anchor_active.detach().cpu().item()) <= 1e-8:
        zero = state.b_j.sum() * 0.0
        return {"far_axis": zero, "overcorrection": zero, "trend": zero}

    excess = F.relu(state.b_j - state.b_near - float(positive_margin))
    far_axis = (support * F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction="none")).sum() / support.sum().clamp_min(1e-6)

    over = F.relu(state.b_near - float(negative_margin) - state.b_j)
    overcorr = (support * F.smooth_l1_loss(over, torch.zeros_like(over), reduction="none")).sum() / support.sum().clamp_min(1e-6)

    trend = _depth_trend_loss(
        b_j=state.b_j,
        q_object=state.q_object,
        depth_norm=state.depth_norm,
        anchor_active=state.anchor_active,
        trend_margin_step=trend_margin_step,
    )
    return {"far_axis": far_axis, "overcorrection": overcorr, "trend": trend}


def _depth_trend_loss(
    *,
    b_j: Tensor,
    q_object: Tensor,
    depth_norm: Tensor,
    anchor_active: Tensor,
    trend_margin_step: float,
) -> Tensor:
    obj = q_object.detach().to(b_j).clamp(0.0, 1.0)
    if float(obj.sum().detach().cpu().item()) <= 1e-8 or float(anchor_active.detach().cpu().item()) <= 1e-8:
        return b_j.sum() * 0.0
    valid = obj.reshape(-1) > 1e-4
    if int(valid.sum().detach().cpu().item()) < 64:
        return b_j.sum() * 0.0
    d_flat = depth_norm.detach().reshape(-1)
    valid_depth = d_flat[valid]
    q25, q50, q75 = torch.quantile(valid_depth, torch.tensor([0.25, 0.50, 0.75], device=valid_depth.device))
    bins = [
        depth_norm <= q25,
        (depth_norm > q25) & (depth_norm <= q50),
        (depth_norm > q50) & (depth_norm <= q75),
        depth_norm > q75,
    ]
    means = []
    for mask in bins:
        w = obj * mask.to(obj.dtype)
        if float(w.sum().detach().cpu().item()) <= 1e-8:
            means.append(None)
        else:
            means.append(_robust_weighted_scalar_mean(b_j, w, detach_result=False))
    if means[0] is None:
        return b_j.sum() * 0.0
    near = means[0].detach()
    losses = []
    for idx, mean in enumerate(means[1:], start=1):
        if mean is None:
            continue
        margin = float(trend_margin_step) * idx
        losses.append(F.relu(mean - near - margin))
    if not losses:
        return b_j.sum() * 0.0
    return torch.stack(losses).mean()


def tmica_tail_lite_loss(
    *,
    medium_rgb: Tensor,
    q_tail: Tensor,
    target_anchor: Tensor,
    tail_active: Tensor,
    tolerance: float,
) -> Tensor:
    """Strictly gated tail-mean-only chroma calibration for A/B_inf."""

    q = q_tail.detach().to(medium_rgb).clamp(0.0, 1.0)
    active = tail_active.detach().reshape(()).to(medium_rgb)
    if float(q.sum().detach().cpu().item()) <= 1e-8 or float(active.detach().cpu().item()) <= 1e-8:
        return medium_rgb.sum() * 0.0
    mean_rgb = _normalize_channels(_weighted_mean(medium_rgb, q).view(1, 3)).reshape(3)
    target = target_anchor.detach().to(medium_rgb)
    dist = (_clr(mean_rgb.view(1, 1, 3)) - _clr(target.view(1, 1, 3))).abs().sum()
    return active * F.smooth_l1_loss(
        F.relu(dist - float(tolerance)),
        dist.new_zeros(()),
        reduction="mean",
    )

