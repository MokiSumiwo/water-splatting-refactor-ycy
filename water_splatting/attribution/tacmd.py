"""Tail-anchored counterfactual medium disentanglement helpers.

The routines here are deliberately stateless.  The model owns the scene-level
EMA anchor because that state has to live in checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class TacmdTailEvidence:
    """Detached tail-water evidence and per-view color anchor candidate."""

    q_infty: Tensor
    q_transmission: Tensor
    q_accumulation: Tensor
    q_depth: Tensor
    q_gradient: Tensor
    depth_norm: Tensor
    confidence: Tensor
    observed_anchor: Tensor
    support_mean: Tensor
    support_sum: Tensor


@dataclass(frozen=True)
class TacmdBsState:
    """Backscatter spectrum decomposition used by TACMD."""

    strength: Tensor
    spectrum: Tensor
    center: Tensor
    z: Tensor
    center_z: Tensor
    radius: Tensor
    depth_gate: Tensor
    q_medium: Tensor
    weight: Tensor
    deviation_l1: Tensor
    over_band: Tensor


def _to_nchw(image: Tensor) -> Tensor:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected [H, W, 3] image, got {tuple(image.shape)}")
    return image.permute(2, 0, 1).unsqueeze(0)


def _from_nchw(image: Tensor) -> Tensor:
    return image.squeeze(0).permute(1, 2, 0)


def _luma(rgb: Tensor) -> Tensor:
    return 0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2] + 0.0722 * rgb[..., 2:3]


def _chroma(rgb: Tensor) -> Tensor:
    return rgb - rgb.mean(dim=-1, keepdim=True)


def _normalize_channels(rgb: Tensor, eps: float = 1e-8) -> Tensor:
    rgb = rgb.clamp_min(0.0)
    return rgb / rgb.sum(dim=-1, keepdim=True).clamp_min(eps)


def _clr(proportion: Tensor, eps: float = 1e-8) -> Tensor:
    logp = torch.log(proportion.clamp_min(eps))
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


def _image_gradient(gt_img: Tensor) -> Tensor:
    image = gt_img.detach().float().clamp(0.0, 1.0)
    grad_x = torch.zeros_like(image[..., 0:1])
    grad_y = torch.zeros_like(image[..., 0:1])
    grad_x[:, 1:, :] = (image[:, 1:, :] - image[:, :-1, :]).abs().sum(dim=-1, keepdim=True)
    grad_y[1:, :, :] = (image[1:, :, :] - image[:-1, :, :]).abs().sum(dim=-1, keepdim=True)
    return grad_x + grad_y


def _weighted_mean_rgb(rgb: Tensor, weight: Tensor, eps: float = 1e-8) -> Tensor:
    if weight.ndim == 2:
        weight = weight[..., None]
    return (rgb * weight).sum(dim=(0, 1)) / weight.sum().clamp_min(eps)


def _weighted_chroma_distance(rgb_or_prop: Tensor, target_prop: Tensor) -> Tensor:
    prop = _normalize_channels(rgb_or_prop)
    target = target_prop.detach().view(1, 1, 3).to(device=prop.device, dtype=prop.dtype)
    return (prop - target).abs().sum(dim=-1, keepdim=True)


def compute_tail_evidence(
    *,
    gt_img: Tensor,
    final_transmittance: Tensor,
    accumulation: Tensor,
    depth: Tensor,
    transmission_mid: float,
    transmission_temp: float,
    accumulation_mid: float,
    accumulation_temp: float,
    depth_mid: float,
    depth_temp: float,
    gradient_scale: float,
    confidence_low: float,
    confidence_high: float,
) -> TacmdTailEvidence:
    """Estimate detached infinite-water evidence and observed tail color."""

    gt = gt_img.detach().float().clamp(0.0, 1.0)
    final_t = final_transmittance.detach().float()
    accum = accumulation.detach().float()
    if final_t.ndim == 2:
        final_t = final_t[..., None]
    if accum.ndim == 2:
        accum = accum[..., None]

    depth_norm = _normalize_depth(depth)
    q_t = torch.sigmoid((final_t - float(transmission_mid)) / max(float(transmission_temp), 1e-6))
    q_a = torch.sigmoid((float(accumulation_mid) - accum) / max(float(accumulation_temp), 1e-6))
    q_d = torch.sigmoid((depth_norm - float(depth_mid)) / max(float(depth_temp), 1e-6))
    q_g = torch.exp(-_image_gradient(gt) / max(float(gradient_scale), 1e-6)).clamp(0.0, 1.0)
    q = (q_t * q_a * q_d * q_g).clamp(0.0, 1.0).detach()

    support_mean = q.mean()
    denom = max(float(confidence_high) - float(confidence_low), 1e-6)
    confidence = ((support_mean - float(confidence_low)) / denom).clamp(0.0, 1.0).detach()

    with torch.no_grad():
        robust_weight = q.clone()
        valid = robust_weight.squeeze(-1) > 1e-4
        if bool(valid.any()):
            luma = _luma(gt).squeeze(-1)
            selected = luma[valid]
            if selected.numel() >= 32:
                lo = torch.quantile(selected, 0.10)
                hi = torch.quantile(selected, 0.90)
                trim = ((luma >= lo) & (luma <= hi)).to(dtype=robust_weight.dtype)[..., None]
                robust_weight = robust_weight * trim
        if float(robust_weight.sum().item()) <= 1e-8:
            observed = gt.new_tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        else:
            observed = _normalize_channels(_weighted_mean_rgb(gt, robust_weight).view(1, 3)).reshape(3)

    return TacmdTailEvidence(
        q_infty=q,
        q_transmission=q_t.detach(),
        q_accumulation=q_a.detach(),
        q_depth=q_d.detach(),
        q_gradient=q_g.detach(),
        depth_norm=depth_norm.detach(),
        confidence=confidence,
        observed_anchor=observed.detach(),
        support_mean=support_mean.detach(),
        support_sum=q.sum().detach(),
    )


def combine_tail_anchor(
    *,
    observed_anchor: Tensor,
    scene_anchor: Tensor,
    scene_anchor_weight: Tensor,
    confidence: Tensor,
    fallback: float,
) -> tuple[Tensor, Tensor]:
    """Blend current tail observation with scene EMA anchor."""

    scene_valid = (scene_anchor_weight.detach().reshape(()) > 1e-4).to(dtype=observed_anchor.dtype)
    conf = confidence.detach().reshape(()).to(dtype=observed_anchor.dtype)
    fallback_weight = float(fallback) * (1.0 - conf) * scene_valid
    anchor = conf * observed_anchor.detach() + fallback_weight * scene_anchor.detach()
    active = (conf + fallback_weight).clamp(0.0, 1.0)
    if float(active.detach().cpu().item()) <= 1e-8:
        anchor = observed_anchor.new_tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    else:
        anchor = _normalize_channels(anchor.view(1, 3)).reshape(3)
    return anchor.detach(), active.detach()


def tail_anchor_losses(
    *,
    medium_rgb: Tensor,
    q_infty: Tensor,
    target_anchor: Tensor,
    confidence: Tensor,
    tolerance: float,
) -> Dict[str, Tensor]:
    """Tail mean and local tolerance-band losses for A/B_inf."""

    q = q_infty.detach().to(device=medium_rgb.device, dtype=medium_rgb.dtype).clamp(0.0, 1.0)
    conf = confidence.detach().to(device=medium_rgb.device, dtype=medium_rgb.dtype)
    target = target_anchor.detach().to(device=medium_rgb.device, dtype=medium_rgb.dtype)
    if float(q.sum().detach().cpu().item()) <= 1e-8 or float(conf.detach().cpu().item()) <= 1e-8:
        zero = medium_rgb.sum() * 0.0
        return {"mean": zero, "band": zero}

    mean_a = _normalize_channels(_weighted_mean_rgb(medium_rgb, q).view(1, 3)).reshape(3)
    mean_loss = F.smooth_l1_loss(mean_a, target, reduction="mean") * conf

    dist = _weighted_chroma_distance(medium_rgb, target)
    band = F.relu(dist - float(tolerance))
    band_loss = conf * (q * band).sum() / q.sum().clamp_min(1e-6)
    return {"mean": mean_loss, "band": band_loss}


def build_bs_state(
    *,
    medium_bs: Tensor,
    rgb_medium_total: Tensor,
    pred_image: Tensor,
    depth: Tensor,
    radius_near: float,
    radius_far: float,
    depth_mid: float,
    depth_temp: float,
) -> TacmdBsState:
    """Build backscatter spectrum state while preserving total strength."""

    bs = medium_bs.clamp_min(0.0)
    strength = bs.mean(dim=-1, keepdim=True)
    spectrum = _normalize_channels(bs)

    with torch.no_grad():
        q_medium = (
            rgb_medium_total.detach().abs().sum(dim=-1, keepdim=True)
            / pred_image.detach().abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        weight = (0.25 + 0.75 * q_medium).clamp(0.0, 1.0)
        center = _normalize_channels(_weighted_mean_rgb(spectrum.detach(), weight).view(1, 3)).reshape(3)

    depth_norm = _normalize_depth(depth)
    depth_gate = torch.sigmoid((depth_norm - float(depth_mid)) / max(float(depth_temp), 1e-6)).clamp(0.0, 1.0)
    radius = float(radius_near) - depth_gate * (float(radius_near) - float(radius_far))
    z = _clr(spectrum)
    center_z = _clr(center.view(1, 1, 3)).reshape(3).detach()
    deviation = (z - center_z.view(1, 1, 3)).abs().sum(dim=-1, keepdim=True)
    over = (deviation > radius.detach()).to(dtype=medium_bs.dtype)
    return TacmdBsState(
        strength=strength,
        spectrum=spectrum,
        center=center.detach(),
        z=z,
        center_z=center_z,
        radius=radius.detach(),
        depth_gate=depth_gate.detach(),
        q_medium=q_medium.detach(),
        weight=weight.detach(),
        deviation_l1=deviation,
        over_band=over.detach(),
    )


def bs_band_loss(bs_state: TacmdBsState) -> Tensor:
    """Depth-adaptive log-ratio band loss for BS spectrum."""

    excess = F.relu(bs_state.deviation_l1 - bs_state.radius)
    return (bs_state.weight * excess).sum() / bs_state.weight.sum().clamp_min(1e-6)


def bs_convergence_losses(
    *,
    medium_rgb: Tensor,
    medium_bs: Tensor,
    depth: Tensor,
    q_infty: Tensor,
    target_anchor: Tensor,
    confidence: Tensor,
) -> Dict[str, Tensor]:
    """Force finite BS chroma to move toward the tail anchor with depth."""

    q = q_infty.detach().to(device=medium_bs.device, dtype=medium_bs.dtype).clamp(0.0, 1.0)
    conf = confidence.detach().to(device=medium_bs.device, dtype=medium_bs.dtype)
    target = target_anchor.detach().to(device=medium_bs.device, dtype=medium_bs.dtype)
    if float(q.sum().detach().cpu().item()) <= 1e-8 or float(conf.detach().cpu().item()) <= 1e-8:
        zero = medium_bs.sum() * 0.0
        return {"monotonic": zero, "terminal": zero}

    d = depth.detach().float()
    if d.ndim == 2:
        d = d[..., None]
    d = d.clamp_min(0.0)
    a = medium_rgb.detach().float().clamp_min(0.0)
    bs = medium_bs.clamp_min(0.0)

    b1 = a * (1.0 - torch.exp(-bs * (0.75 * d)))
    b2 = a * (1.0 - torch.exp(-bs * (1.25 * d)))
    bd = a * (1.0 - torch.exp(-bs * d))
    d1 = _weighted_chroma_distance(b1, target)
    d2 = _weighted_chroma_distance(b2, target)
    dd = _weighted_chroma_distance(bd, target)
    mono = conf * (q * F.relu(d2 - d1)).sum() / q.sum().clamp_min(1e-6)
    terminal = conf * (q * dd).sum() / q.sum().clamp_min(1e-6)
    return {"monotonic": mono, "terminal": terminal}


def build_counterfactual_bs(
    *,
    bs_state: TacmdBsState,
    projection_max: float,
) -> Tensor:
    """Project BS spectrum into the tolerance band while keeping strength."""

    radius = bs_state.radius.clamp_min(1e-6)
    center_z = bs_state.center_z.view(1, 1, 3).to(device=bs_state.z.device, dtype=bs_state.z.dtype)
    z_proj = center_z + radius * torch.tanh((bs_state.z - center_z) / radius)
    c_proj = torch.softmax(z_proj, dim=-1)
    lam = float(projection_max) * bs_state.q_medium * (0.25 + 0.75 * bs_state.depth_gate)
    c_cf = _normalize_channels((1.0 - lam) * bs_state.spectrum + lam * c_proj)
    return (3.0 * bs_state.strength * c_cf).clamp_min(0.0)


def _blur_rgb(rgb: Tensor, kernel: int) -> Tensor:
    k = max(int(kernel), 1)
    if k <= 1:
        return rgb
    if k % 2 == 0:
        k += 1
    return _from_nchw(F.avg_pool2d(_to_nchw(rgb), kernel_size=k, stride=1, padding=k // 2))


def counterfactual_chroma_loss(
    *,
    cf_rgb: Tensor,
    gt_img: Tensor,
    main_rgb: Tensor,
    bs_state: TacmdBsState,
    blur_kernel: int,
    rgb_trust_region: float,
    luma_ratio: float,
) -> Dict[str, Tensor]:
    """Low-frequency chroma/luma loss for the counterfactual branch."""

    gt = gt_img.detach().float().clamp(0.0, 1.0)
    weight = (bs_state.q_medium * (0.25 + 0.75 * bs_state.depth_gate)).detach().to(
        device=cf_rgb.device,
        dtype=cf_rgb.dtype,
    )
    if float(weight.sum().detach().cpu().item()) <= 1e-8:
        zero = cf_rgb.sum() * 0.0
        return {"loss": zero, "chroma": zero, "luma": zero, "safe_gate": zero.detach(), "rgb_delta": zero.detach()}

    cf_chroma = _blur_rgb(_chroma(cf_rgb), blur_kernel)
    gt_chroma = _blur_rgb(_chroma(gt), blur_kernel)
    chroma = (weight * (cf_chroma - gt_chroma).abs().sum(dim=-1, keepdim=True)).sum() / weight.sum().clamp_min(1e-6)
    luma = (weight * (_luma(cf_rgb) - _luma(gt)).abs()).sum() / weight.sum().clamp_min(1e-6)
    rgb_delta = (weight * (cf_rgb.detach() - main_rgb.detach()).abs().sum(dim=-1, keepdim=True)).sum() / (
        weight.sum().clamp_min(1e-6) * 3.0
    )
    safe_gate = (float(rgb_trust_region) / rgb_delta.clamp_min(1e-6)).clamp(0.0, 1.0).detach()
    loss = safe_gate * (chroma + float(luma_ratio) * luma)
    return {
        "loss": loss,
        "chroma": chroma.detach(),
        "luma": luma.detach(),
        "safe_gate": safe_gate,
        "rgb_delta": rgb_delta.detach(),
    }


def bs_state_stats(bs_state: TacmdBsState) -> Dict[str, Tensor]:
    """Compact scalar diagnostics for train logs."""

    strength = bs_state.strength.detach()
    spectrum = bs_state.spectrum.detach()
    return {
        "bs_strength_mean": strength.mean(),
        "bs_strength_std": strength.std(unbiased=False),
        "bs_spectrum_std": spectrum.std(unbiased=False),
        "bs_over_band_fraction": bs_state.over_band.float().mean(),
        "bs_deviation_l1_mean": bs_state.deviation_l1.detach().mean(),
        "bs_radius_mean": bs_state.radius.detach().mean(),
        "q_medium_mean": bs_state.q_medium.detach().mean(),
        "cf_weight_mean": (bs_state.q_medium * (0.25 + 0.75 * bs_state.depth_gate)).detach().mean(),
    }
