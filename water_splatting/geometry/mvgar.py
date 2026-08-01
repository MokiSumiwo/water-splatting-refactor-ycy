"""MV-GAR geometry evidence and candidate selection utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from water_splatting.cleanup import sample_pixel_map_at_gaussians


@dataclass
class MVGAREvidence:
    """Per-Gaussian detached evidence for MV-GAR."""

    weight: torch.Tensor
    detail: torch.Tensor
    log_depth_error: torch.Tensor
    depth_target: torch.Tensor
    sampled_confidence: torch.Tensor
    sampled_structure: torch.Tensor
    sampled_render_reliability: torch.Tensor
    front_gate: torch.Tensor


def _as_hwc1(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float()
    if value.ndim == 2:
        value = value[..., None]
    if value.ndim == 3 and value.shape[0] == 1 and value.shape[-1] != 1:
        value = value.permute(1, 2, 0)
    if value.ndim != 3:
        raise ValueError(f"Expected HxW or HxWxC tensor, got shape {tuple(value.shape)}")
    if value.shape[-1] != 1:
        value = value[..., :1]
    return value.contiguous()


def _resize_hwc(value: torch.Tensor, height: int, width: int, *, mode: str = "bilinear") -> torch.Tensor:
    value = _as_hwc1(value)
    if int(value.shape[0]) == int(height) and int(value.shape[1]) == int(width):
        return value
    chw = value.permute(2, 0, 1)[None]
    if mode == "nearest":
        out = F.interpolate(chw, size=(height, width), mode=mode)
    else:
        out = F.interpolate(chw, size=(height, width), mode=mode, align_corners=False)
    return out[0].permute(1, 2, 0).contiguous()


def load_mvgar_view_payload(
    pseudo_depth_dir: str,
    image_idx: int,
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    cache: Optional[Dict[Tuple[str, int, int, int], Dict[str, torch.Tensor]]] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Load and resize a ``view_XXXX_mvgar.pt`` payload."""

    path = Path(pseudo_depth_dir) / f"view_{int(image_idx):04d}_mvgar.pt"
    key = (str(path), int(image_idx), int(height), int(width))
    if cache is not None and key in cache:
        return {name: value.to(device=device, dtype=dtype) for name, value in cache[key].items()}
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "depth" not in payload:
        raise KeyError(f"{path} must contain at least a 'depth' tensor")

    depth = _resize_hwc(torch.as_tensor(payload["depth"]), height, width).clamp_min(1e-6)
    confidence = _resize_hwc(
        torch.as_tensor(payload.get("pseudo_confidence", torch.ones_like(depth))),
        height,
        width,
    ).clamp(0.0, 1.0)
    structure = _resize_hwc(
        torch.as_tensor(payload.get("structure_confidence", torch.ones_like(depth))),
        height,
        width,
    ).clamp(0.0, 1.0)
    boundary_safe = _resize_hwc(
        torch.as_tensor(payload.get("boundary_safe", torch.ones_like(depth))),
        height,
        width,
    ).clamp(0.0, 1.0)
    valid = torch.isfinite(depth) & (depth > 0)
    out_cpu = {
        "depth": torch.where(valid, depth, torch.zeros_like(depth)).float(),
        "pseudo_confidence": torch.where(valid, confidence, torch.zeros_like(confidence)).float(),
        "structure_confidence": torch.where(valid, structure, torch.zeros_like(structure)).float(),
        "boundary_safe": torch.where(valid, boundary_safe, torch.zeros_like(boundary_safe)).float(),
    }
    if cache is not None:
        cache[key] = out_cpu
    return {name: value.to(device=device, dtype=dtype) for name, value in out_cpu.items()}


def tensor_stats(values: torch.Tensor) -> Dict[str, float]:
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


def build_mvgar_detail_map(pred_img: torch.Tensor, gt_img: torch.Tensor, highpass_weight: float = 0.35) -> torch.Tensor:
    """Detached Sobel + high-pass underwater residual map."""

    pred = pred_img.detach().float().clamp(0.0, 1.0)
    gt = gt_img.detach().float().clamp(0.0, 1.0)
    pred_chw = pred.permute(2, 0, 1).unsqueeze(0)
    gt_chw = gt.permute(2, 0, 1).unsqueeze(0)
    channels = pred_chw.shape[1]

    kx = pred_chw.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = pred_chw.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    kx = kx.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    pred_gx = F.conv2d(pred_chw, kx, padding=1, groups=channels)
    pred_gy = F.conv2d(pred_chw, ky, padding=1, groups=channels)
    gt_gx = F.conv2d(gt_chw, kx, padding=1, groups=channels)
    gt_gy = F.conv2d(gt_chw, ky, padding=1, groups=channels)
    sobel = 0.5 * ((pred_gx - gt_gx).abs() + (pred_gy - gt_gy).abs()).mean(dim=1)

    pred_blur = F.avg_pool2d(F.pad(pred_chw, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
    gt_blur = F.avg_pool2d(F.pad(gt_chw, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
    highpass = ((pred_chw - pred_blur) - (gt_chw - gt_blur)).abs().mean(dim=1)
    detail = sobel + float(highpass_weight) * highpass
    return detail.squeeze(0)[..., None].detach()


def build_mvgar_surface_evidence(
    *,
    payload: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    gaussian_depths: torch.Tensor,
    xys: torch.Tensor,
    radii: torch.Tensor,
    detail_map: torch.Tensor,
    min_pseudo_confidence: float,
    accumulation_mid: float,
    accumulation_temp: float,
    depth_std_kappa: float,
    front_depth_log_tau: float,
) -> MVGAREvidence:
    """Construct detached per-Gaussian MV-GAR support and depth targets."""

    height, width = int(payload["depth"].shape[0]), int(payload["depth"].shape[1])
    pseudo_depth = payload["depth"].detach().float().clamp_min(1e-6)
    pseudo_conf = payload["pseudo_confidence"].detach().float().clamp(0.0, 1.0)
    structure = payload["structure_confidence"].detach().float().clamp(0.0, 1.0)
    boundary_safe = payload.get("boundary_safe", torch.ones_like(structure)).detach().float().clamp(0.0, 1.0)

    accumulation = outputs["accumulation"].detach().float()
    depth_std = outputs["depth_std_relative"].detach().float()
    render_reliability = torch.sigmoid(
        (accumulation - float(accumulation_mid)) / max(float(accumulation_temp), 1e-6)
    )
    render_reliability = render_reliability * torch.exp(-depth_std / max(float(depth_std_kappa), 1e-6))
    render_reliability = render_reliability.clamp(0.0, 1.0)

    sampled_depth = sample_pixel_map_at_gaussians(pseudo_depth, xys.detach(), radii.detach(), height, width).float()
    sampled_conf = sample_pixel_map_at_gaussians(pseudo_conf, xys.detach(), radii.detach(), height, width).float()
    sampled_structure = sample_pixel_map_at_gaussians(structure, xys.detach(), radii.detach(), height, width).float()
    sampled_boundary = sample_pixel_map_at_gaussians(boundary_safe, xys.detach(), radii.detach(), height, width).float()
    sampled_reliability = sample_pixel_map_at_gaussians(
        render_reliability, xys.detach(), radii.detach(), height, width
    ).float()
    sampled_render_depth = sample_pixel_map_at_gaussians(
        outputs["depth"].detach().float().clamp_min(1e-6),
        xys.detach(),
        radii.detach(),
        height,
        width,
    ).float()
    sampled_detail = sample_pixel_map_at_gaussians(detail_map.detach(), xys.detach(), radii.detach(), height, width).float()

    depth_for_loss = gaussian_depths.reshape(-1).float().clamp_min(1e-6)
    target = sampled_depth.reshape(-1).clamp_min(1e-6)
    log_error_for_weight = (depth_for_loss.detach().log() - sampled_render_depth.reshape(-1).clamp_min(1e-6).log()).abs()
    front_gate = torch.exp(-log_error_for_weight / max(float(front_depth_log_tau), 1e-6)).clamp(0.0, 1.0)
    conf_gate = (sampled_conf.reshape(-1) >= float(min_pseudo_confidence)).float()
    valid = torch.isfinite(target) & (target > 0) & torch.isfinite(depth_for_loss.detach())
    weight = (
        sampled_conf.reshape(-1)
        * sampled_structure.reshape(-1)
        * sampled_boundary.reshape(-1)
        * sampled_reliability.reshape(-1)
        * front_gate
        * conf_gate
        * valid.float()
    ).detach()
    log_depth_error = (depth_for_loss.detach().log() - target.log()).detach()

    return MVGAREvidence(
        weight=weight,
        detail=sampled_detail.reshape(-1).detach(),
        log_depth_error=log_depth_error,
        depth_target=target.detach(),
        sampled_confidence=sampled_conf.reshape(-1).detach(),
        sampled_structure=sampled_structure.reshape(-1).detach(),
        sampled_render_reliability=sampled_reliability.reshape(-1).detach(),
        front_gate=front_gate.detach(),
    )


def mvgar_surface_anchor_loss(
    *,
    gaussian_depths: torch.Tensor,
    depth_target: torch.Tensor,
    weight: torch.Tensor,
    huber_delta: float,
) -> torch.Tensor:
    """Weighted log-depth Huber loss that keeps gradients on Gaussian depths."""

    depths = gaussian_depths.reshape(-1).float().clamp_min(1e-6)
    target = depth_target.reshape(-1).detach().float().clamp_min(1e-6)
    w = weight.reshape(-1).detach().float()
    diff = depths.log() - target.log()
    delta = max(float(huber_delta), 1e-6)
    abs_diff = diff.abs()
    loss = torch.where(abs_diff <= delta, 0.5 * diff.square() / delta, abs_diff - 0.5 * delta)
    return (w * loss).sum() / w.sum().clamp_min(1e-6)


def _robust_normalize(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = values.detach().float().reshape(-1)
    out = torch.zeros_like(values)
    keep = mask.reshape(-1) & torch.isfinite(values)
    if int(keep.sum().item()) < 4:
        return out
    ref = values[keep]
    lo = torch.quantile(ref, 0.50)
    hi = torch.quantile(ref, 0.95)
    denom = (hi - lo).clamp_min(1e-12)
    out = ((values - lo) / denom).clamp(0.0, 4.0)
    out[~torch.isfinite(out)] = 0.0
    return out


def select_mvgar_candidates(
    *,
    base_high_grads: torch.Tensor,
    avg_grad_norm: torch.Tensor,
    weight_sum: torch.Tensor,
    detail_sum: torch.Tensor,
    depth_error_sum: torch.Tensor,
    depth_error_sq_sum: torch.Tensor,
    view_count: torch.Tensor,
    min_view_count: int,
    min_mean_weight: float,
    detail_quantile: float,
    depth_variance_threshold: float,
    max_extra_ratio_to_base: float,
    max_extra_fraction_per_refine: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Select conservative MV-GAR refinement candidates."""

    n = int(base_high_grads.reshape(-1).numel())
    device = base_high_grads.device
    empty = torch.zeros(n, device=device, dtype=torch.bool)
    base = base_high_grads.reshape(-1).bool()
    wsum = weight_sum.reshape(-1).float().to(device=device)
    details = detail_sum.reshape(-1).float().to(device=device)
    derr = depth_error_sum.reshape(-1).float().to(device=device)
    derr2 = depth_error_sq_sum.reshape(-1).float().to(device=device)
    vc = view_count.reshape(-1).float().to(device=device)
    denom = wsum.clamp_min(1e-6)
    mean_weight = wsum / vc.clamp_min(1.0)
    mean_detail = details / denom
    mean_error = derr / denom
    error_var = (derr2 / denom - mean_error.square()).clamp_min(0.0)

    eligible = (
        (vc >= max(int(min_view_count), 1))
        & (mean_weight >= float(min_mean_weight))
        & (error_var <= float(depth_variance_threshold))
        & torch.isfinite(mean_detail)
        & torch.isfinite(error_var)
    )
    payload: Dict[str, Any] = {
        "base_high_grad_count": int(base.sum().item()),
        "mvgar_eligible_count": int(eligible.sum().item()),
        "mvgar_extra_candidate_count": 0,
        "mvgar_detail": tensor_stats(mean_detail[eligible]),
        "mvgar_mean_weight": tensor_stats(mean_weight[eligible]),
        "mvgar_depth_error_variance": tensor_stats(error_var[eligible]),
        "mvgar_view_count": tensor_stats(vc[eligible]),
    }
    if not bool(eligible.any().item()):
        return empty, payload

    q = min(max(float(detail_quantile), 0.0), 1.0)
    detail_threshold = torch.quantile(mean_detail[eligible], q)
    extra_pool = eligible & (~base) & (mean_detail >= detail_threshold)
    if not bool(extra_pool.any().item()):
        return empty, payload

    base_count = int(base.sum().item())
    ratio_limit = int(math.ceil(base_count * max(float(max_extra_ratio_to_base), 0.0)))
    fraction_limit = int(math.ceil(n * max(float(max_extra_fraction_per_refine), 0.0)))
    k = min(int(extra_pool.sum().item()), max(ratio_limit, 0), max(fraction_limit, 0))
    if k <= 0:
        return empty, payload

    score = _robust_normalize(avg_grad_norm.reshape(-1).to(device=device), eligible)
    score = score + _robust_normalize(mean_detail, eligible)
    score = score + 0.25 * _robust_normalize(mean_error.abs(), eligible)
    score = torch.where(extra_pool, score, torch.full_like(score, -float("inf")))
    top_idx = torch.topk(score, k=k, largest=True).indices
    candidates = torch.zeros_like(empty)
    candidates[top_idx] = True
    payload["mvgar_extra_candidate_count"] = int(candidates.sum().item())
    return candidates, payload
