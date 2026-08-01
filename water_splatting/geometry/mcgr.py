"""MCGR correspondence-gated refinement utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from water_splatting.cleanup import sample_pixel_map_at_gaussians
from water_splatting.geometry.mvgar import tensor_stats


@dataclass
class MCGREvidence:
    """Detached per-Gaussian MCGR evidence."""

    weight: torch.Tensor
    persistent: torch.Tensor
    sampled_confidence: torch.Tensor
    sampled_valid_neighbors: torch.Tensor
    sampled_render_reliability: torch.Tensor


def _resize_hwc1(value: torch.Tensor, height: int, width: int, *, mode: str = "bilinear") -> torch.Tensor:
    value = value.detach().float()
    if value.ndim == 2:
        value = value[..., None]
    if value.ndim == 3 and value.shape[0] == 1 and value.shape[-1] != 1:
        value = value.permute(1, 2, 0)
    if value.ndim != 3:
        raise ValueError(f"Expected HxW or HxWxC tensor, got {tuple(value.shape)}")
    if value.shape[-1] != 1:
        value = value[..., :1]
    if int(value.shape[0]) == int(height) and int(value.shape[1]) == int(width):
        return value.contiguous()
    chw = value.permute(2, 0, 1)[None]
    if mode == "nearest":
        out = F.interpolate(chw, size=(height, width), mode="nearest")
    else:
        out = F.interpolate(chw, size=(height, width), mode=mode, align_corners=False)
    return out[0].permute(1, 2, 0).contiguous()


def load_mcgr_correspondence_payload(
    correspondence_dir: str,
    image_idx: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    cache: Optional[Dict[Tuple[str, int], Dict[str, torch.Tensor]]] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Load a precomputed ``view_XXXX_mcgr.pt`` correspondence payload."""

    path = Path(correspondence_dir) / f"view_{int(image_idx):04d}_mcgr.pt"
    key = (str(path), int(image_idx))
    if cache is not None and key in cache:
        return {
            name: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            for name, value in cache[key].items()
        }
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    required = ["neighbor_ids", "corr_uv", "corr_confidence", "corr_valid", "valid_neighbor_count", "cross_view_confidence"]
    missing = [name for name in required if name not in payload]
    if missing:
        raise KeyError(f"{path} missing MCGR payload keys: {missing}")
    out_cpu = {
        "neighbor_ids": torch.as_tensor(payload["neighbor_ids"], dtype=torch.long).reshape(-1),
        "corr_uv": torch.as_tensor(payload["corr_uv"]).float(),
        "corr_confidence": torch.as_tensor(payload["corr_confidence"]).float().clamp(0.0, 1.0),
        "corr_valid": torch.as_tensor(payload["corr_valid"]).bool(),
        "valid_neighbor_count": torch.as_tensor(payload["valid_neighbor_count"]).float(),
        "cross_view_confidence": torch.as_tensor(payload["cross_view_confidence"]).float().clamp(0.0, 1.0),
        "hf_confidence": torch.as_tensor(payload.get("hf_confidence", torch.ones_like(payload["cross_view_confidence"]))).float().clamp(0.0, 1.0),
    }
    if cache is not None:
        cache[key] = out_cpu
    return {
        name: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        for name, value in out_cpu.items()
    }


def _robust01(value: torch.Tensor) -> torch.Tensor:
    flat = value.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() < 8:
        return torch.zeros_like(value)
    lo = torch.quantile(flat, 0.50)
    hi = torch.quantile(flat, 0.95)
    return ((value.float() - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def build_mcgr_detail_residual(
    pred_img: torch.Tensor,
    gt_img: torch.Tensor,
    *,
    downscale: int,
    highpass_weight: float,
) -> torch.Tensor:
    """Build detached normalized high-frequency residual at MCGR bank resolution."""

    pred = pred_img.detach().float().clamp(0.0, 1.0)
    gt = gt_img.detach().float().clamp(0.0, 1.0)
    pred_chw = pred.permute(2, 0, 1).unsqueeze(0)
    gt_chw = gt.permute(2, 0, 1).unsqueeze(0)
    channels = int(pred_chw.shape[1])
    kx = pred_chw.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = pred_chw.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    kx = kx.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    sobel = 0.5 * (
        (F.conv2d(pred_chw, kx, padding=1, groups=channels) - F.conv2d(gt_chw, kx, padding=1, groups=channels)).abs()
        + (F.conv2d(pred_chw, ky, padding=1, groups=channels) - F.conv2d(gt_chw, ky, padding=1, groups=channels)).abs()
    ).mean(dim=1, keepdim=True)
    pred_blur = F.avg_pool2d(F.pad(pred_chw, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
    gt_blur = F.avg_pool2d(F.pad(gt_chw, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
    highpass = ((pred_chw - pred_blur) - (gt_chw - gt_blur)).abs().mean(dim=1, keepdim=True)
    residual = sobel + float(highpass_weight) * highpass
    residual = _robust01(residual)
    d = max(int(downscale), 1)
    if d > 1:
        h, w = int(residual.shape[-2]), int(residual.shape[-1])
        residual = F.interpolate(residual, size=(max(h // d, 1), max(w // d, 1)), mode="bilinear", align_corners=False)
    return residual[0, 0, ..., None].detach()


def update_mcgr_residual_bank(
    bank_value: Optional[torch.Tensor],
    residual: torch.Tensor,
    *,
    ema_decay: float,
) -> torch.Tensor:
    """Return an updated CPU float16 residual-bank tensor."""

    current = residual.detach().float().cpu()
    if bank_value is None or tuple(bank_value.shape) != tuple(current.shape):
        return current.half()
    decay = min(max(float(ema_decay), 0.0), 0.999)
    return (decay * bank_value.float() + (1.0 - decay) * current).half()


def _sample_lowres_map(pixel_map: torch.Tensor, xys: torch.Tensor, radii: torch.Tensor, height: int, width: int, downscale: int) -> torch.Tensor:
    scale = max(float(downscale), 1.0)
    return sample_pixel_map_at_gaussians(
        pixel_map,
        xys.detach() / scale,
        radii.detach() / scale,
        height,
        width,
    )


def _grid_sample_map(value: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Sample a HxWx1 map at low-res pixel coordinates uv[..., 0:2]."""

    h, w = int(value.shape[0]), int(value.shape[1])
    src = value.float().permute(2, 0, 1)[None]
    x = uv[..., 0].float()
    y = uv[..., 1].float()
    gx = 2.0 * x / max(w - 1, 1) - 1.0
    gy = 2.0 * y / max(h - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None]
    return F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]


def build_mcgr_persistent_map(
    *,
    payload: Dict[str, torch.Tensor],
    current_residual: torch.Tensor,
    residual_bank: Dict[int, torch.Tensor],
    residual_bank_steps: Dict[int, int],
    step: int,
    max_age_steps: int,
    min_valid_neighbors: int,
    residual_match_tau: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Warp neighbor residual bank entries and build a detached persistent residual map."""

    device = current_residual.device
    dtype = current_residual.dtype
    h, w = int(current_residual.shape[0]), int(current_residual.shape[1])
    neighbor_ids = payload["neighbor_ids"].detach().long().cpu().tolist()
    corr_uv = payload["corr_uv"].float()
    corr_conf = payload["corr_confidence"].float().clamp(0.0, 1.0)
    corr_valid = payload["corr_valid"].bool()

    scores = []
    valid_masks = []
    ages = []
    tau = max(float(residual_match_tau), 1e-6)
    for k, neighbor_id in enumerate(neighbor_ids):
        bank = residual_bank.get(int(neighbor_id))
        bank_step = residual_bank_steps.get(int(neighbor_id), -10**9)
        if bank is None or int(step) - int(bank_step) > int(max_age_steps):
            continue
        neighbor = bank.to(device=device, dtype=dtype)
        if neighbor.ndim == 2:
            neighbor = neighbor[..., None]
        if int(neighbor.shape[0]) != h or int(neighbor.shape[1]) != w:
            neighbor = _resize_hwc1(neighbor, h, w)
        warped = _grid_sample_map(neighbor, corr_uv[k]).clamp(0.0, 1.0)
        valid = corr_valid[k] & torch.isfinite(warped)
        score = torch.sqrt((current_residual[..., 0].float() * warped.float()).clamp_min(0.0))
        score = score * torch.exp(-(current_residual[..., 0].float() - warped.float()).abs() / tau)
        score = score * corr_conf[k].float()
        scores.append(torch.where(valid, score, torch.zeros_like(score)))
        valid_masks.append(valid)
        ages.append(float(int(step) - int(bank_step)))

    if not scores:
        zero = torch.zeros_like(current_residual[..., :1])
        return zero, {
            "mcgr_bank_valid_neighbor_count_mean": 0.0,
            "mcgr_bank_mean_age": 0.0,
            "mcgr_persistent_residual_mean": 0.0,
            "mcgr_persistent_residual_p90": 0.0,
        }

    score_stack = torch.stack(scores, dim=0)
    valid_stack = torch.stack(valid_masks, dim=0)
    valid_count = valid_stack.sum(dim=0)
    masked = torch.where(valid_stack, score_stack, torch.full_like(score_stack, -float("inf")))
    k_top = min(2, int(masked.shape[0]))
    top = torch.topk(masked, k=k_top, dim=0).values
    top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    top_mean = top.mean(dim=0)
    keep = valid_count >= max(int(min_valid_neighbors), 1)
    persistent = torch.where(keep, top_mean, torch.zeros_like(top_mean))
    cross = payload["cross_view_confidence"].float().clamp(0.0, 1.0)
    hf = payload.get("hf_confidence", torch.ones_like(cross)).float().clamp(0.0, 1.0)
    persistent = (persistent * cross * hf).clamp(0.0, 1.0)[..., None].detach()
    flat = persistent.reshape(-1)
    stats = {
        "mcgr_bank_valid_neighbor_count_mean": float(valid_count.float().mean().item()),
        "mcgr_bank_mean_age": float(sum(ages) / max(len(ages), 1)),
        "mcgr_persistent_residual_mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "mcgr_persistent_residual_p90": float(torch.quantile(flat.float(), 0.90).item()) if flat.numel() else 0.0,
    }
    return persistent, stats


def build_mcgr_gaussian_evidence(
    *,
    outputs: Dict[str, torch.Tensor],
    xys: torch.Tensor,
    radii: torch.Tensor,
    persistent_map: torch.Tensor,
    correspondence_payload: Dict[str, torch.Tensor],
    accumulation_mid: float,
    accumulation_temp: float,
    depth_std_kappa: float,
    downscale: int,
) -> MCGREvidence:
    """Sample MCGR maps at Gaussian centers and return detached evidence."""

    h, w = int(persistent_map.shape[0]), int(persistent_map.shape[1])
    accumulation = outputs["accumulation"].detach().float()
    depth_std = outputs["depth_std_relative"].detach().float()
    render_reliability = torch.sigmoid(
        (accumulation - float(accumulation_mid)) / max(float(accumulation_temp), 1e-6)
    )
    render_reliability = (render_reliability * torch.exp(-depth_std / max(float(depth_std_kappa), 1e-6))).clamp(0.0, 1.0)
    sampled_reliability = sample_pixel_map_at_gaussians(
        render_reliability,
        xys.detach(),
        radii.detach(),
        int(accumulation.shape[0]),
        int(accumulation.shape[1]),
    ).float()
    sampled_persistent = _sample_lowres_map(persistent_map, xys, radii, h, w, downscale).float()
    sampled_conf = _sample_lowres_map(
        correspondence_payload["cross_view_confidence"][..., None],
        xys,
        radii,
        h,
        w,
        downscale,
    ).float()
    sampled_valid = _sample_lowres_map(
        correspondence_payload["valid_neighbor_count"][..., None].float(),
        xys,
        radii,
        h,
        w,
        downscale,
    ).float()
    weight = (sampled_conf * sampled_reliability).clamp(0.0, 1.0).detach()
    return MCGREvidence(
        weight=weight.reshape(-1).detach(),
        persistent=sampled_persistent.reshape(-1).detach(),
        sampled_confidence=sampled_conf.reshape(-1).detach(),
        sampled_valid_neighbors=sampled_valid.reshape(-1).detach(),
        sampled_render_reliability=sampled_reliability.reshape(-1).detach(),
    )


def select_mcgr_candidates(
    *,
    base_high_grads: torch.Tensor,
    weight_sum: torch.Tensor,
    persistent_sum: torch.Tensor,
    view_count: torch.Tensor,
    grad_direction_sum: Optional[torch.Tensor] = None,
    grad_weight_sum: Optional[torch.Tensor] = None,
    gradient_coherence_enabled: bool = False,
    gradient_coherence_threshold: float = 0.35,
    min_view_count: int,
    min_mean_confidence: float,
    persistent_quantile: float,
    max_extra_ratio_to_base: float,
    max_extra_fraction_per_refine: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Select conservative MCGR extra refinement candidates."""

    n = int(base_high_grads.reshape(-1).numel())
    device = base_high_grads.device
    empty = torch.zeros(n, device=device, dtype=torch.bool)
    base = base_high_grads.reshape(-1).bool()
    wsum = weight_sum.reshape(-1).float().to(device=device)
    psum = persistent_sum.reshape(-1).float().to(device=device)
    vc = view_count.reshape(-1).float().to(device=device)
    mean_conf = wsum / vc.clamp_min(1.0)
    mean_persistent = psum / wsum.clamp_min(1e-6)
    supported = vc > 0
    view_qualified = vc >= max(int(min_view_count), 1)
    confidence_qualified = view_qualified & (mean_conf >= float(min_mean_confidence))
    grad_coherence = torch.ones(n, device=device, dtype=torch.float32)
    if gradient_coherence_enabled:
        if grad_direction_sum is None or grad_weight_sum is None:
            grad_coherence = torch.zeros(n, device=device, dtype=torch.float32)
        else:
            gsum = grad_direction_sum.reshape(n, -1).float().to(device=device)
            gw = grad_weight_sum.reshape(-1).float().to(device=device).clamp_min(1e-6)
            grad_coherence = (gsum.norm(dim=-1) / gw).clamp(0.0, 1.0)
        confidence_qualified = confidence_qualified & (grad_coherence >= float(gradient_coherence_threshold))
    payload: Dict[str, Any] = {
        "base_high_grad_count": int(base.sum().item()),
        "mcgr_supported_gaussian_count": int(supported.sum().item()),
        "mcgr_min_view_qualified_count": int(view_qualified.sum().item()),
        "mcgr_confidence_qualified_count": int(confidence_qualified.sum().item()),
        "mcgr_gradient_coherent_count": int(
            ((grad_coherence >= float(gradient_coherence_threshold)) & view_qualified).sum().item()
        )
        if gradient_coherence_enabled
        else int(view_qualified.sum().item()),
        "mcgr_extra_candidate_count": 0,
        "mcgr_mean_confidence": tensor_stats(mean_conf[confidence_qualified]),
        "mcgr_mean_persistent": tensor_stats(mean_persistent[confidence_qualified]),
        "mcgr_view_count": tensor_stats(vc[confidence_qualified]),
        "mcgr_grad_coherence": tensor_stats(grad_coherence[confidence_qualified])
        if gradient_coherence_enabled
        else {"mean": 1.0, "p50": 1.0, "p90": 1.0, "p95": 1.0},
    }
    if not bool(confidence_qualified.any().item()):
        return empty, payload
    q = min(max(float(persistent_quantile), 0.0), 1.0)
    threshold = torch.quantile(mean_persistent[confidence_qualified], q)
    extra_pool = confidence_qualified & (~base) & (mean_persistent >= threshold)
    payload["mcgr_persistent_qualified_count"] = int(extra_pool.sum().item())
    if not bool(extra_pool.any().item()):
        return empty, payload
    base_count = int(base.sum().item())
    ratio_limit = int(math.ceil(base_count * max(float(max_extra_ratio_to_base), 0.0)))
    fraction_limit = int(math.ceil(n * max(float(max_extra_fraction_per_refine), 0.0)))
    k = min(int(extra_pool.sum().item()), max(ratio_limit, 0), max(fraction_limit, 0))
    if k <= 0:
        return empty, payload
    score = mean_persistent * mean_conf.clamp_min(0.0).sqrt()
    if gradient_coherence_enabled:
        score = score * grad_coherence.clamp_min(0.0).sqrt()
    score = torch.where(extra_pool, score, torch.full_like(score, -float("inf")))
    top_idx = torch.topk(score, k=k, largest=True).indices
    candidates = torch.zeros_like(empty)
    candidates[top_idx] = True
    payload["mcgr_extra_candidate_count"] = int(candidates.sum().item())
    payload["mcgr_selected_mean_confidence"] = tensor_stats(mean_conf[candidates])
    payload["mcgr_selected_mean_persistent"] = tensor_stats(mean_persistent[candidates])
    payload["mcgr_selected_view_count"] = tensor_stats(vc[candidates])
    if gradient_coherence_enabled:
        payload["mcgr_selected_grad_coherence"] = tensor_stats(grad_coherence[candidates])
    return candidates, payload
