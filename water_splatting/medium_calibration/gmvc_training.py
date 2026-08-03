"""Training utilities for GMVC continuation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .gmvc_losses import charbonnier_loss, invert_intrinsic_radiance


def load_gmvc_training_bank(path: str | Path) -> Dict[str, Any]:
    """Load a CPU GMVC track bank built by scripts/diagnostics/build_gmvc_tracks.py."""

    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def _camera_key(outputs: Dict[str, Tensor]) -> str | None:
    value = outputs.get("camera_index")
    if value is None:
        return None
    return str(int(value.detach().cpu().reshape(-1)[0].item()))


def _sample_hwc(image: Tensor, xy: Tensor) -> Tensor:
    if xy.numel() == 0:
        channels = image.shape[-1] if image.ndim == 3 else 1
        return image.new_empty((0, channels))
    if image.ndim == 2:
        image = image[..., None]
    h, w = image.shape[:2]
    grid_x = 2.0 * xy[:, 0] / max(w - 1, 1) - 1.0
    grid_y = 2.0 * xy[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    nchw = image.permute(2, 0, 1).unsqueeze(0)
    sampled = F.grid_sample(nchw, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled[0, :, :, 0].T.contiguous()


def _weighted_mean(value: Tensor, weight: Tensor, eps: float) -> Tensor:
    return (value * weight[:, None]).sum() / (weight.sum() * value.shape[-1] + float(eps))


def _choose_rows(count: int, max_count: int, step: int, seed: int, device: torch.device) -> Tensor:
    if max_count <= 0 or count <= max_count:
        return torch.arange(count, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + int(step) * 1009)
    return torch.randperm(count, generator=generator, device=device)[:max_count]


def _ramped_weight(weight: float, step: int, start: int, ramp: int, stop: int) -> float:
    if weight <= 0.0 or step < start or step >= stop:
        return 0.0
    if ramp <= 0:
        return float(weight)
    return float(weight) * min((step - start) / max(float(ramp), 1.0), 1.0)


def compute_gmvc_training_terms(
    outputs: Dict[str, Tensor],
    gt_img: Tensor,
    bank: Dict[str, Any],
    step: int,
    config: Any,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """Compute current-view GMVC losses against a detached offline track bank."""

    device = gt_img.device
    eps = float(getattr(config, "gmvc_eps", 1e-4))
    camera_key = _camera_key(outputs)
    zero = gt_img.new_zeros(())
    if camera_key is None:
        return {}, {"gmvc_available_tracks": zero, "gmvc_sampled_tracks": zero}

    per_camera = bank.get("per_camera", {})
    entry = per_camera.get(camera_key)
    if entry is None:
        return {}, {"gmvc_available_tracks": zero, "gmvc_sampled_tracks": zero}

    xy = entry["xy"].to(device=device, dtype=gt_img.dtype)
    count = int(xy.shape[0])
    if count == 0:
        return {}, {"gmvc_available_tracks": zero, "gmvc_sampled_tracks": zero}

    rows = _choose_rows(
        count=count,
        max_count=int(getattr(config, "gmvc_max_tracks_per_step", 4096)),
        step=step,
        seed=int(getattr(config, "gmvc_seed", 42)) + int(camera_key) * 9173,
        device=device,
    )
    xy = xy[rows]
    weight = entry["weight"].to(device=device, dtype=gt_img.dtype)[rows].reshape(-1).clamp_min(0.0)
    j_consensus = entry["j_consensus"].to(device=device, dtype=gt_img.dtype)[rows]
    attn_log_center = entry["medium_attn_log_center"].to(device=device, dtype=gt_img.dtype)[rows]
    bs_log_center = entry["medium_bs_log_center"].to(device=device, dtype=gt_img.dtype)[rows]
    b_inf_center = entry["b_inf_center"].to(device=device, dtype=gt_img.dtype)[rows]

    if weight.sum() <= 0:
        return {}, {"gmvc_available_tracks": gt_img.new_tensor(float(count)), "gmvc_sampled_tracks": zero}

    depth = _sample_hwc(outputs["depth"], xy)
    if bool(getattr(config, "gmvc_detach_depth", True)):
        depth = depth.detach()
    if depth.shape[-1] != 1:
        depth = depth.mean(dim=-1, keepdim=True)
    gt_sample = _sample_hwc(gt_img, xy)
    medium_attn = _sample_hwc(outputs["medium_attn"], xy)
    medium_bs = _sample_hwc(outputs["medium_bs"], xy)
    b_inf = _sample_hwc(outputs.get("b_inf", outputs["medium_rgb"]), xy)

    j_hat = invert_intrinsic_radiance(
        observed_rgb=gt_sample,
        depth=depth,
        medium_attn=medium_attn,
        medium_bs=medium_bs,
        b_inf=b_inf,
        eps=eps,
    )
    valid_j = (
        torch.isfinite(j_hat).all(dim=-1)
        & (j_hat >= float(getattr(config, "gmvc_j_clamp_min", -0.25))).all(dim=-1)
        & (j_hat <= float(getattr(config, "gmvc_j_clamp_max", 1.25))).all(dim=-1)
    )
    weight = torch.where(valid_j, weight, torch.zeros_like(weight))
    if weight.sum() <= 0:
        return {}, {"gmvc_available_tracks": gt_img.new_tensor(float(count)), "gmvc_sampled_tracks": zero}

    huber_eps = float(getattr(config, "gmvc_charbonnier_eps", 1e-6))
    j_residual = charbonnier_loss(j_hat - j_consensus, eps=huber_eps)
    j_loss = _weighted_mean(j_residual, weight, eps)

    range_scale = max(float(getattr(config, "gmvc_range_log_scale", 0.25)), eps)
    attn_log = torch.log(medium_attn.clamp_min(eps))
    bs_log = torch.log(medium_bs.clamp_min(eps))
    range_residual = torch.cat(
        [
            charbonnier_loss((attn_log - attn_log_center) / range_scale, eps=huber_eps),
            charbonnier_loss((bs_log - bs_log_center) / range_scale, eps=huber_eps),
        ],
        dim=-1,
    )
    range_loss = _weighted_mean(range_residual, weight, eps)

    b_inf_residual = charbonnier_loss(b_inf - b_inf_center, eps=huber_eps)
    b_inf_loss = _weighted_mean(b_inf_residual, weight, eps)

    intrinsic_source_key = str(getattr(config, "gmvc_intrinsic_source", "J_proxy_raw"))
    intrinsic_source = outputs.get(intrinsic_source_key)
    if intrinsic_source is None:
        intrinsic_loss = zero
        intrinsic_source_available = zero
    else:
        intrinsic_source_available = gt_img.new_tensor(1.0)
        intrinsic_sample = _sample_hwc(intrinsic_source, xy)
        valid_intrinsic = torch.isfinite(intrinsic_sample).all(dim=-1)
        intrinsic_weight = torch.where(valid_intrinsic, weight, torch.zeros_like(weight))
        if intrinsic_weight.sum() <= 0:
            intrinsic_loss = zero
        else:
            intrinsic_residual = charbonnier_loss(intrinsic_sample - j_consensus.detach(), eps=huber_eps)
            intrinsic_loss = _weighted_mean(intrinsic_residual, intrinsic_weight, eps)

    start = int(getattr(config, "gmvc_start_step", 10000))
    stop = int(getattr(config, "gmvc_stop_step", 15000))
    ramp = int(getattr(config, "gmvc_ramp_steps", 500))
    lambda_j = _ramped_weight(float(getattr(config, "lambda_gmvc_j", 0.0)), step, start, ramp, stop)
    lambda_range = _ramped_weight(float(getattr(config, "lambda_gmvc_range", 0.0)), step, start, ramp, stop)
    lambda_binf = _ramped_weight(float(getattr(config, "lambda_gmvc_binf", 0.0)), step, start, ramp, stop)
    lambda_intrinsic = _ramped_weight(
        float(getattr(config, "lambda_gmvc_intrinsic", 0.0)),
        step,
        start,
        ramp,
        stop,
    )

    losses = {
        "gmvc_j_consistency_loss": j_loss * lambda_j,
        "gmvc_range_loss": range_loss * lambda_range,
        "gmvc_binf_loss": b_inf_loss * lambda_binf,
        "gmvc_intrinsic_loss": intrinsic_loss * lambda_intrinsic,
    }
    metrics = {
        "gmvc_available_tracks": gt_img.new_tensor(float(count)),
        "gmvc_sampled_tracks": gt_img.new_tensor(float(xy.shape[0])),
        "gmvc_valid_weight_sum": weight.detach().sum(),
        "gmvc_valid_fraction": (weight > 0).float().mean().detach(),
        "gmvc_j_consistency_raw": j_loss.detach(),
        "gmvc_range_raw": range_loss.detach(),
        "gmvc_binf_raw": b_inf_loss.detach(),
        "gmvc_intrinsic_raw": intrinsic_loss.detach(),
        "gmvc_intrinsic_source_available": intrinsic_source_available.detach(),
        "gmvc_lambda_j": gt_img.new_tensor(float(lambda_j)),
        "gmvc_lambda_range": gt_img.new_tensor(float(lambda_range)),
        "gmvc_lambda_binf": gt_img.new_tensor(float(lambda_binf)),
        "gmvc_lambda_intrinsic": gt_img.new_tensor(float(lambda_intrinsic)),
    }
    return losses, metrics
