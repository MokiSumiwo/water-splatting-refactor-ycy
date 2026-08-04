"""Training utilities for GMVC continuation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


def _weighted_channel_mean(value: Tensor, weight: Tensor, eps: float) -> Tensor:
    return (value * weight[:, None]).sum(dim=0) / (weight.sum() + float(eps))


def _logit_from_unit(value: Tensor, eps: float) -> Tensor:
    clipped = value.clamp(float(eps), 1.0 - float(eps))
    return torch.log(clipped) - torch.log1p(-clipped)


def _medium_terms(depth: Tensor, medium_attn: Tensor, medium_bs: Tensor, b_inf: Tensor) -> Tuple[Tensor, Tensor]:
    if depth.ndim == medium_attn.ndim - 1:
        depth = depth[..., None]
    transmission = torch.exp(-(medium_attn * depth).clamp_min(0.0))
    backscatter = b_inf * (1.0 - torch.exp(-(medium_bs * depth).clamp_min(0.0)))
    return transmission, backscatter


def _safe_quantile(value: Tensor, q: float, zero: Tensor) -> Tensor:
    if value.numel() == 0:
        return zero.detach()
    return torch.quantile(value.detach().float().reshape(-1), float(q))


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
    state: Optional[Dict[str, Tensor]] = None,
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

    beta_log_scale = max(float(getattr(config, "gmvc_residual_beta_log_scale", 0.15)), eps)
    binf_logit_scale = max(float(getattr(config, "gmvc_residual_binf_logit_scale", 0.10)), eps)
    ema_momentum = min(max(float(getattr(config, "gmvc_residual_ema_momentum", 0.99)), 0.0), 1.0)
    b_inf_logit = _logit_from_unit(b_inf, eps)
    batch_log_attn_center = _weighted_channel_mean(attn_log.detach(), weight.detach(), eps)
    batch_log_bs_center = _weighted_channel_mean(bs_log.detach(), weight.detach(), eps)
    batch_binf_logit_center = _weighted_channel_mean(b_inf_logit.detach(), weight.detach(), eps)

    def _ema_center(name: str, batch_center: Tensor) -> Tensor:
        if state is None:
            return batch_center.detach()
        key = f"gmvc_online_{name}_center"
        with torch.no_grad():
            previous = state.get(key)
            if previous is None or tuple(previous.shape) != tuple(batch_center.shape):
                state[key] = batch_center.detach().clone()
            else:
                state[key] = (
                    previous.to(device=batch_center.device, dtype=batch_center.dtype) * ema_momentum
                    + batch_center.detach() * (1.0 - ema_momentum)
                )
        return state[key].to(device=device, dtype=gt_img.dtype)

    log_attn_center = _ema_center("log_attn", batch_log_attn_center)
    log_bs_center = _ema_center("log_bs", batch_log_bs_center)
    binf_logit_center = _ema_center("binf_logit", batch_binf_logit_center)
    residual_attn = attn_log - log_attn_center.detach()
    residual_bs = bs_log - log_bs_center.detach()
    residual_binf = b_inf_logit - binf_logit_center.detach()
    residual_budget_values = torch.cat(
        [
            residual_attn / beta_log_scale,
            residual_bs / beta_log_scale,
            residual_binf / binf_logit_scale,
        ],
        dim=-1,
    )
    residual_budget_loss = _weighted_mean(charbonnier_loss(residual_budget_values, eps=huber_eps), weight, eps)
    residual_abs = residual_budget_values.detach().abs()
    residual_mean_l2 = (
        _weighted_channel_mean(residual_attn.detach(), weight.detach(), eps).square().mean()
        + _weighted_channel_mean(residual_bs.detach(), weight.detach(), eps).square().mean()
        + _weighted_channel_mean(residual_binf.detach(), weight.detach(), eps).square().mean()
    )

    closure_loss = zero
    closure_raw_l1 = zero
    closure_fixed_norm_l1 = zero
    closure_available = zero
    closure_valid_fraction = zero
    closure_weight_sum = zero
    closure_signal_floor = max(float(getattr(config, "gmvc_closure_signal_floor", 0.03)), eps)
    closure_fields = (
        "closure_partner_gt",
        "closure_partner_depth",
        "closure_partner_medium_attn",
        "closure_partner_medium_bs",
        "closure_partner_b_inf",
        "closure_denom_fixed",
        "closure_weight",
    )
    if all(name in entry for name in closure_fields):
        partner_gt = entry["closure_partner_gt"].to(device=device, dtype=gt_img.dtype)[rows]
        partner_depth = entry["closure_partner_depth"].to(device=device, dtype=gt_img.dtype)[rows].reshape(-1, 1)
        partner_attn = entry["closure_partner_medium_attn"].to(device=device, dtype=gt_img.dtype)[rows]
        partner_bs = entry["closure_partner_medium_bs"].to(device=device, dtype=gt_img.dtype)[rows]
        partner_binf = entry["closure_partner_b_inf"].to(device=device, dtype=gt_img.dtype)[rows]
        fixed_denom = entry["closure_denom_fixed"].to(device=device, dtype=gt_img.dtype)[rows]
        closure_weight = entry["closure_weight"].to(device=device, dtype=gt_img.dtype)[rows].reshape(-1).clamp_min(0.0)
        closure_weight = torch.where(weight > 0, closure_weight, torch.zeros_like(closure_weight))
        closure_weight_sum = closure_weight.detach().sum()
        if closure_weight.sum() > 0:
            current_t, current_b = _medium_terms(depth, medium_attn, medium_bs, b_inf)
            partner_t, partner_b = _medium_terms(
                partner_depth.detach(),
                partner_attn.detach(),
                partner_bs.detach(),
                partner_binf.detach(),
            )
            left = (gt_sample.detach() - current_b) * partner_t.detach()
            right = (partner_gt.detach() - partner_b.detach()) * current_t
            closure_delta = left - right
            closure_denom = torch.clamp(fixed_denom.detach(), min=closure_signal_floor)
            closure_norm = closure_delta / closure_denom.clamp_min(eps)
            closure_loss = _weighted_mean(charbonnier_loss(closure_norm, eps=huber_eps), closure_weight, eps)
            closure_raw_l1 = _weighted_mean(closure_delta.detach().abs(), closure_weight.detach(), eps)
            closure_fixed_norm_l1 = _weighted_mean(closure_norm.detach().abs(), closure_weight.detach(), eps)
            closure_available = gt_img.new_tensor(1.0)
            closure_valid_fraction = (closure_weight > 0).float().mean().detach()

    transmission, backscatter = _medium_terms(depth.detach(), medium_attn.detach(), medium_bs.detach(), b_inf.detach())
    transmission_scalar = transmission.mean(dim=-1)

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
    lambda_residual_budget = _ramped_weight(
        float(getattr(config, "lambda_gmvc_residual_budget", 0.0)),
        step,
        start,
        ramp,
        stop,
    )
    lambda_fixed_closure = _ramped_weight(
        float(getattr(config, "lambda_gmvc_fixed_closure", 0.0)),
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
        "gmvc_residual_budget_loss": residual_budget_loss * lambda_residual_budget,
        "gmvc_fixed_closure_loss": closure_loss * lambda_fixed_closure,
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
        "gmvc_residual_budget_raw": residual_budget_loss.detach(),
        "gmvc_residual_mean_l2": residual_mean_l2.detach(),
        "gmvc_residual_abs_mean": residual_abs.mean(),
        "gmvc_residual_abs_p50": _safe_quantile(residual_abs, 0.50, zero),
        "gmvc_residual_abs_p95": _safe_quantile(residual_abs, 0.95, zero),
        "gmvc_residual_abs_max": residual_abs.max() if residual_abs.numel() > 0 else zero.detach(),
        "gmvc_residual_saturation": (residual_abs > 0.95).float().mean().detach(),
        "gmvc_log_attn_center_r": log_attn_center.detach()[0],
        "gmvc_log_attn_center_g": log_attn_center.detach()[1],
        "gmvc_log_attn_center_b": log_attn_center.detach()[2],
        "gmvc_log_bs_center_r": log_bs_center.detach()[0],
        "gmvc_log_bs_center_g": log_bs_center.detach()[1],
        "gmvc_log_bs_center_b": log_bs_center.detach()[2],
        "gmvc_binf_logit_center_r": binf_logit_center.detach()[0],
        "gmvc_binf_logit_center_g": binf_logit_center.detach()[1],
        "gmvc_binf_logit_center_b": binf_logit_center.detach()[2],
        "gmvc_fixed_closure_raw": closure_loss.detach(),
        "gmvc_fixed_closure_l1": closure_raw_l1.detach(),
        "gmvc_fixed_closure_norm_l1": closure_fixed_norm_l1.detach(),
        "gmvc_fixed_closure_available": closure_available.detach(),
        "gmvc_fixed_closure_valid_fraction": closure_valid_fraction.detach(),
        "gmvc_fixed_closure_weight_sum": closure_weight_sum.detach(),
        "gmvc_transmission_p05": _safe_quantile(transmission_scalar, 0.05, zero),
        "gmvc_transmission_p50": _safe_quantile(transmission_scalar, 0.50, zero),
        "gmvc_transmission_p95": _safe_quantile(transmission_scalar, 0.95, zero),
        "gmvc_backscatter_mean": backscatter.detach().mean(),
        "gmvc_lambda_residual_budget": gt_img.new_tensor(float(lambda_residual_budget)),
        "gmvc_lambda_fixed_closure": gt_img.new_tensor(float(lambda_fixed_closure)),
    }
    return losses, metrics
