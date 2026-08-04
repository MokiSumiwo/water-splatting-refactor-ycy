"""Training utilities for GMVC continuation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .gmvc_losses import charbonnier_loss, invert_intrinsic_radiance


def load_gmvc_training_bank(path: str | Path) -> Dict[str, Any]:
    """Load a CPU GMVC track bank built by scripts/diagnostics/build_gmvc_tracks.py."""

    try:
        bank = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        bank = torch.load(Path(path), map_location="cpu")
    observations = bank.get("observations", {})
    if "camera_to_track_ids" not in bank and "camera_index" in observations and "track_id" in observations:
        camera_to_track_ids = {}
        camera_index = observations["camera_index"].long()
        track_id = observations["track_id"].long()
        for camera in camera_index.unique(sorted=True).tolist():
            rows = camera_index == int(camera)
            camera_to_track_ids[str(int(camera))] = track_id[rows].unique(sorted=True).long()
        bank["camera_to_track_ids"] = camera_to_track_ids
    return bank


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


def _track_balanced_mean(value: Tensor, weight: Tensor, local_track: Tensor, track_count: int, eps: float) -> Tensor:
    if value.numel() == 0 or track_count <= 0:
        return value.new_zeros(())
    weighted_sum = _scatter_sum(value * weight[:, None], local_track, track_count).sum(dim=-1)
    weight_sum = _scatter_sum(weight[:, None], local_track, track_count).reshape(-1)
    valid = weight_sum > 0
    if not bool(valid.any()):
        return value.new_zeros(())
    per_track = weighted_sum[valid] / (weight_sum[valid] * value.shape[-1] + float(eps))
    return per_track.mean()


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


def _sample_v2_track_rows(
    observations: Dict[str, Tensor],
    max_tracks: int,
    step: int,
    seed: int,
    eligible_track_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    track_ids = observations.get("track_ids")
    starts = observations.get("track_starts")
    lengths = observations.get("track_lengths")
    if track_ids is None or starts is None or lengths is None or int(track_ids.numel()) == 0:
        empty = torch.empty((0,), dtype=torch.long)
        return empty, empty, empty

    if eligible_track_ids is not None and int(eligible_track_ids.numel()) > 0:
        chosen_pool = eligible_track_ids.long().cpu()
    else:
        chosen_pool = torch.arange(int(track_ids.shape[0]), dtype=torch.long)
    track_count = int(chosen_pool.shape[0])
    if track_count == 0:
        empty = torch.empty((0,), dtype=torch.long)
        return empty, empty, empty
    if max_tracks <= 0 or track_count <= max_tracks:
        chosen = chosen_pool
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + int(step) * 7919)
        chosen = chosen_pool[torch.randperm(track_count, generator=generator)[:max_tracks]]

    row_chunks = []
    local_chunks = []
    for local_idx, track_idx in enumerate(chosen.tolist()):
        start = int(starts[track_idx].item())
        length = int(lengths[track_idx].item())
        row_chunks.append(torch.arange(start, start + length, dtype=torch.long))
        local_chunks.append(torch.full((length,), int(local_idx), dtype=torch.long))
    if not row_chunks:
        empty = torch.empty((0,), dtype=torch.long)
        return empty, empty, empty
    return torch.cat(row_chunks), torch.cat(local_chunks), track_ids[chosen].long()


def _scatter_sum(value: Tensor, index: Tensor, count: int) -> Tensor:
    out = value.new_zeros((count,) + tuple(value.shape[1:]))
    if value.numel() == 0:
        return out
    expand_index = index.reshape(-1, *([1] * (value.ndim - 1))).expand_as(value)
    return out.scatter_add_(0, expand_index, value)


def _track_scalar_min_max(value: Tensor, index: Tensor, valid: Tensor, count: int) -> Tuple[Tensor, Tensor]:
    mins = value.new_zeros((count,))
    maxs = value.new_zeros((count,))
    for track_idx in range(count):
        rows = torch.nonzero((index == track_idx) & valid, as_tuple=False).reshape(-1)
        if int(rows.numel()) == 0:
            continue
        vals = value[rows].reshape(-1)
        mins[track_idx] = vals.min()
        maxs[track_idx] = vals.max()
    return mins, maxs


def _gmvc_v3_object_phase(config: Any, step: int) -> bool:
    if not bool(getattr(config, "gmvc_v3_enabled", False)):
        return False
    start = int(getattr(config, "gmvc_start_step", 10000))
    stop = int(getattr(config, "gmvc_stop_step", 15000))
    if step < start or step >= stop:
        return False
    medium_steps = max(int(getattr(config, "gmvc_v3_medium_steps", 4)), 0)
    object_steps = max(int(getattr(config, "gmvc_v3_object_steps", 1)), 0)
    cycle = medium_steps + object_steps
    if cycle <= 0 or object_steps <= 0:
        return False
    return ((int(step) - start) % cycle) >= medium_steps


def _compute_v2_pair_indices(local_track: Tensor, depth: Tensor, weight: Tensor, track_count: int) -> Tuple[Tensor, Tensor]:
    near_indices = []
    far_indices = []
    valid = weight > 0
    for track_idx in range(track_count):
        rows = torch.nonzero((local_track == track_idx) & valid, as_tuple=False).reshape(-1)
        if int(rows.numel()) < 2:
            continue
        track_depth = depth[rows].reshape(-1)
        near_indices.append(rows[int(torch.argmin(track_depth).item())])
        far_indices.append(rows[int(torch.argmax(track_depth).item())])
    if not near_indices:
        empty = torch.empty((0,), dtype=torch.long, device=local_track.device)
        return empty, empty
    return torch.stack(near_indices).long(), torch.stack(far_indices).long()


def _compute_gmvc_v2_terms(
    *,
    outputs: Optional[Dict[str, Tensor]] = None,
    gt_img: Tensor,
    bank: Dict[str, Any],
    step: int,
    config: Any,
    state: Optional[Dict[str, Tensor]] = None,
    medium_query_fn: Optional[Callable[[Tensor, Tensor, Tensor, Optional[Tensor]], Dict[str, Tensor]]],
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    device = gt_img.device
    dtype = gt_img.dtype
    zero = gt_img.new_zeros(())
    eps = float(getattr(config, "gmvc_eps", 1e-4))
    observations = bank.get("observations")
    if observations is None or medium_query_fn is None:
        return {}, {
            "gmvc_v2_available_tracks": zero,
            "gmvc_v2_sampled_tracks": zero,
            "gmvc_v2_sampled_observations": zero,
        }

    track_count_total = int(observations.get("track_ids", torch.empty(0)).shape[0])
    if track_count_total == 0:
        return {}, {
            "gmvc_v2_available_tracks": zero,
            "gmvc_v2_sampled_tracks": zero,
            "gmvc_v2_sampled_observations": zero,
        }

    object_phase = _gmvc_v3_object_phase(config, step)
    eligible_track_ids = None
    if (
        object_phase
        and bool(getattr(config, "gmvc_v3_target_current_camera_tracks", False))
        and outputs is not None
    ):
        camera_key = _camera_key(outputs)
        if camera_key is not None:
            eligible_track_ids = bank.get("camera_to_track_ids", {}).get(str(camera_key))

    rows_cpu, local_cpu, sampled_track_ids_cpu = _sample_v2_track_rows(
        observations=observations,
        max_tracks=int(getattr(config, "gmvc_v2_max_tracks_per_step", 512)),
        step=step,
        seed=int(getattr(config, "gmvc_seed", 42)) + 44497,
        eligible_track_ids=eligible_track_ids,
    )
    if int(rows_cpu.numel()) == 0:
        return {}, {
            "gmvc_v2_available_tracks": gt_img.new_tensor(float(track_count_total)),
            "gmvc_v2_sampled_tracks": zero,
            "gmvc_v2_sampled_observations": zero,
        }

    local_track = local_cpu.to(device=device)
    sampled_track_count = int(local_cpu.max().item()) + 1 if int(local_cpu.numel()) > 0 else 0
    gt = observations["gt"][rows_cpu].to(device=device, dtype=dtype)
    weight = observations["weight"][rows_cpu].to(device=device, dtype=dtype).reshape(-1).clamp_min(0.0)
    depth = observations["fixed_depth"][rows_cpu].to(device=device, dtype=dtype).reshape(-1, 1).clamp_min(eps)
    directions = observations["ray_direction"][rows_cpu].to(device=device, dtype=dtype)
    image_xy = observations["image_xy_norm"][rows_cpu].to(device=device, dtype=dtype)
    camera_centers = observations["camera_center"][rows_cpu].to(device=device, dtype=dtype)
    bank_attn = observations["bank_medium_attn"][rows_cpu].to(device=device, dtype=dtype)
    bank_bs = observations["bank_medium_bs"][rows_cpu].to(device=device, dtype=dtype)
    bank_binf = observations["bank_b_inf"][rows_cpu].to(device=device, dtype=dtype)

    current_medium = medium_query_fn(directions, image_xy, camera_centers, None)
    medium_attn = current_medium["medium_attn"]
    medium_bs = current_medium["medium_bs"]
    b_inf = current_medium.get("b_inf", current_medium["medium_rgb"])
    finite = (
        torch.isfinite(gt).all(dim=-1)
        & torch.isfinite(medium_attn).all(dim=-1)
        & torch.isfinite(medium_bs).all(dim=-1)
        & torch.isfinite(b_inf).all(dim=-1)
        & torch.isfinite(depth).all(dim=-1)
    )
    weight = torch.where(finite, weight, torch.zeros_like(weight))
    transmission, backscatter = _medium_terms(depth, medium_attn, medium_bs, b_inf)

    min_obs = max(int(getattr(config, "gmvc_v2_min_observations_per_track", 2)), 2)
    obs_count = _scatter_sum((weight > 0).to(dtype)[:, None], local_track, sampled_track_count).reshape(-1)
    base_track_valid = obs_count >= float(min_obs)
    transmission_scalar_for_gate = transmission.detach().mean(dim=-1)
    valid_rows = weight > 0
    t_min, t_max = _track_scalar_min_max(
        transmission_scalar_for_gate,
        local_track,
        valid_rows,
        sampled_track_count,
    )
    depth_scalar_for_gate = depth.detach().reshape(-1)
    d_min, d_max = _track_scalar_min_max(depth_scalar_for_gate, local_track, valid_rows, sampled_track_count)
    median_depth = _scatter_sum((depth_scalar_for_gate * valid_rows.to(dtype))[:, None], local_track, sampled_track_count).reshape(-1)
    median_depth = median_depth / obs_count.clamp_min(1.0)
    initial_profile_weight = torch.where(base_track_valid[local_track], weight, torch.zeros_like(weight))
    initial_denominator = _scatter_sum(
        initial_profile_weight[:, None] * transmission.detach().square(),
        local_track,
        sampled_track_count,
    )
    hessian_scalar = initial_denominator.detach().mean(dim=-1)
    depth_span_rel = (d_max - d_min) / median_depth.clamp_min(eps)
    track_valid = (
        base_track_valid
        & (hessian_scalar >= float(getattr(config, "gmvc_profile_min_hessian", 0.0)))
        & ((t_max - t_min) >= float(getattr(config, "gmvc_profile_min_transmission_span", 0.0)))
        & (depth_span_rel >= float(getattr(config, "gmvc_profile_min_depth_span_rel", 0.0)))
    )
    profile_weight = torch.where(track_valid[local_track], weight, torch.zeros_like(weight))
    profile_loss_mode = str(getattr(config, "gmvc_profile_loss_mode", "charbonnier"))
    profile_track_balanced = bool(getattr(config, "gmvc_profile_track_balanced", False))
    irls_weight = torch.ones_like(profile_weight)

    if profile_weight.sum() > 0:
        numerator0 = _scatter_sum(
            profile_weight[:, None] * transmission * (gt - backscatter),
            local_track,
            sampled_track_count,
        )
        denominator0 = _scatter_sum(
            profile_weight[:, None] * transmission.square(),
            local_track,
            sampled_track_count,
        )
        j0 = numerator0 / (denominator0 + eps)
        if profile_loss_mode == "irls_l2":
            pred0 = j0.detach()[local_track] * transmission.detach() + backscatter.detach()
            residual_norm = torch.linalg.norm(pred0 - gt.detach(), dim=-1)
            delta = max(float(getattr(config, "gmvc_profile_irls_delta", 0.03)), eps)
            irls_weight = (delta / torch.sqrt(residual_norm.square() + delta * delta)).detach()
            irls_weight = irls_weight.clamp_max(float(getattr(config, "gmvc_profile_irls_max_weight", 1.0)))
            solve_weight = profile_weight * irls_weight
        else:
            solve_weight = profile_weight
        numerator = _scatter_sum(solve_weight[:, None] * transmission * (gt - backscatter), local_track, sampled_track_count)
        denominator = _scatter_sum(solve_weight[:, None] * transmission.square(), local_track, sampled_track_count)
        j_star = numerator / (denominator + eps)
        j_star_for_loss = j_star.detach() if bool(getattr(config, "gmvc_profile_detach_j_star", True)) else j_star
        pred = j_star_for_loss[local_track] * transmission + backscatter
        if profile_loss_mode == "irls_l2":
            profile_loss_values = 0.5 * (pred - gt).square()
            loss_weight = solve_weight
        else:
            profile_loss_values = charbonnier_loss(
                pred - gt,
                eps=float(getattr(config, "gmvc_charbonnier_eps", 1e-6)),
            )
            loss_weight = profile_weight
        if profile_track_balanced:
            profile_loss = _track_balanced_mean(
                profile_loss_values,
                loss_weight,
                local_track,
                sampled_track_count,
                eps,
            )
        else:
            profile_loss = _weighted_mean(profile_loss_values, loss_weight, eps)
        j_obs = (gt.detach() - backscatter.detach()) / (transmission.detach() + eps)
        j_center = j_star.detach()[local_track]
        j_variance_values = (j_obs - j_center).abs()
        if profile_track_balanced:
            j_variance = _track_balanced_mean(
                j_variance_values,
                profile_weight.detach(),
                local_track,
                sampled_track_count,
                eps,
            )
        else:
            j_variance = _weighted_mean(j_variance_values, profile_weight.detach(), eps)
    else:
        profile_loss = zero
        j_variance = zero
        j_star = gt_img.new_zeros((sampled_track_count, 3))

    object_loss = zero
    object_valid_observation_fraction = zero
    object_valid_tracks = zero
    object_weight_sum = zero
    object_source_available = zero
    if object_phase and outputs is not None:
        camera_key = _camera_key(outputs)
        intrinsic_source_key = str(getattr(config, "gmvc_v3_object_source", "J_proxy_raw"))
        intrinsic_source = outputs.get(intrinsic_source_key)
        if camera_key is not None and intrinsic_source is not None:
            object_source_available = gt_img.new_tensor(1.0)
            row_camera = observations["camera_index"][rows_cpu].to(device=device).long()
            current_camera = int(camera_key)
            current_mask = row_camera == current_camera
            object_track_valid = (
                track_valid
                & (hessian_scalar >= float(getattr(config, "gmvc_object_min_hessian", 0.0)))
                & (depth_span_rel >= float(getattr(config, "gmvc_object_min_depth_span_rel", 0.05)))
            )
            j_target = j_star.detach()[local_track]
            j_valid = (
                torch.isfinite(j_target).all(dim=-1)
                & (j_target >= float(getattr(config, "gmvc_object_j_clamp_min", -0.1))).all(dim=-1)
                & (j_target <= float(getattr(config, "gmvc_object_j_clamp_max", 1.1))).all(dim=-1)
            )
            object_weight = torch.where(
                current_mask & object_track_valid[local_track] & j_valid,
                profile_weight.detach(),
                torch.zeros_like(profile_weight),
            )
            if object_weight.sum() > 0:
                xy_abs = observations["xy"][rows_cpu].to(device=device, dtype=dtype)
                intrinsic_sample = _sample_hwc(intrinsic_source, xy_abs)
                valid_intrinsic = torch.isfinite(intrinsic_sample).all(dim=-1)
                object_weight = torch.where(valid_intrinsic, object_weight, torch.zeros_like(object_weight))
                if object_weight.sum() > 0:
                    object_values = charbonnier_loss(
                        intrinsic_sample - j_target,
                        eps=float(getattr(config, "gmvc_charbonnier_eps", 1e-6)),
                    )
                    if bool(getattr(config, "gmvc_object_track_balanced", True)):
                        object_loss = _track_balanced_mean(
                            object_values,
                            object_weight,
                            local_track,
                            sampled_track_count,
                            eps,
                        )
                    else:
                        object_loss = _weighted_mean(object_values, object_weight, eps)
                    object_valid_tracks = (
                        _scatter_sum((object_weight > 0).to(dtype)[:, None], local_track, sampled_track_count)
                        .reshape(-1)
                        .gt(0)
                        .float()
                        .sum()
                    )
                    object_valid_observation_fraction = (object_weight > 0).float().mean().detach()
                    object_weight_sum = object_weight.detach().sum()

    closure_signal_floor = max(float(getattr(config, "gmvc_closure_signal_floor", 0.03)), eps)
    near_idx, far_idx = _compute_v2_pair_indices(local_track, depth.reshape(-1), weight, sampled_track_count)
    if int(near_idx.numel()) > 0:
        bank_t, bank_b = _medium_terms(depth.detach(), bank_attn.detach(), bank_bs.detach(), bank_binf.detach())
        left0 = (gt.detach()[near_idx] - bank_b[near_idx]) * bank_t[far_idx]
        right0 = (gt.detach()[far_idx] - bank_b[far_idx]) * bank_t[near_idx]
        fixed_denom = (left0.abs() + right0.abs()).clamp_min(closure_signal_floor)

        left = (gt.detach()[near_idx] - backscatter[near_idx]) * transmission[far_idx]
        right = (gt.detach()[far_idx] - backscatter[far_idx]) * transmission[near_idx]
        closure_delta = left - right
        closure_norm = closure_delta / fixed_denom.clamp_min(eps)
        pair_weight = torch.sqrt(weight[near_idx].clamp_min(0.0) * weight[far_idx].clamp_min(0.0))
        closure_loss = _weighted_mean(
            charbonnier_loss(closure_norm, eps=float(getattr(config, "gmvc_charbonnier_eps", 1e-6))),
            pair_weight,
            eps,
        )
        closure_l1 = _weighted_mean(closure_delta.detach().abs(), pair_weight.detach(), eps)
        closure_norm_l1 = _weighted_mean(closure_norm.detach().abs(), pair_weight.detach(), eps)
    else:
        closure_loss = zero
        closure_l1 = zero
        closure_norm_l1 = zero
        pair_weight = gt_img.new_empty((0,))

    start = int(getattr(config, "gmvc_start_step", 10000))
    stop = int(getattr(config, "gmvc_stop_step", 15000))
    ramp = int(getattr(config, "gmvc_ramp_steps", 500))
    lambda_profile = _ramped_weight(float(getattr(config, "lambda_gmvc_profile", 0.0)), step, start, ramp, stop)
    lambda_symmetric_closure = _ramped_weight(
        float(getattr(config, "lambda_gmvc_symmetric_closure", 0.0)),
        step,
        start,
        ramp,
        stop,
    )
    lambda_object = _ramped_weight(float(getattr(config, "lambda_gmvc_object", 0.0)), step, start, ramp, stop)
    if bool(getattr(config, "gmvc_v3_enabled", False)):
        if object_phase:
            lambda_profile = 0.0
            lambda_symmetric_closure = 0.0
        else:
            lambda_object = 0.0
    transmission_scalar = transmission.detach().mean(dim=-1)
    residual_abs = torch.cat(
        [
            torch.log(medium_attn.detach().clamp_min(eps)).abs(),
            torch.log(medium_bs.detach().clamp_min(eps)).abs(),
            _logit_from_unit(b_inf.detach(), eps).abs(),
        ],
        dim=-1,
    )

    target_drift_mean = zero.detach()
    target_drift_p95 = zero.detach()
    target_drift_count = zero.detach()
    medium_attn_delta_mean = zero.detach()
    medium_attn_delta_p95 = zero.detach()
    medium_bs_delta_mean = zero.detach()
    medium_bs_delta_p95 = zero.detach()
    b_inf_delta_mean = zero.detach()
    b_inf_delta_p95 = zero.detach()
    transmission_delta_mean = zero.detach()
    transmission_delta_p95 = zero.detach()
    medium_delta_count = zero.detach()

    if state is not None:
        with torch.no_grad():
            sampled_track_ids = sampled_track_ids_cpu.long()
            if int(sampled_track_ids.numel()) > 0:
                cache_size = max(
                    int(observations.get("track_ids", torch.empty(0)).max().item()) + 1
                    if int(observations.get("track_ids", torch.empty(0)).numel()) > 0
                    else 0,
                    int(sampled_track_ids.max().item()) + 1,
                )
                key = "gmvc_v2_prev_j_star_by_track"
                previous = state.get(key)
                if previous is None or previous.ndim != 2 or previous.shape[0] < cache_size:
                    new_previous = torch.full((cache_size, 3), float("nan"), dtype=torch.float32)
                    if previous is not None and previous.ndim == 2:
                        rows_to_copy = min(int(previous.shape[0]), cache_size)
                        cols_to_copy = min(int(previous.shape[1]), 3)
                        new_previous[:rows_to_copy, :cols_to_copy] = previous[:rows_to_copy, :cols_to_copy].cpu()
                    previous = new_previous
                    state[key] = previous
                j_star_cpu = j_star.detach().float().cpu()
                valid_track = track_valid.detach().cpu() & torch.isfinite(j_star_cpu).all(dim=-1)
                prev_values = previous[sampled_track_ids]
                valid_prev = valid_track & torch.isfinite(prev_values).all(dim=-1)
                if bool(valid_prev.any()):
                    drift = (j_star_cpu[valid_prev] - prev_values[valid_prev]).abs().mean(dim=-1)
                    target_drift_mean = gt_img.new_tensor(float(drift.mean().item()))
                    target_drift_p95 = gt_img.new_tensor(float(torch.quantile(drift, 0.95).item()))
                    target_drift_count = gt_img.new_tensor(float(drift.numel()))
                if bool(valid_track.any()):
                    previous[sampled_track_ids[valid_track]] = j_star_cpu[valid_track]

            row_ids = rows_cpu.long()
            if int(row_ids.numel()) > 0:
                obs_count_total = int(observations["gt"].shape[0])

                def _row_delta_cache(name: str, current: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
                    current_cpu = current.detach().float().cpu()
                    key = f"gmvc_v2_prev_{name}_by_row"
                    previous = state.get(key)
                    channels = int(current_cpu.shape[-1])
                    if previous is None or previous.ndim != 2 or previous.shape[0] < obs_count_total or previous.shape[1] != channels:
                        new_previous = torch.full((obs_count_total, channels), float("nan"), dtype=torch.float32)
                        if previous is not None and previous.ndim == 2:
                            rows_to_copy = min(int(previous.shape[0]), obs_count_total)
                            cols_to_copy = min(int(previous.shape[1]), channels)
                            new_previous[:rows_to_copy, :cols_to_copy] = previous[:rows_to_copy, :cols_to_copy].cpu()
                        previous = new_previous
                        state[key] = previous
                    prev_values = previous[row_ids]
                    valid = torch.isfinite(prev_values).all(dim=-1) & torch.isfinite(current_cpu).all(dim=-1)
                    if bool(valid.any()):
                        delta = (current_cpu[valid] - prev_values[valid]).abs().mean(dim=-1)
                        mean = gt_img.new_tensor(float(delta.mean().item()))
                        p95 = gt_img.new_tensor(float(torch.quantile(delta, 0.95).item()))
                        count = gt_img.new_tensor(float(delta.numel()))
                    else:
                        mean = zero.detach()
                        p95 = zero.detach()
                        count = zero.detach()
                    current_valid = torch.isfinite(current_cpu).all(dim=-1)
                    if bool(current_valid.any()):
                        previous[row_ids[current_valid]] = current_cpu[current_valid]
                    return mean, p95, count

                medium_attn_delta_mean, medium_attn_delta_p95, medium_delta_count = _row_delta_cache(
                    "medium_attn",
                    medium_attn,
                )
                medium_bs_delta_mean, medium_bs_delta_p95, _ = _row_delta_cache("medium_bs", medium_bs)
                b_inf_delta_mean, b_inf_delta_p95, _ = _row_delta_cache("b_inf", b_inf)
                transmission_delta_mean, transmission_delta_p95, _ = _row_delta_cache("transmission", transmission)

    losses = {
        "gmvc_profile_loss": profile_loss * lambda_profile,
        "gmvc_symmetric_closure_loss": closure_loss * lambda_symmetric_closure,
        "gmvc_object_loss": object_loss * lambda_object,
    }
    metrics = {
        "gmvc_v2_available_tracks": gt_img.new_tensor(float(track_count_total)),
        "gmvc_v2_sampled_tracks": gt_img.new_tensor(float(sampled_track_count)),
        "gmvc_v2_sampled_observations": gt_img.new_tensor(float(rows_cpu.numel())),
        "gmvc_v2_valid_tracks": track_valid.detach().float().sum(),
        "gmvc_v2_valid_observation_fraction": (profile_weight > 0).detach().float().mean(),
        "gmvc_profile_raw": profile_loss.detach(),
        "gmvc_profile_loss_mode_irls": gt_img.new_tensor(float(profile_loss_mode == "irls_l2")),
        "gmvc_profile_track_balanced": gt_img.new_tensor(float(profile_track_balanced)),
        "gmvc_profile_hessian_p50": _safe_quantile(hessian_scalar, 0.50, zero),
        "gmvc_profile_hessian_p05": _safe_quantile(hessian_scalar, 0.05, zero),
        "gmvc_profile_transmission_span_p50": _safe_quantile(t_max - t_min, 0.50, zero),
        "gmvc_profile_depth_span_rel_p50": _safe_quantile(depth_span_rel, 0.50, zero),
        "gmvc_profile_irls_weight_mean": irls_weight.detach().mean() if irls_weight.numel() else zero.detach(),
        "gmvc_profile_j_variance_l1": j_variance.detach(),
        "gmvc_profile_j_star_mean": j_star.detach().mean() if j_star.numel() else zero.detach(),
        "gmvc_profile_j_star_p05": _safe_quantile(j_star, 0.05, zero),
        "gmvc_profile_j_star_p95": _safe_quantile(j_star, 0.95, zero),
        "gmvc_profile_j_star_drift_l1_mean": target_drift_mean.detach(),
        "gmvc_profile_j_star_drift_l1_p95": target_drift_p95.detach(),
        "gmvc_profile_j_star_drift_count": target_drift_count.detach(),
        "gmvc_symmetric_closure_raw": closure_loss.detach(),
        "gmvc_object_raw": object_loss.detach(),
        "gmvc_object_source_available": object_source_available.detach(),
        "gmvc_object_valid_tracks": object_valid_tracks.detach(),
        "gmvc_object_valid_observation_fraction": object_valid_observation_fraction.detach(),
        "gmvc_object_weight_sum": object_weight_sum.detach(),
        "gmvc_v3_object_phase": gt_img.new_tensor(float(object_phase)),
        "gmvc_symmetric_closure_l1": closure_l1.detach(),
        "gmvc_symmetric_closure_norm_l1": closure_norm_l1.detach(),
        "gmvc_symmetric_closure_pairs": gt_img.new_tensor(float(pair_weight.numel())),
        "gmvc_v2_transmission_p05": _safe_quantile(transmission_scalar, 0.05, zero),
        "gmvc_v2_transmission_p50": _safe_quantile(transmission_scalar, 0.50, zero),
        "gmvc_v2_transmission_p95": _safe_quantile(transmission_scalar, 0.95, zero),
        "gmvc_v2_backscatter_mean": backscatter.detach().mean(),
        "gmvc_v2_residual_abs_p95": _safe_quantile(residual_abs, 0.95, zero),
        "gmvc_medium_attn_delta_l1_mean": medium_attn_delta_mean.detach(),
        "gmvc_medium_attn_delta_l1_p95": medium_attn_delta_p95.detach(),
        "gmvc_medium_bs_delta_l1_mean": medium_bs_delta_mean.detach(),
        "gmvc_medium_bs_delta_l1_p95": medium_bs_delta_p95.detach(),
        "gmvc_b_inf_delta_l1_mean": b_inf_delta_mean.detach(),
        "gmvc_b_inf_delta_l1_p95": b_inf_delta_p95.detach(),
        "gmvc_transmission_delta_l1_mean": transmission_delta_mean.detach(),
        "gmvc_transmission_delta_l1_p95": transmission_delta_p95.detach(),
        "gmvc_medium_delta_count": medium_delta_count.detach(),
        "gmvc_lambda_profile": gt_img.new_tensor(float(lambda_profile)),
        "gmvc_lambda_symmetric_closure": gt_img.new_tensor(float(lambda_symmetric_closure)),
        "gmvc_lambda_object": gt_img.new_tensor(float(lambda_object)),
    }
    return losses, metrics


def _gmvc_v2_requested(config: Any) -> bool:
    return bool(
        getattr(config, "gmvc_v2_enabled", False)
        or getattr(config, "gmvc_v3_enabled", False)
        or float(getattr(config, "lambda_gmvc_profile", 0.0)) > 0.0
        or float(getattr(config, "lambda_gmvc_symmetric_closure", 0.0)) > 0.0
        or float(getattr(config, "lambda_gmvc_object", 0.0)) > 0.0
    )


def compute_gmvc_training_terms(
    outputs: Dict[str, Tensor],
    gt_img: Tensor,
    bank: Dict[str, Any],
    step: int,
    config: Any,
    state: Optional[Dict[str, Tensor]] = None,
    medium_query_fn: Optional[Callable[[Tensor, Tensor, Tensor, Optional[Tensor]], Dict[str, Tensor]]] = None,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """Compute current-view GMVC losses against a detached offline track bank."""

    device = gt_img.device
    eps = float(getattr(config, "gmvc_eps", 1e-4))
    v2_requested = _gmvc_v2_requested(config)
    camera_key = _camera_key(outputs)
    zero = gt_img.new_zeros(())
    if camera_key is None:
        if v2_requested:
            return _compute_gmvc_v2_terms(
                outputs=outputs,
                gt_img=gt_img,
                bank=bank,
                step=step,
                config=config,
                state=state,
                medium_query_fn=medium_query_fn,
            )
        return {}, {"gmvc_available_tracks": zero, "gmvc_sampled_tracks": zero}

    per_camera = bank.get("per_camera", {})
    entry = per_camera.get(camera_key)
    if entry is None:
        if v2_requested:
            return _compute_gmvc_v2_terms(
                outputs=outputs,
                gt_img=gt_img,
                bank=bank,
                step=step,
                config=config,
                state=state,
                medium_query_fn=medium_query_fn,
            )
        return {}, {"gmvc_available_tracks": zero, "gmvc_sampled_tracks": zero}

    xy = entry["xy"].to(device=device, dtype=gt_img.dtype)
    count = int(xy.shape[0])
    if count == 0:
        if v2_requested:
            return _compute_gmvc_v2_terms(
                outputs=outputs,
                gt_img=gt_img,
                bank=bank,
                step=step,
                config=config,
                state=state,
                medium_query_fn=medium_query_fn,
            )
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
        if v2_requested:
            return _compute_gmvc_v2_terms(
                outputs=outputs,
                gt_img=gt_img,
                bank=bank,
                step=step,
                config=config,
                state=state,
                medium_query_fn=medium_query_fn,
            )
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
        if v2_requested:
            return _compute_gmvc_v2_terms(
                outputs=outputs,
                gt_img=gt_img,
                bank=bank,
                step=step,
                config=config,
                state=state,
                medium_query_fn=medium_query_fn,
            )
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
    if v2_requested:
        v2_losses, v2_metrics = _compute_gmvc_v2_terms(
            outputs=outputs,
            gt_img=gt_img,
            bank=bank,
            step=step,
            config=config,
            state=state,
            medium_query_fn=medium_query_fn,
        )
        losses.update(v2_losses)
        metrics.update(v2_metrics)
    return losses, metrics
