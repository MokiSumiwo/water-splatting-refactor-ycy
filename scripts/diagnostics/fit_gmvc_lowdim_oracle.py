#!/usr/bin/env python
"""Fit low-dimensional GMVC physical oracle models on geometry-anchored tracks."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import Tensor

from nerfstudio.utils.eval_utils import eval_setup
from water_splatting.medium_calibration import GMVCTrackConfig, render_gmvc_views
from water_splatting.medium_calibration.gmvc_tracks import (
    _sample_hwc,
    _sample_observations,
    _sample_source_pixels,
    _selected_target_indices,
    project_world,
    unproject_pixels,
)


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _nearest_rank(values: Tensor, q: float) -> float:
    values = values.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return 0.0
    rank = max(1, min(values.numel(), math.ceil(float(q) * values.numel())))
    return float(values.kthvalue(rank).values.item())


def _stats(values: Tensor) -> Dict[str, float]:
    values = values.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "p50": _nearest_rank(values, 0.50),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "max": float(values.max().item()),
    }


def _append_obs(
    obs_lists: List[List[Dict[str, Tensor]]],
    local_idx: int,
    obs: Dict[str, Tensor],
    row_idx: int,
    view_camera_index: int,
    view_image_index: int,
) -> None:
    payload = {key: value[row_idx].detach().cpu() for key, value in obs.items()}
    payload["camera_index"] = torch.tensor(int(view_camera_index), dtype=torch.long)
    payload["image_index"] = torch.tensor(int(view_image_index), dtype=torch.long)
    obs_lists[local_idx].append(payload)


def _track_weighted_payload(
    observations: List[Dict[str, Tensor]], cfg: GMVCTrackConfig
) -> Tuple[bool, Dict[str, Tensor], Dict[str, float]]:
    depth = torch.stack([obs["depth"].reshape(()) for obs in observations]).float()
    gt = torch.stack([obs["gt"].float() for obs in observations])
    alpha = torch.stack([obs["alpha"].reshape(()) for obs in observations]).float()
    depth_err = torch.stack([obs["depth_rel_error"].reshape(()) for obs in observations]).float()
    depth_std = torch.stack([obs["depth_std_relative"].reshape(()) for obs in observations]).float()
    transmission = torch.stack([obs["transmission"].float() for obs in observations])
    t_scalar = transmission.mean(dim=-1)
    medium_attn = torch.stack([obs["medium_attn"].float() for obs in observations])
    medium_bs = torch.stack([obs["medium_bs"].float() for obs in observations])
    b_inf = torch.stack([obs["b_inf"].float() for obs in observations])
    camera_index = torch.stack([obs["camera_index"].long().reshape(()) for obs in observations])
    image_index = torch.stack([obs["image_index"].long().reshape(()) for obs in observations])

    span = depth.max() - depth.min()
    relative_span = float((span / depth.median().clamp_min(float(cfg.eps))).item())
    if relative_span < cfg.relative_depth_span:
        return False, {}, {"relative_depth_span": relative_span}

    valid = (
        torch.isfinite(depth)
        & torch.isfinite(gt).all(dim=-1)
        & (depth > 0.0)
        & (alpha >= cfg.alpha_threshold)
        & (depth_err <= cfg.depth_rel_threshold)
        & (depth_std <= cfg.depth_std_rel_threshold)
        & (t_scalar >= cfg.transmission_min)
    )
    w_alpha = ((alpha - cfg.alpha_threshold) / max(1.0 - cfg.alpha_threshold, cfg.eps)).clamp(0.0, 1.0)
    w_depth = torch.exp(-depth_err / max(cfg.depth_error_sigma, cfg.eps)).clamp(0.0, 1.0)
    w_t = ((t_scalar - cfg.transmission_min) / max(1.0 - cfg.transmission_min, cfg.eps)).clamp(0.0, 1.0)
    w_span = min(max(relative_span / max(cfg.span_weight_high, cfg.eps), 0.0), 1.0)
    weights = torch.where(valid, (w_alpha * w_depth * w_t * w_span).float(), torch.zeros_like(t_scalar))
    keep = weights > 0.0
    if int(keep.sum().item()) < cfg.min_views:
        return False, {}, {
            "relative_depth_span": relative_span,
            "valid_observation_count": float(keep.sum().item()),
        }

    payload = {
        "gt": gt[keep],
        "depth": depth[keep],
        "weight": weights[keep],
        "camera_index": camera_index[keep],
        "image_index": image_index[keep],
        "medium_attn": medium_attn[keep],
        "medium_bs": medium_bs[keep],
        "b_inf": b_inf[keep],
    }
    meta = {
        "relative_depth_span": relative_span,
        "track_length": float(keep.sum().item()),
        "weight_mean": float(weights[keep].mean().item()),
    }
    return True, payload, meta


def build_oracle_dataset(
    pipeline: Any,
    split: str,
    max_images: int,
    cfg: GMVCTrackConfig,
    max_tracks: int,
) -> Dict[str, Any]:
    views = render_gmvc_views(pipeline, split, max_images)
    obs_columns: Dict[str, List[Tensor]] = {
        "track_id": [],
        "gt": [],
        "depth": [],
        "weight": [],
        "camera_index": [],
        "image_index": [],
        "medium_attn": [],
        "medium_bs": [],
        "b_inf": [],
    }
    counters: Dict[str, int] = {
        "source_valid_pixels_total": 0,
        "sampled_source_tracks": 0,
        "candidate_tracks_len_ge_2": 0,
        "accepted_tracks": 0,
        "accepted_observations": 0,
        "target_projection_attempts": 0,
        "invalid_out_of_bounds_count": 0,
        "invalid_depth_count": 0,
        "invalid_alpha_count": 0,
        "invalid_depth_std_count": 0,
        "invalid_low_T_count": 0,
    }
    track_metas: List[Dict[str, float]] = []

    for source_idx, source_view in enumerate(views):
        source_xy, source_valid_count = _sample_source_pixels(source_view, cfg, source_idx)
        counters["source_valid_pixels_total"] += int(source_valid_count)
        counters["sampled_source_tracks"] += int(source_xy.shape[0])
        if source_xy.numel() == 0:
            continue
        source_depth = _sample_hwc(source_view.depth, source_xy).reshape(-1)
        points_world = unproject_pixels(source_view, source_xy, source_depth)
        source_obs = _sample_observations(source_view, source_xy, torch.zeros_like(source_depth))
        obs_lists: List[List[Dict[str, Tensor]]] = [[] for _ in range(source_xy.shape[0])]
        for local_idx in range(source_xy.shape[0]):
            _append_obs(
                obs_lists,
                local_idx,
                source_obs,
                local_idx,
                source_view.camera_index,
                source_view.image_index,
            )

        for target_idx in _selected_target_indices(source_idx, len(views), cfg):
            target_view = views[target_idx]
            xy_target, z_projected = project_world(target_view, points_world, eps=cfg.eps)
            in_bounds = (
                (z_projected > cfg.eps)
                & (xy_target[:, 0] >= cfg.edge_margin)
                & (xy_target[:, 0] < target_view.width - cfg.edge_margin)
                & (xy_target[:, 1] >= cfg.edge_margin)
                & (xy_target[:, 1] < target_view.height - cfg.edge_margin)
            )
            counters["target_projection_attempts"] += int(xy_target.shape[0])
            counters["invalid_out_of_bounds_count"] += int((~in_bounds).sum().item())
            if not in_bounds.any():
                continue

            local_indices = torch.nonzero(in_bounds, as_tuple=False).reshape(-1)
            target_xy = xy_target[local_indices]
            projected_depth = z_projected[local_indices]
            target_depth = _sample_hwc(target_view.depth, target_xy).reshape(-1)
            depth_rel_error = (target_depth - projected_depth).abs() / target_depth.clamp_min(cfg.eps)
            target_obs = _sample_observations(target_view, target_xy, depth_rel_error)
            alpha = target_obs["alpha"].reshape(-1)
            depth_std = target_obs["depth_std_relative"].reshape(-1)
            t_mean = target_obs["transmission"].mean(dim=-1)
            depth_valid = torch.isfinite(target_depth) & (target_depth > 0) & (depth_rel_error <= cfg.depth_rel_threshold)
            alpha_valid = alpha >= cfg.alpha_threshold
            depth_std_valid = depth_std <= cfg.depth_std_rel_threshold
            t_valid = t_mean >= cfg.transmission_min
            counters["invalid_depth_count"] += int((~depth_valid).sum().item())
            counters["invalid_alpha_count"] += int((depth_valid & ~alpha_valid).sum().item())
            counters["invalid_depth_std_count"] += int((depth_valid & alpha_valid & ~depth_std_valid).sum().item())
            counters["invalid_low_T_count"] += int((depth_valid & alpha_valid & depth_std_valid & ~t_valid).sum().item())
            final_valid = depth_valid & alpha_valid & depth_std_valid & t_valid

            for row_idx in torch.nonzero(final_valid, as_tuple=False).reshape(-1).tolist():
                _append_obs(
                    obs_lists,
                    int(local_indices[row_idx].item()),
                    target_obs,
                    int(row_idx),
                    target_view.camera_index,
                    target_view.image_index,
                )

        for observations in obs_lists:
            if len(observations) < 2:
                continue
            counters["candidate_tracks_len_ge_2"] += 1
            accepted, payload, meta = _track_weighted_payload(observations, cfg)
            if not accepted:
                continue
            track_id = counters["accepted_tracks"]
            obs_count = int(payload["gt"].shape[0])
            obs_columns["track_id"].append(torch.full((obs_count,), track_id, dtype=torch.long))
            for key in ["gt", "depth", "weight", "camera_index", "image_index", "medium_attn", "medium_bs", "b_inf"]:
                obs_columns[key].append(payload[key])
            counters["accepted_tracks"] += 1
            counters["accepted_observations"] += obs_count
            track_metas.append(meta)
            if max_tracks > 0 and counters["accepted_tracks"] >= max_tracks:
                break
        if max_tracks > 0 and counters["accepted_tracks"] >= max_tracks:
            break

    if counters["accepted_tracks"] == 0:
        raise RuntimeError("No accepted GMVC oracle tracks. Relax filters or increase sampled views.")

    stacked = {
        key: torch.cat(value, dim=0) if value else torch.empty((0,), dtype=torch.float32)
        for key, value in obs_columns.items()
    }
    for key in ["gt", "medium_attn", "medium_bs", "b_inf"]:
        stacked[key] = stacked[key].float().reshape(-1, 3)
    stacked["depth"] = stacked["depth"].float().reshape(-1, 1)
    stacked["weight"] = stacked["weight"].float().reshape(-1)
    stacked["track_id"] = stacked["track_id"].long().reshape(-1)
    stacked["camera_index_raw"] = stacked.pop("camera_index").long().reshape(-1)
    stacked["image_index"] = stacked["image_index"].long().reshape(-1)

    unique_cameras = torch.unique(stacked["camera_index_raw"], sorted=True)
    camera_map = {int(value.item()): idx for idx, value in enumerate(unique_cameras)}
    stacked["camera_id"] = torch.tensor(
        [camera_map[int(value.item())] for value in stacked["camera_index_raw"]], dtype=torch.long
    )
    summary = {
        "view_count": len(views),
        "counters": counters,
        "camera_index_values": [int(value.item()) for value in unique_cameras],
        "track_length": _stats(torch.tensor([meta["track_length"] for meta in track_metas])),
        "relative_depth_span": _stats(torch.tensor([meta["relative_depth_span"] for meta in track_metas])),
        "weight": _stats(stacked["weight"]),
        "depth": _stats(stacked["depth"]),
        "m1_medium_attn_mean_rgb": [
            float(v) for v in (stacked["medium_attn"] * stacked["weight"][:, None]).sum(dim=0)
            / stacked["weight"].sum().clamp_min(float(cfg.eps))
        ],
        "m1_medium_bs_mean_rgb": [
            float(v) for v in (stacked["medium_bs"] * stacked["weight"][:, None]).sum(dim=0)
            / stacked["weight"].sum().clamp_min(float(cfg.eps))
        ],
        "m1_b_inf_mean_rgb": [
            float(v) for v in (stacked["b_inf"] * stacked["weight"][:, None]).sum(dim=0)
            / stacked["weight"].sum().clamp_min(float(cfg.eps))
        ],
    }
    return {"observations": stacked, "summary": summary, "views": [view.metadata() for view in views]}


def _weighted_track_init(obs: Dict[str, Tensor], track_count: int, eps: float) -> Tensor:
    gt = obs["gt"]
    weights = obs["weight"]
    track_id = obs["track_id"]
    accum = torch.zeros((track_count, 3), dtype=torch.float32, device=gt.device)
    denom = torch.zeros((track_count, 1), dtype=torch.float32, device=gt.device)
    accum.index_add_(0, track_id, gt * weights[:, None])
    denom.index_add_(0, track_id, weights[:, None])
    return accum / denom.clamp_min(float(eps))


def _logit_from_unit(value: Tensor, eps: float = 1e-4) -> Tensor:
    value = value.clamp(float(eps), 1.0 - float(eps))
    return torch.log(value) - torch.log1p(-value)


class LowDimOracle(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        obs: Dict[str, Tensor],
        track_count: int,
        camera_count: int,
        j_min: float,
        j_max: float,
        eps: float,
        beta_residual_scale: float,
        binf_residual_scale: float,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.j_min = float(j_min)
        self.j_max = float(j_max)
        self.eps = float(eps)
        self.beta_residual_scale = float(beta_residual_scale)
        self.binf_residual_scale = float(binf_residual_scale)
        weight = obs["weight"]
        denom = weight.sum().clamp_min(self.eps)
        attn_init = (obs["medium_attn"] * weight[:, None]).sum(dim=0) / denom
        bs_init = (obs["medium_bs"] * weight[:, None]).sum(dim=0) / denom
        binf_init = (obs["b_inf"] * weight[:, None]).sum(dim=0) / denom
        j_init = _weighted_track_init(obs, track_count, eps).clamp(self.j_min + 1e-4, self.j_max - 1e-4)
        j_unit = (j_init - self.j_min) / max(self.j_max - self.j_min, self.eps)

        self.log_beta_d_center = torch.nn.Parameter(torch.log(attn_init.clamp_min(self.eps)))
        self.log_beta_b_center = torch.nn.Parameter(torch.log(bs_init.clamp_min(self.eps)))
        self.b_inf_center_raw = torch.nn.Parameter(_logit_from_unit(binf_init))
        self.j_raw = torch.nn.Parameter(_logit_from_unit(j_unit))
        if model_name == "O1":
            self.delta_log_beta_d = torch.nn.Parameter(torch.zeros((camera_count, 3), dtype=torch.float32))
            self.delta_log_beta_b = torch.nn.Parameter(torch.zeros((camera_count, 3), dtype=torch.float32))
            self.delta_b_inf_raw = torch.nn.Parameter(torch.zeros((camera_count, 3), dtype=torch.float32))
        else:
            self.register_parameter("delta_log_beta_d", None)
            self.register_parameter("delta_log_beta_b", None)
            self.register_parameter("delta_b_inf_raw", None)

    def j(self) -> Tensor:
        return self.j_min + (self.j_max - self.j_min) * torch.sigmoid(self.j_raw)

    def medium(self, camera_id: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.model_name == "O1":
            log_beta_d = self.log_beta_d_center[None] + self.beta_residual_scale * torch.tanh(
                self.delta_log_beta_d[camera_id]
            )
            log_beta_b = self.log_beta_b_center[None] + self.beta_residual_scale * torch.tanh(
                self.delta_log_beta_b[camera_id]
            )
            b_inf_raw = self.b_inf_center_raw[None] + self.binf_residual_scale * torch.tanh(
                self.delta_b_inf_raw[camera_id]
            )
            return torch.exp(log_beta_d), torch.exp(log_beta_b), torch.sigmoid(b_inf_raw)
        return (
            torch.exp(self.log_beta_d_center)[None].expand(camera_id.shape[0], 3),
            torch.exp(self.log_beta_b_center)[None].expand(camera_id.shape[0], 3),
            torch.sigmoid(self.b_inf_center_raw)[None].expand(camera_id.shape[0], 3),
        )

    def predict(self, obs: Dict[str, Tensor], indices: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if indices is None:
            gt = obs["gt"]
            depth = obs["depth"]
            track_id = obs["track_id"]
            camera_id = obs["camera_id"]
        else:
            gt = obs["gt"][indices]
            depth = obs["depth"][indices]
            track_id = obs["track_id"][indices]
            camera_id = obs["camera_id"][indices]
        beta_d, beta_b, b_inf = self.medium(camera_id)
        transmission = torch.exp(-(beta_d * depth).clamp_min(0.0))
        backscatter = b_inf * (1.0 - torch.exp(-(beta_b * depth).clamp_min(0.0)))
        pred = self.j()[track_id] * transmission + backscatter
        return pred, transmission, backscatter, gt

    def residual_budget(self) -> Dict[str, Any]:
        if self.model_name != "O1":
            return {
                "beta_log_residual_abs": {"mean": 0.0, "p95": 0.0, "max": 0.0},
                "b_inf_logit_residual_abs": {"mean": 0.0, "p95": 0.0, "max": 0.0},
                "saturation_ratio_abs_tanh_gt_095": 0.0,
            }
        beta_resid = torch.cat(
            [
                self.beta_residual_scale * torch.tanh(self.delta_log_beta_d.detach()),
                self.beta_residual_scale * torch.tanh(self.delta_log_beta_b.detach()),
            ],
            dim=0,
        ).abs()
        binf_resid = (self.binf_residual_scale * torch.tanh(self.delta_b_inf_raw.detach())).abs()
        tanh_abs = torch.cat(
            [
                torch.tanh(self.delta_log_beta_d.detach()).abs().reshape(-1),
                torch.tanh(self.delta_log_beta_b.detach()).abs().reshape(-1),
                torch.tanh(self.delta_b_inf_raw.detach()).abs().reshape(-1),
            ]
        )
        return {
            "beta_log_residual_abs": _stats(beta_resid),
            "b_inf_logit_residual_abs": _stats(binf_resid),
            "saturation_ratio_abs_tanh_gt_095": float((tanh_abs > 0.95).float().mean().item()),
        }

    def fitted_parameters(self, camera_index_values: List[int]) -> Dict[str, Any]:
        camera_id = torch.arange(len(camera_index_values), dtype=torch.long, device=self.log_beta_d_center.device)
        beta_d, beta_b, b_inf = self.medium(camera_id)
        return {
            "model": self.model_name,
            "beta_d_center_rgb": [float(v) for v in torch.exp(self.log_beta_d_center.detach().cpu())],
            "beta_b_center_rgb": [float(v) for v in torch.exp(self.log_beta_b_center.detach().cpu())],
            "b_inf_center_rgb": [float(v) for v in torch.sigmoid(self.b_inf_center_raw.detach().cpu())],
            "per_camera": [
                {
                    "camera_index": int(raw),
                    "beta_d_rgb": [float(v) for v in beta_d[idx].detach().cpu()],
                    "beta_b_rgb": [float(v) for v in beta_b[idx].detach().cpu()],
                    "b_inf_rgb": [float(v) for v in b_inf[idx].detach().cpu()],
                }
                for idx, raw in enumerate(camera_index_values)
            ],
            "residual_budget": self.residual_budget(),
        }


def _weighted_loss(residual: Tensor, weights: Tensor, charbonnier_eps: float) -> Tensor:
    per_obs = torch.sqrt(residual.square() + float(charbonnier_eps)).mean(dim=-1)
    return (per_obs * weights).sum() / weights.sum().clamp_min(1e-8)


def _make_track_split(track_count: int, train_fraction: float, seed: int) -> Tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(track_count, generator=generator)
    train_count = max(1, min(track_count - 1, int(round(track_count * train_fraction))))
    train_tracks = perm[:train_count]
    heldout_tracks = perm[train_count:]
    if heldout_tracks.numel() == 0:
        heldout_tracks = train_tracks[-1:]
        train_tracks = train_tracks[:-1]
    return train_tracks, heldout_tracks


def _indices_for_tracks(track_id: Tensor, tracks: Tensor) -> Tensor:
    mask = torch.zeros(int(track_id.max().item()) + 1, dtype=torch.bool)
    mask[tracks] = True
    return torch.nonzero(mask[track_id], as_tuple=False).reshape(-1)


def _iter_minibatches(indices: Tensor, batch_size: int, seed: int, step: int) -> Iterable[Tensor]:
    if batch_size <= 0 or indices.numel() <= batch_size:
        yield indices
        return
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(step) * 104729)
    order = indices[torch.randperm(indices.numel(), generator=generator).to(indices.device)]
    for start in range(0, order.numel(), batch_size):
        yield order[start : start + batch_size]


def _weighted_l1_for_indices(model: LowDimOracle, obs: Dict[str, Tensor], indices: Tensor) -> float:
    with torch.no_grad():
        pred, _, _, gt = model.predict(obs, indices)
        weights = obs["weight"][indices]
        value = ((pred - gt).abs().mean(dim=-1) * weights).sum() / weights.sum().clamp_min(1e-8)
    return float(value.detach().cpu().item())


def _fit_model(
    model_name: str,
    obs: Dict[str, Tensor],
    track_count: int,
    camera_count: int,
    train_indices: Tensor,
    args: argparse.Namespace,
) -> Tuple[LowDimOracle, List[Dict[str, float]]]:
    model = LowDimOracle(
        model_name=model_name,
        obs=obs,
        track_count=track_count,
        camera_count=camera_count,
        j_min=args.j_min,
        j_max=args.j_max,
        eps=args.eps,
        beta_residual_scale=args.o1_log_beta_scale,
        binf_residual_scale=args.o1_binf_logit_scale,
    ).to(args.fit_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: List[Dict[str, float]] = []
    for step in range(args.iters):
        total_loss = 0.0
        total_weight = 0.0
        for batch_indices in _iter_minibatches(train_indices, args.batch_size, args.seed, step):
            optimizer.zero_grad(set_to_none=True)
            pred, _, _, gt = model.predict(obs, batch_indices)
            weights = obs["weight"][batch_indices]
            loss = _weighted_loss(pred - gt, weights, args.charbonnier_eps)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * float(weights.sum().detach().cpu().item())
            total_weight += float(weights.sum().detach().cpu().item())
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.iters:
            history.append(
                {
                    "iter": float(step + 1),
                    "weighted_charbonnier": total_loss / max(total_weight, 1e-8),
                    "train_weighted_l1": _weighted_l1_for_indices(model, obs, train_indices),
                }
            )
    return model, history


def _per_track_indices(track_id: Tensor, track_count: int, indices: Tensor) -> List[Tensor]:
    buckets: List[List[int]] = [[] for _ in range(track_count)]
    for idx in indices.detach().cpu().tolist():
        buckets[int(track_id[idx].item())].append(int(idx))
    return [torch.tensor(bucket, dtype=torch.long, device=indices.device) for bucket in buckets if len(bucket) >= 2]


def _pair_metrics(model: LowDimOracle, obs: Dict[str, Tensor], indices: Tensor, eps: float) -> Dict[str, Any]:
    with torch.no_grad():
        track_count = int(obs["track_id"].max().item()) + 1
        track_buckets = _per_track_indices(obs["track_id"], track_count, indices)
        transfer_values: List[Tensor] = []
        closure_values: List[Tensor] = []
        closure_norm_values: List[Tensor] = []
        consensus_recon_values: List[Tensor] = []
        j_var_values: List[Tensor] = []
        pair_weights: List[Tensor] = []
        obs_weights: List[Tensor] = []
        track_weights: List[Tensor] = []
        for bucket in track_buckets:
            pred, transmission, backscatter, gt = model.predict(obs, bucket)
            del pred
            weights = obs["weight"][bucket]
            j_hat = (gt - backscatter) / transmission.clamp_min(float(eps))
            obs_count = int(bucket.numel())
            if obs_count < 2:
                continue
            src = torch.arange(obs_count, device=bucket.device).repeat_interleave(obs_count)
            dst = torch.arange(obs_count, device=bucket.device).repeat(obs_count)
            pair_mask = src != dst
            src = src[pair_mask]
            dst = dst[pair_mask]
            pair_w = torch.sqrt(weights[src] * weights[dst]).clamp_min(0.0)
            pred_dst = j_hat[src] * transmission[dst] + backscatter[dst]
            transfer = (pred_dst - gt[dst]).abs().mean(dim=-1)
            left = (gt[src] - backscatter[src]) * transmission[dst]
            right = (gt[dst] - backscatter[dst]) * transmission[src]
            closure = (left - right).abs().mean(dim=-1)
            closure_norm = (left - right).abs() / (left.abs() + right.abs() + float(eps))
            closure_norm = closure_norm.mean(dim=-1)
            mean_j = (j_hat * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(float(eps))
            consensus_pred = mean_j[None] * transmission + backscatter
            consensus_recon = (consensus_pred - gt).abs().mean(dim=-1)
            j_var = ((j_hat - mean_j[None]).square().mean(dim=-1) * weights).sum() / weights.sum().clamp_min(float(eps))
            transfer_values.append(transfer)
            closure_values.append(closure)
            closure_norm_values.append(closure_norm)
            consensus_recon_values.append(consensus_recon)
            pair_weights.append(pair_w)
            obs_weights.append(weights)
            j_var_values.append(j_var.reshape(1))
            track_weights.append(weights.mean().reshape(1))

        if not transfer_values:
            return {
                "track_count": 0,
                "pair_count": 0,
                "transfer_l1": 0.0,
                "closure_l1": 0.0,
                "closure_norm_l1": 0.0,
                "object_j_variance": 0.0,
            }
        transfer_t = torch.cat(transfer_values)
        closure_t = torch.cat(closure_values)
        closure_norm_t = torch.cat(closure_norm_values)
        consensus_recon_t = torch.cat(consensus_recon_values)
        pair_w_t = torch.cat(pair_weights)
        obs_w_t = torch.cat(obs_weights)
        j_var_t = torch.cat(j_var_values)
        track_w_t = torch.cat(track_weights)
        denom_pair = pair_w_t.sum().clamp_min(float(eps))
        denom_obs = obs_w_t.sum().clamp_min(float(eps))
        denom_track = track_w_t.sum().clamp_min(float(eps))
        return {
            "track_count": len(track_buckets),
            "pair_count": int(transfer_t.numel()),
            "consensus_j_reconstruction_l1": float(
                (consensus_recon_t * obs_w_t).sum().cpu().item() / denom_obs.cpu().item()
            ),
            "transfer_l1": float((transfer_t * pair_w_t).sum().cpu().item() / denom_pair.cpu().item()),
            "closure_l1": float((closure_t * pair_w_t).sum().cpu().item() / denom_pair.cpu().item()),
            "closure_norm_l1": float((closure_norm_t * pair_w_t).sum().cpu().item() / denom_pair.cpu().item()),
            "object_j_variance": float((j_var_t * track_w_t).sum().cpu().item() / denom_track.cpu().item()),
            "transfer_l1_stats": _stats(transfer_t.detach().cpu()),
            "closure_norm_l1_stats": _stats(closure_norm_t.detach().cpu()),
        }


def _evaluate_model(
    model: LowDimOracle,
    obs: Dict[str, Tensor],
    train_indices: Tensor,
    heldout_indices: Tensor,
    eps: float,
) -> Dict[str, Any]:
    train_recon = _weighted_l1_for_indices(model, obs, train_indices)
    heldout_recon = _weighted_l1_for_indices(model, obs, heldout_indices)
    return {
        "reconstruction_weighted_l1": {
            "train": train_recon,
            "heldout": heldout_recon,
        },
        "cross_view": {
            "train": _pair_metrics(model, obs, train_indices, eps),
            "heldout": _pair_metrics(model, obs, heldout_indices, eps),
        },
    }


def _move_obs(obs: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in obs.items()}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    def _update_config(config: Any) -> Any:
        if args.load_step is not None:
            config.load_step = int(args.load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(
        args.load_config,
        eval_num_rays_per_chunk=None,
        test_mode=args.test_mode,
        update_config_callback=_update_config,
    )
    track_cfg = GMVCTrackConfig(
        min_views=args.track_min_views,
        alpha_threshold=args.alpha_threshold,
        depth_rel_threshold=args.depth_rel_threshold,
        depth_std_rel_threshold=args.depth_std_rel_threshold,
        relative_depth_span=args.relative_depth_span,
        transmission_min=args.transmission_min,
        span_weight_high=args.span_weight_high,
        depth_error_sigma=args.depth_error_sigma,
        eps=args.eps,
        j_clamp_min=args.j_min,
        j_clamp_max=args.j_max,
        edge_margin=args.edge_margin,
        samples_per_view=args.samples_per_view,
        seed=args.seed,
        target_neighbor_window=args.target_neighbor_window,
    )
    dataset = build_oracle_dataset(
        pipeline=pipeline,
        split=args.split,
        max_images=args.max_images,
        cfg=track_cfg,
        max_tracks=args.max_tracks,
    )
    obs_cpu = dataset["observations"]
    track_count = int(obs_cpu["track_id"].max().item()) + 1
    camera_count = int(torch.unique(obs_cpu["camera_id"]).numel())
    train_tracks, heldout_tracks = _make_track_split(track_count, args.train_fraction, args.seed)
    train_indices_cpu = _indices_for_tracks(obs_cpu["track_id"], train_tracks)
    heldout_indices_cpu = _indices_for_tracks(obs_cpu["track_id"], heldout_tracks)

    args.fit_device = torch.device(args.fit_device)
    obs = _move_obs(obs_cpu, args.fit_device)
    train_indices = train_indices_cpu.to(args.fit_device)
    heldout_indices = heldout_indices_cpu.to(args.fit_device)

    model_names = [name.strip().upper() for name in args.models.split(",") if name.strip()]
    results: Dict[str, Any] = {}
    fitted_payload: Dict[str, Any] = {
        "train_tracks": train_tracks,
        "heldout_tracks": heldout_tracks,
        "camera_index_values": dataset["summary"]["camera_index_values"],
    }
    for model_name in model_names:
        if model_name not in {"O0", "O1"}:
            raise ValueError(f"Unsupported oracle model: {model_name}. Expected O0 or O1.")
        model, history = _fit_model(
            model_name=model_name,
            obs=obs,
            track_count=track_count,
            camera_count=camera_count,
            train_indices=train_indices,
            args=args,
        )
        metrics = _evaluate_model(model, obs, train_indices, heldout_indices, args.eps)
        results[model_name] = {
            "history": history,
            "metrics": metrics,
            "parameters": model.fitted_parameters(dataset["summary"]["camera_index_values"]),
        }
        fitted_payload[f"{model_name}_state_dict"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    summary: Dict[str, Any] = {
        "diagnostic": "gmvc_lowdim_physical_oracle",
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "experiment_name": getattr(config, "experiment_name", ""),
        "method_name": getattr(config, "method_name", ""),
        "split": args.split,
        "track_config": asdict(track_cfg),
        "fit_config": {
            "models": model_names,
            "iters": args.iters,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "train_fraction": args.train_fraction,
            "j_min": args.j_min,
            "j_max": args.j_max,
            "o1_log_beta_scale": args.o1_log_beta_scale,
            "o1_binf_logit_scale": args.o1_binf_logit_scale,
            "charbonnier_eps": args.charbonnier_eps,
            "fit_device": str(args.fit_device),
        },
        "dataset": dataset["summary"],
        "split_counts": {
            "train_tracks": int(train_tracks.numel()),
            "heldout_tracks": int(heldout_tracks.numel()),
            "train_observations": int(train_indices_cpu.numel()),
            "heldout_observations": int(heldout_indices_cpu.numel()),
        },
        "models": results,
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (args.output_dir / "gmvc_lowdim_oracle.json")
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf8")
    if args.save_fit_state:
        torch.save(fitted_payload, args.output_dir / "gmvc_lowdim_oracle_fit_state.pt")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--test-mode", default="inference")
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--samples-per-view", type=int, default=4096)
    parser.add_argument("--target-neighbor-window", type=int, default=0)
    parser.add_argument("--track-min-views", type=int, default=3)
    parser.add_argument("--alpha-threshold", type=float, default=0.95)
    parser.add_argument("--depth-rel-threshold", type=float, default=0.02)
    parser.add_argument("--depth-std-rel-threshold", type=float, default=0.25)
    parser.add_argument("--relative-depth-span", type=float, default=0.05)
    parser.add_argument("--transmission-min", type=float, default=0.10)
    parser.add_argument("--span-weight-high", type=float, default=0.10)
    parser.add_argument("--depth-error-sigma", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--edge-margin", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--models", default="O0,O1")
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses full-batch fitting.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--charbonnier-eps", type=float, default=1e-6)
    parser.add_argument("--j-min", type=float, default=-0.25)
    parser.add_argument("--j-max", type=float, default=1.25)
    parser.add_argument("--o1-log-beta-scale", type=float, default=0.15)
    parser.add_argument("--o1-binf-logit-scale", type=float, default=0.10)
    parser.add_argument("--fit-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--save-fit-state", action="store_true")
    args = parser.parse_args()

    result = run(args)
    compact = {
        "checkpoint": result["checkpoint"],
        "step": result["step"],
        "split": result["split"],
        "dataset": {
            "accepted_tracks": result["dataset"]["counters"]["accepted_tracks"],
            "accepted_observations": result["dataset"]["counters"]["accepted_observations"],
            "view_count": result["dataset"]["view_count"],
        },
        "split_counts": result["split_counts"],
        "models": {
            name: {
                "train_recon_l1": model_result["metrics"]["reconstruction_weighted_l1"]["train"],
                "heldout_recon_l1": model_result["metrics"]["reconstruction_weighted_l1"]["heldout"],
                "heldout_consensus_j_recon_l1": model_result["metrics"]["cross_view"]["heldout"][
                    "consensus_j_reconstruction_l1"
                ],
                "heldout_transfer_l1": model_result["metrics"]["cross_view"]["heldout"]["transfer_l1"],
                "heldout_closure_norm_l1": model_result["metrics"]["cross_view"]["heldout"]["closure_norm_l1"],
                "heldout_object_j_variance": model_result["metrics"]["cross_view"]["heldout"]["object_j_variance"],
                "residual_saturation": model_result["parameters"]["residual_budget"][
                    "saturation_ratio_abs_tanh_gt_095"
                ],
            }
            for name, model_result in result["models"].items()
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
