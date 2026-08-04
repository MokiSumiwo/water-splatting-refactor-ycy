#!/usr/bin/env python
"""Fit low-dimensional GMVC physical oracle models on geometry-anchored tracks."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass
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


@dataclass
class OracleFitVariant:
    """One low-dimensional oracle fit configuration."""

    name: str
    model_name: str = "O1"
    beta_scale: float = 0.15
    binf_scale: float = 0.10
    lambda_res: float = 0.0
    lambda_sat: float = 0.0
    lambda_closure: float = 0.0
    closure_denominator: str = "current"


def _parse_o1_variants(args: argparse.Namespace) -> List[OracleFitVariant]:
    entries = [entry.strip() for entry in str(args.o1_variants or "").split(";") if entry.strip()]
    if not entries:
        return [
            OracleFitVariant(
                name="O1",
                model_name="O1",
                beta_scale=float(args.o1_log_beta_scale),
                binf_scale=float(args.o1_binf_logit_scale),
                lambda_res=float(args.lambda_res),
                lambda_sat=float(args.lambda_sat),
                lambda_closure=float(args.lambda_closure),
                closure_denominator=str(args.closure_denominator),
            )
        ]
    variants: List[OracleFitVariant] = []
    for entry in entries:
        parts = [part.strip() for part in entry.split(":")]
        if len(parts) not in {6, 7}:
            raise ValueError(
                "Each --o1-variants entry must be "
                "name:beta_scale:binf_scale:lambda_res:lambda_sat:lambda_closure[:closure_denominator]"
            )
        name, beta_scale, binf_scale, lambda_res, lambda_sat, lambda_closure = parts[:6]
        closure_denominator = parts[6] if len(parts) == 7 else str(args.closure_denominator)
        if closure_denominator not in {"current", "detach", "fixed"}:
            raise ValueError(f"Unknown closure denominator mode: {closure_denominator}")
        variants.append(
            OracleFitVariant(
                name=name,
                model_name="O1",
                beta_scale=float(beta_scale),
                binf_scale=float(binf_scale),
                lambda_res=float(lambda_res),
                lambda_sat=float(lambda_sat),
                lambda_closure=float(lambda_closure),
                closure_denominator=closure_denominator,
            )
        )
    return variants


def _camera_weight_vector(obs: Dict[str, Tensor], camera_count: int, eps: float) -> Tensor:
    weights = obs["weight"].detach().float()
    camera_id = obs["camera_id"].detach().long()
    out = torch.zeros((camera_count, 1), dtype=torch.float32, device=weights.device)
    out.index_add_(0, camera_id, weights[:, None])
    return out / out.sum().clamp_min(float(eps))


class LowDimOracle(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        obs: Dict[str, Tensor],
        track_count: int,
        camera_count: int,
        camera_weights: Tensor,
        j_min: float,
        j_max: float,
        eps: float,
        beta_residual_scale: float,
        binf_residual_scale: float,
        center_residuals: bool,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.j_min = float(j_min)
        self.j_max = float(j_max)
        self.eps = float(eps)
        self.beta_residual_scale = float(beta_residual_scale)
        self.binf_residual_scale = float(binf_residual_scale)
        self.center_residuals = bool(center_residuals)
        self.register_buffer("camera_weights", camera_weights.detach().float().reshape(camera_count, 1))
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

    def _center_delta(self, delta: Tensor) -> Tensor:
        if not self.center_residuals:
            return delta
        mean = (delta * self.camera_weights.to(delta.device)).sum(dim=0, keepdim=True)
        return delta - mean

    def _centered_deltas(self) -> Tuple[Tensor, Tensor, Tensor]:
        if self.model_name != "O1":
            empty = torch.empty((0, 3), dtype=self.log_beta_d_center.dtype, device=self.log_beta_d_center.device)
            return empty, empty, empty
        return (
            self._center_delta(self.delta_log_beta_d),
            self._center_delta(self.delta_log_beta_b),
            self._center_delta(self.delta_b_inf_raw),
        )

    def medium(self, camera_id: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.model_name == "O1":
            delta_d, delta_b, delta_inf = self._centered_deltas()
            log_beta_d = self.log_beta_d_center[None] + self.beta_residual_scale * torch.tanh(delta_d[camera_id])
            log_beta_b = self.log_beta_b_center[None] + self.beta_residual_scale * torch.tanh(delta_b[camera_id])
            b_inf_raw = self.b_inf_center_raw[None] + self.binf_residual_scale * torch.tanh(delta_inf[camera_id])
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
        delta_d, delta_b, delta_inf = [value.detach() for value in self._centered_deltas()]
        tanh_d_signed = torch.tanh(delta_d)
        tanh_b_signed = torch.tanh(delta_b)
        tanh_inf_signed = torch.tanh(delta_inf)
        tanh_d = tanh_d_signed.abs()
        tanh_b = tanh_b_signed.abs()
        tanh_inf = tanh_inf_signed.abs()
        beta_d_resid_signed = self.beta_residual_scale * tanh_d_signed
        beta_b_resid_signed = self.beta_residual_scale * tanh_b_signed
        binf_resid_signed = self.binf_residual_scale * tanh_inf_signed
        beta_d_resid = beta_d_resid_signed.abs()
        beta_b_resid = beta_b_resid_signed.abs()
        binf_resid = binf_resid_signed.abs()
        beta_resid = torch.cat([beta_d_resid, beta_b_resid], dim=0)
        tanh_abs = torch.cat([tanh_d.reshape(-1), tanh_b.reshape(-1), tanh_inf.reshape(-1)])
        weights = self.camera_weights.to(beta_d_resid_signed.device)
        beta_d_mean = (beta_d_resid_signed * weights).sum(dim=0)
        beta_b_mean = (beta_b_resid_signed * weights).sum(dim=0)
        binf_mean = (binf_resid_signed * weights).sum(dim=0)
        channel_sat = {
            "r": float((tanh_abs[0::3] > 0.95).float().mean().item()) if tanh_abs.numel() >= 3 else 0.0,
            "g": float((tanh_abs[1::3] > 0.95).float().mean().item()) if tanh_abs.numel() >= 3 else 0.0,
            "b": float((tanh_abs[2::3] > 0.95).float().mean().item()) if tanh_abs.numel() >= 3 else 0.0,
        }
        return {
            "center_residuals": self.center_residuals,
            "beta_d_log_residual_abs": _stats(beta_d_resid),
            "beta_b_log_residual_abs": _stats(beta_b_resid),
            "beta_log_residual_abs": _stats(beta_resid),
            "b_inf_logit_residual_abs": _stats(binf_resid),
            "weighted_mean_residual": {
                "beta_d_log_rgb": [float(v) for v in beta_d_mean.detach().cpu()],
                "beta_b_log_rgb": [float(v) for v in beta_b_mean.detach().cpu()],
                "b_inf_logit_rgb": [float(v) for v in binf_mean.detach().cpu()],
                "l2": float(
                    (
                        beta_d_mean.square().mean()
                        + beta_b_mean.square().mean()
                        + binf_mean.square().mean()
                    )
                    .detach()
                    .cpu()
                    .item()
                ),
            },
            "saturation_ratio_abs_tanh_gt_095": float((tanh_abs > 0.95).float().mean().item()),
            "saturation_ratio_by_parameter": {
                "beta_d": float((tanh_d > 0.95).float().mean().item()),
                "beta_b": float((tanh_b > 0.95).float().mean().item()),
                "b_inf": float((tanh_inf > 0.95).float().mean().item()),
            },
            "saturation_ratio_by_channel": channel_sat,
        }

    def regularization_terms(self, sat_threshold: float, sat_temp: float) -> Dict[str, Tensor]:
        zero = self.log_beta_d_center.new_tensor(0.0)
        if self.model_name != "O1":
            return {"residual_l2": zero, "saturation_softplus": zero, "mean_residual_l2": zero}
        delta_d, delta_b, delta_inf = self._centered_deltas()
        tanh_d = torch.tanh(delta_d)
        tanh_b = torch.tanh(delta_b)
        tanh_inf = torch.tanh(delta_inf)
        residual_l2 = tanh_d.square().mean() + tanh_b.square().mean() + tanh_inf.square().mean()
        weights = self.camera_weights.to(delta_d.device)
        beta_d_mean = (self.beta_residual_scale * tanh_d * weights).sum(dim=0)
        beta_b_mean = (self.beta_residual_scale * tanh_b * weights).sum(dim=0)
        binf_mean = (self.binf_residual_scale * tanh_inf * weights).sum(dim=0)
        mean_residual_l2 = beta_d_mean.square().mean() + beta_b_mean.square().mean() + binf_mean.square().mean()
        tanh_abs = torch.cat(
            [
                tanh_d.abs().reshape(-1),
                tanh_b.abs().reshape(-1),
                tanh_inf.abs().reshape(-1),
            ]
        )
        temp = max(float(sat_temp), self.eps)
        saturation = torch.nn.functional.softplus((tanh_abs - float(sat_threshold)) / temp).mean() * temp
        return {
            "residual_l2": residual_l2,
            "saturation_softplus": saturation,
            "mean_residual_l2": mean_residual_l2,
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


def _build_farthest_pair_indices(obs: Dict[str, Tensor], indices: Tensor, track_count: int) -> Tuple[Tensor, Tensor]:
    buckets: List[List[int]] = [[] for _ in range(track_count)]
    for idx in indices.detach().cpu().tolist():
        buckets[int(obs["track_id"][idx].detach().cpu().item())].append(int(idx))
    src: List[int] = []
    dst: List[int] = []
    depth_cpu = obs["depth"].detach().cpu().reshape(-1)
    for bucket in buckets:
        if len(bucket) < 2:
            continue
        depths = depth_cpu[torch.tensor(bucket, dtype=torch.long)]
        local_min = int(torch.argmin(depths).item())
        local_max = int(torch.argmax(depths).item())
        if local_min == local_max:
            continue
        near_idx = bucket[local_min]
        far_idx = bucket[local_max]
        src.extend([near_idx, far_idx])
        dst.extend([far_idx, near_idx])
    device = indices.device
    return torch.tensor(src, dtype=torch.long, device=device), torch.tensor(dst, dtype=torch.long, device=device)


def _closure_loss_for_pairs(
    model: LowDimOracle,
    obs: Dict[str, Tensor],
    src_indices: Tensor,
    dst_indices: Tensor,
    eps: float,
    signal_floor: float,
    charbonnier_eps: float,
    denominator_mode: str,
) -> Tensor:
    if src_indices.numel() == 0:
        return obs["gt"].new_tensor(0.0)
    _, transmission_src, backscatter_src, gt_src = model.predict(obs, src_indices)
    _, transmission_dst, backscatter_dst, gt_dst = model.predict(obs, dst_indices)
    left = (gt_src - backscatter_src) * transmission_dst
    right = (gt_dst - backscatter_dst) * transmission_src
    if denominator_mode == "current":
        denom_source = left.abs() + right.abs()
    elif denominator_mode == "detach":
        denom_source = (left.abs() + right.abs()).detach()
    elif denominator_mode == "fixed":
        src_depth = obs["depth"][src_indices]
        dst_depth = obs["depth"][dst_indices]
        src_t0 = torch.exp(-(obs["medium_attn"][src_indices] * src_depth).clamp_min(0.0))
        dst_t0 = torch.exp(-(obs["medium_attn"][dst_indices] * dst_depth).clamp_min(0.0))
        src_b0 = obs["b_inf"][src_indices] * (
            1.0 - torch.exp(-(obs["medium_bs"][src_indices] * src_depth).clamp_min(0.0))
        )
        dst_b0 = obs["b_inf"][dst_indices] * (
            1.0 - torch.exp(-(obs["medium_bs"][dst_indices] * dst_depth).clamp_min(0.0))
        )
        left0 = (gt_src.detach() - src_b0.detach()) * dst_t0.detach()
        right0 = (gt_dst.detach() - dst_b0.detach()) * src_t0.detach()
        denom_source = left0.abs() + right0.abs()
    else:
        raise ValueError(f"Unknown closure denominator mode: {denominator_mode}")
    denom = torch.clamp(denom_source, min=float(signal_floor))
    residual = (left - right) / denom.clamp_min(float(eps))
    pair_weights = torch.sqrt(obs["weight"][src_indices] * obs["weight"][dst_indices]).clamp_min(0.0)
    return _weighted_loss(residual, pair_weights, charbonnier_eps)


def _fit_model(
    variant: OracleFitVariant,
    obs: Dict[str, Tensor],
    track_count: int,
    camera_count: int,
    camera_weights: Tensor,
    train_indices: Tensor,
    closure_pair_indices: Tuple[Tensor, Tensor],
    args: argparse.Namespace,
) -> Tuple[LowDimOracle, List[Dict[str, float]]]:
    model = LowDimOracle(
        model_name=variant.model_name,
        obs=obs,
        track_count=track_count,
        camera_count=camera_count,
        camera_weights=camera_weights,
        j_min=args.j_min,
        j_max=args.j_max,
        eps=args.eps,
        beta_residual_scale=variant.beta_scale,
        binf_residual_scale=variant.binf_scale,
        center_residuals=not bool(args.no_center_residuals),
    ).to(args.fit_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: List[Dict[str, float]] = []
    closure_src, closure_dst = closure_pair_indices
    for step in range(args.iters):
        total_loss = 0.0
        total_recon = 0.0
        total_closure = 0.0
        total_res = 0.0
        total_sat = 0.0
        total_mean_res = 0.0
        total_weight = 0.0
        for batch_indices in _iter_minibatches(train_indices, args.batch_size, args.seed, step):
            optimizer.zero_grad(set_to_none=True)
            pred, _, _, gt = model.predict(obs, batch_indices)
            weights = obs["weight"][batch_indices]
            recon_loss = _weighted_loss(pred - gt, weights, args.charbonnier_eps)
            reg_terms = model.regularization_terms(args.sat_threshold, args.sat_temp)
            closure_loss = _closure_loss_for_pairs(
                model,
                obs,
                closure_src,
                closure_dst,
                args.eps,
                args.closure_signal_floor,
                args.charbonnier_eps,
                variant.closure_denominator,
            )
            loss = (
                recon_loss
                + float(variant.lambda_res) * reg_terms["residual_l2"]
                + float(args.lambda_mean_res) * reg_terms["mean_residual_l2"]
                + float(variant.lambda_sat) * reg_terms["saturation_softplus"]
                + float(variant.lambda_closure) * closure_loss
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * float(weights.sum().detach().cpu().item())
            total_recon += float(recon_loss.detach().cpu().item()) * float(weights.sum().detach().cpu().item())
            total_closure += float(closure_loss.detach().cpu().item()) * float(weights.sum().detach().cpu().item())
            total_res += float(reg_terms["residual_l2"].detach().cpu().item()) * float(weights.sum().detach().cpu().item())
            total_mean_res += float(reg_terms["mean_residual_l2"].detach().cpu().item()) * float(
                weights.sum().detach().cpu().item()
            )
            total_sat += float(reg_terms["saturation_softplus"].detach().cpu().item()) * float(
                weights.sum().detach().cpu().item()
            )
            total_weight += float(weights.sum().detach().cpu().item())
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.iters:
            history.append(
                {
                    "iter": float(step + 1),
                    "weighted_objective": total_loss / max(total_weight, 1e-8),
                    "reconstruction_charbonnier": total_recon / max(total_weight, 1e-8),
                    "closure_robust": total_closure / max(total_weight, 1e-8),
                    "residual_l2": total_res / max(total_weight, 1e-8),
                    "mean_residual_l2": total_mean_res / max(total_weight, 1e-8),
                    "saturation_softplus": total_sat / max(total_weight, 1e-8),
                    "train_weighted_l1": _weighted_l1_for_indices(model, obs, train_indices),
                }
            )
    return model, history


def _per_track_indices(track_id: Tensor, track_count: int, indices: Tensor) -> List[Tensor]:
    buckets: List[List[int]] = [[] for _ in range(track_count)]
    for idx in indices.detach().cpu().tolist():
        buckets[int(track_id[idx].item())].append(int(idx))
    return [torch.tensor(bucket, dtype=torch.long, device=indices.device) for bucket in buckets if len(bucket) >= 2]


def _weighted_mean(values: Tensor, weights: Tensor, eps: float) -> float:
    denom = weights.sum().clamp_min(float(eps))
    return float((values * weights).sum().detach().cpu().item() / denom.detach().cpu().item())


def _bucket_summary(values: Tensor, weights: Tensor, key: Tensor, bins: List[Tuple[str, float, float]], eps: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, lo, hi in bins:
        mask = (key >= float(lo)) & (key < float(hi))
        if not mask.any():
            out[name] = {"count": 0, "mean": 0.0}
            continue
        out[name] = {
            "count": int(mask.sum().item()),
            "mean": _weighted_mean(values[mask], weights[mask], eps),
        }
    return out


def _pair_metrics(
    model: LowDimOracle,
    obs: Dict[str, Tensor],
    indices: Tensor,
    eps: float,
    signal_floor: float,
) -> Dict[str, Any]:
    with torch.no_grad():
        track_count = int(obs["track_id"].max().item()) + 1
        track_buckets = _per_track_indices(obs["track_id"], track_count, indices)
        transfer_values: List[Tensor] = []
        closure_values: List[Tensor] = []
        closure_norm_values: List[Tensor] = []
        closure_floor_values: List[Tensor] = []
        consensus_recon_values: List[Tensor] = []
        j_var_values: List[Tensor] = []
        pair_weights: List[Tensor] = []
        obs_weights: List[Tensor] = []
        track_weights: List[Tensor] = []
        pair_t_values: List[Tensor] = []
        pair_signal_values: List[Tensor] = []
        for bucket in track_buckets:
            pred, transmission, backscatter, gt = model.predict(obs, bucket)
            del pred
            weights = obs["weight"][bucket]
            j_hat = (gt - backscatter) / transmission.clamp_min(float(eps))
            t_scalar = transmission.mean(dim=-1)
            signal_abs = (gt - backscatter).abs().mean(dim=-1)
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
            closure_floor = (left - right).abs() / torch.clamp(
                left.abs() + right.abs(),
                min=float(signal_floor),
            )
            closure_floor = closure_floor.mean(dim=-1)
            mean_j = (j_hat * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(float(eps))
            consensus_pred = mean_j[None] * transmission + backscatter
            consensus_recon = (consensus_pred - gt).abs().mean(dim=-1)
            j_var = ((j_hat - mean_j[None]).square().mean(dim=-1) * weights).sum() / weights.sum().clamp_min(float(eps))
            transfer_values.append(transfer)
            closure_values.append(closure)
            closure_norm_values.append(closure_norm)
            closure_floor_values.append(closure_floor)
            consensus_recon_values.append(consensus_recon)
            pair_weights.append(pair_w)
            obs_weights.append(weights)
            j_var_values.append(j_var.reshape(1))
            track_weights.append(weights.mean().reshape(1))
            pair_t_values.append(torch.minimum(t_scalar[src], t_scalar[dst]))
            pair_signal_values.append(torch.minimum(signal_abs[src], signal_abs[dst]))

        if not transfer_values:
            return {
                "track_count": 0,
                "pair_count": 0,
                "transfer_l1": 0.0,
                "closure_l1": 0.0,
                "closure_norm_l1": 0.0,
                "closure_signal_floor_l1": 0.0,
                "object_j_variance": 0.0,
            }
        transfer_t = torch.cat(transfer_values)
        closure_t = torch.cat(closure_values)
        closure_norm_t = torch.cat(closure_norm_values)
        closure_floor_t = torch.cat(closure_floor_values)
        consensus_recon_t = torch.cat(consensus_recon_values)
        pair_w_t = torch.cat(pair_weights)
        obs_w_t = torch.cat(obs_weights)
        j_var_t = torch.cat(j_var_values)
        track_w_t = torch.cat(track_weights)
        pair_t_t = torch.cat(pair_t_values)
        pair_signal_t = torch.cat(pair_signal_values)
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
            "closure_signal_floor_l1": float((closure_floor_t * pair_w_t).sum().cpu().item() / denom_pair.cpu().item()),
            "object_j_variance": float((j_var_t * track_w_t).sum().cpu().item() / denom_track.cpu().item()),
            "transfer_l1_stats": _stats(transfer_t.detach().cpu()),
            "closure_l1_stats": _stats(closure_t.detach().cpu()),
            "closure_norm_l1_stats": _stats(closure_norm_t.detach().cpu()),
            "closure_signal_floor_l1_stats": _stats(closure_floor_t.detach().cpu()),
            "transmission_pair_min_stats": _stats(pair_t_t.detach().cpu()),
            "signal_pair_min_stats": _stats(pair_signal_t.detach().cpu()),
            "transfer_by_transmission_min": _bucket_summary(
                transfer_t,
                pair_w_t,
                pair_t_t,
                [("t_lt_020", 0.0, 0.20), ("t_020_050", 0.20, 0.50), ("t_ge_050", 0.50, float("inf"))],
                eps,
            ),
            "closure_signal_floor_by_signal_min": _bucket_summary(
                closure_floor_t,
                pair_w_t,
                pair_signal_t,
                [
                    ("signal_lt_floor", 0.0, float(signal_floor)),
                    ("signal_floor_010", float(signal_floor), 0.10),
                    ("signal_ge_010", 0.10, float("inf")),
                ],
                eps,
            ),
        }


def _evaluate_model(
    model: LowDimOracle,
    obs: Dict[str, Tensor],
    train_indices: Tensor,
    heldout_indices: Tensor,
    eps: float,
    signal_floor: float,
) -> Dict[str, Any]:
    train_recon = _weighted_l1_for_indices(model, obs, train_indices)
    heldout_recon = _weighted_l1_for_indices(model, obs, heldout_indices)
    return {
        "reconstruction_weighted_l1": {
            "train": train_recon,
            "heldout": heldout_recon,
        },
        "cross_view": {
            "train": _pair_metrics(model, obs, train_indices, eps, signal_floor),
            "heldout": _pair_metrics(model, obs, heldout_indices, eps, signal_floor),
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
    variants: List[OracleFitVariant] = []
    if "O0" in model_names:
        variants.append(OracleFitVariant(name="O0", model_name="O0", beta_scale=0.0, binf_scale=0.0))
    if "O1" in model_names:
        variants.extend(_parse_o1_variants(args))
    if not variants:
        raise ValueError("No oracle variants requested. Use --models O0, O1, or O0,O1.")
    camera_weights = _camera_weight_vector(obs, camera_count, args.eps).to(args.fit_device)
    closure_pair_indices = _build_farthest_pair_indices(obs, train_indices, track_count)
    results: Dict[str, Any] = {}
    fitted_payload: Dict[str, Any] = {
        "train_tracks": train_tracks,
        "heldout_tracks": heldout_tracks,
        "camera_index_values": dataset["summary"]["camera_index_values"],
    }
    for variant in variants:
        model, history = _fit_model(
            variant=variant,
            obs=obs,
            track_count=track_count,
            camera_count=camera_count,
            camera_weights=camera_weights,
            train_indices=train_indices,
            closure_pair_indices=closure_pair_indices,
            args=args,
        )
        metrics = _evaluate_model(model, obs, train_indices, heldout_indices, args.eps, args.closure_signal_floor)
        results[variant.name] = {
            "variant": asdict(variant),
            "history": history,
            "metrics": metrics,
            "parameters": model.fitted_parameters(dataset["summary"]["camera_index_values"]),
        }
        fitted_payload[f"{variant.name}_state_dict"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}

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
            "variants": [asdict(variant) for variant in variants],
            "iters": args.iters,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "train_fraction": args.train_fraction,
            "j_min": args.j_min,
            "j_max": args.j_max,
            "o1_log_beta_scale": args.o1_log_beta_scale,
            "o1_binf_logit_scale": args.o1_binf_logit_scale,
            "lambda_res": args.lambda_res,
            "lambda_sat": args.lambda_sat,
            "lambda_closure": args.lambda_closure,
            "lambda_mean_res": args.lambda_mean_res,
            "closure_denominator": args.closure_denominator,
            "closure_signal_floor": args.closure_signal_floor,
            "sat_threshold": args.sat_threshold,
            "sat_temp": args.sat_temp,
            "center_residuals": not bool(args.no_center_residuals),
            "closure_pair_count": int(closure_pair_indices[0].numel()),
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
    parser.add_argument(
        "--o1-variants",
        default="",
        help=(
            "Semicolon-separated O1 variants: "
            "name:beta_scale:binf_scale:lambda_res:lambda_sat:lambda_closure[:closure_denominator]."
        ),
    )
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
    parser.add_argument("--lambda-res", type=float, default=0.0)
    parser.add_argument("--lambda-sat", type=float, default=0.0)
    parser.add_argument("--lambda-closure", type=float, default=0.0)
    parser.add_argument("--lambda-mean-res", type=float, default=0.0)
    parser.add_argument("--closure-denominator", choices=["current", "detach", "fixed"], default="current")
    parser.add_argument("--closure-signal-floor", type=float, default=0.03)
    parser.add_argument("--sat-threshold", type=float, default=0.80)
    parser.add_argument("--sat-temp", type=float, default=0.05)
    parser.add_argument("--no-center-residuals", action="store_true")
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
                "heldout_closure_l1": model_result["metrics"]["cross_view"]["heldout"]["closure_l1"],
                "heldout_closure_norm_l1": model_result["metrics"]["cross_view"]["heldout"]["closure_norm_l1"],
                "heldout_closure_signal_floor_l1": model_result["metrics"]["cross_view"]["heldout"][
                    "closure_signal_floor_l1"
                ],
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
