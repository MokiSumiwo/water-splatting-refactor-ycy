#!/usr/bin/env python
"""Build detached GMVC training track banks from an M1 checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import torch

from nerfstudio.utils.eval_utils import eval_setup
from water_splatting.medium_calibration import GMVCTrackConfig, render_gmvc_views
from water_splatting.medium_calibration.gmvc_losses import invert_intrinsic_radiance
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


def _append_obs(
    obs_lists: List[List[Dict[str, torch.Tensor]]],
    local_idx: int,
    obs: Dict[str, torch.Tensor],
    row_idx: int,
    xy: torch.Tensor,
    view_camera_index: int,
    view_image_index: int,
) -> None:
    payload = {key: value[row_idx].detach().cpu() for key, value in obs.items()}
    payload["xy"] = xy[row_idx].detach().cpu()
    payload["camera_index"] = torch.tensor(float(view_camera_index))
    payload["image_index"] = torch.tensor(float(view_image_index))
    obs_lists[local_idx].append(payload)


def _track_bank_entries(observations: List[Dict[str, torch.Tensor]], cfg: GMVCTrackConfig) -> List[Dict[str, torch.Tensor]]:
    depth = torch.stack([obs["depth"].reshape(()) for obs in observations]).float()
    alpha = torch.stack([obs["alpha"].reshape(()) for obs in observations]).float()
    depth_err = torch.stack([obs["depth_rel_error"].reshape(()) for obs in observations]).float()
    transmission = torch.stack([obs["transmission"].float() for obs in observations])
    t_scalar = transmission.mean(dim=-1)
    gt = torch.stack([obs["gt"].float() for obs in observations])
    medium_attn = torch.stack([obs["medium_attn"].float() for obs in observations])
    medium_bs = torch.stack([obs["medium_bs"].float() for obs in observations])
    b_inf = torch.stack([obs["b_inf"].float() for obs in observations])
    j_hat = invert_intrinsic_radiance(gt, depth[:, None], medium_attn, medium_bs, b_inf, eps=cfg.eps)
    j_valid = (
        torch.isfinite(j_hat).all(dim=-1)
        & (j_hat >= cfg.j_clamp_min).all(dim=-1)
        & (j_hat <= cfg.j_clamp_max).all(dim=-1)
    )
    span = depth.max() - depth.min()
    relative_span = float((span / depth.median().clamp_min(float(cfg.eps))).item())
    if relative_span < cfg.relative_depth_span:
        return []

    w_alpha = ((alpha - cfg.alpha_threshold) / max(1.0 - cfg.alpha_threshold, cfg.eps)).clamp(0.0, 1.0)
    w_depth = torch.exp(-depth_err / max(cfg.depth_error_sigma, cfg.eps)).clamp(0.0, 1.0)
    w_t = ((t_scalar - cfg.transmission_min) / max(1.0 - cfg.transmission_min, cfg.eps)).clamp(0.0, 1.0)
    w_span = min(max(relative_span / max(cfg.span_weight_high, cfg.eps), 0.0), 1.0)
    weights = torch.where(j_valid, (w_alpha * w_depth * w_t * w_span).float(), torch.zeros_like(t_scalar))
    valid = weights > 0
    if int(valid.sum().item()) < cfg.min_views:
        return []
    denom = weights.sum().clamp_min(float(cfg.eps))
    j_center = (j_hat * weights[:, None]).sum(dim=0) / denom
    attn_log_center = (torch.log(medium_attn.clamp_min(cfg.eps)) * weights[:, None]).sum(dim=0) / denom
    bs_log_center = (torch.log(medium_bs.clamp_min(cfg.eps)) * weights[:, None]).sum(dim=0) / denom
    b_inf_center = (b_inf * weights[:, None]).sum(dim=0) / denom
    valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
    valid_depth = depth[valid_indices]
    near_idx = int(valid_indices[int(torch.argmin(valid_depth).item())].item())
    far_idx = int(valid_indices[int(torch.argmax(valid_depth).item())].item())
    depth_mid = 0.5 * (depth[near_idx] + depth[far_idx])
    baseline_t, baseline_b = _medium_terms(depth, medium_attn, medium_bs, b_inf)

    entries: List[Dict[str, torch.Tensor]] = []
    for row_idx, (obs, weight) in enumerate(zip(observations, weights)):
        if float(weight.item()) <= 0.0:
            continue
        partner_idx = far_idx if depth[row_idx] <= depth_mid else near_idx
        partner_weight = weights[partner_idx].clamp_min(0.0)
        closure_weight = torch.sqrt(weight.clamp_min(0.0) * partner_weight)
        left0 = (gt[row_idx] - baseline_b[row_idx]) * baseline_t[partner_idx]
        right0 = (gt[partner_idx] - baseline_b[partner_idx]) * baseline_t[row_idx]
        closure_denom_fixed = (left0.abs() + right0.abs()).float()
        entries.append(
            {
                "camera_index": obs["camera_index"].long(),
                "image_index": obs["image_index"].long(),
                "xy": obs["xy"].float(),
                "j_consensus": j_center.float(),
                "medium_attn_log_center": attn_log_center.float(),
                "medium_bs_log_center": bs_log_center.float(),
                "b_inf_center": b_inf_center.float(),
                "weight": weight.float(),
                "closure_partner_gt": gt[partner_idx].float(),
                "closure_partner_depth": depth[partner_idx].reshape(()).float(),
                "closure_partner_medium_attn": medium_attn[partner_idx].float(),
                "closure_partner_medium_bs": medium_bs[partner_idx].float(),
                "closure_partner_b_inf": b_inf[partner_idx].float(),
                "closure_denom_fixed": closure_denom_fixed.float(),
                "closure_weight": closure_weight.reshape(()).float(),
                "closure_depth_span": (depth[partner_idx] - depth[row_idx]).abs().reshape(()).float(),
            }
        )
    return entries


def _stack_or_empty(values: List[torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
    if not values:
        return torch.empty(shape, dtype=torch.float32)
    return torch.stack(values).float()


def _medium_terms(
    depth: torch.Tensor,
    medium_attn: torch.Tensor,
    medium_bs: torch.Tensor,
    b_inf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if depth.ndim == 1:
        depth = depth[:, None]
    transmission = torch.exp(-(medium_attn * depth).clamp_min(0.0))
    backscatter = b_inf * (1.0 - torch.exp(-(medium_bs * depth).clamp_min(0.0)))
    return transmission, backscatter


def build_bank(args: argparse.Namespace) -> Dict[str, Any]:
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
    cfg = GMVCTrackConfig(
        min_views=args.track_min_views,
        alpha_threshold=args.alpha_threshold,
        depth_rel_threshold=args.depth_rel_threshold,
        depth_std_rel_threshold=args.depth_std_rel_threshold,
        relative_depth_span=args.relative_depth_span,
        transmission_min=args.transmission_min,
        span_weight_high=args.span_weight_high,
        depth_error_sigma=args.depth_error_sigma,
        eps=args.eps,
        j_clamp_min=args.j_clamp_min,
        j_clamp_max=args.j_clamp_max,
        edge_margin=args.edge_margin,
        samples_per_view=args.samples_per_view,
        seed=args.seed,
        target_neighbor_window=args.target_neighbor_window,
    )
    views = render_gmvc_views(pipeline, args.split, args.max_images)
    per_camera_lists: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    counters: Dict[str, int] = {
        "source_valid_pixels_total": 0,
        "sampled_source_tracks": 0,
        "accepted_tracks": 0,
        "accepted_observations": 0,
    }

    for source_idx, source_view in enumerate(views):
        source_xy, source_valid_count = _sample_source_pixels(source_view, cfg, source_idx)
        counters["source_valid_pixels_total"] += source_valid_count
        counters["sampled_source_tracks"] += int(source_xy.shape[0])
        if source_xy.numel() == 0:
            continue
        source_depth = _sample_hwc(source_view.depth, source_xy).reshape(-1)
        points_world = unproject_pixels(source_view, source_xy, source_depth)
        source_obs = _sample_observations(source_view, source_xy, torch.zeros_like(source_depth))
        obs_lists: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(source_xy.shape[0])]
        for local_idx in range(source_xy.shape[0]):
            _append_obs(
                obs_lists,
                local_idx,
                source_obs,
                local_idx,
                source_xy,
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
            final_valid = (
                torch.isfinite(target_depth)
                & (target_depth > 0)
                & (depth_rel_error <= cfg.depth_rel_threshold)
                & (alpha >= cfg.alpha_threshold)
                & (depth_std <= cfg.depth_std_rel_threshold)
                & (t_mean >= cfg.transmission_min)
            )
            for row_idx in torch.nonzero(final_valid, as_tuple=False).reshape(-1).tolist():
                _append_obs(
                    obs_lists,
                    int(local_indices[row_idx].item()),
                    target_obs,
                    int(row_idx),
                    target_xy,
                    target_view.camera_index,
                    target_view.image_index,
                )

        for observations in obs_lists:
            if len(observations) < cfg.min_views:
                continue
            entries = _track_bank_entries(observations, cfg)
            if not entries:
                continue
            counters["accepted_tracks"] += 1
            counters["accepted_observations"] += len(entries)
            for entry in entries:
                key = str(int(entry["camera_index"].item()))
                bucket = per_camera_lists.setdefault(
                    key,
                    {
                        "xy": [],
                        "j_consensus": [],
                        "medium_attn_log_center": [],
                        "medium_bs_log_center": [],
                        "b_inf_center": [],
                        "weight": [],
                        "closure_partner_gt": [],
                        "closure_partner_depth": [],
                        "closure_partner_medium_attn": [],
                        "closure_partner_medium_bs": [],
                        "closure_partner_b_inf": [],
                        "closure_denom_fixed": [],
                        "closure_weight": [],
                        "closure_depth_span": [],
                    },
                )
                for name in bucket:
                    bucket[name].append(entry[name])

    per_camera: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, bucket in sorted(per_camera_lists.items(), key=lambda item: int(item[0])):
        per_camera[key] = {
            "xy": _stack_or_empty(bucket["xy"], (0, 2)),
            "j_consensus": _stack_or_empty(bucket["j_consensus"], (0, 3)),
            "medium_attn_log_center": _stack_or_empty(bucket["medium_attn_log_center"], (0, 3)),
            "medium_bs_log_center": _stack_or_empty(bucket["medium_bs_log_center"], (0, 3)),
            "b_inf_center": _stack_or_empty(bucket["b_inf_center"], (0, 3)),
            "weight": _stack_or_empty(bucket["weight"], (0,)),
            "closure_partner_gt": _stack_or_empty(bucket["closure_partner_gt"], (0, 3)),
            "closure_partner_depth": _stack_or_empty(bucket["closure_partner_depth"], (0,)),
            "closure_partner_medium_attn": _stack_or_empty(bucket["closure_partner_medium_attn"], (0, 3)),
            "closure_partner_medium_bs": _stack_or_empty(bucket["closure_partner_medium_bs"], (0, 3)),
            "closure_partner_b_inf": _stack_or_empty(bucket["closure_partner_b_inf"], (0, 3)),
            "closure_denom_fixed": _stack_or_empty(bucket["closure_denom_fixed"], (0, 3)),
            "closure_weight": _stack_or_empty(bucket["closure_weight"], (0,)),
            "closure_depth_span": _stack_or_empty(bucket["closure_depth_span"], (0,)),
        }
        if args.max_observations_per_camera > 0:
            n = int(per_camera[key]["xy"].shape[0])
            if n > args.max_observations_per_camera:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(args.seed + int(key) * 7919)
                keep = torch.randperm(n, generator=generator)[: args.max_observations_per_camera]
                per_camera[key] = {name: value[keep] for name, value in per_camera[key].items()}

    metadata = {
        "bank_type": "gmvc_training_track_bank",
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "split": args.split,
        "view_count": len(views),
        "track_config": cfg.__dict__,
        "counters": counters,
        "per_camera_counts": {key: int(value["xy"].shape[0]) for key, value in per_camera.items()},
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    bank = {"metadata": metadata, "per_camera": per_camera}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, args.output_path)
    summary_path = args.output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--test-mode", default="inference")
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--samples-per-view", type=int, default=4096)
    parser.add_argument("--max-observations-per-camera", type=int, default=20000)
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
    parser.add_argument("--j-clamp-min", type=float, default=-0.25)
    parser.add_argument("--j-clamp-max", type=float, default=1.25)
    parser.add_argument("--edge-margin", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    metadata = build_bank(args)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
