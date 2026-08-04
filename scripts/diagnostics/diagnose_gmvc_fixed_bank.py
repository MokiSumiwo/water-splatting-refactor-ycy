#!/usr/bin/env python
"""Evaluate GMVC metrics on a fixed track bank.

Unlike diagnose_gmvc_checkpoint_tracks.py, this script never rebuilds tracks from
the evaluated checkpoint. Track IDs, observation rows, GT RGB, fixed depths, and
bank weights all come from --track-bank.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from nerfstudio.utils.eval_utils import eval_setup
from torch import Tensor


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
    rank = max(1, min(int(values.numel()), math.ceil(float(q) * int(values.numel()))))
    return float(values.kthvalue(rank).values.item())


def _stats(values: Tensor) -> Dict[str, float]:
    values = values.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "p50": _nearest_rank(values, 0.50),
        "p75": _nearest_rank(values, 0.75),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "max": float(values.max().item()),
    }


def _weighted_mean(values: Tensor, weights: Tensor, eps: float) -> float:
    if values.numel() == 0:
        return 0.0
    denom = weights.sum().clamp_min(float(eps))
    return float((values * weights).sum().detach().cpu().item() / denom.detach().cpu().item())


def _sample_hwc(image: Tensor, xy: Tensor) -> Tensor:
    if xy.numel() == 0:
        return torch.empty((0, image.shape[-1]), dtype=image.dtype, device=image.device)
    h, w = image.shape[:2]
    xy = xy.to(device=image.device, dtype=image.dtype)
    grid_x = 2.0 * xy[:, 0] / max(float(w - 1), 1.0) - 1.0
    grid_y = 2.0 * xy[:, 1] / max(float(h - 1), 1.0) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    nchw = image.permute(2, 0, 1).unsqueeze(0)
    sampled = F.grid_sample(nchw, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled[0, :, :, 0].T.contiguous()


def _track_indices(obs: Dict[str, Tensor], track_ids: Tensor) -> Tensor:
    starts = obs["track_starts"].long()
    lengths = obs["track_lengths"].long()
    chunks = []
    for track_id in track_ids.long().tolist():
        start = int(starts[track_id].item())
        length = int(lengths[track_id].item())
        if length > 0:
            chunks.append(torch.arange(start, start + length, dtype=torch.long))
    if not chunks:
        return torch.empty((0,), dtype=torch.long)
    return torch.cat(chunks, dim=0)


def _select_tracks(obs: Dict[str, Tensor], max_tracks: int, seed: int) -> Tensor:
    track_ids = obs["track_ids"].long()
    if max_tracks > 0 and int(track_ids.numel()) > max_tracks:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        keep = torch.randperm(int(track_ids.numel()), generator=generator)[:max_tracks]
        track_ids = track_ids[keep]
    return track_ids.sort().values


def _split_tracks(track_ids: Tensor, train_fraction: float, seed: int) -> Tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 9176)
    perm = track_ids[torch.randperm(int(track_ids.numel()), generator=generator)]
    train_count = int(round(float(train_fraction) * int(track_ids.numel())))
    train_count = max(1, min(train_count, int(track_ids.numel()) - 1)) if int(track_ids.numel()) > 1 else int(track_ids.numel())
    return perm[:train_count].sort().values, perm[train_count:].sort().values


def _camera_for_image(pipeline: Any, split: str, image_index: int) -> Any:
    if split != "train":
        raise NotImplementedError("fixed-bank eval currently supports train-split banks")
    camera = pipeline.datamanager.train_dataset.cameras[int(image_index) : int(image_index) + 1]
    return camera.to(pipeline.model.device) if hasattr(camera, "to") else camera


def _force_dc_proxy_context(model: Any, args: argparse.Namespace) -> Dict[str, Any]:
    attrs = [
        "gmvc_enabled",
        "gmvc_v3_enabled",
        "lambda_gmvc_object",
        "gmvc_start_step",
        "gmvc_stop_step",
        "gmvc_v3_object_source",
        "gmvc_intrinsic_source",
        "gmvc_intrinsic_use_dc_proxy",
    ]
    saved = {name: getattr(model.config, name, None) for name in attrs}
    saved["_training"] = bool(model.training)
    saved["_step"] = int(getattr(model, "step", 0))
    if args.force_dc_proxy:
        model.train()
        model.step = max(1, int(getattr(model, "step", 1)))
        model.config.gmvc_enabled = True
        model.config.gmvc_v3_enabled = True
        model.config.lambda_gmvc_object = 1.0
        model.config.gmvc_start_step = 0
        model.config.gmvc_stop_step = 10**9
        model.config.gmvc_v3_object_source = "J_proxy_raw"
        model.config.gmvc_intrinsic_source = "J_proxy_raw"
        model.config.gmvc_intrinsic_use_dc_proxy = True
    return saved


def _restore_dc_proxy_context(model: Any, saved: Dict[str, Any]) -> None:
    was_training = bool(saved.pop("_training"))
    old_step = int(saved.pop("_step"))
    for name, value in saved.items():
        setattr(model.config, name, value)
    model.step = old_step
    model.train(was_training)


def _render_bank_rows(
    pipeline: Any,
    obs: Dict[str, Tensor],
    row_indices: Tensor,
    split: str,
    args: argparse.Namespace,
) -> Dict[str, Tensor]:
    model = pipeline.model
    device = model.device
    row_count = int(row_indices.numel())
    medium_attn = torch.empty((row_count, 3), dtype=torch.float32)
    medium_bs = torch.empty((row_count, 3), dtype=torch.float32)
    b_inf = torch.empty((row_count, 3), dtype=torch.float32)
    j_proxy = torch.empty((row_count, 3), dtype=torch.float32)
    proxy_available = torch.zeros((row_count,), dtype=torch.bool)

    image_indices = obs["image_index"][row_indices].long()
    xy_all = obs["xy"][row_indices].float()
    saved = _force_dc_proxy_context(model, args)
    try:
        with torch.no_grad():
            for image_index in image_indices.unique(sorted=True).tolist():
                local = torch.nonzero(image_indices == int(image_index), as_tuple=False).reshape(-1)
                if int(local.numel()) == 0:
                    continue
                camera = _camera_for_image(pipeline, split, int(image_index))
                outputs = model.get_outputs(camera)
                xy = xy_all[local].to(device=device)
                medium_attn[local] = _sample_hwc(outputs["medium_attn"].detach(), xy).detach().float().cpu()
                medium_bs[local] = _sample_hwc(outputs["medium_bs"].detach(), xy).detach().float().cpu()
                b_inf_image = outputs.get("b_inf", outputs["medium_rgb"])
                b_inf[local] = _sample_hwc(b_inf_image.detach(), xy).detach().float().cpu()
                proxy_image = outputs.get(args.object_source)
                if proxy_image is not None:
                    j_proxy[local] = _sample_hwc(proxy_image.detach(), xy).detach().float().cpu()
                    proxy_available[local] = True
                else:
                    fallback = outputs.get("J_gaussian_raw", outputs.get("J_raw"))
                    if fallback is None:
                        j_proxy[local] = torch.nan
                    else:
                        j_proxy[local] = _sample_hwc(fallback.detach(), xy).detach().float().cpu()
    finally:
        _restore_dc_proxy_context(model, saved)

    depth = obs["fixed_depth"][row_indices].float().reshape(-1, 1)
    transmission = torch.exp(-(medium_attn * depth).clamp_min(0.0))
    backscatter = b_inf * (1.0 - torch.exp(-(medium_bs * depth).clamp_min(0.0)))
    return {
        "medium_attn": medium_attn,
        "medium_bs": medium_bs,
        "b_inf": b_inf,
        "transmission": transmission,
        "backscatter": backscatter,
        "j_proxy": j_proxy,
        "proxy_available": proxy_available,
    }


def _robust_j_star(gt: Tensor, transmission: Tensor, backscatter: Tensor, weight: Tensor, eps: float, delta: float, max_weight: float) -> Tuple[Tensor, Tensor, Tensor]:
    numerator0 = (weight[:, None] * transmission * (gt - backscatter)).sum(dim=0)
    denominator0 = (weight[:, None] * transmission.square()).sum(dim=0)
    j0 = numerator0 / (denominator0 + float(eps))
    pred0 = j0[None] * transmission + backscatter
    residual_norm = torch.linalg.norm(pred0 - gt, dim=-1)
    irls_weight = (float(delta) / torch.sqrt(residual_norm.square() + float(delta) * float(delta))).clamp_max(float(max_weight))
    solve_weight = weight * irls_weight
    numerator = (solve_weight[:, None] * transmission * (gt - backscatter)).sum(dim=0)
    denominator = (solve_weight[:, None] * transmission.square()).sum(dim=0)
    j_star = numerator / (denominator + float(eps))
    residual = (j_star[None] * transmission + backscatter - gt).abs().mean(dim=-1)
    return j_star, solve_weight, residual


def _evaluate_split(
    data: Dict[str, Tensor],
    local_track_ids: Tensor,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    eps = float(args.eps)
    transfer_values: List[Tensor] = []
    closure_values: List[Tensor] = []
    closure_floor_values: List[Tensor] = []
    consensus_recon_values: List[Tensor] = []
    j_var_values: List[Tensor] = []
    obj_fit_values: List[Tensor] = []
    obj_fit_weights: List[Tensor] = []
    dc_var_values: List[Tensor] = []
    recomp_values: List[Tensor] = []
    recomp_weights: List[Tensor] = []
    pair_weights: List[Tensor] = []
    obs_weights: List[Tensor] = []
    track_weights: List[Tensor] = []
    track_residuals: List[Tensor] = []
    irls_effective_ratios: List[Tensor] = []
    j_star_outside: List[Tensor] = []
    hessian_values: List[Tensor] = []
    t_span_values: List[Tensor] = []
    depth_span_rel_values: List[Tensor] = []
    proxy_available_count = 0
    row_count = 0

    local_track = data["local_track"]
    gt_all = data["gt"]
    depth_all = data["depth"]
    weight_all = data["weight"]
    transmission_all = data["transmission"]
    backscatter_all = data["backscatter"]
    j_proxy_all = data["j_proxy"]
    proxy_available_all = data["proxy_available"]

    for track_id in local_track_ids.long().tolist():
        rows = torch.nonzero(local_track == int(track_id), as_tuple=False).reshape(-1)
        if int(rows.numel()) < 2:
            continue
        gt = gt_all[rows]
        depth = depth_all[rows]
        weight = weight_all[rows].clamp_min(0.0)
        transmission = transmission_all[rows]
        backscatter = backscatter_all[rows]
        j_proxy = j_proxy_all[rows]
        proxy_available = proxy_available_all[rows] & torch.isfinite(j_proxy).all(dim=-1)
        valid = torch.isfinite(gt).all(dim=-1) & torch.isfinite(transmission).all(dim=-1) & torch.isfinite(backscatter).all(dim=-1) & (weight > 0)
        if int(valid.sum().item()) < 2:
            continue
        gt = gt[valid]
        depth = depth[valid]
        weight = weight[valid]
        transmission = transmission[valid]
        backscatter = backscatter[valid]
        j_proxy = j_proxy[valid]
        proxy_available = proxy_available[valid]
        obs_n = int(weight.numel())
        row_count += obs_n
        proxy_available_count += int(proxy_available.sum().item())

        j_obs = (gt - backscatter) / transmission.clamp_min(eps)
        t_scalar = transmission.mean(dim=-1)
        src = torch.arange(obs_n).repeat_interleave(obs_n)
        dst = torch.arange(obs_n).repeat(obs_n)
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        pair_w = torch.sqrt(weight[src] * weight[dst]).clamp_min(0.0)
        pred_dst = j_obs[src] * transmission[dst] + backscatter[dst]
        transfer = (pred_dst - gt[dst]).abs().mean(dim=-1)
        left = (gt[src] - backscatter[src]) * transmission[dst]
        right = (gt[dst] - backscatter[dst]) * transmission[src]
        closure = (left - right).abs().mean(dim=-1)
        closure_floor = ((left - right).abs() / torch.clamp(left.abs() + right.abs(), min=float(args.closure_signal_floor))).mean(dim=-1)
        mean_j = (j_obs * weight[:, None]).sum(dim=0) / weight.sum().clamp_min(eps)
        consensus_pred = mean_j[None] * transmission + backscatter
        consensus_recon = (consensus_pred - gt).abs().mean(dim=-1)
        j_var = ((j_obs - mean_j[None]).square().mean(dim=-1) * weight).sum() / weight.sum().clamp_min(eps)

        j_star, solve_weight, profile_residual = _robust_j_star(
            gt,
            transmission,
            backscatter,
            weight,
            eps=eps,
            delta=float(args.irls_delta),
            max_weight=float(args.irls_max_weight),
        )
        if bool(proxy_available.any()):
            obj_weight = torch.where(proxy_available, weight, torch.zeros_like(weight))
            obj_fit = (j_proxy - j_star[None]).abs().mean(dim=-1)
            dc_center = (j_proxy * obj_weight[:, None]).sum(dim=0) / obj_weight.sum().clamp_min(eps)
            dc_var = ((j_proxy - dc_center[None]).square().mean(dim=-1) * obj_weight).sum() / obj_weight.sum().clamp_min(eps)
            recomp_pred = dc_center[None] * transmission + backscatter
            recomp = (recomp_pred - gt).abs().mean(dim=-1)
            obj_fit_values.append(obj_fit)
            obj_fit_weights.append(obj_weight)
            dc_var_values.append(dc_var.reshape(1))
            recomp_values.append(recomp)
            recomp_weights.append(weight)
        else:
            obj_weight = torch.zeros_like(weight)

        transfer_values.append(transfer)
        closure_values.append(closure)
        closure_floor_values.append(closure_floor)
        consensus_recon_values.append(consensus_recon)
        j_var_values.append(j_var.reshape(1))
        pair_weights.append(pair_w)
        obs_weights.append(weight)
        track_weights.append(weight.mean().reshape(1))
        track_residuals.append(((profile_residual * solve_weight).sum() / solve_weight.sum().clamp_min(eps)).reshape(1))
        irls_effective_ratios.append((solve_weight.sum() / weight.sum().clamp_min(eps)).reshape(1))
        j_star_outside.append(((j_star < float(args.object_j_min)) | (j_star > float(args.object_j_max))).any().float().reshape(1))
        hessian_values.append((weight[:, None] * transmission.square()).sum(dim=0).mean().reshape(1))
        t_span_values.append((t_scalar.max() - t_scalar.min()).reshape(1))
        depth_span_rel_values.append(((depth.max() - depth.min()) / depth.median().clamp_min(eps)).reshape(1))

    if not transfer_values:
        return {"track_count": 0, "row_count": int(row_count)}

    pair_w_t = torch.cat(pair_weights)
    obs_w_t = torch.cat(obs_weights)
    track_w_t = torch.cat(track_weights)
    transfer_t = torch.cat(transfer_values)
    closure_t = torch.cat(closure_values)
    closure_floor_t = torch.cat(closure_floor_values)
    consensus_t = torch.cat(consensus_recon_values)
    j_var_t = torch.cat(j_var_values)
    track_residual_t = torch.cat(track_residuals)
    irls_ratio_t = torch.cat(irls_effective_ratios)
    j_out_t = torch.cat(j_star_outside)
    hessian_t = torch.cat(hessian_values)
    t_span_t = torch.cat(t_span_values)
    depth_span_rel_t = torch.cat(depth_span_rel_values)

    metrics: Dict[str, Any] = {
        "track_count": int(track_w_t.numel()),
        "row_count": int(row_count),
        "proxy_available_fraction": float(proxy_available_count / max(row_count, 1)),
        "transfer_l1": _weighted_mean(transfer_t, pair_w_t, eps),
        "closure_l1": _weighted_mean(closure_t, pair_w_t, eps),
        "closure_signal_floor_l1": _weighted_mean(closure_floor_t, pair_w_t, eps),
        "consensus_j_reconstruction_l1": _weighted_mean(consensus_t, obs_w_t, eps),
        "object_j_variance": _weighted_mean(j_var_t, track_w_t, eps),
        "track_profile_residual": _stats(track_residual_t),
        "irls_effective_weight_ratio": _stats(irls_ratio_t),
        "j_star_outside_ratio": float(j_out_t.mean().item()) if j_out_t.numel() else 0.0,
        "valid_hessian_ratio": float((hessian_t >= float(args.min_hessian)).float().mean().item()) if hessian_t.numel() else 0.0,
        "valid_transmission_span_ratio": float((t_span_t >= float(args.min_transmission_span)).float().mean().item()) if t_span_t.numel() else 0.0,
        "valid_depth_span_ratio": float((depth_span_rel_t >= float(args.min_depth_span_rel)).float().mean().item()) if depth_span_rel_t.numel() else 0.0,
        "hessian_stats": _stats(hessian_t),
        "transmission_span_stats": _stats(t_span_t),
        "depth_span_rel_stats": _stats(depth_span_rel_t),
    }
    if obj_fit_values:
        obj_fit_t = torch.cat(obj_fit_values)
        obj_fit_w_t = torch.cat(obj_fit_weights)
        recomp_t = torch.cat(recomp_values)
        recomp_w_t = torch.cat(recomp_weights)
        metrics["object_target_l1"] = _weighted_mean(obj_fit_t, obj_fit_w_t, eps)
        metrics["dc_cross_view_variance"] = float(torch.cat(dc_var_values).mean().item()) if dc_var_values else 0.0
        metrics["dc_recomposition_l1"] = _weighted_mean(recomp_t, recomp_w_t, eps)
        metrics["object_target_l1_stats"] = _stats(obj_fit_t)
        metrics["dc_recomposition_l1_stats"] = _stats(recomp_t)
    else:
        metrics["object_target_l1"] = 0.0
        metrics["dc_cross_view_variance"] = 0.0
        metrics["dc_recomposition_l1"] = 0.0
    return metrics


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
    bank = torch.load(args.track_bank, map_location="cpu")
    obs = bank["observations"]
    selected_tracks = _select_tracks(obs, args.max_tracks, args.seed)
    train_tracks_old, heldout_tracks_old = _split_tracks(selected_tracks, args.train_fraction, args.seed)
    row_indices = _track_indices(obs, selected_tracks)
    old_to_local = torch.full((int(obs["track_ids"].max().item()) + 1,), -1, dtype=torch.long)
    old_to_local[selected_tracks.long()] = torch.arange(int(selected_tracks.numel()), dtype=torch.long)
    local_track = old_to_local[obs["track_id"][row_indices].long()]
    train_tracks = old_to_local[train_tracks_old.long()]
    heldout_tracks = old_to_local[heldout_tracks_old.long()]

    current = _render_bank_rows(pipeline, obs, row_indices, bank["metadata"].get("split", args.split), args)
    data = {
        "local_track": local_track,
        "gt": obs["gt"][row_indices].float(),
        "depth": obs["fixed_depth"][row_indices].float(),
        "weight": obs["weight"][row_indices].float(),
        **current,
    }
    summary: Dict[str, Any] = {
        "diagnostic": "gmvc_fixed_bank_metrics",
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "experiment_name": getattr(config, "experiment_name", ""),
        "method_name": getattr(config, "method_name", ""),
        "track_bank": str(args.track_bank),
        "bank_metadata": {
            "bank_type": bank["metadata"].get("bank_type", ""),
            "split": bank["metadata"].get("split", ""),
            "step": bank["metadata"].get("step", None),
            "track_config": bank["metadata"].get("track_config", {}),
            "counters": bank["metadata"].get("counters", {}),
            "v2_track_count": int(bank["metadata"].get("v2_track_count", 0)),
            "v2_observation_count": int(bank["metadata"].get("v2_observation_count", 0)),
        },
        "selected": {
            "track_count": int(selected_tracks.numel()),
            "row_count": int(row_indices.numel()),
            "train_tracks": int(train_tracks.numel()),
            "heldout_tracks": int(heldout_tracks.numel()),
            "max_tracks": int(args.max_tracks),
            "seed": int(args.seed),
            "train_fraction": float(args.train_fraction),
        },
        "metrics": {
            "heldout": _evaluate_split(data, heldout_tracks, args),
        },
        "object_source": args.object_source,
        "force_dc_proxy": bool(args.force_dc_proxy),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if args.include_train:
        summary["metrics"]["train"] = _evaluate_split(data, train_tracks, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (args.output_dir / "gmvc_fixed_bank_metrics.json")
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--test-mode", default="inference")
    parser.add_argument("--track-bank", type=Path, required=True)
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--max-tracks", type=int, default=30000)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--include-train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--closure-signal-floor", type=float, default=0.03)
    parser.add_argument("--irls-delta", type=float, default=0.03)
    parser.add_argument("--irls-max-weight", type=float, default=1.0)
    parser.add_argument("--object-j-min", type=float, default=-0.1)
    parser.add_argument("--object-j-max", type=float, default=1.1)
    parser.add_argument("--min-hessian", type=float, default=1e-5)
    parser.add_argument("--min-transmission-span", type=float, default=0.01)
    parser.add_argument("--min-depth-span-rel", type=float, default=0.05)
    parser.add_argument("--object-source", default="J_proxy_raw")
    parser.set_defaults(force_dc_proxy=True)
    parser.add_argument("--force-dc-proxy", dest="force_dc_proxy", action="store_true")
    parser.add_argument("--no-force-dc-proxy", dest="force_dc_proxy", action="store_false")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = run(args)
    heldout = result["metrics"]["heldout"]
    compact = {
        "checkpoint": result["checkpoint"],
        "step": result["step"],
        "track_bank": result["track_bank"],
        "selected": result["selected"],
        "heldout": {
            "transfer_l1": heldout.get("transfer_l1", 0.0),
            "object_j_variance": heldout.get("object_j_variance", 0.0),
            "closure_signal_floor_l1": heldout.get("closure_signal_floor_l1", 0.0),
            "consensus_j_reconstruction_l1": heldout.get("consensus_j_reconstruction_l1", 0.0),
            "object_target_l1": heldout.get("object_target_l1", 0.0),
            "dc_cross_view_variance": heldout.get("dc_cross_view_variance", 0.0),
            "dc_recomposition_l1": heldout.get("dc_recomposition_l1", 0.0),
            "proxy_available_fraction": heldout.get("proxy_available_fraction", 0.0),
            "j_star_outside_ratio": heldout.get("j_star_outside_ratio", 0.0),
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
