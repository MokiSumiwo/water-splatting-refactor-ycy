#!/usr/bin/env python
"""Evaluate current-checkpoint GMVC cross-view metrics without fitting an oracle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from nerfstudio.utils.eval_utils import eval_setup
from torch import Tensor

sys.path.append(str(Path(__file__).resolve().parent))

from fit_gmvc_lowdim_oracle import (
    _bucket_summary,
    _indices_for_tracks,
    _make_track_split,
    _per_track_indices,
    _stats,
    build_oracle_dataset,
)
from water_splatting.medium_calibration import GMVCTrackConfig


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _medium_terms(depth: Tensor, medium_attn: Tensor, medium_bs: Tensor, b_inf: Tensor) -> Tuple[Tensor, Tensor]:
    if depth.ndim == 1:
        depth = depth[:, None]
    transmission = torch.exp(-(medium_attn * depth).clamp_min(0.0))
    backscatter = b_inf * (1.0 - torch.exp(-(medium_bs * depth).clamp_min(0.0)))
    return transmission, backscatter


def _weighted_l1_current(obs: Dict[str, Tensor], indices: Tensor, eps: float) -> float:
    depth = obs["depth"][indices]
    medium_attn = obs["medium_attn"][indices]
    medium_bs = obs["medium_bs"][indices]
    b_inf = obs["b_inf"][indices]
    gt = obs["gt"][indices]
    weights = obs["weight"][indices]
    transmission, backscatter = _medium_terms(depth, medium_attn, medium_bs, b_inf)
    j_hat = (gt - backscatter) / transmission.clamp_min(float(eps))
    pred = j_hat * transmission + backscatter
    value = ((pred - gt).abs().mean(dim=-1) * weights).sum() / weights.sum().clamp_min(float(eps))
    return float(value.detach().cpu().item())


def _pair_metrics_current(obs: Dict[str, Tensor], indices: Tensor, eps: float, signal_floor: float) -> Dict[str, Any]:
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
            depth = obs["depth"][bucket]
            medium_attn = obs["medium_attn"][bucket]
            medium_bs = obs["medium_bs"][bucket]
            b_inf = obs["b_inf"][bucket]
            gt = obs["gt"][bucket]
            weights = obs["weight"][bucket]
            transmission, backscatter = _medium_terms(depth, medium_attn, medium_bs, b_inf)
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
            closure_norm = ((left - right).abs() / (left.abs() + right.abs() + float(eps))).mean(dim=-1)
            closure_floor = (
                (left - right).abs()
                / torch.clamp(left.abs() + right.abs(), min=float(signal_floor))
            ).mean(dim=-1)
            mean_j = (j_hat * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(float(eps))
            consensus_pred = mean_j[None] * transmission + backscatter
            consensus_recon = (consensus_pred - gt).abs().mean(dim=-1)
            j_var = ((j_hat - mean_j[None]).square().mean(dim=-1) * weights).sum()
            j_var = j_var / weights.sum().clamp_min(float(eps))

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
                "consensus_j_reconstruction_l1": 0.0,
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
            "consensus_j_reconstruction_l1": float((consensus_recon_t * obs_w_t).sum().cpu().item() / denom_obs.cpu().item()),
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
    obs = dataset["observations"]
    track_count = int(obs["track_id"].max().item()) + 1
    train_tracks, heldout_tracks = _make_track_split(track_count, args.train_fraction, args.seed)
    train_indices = _indices_for_tracks(obs["track_id"], train_tracks)
    heldout_indices = _indices_for_tracks(obs["track_id"], heldout_tracks)

    summary: Dict[str, Any] = {
        "diagnostic": "gmvc_checkpoint_cross_view_metrics",
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "experiment_name": getattr(config, "experiment_name", ""),
        "method_name": getattr(config, "method_name", ""),
        "split": args.split,
        "track_config": asdict(track_cfg),
        "dataset": dataset["summary"],
        "split_counts": {
            "train_tracks": int(train_tracks.numel()),
            "heldout_tracks": int(heldout_tracks.numel()),
            "train_observations": int(train_indices.numel()),
            "heldout_observations": int(heldout_indices.numel()),
        },
        "metrics": {
            "reconstruction_weighted_l1": {
                "train": _weighted_l1_current(obs, train_indices, args.eps),
                "heldout": _weighted_l1_current(obs, heldout_indices, args.eps),
            },
            "cross_view": {
                "train": _pair_metrics_current(obs, train_indices, args.eps, args.closure_signal_floor),
                "heldout": _pair_metrics_current(obs, heldout_indices, args.eps, args.closure_signal_floor),
            },
        },
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (args.output_dir / "gmvc_checkpoint_tracks.json")
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf8")
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
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--j-min", type=float, default=-0.25)
    parser.add_argument("--j-max", type=float, default=1.25)
    parser.add_argument("--closure-signal-floor", type=float, default=0.03)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = run(args)
    heldout = result["metrics"]["cross_view"]["heldout"]
    compact = {
        "checkpoint": result["checkpoint"],
        "step": result["step"],
        "split": result["split"],
        "dataset": {
            "accepted_tracks": result["dataset"]["counters"]["accepted_tracks"],
            "accepted_observations": result["dataset"]["counters"]["accepted_observations"],
            "view_count": result["dataset"]["view_count"],
        },
        "heldout": {
            "transfer_l1": heldout["transfer_l1"],
            "closure_l1": heldout["closure_l1"],
            "closure_norm_l1": heldout["closure_norm_l1"],
            "closure_signal_floor_l1": heldout["closure_signal_floor_l1"],
            "object_j_variance": heldout["object_j_variance"],
            "consensus_j_reconstruction_l1": heldout["consensus_j_reconstruction_l1"],
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
