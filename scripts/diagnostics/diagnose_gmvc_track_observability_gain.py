#!/usr/bin/env python
"""Attribute GMVC fixed-bank gains to track-level observability.

The script evaluates two checkpoints on the same heldout split of a fixed GMVC
track bank, then records whether candidate-vs-base gains are larger on tracks
with stronger propagation/degradation variation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import torch
from nerfstudio.utils.eval_utils import eval_setup
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics.diagnose_gmvc_fixed_bank import (  # noqa: E402
    _git_commit,
    _nearest_rank,
    _render_bank_rows,
    _robust_j_star,
    _select_tracks,
    _split_tracks,
    _stats,
    _track_indices,
)


LOWER_IS_BETTER = [
    "transfer_l1",
    "object_j_variance",
    "closure_signal_floor_l1",
    "object_target_l1",
    "dc_cross_view_variance",
    "dc_recomposition_l1",
    "consensus_j_reconstruction_l1",
]

OBS_KEYS = [
    "obs_count",
    "camera_count",
    "depth_span",
    "depth_span_rel",
    "bank_transmission_span",
    "bank_backscatter_span",
    "view_angle_span_deg",
    "camera_center_span",
    "geometry_weight_mean",
    "bank_hessian",
]

PRIMARY_OBS_KEYS = ["depth_span_rel", "bank_transmission_span", "bank_backscatter_span"]


def _weighted_mean_tensor(values: Tensor, weights: Tensor, eps: float) -> Tensor:
    if values.numel() == 0:
        return torch.tensor(float("nan"))
    return (values * weights).sum() / weights.sum().clamp_min(float(eps))


def _safe_float(value: Tensor | float | int) -> float:
    if isinstance(value, Tensor):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().float().cpu().item())
    return float(value)


def _finite(values: Iterable[float]) -> Tensor:
    out = torch.tensor(list(values), dtype=torch.float32)
    return out[torch.isfinite(out)]


def _pearson(x: Tensor, y: Tensor) -> float:
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    if int(x.numel()) < 3:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.square().sum() * y.square().sum()).clamp_min(1e-20))
    return float((x * y).sum().item() / float(denom.item()))


def _rank(values: Tensor) -> Tensor:
    values = values.detach().float().reshape(-1)
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(int(values.numel()), dtype=values.dtype)
    return ranks


def _spearman(x: Tensor, y: Tensor) -> float:
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    if int(x.numel()) < 3:
        return 0.0
    return _pearson(_rank(x), _rank(y))


def _model_outputs(
    load_config: Path,
    load_step: int | None,
    test_mode: str,
    obs: Mapping[str, Tensor],
    row_indices: Tensor,
    split: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    def _update_config(config: Any) -> Any:
        if load_step is not None:
            config.load_step = int(load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(
        load_config,
        eval_num_rays_per_chunk=None,
        test_mode=test_mode,
        update_config_callback=_update_config,
    )
    render_args = SimpleNamespace(
        object_source=args.object_source,
        force_dc_proxy=bool(args.force_dc_proxy),
    )
    outputs = _render_bank_rows(pipeline, dict(obs), row_indices, split, render_args)
    metadata = {
        "load_config": str(load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "experiment_name": getattr(config, "experiment_name", ""),
        "method_name": getattr(config, "method_name", ""),
    }
    del pipeline
    del config
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs, metadata


def _bank_terms(obs: Mapping[str, Tensor], row_indices: Tensor) -> Dict[str, Tensor]:
    depth = obs["fixed_depth"][row_indices].float().reshape(-1, 1)
    attn = obs["bank_medium_attn"][row_indices].float()
    bs = obs["bank_medium_bs"][row_indices].float()
    b_inf = obs["bank_b_inf"][row_indices].float()
    transmission = torch.exp(-(attn * depth).clamp_min(0.0))
    backscatter = b_inf * (1.0 - torch.exp(-(bs * depth).clamp_min(0.0)))
    return {"transmission": transmission, "backscatter": backscatter}


def _track_observability(
    obs: Mapping[str, Tensor],
    row_indices: Tensor,
    local_rows: Tensor,
    bank: Mapping[str, Tensor],
    eps: float,
) -> Dict[str, float]:
    rows = row_indices[local_rows]
    depth = obs["fixed_depth"][rows].float()
    weight = obs["weight"][rows].float().clamp_min(0.0)
    valid = torch.isfinite(depth) & torch.isfinite(weight) & (weight > 0)
    if int(valid.sum().item()) < 2:
        return {"valid_observability": 0.0}
    depth = depth[valid]
    weight = weight[valid]
    local_valid = local_rows[valid]
    t_scalar = bank["transmission"][local_valid].float().mean(dim=-1)
    b_scalar = bank["backscatter"][local_valid].float().mean(dim=-1)
    ray = obs["ray_direction"][rows][valid].float()
    centers = obs["camera_center"][rows][valid].float()
    cameras = obs["camera_index"][rows][valid].long()

    ray = ray / ray.norm(dim=-1, keepdim=True).clamp_min(eps)
    if int(ray.shape[0]) >= 2:
        min_dot = (ray @ ray.T).clamp(-1.0, 1.0).min()
        view_angle = torch.rad2deg(torch.acos(min_dot))
    else:
        view_angle = torch.tensor(0.0)
    center_span = (centers.max(dim=0).values - centers.min(dim=0).values).norm() if int(centers.numel()) else torch.tensor(0.0)
    depth_span = depth.max() - depth.min()
    return {
        "valid_observability": 1.0,
        "obs_count": float(depth.numel()),
        "camera_count": float(torch.unique(cameras).numel()),
        "depth_span": _safe_float(depth_span),
        "depth_span_rel": _safe_float(depth_span / depth.median().clamp_min(eps)),
        "bank_transmission_span": _safe_float(t_scalar.max() - t_scalar.min()),
        "bank_backscatter_span": _safe_float(b_scalar.max() - b_scalar.min()),
        "view_angle_span_deg": _safe_float(view_angle),
        "camera_center_span": _safe_float(center_span),
        "geometry_weight_mean": _safe_float(weight.mean()),
        "geometry_weight_p10": _nearest_rank(weight, 0.10),
        "bank_hessian": _safe_float((weight[:, None] * bank["transmission"][local_valid].square()).sum(dim=0).mean()),
    }


def _track_model_metrics(
    gt: Tensor,
    depth: Tensor,
    weight: Tensor,
    current: Mapping[str, Tensor],
    local_rows: Tensor,
    args: argparse.Namespace,
) -> Dict[str, float]:
    eps = float(args.eps)
    transmission = current["transmission"][local_rows].float()
    backscatter = current["backscatter"][local_rows].float()
    j_proxy = current["j_proxy"][local_rows].float()
    proxy_available = current["proxy_available"][local_rows].bool() & torch.isfinite(j_proxy).all(dim=-1)
    valid = (
        torch.isfinite(gt).all(dim=-1)
        & torch.isfinite(depth)
        & torch.isfinite(weight)
        & torch.isfinite(transmission).all(dim=-1)
        & torch.isfinite(backscatter).all(dim=-1)
        & (weight > 0)
    )
    if int(valid.sum().item()) < 2:
        return {"valid_metrics": 0.0}
    gt = gt[valid]
    weight = weight[valid].clamp_min(0.0)
    transmission = transmission[valid]
    backscatter = backscatter[valid]
    j_proxy = j_proxy[valid]
    proxy_available = proxy_available[valid]
    obs_n = int(weight.numel())

    j_obs = (gt - backscatter) / transmission.clamp_min(eps)
    src = torch.arange(obs_n).repeat_interleave(obs_n)
    dst = torch.arange(obs_n).repeat(obs_n)
    pair_mask = src != dst
    src = src[pair_mask]
    dst = dst[pair_mask]
    pair_w = torch.sqrt(weight[src] * weight[dst]).clamp_min(0.0)

    pred_dst = j_obs[src] * transmission[dst] + backscatter[dst]
    transfer = (pred_dst - gt[dst]).abs().mean(dim=-1)
    left = (gt[src] - backscatter[src]) * transmission[dst]
    right = (gt[dst] - backscatter[dst]) * transmission[src]
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

    out: Dict[str, float] = {
        "valid_metrics": 1.0,
        "metric_obs_count": float(obs_n),
        "transfer_l1": _safe_float(_weighted_mean_tensor(transfer, pair_w, eps)),
        "object_j_variance": _safe_float(j_var),
        "closure_signal_floor_l1": _safe_float(_weighted_mean_tensor(closure_floor, pair_w, eps)),
        "consensus_j_reconstruction_l1": _safe_float(_weighted_mean_tensor(consensus_recon, weight, eps)),
        "track_profile_residual": _safe_float(_weighted_mean_tensor(profile_residual, solve_weight, eps)),
        "irls_effective_weight_ratio": _safe_float(solve_weight.sum() / weight.sum().clamp_min(eps)),
        "j_star_outside": float(((j_star < float(args.object_j_min)) | (j_star > float(args.object_j_max))).any().item()),
    }
    if bool(proxy_available.any()):
        obj_weight = torch.where(proxy_available, weight, torch.zeros_like(weight))
        obj_fit = (j_proxy - j_star[None]).abs().mean(dim=-1)
        dc_center = (j_proxy * obj_weight[:, None]).sum(dim=0) / obj_weight.sum().clamp_min(eps)
        dc_var = ((j_proxy - dc_center[None]).square().mean(dim=-1) * obj_weight).sum() / obj_weight.sum().clamp_min(eps)
        recomp_pred = dc_center[None] * transmission + backscatter
        recomp = (recomp_pred - gt).abs().mean(dim=-1)
        out.update(
            {
                "object_target_l1": _safe_float(_weighted_mean_tensor(obj_fit, obj_weight, eps)),
                "dc_cross_view_variance": _safe_float(dc_var),
                "dc_recomposition_l1": _safe_float(_weighted_mean_tensor(recomp, weight, eps)),
                "proxy_available_fraction": _safe_float(proxy_available.float().mean()),
            }
        )
    else:
        out.update(
            {
                "object_target_l1": float("nan"),
                "dc_cross_view_variance": float("nan"),
                "dc_recomposition_l1": float("nan"),
                "proxy_available_fraction": 0.0,
            }
        )
    return out


def _track_records(
    obs: Mapping[str, Tensor],
    row_indices: Tensor,
    track_ids: Tensor,
    bank: Mapping[str, Tensor],
    base: Mapping[str, Tensor],
    candidate: Mapping[str, Tensor],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    offset = 0
    starts = obs["track_starts"].long()
    lengths = obs["track_lengths"].long()
    for track_id in track_ids.long().tolist():
        length = int(lengths[track_id].item())
        local_rows = torch.arange(offset, offset + length, dtype=torch.long)
        offset += length
        if length < 2:
            continue
        rows = row_indices[local_rows]
        gt = obs["gt"][rows].float()
        depth = obs["fixed_depth"][rows].float()
        weight = obs["weight"][rows].float()
        observability = _track_observability(obs, row_indices, local_rows, bank, float(args.eps))
        if observability.get("valid_observability", 0.0) < 1.0:
            continue
        base_metrics = _track_model_metrics(gt, depth, weight, base, local_rows, args)
        candidate_metrics = _track_model_metrics(gt, depth, weight, candidate, local_rows, args)
        if base_metrics.get("valid_metrics", 0.0) < 1.0 or candidate_metrics.get("valid_metrics", 0.0) < 1.0:
            continue
        gains: Dict[str, float] = {}
        gain_pct: Dict[str, float] = {}
        for key in LOWER_IS_BETTER:
            base_value = float(base_metrics.get(key, float("nan")))
            candidate_value = float(candidate_metrics.get(key, float("nan")))
            gain = base_value - candidate_value
            gains[key] = gain
            gain_pct[key] = 0.0 if not math.isfinite(base_value) or abs(base_value) < 1e-12 else 100.0 * gain / base_value
        records.append(
            {
                "track_id": int(track_id),
                "bank_start": int(starts[track_id].item()),
                "bank_length": int(length),
                "observability": observability,
                "base": base_metrics,
                "candidate": candidate_metrics,
                "gain": gains,
                "gain_pct": gain_pct,
            }
        )
    return records


def _values(records: List[Dict[str, Any]], section: str, key: str) -> Tensor:
    values = []
    for record in records:
        value = record.get(section, {}).get(key, float("nan"))
        values.append(float(value))
    return torch.tensor(values, dtype=torch.float32)


def _metric_values(records: List[Dict[str, Any]], label: str, key: str) -> Tensor:
    return torch.tensor([float(record[label].get(key, float("nan"))) for record in records], dtype=torch.float32)


def _gain_values(records: List[Dict[str, Any]], key: str) -> Tensor:
    return torch.tensor([float(record["gain"].get(key, float("nan"))) for record in records], dtype=torch.float32)


def _quartile_bins(records: List[Dict[str, Any]], obs_key: str, gain_keys: Iterable[str]) -> List[Dict[str, Any]]:
    obs = _values(records, "observability", obs_key)
    finite = obs[torch.isfinite(obs)]
    if int(finite.numel()) < 4:
        return []
    qs = [_nearest_rank(finite, q) for q in (0.25, 0.50, 0.75)]
    edges = [-float("inf"), *qs, float("inf")]
    out = []
    for idx in range(4):
        low, high = edges[idx], edges[idx + 1]
        if idx == 0:
            mask = obs <= high
        elif idx == 3:
            mask = obs > low
        else:
            mask = (obs > low) & (obs <= high)
        block: Dict[str, Any] = {
            "bin": idx,
            "range": [low if math.isfinite(low) else None, high if math.isfinite(high) else None],
            "count": int(mask.sum().item()),
            "observability_mean": _safe_float(obs[mask].mean()) if int(mask.sum().item()) else 0.0,
            "gains": {},
        }
        for key in gain_keys:
            gains = _gain_values(records, key)[mask]
            gains = gains[torch.isfinite(gains)]
            if int(gains.numel()) == 0:
                block["gains"][key] = {"mean": 0.0, "p50": 0.0, "positive_ratio": 0.0}
            else:
                block["gains"][key] = {
                    "mean": _safe_float(gains.mean()),
                    "p50": _nearest_rank(gains, 0.50),
                    "positive_ratio": _safe_float((gains > 0).float().mean()),
                }
        out.append(block)
    return out


def _summarize(records: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    gain_keys = ["transfer_l1", "object_j_variance", "dc_cross_view_variance", "object_target_l1", "dc_recomposition_l1"]
    summary: Dict[str, Any] = {
        "track_count": len(records),
        "observability": {},
        "base_metrics": {},
        "candidate_metrics": {},
        "gain": {},
        "correlation": {},
        "quartile_bins": {},
    }
    for key in OBS_KEYS:
        summary["observability"][key] = _stats(_values(records, "observability", key))
    for key in LOWER_IS_BETTER:
        base_values = _metric_values(records, "base", key)
        candidate_values = _metric_values(records, "candidate", key)
        gain_values = _gain_values(records, key)
        finite_gain = gain_values[torch.isfinite(gain_values)]
        summary["base_metrics"][key] = _stats(base_values)
        summary["candidate_metrics"][key] = _stats(candidate_values)
        summary["gain"][key] = {
            **_stats(gain_values),
            "positive_ratio": _safe_float((finite_gain > 0).float().mean()) if int(finite_gain.numel()) else 0.0,
        }
    for gain_key in gain_keys:
        summary["correlation"][gain_key] = {}
        gain_values = _gain_values(records, gain_key)
        for obs_key in OBS_KEYS:
            obs_values = _values(records, "observability", obs_key)
            summary["correlation"][gain_key][obs_key] = {
                "pearson": _pearson(obs_values, gain_values),
                "spearman": _spearman(obs_values, gain_values),
            }
    for obs_key in PRIMARY_OBS_KEYS:
        summary["quartile_bins"][obs_key] = _quartile_bins(records, obs_key, gain_keys)
    return summary


def run(args: argparse.Namespace) -> Dict[str, Any]:
    bank_payload = torch.load(args.track_bank, map_location="cpu")
    obs = bank_payload["observations"]
    selected_tracks = _select_tracks(obs, int(args.max_tracks), int(args.seed))
    _, heldout_tracks = _split_tracks(selected_tracks, float(args.train_fraction), int(args.seed))
    row_indices = _track_indices(obs, heldout_tracks)
    split = bank_payload["metadata"].get("split", args.split)

    bank = _bank_terms(obs, row_indices)
    base, base_meta = _model_outputs(args.base_config, args.base_step, args.test_mode, obs, row_indices, split, args)
    candidate, candidate_meta = _model_outputs(
        args.candidate_config,
        args.candidate_step,
        args.test_mode,
        obs,
        row_indices,
        split,
        args,
    )
    records = _track_records(obs, row_indices, heldout_tracks, bank, base, candidate, args)
    summary = _summarize(records, args)
    payload: Dict[str, Any] = {
        "diagnostic": "gmvc_track_observability_gain",
        "scene_name": args.scene_name,
        "bank_name": args.bank_name,
        "base_label": args.base_label,
        "candidate_label": args.candidate_label,
        "base": base_meta,
        "candidate": candidate_meta,
        "track_bank": str(args.track_bank),
        "bank_metadata": {
            "bank_type": bank_payload["metadata"].get("bank_type", ""),
            "split": bank_payload["metadata"].get("split", ""),
            "step": bank_payload["metadata"].get("step"),
            "track_config": bank_payload["metadata"].get("track_config", {}),
            "v2_track_count": int(bank_payload["metadata"].get("v2_track_count", 0)),
            "v2_observation_count": int(bank_payload["metadata"].get("v2_observation_count", 0)),
        },
        "selected": {
            "max_tracks": int(args.max_tracks),
            "seed": int(args.seed),
            "train_fraction": float(args.train_fraction),
            "heldout_tracks": int(heldout_tracks.numel()),
            "heldout_rows": int(row_indices.numel()),
            "valid_tracks": int(summary["track_count"]),
        },
        "summary": summary,
        "git_commit": _git_commit(REPO_ROOT),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_jsonl:
        jsonl_path = args.output_dir / "track_observability_gain_records.jsonl"
        with jsonl_path.open("w", encoding="utf8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        payload["records_jsonl"] = str(jsonl_path)
    output_json = args.output_json or (args.output_dir / "track_observability_gain_summary.json")
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--bank-name", required=True)
    parser.add_argument("--base-label", default="A0_MHOLD")
    parser.add_argument("--candidate-label", default="P30_MHOLD")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--base-step", type=int, default=None)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--candidate-step", type=int, default=None)
    parser.add_argument("--test-mode", default="inference")
    parser.add_argument("--track-bank", type=Path, required=True)
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--max-tracks", type=int, default=30000)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--closure-signal-floor", type=float, default=0.03)
    parser.add_argument("--irls-delta", type=float, default=0.03)
    parser.add_argument("--irls-max-weight", type=float, default=1.0)
    parser.add_argument("--object-j-min", type=float, default=-0.1)
    parser.add_argument("--object-j-max", type=float, default=1.1)
    parser.add_argument("--object-source", default="J_proxy_raw")
    parser.set_defaults(force_dc_proxy=True)
    parser.add_argument("--force-dc-proxy", dest="force_dc_proxy", action="store_true")
    parser.add_argument("--no-force-dc-proxy", dest="force_dc_proxy", action="store_false")
    parser.set_defaults(save_jsonl=True)
    parser.add_argument("--save-jsonl", dest="save_jsonl", action="store_true")
    parser.add_argument("--no-save-jsonl", dest="save_jsonl", action="store_false")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    payload = run(args)
    compact = {
        "scene_name": payload["scene_name"],
        "bank_name": payload["bank_name"],
        "base_label": payload["base_label"],
        "candidate_label": payload["candidate_label"],
        "valid_tracks": payload["selected"]["valid_tracks"],
        "gain": {
            key: {
                "mean": payload["summary"]["gain"][key]["mean"],
                "positive_ratio": payload["summary"]["gain"][key]["positive_ratio"],
            }
            for key in ["transfer_l1", "object_j_variance", "dc_cross_view_variance"]
        },
        "spearman": {
            key: {
                obs_key: payload["summary"]["correlation"][key][obs_key]["spearman"]
                for obs_key in PRIMARY_OBS_KEYS
            }
            for key in ["transfer_l1", "object_j_variance", "dc_cross_view_variance"]
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
