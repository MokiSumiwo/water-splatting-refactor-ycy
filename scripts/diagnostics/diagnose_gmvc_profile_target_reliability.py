#!/usr/bin/env python
"""Audit whether GMVC profile-target reliability is predictable from track signals.

This diagnostic is offline only. It computes detached per-track features from a
fixed GMVC track bank, joins them with previously measured track-level utility
records, and reports whether allowed training-time signals can predict which
tracks benefited from profile calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import torch
from torch import Tensor


LOWER_IS_BETTER = [
    "transfer_l1",
    "object_j_variance",
    "closure_signal_floor_l1",
    "dc_recomposition_l1",
    "dc_cross_view_variance",
    "object_target_l1",
]

FEATURE_DIRECTIONS: Dict[str, int] = {
    "valid_observation_count": 1,
    "camera_count": 1,
    "depth_span": 1,
    "depth_span_rel": 1,
    "depth_iqr": 1,
    "depth_iqr_rel": 1,
    "transmission_span": 1,
    "backscatter_span": 1,
    "view_direction_spread_deg": 1,
    "profile_denominator_mean": 1,
    "profile_denominator_min": 1,
    "profile_denominator_signal_ratio": 1,
    "geometry_weight_mean": 1,
    "geometry_weight_min": 1,
    "geometry_weight_p10": 1,
    "irls_inlier_ratio": 1,
    "irls_effective_observation_count": 1,
    "irls_effective_observation_fraction": 1,
    "weighted_profile_residual_mean": -1,
    "weighted_profile_residual_median": -1,
    "weighted_profile_residual_p90": -1,
    "maximum_single_observation_weight_share": -1,
    "jstar_channel_saturation_ratio": -1,
    "jstar_any_saturation": -1,
    "loo_observation_error_mean": -1,
    "loo_observation_error_median": -1,
    "loo_observation_error_p90": -1,
    "loo_observation_error_worst": -1,
    "loo_jstar_drift_mean": -1,
    "loo_jstar_drift_median": -1,
    "loo_jstar_drift_p90": -1,
    "loo_jstar_drift_worst": -1,
    "loo_successful_solve_ratio": 1,
}

PRIMARY_FEATURES = [
    "loo_observation_error_median",
    "loo_jstar_drift_median",
    "weighted_profile_residual_median",
    "irls_inlier_ratio",
    "irls_effective_observation_count",
    "maximum_single_observation_weight_share",
    "depth_span_rel",
    "transmission_span",
    "backscatter_span",
]


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _safe_float(value: Union[Tensor, float, int]) -> float:
    if isinstance(value, Tensor):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().float().cpu().item())
    return float(value)


def _finite(values: Tensor) -> Tensor:
    values = values.detach().float().reshape(-1)
    return values[torch.isfinite(values)]


def _nearest_rank(values: Tensor, q: float) -> float:
    values = _finite(values)
    if int(values.numel()) == 0:
        return float("nan")
    rank = max(1, min(int(values.numel()), math.ceil(float(q) * int(values.numel()))))
    return float(values.kthvalue(rank).values.item())


def _stats(values: Tensor) -> Dict[str, float]:
    values = _finite(values)
    if int(values.numel()) == 0:
        return {"count": 0, "mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "p25": _nearest_rank(values, 0.25),
        "p50": _nearest_rank(values, 0.50),
        "p75": _nearest_rank(values, 0.75),
        "p90": _nearest_rank(values, 0.90),
        "max": float(values.max().item()),
    }


def _rank(values: Tensor) -> Tensor:
    values = values.detach().float().reshape(-1)
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(int(values.numel()), dtype=values.dtype)
    return ranks


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


def _spearman(x: Tensor, y: Tensor) -> float:
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    if int(x.numel()) < 3:
        return 0.0
    return _pearson(_rank(x), _rank(y))


def _weighted_quantile(values: Tensor, weights: Tensor, q: float) -> float:
    values = values.detach().float().reshape(-1)
    weights = weights.detach().float().reshape(-1)
    mask = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0)
    if int(mask.sum().item()) == 0:
        return float("nan")
    values = values[mask]
    weights = weights[mask]
    order = torch.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = torch.cumsum(weights, dim=0)
    threshold = float(q) * float(cdf[-1].item())
    idx = int(torch.searchsorted(cdf, torch.tensor(threshold, dtype=cdf.dtype), right=False).item())
    idx = max(0, min(idx, int(values.numel()) - 1))
    return float(values[idx].item())


def _weighted_mean(values: Tensor, weights: Tensor, eps: float) -> float:
    mask = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0)
    if int(mask.sum().item()) == 0:
        return float("nan")
    values = values[mask].float()
    weights = weights[mask].float()
    return float((values * weights).sum().item() / weights.sum().clamp_min(float(eps)).item())


def _robust_j_star(
    gt: Tensor,
    transmission: Tensor,
    backscatter: Tensor,
    weight: Tensor,
    eps: float,
    delta: float,
    max_weight: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    numerator0 = (weight[:, None] * transmission * (gt - backscatter)).sum(dim=0)
    denominator0 = (weight[:, None] * transmission.square()).sum(dim=0)
    j0 = numerator0 / denominator0.clamp_min(float(eps))
    pred0 = j0[None] * transmission + backscatter
    residual_norm = torch.linalg.norm(pred0 - gt, dim=-1)
    irls = (float(delta) / torch.sqrt(residual_norm.square() + float(delta) * float(delta))).clamp_max(
        float(max_weight)
    )
    solve_weight = weight * irls
    numerator = (solve_weight[:, None] * transmission * (gt - backscatter)).sum(dim=0)
    denominator = (solve_weight[:, None] * transmission.square()).sum(dim=0)
    j_star = numerator / denominator.clamp_min(float(eps))
    residual = (j_star[None] * transmission + backscatter - gt).abs().mean(dim=-1)
    return j_star, solve_weight, residual, denominator


def _medium_terms(obs: Mapping[str, Tensor], rows: Tensor) -> Tuple[Tensor, Tensor]:
    depth = obs["fixed_depth"][rows].float().reshape(-1, 1)
    attn = obs["bank_medium_attn"][rows].float()
    bs = obs["bank_medium_bs"][rows].float()
    b_inf = obs["bank_b_inf"][rows].float()
    transmission = torch.exp(-(attn * depth).clamp_min(0.0))
    backscatter = b_inf * (1.0 - torch.exp(-(bs * depth).clamp_min(0.0)))
    return transmission, backscatter


def _view_spread_deg(rays: Tensor, eps: float) -> float:
    if int(rays.shape[0]) < 2:
        return 0.0
    rays = rays.float()
    rays = rays / rays.norm(dim=-1, keepdim=True).clamp_min(float(eps))
    min_dot = (rays @ rays.T).clamp(-1.0, 1.0).min()
    return float(torch.rad2deg(torch.acos(min_dot)).item())


def _loo_features(
    gt: Tensor,
    transmission: Tensor,
    backscatter: Tensor,
    weight: Tensor,
    j_star: Tensor,
    eps: float,
    delta: float,
    max_weight: float,
) -> Dict[str, float]:
    errors: List[Tensor] = []
    drifts: List[Tensor] = []
    obs_count = int(weight.numel())
    if obs_count < 3:
        return {
            "loo_observation_error_mean": float("nan"),
            "loo_observation_error_median": float("nan"),
            "loo_observation_error_p90": float("nan"),
            "loo_observation_error_worst": float("nan"),
            "loo_jstar_drift_mean": float("nan"),
            "loo_jstar_drift_median": float("nan"),
            "loo_jstar_drift_p90": float("nan"),
            "loo_jstar_drift_worst": float("nan"),
            "loo_successful_solve_ratio": 0.0,
        }
    for idx in range(obs_count):
        keep = torch.ones((obs_count,), dtype=torch.bool)
        keep[idx] = False
        if int((weight[keep] > 0).sum().item()) < 2:
            continue
        j_loo, solve_weight, _, denominator = _robust_j_star(
            gt[keep],
            transmission[keep],
            backscatter[keep],
            weight[keep],
            eps=eps,
            delta=delta,
            max_weight=max_weight,
        )
        if not bool(torch.isfinite(j_loo).all()) or not bool((solve_weight > 0).any()):
            continue
        if float(denominator.mean().item()) <= eps:
            continue
        pred = j_loo[None] * transmission[idx : idx + 1] + backscatter[idx : idx + 1]
        errors.append((pred[0] - gt[idx]).abs().mean())
        drifts.append((j_loo - j_star).abs().mean())
    if not errors:
        return {
            "loo_observation_error_mean": float("nan"),
            "loo_observation_error_median": float("nan"),
            "loo_observation_error_p90": float("nan"),
            "loo_observation_error_worst": float("nan"),
            "loo_jstar_drift_mean": float("nan"),
            "loo_jstar_drift_median": float("nan"),
            "loo_jstar_drift_p90": float("nan"),
            "loo_jstar_drift_worst": float("nan"),
            "loo_successful_solve_ratio": 0.0,
        }
    error = torch.stack(errors)
    drift = torch.stack(drifts)
    return {
        "loo_observation_error_mean": _safe_float(error.mean()),
        "loo_observation_error_median": _nearest_rank(error, 0.50),
        "loo_observation_error_p90": _nearest_rank(error, 0.90),
        "loo_observation_error_worst": _safe_float(error.max()),
        "loo_jstar_drift_mean": _safe_float(drift.mean()),
        "loo_jstar_drift_median": _nearest_rank(drift, 0.50),
        "loo_jstar_drift_p90": _nearest_rank(drift, 0.90),
        "loo_jstar_drift_worst": _safe_float(drift.max()),
        "loo_successful_solve_ratio": float(len(errors) / max(obs_count, 1)),
    }


def _track_features(obs: Mapping[str, Tensor], track_id: int, args: argparse.Namespace) -> Dict[str, float]:
    start = int(obs["track_starts"][track_id].item())
    length = int(obs["track_lengths"][track_id].item())
    rows = torch.arange(start, start + length, dtype=torch.long)
    gt = obs["gt"][rows].float()
    depth = obs["fixed_depth"][rows].float()
    weight = obs["weight"][rows].float().clamp_min(0.0)
    transmission, backscatter = _medium_terms(obs, rows)
    valid = (
        torch.isfinite(gt).all(dim=-1)
        & torch.isfinite(depth)
        & torch.isfinite(weight)
        & torch.isfinite(transmission).all(dim=-1)
        & torch.isfinite(backscatter).all(dim=-1)
        & (weight > 0)
    )
    if int(valid.sum().item()) < 2:
        return {"valid_feature_track": 0.0, "track_id": float(track_id)}
    gt = gt[valid]
    depth = depth[valid]
    weight = weight[valid]
    transmission = transmission[valid]
    backscatter = backscatter[valid]
    valid_rows = rows[valid]
    t_scalar = transmission.mean(dim=-1)
    b_scalar = backscatter.mean(dim=-1)
    depth_sorted = depth.sort().values
    q25 = _nearest_rank(depth_sorted, 0.25)
    q75 = _nearest_rank(depth_sorted, 0.75)
    depth_median = _nearest_rank(depth, 0.50)
    j_star, solve_weight, residual, denominator = _robust_j_star(
        gt,
        transmission,
        backscatter,
        weight,
        eps=float(args.eps),
        delta=float(args.irls_delta),
        max_weight=float(args.irls_max_weight),
    )
    solve_sum = solve_weight.sum().clamp_min(float(args.eps))
    irls_ratio = solve_weight / weight.clamp_min(float(args.eps))
    inlier = irls_ratio >= float(args.irls_inlier_threshold)
    effective_obs = solve_sum.square() / solve_weight.square().sum().clamp_min(float(args.eps))
    features: Dict[str, float] = {
        "valid_feature_track": 1.0,
        "track_id": float(track_id),
        "valid_observation_count": float(valid.sum().item()),
        "camera_count": float(torch.unique(obs["camera_index"][valid_rows].long()).numel()),
        "depth_span": _safe_float(depth.max() - depth.min()),
        "depth_span_rel": _safe_float((depth.max() - depth.min()) / max(depth_median, float(args.eps))),
        "depth_iqr": float(q75 - q25),
        "depth_iqr_rel": float((q75 - q25) / max(depth_median, float(args.eps))),
        "transmission_span": _safe_float(t_scalar.max() - t_scalar.min()),
        "backscatter_span": _safe_float(b_scalar.max() - b_scalar.min()),
        "view_direction_spread_deg": _view_spread_deg(obs["ray_direction"][valid_rows].float(), float(args.eps)),
        "profile_denominator_mean": _safe_float(denominator.mean()),
        "profile_denominator_min": _safe_float(denominator.min()),
        "profile_denominator_signal_ratio": _safe_float(denominator.mean() / float(args.closure_signal_floor)),
        "geometry_weight_mean": _safe_float(weight.mean()),
        "geometry_weight_min": _safe_float(weight.min()),
        "geometry_weight_p10": _nearest_rank(weight, 0.10),
        "irls_inlier_ratio": _safe_float(inlier.float().mean()),
        "irls_effective_observation_count": _safe_float(effective_obs),
        "irls_effective_observation_fraction": _safe_float(effective_obs / float(valid.sum().item())),
        "weighted_profile_residual_mean": _weighted_mean(residual, solve_weight, float(args.eps)),
        "weighted_profile_residual_median": _weighted_quantile(residual, solve_weight, 0.50),
        "weighted_profile_residual_p90": _weighted_quantile(residual, solve_weight, 0.90),
        "maximum_single_observation_weight_share": _safe_float(solve_weight.max() / solve_sum),
        "jstar_channel_saturation_ratio": _safe_float(
            ((j_star < float(args.object_j_min)) | (j_star > float(args.object_j_max))).float().mean()
        ),
        "jstar_any_saturation": float(((j_star < float(args.object_j_min)) | (j_star > float(args.object_j_max))).any()),
    }
    features.update(
        _loo_features(
            gt,
            transmission,
            backscatter,
            weight,
            j_star,
            eps=float(args.eps),
            delta=float(args.irls_delta),
            max_weight=float(args.irls_max_weight),
        )
    )
    return features


def _load_gain_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _metric_value(record: Mapping[str, Any], section: str, key: str) -> float:
    try:
        return float(record[section][key])
    except Exception:
        return float("nan")


def _utility(record: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in LOWER_IS_BETTER:
        base = _metric_value(record, "base", key)
        cand = _metric_value(record, "candidate", key)
        denom = max(abs(base), float(args.eps))
        gain = base - cand
        out[f"{key}_gain"] = gain
        out[f"{key}_relative_gain"] = gain / denom
        out[f"{key}_relative_change"] = (cand - base) / denom
    closure_worsen = max(0.0, -out["closure_signal_floor_l1_relative_gain"])
    utility = (
        out["transfer_l1_relative_gain"]
        + out["object_j_variance_relative_gain"]
        + 0.5 * out["dc_recomposition_l1_relative_gain"]
        - closure_worsen
    )
    out["utility"] = utility
    out["closure_worsen_relative"] = closure_worsen
    out["positive_label"] = float(
        out["transfer_l1_gain"] > 0.0
        and out["object_j_variance_gain"] > 0.0
        and closure_worsen <= float(args.closure_worsen_threshold)
    )
    return out


def _tensor(records: Sequence[Mapping[str, Any]], key: str) -> Tensor:
    values = []
    for record in records:
        value = record.get(key, float("nan"))
        try:
            values.append(float(value))
        except Exception:
            values.append(float("nan"))
    return torch.tensor(values, dtype=torch.float32)


def _bootstrap_ci(x: Tensor, y: Tensor, samples: int, seed: int) -> Dict[str, float]:
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    if samples <= 0 or int(x.numel()) < 8:
        rho = _spearman(x, y)
        return {"mean": rho, "p025": rho, "p975": rho, "samples": 0}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    values = []
    n = int(x.numel())
    for _ in range(int(samples)):
        idx = torch.randint(0, n, (n,), generator=generator)
        values.append(_spearman(x[idx], y[idx]))
    t = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(t.mean().item()),
        "p025": _nearest_rank(t, 0.025),
        "p975": _nearest_rank(t, 0.975),
        "samples": int(samples),
    }


def _quartile_summary(score: Tensor, utility: Tensor, positive: Tensor) -> Dict[str, Any]:
    mask = torch.isfinite(score) & torch.isfinite(utility) & torch.isfinite(positive)
    score = score[mask]
    utility = utility[mask]
    positive = positive[mask]
    if int(score.numel()) < 4:
        return {}
    count = max(1, int(round(0.25 * int(score.numel()))))
    order = torch.argsort(score, descending=True)
    top = order[:count]
    bottom = order[-count:]
    return {
        "count": int(score.numel()),
        "quartile_count": int(count),
        "top_positive_rate": _safe_float(positive[top].mean()),
        "bottom_positive_rate": _safe_float(positive[bottom].mean()),
        "top_utility_mean": _safe_float(utility[top].mean()),
        "bottom_utility_mean": _safe_float(utility[bottom].mean()),
        "top_utility_median": _nearest_rank(utility[top], 0.50),
        "bottom_utility_median": _nearest_rank(utility[bottom], 0.50),
        "positive_rate_lift": _safe_float(positive[top].mean() - positive[bottom].mean()),
        "utility_mean_lift": _safe_float(utility[top].mean() - utility[bottom].mean()),
    }


def _summarize(records: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    utility = _tensor(records, "utility")
    positive = _tensor(records, "positive_label")
    summary: Dict[str, Any] = {
        "track_count": len(records),
        "positive_label_rate": _safe_float(positive[torch.isfinite(positive)].mean()) if len(records) else 0.0,
        "utility": _stats(utility),
        "relative_gain": {},
        "feature_stats": {},
        "feature_audit": {},
        "primary_features": PRIMARY_FEATURES,
    }
    for key in LOWER_IS_BETTER:
        summary["relative_gain"][key] = _stats(_tensor(records, f"{key}_relative_gain"))
    for feature, direction in FEATURE_DIRECTIONS.items():
        value = _tensor(records, feature)
        score = value * float(direction)
        stable_seed = int(args.seed) + sum((idx + 1) * ord(ch) for idx, ch in enumerate(feature))
        summary["feature_stats"][feature] = _stats(value)
        summary["feature_audit"][feature] = {
            "direction": "higher_is_more_reliable" if direction > 0 else "lower_is_more_reliable",
            "spearman_with_utility": _spearman(score, utility),
            "spearman_with_positive_label": _spearman(score, positive),
            "spearman_utility_bootstrap_ci": _bootstrap_ci(
                score,
                utility,
                int(args.bootstrap_samples),
                stable_seed,
            ),
            "quartile": _quartile_summary(score, utility, positive),
        }
    return summary


def run(args: argparse.Namespace) -> Dict[str, Any]:
    bank = torch.load(args.track_bank, map_location="cpu")
    obs = bank["observations"]
    gain_records = _load_gain_records(args.gain_records)
    if args.max_tracks > 0:
        gain_records = gain_records[: int(args.max_tracks)]

    records: List[Dict[str, Any]] = []
    for gain_record in gain_records:
        track_id = int(gain_record["track_id"])
        features = _track_features(obs, track_id, args)
        if features.get("valid_feature_track", 0.0) < 1.0:
            continue
        utility = _utility(gain_record, args)
        record = {
            "track_id": track_id,
            **features,
            **utility,
        }
        records.append(record)

    payload: Dict[str, Any] = {
        "diagnostic": "gmvc_profile_target_reliability",
        "scene_name": args.scene_name,
        "bank_name": args.bank_name,
        "track_bank": str(args.track_bank),
        "gain_records": str(args.gain_records),
        "gain_labels": {
            "base_label": args.base_label,
            "candidate_label": args.candidate_label,
            "positive_label": "transfer improves, J-var improves, closure relative worsening <= threshold",
            "closure_worsen_threshold": float(args.closure_worsen_threshold),
            "utility": "relative transfer gain + relative J-var gain + 0.5 relative recomposition gain - positive closure worsening",
        },
        "feature_source": "fixed-bank M1 medium parameters and track observations",
        "summary": _summarize(records, args),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_jsonl:
        jsonl_path = args.output_dir / "profile_target_reliability_records.jsonl"
        with jsonl_path.open("w", encoding="utf8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        payload["records_jsonl"] = str(jsonl_path)
    output_json = args.output_json or (args.output_dir / "profile_target_reliability_summary.json")
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--bank-name", required=True)
    parser.add_argument("--track-bank", type=Path, required=True)
    parser.add_argument("--gain-records", type=Path, required=True)
    parser.add_argument("--base-label", default="A0_MHOLD")
    parser.add_argument("--candidate-label", default="P30_MHOLD")
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--closure-signal-floor", type=float, default=0.03)
    parser.add_argument("--closure-worsen-threshold", type=float, default=0.005)
    parser.add_argument("--irls-delta", type=float, default=0.03)
    parser.add_argument("--irls-max-weight", type=float, default=1.0)
    parser.add_argument("--irls-inlier-threshold", type=float, default=0.5)
    parser.add_argument("--object-j-min", type=float, default=-0.1)
    parser.add_argument("--object-j-max", type=float, default=1.1)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
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
        "track_count": payload["summary"]["track_count"],
        "positive_label_rate": payload["summary"]["positive_label_rate"],
        "utility_mean": payload["summary"]["utility"]["mean"],
        "primary_features": {
            key: {
                "rho": payload["summary"]["feature_audit"][key]["spearman_with_utility"],
                "top_positive_rate": payload["summary"]["feature_audit"][key]["quartile"].get(
                    "top_positive_rate", 0.0
                ),
                "bottom_positive_rate": payload["summary"]["feature_audit"][key]["quartile"].get(
                    "bottom_positive_rate", 0.0
                ),
            }
            for key in PRIMARY_FEATURES
            if key in payload["summary"]["feature_audit"]
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
