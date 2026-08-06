#!/usr/bin/env python
"""Audit observability/reliability weighting candidates for GMVC tracks.

This is an offline audit over track-level attribution JSONL files. It does not
use the measured gain to build the score; gain is only used afterward to test
whether a score would upweight tracks where P30-MHOLD helped.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor


DEFAULT_GAIN_KEYS = [
    "transfer_l1",
    "object_j_variance",
    "dc_cross_view_variance",
    "object_target_l1",
    "dc_recomposition_l1",
]


def _parse_label_path(items: Iterable[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=PATH, got {item}")
        label, path = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty label in {item}")
        out[label] = Path(path)
    return out


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _get(record: Mapping[str, Any], dotted: str, default: float = float("nan")) -> float:
    cur: Any = record
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    try:
        return float(cur)
    except Exception:
        return default


def _tensor(records: Sequence[Mapping[str, Any]], dotted: str) -> Tensor:
    return torch.tensor([_get(record, dotted) for record in records], dtype=torch.float32)


def _finite(values: Tensor) -> Tensor:
    return values[torch.isfinite(values)]


def _stats(values: Tensor) -> Dict[str, float]:
    values = _finite(values.detach().float().reshape(-1))
    if int(values.numel()) == 0:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0}
    sorted_values = values.sort().values

    def q(frac: float) -> float:
        idx = max(0, min(int(sorted_values.numel()) - 1, math.ceil(frac * int(sorted_values.numel())) - 1))
        return float(sorted_values[idx].item())

    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": float(sorted_values[-1].item()),
    }


def _rank_percentile(values: Tensor, invert: bool = False) -> Tensor:
    values = values.detach().float().reshape(-1)
    finite = torch.isfinite(values)
    safe = values.clone()
    if int(finite.sum().item()) == 0:
        return torch.full_like(values, 0.5)
    fill = values[finite].median()
    safe[~finite] = fill
    if invert:
        safe = -safe
    order = torch.argsort(safe)
    ranks = torch.empty_like(safe)
    ranks[order] = torch.arange(int(values.numel()), dtype=torch.float32)
    return (ranks + 0.5) / max(float(values.numel()), 1.0)


def _geomean(features: Sequence[Tensor], eps: float = 1e-4) -> Tensor:
    if not features:
        raise ValueError("At least one feature is required")
    stacked = torch.stack([feature.clamp(float(eps), 1.0) for feature in features], dim=0)
    return torch.exp(torch.log(stacked).mean(dim=0))


def _pearson(x: Tensor, y: Tensor) -> float:
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask].float()
    y = y[mask].float()
    if int(x.numel()) < 3:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.square().sum() * y.square().sum()).clamp_min(1e-20))
    return float((x * y).sum().item() / float(denom.item()))


def _spearman(x: Tensor, y: Tensor) -> float:
    return _pearson(_rank_percentile(x), _rank_percentile(y))


def _weighted_mean(values: Tensor, weights: Tensor) -> float:
    mask = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0)
    if int(mask.sum().item()) == 0:
        return 0.0
    return float((values[mask] * weights[mask]).sum().item() / weights[mask].sum().clamp_min(1e-12).item())


def _weighted_positive_ratio(values: Tensor, weights: Tensor) -> float:
    mask = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0)
    if int(mask.sum().item()) == 0:
        return 0.0
    return float((((values[mask] > 0).float()) * weights[mask]).sum().item() / weights[mask].sum().clamp_min(1e-12).item())


def _score_features(records: Sequence[Mapping[str, Any]]) -> Dict[str, Tensor]:
    depth = _rank_percentile(_tensor(records, "observability.depth_span_rel"))
    transmission = _rank_percentile(_tensor(records, "observability.bank_transmission_span"))
    backscatter = _rank_percentile(_tensor(records, "observability.bank_backscatter_span"))
    view = _rank_percentile(_tensor(records, "observability.view_angle_span_deg"))
    obs_count = _rank_percentile(_tensor(records, "observability.obs_count"))
    geom_mean = _rank_percentile(_tensor(records, "observability.geometry_weight_mean"))
    geom_p10 = _rank_percentile(_tensor(records, "observability.geometry_weight_p10"))
    hessian = _rank_percentile(_tensor(records, "observability.bank_hessian"))
    irls = _rank_percentile(_tensor(records, "base.irls_effective_weight_ratio"))
    residual_inv = _rank_percentile(_tensor(records, "base.track_profile_residual"), invert=True)
    j_inside = 1.0 - _tensor(records, "base.j_star_outside").clamp(0.0, 1.0)
    proxy = _tensor(records, "base.proxy_available_fraction").clamp(0.0, 1.0)

    observability = _geomean([depth, transmission, backscatter])
    observability_view = _geomean([depth, transmission, backscatter, view, obs_count])
    geometry = _geomean([geom_mean, geom_p10, hessian])
    reliability = _geomean([geometry, irls, residual_inv, j_inside.clamp_min(1e-4), proxy.clamp_min(1e-4)])
    return {
        "obs_dtb": observability,
        "obs_dtb_view": observability_view,
        "geometry": geometry,
        "reliability": reliability,
        "obs_x_reliability": _geomean([observability, reliability]),
        "obs_view_x_reliability": _geomean([observability_view, reliability]),
        "obs_x_geom": _geomean([observability, geometry]),
        "residual_inv_only": residual_inv,
        "irls_only": irls,
    }


def _evaluate_score(
    score: Tensor,
    gains: Mapping[str, Tensor],
    top_fractions: Sequence[float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "score_stats": _stats(score),
        "gain_spearman": {},
        "weighted": {},
        "top_fraction": {},
    }
    for gain_key, gain in gains.items():
        all_finite = gain[torch.isfinite(gain)]
        all_mean = float(all_finite.mean().item()) if int(all_finite.numel()) else 0.0
        all_positive = float((all_finite > 0).float().mean().item()) if int(all_finite.numel()) else 0.0
        weighted_mean = _weighted_mean(gain, score)
        weighted_pos = _weighted_positive_ratio(gain, score)
        out["gain_spearman"][gain_key] = _spearman(score, gain)
        out["weighted"][gain_key] = {
            "mean_gain": weighted_mean,
            "positive_ratio": weighted_pos,
            "mean_gain_lift": weighted_mean - all_mean,
            "positive_ratio_lift": weighted_pos - all_positive,
        }
    for frac in top_fractions:
        count = max(1, int(round(float(frac) * int(score.numel()))))
        order = torch.argsort(score, descending=True)[:count]
        block: Dict[str, Any] = {"count": int(count), "score_mean": float(score[order].mean().item()), "gains": {}}
        for gain_key, gain in gains.items():
            top_gain = gain[order]
            top_gain = top_gain[torch.isfinite(top_gain)]
            all_finite = gain[torch.isfinite(gain)]
            all_mean = float(all_finite.mean().item()) if int(all_finite.numel()) else 0.0
            all_positive = float((all_finite > 0).float().mean().item()) if int(all_finite.numel()) else 0.0
            if int(top_gain.numel()) == 0:
                mean_gain = 0.0
                positive = 0.0
            else:
                mean_gain = float(top_gain.mean().item())
                positive = float((top_gain > 0).float().mean().item())
            block["gains"][gain_key] = {
                "mean_gain": mean_gain,
                "positive_ratio": positive,
                "mean_gain_lift": mean_gain - all_mean,
                "positive_ratio_lift": positive - all_positive,
            }
        out["top_fraction"][f"{float(frac):.2f}"] = block
    return out


def _summarize_records(
    label: str,
    path: Path,
    gain_keys: Sequence[str],
    top_fractions: Sequence[float],
) -> Dict[str, Any]:
    records = _read_jsonl(path)
    gains = {key: _tensor(records, f"gain.{key}") for key in gain_keys}
    scores = _score_features(records)
    summary: Dict[str, Any] = {
        "label": label,
        "records_path": str(path),
        "track_count": len(records),
        "gain": {},
        "scores": {},
    }
    for key, values in gains.items():
        finite = values[torch.isfinite(values)]
        summary["gain"][key] = {
            **_stats(values),
            "positive_ratio": float((finite > 0).float().mean().item()) if int(finite.numel()) else 0.0,
        }
    for score_name, score in scores.items():
        summary["scores"][score_name] = _evaluate_score(score, gains, top_fractions)
    return summary


def _aggregate(
    datasets: Mapping[str, Dict[str, Any]],
    score_names: Sequence[str],
    gain_keys: Sequence[str],
    top_fractions: Sequence[float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for score_name in score_names:
        out[score_name] = {}
        for gain_key in gain_keys:
            score_block: Dict[str, Any] = {}
            weighted_lifts = [
                datasets[label]["scores"][score_name]["weighted"][gain_key]["positive_ratio_lift"]
                for label in datasets
            ]
            score_block["weighted_positive_lift_mean"] = float(sum(weighted_lifts) / max(len(weighted_lifts), 1))
            score_block["weighted_positive_lift_all_positive"] = bool(all(value > 0 for value in weighted_lifts))
            top_blocks: Dict[str, Any] = {}
            for frac in top_fractions:
                frac_key = f"{float(frac):.2f}"
                lifts = [
                    datasets[label]["scores"][score_name]["top_fraction"][frac_key]["gains"][gain_key]["positive_ratio_lift"]
                    for label in datasets
                ]
                mean_gains = [
                    datasets[label]["scores"][score_name]["top_fraction"][frac_key]["gains"][gain_key]["mean_gain_lift"]
                    for label in datasets
                ]
                top_blocks[frac_key] = {
                    "positive_ratio_lift_mean": float(sum(lifts) / max(len(lifts), 1)),
                    "positive_ratio_lift_min": float(min(lifts)) if lifts else 0.0,
                    "positive_ratio_lift_all_positive": bool(all(value > 0 for value in lifts)),
                    "mean_gain_lift_mean": float(sum(mean_gains) / max(len(mean_gains), 1)),
                    "mean_gain_lift_min": float(min(mean_gains)) if mean_gains else 0.0,
                    "mean_gain_lift_all_positive": bool(all(value > 0 for value in mean_gains)),
                }
            score_block["top_fraction"] = top_blocks
            out[score_name][gain_key] = score_block
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", action="append", default=[], help="Dataset as LABEL=track_observability_gain_records.jsonl")
    parser.add_argument("--gain-keys", default=",".join(DEFAULT_GAIN_KEYS))
    parser.add_argument("--top-fractions", default="0.10,0.25,0.50")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    record_paths = _parse_label_path(args.records)
    if not record_paths:
        raise ValueError("At least one --records LABEL=PATH is required")
    gain_keys = [item.strip() for item in args.gain_keys.split(",") if item.strip()]
    top_fractions = [float(item.strip()) for item in args.top_fractions.split(",") if item.strip()]

    datasets = {
        label: _summarize_records(label, path, gain_keys, top_fractions)
        for label, path in record_paths.items()
    }
    score_names = list(next(iter(datasets.values()))["scores"].keys())
    payload = {
        "diagnostic": "gmvc_observability_weight_audit",
        "gain_keys": gain_keys,
        "top_fractions": top_fractions,
        "datasets": datasets,
        "aggregate": _aggregate(datasets, score_names, gain_keys, top_fractions),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf8")

    compact: Dict[str, Any] = {"output_json": str(args.output_json), "score_ranking": {}}
    for score_name in score_names:
        transfer = payload["aggregate"][score_name]["transfer_l1"]["top_fraction"]["0.25"]
        jvar = payload["aggregate"][score_name]["object_j_variance"]["top_fraction"]["0.25"]
        compact["score_ranking"][score_name] = {
            "top25_transfer_pos_lift_mean": transfer["positive_ratio_lift_mean"],
            "top25_transfer_all_positive": transfer["positive_ratio_lift_all_positive"],
            "top25_jvar_pos_lift_mean": jvar["positive_ratio_lift_mean"],
            "top25_jvar_all_positive": jvar["positive_ratio_lift_all_positive"],
        }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
