"""Summary statistics for GMVC track diagnostics."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import torch


def _finite(values: Iterable[float]) -> torch.Tensor:
    tensor = torch.tensor(list(values), dtype=torch.float32)
    return tensor[torch.isfinite(tensor)]


def _nearest_rank(values: torch.Tensor, q: float) -> float:
    values = values.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return 0.0
    rank = max(1, min(values.numel(), math.ceil(float(q) * values.numel())))
    return float(values.kthvalue(rank).values.item())


def _stats(values: Iterable[float]) -> Dict[str, float]:
    tensor = _finite(values)
    if tensor.numel() == 0:
        return {"count": 0, "mean": 0.0, "p05": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "p05": _nearest_rank(tensor, 0.05),
        "p50": _nearest_rank(tensor, 0.50),
        "p90": _nearest_rank(tensor, 0.90),
        "p95": _nearest_rank(tensor, 0.95),
        "max": float(tensor.max().item()),
    }


def _corr(x_values: Iterable[float], y_values: Iterable[float]) -> float:
    x = _finite(x_values)
    y = _finite(y_values)
    n = min(x.numel(), y.numel())
    if n < 2:
        return 0.0
    x = x[:n]
    y = y[:n]
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((x * y).sum().item() / denom.item())


def _rgb_mean(rows: List[Dict[str, Any]], key: str) -> List[float]:
    if not rows:
        return [0.0, 0.0, 0.0]
    out = []
    for idx in range(3):
        out.append(float(torch.tensor([row[key][idx] for row in rows], dtype=torch.float32).mean().item()))
    return out


def summarize_gmvc_tracks(
    track_rows: List[Dict[str, Any]],
    counters: Dict[str, int],
    view_parameter_rows: List[Dict[str, Any]],
    min_views: int,
    relative_depth_span: float,
) -> Dict[str, Any]:
    """Aggregate per-track GMVC diagnostics into a compact JSON-ready summary."""

    length_ge_min = [row for row in track_rows if row["track_length"] >= min_views]
    final_rows = [
        row
        for row in length_ge_min
        if row["relative_depth_span"] >= relative_depth_span
        and row["valid_j_observation_count"] >= min_views
    ]
    sampled_tracks = max(counters.get("sampled_source_tracks", 0), 1)
    source_valid = max(counters.get("source_valid_pixels_total", 0), sampled_tracks)
    scale = float(source_valid) / float(sampled_tracks)

    total_t_checked = counters.get("target_valid_after_depth_std", 0) + counters.get("invalid_low_T_count", 0)
    valid_t_ratio = 0.0
    if total_t_checked > 0:
        valid_t_ratio = counters.get("target_valid_after_low_T", 0) / total_t_checked

    summary = {
        "counts": {
            **counters,
            "tracks_with_min_views": len(length_ge_min),
            "tracks_final": len(final_rows),
            "estimated_tracks_final_full_source": int(round(len(final_rows) * scale)),
        },
        "phase_a_gate": {
            "effective_tracks": len(final_rows),
            "estimated_effective_tracks_full_source": int(round(len(final_rows) * scale)),
            "length_ge_min_views_ratio": len(length_ge_min) / sampled_tracks,
            "relative_depth_span_ge_threshold_count": len(final_rows),
            "valid_observation_T_ratio": valid_t_ratio,
            "passes_sampled_gate": bool(
                len(final_rows) >= 10000
                and (len(length_ge_min) / sampled_tracks) >= 0.30
                and len(final_rows) >= 5000
                and valid_t_ratio >= 0.80
            ),
            "passes_estimated_gate": bool(
                int(round(len(final_rows) * scale)) >= 10000
                and (len(length_ge_min) / sampled_tracks) >= 0.30
                and int(round(len(final_rows) * scale)) >= 5000
                and valid_t_ratio >= 0.80
            ),
        },
        "track_length": _stats(row["track_length"] for row in track_rows),
        "relative_depth_span": _stats(row["relative_depth_span"] for row in length_ge_min),
        "alpha_mean": _stats(row["alpha_mean"] for row in final_rows),
        "depth_consistency_error": _stats(row["depth_consistency_error_mean"] for row in final_rows),
        "transmission_mean": _stats(row["transmission_mean"] for row in final_rows),
        "transmission_p05": _stats(row["transmission_p05"] for row in final_rows),
        "inverse_radiance_consistency_E_J": _stats(row["j_consistency_l1"] for row in final_rows),
        "medium_attn_track_variance": _stats(row["medium_attn_track_l1"] for row in final_rows),
        "medium_bs_track_variance": _stats(row["medium_bs_track_l1"] for row in final_rows),
        "b_inf_track_variance": _stats(row["b_inf_track_l1"] for row in final_rows),
        "endpoint_vs_actual_backscatter_l1": _stats(row["endpoint_actual_l1"] for row in final_rows),
        "invalid_j_observation_ratio": _stats(row["invalid_j_ratio"] for row in length_ge_min),
        "compensation_correlation": {
            "track_delta_attn_vs_b_inf": _corr(
                (v for row in final_rows for v in row["attn_delta_scalar"]),
                (v for row in final_rows for v in row["b_inf_delta_scalar"]),
            ),
            "track_delta_bs_vs_b_inf": _corr(
                (v for row in final_rows for v in row["bs_delta_scalar"]),
                (v for row in final_rows for v in row["b_inf_delta_scalar"]),
            ),
        },
        "view_parameter_means": view_parameter_rows,
        "view_parameter_mean_rgb": {
            "medium_attn": _rgb_mean(view_parameter_rows, "medium_attn_mean_rgb"),
            "medium_bs": _rgb_mean(view_parameter_rows, "medium_bs_mean_rgb"),
            "b_inf": _rgb_mean(view_parameter_rows, "b_inf_mean_rgb"),
        },
    }
    return summary
