#!/usr/bin/env python
"""Summarize scene observability from GMVC geometry-only track banks."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import torch
from torch import Tensor


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _parse_label_path(items: Iterable[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=PATH, got: {item}")
        label, path = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty label in: {item}")
        out[label] = Path(path.strip())
    return out


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


def _select_tracks(obs: Mapping[str, Tensor], max_tracks: int, seed: int) -> Tensor:
    track_ids = obs["track_ids"].long()
    if max_tracks > 0 and int(track_ids.numel()) > max_tracks:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        keep = torch.randperm(int(track_ids.numel()), generator=generator)[:max_tracks]
        track_ids = track_ids[keep]
    return track_ids.sort().values


def _split_tracks(track_ids: Tensor, train_fraction: float, seed: int) -> Tuple[Tensor, Tensor]:
    if int(track_ids.numel()) == 0:
        return track_ids, track_ids
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 9176)
    perm = track_ids[torch.randperm(int(track_ids.numel()), generator=generator)]
    train_count = int(round(float(train_fraction) * int(track_ids.numel())))
    train_count = max(1, min(train_count, int(track_ids.numel()) - 1)) if int(track_ids.numel()) > 1 else int(track_ids.numel())
    return perm[:train_count].sort().values, perm[train_count:].sort().values


def _indices_for_tracks(obs: Mapping[str, Tensor], track_ids: Tensor) -> Tensor:
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


def _medium_terms(obs: Mapping[str, Tensor], row_indices: Tensor, eps: float) -> Tuple[Tensor, Tensor, Tensor]:
    depth = obs["fixed_depth"][row_indices].float().reshape(-1, 1)
    attn = obs["bank_medium_attn"][row_indices].float()
    bs = obs["bank_medium_bs"][row_indices].float()
    b_inf = obs["bank_b_inf"][row_indices].float()
    transmission = torch.exp(-(attn * depth).clamp_min(0.0))
    backscatter = b_inf * (1.0 - torch.exp(-(bs * depth).clamp_min(0.0)))
    return depth.reshape(-1).clamp_min(float(eps)), transmission, backscatter


def _track_span_stats(values: Tensor, obs: Mapping[str, Tensor], selected_tracks: Tensor) -> Dict[str, float]:
    spans = []
    starts = obs["track_starts"].long()
    lengths = obs["track_lengths"].long()
    for track_id in selected_tracks.long().tolist():
        start = int(starts[track_id].item())
        length = int(lengths[track_id].item())
        if length < 2:
            continue
        track_values = values[start : start + length].float()
        finite = torch.isfinite(track_values)
        if track_values.ndim > 1:
            finite = finite.all(dim=-1)
        if int(finite.sum().item()) < 2:
            continue
        valid = track_values[finite]
        if valid.ndim == 1:
            span = valid.max() - valid.min()
        else:
            span = (valid.max(dim=0).values - valid.min(dim=0).values).mean()
        spans.append(span.reshape(1))
    return _stats(torch.cat(spans) if spans else torch.empty((0,), dtype=torch.float32))


def _relative_depth_span_stats(depth: Tensor, obs: Mapping[str, Tensor], selected_tracks: Tensor, eps: float) -> Dict[str, float]:
    spans = []
    starts = obs["track_starts"].long()
    lengths = obs["track_lengths"].long()
    for track_id in selected_tracks.long().tolist():
        start = int(starts[track_id].item())
        length = int(lengths[track_id].item())
        if length < 2:
            continue
        track_depth = depth[start : start + length].float()
        track_depth = track_depth[torch.isfinite(track_depth)]
        if int(track_depth.numel()) < 2:
            continue
        spans.append(((track_depth.max() - track_depth.min()) / track_depth.median().clamp_min(float(eps))).reshape(1))
    return _stats(torch.cat(spans) if spans else torch.empty((0,), dtype=torch.float32))


def _load_fixed_metrics(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf8"))
    heldout = data.get("metrics", {}).get("heldout", {})
    selected = data.get("selected", {})
    return {
        "path": str(path),
        "step": data.get("step"),
        "selected": selected,
        "heldout": {
            "transfer_l1": heldout.get("transfer_l1", 0.0),
            "object_j_variance": heldout.get("object_j_variance", 0.0),
            "closure_l1": heldout.get("closure_l1", 0.0),
            "closure_signal_floor_l1": heldout.get("closure_signal_floor_l1", 0.0),
            "consensus_j_reconstruction_l1": heldout.get("consensus_j_reconstruction_l1", 0.0),
            "object_target_l1": heldout.get("object_target_l1", 0.0),
            "dc_cross_view_variance": heldout.get("dc_cross_view_variance", 0.0),
            "dc_recomposition_l1": heldout.get("dc_recomposition_l1", 0.0),
            "proxy_available_fraction": heldout.get("proxy_available_fraction", 0.0),
            "transmission_span_stats": heldout.get("transmission_span_stats", {}),
            "depth_span_rel_stats": heldout.get("depth_span_rel_stats", {}),
        },
    }


def summarize_bank(label: str, bank_path: Path, fixed_path: Path | None, args: argparse.Namespace) -> Dict[str, Any]:
    bank = torch.load(bank_path, map_location="cpu")
    obs = bank["observations"]
    selected_tracks = _select_tracks(obs, int(args.max_tracks), int(args.seed))
    train_tracks, heldout_tracks = _split_tracks(selected_tracks, float(args.train_fraction), int(args.seed))
    selected_rows = _indices_for_tracks(obs, selected_tracks)
    heldout_rows = _indices_for_tracks(obs, heldout_tracks)

    depth, transmission, backscatter = _medium_terms(obs, torch.arange(int(obs["track_id"].numel()), dtype=torch.long), float(args.eps))
    t_scalar = transmission.mean(dim=-1)
    b_scalar = backscatter.mean(dim=-1)
    track_lengths = obs["track_lengths"][selected_tracks].float()
    cameras = obs["camera_index"][selected_rows].long() if int(selected_rows.numel()) else torch.empty((0,), dtype=torch.long)
    heldout_cameras = obs["camera_index"][heldout_rows].long() if int(heldout_rows.numel()) else torch.empty((0,), dtype=torch.long)
    camera_count = int(torch.unique(cameras).numel()) if int(cameras.numel()) else 0
    heldout_camera_count = int(torch.unique(heldout_cameras).numel()) if int(heldout_cameras.numel()) else 0

    metadata = bank.get("metadata", {})
    counters = metadata.get("counters", {})
    sampled_tracks = int(counters.get("sampled_source_tracks", 0))
    accepted_tracks = int(metadata.get("v2_track_count", int(obs["track_ids"].numel())))
    acceptance_ratio = float(accepted_tracks / max(sampled_tracks, 1))
    heldout_row_ratio = float(int(heldout_rows.numel()) / max(int(selected_rows.numel()), 1))

    result: Dict[str, Any] = {
        "scene": label,
        "track_bank": str(bank_path),
        "bank_step": metadata.get("step"),
        "split": metadata.get("split"),
        "view_count": metadata.get("view_count"),
        "geometry_only_bank": bool(metadata.get("track_config", {}).get("geometry_only_bank", False)),
        "sampled_source_tracks": sampled_tracks,
        "accepted_tracks": accepted_tracks,
        "accepted_observations": int(metadata.get("v2_observation_count", int(obs["track_id"].numel()))),
        "acceptance_ratio": acceptance_ratio,
        "selected_track_count": int(selected_tracks.numel()),
        "selected_observation_count": int(selected_rows.numel()),
        "observation_length": _stats(track_lengths),
        "depth_span": _track_span_stats(depth, obs, selected_tracks),
        "relative_depth_span": _relative_depth_span_stats(depth, obs, selected_tracks, float(args.eps)),
        "transmission_span": _track_span_stats(t_scalar, obs, selected_tracks),
        "backscatter_span": _track_span_stats(b_scalar, obs, selected_tracks),
        "heldout": {
            "train_fraction": float(args.train_fraction),
            "track_count": int(heldout_tracks.numel()),
            "track_ratio": float(int(heldout_tracks.numel()) / max(int(selected_tracks.numel()), 1)),
            "observation_count": int(heldout_rows.numel()),
            "observation_ratio": heldout_row_ratio,
            "camera_count": heldout_camera_count,
            "camera_ratio": float(heldout_camera_count / max(camera_count, 1)),
        },
        "a0_fixed_bank": _load_fixed_metrics(fixed_path),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", action="append", default=[], help="Scene bank as SCENE=PATH")
    parser.add_argument("--fixed-metrics", action="append", default=[], help="A0 fixed metrics as SCENE=PATH")
    parser.add_argument("--max-tracks", type=int, default=30000)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    banks = _parse_label_path(args.bank)
    fixed = _parse_label_path(args.fixed_metrics)
    if not banks:
        raise ValueError("At least one --bank SCENE=PATH is required")
    results = [summarize_bank(label, path, fixed.get(label), args) for label, path in banks.items()]
    payload = {
        "diagnostic": "gmvc_geometry_observability_summary",
        "max_tracks": int(args.max_tracks),
        "train_fraction": float(args.train_fraction),
        "seed": int(args.seed),
        "scenes": results,
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
