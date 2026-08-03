#!/usr/bin/env python
"""Diagnose whether M1 exposes enough multi-view medium-calibration signal."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import torch

from nerfstudio.utils.eval_utils import eval_setup
from water_splatting.medium_calibration import (
    GMVCTrackConfig,
    build_gmvc_track_metrics,
    render_gmvc_views,
    summarize_gmvc_tracks,
)


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
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
    pipeline.eval()
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
        j_clamp_min=args.j_clamp_min,
        j_clamp_max=args.j_clamp_max,
        edge_margin=args.edge_margin,
        samples_per_view=args.samples_per_view,
        seed=args.seed,
        target_neighbor_window=args.target_neighbor_window,
    )
    views = render_gmvc_views(pipeline, args.split, args.max_images)
    track_rows, counters, view_rows = build_gmvc_track_metrics(views, track_cfg)
    summary = summarize_gmvc_tracks(
        track_rows=track_rows,
        counters=counters,
        view_parameter_rows=view_rows,
        min_views=track_cfg.min_views,
        relative_depth_span=track_cfg.relative_depth_span,
    )
    result: Dict[str, Any] = {
        "diagnostic": "gmvc_identifiability_phase_a",
        "experiment_name": getattr(config, "experiment_name", ""),
        "method_name": getattr(config, "method_name", ""),
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "split": args.split,
        "view_count": len(views),
        "track_config": track_cfg.__dict__,
        "summary": summary,
        "views": [view.metadata() for view in views],
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (args.output_dir / "gmvc_identifiability.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    if args.save_track_metrics:
        torch.save({"track_rows": track_rows, "counters": counters, "view_rows": view_rows}, args.output_dir / "gmvc_track_metrics.pt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--test-mode", default="inference")
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images in the selected split.")
    parser.add_argument("--samples-per-view", type=int, default=4096)
    parser.add_argument("--target-neighbor-window", type=int, default=0, help="0 compares every other rendered view.")
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--save-track-metrics", action="store_true")
    args = parser.parse_args()

    result = diagnose(args)
    print(
        json.dumps(
            {
                "checkpoint": result["checkpoint"],
                "step": result["step"],
                "split": result["split"],
                "view_count": result["view_count"],
                "counts": result["summary"]["counts"],
                "phase_a_gate": result["summary"]["phase_a_gate"],
                "E_J": result["summary"]["inverse_radiance_consistency_E_J"],
                "compensation_correlation": result["summary"]["compensation_correlation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
