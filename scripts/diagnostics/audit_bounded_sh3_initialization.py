#!/usr/bin/env python
"""Audit bounded-SH3 color initialization without training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict

import torch
from nerfstudio.utils.eval_utils import eval_setup
from plyfile import PlyData


C0 = 0.28209479177387814


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_seed_rgb(ply_path: Path) -> torch.Tensor:
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"].data
    rgb = torch.stack(
        [
            torch.from_numpy(vertex["red"].astype("float32")),
            torch.from_numpy(vertex["green"].astype("float32")),
            torch.from_numpy(vertex["blue"].astype("float32")),
        ],
        dim=-1,
    )
    return (rgb / 255.0).clamp(0.0, 1.0)


def _rgb_stats(value: torch.Tensor) -> Dict[str, float]:
    finite = value.detach().float().reshape(-1)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0}
    rank = max(1, min(int(finite.numel()), int(math.ceil(0.95 * float(finite.numel())))))
    return {
        "mean": float(finite.mean().item()),
        "p95": float(finite.kthvalue(rank).values.item()),
        "max": float(finite.max().item()),
        "min": float(finite.min().item()),
    }


def _legacy_forward_equivalence(config_path: Path, load_step: int, max_views: int) -> Dict[str, Any]:
    if not config_path.exists():
        return {"available": False, "reason": f"missing config: {config_path}"}

    def _update_config(config: Any) -> Any:
        config.load_step = int(load_step)
        return config

    _, pipeline, checkpoint_path, loaded_step = eval_setup(
        config_path,
        eval_num_rays_per_chunk=None,
        test_mode="test",
        update_config_callback=_update_config,
    )
    model = pipeline.model
    model.eval()
    mode = str(getattr(model.config, "intrinsic_color_parameterization", "legacy"))
    rows = []
    with torch.no_grad():
        for image_idx, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= int(max_views):
                break
            camera = camera.to(model.device) if hasattr(camera, "to") else camera
            out_a = model.get_outputs_for_camera(camera)
            out_b = model.get_outputs_for_camera(camera)
            row = {"image_idx": int(image_idx)}
            for key in ("pred_image", "rgb_object", "J_gaussian_raw"):
                a = out_a[key].detach().float()
                b = out_b[key].detach().float()
                row[f"{key}_max_abs_diff"] = float(torch.abs(a - b).max().item())
                row[f"{key}_mean_abs_diff"] = float(torch.abs(a - b).mean().item())
            rows.append(row)
    return {
        "available": True,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "requested_step": int(load_step),
        "loaded_step": int(loaded_step),
        "intrinsic_color_parameterization": mode,
        "comparison": "duplicate deterministic forward with bounded flag off in the loaded legacy config",
        "rows": rows,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    seed_rgb = _load_seed_rgb(args.seed_ply)
    legacy_dc = (seed_rgb - 0.5) / C0
    legacy_rgb = legacy_dc * C0 + 0.5
    bounded_dc = torch.logit(seed_rgb, eps=float(args.bound_logit_eps)) / C0
    bounded_logits = bounded_dc * C0
    bounded_rgb = torch.sigmoid(bounded_logits)
    diff = torch.abs(legacy_rgb - bounded_rgb)
    finite_rgb = torch.isfinite(bounded_rgb)
    summary: Dict[str, Any] = {
        "diagnostic": "bounded_sh3_initialization_audit",
        "scene": args.scene,
        "seed_ply": str(args.seed_ply),
        "num_seed_points": int(seed_rgb.shape[0]),
        "water_splatting_commit": _git_commit(repo),
        "seafree_reference_commit": args.seafree_reference_commit,
        "BOUND_LOGIT_EPS": float(args.bound_logit_eps),
        "definition": {
            "legacy_seed_rgb": "C0 * RGB2SH(seed_rgb) + 0.5",
            "bounded_seed_rgb": "sigmoid(C0 * (logit(seed_rgb, eps) / C0))",
            "features_rest": "zero in both seed initializations",
        },
        "rgb_equivalence": {
            "mean_abs_error": float(diff.mean().item()),
            "p95_abs_error": _rgb_stats(diff)["p95"],
            "max_abs_error": float(diff.max().item()),
            "target_mean_lt_1e-5": bool(float(diff.mean().item()) < 1e-5),
            "target_max_lt_1e-4": bool(float(diff.max().item()) < 1e-4),
        },
        "bounded_rgb": {
            **_rgb_stats(bounded_rgb),
            "all_finite": bool(finite_rgb.all().item()),
            "strictly_between_0_and_1": bool(((bounded_rgb > 0.0) & (bounded_rgb < 1.0)).all().item()),
        },
        "bounded_logits": _rgb_stats(bounded_logits),
        "legacy_forward_equivalence": _legacy_forward_equivalence(
            args.legacy_config, args.legacy_load_step, args.legacy_max_views
        )
        if args.legacy_config is not None
        else {"available": False, "reason": "no --legacy-config provided"},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "bounded_sh3_initialization_audit.json"
    csv_path = args.output_dir / "bounded_sh3_initialization_audit.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf8")
    with csv_path.open("w", newline="", encoding="utf8") as handle:
        fieldnames = [
            "scene",
            "num_seed_points",
            "BOUND_LOGIT_EPS",
            "mean_abs_error",
            "p95_abs_error",
            "max_abs_error",
            "bounded_rgb_min",
            "bounded_rgb_max",
            "all_finite",
            "strictly_between_0_and_1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "scene": summary["scene"],
                "num_seed_points": summary["num_seed_points"],
                "BOUND_LOGIT_EPS": summary["BOUND_LOGIT_EPS"],
                "mean_abs_error": summary["rgb_equivalence"]["mean_abs_error"],
                "p95_abs_error": summary["rgb_equivalence"]["p95_abs_error"],
                "max_abs_error": summary["rgb_equivalence"]["max_abs_error"],
                "bounded_rgb_min": summary["bounded_rgb"]["min"],
                "bounded_rgb_max": summary["bounded_rgb"]["max"],
                "all_finite": summary["bounded_rgb"]["all_finite"],
                "strictly_between_0_and_1": summary["bounded_rgb"]["strictly_between_0_and_1"],
            }
        )
    return {"json": str(json_path), "csv": str(csv_path), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="Curasao")
    parser.add_argument(
        "--seed-ply",
        type=Path,
        default=Path("undistorted_data/undistorted_Curasao/sparse/0/points3D.ply"),
    )
    parser.add_argument("--bound-logit-eps", type=float, default=1e-7)
    parser.add_argument("--legacy-config", type=Path, default=None)
    parser.add_argument("--legacy-load-step", type=int, default=15000)
    parser.add_argument("--legacy-max-views", type=int, default=1)
    parser.add_argument("--seafree-reference-commit", default="7797e97dae831029ac89ae9f37b3c3d69ec2cf6c")
    parser.add_argument("--output-dir", type=Path, default=Path("renders/dewater_bounded_sh3_scratch_20260808"))
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
