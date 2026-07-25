#!/usr/bin/env python
"""Check analytic closed-tail medium decomposition without changing CUDA outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import torch

from nerfstudio.utils.eval_utils import eval_setup


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _stats(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        return {"mean": 0.0, "max": 0.0, "p95": 0.0}
    return {
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
    }


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    config, pipeline, checkpoint_path, step = eval_setup(args.load_config)
    pipeline.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images: List[Dict[str, Any]] = []
    all_medium_abs: List[torch.Tensor] = []
    all_rgb_abs: List[torch.Tensor] = []

    with torch.no_grad():
        for image_idx, (camera, _batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= args.max_images:
                break
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            rgb_medium = outputs["rgb_medium"].detach().float()
            rgb_medium_finite = outputs["rgb_medium_finite"].detach().float()
            tail_medium = outputs["tail_medium_original"].detach().float()
            rgb_object = outputs["rgb_object"].detach().float()
            rgb = outputs["rgb"].detach().float()

            medium_recompose = rgb_medium_finite + tail_medium
            rgb_recompose = rgb_object + medium_recompose
            medium_abs = (medium_recompose - rgb_medium).abs()
            rgb_abs = (rgb_recompose - rgb).abs()
            all_medium_abs.append(medium_abs.reshape(-1).cpu())
            all_rgb_abs.append(rgb_abs.reshape(-1).cpu())
            images.append(
                {
                    "image_index": image_idx,
                    "medium_abs_error": _stats(medium_abs),
                    "rgb_abs_error": _stats(rgb_abs),
                }
            )

    medium_all = torch.cat(all_medium_abs) if all_medium_abs else torch.empty(0)
    rgb_all = torch.cat(all_rgb_abs) if all_rgb_abs else torch.empty(0)
    aggregate = {
        "medium_abs_error": _stats(medium_all),
        "rgb_abs_error": _stats(rgb_all),
        "passes_t1_threshold": bool(
            medium_all.numel() > 0
            and rgb_all.numel() > 0
            and medium_all.max().item() < args.max_abs_threshold
            and medium_all.mean().item() < args.mean_abs_threshold
            and rgb_all.max().item() < args.max_abs_threshold
            and rgb_all.mean().item() < args.mean_abs_threshold
        ),
        "max_abs_threshold": args.max_abs_threshold,
        "mean_abs_threshold": args.mean_abs_threshold,
    }

    repo = Path(__file__).resolve().parents[2]
    result = {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "load_config": str(args.load_config),
        "git_commit": _git_commit(repo),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "aggregate": aggregate,
        "images": images,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--max-abs-threshold", type=float, default=1e-5)
    parser.add_argument("--mean-abs-threshold", type=float, default=1e-6)
    args = parser.parse_args()

    result = diagnose(args)
    output_json = args.output_dir / "closed_tail_recompose_diagnostic.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"saved={output_json}")


if __name__ == "__main__":
    main()
