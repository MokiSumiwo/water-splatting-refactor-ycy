#!/usr/bin/env python
"""Evaluate per-image reconstruction metrics for a saved checkpoint step."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from nerfstudio.utils.eval_utils import eval_setup


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _mean_std(images: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    values = np.asarray([float(item[key]) for item in images], dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    def _update_config(config: Any) -> Any:
        config.load_step = int(args.load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(
        args.load_config,
        eval_num_rays_per_chunk=args.eval_num_rays_per_chunk,
        test_mode=args.test_mode,
        update_config_callback=_update_config,
    )
    pipeline.eval()
    model = pipeline.model

    images: List[Dict[str, Any]] = []
    max_images = args.max_images if args.max_images > 0 else 10**9
    with torch.no_grad():
        for image_idx, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= max_images:
                break
            outputs = model.get_outputs_for_camera(camera=camera)
            metrics, _images = model.get_image_metrics_and_images(outputs, batch)
            row: Dict[str, Any] = {"image_index": image_idx}
            row.update({key: float(value) for key, value in metrics.items()})
            images.append(row)

    aggregate = {
        key: _mean_std(images, key)
        for key in (
            "psnr",
            "ssim",
            "lpips",
            "J_white_ratio",
            "J_saturation_ratio",
            "J_red_dominance_ratio",
            "J_green_dominance_ratio",
            "J_blue_dominance_ratio",
        )
        if images and key in images[0]
    }

    repo = Path(__file__).resolve().parents[2]
    result = {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "requested_load_step": int(args.load_step),
        "loaded_step": int(step),
        "test_mode": args.test_mode,
        "git_commit": _git_commit(repo),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "aggregate": aggregate,
        "images": images,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--test-mode", choices=("test", "val"), default="test")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=None)
    args = parser.parse_args()

    result = evaluate(args)
    summary = {
        "experiment_name": result["experiment_name"],
        "checkpoint": result["checkpoint"],
        "loaded_step": result["loaded_step"],
        "aggregate": result["aggregate"],
        "output_json": str(args.output_json),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

