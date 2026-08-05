#!/usr/bin/env python
"""Evaluate Nerfstudio image metrics for an explicit checkpoint step."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import torch
from nerfstudio.utils.eval_utils import eval_setup


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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
    render_output_path = args.render_output_path
    if render_output_path is not None:
        render_output_path.mkdir(parents=True, exist_ok=True)
    metrics_dict = pipeline.get_average_eval_image_metrics(output_path=render_output_path, get_std=True)
    result: Dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "load_config": str(args.load_config),
        "test_mode": args.test_mode,
        "results": metrics_dict,
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--test-mode", choices=["test", "val", "inference"], default="test")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--render-output-path", type=Path, default=None)
    args = parser.parse_args()

    result = run(args)
    compact = {
        "checkpoint": result["checkpoint"],
        "step": result["step"],
        "results": {
            "psnr": result["results"].get("psnr", 0.0),
            "ssim": result["results"].get("ssim", 0.0),
            "lpips": result["results"].get("lpips", 0.0),
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
