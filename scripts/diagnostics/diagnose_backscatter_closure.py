#!/usr/bin/env python
"""Diagnose explicit B_inf=A closure against the current implicit M1 tail."""

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
    all_tied_abs: List[torch.Tensor] = []
    all_model_abs: List[torch.Tensor] = []
    all_binf_abs: List[torch.Tensor] = []

    with torch.no_grad():
        for image_idx, (camera, _batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= args.max_images:
                break
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            implicit_rgb = outputs["rgb_implicit_tail"].detach().float()
            tied_rgb = (
                outputs["rgb_object"].detach().float()
                + outputs["rgb_medium_finite"].detach().float()
                + outputs["tail_weight_last"].detach().float() * outputs["medium_rgb"].detach().float()
            )
            tied_abs = (tied_rgb - implicit_rgb).abs()
            model_abs = (outputs["rgb"].detach().float() - implicit_rgb).abs()
            all_tied_abs.append(tied_abs.reshape(-1).cpu())
            all_model_abs.append(model_abs.reshape(-1).cpu())

            entry: Dict[str, Any] = {
                "image_index": image_idx,
                "explicit_tied_vs_implicit_rgb_abs_error": _stats(tied_abs),
                "model_rgb_vs_implicit_rgb_abs_error": _stats(model_abs),
            }
            if "b_inf_minus_A_abs" in outputs:
                binf_abs = outputs["b_inf_minus_A_abs"].detach().float()
                all_binf_abs.append(binf_abs.reshape(-1).cpu())
                entry["b_inf_minus_A_abs_error"] = _stats(binf_abs)
            images.append(entry)

    tied_all = torch.cat(all_tied_abs) if all_tied_abs else torch.empty(0)
    model_all = torch.cat(all_model_abs) if all_model_abs else torch.empty(0)
    binf_all = torch.cat(all_binf_abs) if all_binf_abs else torch.empty(0)
    aggregate = {
        "explicit_tied_vs_implicit_rgb_abs_error": _stats(tied_all),
        "model_rgb_vs_implicit_rgb_abs_error": _stats(model_all),
        "b_inf_minus_A_abs_error": _stats(binf_all),
        "passes_a0_a1_threshold": bool(
            tied_all.numel() > 0
            and tied_all.mean().item() < args.mean_abs_threshold
            and tied_all.max().item() < args.max_abs_threshold
        ),
        "max_abs_threshold": args.max_abs_threshold,
        "mean_abs_threshold": args.mean_abs_threshold,
    }

    repo = Path(__file__).resolve().parents[2]
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--max-abs-threshold", type=float, default=1e-5)
    parser.add_argument("--mean-abs-threshold", type=float, default=1e-6)
    args = parser.parse_args()

    result = diagnose(args)
    output_json = args.output_dir / "backscatter_closure_diagnostic.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"saved={output_json}")


if __name__ == "__main__":
    main()
