#!/usr/bin/env python
"""Check bounded-SH3 smoke checkpoints for finite losses and gradients."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from nerfstudio.utils.eval_utils import eval_setup


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _parse_run(text: str) -> Tuple[str, Path, int]:
    parts = text.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"--run must be NAME:STEP:CONFIG, got {text}")
    return parts[0], Path(parts[2]), int(parts[1])


def _grad_stats(param: torch.Tensor | None) -> Dict[str, Any]:
    if param is None or param.grad is None:
        return {"available": False, "finite": False, "norm": 0.0, "max_abs": 0.0}
    grad = param.grad.detach().float()
    finite = torch.isfinite(grad)
    return {
        "available": True,
        "finite": bool(finite.all().item()),
        "norm": float(torch.linalg.vector_norm(grad[finite]).item()) if finite.any() else 0.0,
        "max_abs": float(grad[finite].abs().max().item()) if finite.any() else 0.0,
    }


def _param_grad_stats(module: torch.nn.Module) -> Dict[str, Any]:
    grads = []
    for param in module.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().float().reshape(-1))
    if not grads:
        return {"available": False, "finite": False, "norm": 0.0, "max_abs": 0.0}
    flat = torch.cat(grads)
    finite = torch.isfinite(flat)
    return {
        "available": True,
        "finite": bool(finite.all().item()),
        "norm": float(torch.linalg.vector_norm(flat[finite]).item()) if finite.any() else 0.0,
        "max_abs": float(flat[finite].abs().max().item()) if finite.any() else 0.0,
    }


def _tensor_range(value: torch.Tensor | None) -> Dict[str, Any]:
    if value is None:
        return {"available": False, "finite": False, "min": 0.0, "max": 0.0}
    flat = value.detach().float().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    return {
        "available": True,
        "finite": bool(finite.numel() == flat.numel()),
        "min": float(finite.min().item()) if finite.numel() else 0.0,
        "max": float(finite.max().item()) if finite.numel() else 0.0,
    }


def _check_run(name: str, config_path: Path, load_step: int, train_index: int) -> Dict[str, Any]:
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
    model.train()
    dataset = pipeline.datamanager.train_dataset
    index = min(max(int(train_index), 0), len(dataset.cameras) - 1)
    camera = dataset.cameras[index : index + 1]
    camera = camera.to(model.device) if hasattr(camera, "to") else camera
    batch = {"image": dataset[index]["image"]}
    model.zero_grad(set_to_none=True)
    outputs = model(camera)
    metrics = model.get_metrics_dict(outputs, batch)
    losses = model.get_loss_dict(outputs, batch, metrics)
    total_loss = sum(value for value in losses.values())
    total_loss.backward()
    row = {
        "run": name,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "requested_step": int(load_step),
        "loaded_step": int(loaded_step),
        "train_index": int(index),
        "loss": float(total_loss.detach().item()),
        "loss_finite": bool(torch.isfinite(total_loss.detach()).item()),
        "psnr": float(metrics["psnr"].detach().item()),
        "psnr_finite": bool(torch.isfinite(metrics["psnr"].detach()).item()),
        "intrinsic_color_parameterization": str(getattr(model.config, "intrinsic_color_parameterization", "legacy")),
        "direct_optical_depth_scale": float(getattr(model.config, "direct_optical_depth_scale", 1.0)),
        "features_dc_grad": _grad_stats(model.gauss_params.get("features_dc")),
        "features_rest_grad": _grad_stats(model.gauss_params.get("features_rest")),
        "medium_mlp_grad": _param_grad_stats(model.medium_mlp),
        "gaussian_view_rgb_range": _tensor_range(outputs.get("gaussian_view_rgb")),
        "gaussian_view_logits_range": _tensor_range(outputs.get("gaussian_view_logits")),
        "gaussian_sigmoid_derivative_range": _tensor_range(outputs.get("gaussian_sigmoid_derivative")),
    }
    row["all_checks_pass"] = bool(
        row["loss_finite"]
        and row["psnr_finite"]
        and row["features_dc_grad"]["finite"]
        and row["features_rest_grad"]["finite"]
        and row["medium_mlp_grad"]["finite"]
        and row["gaussian_view_rgb_range"]["finite"]
        and row["gaussian_view_logits_range"]["finite"]
        and row["gaussian_sigmoid_derivative_range"]["finite"]
        and row["gaussian_view_rgb_range"]["min"] > 0.0
        and row["gaussian_view_rgb_range"]["max"] < 1.0
    )
    return row


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rows = [_check_run(name, config, step, args.train_index) for name, config, step in map(_parse_run, args.run)]
    payload = {
        "diagnostic": "bounded_sh3_smoke_gradient_check",
        "water_splatting_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "rows": rows,
        "all_runs_pass": all(bool(row["all_checks_pass"]) for row in rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "bounded_sh3_smoke_gradient_check.json"
    csv_path = args.output_dir / "bounded_sh3_smoke_gradient_check.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf8")
    with csv_path.open("w", newline="", encoding="utf8") as handle:
        fieldnames = [
            "run",
            "requested_step",
            "loaded_step",
            "loss",
            "loss_finite",
            "psnr",
            "psnr_finite",
            "direct_optical_depth_scale",
            "intrinsic_color_parameterization",
            "features_dc_grad_finite",
            "features_rest_grad_finite",
            "medium_mlp_grad_finite",
            "c_min",
            "c_max",
            "logit_min",
            "logit_max",
            "all_checks_pass",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run": row["run"],
                    "requested_step": row["requested_step"],
                    "loaded_step": row["loaded_step"],
                    "loss": row["loss"],
                    "loss_finite": row["loss_finite"],
                    "psnr": row["psnr"],
                    "psnr_finite": row["psnr_finite"],
                    "direct_optical_depth_scale": row["direct_optical_depth_scale"],
                    "intrinsic_color_parameterization": row["intrinsic_color_parameterization"],
                    "features_dc_grad_finite": row["features_dc_grad"]["finite"],
                    "features_rest_grad_finite": row["features_rest_grad"]["finite"],
                    "medium_mlp_grad_finite": row["medium_mlp_grad"]["finite"],
                    "c_min": row["gaussian_view_rgb_range"]["min"],
                    "c_max": row["gaussian_view_rgb_range"]["max"],
                    "logit_min": row["gaussian_view_logits_range"]["min"],
                    "logit_max": row["gaussian_view_logits_range"]["max"],
                    "all_checks_pass": row["all_checks_pass"],
                }
            )
    return {"json": str(json_path), "csv": str(csv_path), "payload": payload}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME:STEP:config.yml")
    parser.add_argument("--train-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("renders/dewater_bounded_sh3_scratch_20260808"))
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
