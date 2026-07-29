#!/usr/bin/env python
"""Diagnose reconstruction-vs-capacity gradient conflict.

This diagnostic loads an existing checkpoint, evaluates a small set of fixed
views, and separately differentiates the reconstruction loss and budgeted
capacity loss with respect to Gaussian opacity, scale, and means. It is
diagnostic-only and does not use region masks for training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from nerfstudio.utils.eval_utils import eval_setup


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _clear_grads(model: torch.nn.Module) -> None:
    model.zero_grad(set_to_none=True)
    for name in ("xys", "xys_grad_abs", "xys_grad_abs_proxy", "xys_grad_abs_capacity"):
        value = getattr(model, name, None)
        if value is not None and getattr(value, "retains_grad", False) and value.grad is not None:
            value.grad = None


def _safe_grad(
    loss: torch.Tensor,
    params: Iterable[torch.Tensor],
    *,
    retain_graph: bool,
) -> List[Optional[torch.Tensor]]:
    params = list(params)
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    return [grad.detach() if grad is not None else None for grad in grads]


def _flat_finite(value: torch.Tensor) -> torch.Tensor:
    flat = value.detach().float().reshape(-1)
    return flat[torch.isfinite(flat)]


def _stats(value: Optional[torch.Tensor]) -> Dict[str, float]:
    if value is None:
        return {"norm": 0.0, "abs_mean": 0.0, "abs_median": 0.0, "abs_p95": 0.0, "abs_max": 0.0, "nonzero_ratio": 0.0}
    flat = _flat_finite(value)
    if flat.numel() == 0:
        return {"norm": 0.0, "abs_mean": 0.0, "abs_median": 0.0, "abs_p95": 0.0, "abs_max": 0.0, "nonzero_ratio": 0.0}
    abs_flat = flat.abs()
    return {
        "norm": float(torch.linalg.vector_norm(flat).item()),
        "abs_mean": float(abs_flat.mean().item()),
        "abs_median": float(torch.quantile(abs_flat, 0.50).item()),
        "abs_p95": float(torch.quantile(abs_flat, 0.95).item()),
        "abs_max": float(abs_flat.max().item()),
        "nonzero_ratio": float((abs_flat > 1e-12).float().mean().item()),
    }


def _per_gaussian_norm(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if value.ndim <= 1:
        return value.detach().float().abs()
    return torch.linalg.vector_norm(value.detach().float().reshape(value.shape[0], -1), dim=-1)


def _conflict_stats(rec_opacity: Optional[torch.Tensor], cap_opacity: Optional[torch.Tensor]) -> Dict[str, float]:
    if rec_opacity is None or cap_opacity is None:
        return {
            "cap_positive_fraction": 0.0,
            "rec_negative_fraction": 0.0,
            "conflict_fraction_all": 0.0,
            "conflict_fraction_cap_positive": 0.0,
            "conflicting_capacity_mass_fraction": 0.0,
            "nonconflicting_capacity_mass_fraction": 0.0,
            "cap_positive_mass": 0.0,
            "conflicting_capacity_mass": 0.0,
            "nonconflicting_capacity_mass": 0.0,
        }
    rec = rec_opacity.detach().float().reshape(-1)
    cap = cap_opacity.detach().float().reshape(-1)
    finite = torch.isfinite(rec) & torch.isfinite(cap)
    rec = rec[finite]
    cap = cap[finite]
    if rec.numel() == 0:
        return {
            "cap_positive_fraction": 0.0,
            "rec_negative_fraction": 0.0,
            "conflict_fraction_all": 0.0,
            "conflict_fraction_cap_positive": 0.0,
            "conflicting_capacity_mass_fraction": 0.0,
            "nonconflicting_capacity_mass_fraction": 0.0,
            "cap_positive_mass": 0.0,
            "conflicting_capacity_mass": 0.0,
            "nonconflicting_capacity_mass": 0.0,
        }
    cap_positive = cap > 0.0
    rec_negative = rec < 0.0
    conflict = cap_positive & rec_negative
    cap_positive_mass = cap[cap_positive].abs().sum()
    conflicting_mass = cap[conflict].abs().sum()
    nonconflicting_mass = (cap[cap_positive & ~rec_negative].abs().sum())
    denom = cap_positive_mass.clamp_min(1e-20)
    return {
        "cap_positive_fraction": float(cap_positive.float().mean().item()),
        "rec_negative_fraction": float(rec_negative.float().mean().item()),
        "conflict_fraction_all": float(conflict.float().mean().item()),
        "conflict_fraction_cap_positive": float(conflict.float().sum().item() / cap_positive.float().sum().clamp_min(1.0).item()),
        "conflicting_capacity_mass_fraction": float((conflicting_mass / denom).item()),
        "nonconflicting_capacity_mass_fraction": float((nonconflicting_mass / denom).item()),
        "cap_positive_mass": float(cap_positive_mass.item()),
        "conflicting_capacity_mass": float(conflicting_mass.item()),
        "nonconflicting_capacity_mass": float(nonconflicting_mass.item()),
    }


def _image_result(model: torch.nn.Module, camera: Any, batch: Dict[str, Any]) -> Dict[str, Any]:
    _clear_grads(model)
    outputs = model.get_outputs(camera)
    metrics: Dict[str, torch.Tensor] = {}
    loss_dict = model.get_loss_dict(outputs, batch, metrics)
    if "budgeted_capacity_loss" not in loss_dict:
        raise RuntimeError("budgeted_capacity_loss is not active for this checkpoint/config")
    main_loss = loss_dict["main_loss"]
    cap_loss = loss_dict["budgeted_capacity_loss"]
    params = [model.opacities, model.scales, model.means]
    rec_opacity, rec_scales, rec_means = _safe_grad(main_loss, params, retain_graph=True)
    cap_opacity, cap_scales, cap_means = _safe_grad(cap_loss, params, retain_graph=False)

    cap_opacity_norm = _per_gaussian_norm(cap_opacity)
    cap_scales_norm = _per_gaussian_norm(cap_scales)
    cap_means_norm = _per_gaussian_norm(cap_means)

    def _mass_ratio(numer: Optional[torch.Tensor], denom: Optional[torch.Tensor]) -> float:
        if numer is None or denom is None:
            return 0.0
        return float(numer.sum().item() / denom.sum().clamp_min(1e-20).item())

    result = {
        "main_loss": float(main_loss.detach().item()),
        "budgeted_capacity_loss": float(cap_loss.detach().item()),
        "support_capacity_mean": float(outputs.get("medium_support_capacity", torch.zeros((), device=main_loss.device)).detach().float().mean().item()),
        "support_capacity_gt_0p25_fraction": float((outputs.get("medium_support_capacity", torch.zeros_like(outputs["accumulation"])) > 0.25).detach().float().mean().item()),
        "rec_grad": {
            "opacities": _stats(rec_opacity),
            "scales": _stats(rec_scales),
            "means": _stats(rec_means),
        },
        "capacity_grad": {
            "opacities": _stats(cap_opacity),
            "scales": _stats(cap_scales),
            "means": _stats(cap_means),
            "scale_to_opacity_mass_ratio": _mass_ratio(cap_scales_norm, cap_opacity_norm),
            "mean_to_opacity_mass_ratio": _mass_ratio(cap_means_norm, cap_opacity_norm),
        },
        "opacity_conflict": _conflict_stats(rec_opacity, cap_opacity),
    }
    _clear_grads(model)
    return result


def _mean(rows: List[Dict[str, Any]], dotted_key: str) -> float:
    values = []
    for row in rows:
        value: Any = row
        for part in dotted_key.split("."):
            value = value[part]
        values.append(float(value))
    if not values:
        return 0.0
    return float(torch.tensor(values, dtype=torch.float64).mean().item())


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    def _update_config(config: Any) -> Any:
        if args.load_step is not None:
            config.load_step = int(args.load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(
        args.load_config,
        update_config_callback=_update_config,
    )
    pipeline.eval()
    model = pipeline.model
    model.step = int(step)
    if args.force_step is not None:
        model.step = int(args.force_step)
    images: List[Dict[str, Any]] = []
    for image_idx, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        if image_idx >= args.max_images:
            break
        images.append({"image_index": image_idx, **_image_result(model, camera, batch)})

    aggregate_keys = [
        "main_loss",
        "budgeted_capacity_loss",
        "support_capacity_mean",
        "support_capacity_gt_0p25_fraction",
        "capacity_grad.opacities.norm",
        "capacity_grad.scales.norm",
        "capacity_grad.means.norm",
        "capacity_grad.scale_to_opacity_mass_ratio",
        "capacity_grad.mean_to_opacity_mass_ratio",
        "opacity_conflict.cap_positive_fraction",
        "opacity_conflict.rec_negative_fraction",
        "opacity_conflict.conflict_fraction_all",
        "opacity_conflict.conflict_fraction_cap_positive",
        "opacity_conflict.conflicting_capacity_mass_fraction",
        "opacity_conflict.nonconflicting_capacity_mass_fraction",
    ]
    aggregate = {key: _mean(images, key) for key in aggregate_keys}
    repo = Path(__file__).resolve().parents[2]
    return {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(step),
        "model_step_used": int(model.step),
        "git_commit": _git_commit(repo),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "max_images": int(args.max_images),
        "aggregate": aggregate,
        "images": images,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--force-step", type=int, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=4)
    args = parser.parse_args()

    result = diagnose(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()
