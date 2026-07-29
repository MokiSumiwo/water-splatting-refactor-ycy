#!/usr/bin/env python
"""Verify branch-local capacity opacity gradient isolation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from nerfstudio.utils.eval_utils import eval_setup


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _grad(loss: torch.Tensor, param: torch.Tensor, *, retain_graph: bool) -> torch.Tensor:
    grad = torch.autograd.grad(
        loss,
        param,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )[0]
    return grad.detach()


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    diff = (a - b).detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(b.detach().float().reshape(-1)).clamp_min(1e-20)
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "relative_norm": float((torch.linalg.vector_norm(diff) / denom).item()),
    }


def _configure_model(model: Any, args: argparse.Namespace, *, conflict_gate: bool) -> None:
    model.config.capacity_control_enabled = True
    model.config.capacity_control_geometry_gradient_scale = float(args.geometry_scale)
    model.config.capacity_control_position_gradient_scale = float(args.position_scale)
    model.config.capacity_control_depth_gradient_scale = float(args.depth_scale)
    model.config.capacity_control_footprint_gradient_scale = float(args.footprint_scale)
    model.config.capacity_control_opacity_gradient_scale = float(args.opacity_scale)
    model.config.capacity_conflict_gate_enabled = bool(conflict_gate)
    model.config.capacity_conflict_rho = float(args.rho)
    model.config.capacity_conflict_rec_grad_threshold = float(args.rec_threshold)
    model.config.budgeted_capacity_enabled = True
    if float(model.config.lambda_budgeted_capacity) <= 0.0:
        model.config.lambda_budgeted_capacity = float(args.lambda_budgeted_capacity)
    model.config.budgeted_capacity_start_step = min(int(model.config.budgeted_capacity_start_step), int(args.force_step))
    model.config.budgeted_capacity_ramp_steps = 0


def _forward_losses(model: Any, camera: Any, batch: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    outputs = model.get_outputs(camera)
    loss_dict = model.get_loss_dict(outputs, batch, {})
    if "budgeted_capacity_loss" not in loss_dict:
        raise RuntimeError("budgeted_capacity_loss was not active")
    if "capacity_control_opacities" not in outputs:
        raise RuntimeError("capacity_control_opacities missing; capacity control did not activate")
    return outputs, loss_dict


def _run_no_gate(model: Any, camera: Any, batch: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    model.zero_grad(set_to_none=True)
    _configure_model(model, args, conflict_gate=False)
    outputs, loss_dict = _forward_losses(model, camera, batch)
    main_grad = _grad(loss_dict["main_loss"], model.opacities, retain_graph=True)
    cap_grad = _grad(loss_dict["budgeted_capacity_loss"], model.opacities, retain_graph=True)
    total_grad = _grad(loss_dict["main_loss"] + loss_dict["budgeted_capacity_loss"], model.opacities, retain_graph=False)
    cap_op = outputs["capacity_control_opacities"]
    main_op = outputs["main_render_opacities"]
    return {
        "main_grad": main_grad,
        "cap_grad": cap_grad,
        "total_grad": total_grad,
        "branch_identity": {
            "capacity_control_opacities_is_main_render_opacities": bool(cap_op is main_op),
            "capacity_control_opacities_data_ptr_equals_main": bool(cap_op.data_ptr() == main_op.data_ptr()),
            "capacity_control_opacities_grad_fn": str(type(cap_op.grad_fn).__name__) if cap_op.grad_fn is not None else None,
            "main_render_opacities_grad_fn": str(type(main_op.grad_fn).__name__) if main_op.grad_fn is not None else None,
        },
    }


def _run_gate(model: Any, camera: Any, batch: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    model.zero_grad(set_to_none=True)
    _configure_model(model, args, conflict_gate=True)
    outputs, loss_dict = _forward_losses(model, camera, batch)
    main_grad_after_hook = _grad(loss_dict["main_loss"], model.opacities, retain_graph=True)
    total_grad_gate = _grad(loss_dict["main_loss"] + loss_dict["budgeted_capacity_loss"], model.opacities, retain_graph=False)
    cap_op = outputs["capacity_control_opacities"]
    main_op = outputs["main_render_opacities"]
    return {
        "main_grad_after_hook": main_grad_after_hook,
        "total_grad_gate": total_grad_gate,
        "branch_identity": {
            "capacity_control_opacities_is_main_render_opacities": bool(cap_op is main_op),
            "capacity_control_opacities_data_ptr_equals_main": bool(cap_op.data_ptr() == main_op.data_ptr()),
            "capacity_control_opacities_grad_fn": str(type(cap_op.grad_fn).__name__) if cap_op.grad_fn is not None else None,
            "main_render_opacities_grad_fn": str(type(main_op.grad_fn).__name__) if main_op.grad_fn is not None else None,
        },
    }


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    def _update_config(config: Any) -> Any:
        if args.load_step is not None:
            config.load_step = int(args.load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(args.load_config, update_config_callback=_update_config)
    pipeline.eval()
    model = pipeline.model
    model.step = int(args.force_step if args.force_step is not None else step)

    camera = None
    batch = None
    for idx, (candidate_camera, candidate_batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        if idx == int(args.image_index):
            camera = candidate_camera
            batch = candidate_batch
            break
    if camera is None or batch is None:
        raise RuntimeError(f"Could not load eval image index {args.image_index}")

    no_gate = _run_no_gate(model, camera, batch, args)
    gate = _run_gate(model, camera, batch, args)

    rho = float(args.rho)
    rec_threshold = float(args.rec_threshold)
    conflict = (no_gate["cap_grad"] > 0.0) & (no_gate["main_grad"] < -rec_threshold)
    expected_gated_cap = torch.where(conflict, rho * no_gate["cap_grad"], no_gate["cap_grad"])
    expected_total_gate = gate["main_grad_after_hook"] + expected_gated_cap
    expected_total_no_gate = no_gate["main_grad"] + no_gate["cap_grad"]

    repo = Path(__file__).resolve().parents[2]
    result = {
        "experiment_name": config.experiment_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(step),
        "model_step_used": int(model.step),
        "load_config": str(args.load_config),
        "git_commit": _git_commit(repo),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "image_index": int(args.image_index),
        "rho": rho,
        "rec_threshold": rec_threshold,
        "branch_identity_no_gate": no_gate["branch_identity"],
        "branch_identity_gate": gate["branch_identity"],
        "ungated_total_additivity": _diff_stats(no_gate["total_grad"], expected_total_no_gate),
        "main_only_before_vs_after_hook": _diff_stats(gate["main_grad_after_hook"], no_gate["main_grad"]),
        "gated_total_vs_expected": _diff_stats(gate["total_grad_gate"], expected_total_gate),
        "conflict_fraction": float(conflict.float().mean().item()),
        "conflicting_cap_mass_fraction": float(
            no_gate["cap_grad"][conflict].abs().sum().item()
            / no_gate["cap_grad"][no_gate["cap_grad"] > 0.0].abs().sum().clamp_min(1e-20).item()
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--force-step", type=int, default=14999)
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--geometry-scale", type=float, default=0.0)
    parser.add_argument("--position-scale", type=float, default=-1.0)
    parser.add_argument("--depth-scale", type=float, default=-1.0)
    parser.add_argument("--footprint-scale", type=float, default=-1.0)
    parser.add_argument("--opacity-scale", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--rec-threshold", type=float, default=1e-10)
    parser.add_argument("--lambda-budgeted-capacity", type=float, default=0.0002)
    args = parser.parse_args()

    result = diagnose(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
