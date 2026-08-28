#!/usr/bin/env python3
"""Run a short matched reference/fused RAOC engineering smoke.

The smoke starts both backends from the same archived C1 checkpoint and uses
the same cached camera sequence.  It is deliberately capped at 200 steps and
does not write checkpoints or run densification/refinement callbacks.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_raoc_q50_q80_causal_scene as RAOC


ARCHIVE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "raoc_cuda_fusion_optimization_20260828" / "smoke"
SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf8")


def _default(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist() if value.numel() != 1 else float(value.detach().cpu().item())
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _flat_norm(model: Any) -> float:
    return float(torch.cat([p.detach().float().reshape(-1).cpu() for p in model.parameters()]).norm().item())


def _run_backend(scene: str, gpu: str, backend: str, steps: int) -> Dict[str, Any]:
    scene_cfg = RAOC.SCENES[scene]
    branch = RAOC._setup_branch(REPO_ROOT, scene_cfg, "C1")
    checkpoint = ARCHIVE_ROOT / scene / "checkpoints" / "C1" / "step-000003000.ckpt"
    try:
        ckpt = RAOC._load_checkpoint(branch, checkpoint)
        model = branch.pipeline.model
        model.config.camera_medium_raoc_backend = backend
        records = RAOC._train_records(branch.pipeline)
        cameras = getattr(branch.pipeline.datamanager, "train_cameras", branch.pipeline.datamanager.train_dataset.cameras).to(model.device)
        cached = branch.pipeline.datamanager.cached_train
        sequence = json.loads((ARCHIVE_ROOT / scene / "camera_sequence.json").read_text(encoding="utf8"))["rows"]
        sequence = [int(row["camera_index"]) for row in sequence[:steps]]
        rows: List[Dict[str, Any]] = []
        torch.cuda.reset_peak_memory_stats()
        for local_step, camera_index in enumerate(sequence):
            torch.manual_seed(20260828 + local_step)
            model.train()
            model.zero_grad(set_to_none=True)
            batch = {key: value.to(model.device) if isinstance(value, Tensor) else value for key, value in cached[camera_index].copy().items()}
            camera = cameras[camera_index : camera_index + 1]
            outputs = model.get_outputs(camera)
            gt = RAOC.MI.PW._get_gt(model, batch, outputs["background"])
            losses = model.get_loss_dict(outputs, batch, {})
            total = sum(losses.values())
            finite_forward = bool(torch.isfinite(total).item()) and all(
                bool(torch.isfinite(value).all().item()) for value in outputs.values() if isinstance(value, Tensor)
            )
            total.backward()
            finite_grad = all(
                param.grad is None or bool(torch.isfinite(param.grad).all().item())
                for param in model.parameters()
            )
            branch.optimizers.optimizer_step_all()
            branch.optimizers.scheduler_step_all(3000 + local_step)
            gate = outputs.get("camera_medium_keep_gate")
            rows.append({
                "local_step": local_step,
                "camera_index": camera_index,
                "loss": float(total.detach().cpu().item()),
                "finite_forward": finite_forward,
                "finite_grad": finite_grad,
                "gaussian_count": int(model.means.shape[0]),
                "keep_gate_mean": float(gate.detach().float().mean().cpu().item()) if gate is not None else None,
                "keep_gate_max": float(gate.detach().float().max().cpu().item()) if gate is not None else None,
                "parameter_norm": _flat_norm(model),
                "allocated_MB": torch.cuda.memory_allocated() / 2**20,
                "reserved_MB": torch.cuda.memory_reserved() / 2**20,
            })
            del outputs, losses, total, gt, batch
        return {
            "scene": scene,
            "gpu": gpu,
            "backend": backend,
            "checkpoint": str(checkpoint),
            "steps": len(rows),
            "rows": rows,
            "all_finite": all(row["finite_forward"] and row["finite_grad"] for row in rows),
            "gaussian_count_constant": len({row["gaussian_count"] for row in rows}) == 1,
            "peak_allocated_MB": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_MB": torch.cuda.max_memory_reserved() / 2**20,
            "densification_or_refinement": False,
        }
    finally:
        RAOC._release(branch)
        gc.collect()


def run(scene: str, gpu: str, steps: int, output: Path) -> Dict[str, Any]:
    steps = min(max(int(steps), 1), 200)
    reference = _run_backend(scene, gpu, "reference", steps)
    fused = _run_backend(scene, gpu, "cuda_fused", steps)
    rows: List[Dict[str, Any]] = []
    for left, right in zip(reference["rows"], fused["rows"]):
        rows.append({
            "local_step": left["local_step"],
            "reference_loss": left["loss"],
            "fused_loss": right["loss"],
            "loss_abs_diff": abs(left["loss"] - right["loss"]),
            "reference_gaussian_count": left["gaussian_count"],
            "fused_gaussian_count": right["gaussian_count"],
            "gaussian_count_match": left["gaussian_count"] == right["gaussian_count"],
            "reference_keep_gate_mean": left["keep_gate_mean"],
            "fused_keep_gate_mean": right["keep_gate_mean"],
            "reference_allocated_MB": left["allocated_MB"],
            "fused_allocated_MB": right["allocated_MB"],
            "reference_reserved_MB": left["reserved_MB"],
            "fused_reserved_MB": right["reserved_MB"],
            "reference_parameter_norm": left["parameter_norm"],
            "fused_parameter_norm": right["parameter_norm"],
        })
    result = {
        "scene": scene,
        "gpu": gpu,
        "steps": steps,
        "reference": reference,
        "cuda_fused": fused,
        "matched_rows": rows,
        "all_finite": reference["all_finite"] and fused["all_finite"],
        "gaussian_count_match_all": all(row["gaussian_count_match"] for row in rows),
        "max_loss_abs_diff": max((row["loss_abs_diff"] for row in rows), default=0.0),
        "optimizer_steps": steps * 2,
        "new_formal_15k_experiment": False,
    }
    _write(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=SCENES)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or OUTPUT_ROOT / f"smoke_{args.scene}.json"
    result = run(args.scene, args.gpu, args.steps, output)
    print(json.dumps({key: result[key] for key in ("scene", "gpu", "steps", "all_finite", "gaussian_count_match_all", "max_loss_abs_diff")}, sort_keys=True))


if __name__ == "__main__":
    main()
