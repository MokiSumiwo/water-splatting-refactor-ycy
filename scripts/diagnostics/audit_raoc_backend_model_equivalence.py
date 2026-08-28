#!/usr/bin/env python3
"""Compare complete model forward/backward results for both RAOC backends."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_raoc_q50_q80_causal_scene as RAOC


ARCHIVE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "raoc_cuda_fusion_optimization_20260828" / "equivalence"
PARAMETER_NAMES = ("medium_mlp", "direction_encoding", "means", "features_dc", "features_rest", "scales", "quats", "opacities")
OUTPUT_NAMES = ("pred_image", "depth", "accumulation", "medium_rgb", "medium_bs", "medium_attn", "b_inf")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf8")


def _default(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist() if value.numel() != 1 else float(value.detach().cpu().item())
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _tensor_diff(left: Tensor, right: Tensor) -> Dict[str, float]:
    delta = (left.detach().float().cpu() - right.detach().float().cpu()).abs()
    left_norm = left.detach().float().cpu().norm()
    return {
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
        "relative_l2": float(delta.norm().item() / max(float(left_norm.item()), 1e-12)),
    }


def _parameter_gradients(model: Any) -> Dict[str, Tensor]:
    result: Dict[str, Tensor] = {}
    for name in PARAMETER_NAMES:
        value = getattr(model, name)
        if isinstance(value, torch.nn.Module):
            values = [parameter.grad.detach().float().cpu().reshape(-1) for parameter in value.parameters() if parameter.grad is not None]
            result[name] = torch.cat(values) if values else torch.empty(0)
        else:
            result[name] = value.grad.detach().float().cpu().clone() if value.grad is not None else torch.empty(0)
    return result


def _run(scene: str, gpu: str, step: int, backend: str) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], float, Dict[str, Any]]:
    holder = RAOC._setup_branch(REPO_ROOT, RAOC.SCENES[scene], "C1")
    checkpoint = ARCHIVE_ROOT / scene / "checkpoints" / "C1" / f"step-{step:09d}.ckpt"
    try:
        ckpt = RAOC._load_checkpoint(holder, checkpoint)
        model = holder.pipeline.model
        model.config.camera_medium_raoc_backend = backend
        model.train()
        model.zero_grad(set_to_none=True)
        _idx, _view_id, camera, batch = RAOC._train_records(holder.pipeline)[0]
        batch = {key: value.to(model.device) if isinstance(value, Tensor) else value for key, value in batch.items()}
        outputs = model.get_outputs(camera)
        losses = model.get_loss_dict(outputs, batch, {})
        loss = sum(losses.values())
        loss.backward()
        output_values = {name: outputs[name].detach().float().cpu().clone() for name in OUTPUT_NAMES if name in outputs}
        gradients = _parameter_gradients(model)
        metadata = {
            "scene": scene,
            "gpu": gpu,
            "step": step,
            "backend": backend,
            "gaussian_count": int(model.means.shape[0]),
            "loss_terms": {key: float(value.detach().cpu().item()) for key, value in losses.items()},
            "loss": float(loss.detach().cpu().item()),
            "finite_outputs": all(bool(torch.isfinite(value).all().item()) for value in outputs.values() if isinstance(value, Tensor)),
            "finite_gradients": all(bool(torch.isfinite(value).all().item()) for value in gradients.values() if value.numel()),
        }
        return output_values, gradients, metadata["loss"], metadata
    finally:
        RAOC._release(holder)
        gc.collect()


def audit(scene: str, gpu: str, step: int, output: Path) -> Dict[str, Any]:
    ref_outputs, ref_grads, ref_loss, ref_meta = _run(scene, gpu, step, "reference")
    fused_outputs, fused_grads, fused_loss, fused_meta = _run(scene, gpu, step, "cuda_fused")
    output_diffs = {name: _tensor_diff(ref_outputs[name], fused_outputs[name]) for name in ref_outputs if name in fused_outputs}
    gradient_diffs = {name: _tensor_diff(ref_grads[name], fused_grads[name]) for name in ref_grads if ref_grads[name].shape == fused_grads[name].shape and ref_grads[name].numel()}
    result = {
        "scene": scene,
        "gpu": gpu,
        "step": step,
        "reference": ref_meta,
        "cuda_fused": fused_meta,
        "loss_abs_diff": abs(ref_loss - fused_loss),
        "output_diffs": output_diffs,
        "gradient_diffs": gradient_diffs,
        "forward_finite": bool(ref_meta["finite_outputs"] and fused_meta["finite_outputs"]),
        "backward_finite": bool(ref_meta["finite_gradients"] and fused_meta["finite_gradients"]),
        "gaussian_count_match": ref_meta["gaussian_count"] == fused_meta["gaussian_count"],
        "model_level_strict_forward_pass": all(row["max_abs"] <= 1e-6 for row in output_diffs.values()),
        "model_level_strict_backward_pass": all(row["relative_l2"] <= 1e-6 for row in gradient_diffs.values()),
    }
    _write(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="IUI3-RedSea", choices=sorted(RAOC.SCENES))
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--step", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "model_level_equivalence.json")
    args = parser.parse_args()
    result = audit(args.scene, args.gpu, args.step, args.output)
    print(json.dumps({key: result[key] for key in ("scene", "step", "loss_abs_diff", "forward_finite", "backward_finite", "gaussian_count_match", "model_level_strict_forward_pass", "model_level_strict_backward_pass")}, sort_keys=True))


if __name__ == "__main__":
    main()
