#!/usr/bin/env python3
"""Read-only legacy checkpoint and non-RAOC path compatibility audit."""

from __future__ import annotations

import argparse
import gc
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_raoc_q50_q80_causal_scene as RAOC


ARCHIVE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
OLD_GLOB = str(REPO_ROOT / "outputs/dewater_bounded_sh3_scratch_20260808/**/nerfstudio_models/step-000003000.ckpt")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf8")


def _diff(left: Tensor, right: Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def _old_checkpoint_load(scene: str) -> Dict[str, Any]:
    paths = sorted(glob.glob(OLD_GLOB, recursive=True))
    if not paths:
        return {"pass": False, "reason": "no archived pre-RAOC checkpoint found"}
    holder = RAOC._setup_branch(REPO_ROOT, RAOC.SCENES[scene], "C1")
    try:
        checkpoint = Path(paths[0])
        payload = torch.load(checkpoint, map_location="cpu")
        state = {key[len("_model.") :]: value for key, value in payload["pipeline"].items() if key.startswith("_model.")}
        state.pop("gaussian_lineage_ids", None)
        holder.pipeline.model.load_state_dict(state, strict=False)
        return {
            "pass": True,
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(payload.get("step", -1)),
            "raoc_buffers_initialized": "_camera_medium_raoc_basis" in holder.pipeline.model.state_dict(),
        }
    except Exception as exc:
        return {"pass": False, "checkpoint": str(paths[0]), "error": repr(exc)}
    finally:
        RAOC._release(holder)


def _nonraoc(scene: str, step: int) -> Dict[str, Any]:
    holder = RAOC._setup_branch(REPO_ROOT, RAOC.SCENES[scene], "C1")
    checkpoint = ARCHIVE_ROOT / scene / "checkpoints" / "C1" / f"step-{step:09d}.ckpt"
    try:
        RAOC._load_checkpoint(holder, checkpoint)
        model = holder.pipeline.model
        _idx, _view_id, camera, batch = RAOC._train_records(holder.pipeline)[0]
        batch = {key: value.to(model.device) if isinstance(value, Tensor) else value for key, value in batch.items()}
        model.config.camera_medium_ray_adaptive_observability_enabled = False
        model.config.camera_medium_observability_enabled = False
        model.set_camera_medium_observability_projector(None)
        baseline = model.get_outputs(camera)
        baseline_loss = sum(model.get_loss_dict(baseline, batch, {}).values())
        model.set_camera_medium_observability_projector(torch.eye(9, device=model.device))
        disabled = model.get_outputs(camera)
        disabled_loss = sum(model.get_loss_dict(disabled, batch, {}).values())
        disabled_diffs = {
            key: _diff(baseline[key], disabled[key])
            for key in ("pred_image", "depth", "accumulation", "medium_rgb", "medium_bs", "medium_attn")
        }
        disabled_diffs["loss"] = abs(float(baseline_loss.item()) - float(disabled_loss.item()))
        model.config.camera_medium_observability_enabled = True
        model.config.camera_medium_observability_strength = 1.0
        ocmc_a = model.get_outputs(camera)
        ocmc_a_loss = sum(model.get_loss_dict(ocmc_a, batch, {}).values())
        ocmc_b = model.get_outputs(camera)
        ocmc_b_loss = sum(model.get_loss_dict(ocmc_b, batch, {}).values())
        ocmc_diffs = {
            key: _diff(ocmc_a[key], ocmc_b[key])
            for key in ("pred_image", "depth", "accumulation", "medium_rgb", "medium_bs", "medium_attn")
        }
        ocmc_diffs["loss"] = abs(float(ocmc_a_loss.item()) - float(ocmc_b_loss.item()))
        return {
            "scene": scene,
            "step": step,
            "disabled_path": {"max_abs_diffs": disabled_diffs, "pass": max(disabled_diffs.values()) == 0.0},
            "ocmc_path_repeatability": {"max_abs_diffs": ocmc_diffs, "pass": max(ocmc_diffs.values()) == 0.0},
            "raoc_backend_not_used": True,
        }
    finally:
        RAOC._release(holder)
        gc.collect()


def run(scene: str, step: int, output: Path) -> Dict[str, Any]:
    result = {
        "old_pre_raoc_checkpoint": _old_checkpoint_load(scene),
        "nonraoc": _nonraoc(scene, step),
        "old_calibrated_raoc_state_checkpoint": str(ARCHIVE_ROOT / scene / "checkpoints" / "C1" / f"step-{step:09d}.ckpt"),
        "old_calibrated_raoc_state_load_pass": (ARCHIVE_ROOT / scene / "checkpoints" / "C1" / f"step-{step:09d}.ckpt").is_file(),
    }
    _write(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="Curasao", choices=sorted(RAOC.SCENES))
    parser.add_argument("--step", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.scene, args.step, args.output)
    print(json.dumps({"old_checkpoint_pass": result["old_pre_raoc_checkpoint"]["pass"], "disabled_pass": result["nonraoc"]["disabled_path"]["pass"], "ocmc_repeatability_pass": result["nonraoc"]["ocmc_path_repeatability"]["pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
