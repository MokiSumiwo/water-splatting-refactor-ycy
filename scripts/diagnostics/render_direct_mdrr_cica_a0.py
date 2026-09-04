#!/usr/bin/env python3
"""Render all fixed heldout views from the historical MDRR/CICA A0 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_direct_mdrr_cica_scene as WORKER


AUX_ROOT = REPO_ROOT / "outputs" / "identifiability_module_causal_iui3_20260902"
FINAL_STEP = 14999
ALLOWED_GPUS = frozenset(("6", "7", "8", "9"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf8")


def _runtime() -> Dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in ALLOWED_GPUS:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be one of {sorted(ALLOWED_GPUS)}, got {visible!r}")
    if os.environ.get("CONDA_DEFAULT_ENV") != "water_splatting":
        raise RuntimeError("CONDA_DEFAULT_ENV must be water_splatting")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("A0 renderer must see exactly logical cuda:0")
    return {
        "physical_gpu": visible,
        "logical_gpu": 0,
        "gpu_name": torch.cuda.get_device_properties(0).name,
        "python": sys.executable,
        "torch_version": torch.__version__,
    }


def _checkpoint(scene: str) -> Path:
    return AUX_ROOT / scene / "checkpoints" / "C1" / f"step-{FINAL_STEP:09d}.ckpt"


def _distribution(value: Tensor) -> Dict[str, float]:
    return WORKER._clear_distribution(value)


def run(scene: str, output_dir: Path) -> Dict[str, Any]:
    runtime = _runtime()
    checkpoint = _checkpoint(scene)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    branch = None
    rows = []
    try:
        branch = WORKER._new_branch(scene)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            payload.get("branch") != "C1"
            or int(payload.get("absolute_step", -1)) != FINAL_STEP
            or payload.get("ocmc_bundle") is None
            or payload.get("raoc_state") is not None
            or not bool(payload.get("metadata", {}).get("module_enabled", False))
        ):
            raise RuntimeError(f"invalid historical A0 checkpoint: {checkpoint}")
        model = branch.pipeline.model
        model.load_state_dict(payload["model"], strict=True)
        model.step = FINAL_STEP
        WORKER._configure_model(model)
        WORKER.FORMAL._install_condition(model, "C0", payload["ocmc_bundle"], None)
        branch.pipeline.eval()
        model.eval()
        eval_files = list(getattr(branch.pipeline.datamanager.eval_dataset, "image_filenames", []))
        records = WORKER._eval_records(branch)
        if not records:
            raise RuntimeError(f"no fixed heldout cameras for {scene}")
        for eval_index, view_id, camera, batch in records:
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera.to(model.device))
                observed = WORKER._gt(
                    model,
                    WORKER._batch_to_device(batch, model.device),
                    outputs["background"],
                ).detach().float().clamp(0.0, 1.0)
            underwater = outputs["pred_image"].detach().float().clamp(0.0, 1.0)
            clear = outputs.get("rgb_clear")
            clear_raw = outputs.get("clear_object_fullsh_raw")
            if not isinstance(clear, Tensor) or not isinstance(clear_raw, Tensor):
                raise RuntimeError(f"native clear outputs missing for {scene}/{view_id}")
            view_dir = output_dir / str(view_id)
            view_dir.mkdir(parents=True, exist_ok=True)
            paths = {
                "input": view_dir / "input.png",
                "underwater": view_dir / "underwater.png",
                "clear_native": view_dir / "clear.png",
                "clear_raw_display_clamp01": view_dir / "clear_raw_display_clamp01.png",
            }
            WORKER._save_native_rgb(paths["input"], observed)
            WORKER._save_native_rgb(paths["underwater"], underwater)
            WORKER._save_native_rgb(paths["clear_native"], clear)
            WORKER._save_native_rgb(paths["clear_raw_display_clamp01"], clear_raw)
            source_image = eval_files[eval_index] if eval_index < len(eval_files) else None
            for output_type, path in paths.items():
                rows.append(
                    {
                        "scene": scene,
                        "view_id": str(view_id),
                        "configuration": "A0",
                        "absolute_step": FINAL_STEP,
                        "output_type": output_type,
                        "path": str(path),
                        "sha256": WORKER._sha256(path),
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": WORKER._sha256(checkpoint),
                        "source_image": str(source_image) if source_image is not None else None,
                        "renderer": "native WaterSplatting classic rasterizer",
                        "postprocessing": "clamp [0,1] and PNG encode only",
                    }
                )
            rows[-1].update(_distribution(clear_raw))
            del outputs, observed, underwater, clear, clear_raw
        manifest = {
            "experiment": "DIRECT_TRAINING_MDRR_CICA_AND_COMBINED_FOUR_SCENE",
            "configuration": "A0",
            "scene": scene,
            "runtime": runtime,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": WORKER._sha256(checkpoint),
            "absolute_step": FINAL_STEP,
            "ocmc_on": True,
            "raoc_off": True,
            "auxiliary_appearance_regularization": True,
            "paired_real_clear_gt_found": False,
            "view_count": len(records),
            "rows": rows,
        }
        _write_json(output_dir / "render_manifest.json", manifest)
        return manifest
    finally:
        WORKER._release(branch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=sorted(WORKER.SCENE_GPUS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.scene, args.output_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
