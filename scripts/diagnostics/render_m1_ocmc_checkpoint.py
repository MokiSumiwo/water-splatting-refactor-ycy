#!/usr/bin/env python3
"""Render held-out RGB images from an existing pure M1+OCMC checkpoint.

This is an analysis-only utility.  It never trains, changes model parameters,
or writes into a historical render directory.  The checkpoint must identify
the legacy C0 OCMC source arm and contain an OCMC projector without RAOC.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_legacy_ocmc_scene as SOURCE
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL


FINAL_STEP = 14999
SCENES = tuple(FORMAL.SCENES)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_rgb(path: Path, value: torch.Tensor) -> None:
    array = torch.nan_to_num(value.detach().float(), nan=0.0, posinf=1.0, neginf=0.0)
    array = array.clamp(0.0, 1.0).cpu().numpy()
    if array.ndim != 3 or array.shape[-1] != 3:
        raise RuntimeError(f"expected HxWx3 RGB tensor, got {tuple(array.shape)}")
    Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB").save(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pure_checkpoint(branch: Any, checkpoint: Path) -> Mapping[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("branch") != "C0":
        raise RuntimeError(f"expected pure OCMC branch C0, got {payload.get('branch')!r}")
    if int(payload.get("absolute_step", -1)) != FINAL_STEP:
        raise RuntimeError(f"expected checkpoint step {FINAL_STEP}, got {payload.get('absolute_step')!r}")
    if payload.get("ocmc_bundle") is None:
        raise RuntimeError("checkpoint does not contain an OCMC projector bundle")
    if payload.get("raoc_state") is not None:
        raise RuntimeError("checkpoint unexpectedly contains RAOC state")
    metadata = payload.get("metadata", {})
    if metadata.get("intrinsic_color_parameterization") not in (None, "legacy"):
        raise RuntimeError(
            "pure M1+OCMC render requires legacy parameterization, got "
            f"{metadata.get('intrinsic_color_parameterization')!r}"
        )
    branch.pipeline.model.load_state_dict(payload["model"], strict=True)
    branch.pipeline.model.step = FINAL_STEP
    FORMAL._configure_model(branch.pipeline.model, "C0", "legacy")
    FORMAL._install_condition(branch.pipeline.model, "C0", payload["ocmc_bundle"], None)
    branch.pipeline.eval()
    branch.pipeline.model.eval()
    return payload


def render(scene: str, gpu: str, checkpoint: Path, output_dir: Path) -> Dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != str(gpu):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must expose exactly {gpu}, got {visible!r}")
    if os.environ.get("CONDA_DEFAULT_ENV") != "water_splatting":
        raise RuntimeError("CONDA_DEFAULT_ENV must be water_splatting")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("render worker must see exactly one CUDA device")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_cfg = SOURCE._scene_cfg(scene)
    # _install_condition() re-applies the module-global parameterization.  The
    # retained C0 checkpoints are legacy SH3, so set this before construction
    # and before checkpoint condition installation.
    FORMAL.ACTIVE_PARAMETERIZATION = "legacy"
    branch = FORMAL._setup_branch(REPO_ROOT, scene_cfg, "C0", "legacy")
    try:
        payload = _load_pure_checkpoint(branch, checkpoint)
        records = FORMAL._eval_records(branch.pipeline)
        rows: List[Dict[str, Any]] = []
        for index, view_id, camera, _batch in records:
            with torch.no_grad():
                outputs = branch.pipeline.model.get_outputs_for_camera(camera.to(branch.pipeline.model.device))
                prediction = outputs["pred_image"].detach().float().clamp(0.0, 1.0)
            path = output_dir / f"{view_id}.png"
            _save_rgb(path, prediction)
            rows.append(
                {
                    "scene": scene,
                    "branch": "C0",
                    "absolute_step": FINAL_STEP,
                    "split": "eval",
                    "view_index": int(index),
                    "view_id": str(view_id),
                    "output_type": "underwater",
                    "path": str(path.resolve()),
                    "height": int(prediction.shape[0]),
                    "width": int(prediction.shape[1]),
                    "sha256": _sha256(path),
                }
            )
            del outputs, prediction
        _write_csv(output_dir / "render_manifest.csv", rows)
        _write_json(
            output_dir / "render_manifest.json",
            {
                "experiment": "M1_OCMC_ANALYSIS_ONLY_RENDER",
                "scene": scene,
                "branch": "C0",
                "absolute_step": FINAL_STEP,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_experiment": payload.get("experiment"),
                "parameterization": "legacy",
                "ocmc_enabled": True,
                "raoc_enabled": False,
                "split": "eval",
                "view_count": len(rows),
                "rows": rows,
            },
        )
        return {"scene": scene, "view_count": len(rows), "output_dir": str(output_dir.resolve()), "rows": rows}
    finally:
        FORMAL._release(branch)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=sorted(SCENES), required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.scene, str(args.gpu), args.checkpoint.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
