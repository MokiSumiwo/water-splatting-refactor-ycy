#!/usr/bin/env python
"""Audit camera/image indexing and camera optimizer state for a trained run."""

from __future__ import annotations

import argparse
import json
import re
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


def _natural_key(path: Path) -> List[Any]:
    text = str(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _safe_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def _safe_list(value: Any) -> List[float]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]
    try:
        return [float(v) for v in value]
    except TypeError:
        return [float(value)]


def _camera_metadata_index(camera: Any) -> Optional[int]:
    metadata = getattr(camera, "metadata", None)
    if not metadata or "cam_idx" not in metadata:
        return None
    value = metadata["cam_idx"]
    if torch.is_tensor(value):
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(value)


def _camera_row(dataset: Any, split: str, index: int) -> Dict[str, Any]:
    camera = dataset.cameras[index : index + 1]
    c2w = camera.camera_to_worlds[0].detach().cpu()
    center = c2w[:3, 3]
    # Nerfstudio camera convention uses the local z axis as the camera forward axis.
    view_direction = c2w[:3, 2]
    image_filename = None
    filenames = getattr(dataset, "image_filenames", None)
    if filenames is not None and index < len(filenames):
        image_filename = str(filenames[index])
    image_height = None
    image_width = None
    try:
        image = dataset[index]["image"]
        image_height = int(image.shape[0])
        image_width = int(image.shape[1])
    except Exception:
        pass
    distortion = getattr(camera, "distortion_params", None)
    return {
        "split": split,
        "dataset_index": int(index),
        "metadata_cam_idx": _camera_metadata_index(camera),
        "image_filename": image_filename,
        "image_width": image_width,
        "image_height": image_height,
        "camera_width": int(_safe_float(camera.width)),
        "camera_height": int(_safe_float(camera.height)),
        "fx": _safe_float(camera.fx),
        "fy": _safe_float(camera.fy),
        "cx": _safe_float(camera.cx),
        "cy": _safe_float(camera.cy),
        "camera_center": [float(v) for v in center.tolist()],
        "view_direction": [float(v) for v in view_direction.tolist()],
        "distortion_params": _safe_list(distortion),
    }


def _split_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    filenames = [row["image_filename"] for row in rows if row.get("image_filename")]
    metadata = [row["metadata_cam_idx"] for row in rows if row.get("metadata_cam_idx") is not None]
    sizes_match = [
        row["image_width"] == row["camera_width"] and row["image_height"] == row["camera_height"]
        for row in rows
        if row.get("image_width") is not None
    ]
    distortion_nonzero = []
    for row in rows:
        params = row.get("distortion_params") or []
        distortion_nonzero.append(any(abs(float(v)) > 1e-12 for v in params))
    return {
        "count": len(rows),
        "duplicate_filenames": sorted({name for name in filenames if filenames.count(name) > 1}),
        "filenames_natural_sorted": filenames == [str(p) for p in sorted([Path(name) for name in filenames], key=_natural_key)],
        "metadata_cam_idx_unique": len(metadata) == len(set(metadata)),
        "all_image_camera_sizes_match": bool(all(sizes_match)) if sizes_match else None,
        "distortion_nonzero_count": int(sum(distortion_nonzero)),
    }


def _camera_opt_summary(model: Any, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    param_groups = model.get_param_groups()
    camera_params = param_groups.get("camera_opt", [])
    optimizers = checkpoint.get("optimizers", {}) if isinstance(checkpoint, dict) else {}
    schedulers = checkpoint.get("schedulers", {}) if isinstance(checkpoint, dict) else {}
    opt_state = optimizers.get("camera_opt")
    state_count = 0
    if isinstance(opt_state, dict):
        state = opt_state.get("state", {})
        state_count = len(state)
    return {
        "param_group_present": "camera_opt" in param_groups,
        "parameter_count": int(sum(p.numel() for p in camera_params)),
        "trainable_parameter_count": int(sum(p.numel() for p in camera_params if p.requires_grad)),
        "checkpoint_optimizer_present": "camera_opt" in optimizers,
        "checkpoint_scheduler_present": "camera_opt" in schedulers,
        "checkpoint_optimizer_state_count": int(state_count),
        "model_has_camera_optimizer_attr": hasattr(model, "camera_optimizer"),
        "interpretation": (
            "camera_opt is configured in TrainerConfig but absent from model param groups/checkpoint optimizers"
            if "camera_opt" not in param_groups and "camera_opt" not in optimizers
            else "camera_opt appears active or partially active"
        ),
    }


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    config, pipeline, checkpoint_path, step = eval_setup(args.load_config)
    model = pipeline.model
    checkpoint = _load_checkpoint(Path(checkpoint_path))
    train_rows = [_camera_row(pipeline.datamanager.train_dataset, "train", i) for i in range(len(pipeline.datamanager.train_dataset.cameras))]
    eval_rows = [_camera_row(pipeline.datamanager.eval_dataset, "eval", i) for i in range(len(pipeline.datamanager.eval_dataset.cameras))]
    train_files = {row["image_filename"] for row in train_rows}
    eval_files = {row["image_filename"] for row in eval_rows}
    train_meta = {row["metadata_cam_idx"] for row in train_rows if row["metadata_cam_idx"] is not None}
    eval_meta = {row["metadata_cam_idx"] for row in eval_rows if row["metadata_cam_idx"] is not None}
    return {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "train_summary": _split_summary(train_rows),
        "eval_summary": _split_summary(eval_rows),
        "cross_split": {
            "shared_filenames": sorted(name for name in train_files & eval_files if name is not None),
            "shared_metadata_cam_idx": sorted(int(v) for v in train_meta & eval_meta),
        },
        "camera_opt": _camera_opt_summary(model, checkpoint),
        "rows": train_rows + eval_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({k: result[k] for k in ["train_summary", "eval_summary", "cross_split", "camera_opt"]}, indent=2))
    print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()
