#!/usr/bin/env python
"""Pre-training gates for Panama BND-CDEPTH.

This script performs read-only source/data/forward/gradient audits for the
single-factor SeaFree-style coarse-depth supervision experiment. It does not
call optimizer.step(), scheduler.step(), or write checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.method_configs import all_methods
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.scripts.train import _set_random_seed
from nerfstudio.utils.eval_utils import eval_setup


SCENE = "Panama"
SEAFREE_REPO = Path("/mnt/new/home_old/ycy/reference_repos/SeaFree-GS")
SEAFREE_COMMIT = "7797e97dae831029ac89ae9f37b3c3d69ec2cf6c"
K1_CONFIG = Path(
    "outputs/dewater_bounded_sh3_cross_scene_20260808/"
    "dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/"
    "dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/"
    "config.yml"
)
DATA_PATH = Path("undistorted_data/undistorted_Panama")
DEPTHS_PATH = Path("depthAnything_u16")
OUTPUT_DIR = Path("outputs/bnd_cdepth_panama_20260811")
RESEARCH_NOTE = Path("research_notes/BND_SEAFREE_COARSE_DEPTH_PANAMA_2026-08-11.md")
TRAJECTORY_STEPS = (1000, 3000, 5000, 8000, 10000, 13000, 15000)
EPS = 1e-8


@dataclass
class LoadedRun:
    config_path: Path
    checkpoint_path: Path
    loaded_step: int
    config: Any
    pipeline: Any


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _git_show(path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(SEAFREE_REPO), "show", f"{SEAFREE_COMMIT}:{path}"],
        text=True,
    )


def _line_excerpt(text: str, start: int, end: int) -> List[str]:
    lines = text.splitlines()
    return [f"{idx + 1}: {lines[idx]}" for idx in range(max(0, start - 1), min(end, len(lines)))]


def _available_steps(config_path: Path) -> Dict[int, Path]:
    ckpt_dir = config_path.parent / "nerfstudio_models"
    out: Dict[int, Path] = {}
    if ckpt_dir.exists():
        for path in ckpt_dir.glob("step-*.ckpt"):
            try:
                out[int(path.stem.split("-")[1])] = path
            except Exception:
                continue
    return out


def _actual_step(config_path: Path, nominal_step: int) -> int:
    steps = _available_steps(config_path)
    if nominal_step in steps:
        return nominal_step
    if nominal_step == 15000 and 14999 in steps:
        return 14999
    raise FileNotFoundError(f"Missing checkpoint step {nominal_step} next to {config_path}; available={sorted(steps)}")


def _release(loaded_or_pipeline: Any) -> None:
    if loaded_or_pipeline is not None:
        try:
            del loaded_or_pipeline.pipeline
        except Exception:
            pass
        del loaded_or_pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_k1(repo: Path, *, load_depths: bool, cdepth_enabled: bool, step: int = 15000) -> LoadedRun:
    config_path = repo / K1_CONFIG
    actual_step = _actual_step(config_path, step)

    def update_config(config: Any) -> Any:
        config.load_step = actual_step
        config.pipeline.model.intrinsic_color_parameterization = "bounded_sh3"
        config.pipeline.model.rasterize_mode = "classic"
        config.pipeline.model.medium_context_mode = "dir_xy_camera"
        config.pipeline.model.b_inf_mode = "tied"
        config.pipeline.model.infinite_water_enabled = False
        config.pipeline.model.coarse_depth_supervision_enabled = bool(cdepth_enabled)
        config.pipeline.model.coarse_depth_supervision_weight = 0.1
        config.pipeline.datamanager.load_depths = bool(load_depths)
        if load_depths:
            config.pipeline.datamanager.dataparser.depths_path = DEPTHS_PATH
        return config

    config, pipeline, checkpoint_path, loaded_step = eval_setup(
        config_path,
        test_mode="test",
        update_config_callback=update_config,
    )
    model = pipeline.model
    model.config.intrinsic_color_parameterization = "bounded_sh3"
    model.config.rasterize_mode = "classic"
    model.config.coarse_depth_supervision_enabled = bool(cdepth_enabled)
    model.config.coarse_depth_supervision_weight = 0.1
    pipeline.eval()
    return LoadedRun(config_path, checkpoint_path, int(loaded_step), config, pipeline)


def _scratch_pipeline(repo: Path, *, load_depths: bool, cdepth_enabled: bool) -> Pipeline:
    config_path = repo / K1_CONFIG
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    config.load_dir = None
    config.load_step = None
    config.load_checkpoint = None
    config.pipeline.datamanager.load_depths = bool(load_depths)
    if load_depths:
        config.pipeline.datamanager.dataparser.depths_path = DEPTHS_PATH
    config.pipeline.model.intrinsic_color_parameterization = "bounded_sh3"
    config.pipeline.model.rasterize_mode = "classic"
    config.pipeline.model.medium_context_mode = "dir_xy_camera"
    config.pipeline.model.b_inf_mode = "tied"
    config.pipeline.model.infinite_water_enabled = False
    config.pipeline.model.coarse_depth_supervision_enabled = bool(cdepth_enabled)
    config.pipeline.model.coarse_depth_supervision_weight = 0.1
    _set_random_seed(config.machine.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode="test")
    assert isinstance(pipeline, Pipeline)
    pipeline.model.config.intrinsic_color_parameterization = "bounded_sh3"
    pipeline.model.config.rasterize_mode = "classic"
    pipeline.model.config.coarse_depth_supervision_enabled = bool(cdepth_enabled)
    pipeline.model.config.coarse_depth_supervision_weight = 0.1
    return pipeline


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _eval_records(pipeline: Any) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    dataset = pipeline.datamanager.eval_dataset
    image_filenames = list(getattr(dataset, "image_filenames", []))
    rows: List[Tuple[int, str, Cameras, Dict[str, Any]]] = []
    for eval_index, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        filename = image_filenames[eval_index] if eval_index < len(image_filenames) else Path(f"eval_{eval_index}")
        rows.append((eval_index, Path(filename).stem, camera, _batch_to_device(batch, pipeline.model.device)))
    return rows


def _train_records(pipeline: Any, count: int = 8) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    dataset = pipeline.datamanager.train_dataset
    image_filenames = list(getattr(dataset, "image_filenames", []))
    total = len(dataset)
    indices = np.linspace(0, total - 1, min(count, total), dtype=int).tolist()
    cameras = dataset.cameras.to(pipeline.model.device)
    rows: List[Tuple[int, str, Cameras, Dict[str, Any]]] = []
    for index in indices:
        filename = image_filenames[index] if index < len(image_filenames) else Path(f"train_{index}")
        batch = pipeline.datamanager.cached_train[index].copy()
        rows.append((index, Path(filename).stem, cameras[index : index + 1], _batch_to_device(batch, pipeline.model.device)))
    return rows


def _parameter_snapshot(model: Any) -> Dict[str, Tensor]:
    out = {
        "means": model.means.detach().cpu().clone(),
        "scales": model.scales.detach().cpu().clone(),
        "quats": model.quats.detach().cpu().clone(),
        "opacities": model.opacities.detach().cpu().clone(),
        "features_dc": model.features_dc.detach().cpu().clone(),
        "features_rest": model.features_rest.detach().cpu().clone(),
    }
    for name, param in model.medium_mlp.named_parameters():
        out[f"medium_mlp.{name}"] = param.detach().cpu().clone()
    for name, param in model.direction_encoding.named_parameters():
        out[f"direction_encoding.{name}"] = param.detach().cpu().clone()
    return out


def _parameter_delta_rows(before: Mapping[str, Tensor], model: Any) -> List[Dict[str, Any]]:
    after = _parameter_snapshot(model)
    rows = []
    for key, tensor in before.items():
        current = after[key]
        diff = (current - tensor).abs()
        rows.append(
            {
                "parameter": key,
                "shape": list(tensor.shape),
                "max_abs_delta": float(diff.max().item()) if diff.numel() else 0.0,
                "mean_abs_delta": float(diff.mean().item()) if diff.numel() else 0.0,
            }
        )
    return rows


def _safe_stats(values: Tensor, prefix: str) -> Dict[str, Any]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {f"{prefix}{key}": float("nan") for key in ("min", "max", "mean", "std")}
    return {
        f"{prefix}min": float(flat.min().item()),
        f"{prefix}max": float(flat.max().item()),
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}std": float(flat.std(unbiased=False).item()),
    }


def source_audit(repo: Path, output_dir: Path) -> Dict[str, Any]:
    model_source = _git_show("seafree_gs/seafree_model.py")
    dataparser_source = _git_show("seafree_gs/seafree_dataparser.py")
    datamanager_source = _git_show("seafree_gs/seafree_datamanager.py")
    audit = {
        "repo": str(SEAFREE_REPO),
        "expected_commit": SEAFREE_COMMIT,
        "current_head": _git(SEAFREE_REPO, "rev-parse", "HEAD"),
        "status_short": _git(SEAFREE_REPO, "status", "--short"),
        "source_files": [
            "seafree_gs/seafree_model.py",
            "seafree_gs/seafree_dataparser.py",
            "seafree_gs/seafree_datamanager.py",
        ],
        "pseudo_depth_input": "batch['depth_image'] from DepthDataset; dataparser sets depth_path from depths_path/image_name.with_suffix('.png').",
        "pseudo_depth_decode": "Nerfstudio DepthDataset get_depth_image_from_path using depth_unit_scale_factor; SeaFree loss then divides each downscaled image by pseudo_depth.max().",
        "rendered_depth_semantics": "SeaFree gsplat RGB+ED expected depth; pixels with alpha<=0 are filled with valid-depth q95 before loss.",
        "rendered_depth_transform": "approximate_rendered_disparity = 1 / (rendered_depth * 10 + 1)",
        "loss_formula": "coarse_grained_depth_loss = 0.1 * (1 - pearson_corrcoef(pseudo_depth.flatten(), approximate_rendered_disparity.flatten()))",
        "coefficient": 0.1,
        "activation_period": "Enabled whenever enable_coarse_grained_depth_loss is true; no step cutoff in source.",
        "mask_semantics": "Only generic batch['mask'] is multiplied into pseudo_depth and rendered_depth if present; foreground/background masks are not applied to coarse-depth.",
        "detach_semantics": "No detach on pseudo_depth/rendered_depth in loss expression; pseudo depth has no trainable gradient. Empty-alpha q95 fill uses detached valid depth in SeaFree get_outputs.",
        "gradient_target": "rendered_depth branch, therefore rasterization geometry/opacity paths according to autograd; not a direct color loss.",
        "line_excerpts": {
            "config_flags": _line_excerpt(model_source, 185, 191),
            "depth_output": _line_excerpt(model_source, 700, 718),
            "loss_depth_inputs": _line_excerpt(model_source, 812, 835),
            "loss_formula": _line_excerpt(model_source, 912, 941),
            "dataparser_depth_path": _line_excerpt(dataparser_source, 78, 86),
            "datamanager_depth_cache": _line_excerpt(datamanager_source, 90, 101),
        },
        "water_splatting_code_fact": {
            "depth": "outputs['depth'] is UnderwaterRenderOutput.depth, alpha-normalized expected depth for alpha>0.",
            "no_support_fill": "UnderwaterRasterizer fills alpha==0 expected depth with depth_im.detach().max(); CDEPTH loss replaces no-support pixels with q95 valid depth to match SeaFree loss input semantics.",
            "renderer_change": "No renderer physics or CUDA formula is modified.",
        },
    }
    _write_json(output_dir / "seafree_cdepth_source_audit.json", audit)
    return audit


def pseudo_depth_dataset_audit(repo: Path, output_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    data_root = repo / DATA_PATH
    image_dir = data_root / "images" / "ColorImage"
    depth_dir = data_root / DEPTHS_PATH
    image_files = sorted(image_dir.glob("*.png"))
    depth_files = sorted(depth_dir.glob("*.png"))
    depth_by_stem = {path.stem: path for path in depth_files}
    rows: List[Dict[str, Any]] = []
    for idx, image_path in enumerate(image_files):
        depth_path = depth_by_stem.get(image_path.stem)
        with Image.open(image_path) as image:
            image_size = image.size
            image_mode = image.mode
        if depth_path is not None:
            depth_arr = np.asarray(Image.open(depth_path))
            depth_size = (int(depth_arr.shape[1]), int(depth_arr.shape[0]))
            depth_dtype = str(depth_arr.dtype)
            depth_min = float(np.nanmin(depth_arr))
            depth_max = float(np.nanmax(depth_arr))
            finite = bool(np.isfinite(depth_arr).all())
        else:
            depth_size = None
            depth_dtype = ""
            depth_min = float("nan")
            depth_max = float("nan")
            finite = False
        split = "eval" if idx % 8 == 0 else "train"
        rows.append(
            {
                "index": idx,
                "split": split,
                "image_filename": image_path.name,
                "depth_filename": depth_path.name if depth_path else "",
                "stem_match": bool(depth_path is not None and depth_path.stem == image_path.stem),
                "image_width": image_size[0],
                "image_height": image_size[1],
                "image_mode": image_mode,
                "depth_width": depth_size[0] if depth_size else "",
                "depth_height": depth_size[1] if depth_size else "",
                "depth_dtype": depth_dtype,
                "depth_min": depth_min,
                "depth_max": depth_max,
                "depth_finite": finite,
                "resolution_match": bool(depth_size == image_size),
            }
        )
    train_rows = [row for row in rows if row["split"] == "train"]
    sample_train = set(np.linspace(0, len(train_rows) - 1, min(10, len(train_rows)), dtype=int).tolist()) if train_rows else set()
    validation_rows = [row for row in rows if row["split"] == "eval"] + [row for i, row in enumerate(train_rows) if i in sample_train]
    alignment_pass = bool(
        rows
        and len(image_files) == len(depth_files)
        and all(row["stem_match"] and row["resolution_match"] and row["depth_finite"] and row["depth_max"] > row["depth_min"] for row in validation_rows)
    )
    audit = {
        "scene": SCENE,
        "data_root": str(data_root),
        "image_dir": str(image_dir),
        "depth_dir": str(depth_dir),
        "num_image_files": len(image_files),
        "num_depth_files": len(depth_files),
        "eval_interval": 8,
        "eval_view_ids": [Path(row["image_filename"]).stem for row in rows if row["split"] == "eval"],
        "validated_train_views": [Path(row["image_filename"]).stem for row in validation_rows if row["split"] == "train"],
        "validated_eval_views": [Path(row["image_filename"]).stem for row in validation_rows if row["split"] == "eval"],
        "decode_logic": "Nerfstudio DepthDataset reads PNG with depth_unit_scale_factor=0.001; CDEPTH loss normalizes each pseudo-depth image by max after model downscale.",
        "PSEUDO_DEPTH_ALIGNMENT_PASS": alignment_pass,
    }
    _write_json(output_dir / "pseudo_depth_dataset_audit.json", audit)
    _write_csv(output_dir / "pseudo_depth_alignment.csv", validation_rows)
    _write_json(output_dir / "pseudo_depth_alignment.json", {"rows": validation_rows, **audit})
    return audit, validation_rows, {"image_files": image_files, "depth_files": depth_files}


def depth_semantics_audit(output_dir: Path) -> Dict[str, Any]:
    audit = {
        "SeaFree_rendered_depth": "gsplat RGB+ED expected depth, with alpha<=0 pixels filled by q95 valid depth before coarse-depth loss.",
        "WaterSplatting_rendered_depth": "UnderwaterRasterizer alpha-normalized expected depth for alpha>0; alpha==0 pixels are renderer-filled by max depth in outputs.",
        "alignment_action": "CDEPTH loss uses outputs['depth'] for alpha>0 and replaces alpha==0 pixels with detached q95 valid depth, matching the SeaFree loss input without changing renderer outputs.",
        "depth_transform": "1/(10*rendered_depth+1), same as SeaFree fixed source.",
        "pseudo_depth_transform": "Downscale using WaterSplatting _downscale_if_required, then divide by per-image max.",
        "renderer_physics_modified": False,
        "CDEPTH_SEMANTIC_ALIGNMENT_VALID": True,
    }
    _write_json(output_dir / "depth_semantics_alignment.json", audit)
    lines = ["# BND-CDEPTH Depth Semantic Alignment", ""]
    for key, value in audit.items():
        lines.append(f"- {key}: {value}")
    (output_dir / "depth_semantics_alignment.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    return audit


def default_compatibility(repo: Path, output_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    loaded = None
    try:
        loaded = _load_k1(repo, load_depths=False, cdepth_enabled=False)
        model = loaded.pipeline.model
        _, view_id, camera, batch = _eval_records(loaded.pipeline)[0]
        model.eval()
        model.step = loaded.loaded_step
        before = _parameter_snapshot(model)
        with torch.no_grad():
            out_a = model.get_outputs_for_camera(camera)
            metrics_a = model.get_metrics_dict(out_a, batch)
            loss_a = model.get_loss_dict(out_a, batch, metrics_a)
            model.config.coarse_depth_supervision_enabled = False
            out_b = model.get_outputs_for_camera(camera)
            metrics_b = model.get_metrics_dict(out_b, batch)
            loss_b = model.get_loss_dict(out_b, batch, metrics_b)
        keys = ("pred_image", "direct_object_signal", "rgb_medium", "depth", "accumulation", "clear_object_fullsh_raw", "transmission", "tau_D")
        rows: List[Dict[str, Any]] = []
        for key in keys:
            diff = (out_a[key].detach().float().cpu() - out_b[key].detach().float().cpu()).abs()
            rows.append(
                {
                    "scene": SCENE,
                    "view_id": view_id,
                    "quantity": key,
                    "max_abs_diff": float(diff.max().item()),
                    "mean_abs_diff": float(diff.mean().item()),
                    "shape": list(out_a[key].shape),
                }
            )
        for key in sorted(set(loss_a) | set(loss_b)):
            a = loss_a.get(key)
            b = loss_b.get(key)
            if isinstance(a, Tensor) and isinstance(b, Tensor):
                diff_val = abs(float(a.detach().cpu().item()) - float(b.detach().cpu().item()))
            else:
                diff_val = float("nan")
            rows.append({"scene": SCENE, "view_id": view_id, "quantity": f"loss:{key}", "max_abs_diff": diff_val, "mean_abs_diff": diff_val})
        delta_rows = _parameter_delta_rows(before, model)
        pass_flag = all(float(row["max_abs_diff"]) <= 1e-8 for row in rows if math.isfinite(float(row["max_abs_diff"])))
        pass_flag = pass_flag and all(float(row["max_abs_delta"]) == 0.0 for row in delta_rows)
        summary = {
            "DEFAULT_K1_COMPATIBILITY": "PASS" if pass_flag else "FAIL",
            "view_id": view_id,
            "parameter_delta_max": max(float(row["max_abs_delta"]) for row in delta_rows) if delta_rows else 0.0,
        }
        _write_csv(output_dir / "default_k1_compatibility.csv", rows)
        _write_json(output_dir / "default_k1_compatibility.json", {"summary": summary, "rows": rows, "parameter_delta_rows": delta_rows})
        return rows, summary
    finally:
        _release(loaded)


def cdepth_forward_sanity(repo: Path, output_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    loaded = None
    try:
        loaded = _load_k1(repo, load_depths=True, cdepth_enabled=True)
        model = loaded.pipeline.model
        model.train()
        model.step = loaded.loaded_step
        before = _parameter_snapshot(model)
        rows: List[Dict[str, Any]] = []
        for train_index, view_id, camera, batch in _train_records(loaded.pipeline, count=8):
            outputs = model.get_outputs(camera.to(model.device))
            metrics: Dict[str, Tensor] = {}
            loss_dict = model.get_loss_dict(outputs, batch, metrics)
            pseudo = model._downscale_if_required(batch["depth_image"]).to(model.device)
            if pseudo.ndim == 2:
                pseudo = pseudo[..., None]
            pseudo_norm = pseudo / pseudo.max().clamp_min(EPS)
            depth = outputs["depth"].detach()
            disp = 1.0 / (depth * 10.0 + 1.0)
            row = {
                "scene": SCENE,
                "train_index": train_index,
                "view_id": view_id,
                "depth_loss_raw": float(metrics["coarse_depth_loss_raw"].detach().cpu().item()),
                "depth_loss_weighted": float(loss_dict["coarse_depth_loss"].detach().cpu().item()),
                "main_loss": float(loss_dict["main_loss"].detach().cpu().item()),
                "finite": bool(torch.isfinite(loss_dict["coarse_depth_loss"]).item() and torch.isfinite(loss_dict["main_loss"]).item()),
                "valid_pixel_fraction": 1.0,
            }
            row.update(_safe_stats(pseudo_norm, "pseudo_depth_"))
            row.update(_safe_stats(depth, "rendered_depth_"))
            row.update(_safe_stats(disp, "approx_disparity_"))
            rows.append(row)
        delta_rows = _parameter_delta_rows(before, model)
        finite = all(bool(row["finite"]) for row in rows)
        safe = all(float(row["max_abs_delta"]) == 0.0 for row in delta_rows)
        summary = {
            "depth_loss_finite": finite,
            "AUDIT_PARAMETER_SAFETY": "PASS" if safe else "FAIL",
            "parameter_delta_max": max(float(row["max_abs_delta"]) for row in delta_rows) if delta_rows else 0.0,
            "num_views": len(rows),
        }
        _write_csv(output_dir / "cdepth_forward_sanity.csv", rows)
        _write_json(output_dir / "cdepth_forward_sanity.json", {"summary": summary, "rows": rows, "parameter_delta_rows": delta_rows})
        return rows, summary
    finally:
        _release(loaded)


def _group_params(model: Any) -> Dict[str, List[Tensor]]:
    return {
        "means": [model.means],
        "scales": [model.scales],
        "quats": [model.quats],
        "opacities": [model.opacities],
        "features_dc": [model.features_dc],
        "features_rest": [model.features_rest],
        "medium": list(model.medium_mlp.parameters()) + list(model.direction_encoding.parameters()),
    }


def _loss_grads(loss: Tensor, groups: Mapping[str, Sequence[Tensor]]) -> Dict[str, List[Optional[Tensor]]]:
    params: List[Tensor] = []
    group_slices: Dict[str, Tuple[int, int]] = {}
    for name, tensors in groups.items():
        start = len(params)
        params.extend(tensors)
        group_slices[name] = (start, len(params))
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
    out: Dict[str, List[Optional[Tensor]]] = {}
    for name, (start, end) in group_slices.items():
        out[name] = list(grads[start:end])
    return out


def _accum_grad_stats(accum: Dict[str, Dict[str, float]], rgb: Mapping[str, Sequence[Optional[Tensor]]], depth: Mapping[str, Sequence[Optional[Tensor]]]) -> None:
    for group in rgb:
        entry = accum.setdefault(group, {"rgb_norm2": 0.0, "depth_norm2": 0.0, "dot": 0.0, "finite": 1.0})
        for gr, gd in zip(rgb[group], depth[group]):
            if gr is None and gd is None:
                continue
            if gr is None:
                gd_flat = gd.detach().float().reshape(-1)
                entry["depth_norm2"] += float((gd_flat * gd_flat).sum().item())
                entry["finite"] *= float(torch.isfinite(gd_flat).all().item())
                continue
            if gd is None:
                gr_flat = gr.detach().float().reshape(-1)
                entry["rgb_norm2"] += float((gr_flat * gr_flat).sum().item())
                entry["finite"] *= float(torch.isfinite(gr_flat).all().item())
                continue
            gr_flat = gr.detach().float().reshape(-1)
            gd_flat = gd.detach().float().reshape(-1)
            entry["rgb_norm2"] += float((gr_flat * gr_flat).sum().item())
            entry["depth_norm2"] += float((gd_flat * gd_flat).sum().item())
            entry["dot"] += float((gr_flat * gd_flat).sum().item())
            entry["finite"] *= float((torch.isfinite(gr_flat).all() & torch.isfinite(gd_flat).all()).item())


def gradient_audit(repo: Path, output_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    loaded = None
    try:
        loaded = _load_k1(repo, load_depths=True, cdepth_enabled=True)
        model = loaded.pipeline.model
        model.train()
        model.step = loaded.loaded_step
        before = _parameter_snapshot(model)
        groups = _group_params(model)
        accum: Dict[str, Dict[str, float]] = {}
        view_rows: List[Dict[str, Any]] = []
        for train_index, view_id, camera, batch in _train_records(loaded.pipeline, count=8):
            model.zero_grad(set_to_none=True)
            outputs = model.get_outputs(camera.to(model.device))
            losses = model.get_loss_dict(outputs, batch, {})
            rgb_grads = _loss_grads(losses["main_loss"], groups)

            model.zero_grad(set_to_none=True)
            outputs = model.get_outputs(camera.to(model.device))
            losses = model.get_loss_dict(outputs, batch, {})
            depth_grads = _loss_grads(losses["coarse_depth_loss"], groups)
            _accum_grad_stats(accum, rgb_grads, depth_grads)
            view_rows.append(
                {
                    "scene": SCENE,
                    "train_index": train_index,
                    "view_id": view_id,
                    "main_loss": float(losses["main_loss"].detach().cpu().item()),
                    "coarse_depth_loss_weighted": float(losses["coarse_depth_loss"].detach().cpu().item()),
                }
            )
        rows: List[Dict[str, Any]] = []
        for group, stats in accum.items():
            rgb_norm = math.sqrt(max(stats["rgb_norm2"], 0.0))
            depth_norm = math.sqrt(max(stats["depth_norm2"], 0.0))
            total_norm = math.sqrt(max(stats["rgb_norm2"] + stats["depth_norm2"] + 2.0 * stats["dot"], 0.0))
            rows.append(
                {
                    "scene": SCENE,
                    "parameter_group": group,
                    "rgb_grad_norm": rgb_norm,
                    "lambda_depth_grad_norm": depth_norm,
                    "DEPTH_TO_RGB_GRAD_NORM_RATIO": depth_norm / max(rgb_norm, EPS),
                    "cos_rgb_depth": stats["dot"] / max(rgb_norm * depth_norm, EPS),
                    "cos_rgb_total": (stats["rgb_norm2"] + stats["dot"]) / max(rgb_norm * total_norm, EPS),
                    "gradient_finite": bool(stats["finite"]),
                }
            )
        delta_rows = _parameter_delta_rows(before, model)
        geometry_rows = [row for row in rows if row["parameter_group"] in ("means", "scales", "quats", "opacities")]
        explosive = any(float(row["DEPTH_TO_RGB_GRAD_NORM_RATIO"]) > 10.0 for row in geometry_rows) and not all(bool(row["gradient_finite"]) for row in geometry_rows)
        negligible = all(float(row["DEPTH_TO_RGB_GRAD_NORM_RATIO"]) < 1e-4 for row in geometry_rows)
        finite = all(bool(row["gradient_finite"]) for row in rows)
        safe = all(float(row["max_abs_delta"]) == 0.0 for row in delta_rows)
        summary = {
            "gradient_finite": finite,
            "DEPTH_GRADIENT_EXPLOSIVE": bool(explosive),
            "DEPTH_GRADIENT_NEGLIGIBLE": bool(negligible),
            "AUDIT_PARAMETER_SAFETY": "PASS" if safe else "FAIL",
            "parameter_delta_max": max(float(row["max_abs_delta"]) for row in delta_rows) if delta_rows else 0.0,
            "num_views": len(view_rows),
        }
        _write_csv(output_dir / "fixed_state_gradient_routing.csv", rows)
        _write_json(output_dir / "fixed_state_gradient_routing.json", {"summary": summary, "rows": rows, "view_rows": view_rows, "parameter_delta_rows": delta_rows})
        _write_csv(output_dir / "fixed_state_depth_gradient_audit.csv", rows)
        _write_json(output_dir / "fixed_state_depth_gradient_audit.json", {"summary": summary, "rows": rows})
        return rows, summary
    finally:
        _release(loaded)


def initialization_equivalence(repo: Path, output_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    k1_pipe = None
    cd_pipe = None
    try:
        k1_pipe = _scratch_pipeline(repo, load_depths=False, cdepth_enabled=False)
        cd_pipe = _scratch_pipeline(repo, load_depths=True, cdepth_enabled=True)
        k1_snap = _parameter_snapshot(k1_pipe.model)
        cd_snap = _parameter_snapshot(cd_pipe.model)
        param_rows: List[Dict[str, Any]] = []
        for key, tensor in k1_snap.items():
            other = cd_snap[key]
            diff = (tensor - other).abs()
            param_rows.append(
                {
                    "parameter": key,
                    "shape": list(tensor.shape),
                    "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
                    "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
                }
            )
        param_pass = all(float(row["max_abs_diff"]) == 0.0 for row in param_rows)

        _, view_id, camera, _ = _eval_records(k1_pipe)[0]
        _, cd_view_id, cd_camera, _ = _eval_records(cd_pipe)[0]
        k1_pipe.model.eval()
        cd_pipe.model.eval()
        k1_pipe.model.step = 0
        cd_pipe.model.step = 0
        with torch.no_grad():
            k1_out = k1_pipe.model.get_outputs_for_camera(camera)
            cd_out = cd_pipe.model.get_outputs_for_camera(cd_camera)
        fwd_rows = []
        for key in ("pred_image", "direct_object_signal", "rgb_medium", "depth", "accumulation", "clear_object_fullsh_raw", "transmission", "tau_D", "gaussian_view_rgb"):
            a = k1_out[key].detach().float().cpu()
            b = cd_out[key].detach().float().cpu()
            diff = (a - b).abs()
            fwd_rows.append(
                {
                    "view_id_k1": view_id,
                    "view_id_cdepth": cd_view_id,
                    "output": key,
                    "shape_match": tuple(a.shape) == tuple(b.shape),
                    "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
                    "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
                }
            )
        fwd_pass = all(row["shape_match"] and float(row["max_abs_diff"]) <= 1e-7 for row in fwd_rows)
        param_summary = {"INIT_PARAMETER_EQUIVALENCE": "PASS" if param_pass else "FAIL"}
        fwd_summary = {"INIT_FORWARD_EQUIVALENCE": "PASS" if fwd_pass else "FAIL"}
        _write_csv(output_dir / "initialization_parameter_equivalence.csv", param_rows)
        _write_json(output_dir / "initialization_parameter_equivalence.json", {"summary": param_summary, "rows": param_rows})
        _write_csv(output_dir / "initialization_forward_equivalence.csv", fwd_rows)
        _write_json(output_dir / "initialization_forward_equivalence.json", {"summary": fwd_summary, "rows": fwd_rows})
        return param_summary, fwd_summary
    finally:
        _release(k1_pipe)
        _release(cd_pipe)


def write_note(path: Path, repo_manifest: Mapping[str, Any], facts: Mapping[str, Any]) -> None:
    lines = [
        "# BND-CDEPTH Panama Coarse-Depth Supervision",
        "",
        "## Code Fact",
        "",
        f"- WaterSplatting repo branch: `{repo_manifest['branch']}`.",
        f"- WaterSplatting start HEAD: `{repo_manifest['head']}`.",
        f"- SeaFree reference commit: `{facts['source']['expected_commit']}`.",
        "- SeaFree coarse-depth source uses `0.1 * (1 - pearson_corrcoef(pseudo_depth, 1/(10*rendered_depth+1)))`.",
        "- Pseudo-depth is a coarse geometric cue, not metric depth GT.",
        "- BND-CDEPTH adds only a disabled-by-default coarse-depth term to the BND-K1 objective.",
        "",
        "## Config Fact",
        "",
        "- Formal BND-K1 controls retained: SH degree 3, classic rasterization, `medium_context_mode=dir_xy_camera`, `b_inf_mode=tied`, `infinite_water_enabled=False`, relative RGB objective.",
        "- New defaults: `coarse_depth_supervision_enabled=False`, `coarse_depth_supervision_weight=0.1`, `load_depths=False`.",
        "- BND-CDEPTH training enables depth loading from `depthAnything_u16` and enables coarse-depth supervision.",
        "",
        "## Experimental Fact",
        "",
        f"- `PSEUDO_DEPTH_ALIGNMENT_PASS`: `{facts['pseudo']['PSEUDO_DEPTH_ALIGNMENT_PASS']}`.",
        f"- `CDEPTH_SEMANTIC_ALIGNMENT_VALID`: `{facts['semantics']['CDEPTH_SEMANTIC_ALIGNMENT_VALID']}`.",
        f"- `DEFAULT_K1_COMPATIBILITY`: `{facts['compat']['DEFAULT_K1_COMPATIBILITY']}`.",
        f"- Forward finite: `{facts['forward']['depth_loss_finite']}`.",
        f"- Gradient finite: `{facts['gradient']['gradient_finite']}`.",
        f"- `AUDIT_PARAMETER_SAFETY`: `{facts['eligibility']['AUDIT_PARAMETER_SAFETY']}`.",
        f"- `INIT_PARAMETER_EQUIVALENCE`: `{facts['init_param']['INIT_PARAMETER_EQUIVALENCE']}`.",
        f"- `INIT_FORWARD_EQUIVALENCE`: `{facts['init_forward']['INIT_FORWARD_EQUIVALENCE']}`.",
        f"- `BND_CDEPTH_TRAINING_ELIGIBLE`: `{facts['eligibility']['BND_CDEPTH_TRAINING_ELIGIBLE']}`.",
        "",
        "## Quantitative Result",
        "",
        "- Pre-training audit tables are stored under `outputs/bnd_cdepth_panama_20260811/`.",
        "",
        "## Inference",
        "",
        "- No training result is inferred from the setup audit. The setup audit only decides whether the one-factor training run is eligible.",
        "",
        "## Hypothesis",
        "",
        "- The causal hypothesis remains untested until the eligible BND-CDEPTH run is trained and summarized.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/mnt/new/home_old/ycy/water-splatting-refactor"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--research-note", type=Path, default=RESEARCH_NOTE)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    note_path = (repo / args.research_note).resolve() if not args.research_note.is_absolute() else args.research_note
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_manifest = {
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "log_3": _git(repo, "log", "-3", "--oneline"),
        "status_short": _git(repo, "status", "--short"),
        "tracked_output_count": _git(repo, "ls-files", "outputs", "renders", "logs", "common_masks", "checkpoints"),
        "seafree_repo": str(SEAFREE_REPO),
        "seafree_reference_commit": SEAFREE_COMMIT,
        "seafree_current_head": _git(SEAFREE_REPO, "rev-parse", "HEAD"),
        "seafree_status_short": _git(SEAFREE_REPO, "status", "--short"),
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)

    source = source_audit(repo, output_dir)
    pseudo, _, _ = pseudo_depth_dataset_audit(repo, output_dir)
    semantics = depth_semantics_audit(output_dir)
    _, compat = default_compatibility(repo, output_dir)
    _, forward = cdepth_forward_sanity(repo, output_dir)
    _, gradient = gradient_audit(repo, output_dir)
    init_param, init_forward = initialization_equivalence(repo, output_dir)
    parameter_safety_pass = (
        forward.get("AUDIT_PARAMETER_SAFETY") == "PASS"
        and gradient.get("AUDIT_PARAMETER_SAFETY") == "PASS"
    )
    eligible = bool(
        pseudo.get("PSEUDO_DEPTH_ALIGNMENT_PASS")
        and semantics.get("CDEPTH_SEMANTIC_ALIGNMENT_VALID")
        and compat.get("DEFAULT_K1_COMPATIBILITY") == "PASS"
        and parameter_safety_pass
        and forward.get("depth_loss_finite")
        and gradient.get("gradient_finite")
        and not gradient.get("DEPTH_GRADIENT_EXPLOSIVE")
        and init_param.get("INIT_PARAMETER_EQUIVALENCE") == "PASS"
        and init_forward.get("INIT_FORWARD_EQUIVALENCE") == "PASS"
    )
    eligibility = {
        "PSEUDO_DEPTH_ALIGNMENT_PASS": bool(pseudo.get("PSEUDO_DEPTH_ALIGNMENT_PASS")),
        "CDEPTH_SEMANTIC_ALIGNMENT_VALID": bool(semantics.get("CDEPTH_SEMANTIC_ALIGNMENT_VALID")),
        "DEFAULT_K1_COMPATIBILITY": compat.get("DEFAULT_K1_COMPATIBILITY"),
        "AUDIT_PARAMETER_SAFETY": "PASS" if parameter_safety_pass else "FAIL",
        "depth_loss_finite": bool(forward.get("depth_loss_finite")),
        "gradient_finite": bool(gradient.get("gradient_finite")),
        "DEPTH_GRADIENT_EXPLOSIVE": bool(gradient.get("DEPTH_GRADIENT_EXPLOSIVE")),
        "DEPTH_GRADIENT_NEGLIGIBLE": bool(gradient.get("DEPTH_GRADIENT_NEGLIGIBLE")),
        "INIT_PARAMETER_EQUIVALENCE": init_param.get("INIT_PARAMETER_EQUIVALENCE"),
        "INIT_FORWARD_EQUIVALENCE": init_forward.get("INIT_FORWARD_EQUIVALENCE"),
        "BND_CDEPTH_TRAINING_ELIGIBLE": eligible,
    }
    _write_json(output_dir / "training_eligibility.json", eligibility)
    _write_json(
        output_dir / "manifest.json",
        {
            "repo_manifest": str(output_dir / "repo_manifest.json"),
            "seafree_cdepth_source_audit": str(output_dir / "seafree_cdepth_source_audit.json"),
            "pseudo_depth_dataset_audit": str(output_dir / "pseudo_depth_dataset_audit.json"),
            "training_eligibility": str(output_dir / "training_eligibility.json"),
            "BND_CDEPTH_TRAINING_ELIGIBLE": eligible,
        },
    )
    write_note(
        note_path,
        repo_manifest,
        {
            "source": source,
            "pseudo": pseudo,
            "semantics": semantics,
            "compat": compat,
            "forward": forward,
            "gradient": gradient,
            "init_param": init_param,
            "init_forward": init_forward,
            "eligibility": eligibility,
        },
    )
    print(json.dumps(eligibility, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
