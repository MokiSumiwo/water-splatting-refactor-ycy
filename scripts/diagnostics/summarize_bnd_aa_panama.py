#!/usr/bin/env python
"""Summarize the Panama BND antialiased rasterization experiment."""

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
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.method_configs import all_methods
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.scripts.train import _set_random_seed
from nerfstudio.utils.eval_utils import eval_setup


SCENE = "Panama"
CHANNELS = ("r", "g", "b")
TRAJECTORY_STEPS = (1000, 3000, 5000, 8000, 10000, 13000, 15000)
FINAL_STEP = 15000
EPS = 1e-8
LUMA_WEIGHTS = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)


@dataclass(frozen=True)
class RunSpec:
    name: str
    config_relpath: str
    parameterization: str
    rasterize_mode: str
    role: str
    reused: bool


RUNS: Dict[str, RunSpec] = {
    "M1": RunSpec(
        "M1",
        "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/"
        "cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml",
        "legacy",
        "classic",
        "reference_m1",
        True,
    ),
    "K1": RunSpec(
        "K1",
        "outputs/dewater_bounded_sh3_cross_scene_20260808/"
        "dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/"
        "dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/"
        "config.yml",
        "bounded_sh3",
        "classic",
        "bounded_classic_control",
        True,
    ),
    "AA": RunSpec(
        "AA",
        "outputs/bnd_aa_panama_20260810/panama_bnd_aa_seed42_step0_to_15000/water-splatting/20260810_bnd_aa/config.yml",
        "bounded_sh3",
        "antialiased",
        "bounded_antialiased_candidate",
        False,
    ),
}


@dataclass
class LoadedRun:
    run: str
    nominal_step: int
    loaded_step: int
    config_path: Path
    checkpoint_path: Path
    config: Any
    pipeline: Any

    @property
    def model(self) -> Any:
        return self.pipeline.model


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
    except Exception:
        return "unknown"


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if float(v) == float(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _safe_quantile(values: Tensor, q: float) -> float:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    if q <= 0.0:
        return float(flat.min().item())
    if q >= 1.0:
        return float(flat.max().item())
    rank = max(1, min(flat.numel(), int(math.ceil(float(q) * flat.numel()))))
    return float(torch.kthvalue(flat, rank).values.item())


def _stats(values: Tensor, prefix: str) -> Dict[str, float]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    names = ("count", "mean", "p01", "p05", "p10", "p50", "p90", "p95", "p99", "max")
    if flat.numel() == 0:
        return {f"{prefix}{name}": float("nan") for name in names}
    return {
        f"{prefix}count": int(flat.numel()),
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}p01": _safe_quantile(flat, 0.01),
        f"{prefix}p05": _safe_quantile(flat, 0.05),
        f"{prefix}p10": _safe_quantile(flat, 0.10),
        f"{prefix}p50": _safe_quantile(flat, 0.50),
        f"{prefix}p90": _safe_quantile(flat, 0.90),
        f"{prefix}p95": _safe_quantile(flat, 0.95),
        f"{prefix}p99": _safe_quantile(flat, 0.99),
        f"{prefix}max": float(flat.max().item()),
    }


def _threshold_fraction(values: Tensor, threshold: float, op: str) -> float:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    if op == "gt":
        return float((flat > threshold).float().mean().item())
    if op == "lt":
        return float((flat < threshold).float().mean().item())
    if op == "abs_gt":
        return float((flat.abs() > threshold).float().mean().item())
    raise ValueError(op)


def _threshold_rows(values: Tensor, prefix: str, thresholds: Sequence[float], op: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    values = values.detach().float()
    if values.ndim > 0 and values.shape[-1] == 3:
        for idx, channel in enumerate(CHANNELS):
            for threshold in thresholds:
                out[f"{prefix}_{channel}_{op}_{threshold:g}"] = _threshold_fraction(values[..., idx], threshold, op)
        for threshold in thresholds:
            out[f"{prefix}_all_{op}_{threshold:g}"] = _threshold_fraction(values.reshape(-1), threshold, op)
    else:
        for threshold in thresholds:
            out[f"{prefix}_{op}_{threshold:g}"] = _threshold_fraction(values, threshold, op)
    return out


def _channel_stats(values: Tensor, prefix: str) -> Dict[str, float]:
    values = values.detach().float()
    out: Dict[str, float] = {}
    if values.ndim > 0 and values.shape[-1] == 3:
        for idx, channel in enumerate(CHANNELS):
            out.update(_stats(values[..., idx], f"{prefix}_{channel}_"))
        out.update(_stats(values.reshape(-1), f"{prefix}_all_"))
    else:
        out.update(_stats(values.reshape(-1), f"{prefix}_"))
    return out


def _pearson(a: Tensor, b: Tensor) -> float:
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    finite = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum()).clamp_min(EPS)
    return float(((x * y).sum() / denom).item())


def _aggregate_numeric(rows: Sequence[Mapping[str, Any]], id_fields: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(id_fields)
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (float, int, bool))})
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and isinstance(row[key], (float, int, bool)) and float(row[key]) == float(row[key])]
        if vals:
            out[key] = _mean(vals)
    return out


def _available_steps(config_path: Path) -> Dict[int, Path]:
    ckpt_dir = config_path.parent / "nerfstudio_models"
    if not ckpt_dir.exists():
        return {}
    out: Dict[int, Path] = {}
    for path in ckpt_dir.glob("step-*.ckpt"):
        try:
            out[int(path.stem.split("-")[1])] = path
        except ValueError:
            continue
    return out


def _actual_step(config_path: Path, nominal_step: int) -> Optional[int]:
    steps = _available_steps(config_path)
    if nominal_step in steps:
        return nominal_step
    if nominal_step == 15000 and 14999 in steps:
        return 14999
    return None


def _load_run(repo: Path, run: str, nominal_step: int) -> LoadedRun:
    spec = RUNS[run]
    config_path = repo / spec.config_relpath
    actual_step = _actual_step(config_path, nominal_step)
    if actual_step is None:
        raise FileNotFoundError(f"Missing {run} checkpoint for nominal step {nominal_step}: {config_path}")

    def update_config(config: Any) -> Any:
        config.load_step = actual_step
        return config

    config, pipeline, checkpoint_path, loaded_step = eval_setup(config_path, test_mode="test", update_config_callback=update_config)
    pipeline.model.config.intrinsic_color_parameterization = spec.parameterization
    pipeline.model.config.rasterize_mode = spec.rasterize_mode
    pipeline.eval()
    return LoadedRun(run, nominal_step, int(loaded_step), config_path, checkpoint_path, config, pipeline)


def _release_loaded(loaded: Optional[LoadedRun]) -> None:
    if loaded is None:
        return
    try:
        del loaded.pipeline
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _view_records(loaded: LoadedRun) -> List[Tuple[int, str, Any, Mapping[str, Any]]]:
    dataset = loaded.pipeline.datamanager.eval_dataset
    image_filenames = list(getattr(dataset, "image_filenames", []))
    rows = []
    for eval_index, (camera, batch) in enumerate(loaded.pipeline.datamanager.fixed_indices_eval_dataloader):
        filename = image_filenames[eval_index] if eval_index < len(image_filenames) else Path(f"eval_{eval_index}")
        rows.append((eval_index, Path(filename).stem, camera, batch))
    return rows


def _metric_images(model: Any, pred: Tensor, gt: Tensor) -> Dict[str, float]:
    pred = pred.detach().float().clamp(0.0, 1.0).to(model.device)
    gt = gt.detach().float().clamp(0.0, 1.0).to(model.device)
    pred_nchw = torch.moveaxis(pred, -1, 0)[None, ...]
    gt_nchw = torch.moveaxis(gt, -1, 0)[None, ...]
    mse = float(((pred - gt) ** 2).mean().item())
    return {
        "psnr": float(model.psnr(gt_nchw, pred_nchw).item()),
        "ssim": float(model.ssim(gt_nchw, pred_nchw).item()),
        "lpips": float(model.lpips(gt_nchw, pred_nchw).item()),
        "mse": mse,
    }


def _safe_cpu(tensor: Tensor) -> Tensor:
    return tensor.detach().float().cpu()


def _luma(rgb: Tensor) -> Tensor:
    return (rgb.detach().float() * LUMA_WEIGHTS).sum(dim=-1)


def _rgb_l1(image: Tensor) -> Tensor:
    return image.detach().float().abs().sum(dim=-1)


def _rgb_l2(image: Tensor) -> Tensor:
    return torch.linalg.norm(image.detach().float(), dim=-1)


def _object_support(item: Mapping[str, Any]) -> Tensor:
    return item["outputs"]["accumulation"].detach().float()[..., 0] > 0.01


def _masked_values(values: Tensor, mask: Tensor) -> Tensor:
    if values.ndim == mask.ndim:
        return values[mask]
    while mask.ndim < values.ndim:
        mask = mask[..., None].expand(*values.shape)
    return values[mask]


def _image_values(items: Sequence[Mapping[str, Any]], key: str, channel: int, support: str) -> Tensor:
    vals: List[Tensor] = []
    for item in items:
        tensor = item["outputs"][key].detach().float()
        if tensor.ndim == 2:
            tensor = tensor[..., None]
        image = tensor[..., 0] if tensor.shape[-1] == 1 else tensor[..., channel]
        if support == "object":
            image = image[_object_support(item)]
        vals.append(image.reshape(-1))
    return torch.cat(vals, dim=0) if vals else torch.empty(0)


def _pooled_channel_stat(items: Sequence[Mapping[str, Any]], key: str, stat: str, support: str) -> float:
    q_map = {"p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99}
    values = []
    for channel in range(3):
        flat = _image_values(items, key, channel, support)
        if stat == "mean":
            values.append(float(flat.mean().item()) if flat.numel() else float("nan"))
        else:
            values.append(_safe_quantile(flat, q_map[stat]))
    return _mean(values)


def _threshold_pooled_channel(items: Sequence[Mapping[str, Any]], key: str, threshold: float, op: str, support: str) -> float:
    values = []
    for channel in range(3):
        flat = _image_values(items, key, channel, support)
        if flat.numel() == 0:
            continue
        if op == "lt":
            values.append(float((flat < threshold).float().mean().item()))
        elif op == "gt":
            values.append(float((flat > threshold).float().mean().item()))
        else:
            raise ValueError(op)
    return _mean(values)


def _append_component_stats(row: Dict[str, Any], items: Sequence[Mapping[str, Any]]) -> None:
    row["tau_eval_object_support_pooled_channel_mean_p90"] = _pooled_channel_stat(items, "tau_D", "p90", "object")
    row["J_clear_eval_object_support_pooled_channel_mean_p99"] = _pooled_channel_stat(
        items, "clear_object_fullsh_raw", "p99", "object"
    )
    row["P_T_lt_0.1_object_support_pooled_channel_mean"] = _threshold_pooled_channel(items, "transmission", 0.1, "lt", "object")
    row["P_J_gt_1_object_support_pooled_channel_mean"] = _threshold_pooled_channel(
        items, "clear_object_fullsh_raw", 1.0, "gt", "object"
    )
    for key, prefix in (
        ("medium_attn", "beta_D_raw"),
        ("medium_bs", "beta_B"),
        ("medium_rgb", "medium_rgb"),
        ("rgb_medium", "rgb_medium"),
        ("tau_D", "tau_D"),
        ("transmission", "T_D"),
        ("clear_object_fullsh_raw", "J"),
        ("depth", "depth"),
    ):
        vals = []
        obj_vals = []
        for item in items:
            if key not in item["outputs"]:
                continue
            tensor = item["outputs"][key]
            vals.append(tensor.reshape(-1, tensor.shape[-1] if tensor.ndim == 3 else 1))
            obj_vals.append(_masked_values(tensor, _object_support(item)).reshape(-1))
        if vals:
            row.update(_channel_stats(torch.cat(vals, dim=0), f"{prefix}_all"))
        if obj_vals:
            row.update(_stats(torch.cat(obj_vals, dim=0), f"{prefix}_object_"))
    row["P_T_lt_0.3_object"] = _threshold_pooled_channel(items, "transmission", 0.3, "lt", "object")
    row["P_T_lt_0.2_object"] = _threshold_pooled_channel(items, "transmission", 0.2, "lt", "object")
    row["P_T_lt_0.05_object"] = _threshold_pooled_channel(items, "transmission", 0.05, "lt", "object")


def _model_geometry_stats(model: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    if hasattr(model, "scales"):
        # Model scales are stored in log space and exponentiated before rasterization.
        row.update(_stats(torch.exp(model.scales.detach().float()).reshape(-1), "gaussian_scale_"))
    if hasattr(model, "means"):
        row["gaussian_means_finite"] = bool(torch.isfinite(model.means.detach()).all().item())
    return row


def _boundary_stats(items: Sequence[Mapping[str, Any]], run: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"scene": SCENE, "run": run}
    if RUNS[run].parameterization != "bounded_sh3":
        row["boundary_stats_available"] = False
        row["BOUNDARY_ESCAPE"] = False
        return row
    colors: List[Tensor] = []
    logits: List[Tensor] = []
    derivs: List[Tensor] = []
    for item in items:
        out = item["outputs"]
        c = out.get("gaussian_view_rgb")
        s = out.get("gaussian_view_logits")
        d = out.get("gaussian_sigmoid_derivative")
        visible = out.get("gaussian_visible_mask")
        if not isinstance(c, Tensor) or not isinstance(s, Tensor):
            continue
        if isinstance(visible, Tensor) and visible.numel() == c.shape[0]:
            mask = visible.bool()
            c = c[mask]
            s = s[mask]
            if isinstance(d, Tensor):
                d = d[mask]
        colors.append(c.reshape(-1, 3))
        logits.append(s.reshape(-1, 3))
        if isinstance(d, Tensor):
            derivs.append(d.reshape(-1, 3))
    if not colors:
        row["boundary_stats_available"] = False
        row["BOUNDARY_ESCAPE"] = False
        return row
    c_all = torch.cat(colors, dim=0)
    s_all = torch.cat(logits, dim=0)
    row["boundary_stats_available"] = True
    row.update(_channel_stats(c_all, "c"))
    row.update(_threshold_rows(c_all, "c", (0.95, 0.99), "gt"))
    row.update(_channel_stats(s_all, "s"))
    row.update(_threshold_rows(s_all, "s_abs", (5.0, 8.0), "abs_gt"))
    if derivs:
        row.update(_channel_stats(torch.cat(derivs, dim=0), "sigmoid_derivative"))
    row["BOUNDARY_ESCAPE"] = bool(row.get("c_all_gt_0.99", 0.0) > 0.05 or row.get("s_abs_all_abs_gt_5", 0.0) > 0.05)
    return row


def _cache_outputs(repo: Path, run: str, nominal_step: int, include_opacity: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    loaded: Optional[LoadedRun] = None
    opacity_rows: List[Dict[str, Any]] = []
    try:
        loaded = _load_run(repo, run, nominal_step)
        model = loaded.model
        items: List[Dict[str, Any]] = []
        for eval_index, view_id, camera, batch in _view_records(loaded):
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera)
                gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
                metrics = _metric_images(model, outputs["pred_image"], gt)
                if include_opacity:
                    opacity_rows.append(_opacity_compensation_stats(model, camera, run, nominal_step, view_id))
            keep = (
                "pred_image",
                "direct_object_signal",
                "clear_object_fullsh_raw",
                "transmission",
                "tau_D",
                "rgb_medium",
                "medium_rgb",
                "medium_bs",
                "medium_attn",
                "depth",
                "accumulation",
                "gaussian_view_rgb",
                "gaussian_view_logits",
                "gaussian_sigmoid_derivative",
                "gaussian_visible_mask",
            )
            tensors = {key: _safe_cpu(outputs[key]) for key in keep if key in outputs and isinstance(outputs[key], Tensor)}
            items.append(
                {
                    "scene": SCENE,
                    "run": run,
                    "nominal_step": nominal_step,
                    "loaded_step": int(loaded.loaded_step),
                    "eval_index": eval_index,
                    "view_id": view_id,
                    "camera_id": eval_index,
                    "gt": _safe_cpu(gt),
                    "outputs": tensors,
                    "metrics": metrics,
                }
            )
        meta = {
            "scene": SCENE,
            "run": run,
            "nominal_step": nominal_step,
            "loaded_step": int(loaded.loaded_step),
            "role": RUNS[run].role,
            "config_path": str(loaded.config_path),
            "checkpoint_path": str(loaded.checkpoint_path),
            "parameterization": RUNS[run].parameterization,
            "rasterize_mode": RUNS[run].rasterize_mode,
            "reused": RUNS[run].reused,
            "seed": getattr(getattr(loaded.config, "machine", None), "seed", ""),
            "sh_degree": getattr(model.config, "sh_degree", ""),
            "medium_context_mode": getattr(model.config, "medium_context_mode", ""),
            "b_inf_mode": getattr(model.config, "b_inf_mode", ""),
            "infinite_water_enabled": getattr(model.config, "infinite_water_enabled", ""),
            "gaussian_count": int(model.num_points),
            "num_eval_views": len(items),
            "view_ids": ";".join(item["view_id"] for item in items),
        }
        meta.update(_model_geometry_stats(model))
        return items, meta, opacity_rows
    finally:
        _release_loaded(loaded)


def _opacity_compensation_stats(model: Any, camera: Cameras, run: str, nominal_step: int, view_id: str) -> Dict[str, Any]:
    camera = camera.to(model.device)
    camera_downscale = model._get_downscale_factor()
    camera.rescale_output_resolution(1 / camera_downscale)
    try:
        R = camera.camera_to_worlds[0, :3, :3]
        T = camera.camera_to_worlds[0, :3, 3:4]
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=model.device, dtype=R.dtype))
        R = R @ R_edit
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        if model.crop_box is not None and not model.training:
            crop_ids = model.crop_box.within(model.means).squeeze()
        else:
            crop_ids = None
        if crop_ids is not None and crop_ids.sum() != 0:
            opacities_crop = model.opacities[crop_ids]
            means_crop = model.means[crop_ids]
            scales_crop = model.scales[crop_ids]
            quats_crop = model.quats[crop_ids]
        else:
            opacities_crop = model.opacities
            means_crop = model.means
            scales_crop = model.scales
            quats_crop = model.quats
        _, _, radii, _, comp, _, _ = model.underwater_rasterizer.project(
            means=means_crop,
            scales=scales_crop,
            quats=quats_crop,
            viewmat=viewmat,
            fx=camera.fx.item(),
            fy=camera.fy.item(),
            cx=camera.cx.item(),
            cy=camera.cy.item(),
            height=int(camera.height.item()),
            width=int(camera.width.item()),
            clip_thresh=model.config.clip_thresh,
        )
    finally:
        camera.rescale_output_resolution(camera_downscale)
    visible = radii > 0
    raw = torch.sigmoid(opacities_crop).reshape(-1)[visible.reshape(-1)].detach().float()
    comp_v = comp.reshape(-1)[visible.reshape(-1)].detach().float()
    effective = raw * comp_v if model.config.rasterize_mode == "antialiased" else raw
    row: Dict[str, Any] = {
        "scene": SCENE,
        "run": run,
        "nominal_step": nominal_step,
        "view_id": view_id,
        "rasterize_mode": model.config.rasterize_mode,
        "visible_gaussian_count": int(raw.numel()),
    }
    row.update(_stats(raw, "raw_opacity_"))
    row.update(_stats(comp_v, "aa_compensation_"))
    row.update(_stats(effective, "effective_opacity_"))
    row["AA_OPACITY_COLLAPSE"] = bool(_threshold_fraction(effective, 1e-4, "lt") > 0.25) if effective.numel() else False
    row["AA_OPACITY_EXTREME"] = bool(_threshold_fraction(effective, 0.99, "gt") > 0.05) if effective.numel() else False
    return row


def trajectory_audit(repo: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    trajectory_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    opacity_rows: List[Dict[str, Any]] = []
    for run in ("K1", "AA"):
        config_path = repo / RUNS[run].config_relpath
        for step in TRAJECTORY_STEPS:
            actual = _actual_step(config_path, step)
            if actual is None:
                missing_rows.append(
                    {
                        "scene": SCENE,
                        "run": run,
                        "nominal_step": step,
                        "config_path": str(config_path),
                        "available_steps": ";".join(str(s) for s in sorted(_available_steps(config_path))),
                        "status": "MISSING_CHECKPOINT",
                    }
                )
                continue
            items, meta, opacity = _cache_outputs(repo, run, step, include_opacity=True)
            checkpoint_rows.append(meta)
            opacity_rows.extend(opacity)
            row: Dict[str, Any] = {
                "scene": SCENE,
                "run": run,
                "nominal_step": step,
                "loaded_step": meta["loaded_step"],
                "config_path": meta["config_path"],
                "checkpoint_path": meta["checkpoint_path"],
                "rasterize_mode": RUNS[run].rasterize_mode,
                "parameterization": RUNS[run].parameterization,
                "num_eval_views": len(items),
                "gaussian_count": meta["gaussian_count"],
            }
            row.update({key: value for key, value in meta.items() if key.startswith("gaussian_scale_") or key == "gaussian_means_finite"})
            for key in ("psnr", "ssim", "lpips", "mse"):
                row[key] = _mean(item["metrics"][key] for item in items)
            _append_component_stats(row, items)
            trajectory_rows.append(row)
            boundary = _boundary_stats(items, run)
            boundary.update({"nominal_step": step, "loaded_step": meta["loaded_step"]})
            boundary_rows.append(boundary)
    # Final M1 only.
    items, meta, opacity = _cache_outputs(repo, "M1", FINAL_STEP, include_opacity=True)
    checkpoint_rows.append(meta)
    opacity_rows.extend(opacity)
    row = {
        "scene": SCENE,
        "run": "M1",
        "nominal_step": FINAL_STEP,
        "loaded_step": meta["loaded_step"],
        "config_path": meta["config_path"],
        "checkpoint_path": meta["checkpoint_path"],
        "rasterize_mode": "classic",
        "parameterization": "legacy",
        "num_eval_views": len(items),
        "gaussian_count": meta["gaussian_count"],
    }
    row.update({key: value for key, value in meta.items() if key.startswith("gaussian_scale_") or key == "gaussian_means_finite"})
    for key in ("psnr", "ssim", "lpips", "mse"):
        row[key] = _mean(item["metrics"][key] for item in items)
    _append_component_stats(row, items)
    trajectory_rows.append(row)
    boundary_rows.append({**_boundary_stats(items, "M1"), "nominal_step": FINAL_STEP, "loaded_step": meta["loaded_step"]})
    return trajectory_rows, missing_rows, boundary_rows, checkpoint_rows, opacity_rows


def _gaussian_kernel1d(sigma: float) -> Tensor:
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _gaussian_blur(image: Tensor, sigma: float) -> Tensor:
    img = image.detach().float().permute(2, 0, 1)[None, ...]
    c = img.shape[1]
    k = _gaussian_kernel1d(sigma)
    pad = k.numel() // 2
    kh = k.reshape(1, 1, 1, -1).expand(c, 1, 1, -1)
    kv = k.reshape(1, 1, -1, 1).expand(c, 1, -1, 1)
    out = F.conv2d(F.pad(img, (pad, pad, 0, 0), mode="reflect"), kh, groups=c)
    out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode="reflect"), kv, groups=c)
    return out[0].permute(1, 2, 0)


def _gradient_magnitude_luma(image: Tensor) -> Tensor:
    lum = _luma(image).float()
    dx = torch.zeros_like(lum)
    dy = torch.zeros_like(lum)
    dx[:, 1:] = lum[:, 1:] - lum[:, :-1]
    dy[1:, :] = lum[1:, :] - lum[:-1, :]
    return torch.sqrt(dx.square() + dy.square() + EPS)


def final_view_audit(repo: Path, render_dir: Path, tile_width: int) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    cached: Dict[str, List[Dict[str, Any]]] = {}
    checkpoint_rows: List[Dict[str, Any]] = []
    for run in ("M1", "K1", "AA"):
        items, meta, _ = _cache_outputs(repo, run, FINAL_STEP)
        cached[run] = items
        checkpoint_rows.append(meta)
    view_ids = [item["view_id"] for item in cached["M1"]]
    for run in ("K1", "AA"):
        if [item["view_id"] for item in cached[run]] != view_ids:
            raise RuntimeError(f"view mismatch for {run}")
    by_run_view = {run: {item["view_id"]: item for item in items} for run, items in cached.items()}
    per_view_rows: List[Dict[str, Any]] = []
    region_rows: List[Dict[str, Any]] = []
    frequency_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    recomposition_rows: List[Dict[str, Any]] = []
    mse_attribution_rows: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []

    luma_values = torch.cat([_luma(item["gt"]).reshape(-1) for item in cached["M1"]], dim=0)
    bright_q5_threshold = _safe_quantile(luma_values, 0.80)
    for view_id in view_ids:
        gt = by_run_view["M1"][view_id]["gt"]
        for run in ("M1", "K1", "AA"):
            item = by_run_view[run][view_id]
            row = {
                "scene": SCENE,
                "view_id": view_id,
                "run": run,
                "psnr": item["metrics"]["psnr"],
                "ssim": item["metrics"]["ssim"],
                "lpips": item["metrics"]["lpips"],
                "mse": item["metrics"]["mse"],
            }
            if run == "AA":
                row["delta_psnr_vs_K1"] = item["metrics"]["psnr"] - by_run_view["K1"][view_id]["metrics"]["psnr"]
                row["delta_ssim_vs_K1"] = item["metrics"]["ssim"] - by_run_view["K1"][view_id]["metrics"]["ssim"]
                row["delta_lpips_vs_K1"] = item["metrics"]["lpips"] - by_run_view["K1"][view_id]["metrics"]["lpips"]
            per_view_rows.append(row)

        h, w = gt.shape[:2]
        yy = torch.linspace(0.0, 1.0, h).reshape(h, 1).expand(h, w)
        support = _object_support(by_run_view["M1"][view_id])
        jmax = by_run_view["M1"][view_id]["outputs"]["clear_object_fullsh_raw"].amax(dim=-1)
        masks = {
            "M1_J_gt_1": support & (jmax > 1.0),
            "M1_J_le_1": support & (jmax <= 1.0),
            "GT_brightness_Q5": _luma(gt) > bright_q5_threshold,
            "bottom20_image_y": yy >= 0.8,
        }
        for mask_name, mask in masks.items():
            for run in ("M1", "K1", "AA"):
                mse_map = (by_run_view[run][view_id]["outputs"]["pred_image"] - gt).square().mean(dim=-1)
                vals = mse_map[mask]
                region_rows.append(
                    {
                        "scene": SCENE,
                        "view_id": view_id,
                        "mask": mask_name,
                        "run": run,
                        "pixel_count": int(mask.sum().item()),
                        "pixel_fraction": float(mask.float().mean().item()),
                        "mse": float(vals.mean().item()) if vals.numel() else float("nan"),
                        "mean_abs_residual_l1": float(_rgb_l1(by_run_view[run][view_id]["outputs"]["pred_image"] - gt)[mask].mean().item()) if vals.numel() else float("nan"),
                    }
                )
        for run in ("K1", "AA"):
            residual = gt - by_run_view[run][view_id]["outputs"]["pred_image"]
            total_e = float(_rgb_l2(residual).square().sum().item())
            for sigma in (3.0, 9.0):
                low = _gaussian_blur(residual, sigma)
                high = residual - low
                low_e = float(low.square().sum().item())
                high_e = float(high.square().sum().item())
                frequency_rows.append(
                    {
                        "scene": SCENE,
                        "view_id": view_id,
                        "run": run,
                        "sigma_px": sigma,
                        "LOW_FREQ_ENERGY_FRACTION": low_e / max(low_e + high_e, EPS),
                        "HIGH_FREQ_ENERGY_FRACTION": high_e / max(low_e + high_e, EPS),
                        "low_energy": low_e,
                        "high_energy": high_e,
                    }
                )
            edge = _gradient_magnitude_luma(gt)
            resid_mag = _rgb_l2(residual)
            edge_thresh = _safe_quantile(edge.reshape(-1), 0.80)
            edge_mask = edge >= edge_thresh
            edge_energy = float(resid_mag.square()[edge_mask].sum().item())
            edge_rows.append(
                {
                    "scene": SCENE,
                    "view_id": view_id,
                    "run": run,
                    "edge_definition": "top20 percent GT luminance gradient magnitude",
                    "residual_edge_pearson": _pearson(resid_mag, edge),
                    "edge_pixel_fraction": float(edge_mask.float().mean().item()),
                    "residual_energy_fraction_top20_edge": edge_energy / max(total_e, EPS),
                    "residual_energy_fraction_non_edge": float(resid_mag.square()[~edge_mask].sum().item() / max(total_e, EPS)),
                    "edge_residual_energy": edge_energy,
                    "total_residual_energy": total_e,
                    "EDGE_ENRICHMENT": edge_energy / max(total_e, EPS) / max(float(edge_mask.float().mean().item()), EPS),
                }
            )
    region_rows.extend(_aggregate_region_rows(region_rows))
    frequency_rows.extend(_aggregate_frequency_rows(frequency_rows))
    edge_rows.extend(_aggregate_edge_rows(edge_rows))
    recomposition_rows.extend(_recomposition_rows(by_run_view, view_ids))
    mse_attribution_rows.extend(_mse_attribution_rows(by_run_view, view_ids))
    _write_visuals(render_dir, by_run_view, view_ids, tile_width, manifest)
    return per_view_rows, region_rows, frequency_rows, edge_rows, recomposition_rows, mse_attribution_rows, checkpoint_rows, manifest


def _aggregate_region_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for mask_name in sorted({row["mask"] for row in rows}):
        for run in ("M1", "K1", "AA"):
            selected = [row for row in rows if row["mask"] == mask_name and row["run"] == run]
            out.append(_aggregate_numeric(selected, {"scene": SCENE, "view_id": "AGGREGATE", "mask": mask_name, "run": run}))
    return out


def _aggregate_frequency_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for run in ("K1", "AA"):
        for sigma in (3.0, 9.0):
            selected = [row for row in rows if row["run"] == run and float(row["sigma_px"]) == sigma]
            out.append(_aggregate_numeric(selected, {"scene": SCENE, "view_id": "AGGREGATE", "run": run, "sigma_px": sigma}))
    return out


def _aggregate_edge_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for run in ("K1", "AA"):
        selected = [row for row in rows if row["run"] == run]
        out.append(_aggregate_numeric(selected, {"scene": SCENE, "view_id": "AGGREGATE", "run": run}))
    return out


def _cancellation_metrics(delta_d: Tensor, delta_b: Tensor, mask: Tensor) -> Dict[str, Any]:
    dd = delta_d[mask].reshape(-1, 3).float()
    db = delta_b[mask].reshape(-1, 3).float()
    if dd.numel() == 0:
        return {}
    sum_abs = (dd + db).abs().sum(dim=-1)
    raw_abs = dd.abs().sum(dim=-1) + db.abs().sum(dim=-1)
    dot = (dd * db).sum(dim=-1)
    nd = torch.linalg.norm(dd, dim=-1)
    nb = torch.linalg.norm(db, dim=-1)
    cos = dot / (nd * nb + EPS)
    return {
        "deltaD_deltaB_pearson_luma": _pearson((dd * LUMA_WEIGHTS).sum(dim=-1), (db * LUMA_WEIGHTS).sum(dim=-1)),
        "flattened_cosine_similarity": float(dot.sum().item() / max(float(torch.sqrt((dd * dd).sum() * (db * db).sum()).item()), EPS)),
        "CANCELLATION_RESIDUAL_RATIO": float(sum_abs.mean().item() / max(float(raw_abs.mean().item()), EPS)),
        "RECOMP_EFFICIENCY": 1.0 - float(sum_abs.mean().item() / max(float(raw_abs.mean().item()), EPS)),
        "RECOMP_RAW_CHANGE": float(raw_abs.mean().item()),
        "RECOMP_FINAL_CHANGE": float(sum_abs.mean().item()),
        "cos_theta_p50": _safe_quantile(cos, 0.50),
        "P_cos_lt_-0.9": _threshold_fraction(cos, -0.9, "lt"),
    }


def _recomposition_rows(by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], view_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for view_id in view_ids:
        m1 = by_run_view["M1"][view_id]
        for run in ("K1", "AA"):
            item = by_run_view[run][view_id]
            mask = _object_support(m1)
            d_d = item["outputs"]["direct_object_signal"] - m1["outputs"]["direct_object_signal"]
            d_b = item["outputs"]["rgb_medium"] - m1["outputs"]["rgb_medium"]
            d_i = item["outputs"]["pred_image"] - m1["outputs"]["pred_image"]
            row = {
                "scene": SCENE,
                "view_id": view_id,
                "run": run,
                "reference": "M1",
                "mean_abs_DeltaD_l1": float(_rgb_l1(d_d)[mask].mean().item()),
                "mean_abs_DeltaB_l1": float(_rgb_l1(d_b)[mask].mean().item()),
                "mean_abs_DeltaI_l1": float(_rgb_l1(d_i)[mask].mean().item()),
            }
            row.update(_cancellation_metrics(d_d, d_b, mask))
            rows.append(row)
    for run in ("K1", "AA"):
        rows.append(_aggregate_numeric([row for row in rows if row["run"] == run], {"scene": SCENE, "view_id": "AGGREGATE", "run": run, "reference": "M1"}))
    return rows


def _mse_attribution_rows(by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], view_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for run in ("K1", "AA"):
        for view_id in view_ids:
            m1 = by_run_view["M1"][view_id]
            item = by_run_view[run][view_id]
            gt = m1["gt"].float()
            d_d = item["outputs"]["direct_object_signal"] - m1["outputs"]["direct_object_signal"]
            d_b = item["outputs"]["rgb_medium"] - m1["outputs"]["rgb_medium"]
            e0 = m1["outputs"]["pred_image"] - gt
            e1 = item["outputs"]["pred_image"] - gt
            delta_mse = float((e1.square().mean() - e0.square().mean()).item())
            c_direct = float((2.0 * e0 * d_d + d_d.square()).mean().item())
            c_medium = float((2.0 * e0 * d_b + d_b.square()).mean().item())
            c_cross = float((2.0 * d_d * d_b).mean().item())
            rows.append(
                {
                    "scene": SCENE,
                    "view_id": view_id,
                    "run": run,
                    "reference": "M1",
                    "DeltaMSE_actual": delta_mse,
                    "C_direct": c_direct,
                    "C_medium": c_medium,
                    "C_cross": c_cross,
                    "component_sum": c_direct + c_medium + c_cross,
                    "absolute_closure_error": abs(delta_mse - (c_direct + c_medium + c_cross)),
                }
            )
        rows.append(_aggregate_numeric([row for row in rows if row["run"] == run], {"scene": SCENE, "view_id": "AGGREGATE", "run": run, "reference": "M1"}))
    return rows


def _rgb_to_uint8(image: Tensor) -> Image.Image:
    arr = (image.detach().float().clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _gray_to_uint8(values: Tensor, scale: float) -> Image.Image:
    scale = max(float(scale), EPS)
    arr = (values.detach().float().clamp_min(0.0) / scale).clamp(0.0, 1.0)
    return Image.fromarray((arr * 255.0).round().byte().cpu().numpy(), mode="L").convert("RGB")


def _signed_to_rgb(values: Tensor, scale: float) -> Image.Image:
    scale = max(float(scale), EPS)
    v = (values.detach().float() / scale).clamp(-1.0, 1.0)
    pos = v.clamp_min(0)
    neg = (-v).clamp_min(0)
    white = torch.ones((*v.shape, 3), dtype=torch.float32)
    red = torch.tensor([1.0, 0.12, 0.08])
    blue = torch.tensor([0.08, 0.28, 1.0])
    rgb = white * (1 - pos[..., None]) + red * pos[..., None]
    rgb = rgb * (1 - neg[..., None]) + blue * neg[..., None]
    return Image.fromarray((rgb.clamp(0, 1) * 255).round().byte().cpu().numpy(), mode="RGB")


def _mask_to_uint8(mask: Tensor) -> Image.Image:
    arr = (mask.detach().bool().cpu().numpy().astype("uint8") * 255)
    return Image.fromarray(arr, mode="L").convert("RGB")


def _overlay_mask(base: Image.Image, mask: Tensor) -> Image.Image:
    image = base.convert("RGB")
    overlay = Image.new("RGB", image.size, (255, 40, 20))
    mask_img = Image.fromarray((mask.detach().bool().cpu().numpy().astype("uint8") * 120), mode="L")
    image.paste(overlay, (0, 0), mask_img)
    return image


def _tile(image: Image.Image, label: str, width: int) -> Image.Image:
    if width > 0 and image.width != width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    label_h = 30
    out = Image.new("RGB", (image.width, image.height + label_h), "white")
    out.paste(image, (0, label_h))
    ImageDraw.Draw(out).text((6, 8), label, fill="black")
    return out


def _save_sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Image.Image]]], tile_width: int, manifest: List[Dict[str, Any]], output_type: str, view_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = []
    for row in rows:
        tiles = [_tile(img, label, tile_width) for label, img in row]
        width = sum(tile.width for tile in tiles) + 6 * (len(tiles) - 1)
        height = max(tile.height for tile in tiles)
        canvas = Image.new("RGB", (width, height), "white")
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, 0))
            x += tile.width + 6
        rendered.append(canvas)
    if not rendered:
        return
    width = max(row.width for row in rendered)
    height = sum(row.height for row in rendered) + 6 * (len(rendered) - 1)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rendered:
        sheet.paste(row, (0, y))
        y += row.height + 6
    sheet.save(path)
    manifest.append(
        {
            "file_path": str(path),
            "scene": SCENE,
            "runs": "M1;K1;AA",
            "step": FINAL_STEP,
            "output_type": output_type,
            "view_ids": ";".join(view_ids),
            "width": sheet.width,
            "height": sheet.height,
        }
    )


def _write_visuals(render_dir: Path, by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], view_ids: Sequence[str], tile_width: int, manifest: List[Dict[str, Any]]) -> None:
    residual_scale = 0.1
    hf_scale = 0.05
    rgb_delta_scale = 0.1
    for view_id in view_ids:
        gt = by_run_view["M1"][view_id]["gt"]
        for run in ("M1", "K1", "AA"):
            residual_scale = max(residual_scale, float(_rgb_l2(by_run_view[run][view_id]["outputs"]["pred_image"] - gt).max().item()))
        hf_scale = max(
            hf_scale,
            float(_luma(_gaussian_blur(gt - by_run_view["K1"][view_id]["outputs"]["pred_image"], 3.0)).abs().max().item()),
            float(_luma(_gaussian_blur(gt - by_run_view["AA"][view_id]["outputs"]["pred_image"], 3.0)).abs().max().item()),
        )
        rgb_delta_scale = max(
            rgb_delta_scale,
            float((by_run_view["AA"][view_id]["outputs"]["direct_object_signal"] - by_run_view["K1"][view_id]["outputs"]["direct_object_signal"]).abs().max().item()),
            float((by_run_view["AA"][view_id]["outputs"]["rgb_medium"] - by_run_view["K1"][view_id]["outputs"]["rgb_medium"]).abs().max().item()),
        )
    rows_underwater: List[List[Tuple[str, Image.Image]]] = []
    rows_clear: List[List[Tuple[str, Image.Image]]] = []
    rows_residual: List[List[Tuple[str, Image.Image]]] = []
    rows_hf: List[List[Tuple[str, Image.Image]]] = []
    rows_edge: List[List[Tuple[str, Image.Image]]] = []
    rows_highj: List[List[Tuple[str, Image.Image]]] = []
    rows_direct: List[List[Tuple[str, Image.Image]]] = []
    rows_medium: List[List[Tuple[str, Image.Image]]] = []
    for view_id in view_ids:
        gt = by_run_view["M1"][view_id]["gt"]
        rows_underwater.append([(f"{view_id} GT", _rgb_to_uint8(gt))] + [(run, _rgb_to_uint8(by_run_view[run][view_id]["outputs"]["pred_image"])) for run in ("M1", "K1", "AA")])
        rows_clear.append([(f"{view_id} {run}", _rgb_to_uint8(by_run_view[run][view_id]["outputs"]["clear_object_fullsh_raw"])) for run in ("M1", "K1", "AA")])
        m1_resid = _rgb_l2(by_run_view["M1"][view_id]["outputs"]["pred_image"] - gt)
        k1_resid = _rgb_l2(by_run_view["K1"][view_id]["outputs"]["pred_image"] - gt)
        aa_resid = _rgb_l2(by_run_view["AA"][view_id]["outputs"]["pred_image"] - gt)
        rows_residual.append(
            [
                (f"{view_id} M1 residual", _gray_to_uint8(m1_resid, residual_scale)),
                ("K1 residual", _gray_to_uint8(k1_resid, residual_scale)),
                ("AA residual", _gray_to_uint8(aa_resid, residual_scale)),
                ("K1 excess", _gray_to_uint8((k1_resid - m1_resid).clamp_min(0.0), residual_scale)),
                ("AA excess", _gray_to_uint8((aa_resid - m1_resid).clamp_min(0.0), residual_scale)),
            ]
        )
        k1_res = gt - by_run_view["K1"][view_id]["outputs"]["pred_image"]
        aa_res = gt - by_run_view["AA"][view_id]["outputs"]["pred_image"]
        k1_h3 = k1_res - _gaussian_blur(k1_res, 3.0)
        aa_h3 = aa_res - _gaussian_blur(aa_res, 3.0)
        k1_h9 = k1_res - _gaussian_blur(k1_res, 9.0)
        aa_h9 = aa_res - _gaussian_blur(aa_res, 9.0)
        rows_hf.append(
            [
                (f"{view_id} K1 HF s3", _signed_to_rgb(_luma(k1_h3), hf_scale)),
                ("AA HF s3", _signed_to_rgb(_luma(aa_h3), hf_scale)),
                ("K1 HF s9", _signed_to_rgb(_luma(k1_h9), hf_scale)),
                ("AA HF s9", _signed_to_rgb(_luma(aa_h9), hf_scale)),
            ]
        )
        edge = _gradient_magnitude_luma(gt)
        edge_mask = edge >= _safe_quantile(edge.reshape(-1), 0.80)
        rows_edge.append(
            [
                (f"{view_id} GT edge top20", _mask_to_uint8(edge_mask)),
                ("K1 residual edge", _overlay_mask(_gray_to_uint8(k1_resid, residual_scale), edge_mask)),
                ("AA residual edge", _overlay_mask(_gray_to_uint8(aa_resid, residual_scale), edge_mask)),
            ]
        )
        high_j = _object_support(by_run_view["M1"][view_id]) & (by_run_view["M1"][view_id]["outputs"]["clear_object_fullsh_raw"].amax(dim=-1) > 1.0)
        rows_highj.append(
            [
                (f"{view_id} M1 J>1", _mask_to_uint8(high_j)),
                ("K1 residual overlay", _overlay_mask(_gray_to_uint8(k1_resid, residual_scale), high_j)),
                ("AA residual overlay", _overlay_mask(_gray_to_uint8(aa_resid, residual_scale), high_j)),
            ]
        )
        d_delta = by_run_view["AA"][view_id]["outputs"]["direct_object_signal"] - by_run_view["K1"][view_id]["outputs"]["direct_object_signal"]
        b_delta = by_run_view["AA"][view_id]["outputs"]["rgb_medium"] - by_run_view["K1"][view_id]["outputs"]["rgb_medium"]
        rows_direct.append(
            [
                (f"{view_id} K1 direct", _rgb_to_uint8(by_run_view["K1"][view_id]["outputs"]["direct_object_signal"])),
                ("AA direct", _rgb_to_uint8(by_run_view["AA"][view_id]["outputs"]["direct_object_signal"])),
                ("abs delta", _rgb_to_uint8(d_delta.abs() / max(rgb_delta_scale, EPS))),
            ]
        )
        rows_medium.append(
            [
                (f"{view_id} K1 medium", _rgb_to_uint8(by_run_view["K1"][view_id]["outputs"]["rgb_medium"])),
                ("AA medium", _rgb_to_uint8(by_run_view["AA"][view_id]["outputs"]["rgb_medium"])),
                ("abs delta", _rgb_to_uint8(b_delta.abs() / max(rgb_delta_scale, EPS))),
            ]
        )
    for filename, rows, output_type in (
        ("contact_sheet_underwater_m1_k1_aa.png", rows_underwater, "underwater"),
        ("contact_sheet_clear_raw_m1_k1_aa.png", rows_clear, "clear_object_fullsh_raw_display_clamp01"),
        ("contact_sheet_residual_m1_k1_aa.png", rows_residual, "underwater_residual_and_excess"),
        ("contact_sheet_high_frequency_residual_k1_aa.png", rows_hf, "high_frequency_residual_sigma3_sigma9"),
        ("contact_sheet_edge_residual_k1_aa.png", rows_edge, "edge_residual_overlay"),
        ("contact_sheet_high_j_mask_k1_aa.png", rows_highj, "fixed_m1_high_j_mask_overlay"),
        ("contact_sheet_direct_k1_aa_delta.png", rows_direct, "direct_object_signal_delta"),
        ("contact_sheet_medium_k1_aa_delta.png", rows_medium, "rgb_medium_delta"),
    ):
        _save_sheet(render_dir / filename, rows, tile_width, manifest, output_type, view_ids)


def initialization_and_forward_audit(repo: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    config_path = repo / RUNS["AA"].config_relpath
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    config.load_dir = None
    config.load_step = None
    config.pipeline.model.rasterize_mode = "classic"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_random_seed(config.machine.seed)
    classic_pipeline = config.pipeline.setup(device=device, test_mode="test")
    assert isinstance(classic_pipeline, Pipeline)
    classic_model = classic_pipeline.model
    param_keys = ("features_dc", "features_rest", "means", "scales", "quats", "opacities")
    classic_params = {key: getattr(classic_model, key).detach().cpu().clone() for key in param_keys}
    classic_medium = classic_model.medium_mlp.tcnn_encoding.params.detach().cpu().clone()
    _release_pipeline(classic_pipeline)

    config2 = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    config2.pipeline.datamanager._target = all_methods[config2.method_name].pipeline.datamanager._target
    config2.load_dir = None
    config2.load_step = None
    config2.pipeline.model.rasterize_mode = "antialiased"
    _set_random_seed(config2.machine.seed)
    aa_pipeline = config2.pipeline.setup(device=device, test_mode="test")
    assert isinstance(aa_pipeline, Pipeline)
    aa_model = aa_pipeline.model
    init_rows = []
    for key in param_keys:
        candidate = getattr(aa_model, key).detach().cpu()
        base = classic_params[key]
        row = {"scene": SCENE, "parameter": key, "classic_shape": list(base.shape), "aa_shape": list(candidate.shape)}
        if tuple(base.shape) == tuple(candidate.shape):
            diff = candidate.float() - base.float()
            row.update({"max_abs_diff": float(diff.abs().max().item()) if diff.numel() else 0.0, "mean_abs_diff": float(diff.abs().mean().item()) if diff.numel() else 0.0})
        else:
            row.update({"max_abs_diff": float("nan"), "mean_abs_diff": float("nan")})
        row["INITIALIZATION_MATCH"] = bool(row["max_abs_diff"] == 0.0)
        init_rows.append(row)
    med = aa_model.medium_mlp.tcnn_encoding.params.detach().cpu()
    diff = med.float() - classic_medium.float()
    init_rows.append(
        {
            "scene": SCENE,
            "parameter": "medium_mlp.tcnn_encoding.params",
            "classic_shape": list(classic_medium.shape),
            "aa_shape": list(med.shape),
            "max_abs_diff": float(diff.abs().max().item()) if diff.numel() else 0.0,
            "mean_abs_diff": float(diff.abs().mean().item()) if diff.numel() else 0.0,
            "INITIALIZATION_MATCH": bool(float(diff.abs().max().item()) == 0.0) if diff.numel() else True,
        }
    )

    # Same initialized parameters, same camera, toggle rasterize mode for a forward-difference audit.
    aa_pipeline.eval()
    eval_index, view_id, camera, batch = _view_records_for_pipeline(aa_pipeline)[0]
    aa_model.config.rasterize_mode = "classic"
    with torch.no_grad():
        classic_out = aa_model.get_outputs_for_camera(camera)
        gt = aa_model.composite_with_background(aa_model.get_gt_img(batch["image"]), classic_out["background"])
        classic_metrics = _metric_images(aa_model, classic_out["pred_image"], gt)
        classic_opacity = _opacity_compensation_stats(aa_model, camera, "INIT_CLASSIC", 0, view_id)
    aa_model.config.rasterize_mode = "antialiased"
    with torch.no_grad():
        aa_out = aa_model.get_outputs_for_camera(camera)
        aa_metrics = _metric_images(aa_model, aa_out["pred_image"], gt)
        aa_opacity = _opacity_compensation_stats(aa_model, camera, "INIT_AA", 0, view_id)
    forward_rows = []
    for key in ("pred_image", "accumulation", "depth", "direct_object_signal", "rgb_medium", "clear_object_fullsh_raw", "transmission", "tau_D"):
        if key in classic_out and key in aa_out and isinstance(classic_out[key], Tensor) and isinstance(aa_out[key], Tensor):
            diff = aa_out[key].detach().float() - classic_out[key].detach().float()
            forward_rows.append(
                {
                    "scene": SCENE,
                    "view_id": view_id,
                    "output": key,
                    "max_abs_diff_AA_minus_classic": float(diff.abs().max().item()),
                    "mean_abs_diff_AA_minus_classic": float(diff.abs().mean().item()),
                    "classic_finite": bool(torch.isfinite(classic_out[key]).all().item()),
                    "aa_finite": bool(torch.isfinite(aa_out[key]).all().item()),
                }
            )
    forward_rows.append({"scene": SCENE, "view_id": view_id, "output": "classic_psnr", "value": classic_metrics["psnr"]})
    forward_rows.append({"scene": SCENE, "view_id": view_id, "output": "aa_psnr", "value": aa_metrics["psnr"]})
    forward_rows.append({"scene": SCENE, "view_id": view_id, "output": "classic_opacity_stats", **classic_opacity})
    forward_rows.append({"scene": SCENE, "view_id": view_id, "output": "aa_opacity_stats", **aa_opacity})
    _release_pipeline(aa_pipeline)
    return init_rows, forward_rows


def _release_pipeline(pipeline: Any) -> None:
    try:
        del pipeline
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _view_records_for_pipeline(pipeline: Any) -> List[Tuple[int, str, Any, Mapping[str, Any]]]:
    dataset = pipeline.datamanager.eval_dataset
    image_filenames = list(getattr(dataset, "image_filenames", []))
    rows = []
    for eval_index, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        filename = image_filenames[eval_index] if eval_index < len(image_filenames) else Path(f"eval_{eval_index}")
        rows.append((eval_index, Path(filename).stem, camera, batch))
    return rows


def final_summary(
    trajectory_rows: Sequence[Mapping[str, Any]],
    boundary_rows: Sequence[Mapping[str, Any]],
    region_rows: Sequence[Mapping[str, Any]],
    frequency_rows: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
    mse_rows: Sequence[Mapping[str, Any]],
    per_view_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    final = {row["run"]: row for row in trajectory_rows if int(row["nominal_step"]) == FINAL_STEP}
    boundary = {row["run"]: row for row in boundary_rows if int(row["nominal_step"]) == FINAL_STEP}
    m1, k1, aa = final["M1"], final["K1"], final["AA"]
    aa_psnr_gain = float(aa["psnr"]) - float(k1["psnr"])
    mse_recovery = (float(k1["mse"]) - float(aa["mse"])) / max(float(k1["mse"]) - float(m1["mse"]), EPS)
    tau_retention = (
        float(m1["tau_eval_object_support_pooled_channel_mean_p90"]) - float(aa["tau_eval_object_support_pooled_channel_mean_p90"])
    ) / max(float(m1["tau_eval_object_support_pooled_channel_mean_p90"]) - float(k1["tau_eval_object_support_pooled_channel_mean_p90"]), EPS)
    high = {row["run"]: row for row in region_rows if row.get("view_id") == "AGGREGATE" and row.get("mask") == "M1_J_gt_1"}
    low = {row["run"]: row for row in region_rows if row.get("view_id") == "AGGREGATE" and row.get("mask") == "M1_J_le_1"}
    high_recovery = (float(high["K1"]["mse"]) - float(high["AA"]["mse"])) / max(float(high["K1"]["mse"]) - float(high["M1"]["mse"]), EPS)
    low_damage = float(low["AA"]["mse"]) - float(low["K1"]["mse"])
    freq = {(row["run"], float(row["sigma_px"])): row for row in frequency_rows if row.get("view_id") == "AGGREGATE"}
    edge = {row["run"]: row for row in edge_rows if row.get("view_id") == "AGGREGATE"}
    hf_red3 = (float(freq[("K1", 3.0)]["high_energy"]) - float(freq[("AA", 3.0)]["high_energy"])) / max(float(freq[("K1", 3.0)]["high_energy"]), EPS)
    hf_red9 = (float(freq[("K1", 9.0)]["high_energy"]) - float(freq[("AA", 9.0)]["high_energy"])) / max(float(freq[("K1", 9.0)]["high_energy"]), EPS)
    edge_red = (float(edge["K1"]["edge_residual_energy"]) - float(edge["AA"]["edge_residual_energy"])) / max(float(edge["K1"]["edge_residual_energy"]), EPS)
    boundary_escape = bool(boundary["AA"].get("BOUNDARY_ESCAPE", False))
    rgb_safety = bool(
        float(aa["psnr"]) - float(m1["psnr"]) >= -0.15
        and float(aa["ssim"]) - float(m1["ssim"]) >= -0.0015
        and float(aa["lpips"]) - float(m1["lpips"]) <= 0.003
    )
    aa_view_rows = [row for row in per_view_rows if row["run"] == "AA"]
    improved = sum(1 for row in aa_view_rows if float(row.get("delta_psnr_vs_K1", 0.0)) > 0)
    degraded = sum(1 for row in aa_view_rows if float(row.get("delta_psnr_vs_K1", 0.0)) < 0)
    hf_or_edge_20 = hf_red3 >= 0.20 or hf_red9 >= 0.20 or edge_red >= 0.20
    strong = bool((aa_psnr_gain >= 0.30 or mse_recovery >= 0.50) and tau_retention >= 0.75 and not boundary_escape and (high_recovery >= 0.20 or hf_or_edge_20) and low_damage <= 1e-4)
    partial = bool(not strong and (aa_psnr_gain >= 0.10 or mse_recovery >= 0.20) and tau_retention >= 0.75 and not boundary_escape and (hf_red3 > 0 or hf_red9 > 0 or edge_red > 0))
    no_recovery = bool(abs(aa_psnr_gain) < 0.05 and not (hf_red3 > 0.05 or hf_red9 > 0.05 or edge_red > 0.05))
    rgb_only = bool((aa_psnr_gain >= 0.10 or mse_recovery >= 0.20) and not (hf_red3 > 0 or hf_red9 > 0 or edge_red > 0))
    decomp_regression = bool(tau_retention < 0.75 or boundary_escape)
    harmful = bool(aa_psnr_gain <= -0.10 or float(aa["ssim"]) < float(k1["ssim"]) - 0.0015 or float(aa["lpips"]) > float(k1["lpips"]) + 0.003)
    if strong:
        mechanism = "SUPPORTED"
    elif partial:
        mechanism = "PARTIALLY_SUPPORTED"
    else:
        mechanism = "NOT_SUPPORTED"
    k1_mse = next(row for row in mse_rows if row.get("view_id") == "AGGREGATE" and row.get("run") == "K1")
    aa_mse = next(row for row in mse_rows if row.get("view_id") == "AGGREGATE" and row.get("run") == "AA")
    return [
        {
            "scene": SCENE,
            "M1_PSNR": m1["psnr"],
            "K1_PSNR": k1["psnr"],
            "AA_PSNR": aa["psnr"],
            "AA_PSNR_GAIN": aa_psnr_gain,
            "M1_MSE": m1["mse"],
            "K1_MSE": k1["mse"],
            "AA_MSE": aa["mse"],
            "GLOBAL_MSE_GAP_RECOVERY": mse_recovery,
            "RGB_SAFETY": rgb_safety,
            "M1_tau_p90": m1["tau_eval_object_support_pooled_channel_mean_p90"],
            "K1_tau_p90": k1["tau_eval_object_support_pooled_channel_mean_p90"],
            "AA_tau_p90": aa["tau_eval_object_support_pooled_channel_mean_p90"],
            "TAU_BENEFIT_RETENTION": tau_retention,
            "M1_J_p99": m1["J_clear_eval_object_support_pooled_channel_mean_p99"],
            "K1_J_p99": k1["J_clear_eval_object_support_pooled_channel_mean_p99"],
            "AA_J_p99": aa["J_clear_eval_object_support_pooled_channel_mean_p99"],
            "AA_P_J_gt_1": aa["P_J_gt_1_object_support_pooled_channel_mean"],
            "AA_P_T_lt_0.1": aa["P_T_lt_0.1_object_support_pooled_channel_mean"],
            "BOUNDARY_ESCAPE": boundary_escape,
            "HIGH_J_MSE_GAP_RECOVERY": high_recovery,
            "LOW_J_DAMAGE": low_damage,
            "HF_RESIDUAL_REDUCTION_sigma3": hf_red3,
            "HF_RESIDUAL_REDUCTION_sigma9": hf_red9,
            "EDGE_RESIDUAL_REDUCTION": edge_red,
            "AA_vs_K1_views_improved": improved,
            "AA_vs_K1_views_degraded": degraded,
            "K1_C_direct": k1_mse["C_direct"],
            "K1_C_medium": k1_mse["C_medium"],
            "K1_C_cross": k1_mse["C_cross"],
            "AA_C_direct": aa_mse["C_direct"],
            "AA_C_medium": aa_mse["C_medium"],
            "AA_C_cross": aa_mse["C_cross"],
            "STRONG_AA_RECOVERY": strong,
            "PARTIAL_AA_RECOVERY": partial,
            "NO_AA_RECOVERY": no_recovery,
            "AA_RGB_ONLY_GAIN": rgb_only,
            "AA_DECOMPOSITION_REGRESSION": decomp_regression,
            "AA_HARMFUL": harmful,
            "MECHANISM_SUPPORT": mechanism,
        }
    ]


def write_visual_index(render_dir: Path, manifest: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Panama BND-AA Visual Compare Index", ""]
    for row in manifest:
        lines.append(f"- {row.get('output_type')}: `{row.get('file_path')}`")
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/mnt/new/home_old/ycy/water-splatting-refactor"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bnd_aa_panama_20260810"))
    parser.add_argument("--render-dir", type=Path, default=Path("renders/bnd_aa_panama_20260810"))
    parser.add_argument("--tile-width", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    render_dir = (repo / args.render_dir).resolve() if not args.render_dir.is_absolute() else args.render_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    init_rows, forward_rows = initialization_and_forward_audit(repo)
    trajectory_rows, missing_rows, boundary_rows, checkpoint_rows, opacity_rows = trajectory_audit(repo)
    (
        per_view_rows,
        region_rows,
        frequency_rows,
        edge_rows,
        recomposition_rows,
        mse_rows,
        final_checkpoint_rows,
        visual_manifest,
    ) = final_view_audit(repo, render_dir, args.tile_width)
    checkpoint_rows.extend(final_checkpoint_rows)
    summary_rows = final_summary(trajectory_rows, boundary_rows, region_rows, frequency_rows, edge_rows, mse_rows, per_view_rows)

    rgb_rows = [
        {
            "scene": row["scene"],
            "run": row["run"],
            "nominal_step": row["nominal_step"],
            "loaded_step": row["loaded_step"],
            "psnr": row.get("psnr"),
            "ssim": row.get("ssim"),
            "lpips": row.get("lpips"),
            "mse": row.get("mse"),
            "gaussian_count": row.get("gaussian_count"),
        }
        for row in trajectory_rows
    ]
    decomp_keys = {
        "scene",
        "run",
        "nominal_step",
        "loaded_step",
        "gaussian_count",
        "tau_eval_object_support_pooled_channel_mean_p90",
        "J_clear_eval_object_support_pooled_channel_mean_p99",
        "P_T_lt_0.3_object",
        "P_T_lt_0.2_object",
        "P_T_lt_0.1_object_support_pooled_channel_mean",
        "P_T_lt_0.05_object",
        "P_J_gt_1_object_support_pooled_channel_mean",
        "beta_D_raw_all_all_mean",
        "T_D_all_all_mean",
    }
    decomp_rows = [{key: value for key, value in row.items() if key in decomp_keys} for row in trajectory_rows]
    geometry_prefixes = ("depth_", "gaussian_scale_")
    geometry_keys = {"scene", "run", "nominal_step", "loaded_step", "gaussian_count", "gaussian_means_finite"}
    geometry_rows = [
        {
            key: value
            for key, value in row.items()
            if key in geometry_keys or any(key.startswith(prefix) for prefix in geometry_prefixes)
        }
        for row in trajectory_rows
    ]
    population_rows = [
        {"scene": row["scene"], "run": row["run"], "nominal_step": row["nominal_step"], "loaded_step": row["loaded_step"], "gaussian_count": row["gaussian_count"]}
        for row in trajectory_rows
        if row["run"] in ("K1", "AA")
    ]

    outputs: List[Tuple[str, Sequence[Mapping[str, Any]]]] = [
        ("aa_initialization_audit", init_rows),
        ("aa_forward_smoke_audit", forward_rows),
        ("aa_training_trajectory", trajectory_rows),
        ("aa_missing_checkpoints", missing_rows),
        ("aa_rgb_metrics", rgb_rows),
        ("aa_decomposition_metrics", decomp_rows),
        ("aa_geometry_metrics", geometry_rows),
        ("aa_boundary_metrics", boundary_rows),
        ("aa_region_metrics", region_rows),
        ("aa_frequency_edge_metrics", frequency_rows + edge_rows),
        ("aa_frequency_metrics", frequency_rows),
        ("aa_edge_metrics", edge_rows),
        ("aa_gaussian_population", population_rows),
        ("aa_opacity_compensation", opacity_rows),
        ("aa_per_view_metrics", per_view_rows),
        ("aa_recomposition_metrics", recomposition_rows),
        ("aa_mse_attribution", mse_rows),
        ("aa_checkpoint_audit", checkpoint_rows),
        ("aa_final_summary", summary_rows),
    ]
    for stem, rows in outputs:
        _write_json(output_dir / f"{stem}.json", rows)
        _write_csv(output_dir / f"{stem}.csv", rows)
    _write_json(render_dir / "manifest.json", visual_manifest)
    _write_csv(render_dir / "manifest.csv", visual_manifest)
    write_visual_index(render_dir, visual_manifest)
    _write_json(
        output_dir / "manifest.json",
        {
            "scene": SCENE,
            "repo": str(repo),
            "branch": _git(repo, "branch", "--show-current"),
            "head": _git(repo, "rev-parse", "HEAD"),
            "runs": {key: spec.__dict__ for key, spec in RUNS.items()},
            "summary": summary_rows[0],
            "visual_manifest": str(render_dir / "manifest.json"),
            "visual_index": str(render_dir / "VISUAL_COMPARE_INDEX.md"),
        },
    )
    print(json.dumps({"summary": summary_rows, "output_dir": str(output_dir), "render_dir": str(render_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
