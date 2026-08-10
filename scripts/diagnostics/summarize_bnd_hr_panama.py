#!/usr/bin/env python
"""Summarize the Panama bounded-headroom SH3 appearance experiment."""

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
from PIL import Image, ImageDraw
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup
from water_splatting.fields import (
    compute_bounded_gaussian_colors,
    compute_bounded_headroom_gaussian_colors,
    compute_gaussian_colors,
)


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
    "HR": RunSpec(
        "HR",
        "outputs/bnd_hr_panama_20260810/panama_bnd_hr_seed42_step0_to_15000/water-splatting/20260810_bnd_hr/config.yml",
        "bounded_headroom_sh3",
        "classic",
        "bounded_headroom_candidate",
        False,
    ),
    "AA": RunSpec(
        "AA",
        "outputs/bnd_aa_panama_20260810/panama_bnd_aa_seed42_step0_to_15000/water-splatting/20260810_bnd_aa/config.yml",
        "bounded_sh3",
        "antialiased",
        "secondary_aa_reference",
        True,
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


def _stats(values: Tensor, prefix: str) -> Dict[str, Any]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    names = ("count", "mean", "p01", "p05", "p10", "p50", "p75", "p90", "p95", "p99", "min", "max")
    if flat.numel() == 0:
        return {f"{prefix}{name}": float("nan") for name in names}
    return {
        f"{prefix}count": int(flat.numel()),
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}p01": _safe_quantile(flat, 0.01),
        f"{prefix}p05": _safe_quantile(flat, 0.05),
        f"{prefix}p10": _safe_quantile(flat, 0.10),
        f"{prefix}p50": _safe_quantile(flat, 0.50),
        f"{prefix}p75": _safe_quantile(flat, 0.75),
        f"{prefix}p90": _safe_quantile(flat, 0.90),
        f"{prefix}p95": _safe_quantile(flat, 0.95),
        f"{prefix}p99": _safe_quantile(flat, 0.99),
        f"{prefix}min": float(flat.min().item()),
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


def _channel_stats(values: Tensor, prefix: str) -> Dict[str, Any]:
    values = values.detach().float()
    out: Dict[str, Any] = {}
    if values.ndim > 0 and values.shape[-1] == 3:
        for index, channel in enumerate(CHANNELS):
            out.update(_stats(values[..., index], f"{prefix}_{channel}_"))
        out.update(_stats(values.reshape(-1), f"{prefix}_all_"))
    else:
        out.update(_stats(values.reshape(-1), f"{prefix}_"))
    return out


def _threshold_rows(values: Tensor, prefix: str, thresholds: Sequence[float], op: str) -> Dict[str, Any]:
    values = values.detach().float()
    out: Dict[str, Any] = {}
    if values.ndim > 0 and values.shape[-1] == 3:
        for index, channel in enumerate(CHANNELS):
            for threshold in thresholds:
                out[f"{prefix}_{channel}_{op}_{threshold:g}"] = _threshold_fraction(values[..., index], threshold, op)
        for threshold in thresholds:
            out[f"{prefix}_all_{op}_{threshold:g}"] = _threshold_fraction(values.reshape(-1), threshold, op)
    else:
        for threshold in thresholds:
            out[f"{prefix}_{op}_{threshold:g}"] = _threshold_fraction(values, threshold, op)
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
    row["J_clear_eval_object_support_pooled_channel_mean_p99"] = _pooled_channel_stat(items, "clear_object_fullsh_raw", "p99", "object")
    row["P_T_lt_0.1_object_support_pooled_channel_mean"] = _threshold_pooled_channel(items, "transmission", 0.1, "lt", "object")
    row["P_J_gt_1_object_support_pooled_channel_mean"] = _threshold_pooled_channel(items, "clear_object_fullsh_raw", 1.0, "gt", "object")
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


def _model_geometry_stats(model: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    if hasattr(model, "scales"):
        row.update(_stats(torch.exp(model.scales.detach().float()).reshape(-1), "gaussian_scale_"))
    if hasattr(model, "means"):
        row["gaussian_means_finite"] = bool(torch.isfinite(model.means.detach()).all().item())
    return row


def _visible_values(items: Sequence[Mapping[str, Any]], key: str) -> Tensor:
    vals = []
    for item in items:
        out = item["outputs"]
        tensor = out.get(key)
        visible = out.get("gaussian_visible_mask")
        if not isinstance(tensor, Tensor):
            continue
        if isinstance(visible, Tensor) and visible.numel() == tensor.shape[0]:
            tensor = tensor[visible.bool()]
        vals.append(tensor.reshape(-1, tensor.shape[-1] if tensor.ndim > 1 else 1))
    return torch.cat(vals, dim=0) if vals else torch.empty(0, 3)


def _representation_rows(run: str, items: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    c0 = _visible_values(items, "gaussian_view_dc_rgb")
    full = _visible_values(items, "gaussian_view_rgb")
    delta = _visible_values(items, "gaussian_color_residual")
    raw_r = _visible_values(items, "gaussian_sh_residual")
    if delta.numel() == 0 and c0.numel() and full.numel():
        delta = full - c0
    u_pos = _visible_values(items, "gaussian_headroom_u_pos")
    u_neg = _visible_values(items, "gaussian_headroom_u_neg")
    if u_pos.numel() == 0 and c0.numel() and delta.numel():
        u_pos = torch.where(delta > 0, delta / (1.0 - c0).clamp_min(EPS), torch.zeros_like(delta))
        u_neg = torch.where(delta < 0, (-delta) / c0.clamp_min(EPS), torch.zeros_like(delta))

    base_row: Dict[str, Any] = {"scene": SCENE, "run": run}
    full_row: Dict[str, Any] = {"scene": SCENE, "run": run}
    util_row: Dict[str, Any] = {"scene": SCENE, "run": run}
    cap_row: Dict[str, Any] = {"scene": SCENE, "run": run}
    balance_row: Dict[str, Any] = {"scene": SCENE, "run": run}
    if c0.numel():
        base_row.update(_channel_stats(c0, "c0"))
        base_row.update(_threshold_rows(c0, "c0", (0.95, 0.99), "gt"))
        base_row.update(_threshold_rows(c0, "c0", (0.05, 0.01), "lt"))
    if full.numel():
        full_row.update(_channel_stats(full, "c"))
        full_row.update(_threshold_rows(full, "c", (0.95, 0.99), "gt"))
        full_row.update(_threshold_rows(full, "c", (0.05, 0.01), "lt"))
    if u_pos.numel():
        pos_vals = u_pos[u_pos > 0]
        neg_vals = u_neg[u_neg > 0]
        u_pos_positive = torch.where(u_pos > 0, u_pos, torch.full_like(u_pos, float("nan")))
        u_neg_negative = torch.where(u_neg > 0, u_neg, torch.full_like(u_neg, float("nan")))
        util_row.update(_channel_stats(torch.where(u_pos > 0, u_pos, torch.full_like(u_pos, float("nan"))), "u_pos"))
        util_row.update(_channel_stats(torch.where(u_neg > 0, u_neg, torch.full_like(u_neg, float("nan"))), "u_neg"))
        util_row.update(_threshold_rows(u_pos_positive, "u_pos_positive_only", (0.50, 0.75, 0.90, 0.99), "gt"))
        util_row.update(_threshold_rows(u_neg_negative, "u_neg_negative_only", (0.50, 0.75, 0.90, 0.99), "gt"))
        util_row.update(_threshold_rows(pos_vals, "u_pos_all_positive_only", (0.50, 0.75, 0.90, 0.99), "gt"))
        util_row.update(_threshold_rows(neg_vals, "u_neg_all_negative_only", (0.50, 0.75, 0.90, 0.99), "gt"))
    if delta.numel():
        cap = torch.linalg.norm(delta, dim=-1)
        cap_row.update(_stats(cap, "R_SH_COLOR_"))
        pos = delta > 0
        neg = delta < 0
        balance_row.update(_threshold_rows(delta, "Delta_c", (0.0,), "gt"))
        balance_row.update(_threshold_rows(delta, "Delta_c", (0.0,), "lt"))
        pos_energy = delta[pos].square().sum()
        neg_energy = delta[neg].square().sum()
        total_energy = pos_energy + neg_energy
        balance_row["positive_energy_fraction_all"] = float(pos_energy.item() / max(float(total_energy.item()), EPS))
        balance_row["negative_energy_fraction_all"] = float(neg_energy.item() / max(float(total_energy.item()), EPS))
        for index, channel in enumerate(CHANNELS):
            d = delta[:, index]
            p = d > 0
            n = d < 0
            ep = d[p].square().sum()
            en = d[n].square().sum()
            et = ep + en
            balance_row[f"{channel}_positive_energy_fraction"] = float(ep.item() / max(float(et.item()), EPS))
            balance_row[f"{channel}_negative_energy_fraction"] = float(en.item() / max(float(et.item()), EPS))
        luma = (delta * LUMA_WEIGHTS).sum(dim=-1)
        luma_projection = luma[:, None] * LUMA_WEIGHTS[None, :] / max(float((LUMA_WEIGHTS * LUMA_WEIGHTS).sum().item()), EPS)
        balance_row["luma_positive_fraction"] = _threshold_fraction(luma, 0.0, "gt")
        balance_row["luma_negative_fraction"] = _threshold_fraction(luma, 0.0, "lt")
        balance_row.update(_stats(torch.linalg.norm(delta - luma_projection, dim=-1), "chroma_residual_"))
    if raw_r.numel():
        util_row.update(_channel_stats(raw_r.abs(), "abs_raw_r"))
        util_row.update(_threshold_rows(raw_r, "raw_r_abs", (5.0,), "abs_gt"))

    boundary_pressure = bool(
        base_row.get("c0_all_gt_0.99", 0.0) > 0.05
        or full_row.get("c_all_gt_0.99", 0.0) > 0.05
        or util_row.get("u_pos_all_positive_only_gt_0.99", 0.0) > 0.05
        or util_row.get("u_neg_all_negative_only_gt_0.99", 0.0) > 0.05
    )
    residual_saturation = bool(
        util_row.get("u_pos_all_positive_only_gt_0.99", 0.0) > 0.05
        or util_row.get("u_neg_all_negative_only_gt_0.99", 0.0) > 0.05
    )
    util_row["RESIDUAL_SATURATION"] = residual_saturation
    util_row["BOUNDARY_PRESSURE"] = boundary_pressure
    return base_row, full_row, util_row, cap_row, balance_row


def _cache_outputs(repo: Path, run: str, nominal_step: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    loaded: Optional[LoadedRun] = None
    try:
        loaded = _load_run(repo, run, nominal_step)
        model = loaded.model
        items: List[Dict[str, Any]] = []
        for eval_index, view_id, camera, batch in _view_records(loaded):
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera)
                gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
                metrics = _metric_images(model, outputs["pred_image"], gt)
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
                "gaussian_view_dc_rgb",
                "gaussian_sh_residual",
                "gaussian_color_residual",
                "gaussian_headroom_u_pos",
                "gaussian_headroom_u_neg",
                "gaussian_visible_mask",
            )
            tensors = {key: _safe_cpu(outputs[key]) for key in keep if key in outputs and isinstance(outputs[key], Tensor)}
            base_render = None
            if run == "HR":
                with torch.no_grad():
                    base_render = _render_with_color_mode(model, camera, "dc_only")
                for key, value in base_render.items():
                    if isinstance(value, Tensor):
                        tensors[f"base_{key}"] = _safe_cpu(value)
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
        return items, meta
    finally:
        _release_loaded(loaded)


def _render_with_color_mode(model: Any, camera: Cameras, mode: str) -> Dict[str, Tensor]:
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
        cx = camera.cx.item()
        cy = camera.cy.item()
        H, W = int(camera.height.item()), int(camera.width.item())
        medium = model._predict_medium(camera=camera, rotation_world_from_camera=R, height=H, width=W, cx=cx, cy=cy)
        if model.crop_box is not None and not model.training:
            crop_ids = model.crop_box.within(model.means).squeeze()
        else:
            crop_ids = None
        if crop_ids is not None and crop_ids.sum() != 0:
            opacities_crop = model.opacities[crop_ids]
            means_crop = model.means[crop_ids]
            features_dc_crop = model.features_dc[crop_ids]
            features_rest_crop = model.features_rest[crop_ids]
            scales_crop = model.scales[crop_ids]
            quats_crop = model.quats[crop_ids]
        else:
            opacities_crop = model.opacities
            means_crop = model.means
            features_dc_crop = model.features_dc
            features_rest_crop = model.features_rest
            scales_crop = model.scales
            quats_crop = model.quats
        xys, depths, radii, conics, comp, num_tiles_hit, _ = model.underwater_rasterizer.project(
            means=means_crop,
            scales=scales_crop,
            quats=quats_crop,
            viewmat=viewmat,
            fx=camera.fx.item(),
            fy=camera.fy.item(),
            cx=cx,
            cy=cy,
            height=H,
            width=W,
            clip_thresh=model.config.clip_thresh,
        )
    finally:
        camera.rescale_output_resolution(camera_downscale)

    active_sh_degree = min(model.step // model.config.sh_degree_interval, model.config.sh_degree)
    parameterization = getattr(model.config, "intrinsic_color_parameterization", "legacy")
    if parameterization == "legacy":
        full = compute_gaussian_colors(
            means=means_crop,
            features_dc=features_dc_crop,
            features_rest=features_rest_crop,
            camera_position=camera.camera_to_worlds[..., :3, 3],
            sh_degree=model.config.sh_degree,
            active_sh_degree=active_sh_degree,
        )
        rgbs = full
    elif parameterization == "bounded_sh3":
        color = compute_bounded_gaussian_colors(
            means=means_crop,
            features_dc=features_dc_crop,
            features_rest=features_rest_crop,
            camera_position=camera.camera_to_worlds[..., :3, 3],
            sh_degree=model.config.sh_degree,
            active_sh_degree=active_sh_degree,
        )
        rgbs = color.dc_rgb if mode == "dc_only" else color.rgb
    elif parameterization == "bounded_headroom_sh3":
        color = compute_bounded_headroom_gaussian_colors(
            means=means_crop,
            features_dc=features_dc_crop,
            features_rest=features_rest_crop,
            camera_position=camera.camera_to_worlds[..., :3, 3],
            sh_degree=model.config.sh_degree,
            active_sh_degree=active_sh_degree,
        )
        rgbs = color.dc_rgb if mode == "dc_only" else color.rgb
    else:
        raise ValueError(parameterization)

    if model.config.rasterize_mode == "antialiased":
        opacities = torch.sigmoid(opacities_crop) * comp[:, None]
    else:
        opacities = torch.sigmoid(opacities_crop)
    render = model.underwater_rasterizer.rasterize(
        xys=xys,
        xys_grad_abs=torch.zeros_like(xys),
        depths=depths,
        radii=radii,
        conics=conics,
        num_tiles_hit=num_tiles_hit,
        colors=rgbs,
        opacities=opacities,
        medium_rgb=medium.rgb,
        medium_bs=medium.bs,
        medium_attn=medium.attn,
        height=H,
        width=W,
        background=medium.rgb,
        step=model.step,
    )
    rgb = render.rgb
    rgb_medium = render.rgb_medium
    b_inf = medium.b_inf
    if model._effective_b_inf_mode() == "tied":
        if b_inf is None:
            raise RuntimeError("b_inf_mode='tied' requires b_inf")
        tail_weight = render.final_transmittance * torch.exp(-medium.bs * render.last_depth)
        rgb_medium = rgb_medium - tail_weight * medium.rgb + tail_weight * b_inf
        rgb = render.rgb_object + rgb_medium
    tau_d = medium.attn * render.depth
    return {
        "pred_image": rgb,
        "direct_object_signal": render.rgb_object,
        "clear_object_fullsh_raw": render.j_raw,
        "rgb_medium": rgb_medium,
        "depth": render.depth,
        "accumulation": render.accumulation,
        "transmission": torch.exp(-tau_d.clamp_min(0.0)).clamp(0.0, 1.0),
        "tau_D": tau_d,
    }


def trajectory_audit(repo: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    trajectory_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    representation_rows: List[Dict[str, Any]] = []
    for run in ("K1", "HR"):
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
            items, meta = _cache_outputs(repo, run, step)
            checkpoint_rows.append(meta)
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
            if run == "HR":
                base, full, util, cap, balance = _representation_rows(run, items)
                representation_rows.append({"nominal_step": step, "loaded_step": meta["loaded_step"], **base, **full, **util, **cap, **balance})
    for run in ("M1", "AA"):
        items, meta = _cache_outputs(repo, run, FINAL_STEP)
        checkpoint_rows.append(meta)
        row = {
            "scene": SCENE,
            "run": run,
            "nominal_step": FINAL_STEP,
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
    return trajectory_rows, missing_rows, checkpoint_rows, representation_rows


def _gaussian_kernel1d(sigma: float) -> Tensor:
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _gaussian_blur(image: Tensor, sigma: float) -> Tensor:
    img = image.detach().float().permute(2, 0, 1)[None, ...]
    channels = img.shape[1]
    k = _gaussian_kernel1d(sigma)
    pad = k.numel() // 2
    kh = k.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
    kv = k.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
    out = F.conv2d(F.pad(img, (pad, pad, 0, 0), mode="reflect"), kh, groups=channels)
    out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode="reflect"), kv, groups=channels)
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
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    cached: Dict[str, List[Dict[str, Any]]] = {}
    checkpoint_rows: List[Dict[str, Any]] = []
    for run in ("M1", "K1", "HR", "AA"):
        items, meta = _cache_outputs(repo, run, FINAL_STEP)
        cached[run] = items
        checkpoint_rows.append(meta)
    view_ids = [item["view_id"] for item in cached["M1"]]
    for run in ("K1", "HR", "AA"):
        if [item["view_id"] for item in cached[run]] != view_ids:
            raise RuntimeError(f"view mismatch for {run}")
    by_run_view = {run: {item["view_id"]: item for item in items} for run, items in cached.items()}
    per_view_rows: List[Dict[str, Any]] = []
    region_rows: List[Dict[str, Any]] = []
    frequency_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    recomposition_rows: List[Dict[str, Any]] = []
    mse_rows: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []

    luma_values = torch.cat([_luma(item["gt"]).reshape(-1) for item in cached["M1"]], dim=0)
    bright_q5_threshold = _safe_quantile(luma_values, 0.80)
    for view_id in view_ids:
        gt = by_run_view["M1"][view_id]["gt"]
        for run in ("M1", "K1", "HR", "AA"):
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
            if run == "HR":
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
            for run in ("M1", "K1", "HR"):
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
        for run in ("K1", "HR"):
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
    mse_rows.extend(_mse_attribution_rows(by_run_view, view_ids))
    _write_visuals(render_dir, by_run_view, view_ids, tile_width, manifest, bright_q5_threshold)
    base, full, util, cap, balance = _representation_rows("HR", cached["HR"])
    return (
        per_view_rows,
        region_rows,
        frequency_rows,
        edge_rows,
        recomposition_rows,
        mse_rows,
        checkpoint_rows,
        manifest,
        [base, full, util, cap, balance],
        cached["HR"],
    )


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


def _aggregate_region_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for mask_name in sorted({row["mask"] for row in rows}):
        for run in ("M1", "K1", "HR"):
            out.append(_aggregate_numeric([row for row in rows if row["mask"] == mask_name and row["run"] == run], {"scene": SCENE, "view_id": "AGGREGATE", "mask": mask_name, "run": run}))
    return out


def _aggregate_frequency_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for run in ("K1", "HR"):
        for sigma in (3.0, 9.0):
            out.append(_aggregate_numeric([row for row in rows if row["run"] == run and float(row["sigma_px"]) == sigma], {"scene": SCENE, "view_id": "AGGREGATE", "run": run, "sigma_px": sigma}))
    return out


def _aggregate_edge_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [_aggregate_numeric([row for row in rows if row["run"] == run], {"scene": SCENE, "view_id": "AGGREGATE", "run": run}) for run in ("K1", "HR")]


def _cancellation_metrics(delta_d: Tensor, delta_b: Tensor, mask: Tensor) -> Dict[str, Any]:
    dd = delta_d[mask].reshape(-1, 3).float()
    db = delta_b[mask].reshape(-1, 3).float()
    if dd.numel() == 0:
        return {}
    sum_abs = (dd + db).abs().sum(dim=-1)
    raw_abs = dd.abs().sum(dim=-1) + db.abs().sum(dim=-1)
    dot = (dd * db).sum(dim=-1)
    denom = float(torch.sqrt((dd * dd).sum() * (db * db).sum()).item())
    cos = dot / (torch.linalg.norm(dd, dim=-1) * torch.linalg.norm(db, dim=-1) + EPS)
    return {
        "flattened_cosine_similarity": float(dot.sum().item() / max(denom, EPS)),
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
        for run in ("K1", "HR"):
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
    for run in ("K1", "HR"):
        rows.append(_aggregate_numeric([row for row in rows if row["run"] == run], {"scene": SCENE, "view_id": "AGGREGATE", "run": run, "reference": "M1"}))
    return rows


def _mse_attribution_rows(by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], view_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for run in ("K1", "HR"):
        for view_id in view_ids:
            m1 = by_run_view["M1"][view_id]
            item = by_run_view[run][view_id]
            gt = m1["gt"].float()
            d_d = item["outputs"]["direct_object_signal"] - m1["outputs"]["direct_object_signal"]
            d_b = item["outputs"]["rgb_medium"] - m1["outputs"]["rgb_medium"]
            e0 = m1["outputs"]["pred_image"] - gt
            e1 = item["outputs"]["pred_image"] - gt
            c_direct = float((2.0 * e0 * d_d + d_d.square()).mean().item())
            c_medium = float((2.0 * e0 * d_b + d_b.square()).mean().item())
            c_cross = float((2.0 * d_d * d_b).mean().item())
            delta_mse = float((e1.square().mean() - e0.square().mean()).item())
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


def _overlay_mask(base: Image.Image, mask: Tensor, color: Tuple[int, int, int] = (255, 40, 20)) -> Image.Image:
    image = base.convert("RGB")
    overlay = Image.new("RGB", image.size, color)
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
    manifest.append({"file_path": str(path), "scene": SCENE, "runs": "M1;K1;HR", "step": FINAL_STEP, "output_type": output_type, "view_ids": ";".join(view_ids), "width": sheet.width, "height": sheet.height})


def _write_visuals(render_dir: Path, by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], view_ids: Sequence[str], tile_width: int, manifest: List[Dict[str, Any]], bright_q5_threshold: float) -> None:
    residual_scale = 0.1
    delta_scale = 0.1
    util_scale = 1.0
    for view_id in view_ids:
        gt = by_run_view["M1"][view_id]["gt"]
        for run in ("M1", "K1", "HR"):
            residual_scale = max(residual_scale, float(_rgb_l2(by_run_view[run][view_id]["outputs"]["pred_image"] - gt).max().item()))
        hr = by_run_view["HR"][view_id]["outputs"]
        delta_scale = max(delta_scale, float((hr["clear_object_fullsh_raw"] - hr["base_clear_object_fullsh_raw"]).abs().max().item()))

    rows_underwater: List[List[Tuple[str, Image.Image]]] = []
    rows_clear: List[List[Tuple[str, Image.Image]]] = []
    rows_base_full: List[List[Tuple[str, Image.Image]]] = []
    rows_sh_residual: List[List[Tuple[str, Image.Image]]] = []
    rows_util: List[List[Tuple[str, Image.Image]]] = []
    rows_highj: List[List[Tuple[str, Image.Image]]] = []
    rows_bright: List[List[Tuple[str, Image.Image]]] = []
    rows_direct: List[List[Tuple[str, Image.Image]]] = []
    rows_medium: List[List[Tuple[str, Image.Image]]] = []
    rows_boundary: List[List[Tuple[str, Image.Image]]] = []
    for view_id in view_ids:
        gt = by_run_view["M1"][view_id]["gt"]
        rows_underwater.append([(f"{view_id} GT", _rgb_to_uint8(gt))] + [(run, _rgb_to_uint8(by_run_view[run][view_id]["outputs"]["pred_image"])) for run in ("M1", "K1", "HR")])
        rows_clear.append([(f"{view_id} {run}", _rgb_to_uint8(by_run_view[run][view_id]["outputs"]["clear_object_fullsh_raw"])) for run in ("M1", "K1", "HR")])
        hr = by_run_view["HR"][view_id]["outputs"]
        base_clear = hr["base_clear_object_fullsh_raw"]
        full_clear = hr["clear_object_fullsh_raw"]
        delta = full_clear - base_clear
        rows_base_full.append(
            [
                (f"{view_id} HR base", _rgb_to_uint8(base_clear)),
                ("HR full", _rgb_to_uint8(full_clear)),
                ("abs full-base", _rgb_to_uint8(delta.abs() / max(delta_scale, EPS))),
            ]
        )
        rows_sh_residual.append(
            [
                (f"{view_id} signed luma", _signed_to_rgb(_luma(delta), delta_scale)),
                ("RGB magnitude", _gray_to_uint8(torch.linalg.norm(delta, dim=-1), delta_scale)),
            ]
        )
        u_signed = torch.where(delta >= 0, delta / (1.0 - base_clear).clamp_min(EPS), delta / base_clear.clamp_min(EPS))
        rows_util.append(
            [
                (f"{view_id} u signed luma", _signed_to_rgb(_luma(u_signed), util_scale)),
                ("u magnitude", _gray_to_uint8(torch.linalg.norm(u_signed, dim=-1), util_scale)),
            ]
        )
        m1_resid = _rgb_l2(by_run_view["M1"][view_id]["outputs"]["pred_image"] - gt)
        k1_resid = _rgb_l2(by_run_view["K1"][view_id]["outputs"]["pred_image"] - gt)
        hr_resid = _rgb_l2(hr["pred_image"] - gt)
        high_j = _object_support(by_run_view["M1"][view_id]) & (by_run_view["M1"][view_id]["outputs"]["clear_object_fullsh_raw"].amax(dim=-1) > 1.0)
        bright = _luma(gt) > bright_q5_threshold
        rows_highj.append(
            [
                (f"{view_id} M1 J>1", _mask_to_uint8(high_j)),
                ("K1 residual", _overlay_mask(_gray_to_uint8(k1_resid, residual_scale), high_j)),
                ("HR residual", _overlay_mask(_gray_to_uint8(hr_resid, residual_scale), high_j)),
            ]
        )
        rows_bright.append(
            [
                (f"{view_id} brightness Q5", _mask_to_uint8(bright)),
                ("K1 residual", _overlay_mask(_gray_to_uint8(k1_resid, residual_scale), bright, (255, 180, 20))),
                ("HR residual", _overlay_mask(_gray_to_uint8(hr_resid, residual_scale), bright, (255, 180, 20))),
            ]
        )
        d_delta = hr["direct_object_signal"] - by_run_view["K1"][view_id]["outputs"]["direct_object_signal"]
        b_delta = hr["rgb_medium"] - by_run_view["K1"][view_id]["outputs"]["rgb_medium"]
        rows_direct.append(
            [
                (f"{view_id} K1 direct", _rgb_to_uint8(by_run_view["K1"][view_id]["outputs"]["direct_object_signal"])),
                ("HR direct", _rgb_to_uint8(hr["direct_object_signal"])),
                ("abs delta", _rgb_to_uint8(d_delta.abs() / max(float(d_delta.abs().max().item()), EPS))),
            ]
        )
        rows_medium.append(
            [
                (f"{view_id} K1 medium", _rgb_to_uint8(by_run_view["K1"][view_id]["outputs"]["rgb_medium"])),
                ("HR medium", _rgb_to_uint8(hr["rgb_medium"])),
                ("abs delta", _rgb_to_uint8(b_delta.abs() / max(float(b_delta.abs().max().item()), EPS))),
            ]
        )
        boundary = (base_clear.amax(dim=-1) > 0.99) | (full_clear.amax(dim=-1) > 0.99) | (u_signed.abs().amax(dim=-1) > 0.99)
        rows_boundary.append(
            [
                (f"{view_id} c0>0.99", _mask_to_uint8(base_clear.amax(dim=-1) > 0.99)),
                ("cHR>0.99", _mask_to_uint8(full_clear.amax(dim=-1) > 0.99)),
                ("|u|>0.99", _mask_to_uint8(u_signed.abs().amax(dim=-1) > 0.99)),
                ("union", _mask_to_uint8(boundary)),
            ]
        )
    for filename, rows, output_type in (
        ("contact_sheet_underwater_m1_k1_hr.png", rows_underwater, "underwater"),
        ("contact_sheet_clear_raw_m1_k1_hr.png", rows_clear, "clear_object_fullsh_raw_display_clamp01"),
        ("contact_sheet_hr_base_full_delta.png", rows_base_full, "hr_base_full_delta"),
        ("contact_sheet_hr_signed_residual.png", rows_sh_residual, "hr_signed_color_residual"),
        ("contact_sheet_hr_headroom_utilization.png", rows_util, "hr_headroom_utilization"),
        ("contact_sheet_high_j_region_k1_hr.png", rows_highj, "fixed_m1_high_j_mask_overlay"),
        ("contact_sheet_brightness_q5_k1_hr.png", rows_bright, "fixed_brightness_q5_overlay"),
        ("contact_sheet_direct_k1_hr_delta.png", rows_direct, "direct_object_signal_delta"),
        ("contact_sheet_medium_k1_hr_delta.png", rows_medium, "rgb_medium_delta"),
        ("contact_sheet_boundary_pressure_hr.png", rows_boundary, "hr_boundary_pressure_masks"),
    ):
        _save_sheet(render_dir / filename, rows, tile_width, manifest, output_type, view_ids)


def final_summary(
    trajectory_rows: Sequence[Mapping[str, Any]],
    representation_rows: Sequence[Mapping[str, Any]],
    region_rows: Sequence[Mapping[str, Any]],
    frequency_rows: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
    mse_rows: Sequence[Mapping[str, Any]],
    per_view_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    final = {row["run"]: row for row in trajectory_rows if int(row["nominal_step"]) == FINAL_STEP}
    m1, k1, hr = final["M1"], final["K1"], final["HR"]
    aa = final.get("AA")
    hr_psnr_gain = float(hr["psnr"]) - float(k1["psnr"])
    mse_recovery = (float(k1["mse"]) - float(hr["mse"])) / max(float(k1["mse"]) - float(m1["mse"]), EPS)
    tau_retention = (
        float(m1["tau_eval_object_support_pooled_channel_mean_p90"]) - float(hr["tau_eval_object_support_pooled_channel_mean_p90"])
    ) / max(float(m1["tau_eval_object_support_pooled_channel_mean_p90"]) - float(k1["tau_eval_object_support_pooled_channel_mean_p90"]), EPS)
    reps = {row["run"]: row for row in representation_rows if row.get("run") == "HR"}
    # representation_rows contains five HR rows; merge by key.
    hr_rep: Dict[str, Any] = {}
    for row in representation_rows:
        if row.get("run") == "HR":
            hr_rep.update(row)
    boundary_pressure = bool(hr_rep.get("BOUNDARY_PRESSURE", False))
    high = {row["run"]: row for row in region_rows if row.get("view_id") == "AGGREGATE" and row.get("mask") == "M1_J_gt_1"}
    low = {row["run"]: row for row in region_rows if row.get("view_id") == "AGGREGATE" and row.get("mask") == "M1_J_le_1"}
    high_recovery = (float(high["K1"]["mse"]) - float(high["HR"]["mse"])) / max(float(high["K1"]["mse"]) - float(high["M1"]["mse"]), EPS)
    low_damage = float(low["HR"]["mse"]) - float(low["K1"]["mse"])
    freq = {(row["run"], float(row["sigma_px"])): row for row in frequency_rows if row.get("view_id") == "AGGREGATE"}
    edge = {row["run"]: row for row in edge_rows if row.get("view_id") == "AGGREGATE"}
    hf_red3 = (float(freq[("K1", 3.0)]["high_energy"]) - float(freq[("HR", 3.0)]["high_energy"])) / max(float(freq[("K1", 3.0)]["high_energy"]), EPS)
    hf_red9 = (float(freq[("K1", 9.0)]["high_energy"]) - float(freq[("HR", 9.0)]["high_energy"])) / max(float(freq[("K1", 9.0)]["high_energy"]), EPS)
    edge_red = (float(edge["K1"]["edge_residual_energy"]) - float(edge["HR"]["edge_residual_energy"])) / max(float(edge["K1"]["edge_residual_energy"]), EPS)
    rgb_safety = bool(
        float(hr["psnr"]) - float(m1["psnr"]) >= -0.15
        and float(hr["ssim"]) - float(m1["ssim"]) >= -0.0015
        and float(hr["lpips"]) - float(m1["lpips"]) <= 0.003
    )
    pareto_closed = bool(
        rgb_safety
        and tau_retention >= 0.75
        and float(hr["P_J_gt_1_object_support_pooled_channel_mean"]) == 0.0
        and not boundary_pressure
    )
    hr_view_rows = [row for row in per_view_rows if row["run"] == "HR"]
    improved = sum(1 for row in hr_view_rows if float(row.get("delta_psnr_vs_K1", 0.0)) > 0)
    degraded = sum(1 for row in hr_view_rows if float(row.get("delta_psnr_vs_K1", 0.0)) < 0)
    deltas = [float(row.get("delta_psnr_vs_K1", 0.0)) for row in hr_view_rows]
    strong = bool(
        (hr_psnr_gain >= 0.30 or mse_recovery >= 0.50)
        and tau_retention >= 0.75
        and not boundary_pressure
        and high_recovery >= 0.30
        and low_damage <= 0.00001
        and float(hr["ssim"]) - float(k1["ssim"]) >= -0.0005
        and float(hr["lpips"]) - float(k1["lpips"]) <= 0.001
    )
    partial = bool(
        not strong
        and (hr_psnr_gain >= 0.10 or mse_recovery >= 0.20)
        and tau_retention >= 0.75
        and not boundary_pressure
        and high_recovery > 0.0
        and low_damage <= 0.00001
    )
    no_recovery = bool(abs(hr_psnr_gain) < 0.05 and high_recovery < 0.10)
    harmful = bool(hr_psnr_gain <= -0.10 or float(hr["lpips"]) - float(k1["lpips"]) > 0.003 or tau_retention < 0.75)
    boundary_failure = bool(boundary_pressure)
    if strong:
        hypothesis = "SUPPORTED"
    elif partial and (hr_rep.get("R_SH_COLOR_p90", 0.0) > 0.0):
        hypothesis = "PARTIALLY_SUPPORTED"
    else:
        hypothesis = "NOT_SUPPORTED"
    k1_mse = next(row for row in mse_rows if row.get("view_id") == "AGGREGATE" and row.get("run") == "K1")
    hr_mse = next(row for row in mse_rows if row.get("view_id") == "AGGREGATE" and row.get("run") == "HR")
    return [
        {
            "scene": SCENE,
            "M1_PSNR": m1["psnr"],
            "K1_PSNR": k1["psnr"],
            "HR_PSNR": hr["psnr"],
            "AA_PSNR_secondary": aa.get("psnr") if aa else float("nan"),
            "HR_PSNR_GAIN": hr_psnr_gain,
            "M1_MSE": m1["mse"],
            "K1_MSE": k1["mse"],
            "HR_MSE": hr["mse"],
            "GLOBAL_MSE_GAP_RECOVERY": mse_recovery,
            "RGB_SAFETY": rgb_safety,
            "PANAMA_PARETO_CLOSED": pareto_closed,
            "M1_tau_p90": m1["tau_eval_object_support_pooled_channel_mean_p90"],
            "K1_tau_p90": k1["tau_eval_object_support_pooled_channel_mean_p90"],
            "HR_tau_p90": hr["tau_eval_object_support_pooled_channel_mean_p90"],
            "TAU_BENEFIT_RETENTION": tau_retention,
            "M1_J_p99": m1["J_clear_eval_object_support_pooled_channel_mean_p99"],
            "K1_J_p99": k1["J_clear_eval_object_support_pooled_channel_mean_p99"],
            "HR_J_p99": hr["J_clear_eval_object_support_pooled_channel_mean_p99"],
            "HR_P_J_gt_1": hr["P_J_gt_1_object_support_pooled_channel_mean"],
            "HR_P_T_lt_0.1": hr["P_T_lt_0.1_object_support_pooled_channel_mean"],
            "BOUNDARY_PRESSURE": boundary_pressure,
            "RESIDUAL_SATURATION": bool(hr_rep.get("RESIDUAL_SATURATION", False)),
            "HIGH_J_MSE_GAP_RECOVERY": high_recovery,
            "LOW_J_DAMAGE": low_damage,
            "HF_RESIDUAL_REDUCTION_sigma3": hf_red3,
            "HF_RESIDUAL_REDUCTION_sigma9": hf_red9,
            "EDGE_RESIDUAL_REDUCTION": edge_red,
            "HR_vs_K1_views_improved": improved,
            "HR_vs_K1_views_degraded": degraded,
            "HR_delta_psnr_mean": _mean(deltas),
            "HR_delta_psnr_median": float(np.median(deltas)) if deltas else float("nan"),
            "HR_delta_psnr_min": min(deltas) if deltas else float("nan"),
            "HR_delta_psnr_max": max(deltas) if deltas else float("nan"),
            "K1_C_direct": k1_mse["C_direct"],
            "K1_C_medium": k1_mse["C_medium"],
            "K1_C_cross": k1_mse["C_cross"],
            "HR_C_direct": hr_mse["C_direct"],
            "HR_C_medium": hr_mse["C_medium"],
            "HR_C_cross": hr_mse["C_cross"],
            "STRONG_HR_RECOVERY": strong,
            "PARTIAL_HR_RECOVERY": partial,
            "HR_BOUNDARY_FAILURE": boundary_failure,
            "NO_HR_RECOVERY": no_recovery,
            "HR_HARMFUL": harmful,
            "HYPOTHESIS_ASSESSMENT": hypothesis,
            **{f"HR_REP_{key}": value for key, value in hr_rep.items() if key not in {"scene", "run"}},
        }
    ]


def write_visual_index(render_dir: Path, manifest: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Panama BND-HR Visual Compare Index", ""]
    for row in manifest:
        lines.append(f"- {row.get('output_type')}: `{row.get('file_path')}`")
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/mnt/new/home_old/ycy/water-splatting-refactor"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bnd_hr_panama_20260810"))
    parser.add_argument("--render-dir", type=Path, default=Path("renders/bnd_hr_panama_20260810"))
    parser.add_argument("--tile-width", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    render_dir = (repo / args.render_dir).resolve() if not args.render_dir.is_absolute() else args.render_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    trajectory_rows, missing_rows, checkpoint_rows, trajectory_rep_rows = trajectory_audit(repo)
    (
        per_view_rows,
        region_rows,
        frequency_rows,
        edge_rows,
        recomposition_rows,
        mse_rows,
        final_checkpoint_rows,
        visual_manifest,
        final_rep_rows,
        _hr_items,
    ) = final_view_audit(repo, render_dir, args.tile_width)
    checkpoint_rows.extend(final_checkpoint_rows)
    representation_named = [
        {"metric_group": "base_color_boundary", **final_rep_rows[0]},
        {"metric_group": "full_color_boundary", **final_rep_rows[1]},
        {"metric_group": "headroom_utilization", **final_rep_rows[2]},
        {"metric_group": "color_sh_capacity", **final_rep_rows[3]},
        {"metric_group": "positive_negative_residual_balance", **final_rep_rows[4]},
    ]
    summary_rows = final_summary(trajectory_rows, final_rep_rows, region_rows, frequency_rows, edge_rows, mse_rows, per_view_rows)

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
        "P_T_lt_0.1_object_support_pooled_channel_mean",
        "P_J_gt_1_object_support_pooled_channel_mean",
        "beta_D_raw_all_all_mean",
        "T_D_all_all_mean",
    }
    decomp_rows = [{key: value for key, value in row.items() if key in decomp_keys} for row in trajectory_rows]
    geometry_prefixes = ("depth_", "gaussian_scale_")
    geometry_keys = {"scene", "run", "nominal_step", "loaded_step", "gaussian_count", "gaussian_means_finite"}
    geometry_rows = [
        {key: value for key, value in row.items() if key in geometry_keys or any(key.startswith(prefix) for prefix in geometry_prefixes)}
        for row in trajectory_rows
    ]
    population_rows = [
        {"scene": row["scene"], "run": row["run"], "nominal_step": row["nominal_step"], "loaded_step": row["loaded_step"], "gaussian_count": row["gaussian_count"]}
        for row in trajectory_rows
        if row["run"] in ("K1", "HR")
    ]

    outputs: List[Tuple[str, Sequence[Mapping[str, Any]]]] = [
        ("training_trajectory", trajectory_rows),
        ("missing_checkpoints", missing_rows),
        ("final_rgb_metrics", rgb_rows),
        ("canonical_decomposition_metrics", decomp_rows),
        ("geometry_metrics", geometry_rows),
        ("base_color_boundary", [final_rep_rows[0]]),
        ("full_color_boundary", [final_rep_rows[1]]),
        ("headroom_utilization", [final_rep_rows[2]]),
        ("residual_saturation", [final_rep_rows[2]]),
        ("color_sh_capacity", [final_rep_rows[3]]),
        ("positive_negative_residual_balance", [final_rep_rows[4]]),
        ("representation_trajectory", trajectory_rep_rows),
        ("representation_metrics", representation_named),
        ("high_j_region_metrics", [row for row in region_rows if row.get("mask") == "M1_J_gt_1"]),
        ("control_region_metrics", [row for row in region_rows if row.get("mask") != "M1_J_gt_1"]),
        ("region_metrics", region_rows),
        ("frequency_metrics", frequency_rows),
        ("edge_metrics", edge_rows),
        ("frequency_edge_metrics", frequency_rows + edge_rows),
        ("per_view_metrics", per_view_rows),
        ("recomposition_metrics", recomposition_rows),
        ("mse_attribution", mse_rows),
        ("gaussian_population", population_rows),
        ("checkpoint_audit", checkpoint_rows),
        ("bnd_hr_final_summary", summary_rows),
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
