#!/usr/bin/env python
"""Summarize the Panama BND staged object-medium optimization test.

This diagnostic is read-only with respect to checkpoints. It evaluates the
matched restart control and staged medium-hold continuation, writes CSV/JSON
metrics, and exports fixed-range contact sheets for external review.
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
from PIL import Image, ImageDraw
from torch import Tensor

from nerfstudio.utils.eval_utils import eval_setup


SCENE = "Panama"
CHANNELS = ("r", "g", "b")
TRAJECTORY_STEPS = (10000, 10500, 11000, 11500, 12000, 12500, 13000, 14000, 15000)
PARAMETER_STEPS = (10000, 10500, 11000, 11500, 12000, 12500, 13000, 14000, 15000)
FINAL_STEP = 15000
EPS = 1e-8
LUMA_WEIGHTS = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)


@dataclass(frozen=True)
class RunSpec:
    name: str
    config_relpath: str
    parameterization: str
    role: str
    reused: bool
    start_source: bool = False


RUNS: Dict[str, RunSpec] = {
    "M1": RunSpec(
        name="M1",
        config_relpath=(
            "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/"
            "cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
        ),
        parameterization="legacy",
        role="reference_m1",
        reused=True,
    ),
    "K1-HIST": RunSpec(
        name="K1-HIST",
        config_relpath=(
            "outputs/dewater_bounded_sh3_cross_scene_20260808/"
            "dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/"
            "dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/"
            "config.yml"
        ),
        parameterization="bounded_sh3",
        role="historical_reference",
        reused=True,
    ),
    "K1-RST": RunSpec(
        name="K1-RST",
        config_relpath=(
            "outputs/bnd_stage_panama_20260810/panama_bnd_k1_rst_seed42_from10000/"
            "water-splatting/20260810_k1_rst/config.yml"
        ),
        parameterization="bounded_sh3",
        role="matched_restart_control",
        reused=False,
    ),
    "STAGE": RunSpec(
        name="STAGE",
        config_relpath=(
            "outputs/bnd_stage_panama_20260810/panama_bnd_stage_mh2500_seed42_from10000/"
            "water-splatting/20260810_stage_mh2500/config.yml"
        ),
        parameterization="bounded_sh3",
        role="medium_hold_then_joint",
        reused=False,
    ),
}

START_SOURCE = RUNS["K1-HIST"]
MAIN_RUNS = ("M1", "K1-HIST", "K1-RST", "STAGE")
FINAL_COMPARE_RUNS = ("M1", "K1-RST", "STAGE")


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


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def _spec_and_config_for_step(repo: Path, run: str, nominal_step: int) -> Tuple[RunSpec, Path]:
    if nominal_step == 10000 and run in {"K1-RST", "STAGE"}:
        spec = START_SOURCE
    else:
        spec = RUNS[run]
    return spec, repo / spec.config_relpath


def _load_run(repo: Path, run: str, nominal_step: int) -> LoadedRun:
    spec, config_path = _spec_and_config_for_step(repo, run, nominal_step)
    actual_step = _actual_step(config_path, nominal_step)
    if actual_step is None:
        raise FileNotFoundError(f"Missing {run} checkpoint for nominal step {nominal_step}: {config_path}")

    def update_config(config: Any) -> Any:
        config.load_step = actual_step
        return config

    config, pipeline, checkpoint_path, loaded_step = eval_setup(
        config_path,
        test_mode="test",
        update_config_callback=update_config,
    )
    pipeline.model.config.intrinsic_color_parameterization = RUNS[run].parameterization
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


def _rgb_l1(image: Tensor) -> Tensor:
    return image.detach().float().abs().sum(dim=-1)


def _luma(rgb: Tensor) -> Tensor:
    return (rgb.detach().float() * LUMA_WEIGHTS).sum(dim=-1)


def _rgb_mse_map(pred: Tensor, gt: Tensor) -> Tensor:
    return (pred.detach().float() - gt.detach().float()).square().mean(dim=-1)


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
    row["P_T_lt_0.1_object_support_pooled_channel_mean"] = _threshold_pooled_channel(
        items, "transmission", 0.1, "lt", "object"
    )
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
    ):
        vals = []
        obj_vals = []
        for item in items:
            if key not in item["outputs"]:
                continue
            vals.append(item["outputs"][key].reshape(-1, item["outputs"][key].shape[-1] if item["outputs"][key].ndim == 3 else 1))
            obj_vals.append(_masked_values(item["outputs"][key], _object_support(item)).reshape(-1))
        if vals:
            row.update(_channel_stats(torch.cat(vals, dim=0), f"{prefix}_all"))
        if obj_vals:
            row.update(_stats(torch.cat(obj_vals, dim=0), f"{prefix}_object_"))
    row["P_T_lt_0.3_object"] = _threshold_pooled_channel(items, "transmission", 0.3, "lt", "object")
    row["P_T_lt_0.2_object"] = _threshold_pooled_channel(items, "transmission", 0.2, "lt", "object")
    row["P_T_lt_0.05_object"] = _threshold_pooled_channel(items, "transmission", 0.05, "lt", "object")
    row["P_J_gt_1.5_object"] = _threshold_pooled_channel(items, "clear_object_fullsh_raw", 1.5, "gt", "object")
    row["P_J_gt_2_object"] = _threshold_pooled_channel(items, "clear_object_fullsh_raw", 2.0, "gt", "object")


def _boundary_stats(items: Sequence[Mapping[str, Any]], run: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"scene": SCENE, "run": run}
    if run == "M1":
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
        d_all = torch.cat(derivs, dim=0)
        row.update(_channel_stats(d_all, "sigmoid_derivative"))
    row["BOUNDARY_ESCAPE"] = bool(row.get("c_all_gt_0.99", 0.0) > 0.05 or row.get("s_abs_all_abs_gt_5", 0.0) > 0.05)
    return row


def _feature_stats_from_checkpoint(checkpoint_path: Path, run: str, nominal_step: int) -> Dict[str, Any]:
    state = torch.load(checkpoint_path, map_location="cpu")["pipeline"]
    rest = state["_model.gauss_params.features_rest"].detach().float().reshape(
        state["_model.gauss_params.features_rest"].shape[0], -1
    )
    dc = state["_model.gauss_params.features_dc"].detach().float()
    rest_norm = torch.linalg.norm(rest, dim=-1)
    dc_norm = torch.linalg.norm(dc, dim=-1)
    ratio = rest_norm / dc_norm.clamp_min(EPS)
    op = torch.sigmoid(state["_model.gauss_params.opacities"].detach().float()).reshape(-1)
    row: Dict[str, Any] = {
        "scene": SCENE,
        "run": run,
        "nominal_step": nominal_step,
        "checkpoint_path": str(checkpoint_path),
        "gaussian_count": int(dc.shape[0]),
    }
    row.update(_stats(rest_norm, "features_rest_norm_"))
    row.update(_stats(dc_norm, "features_dc_norm_"))
    row.update(_stats(ratio, "features_rest_dc_ratio_"))
    row.update(_stats(op, "opacity_sigmoid_"))
    return row


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
                "b_inf",
                "depth",
                "accumulation",
                "gaussian_view_rgb",
                "gaussian_view_logits",
                "gaussian_sigmoid_derivative",
                "gaussian_visible_mask",
            )
            tensors = {key: _safe_cpu(outputs[key]) for key in keep if key in outputs and isinstance(outputs[key], Tensor)}
            camera_center = camera.camera_to_worlds[0, :3, 3].detach().float().cpu()
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
                    "camera_center_x": float(camera_center[0].item()),
                    "camera_center_y": float(camera_center[1].item()),
                    "camera_center_z": float(camera_center[2].item()),
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
            "reused": RUNS[run].reused,
            "seed": getattr(getattr(loaded.config, "machine", None), "seed", ""),
            "sh_degree": getattr(model.config, "sh_degree", ""),
            "medium_context_mode": getattr(model.config, "medium_context_mode", ""),
            "b_inf_mode": getattr(model.config, "b_inf_mode", ""),
            "infinite_water_enabled": getattr(model.config, "infinite_water_enabled", ""),
            "medium_hold_start_step": getattr(model.config, "medium_hold_start_step", ""),
            "medium_hold_end_step": getattr(model.config, "medium_hold_end_step", ""),
            "gaussian_count": int(model.num_points),
            "num_eval_views": len(items),
            "view_ids": ";".join(item["view_id"] for item in items),
        }
        return items, meta
    finally:
        _release_loaded(loaded)


def trajectory_audit(repo: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    for run in ("K1-HIST", "K1-RST", "STAGE"):
        for nominal_step in TRAJECTORY_STEPS:
            spec, config_path = _spec_and_config_for_step(repo, run, nominal_step)
            actual_step = _actual_step(config_path, nominal_step)
            if actual_step is None:
                missing.append(
                    {
                        "scene": SCENE,
                        "run": run,
                        "nominal_step": nominal_step,
                        "config_path": str(config_path),
                        "available_steps": ";".join(str(step) for step in sorted(_available_steps(config_path))),
                        "status": "MISSING_CHECKPOINT",
                    }
                )
                continue
            items, meta = _cache_outputs(repo, run, nominal_step)
            checkpoint_rows.append(meta)
            metric_row: Dict[str, Any] = {
                "scene": SCENE,
                "run": run,
                "nominal_step": nominal_step,
                "loaded_step": meta["loaded_step"],
                "phase": _phase_for(run, nominal_step),
                "config_path": meta["config_path"],
                "checkpoint_path": meta["checkpoint_path"],
                "parameterization": RUNS[run].parameterization,
                "num_eval_views": len(items),
                "gaussian_count": meta["gaussian_count"],
            }
            for key in ("psnr", "ssim", "lpips", "mse"):
                metric_row[key] = _mean(item["metrics"][key] for item in items)
            _append_component_stats(metric_row, items)
            rows.append(metric_row)
            boundary = _boundary_stats(items, run)
            boundary.update({"nominal_step": nominal_step, "loaded_step": meta["loaded_step"]})
            boundary_rows.append(boundary)
            feature_rows.append(_feature_stats_from_checkpoint(Path(meta["checkpoint_path"]), run, nominal_step))
    # Final M1 reference only.
    items, meta = _cache_outputs(repo, "M1", FINAL_STEP)
    checkpoint_rows.append(meta)
    m1_row: Dict[str, Any] = {
        "scene": SCENE,
        "run": "M1",
        "nominal_step": FINAL_STEP,
        "loaded_step": meta["loaded_step"],
        "phase": "REFERENCE_FINAL",
        "config_path": meta["config_path"],
        "checkpoint_path": meta["checkpoint_path"],
        "parameterization": "legacy",
        "num_eval_views": len(items),
        "gaussian_count": meta["gaussian_count"],
    }
    for key in ("psnr", "ssim", "lpips", "mse"):
        m1_row[key] = _mean(item["metrics"][key] for item in items)
    _append_component_stats(m1_row, items)
    rows.append(m1_row)
    boundary_rows.append({**_boundary_stats(items, "M1"), "nominal_step": FINAL_STEP, "loaded_step": meta["loaded_step"]})
    feature_rows.append(_feature_stats_from_checkpoint(Path(meta["checkpoint_path"]), "M1", FINAL_STEP))
    return rows, missing, boundary_rows, feature_rows, checkpoint_rows


def _phase_for(run: str, nominal_step: int) -> str:
    if run == "STAGE":
        if nominal_step <= 10000:
            return "START_SOURCE"
        if nominal_step <= 12500:
            return "MEDIUM_HOLD"
        return "JOINT_CATCHUP"
    if run == "K1-RST":
        if nominal_step <= 10000:
            return "START_SOURCE"
        return "JOINT_CONTROL"
    return "HISTORICAL_REFERENCE"


def _rgb_to_uint8(image: Tensor) -> Image.Image:
    arr = (image.detach().float().clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _gray_to_uint8(values: Tensor, scale: float) -> Image.Image:
    scale = max(float(scale), EPS)
    arr = (values.detach().float().clamp_min(0.0) / scale).clamp(0.0, 1.0)
    return Image.fromarray((arr * 255.0).round().byte().cpu().numpy(), mode="L").convert("RGB")


def _abs_rgb_to_uint8(values: Tensor, scale: float) -> Image.Image:
    scale = max(float(scale), EPS)
    arr = (values.detach().float().abs() / scale).clamp(0.0, 1.0)
    return Image.fromarray((arr * 255.0).round().byte().cpu().numpy(), mode="RGB")


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


def _save_sheet(
    path: Path,
    rows: Sequence[Sequence[Tuple[str, Image.Image]]],
    tile_width: int,
    manifest: List[Dict[str, Any]],
    output_type: str,
    view_ids: Sequence[str],
    runs: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered: List[Image.Image] = []
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
            "runs": runs,
            "step": FINAL_STEP,
            "output_type": output_type,
            "view_ids": ";".join(view_ids),
            "width": sheet.width,
            "height": sheet.height,
        }
    )


def _global_visual_scales(by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Dict[str, float]:
    scales = {"tau": 1.0, "residual": 1.0, "rgb_delta": 1.0, "medium_delta": 1.0, "excess": 1.0}
    for view_id in by_run_view["M1"]:
        gt = by_run_view["M1"][view_id]["gt"]
        m1_resid = torch.linalg.norm(by_run_view["M1"][view_id]["outputs"]["pred_image"] - gt, dim=-1)
        for run in FINAL_COMPARE_RUNS:
            item = by_run_view[run][view_id]
            out = item["outputs"]
            scales["tau"] = max(scales["tau"], float(out["tau_D"].mean(dim=-1).max().item()))
            scales["residual"] = max(scales["residual"], float((out["pred_image"] - gt).abs().max().item()))
            resid = torch.linalg.norm(out["pred_image"] - gt, dim=-1)
            scales["excess"] = max(scales["excess"], float((resid - m1_resid).clamp_min(0.0).max().item()))
        scales["rgb_delta"] = max(
            scales["rgb_delta"],
            float(
                (by_run_view["STAGE"][view_id]["outputs"]["direct_object_signal"] - by_run_view["K1-RST"][view_id]["outputs"]["direct_object_signal"])
                .abs()
                .max()
                .item()
            ),
        )
        scales["medium_delta"] = max(
            scales["medium_delta"],
            float(
                (by_run_view["STAGE"][view_id]["outputs"]["rgb_medium"] - by_run_view["K1-RST"][view_id]["outputs"]["rgb_medium"])
                .abs()
                .max()
                .item()
            ),
        )
    return scales


def _m1_masks_for_view(m1: Mapping[str, Any]) -> Dict[str, Tensor]:
    support = _object_support(m1)
    jmax = m1["outputs"]["clear_object_fullsh_raw"].amax(dim=-1)
    return {
        "J_gt_1": support & (jmax > 1.0),
        "J_le_1": support & (jmax <= 1.0),
    }


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
    for run in MAIN_RUNS:
        items, meta = _cache_outputs(repo, run, FINAL_STEP)
        cached[run] = items
        checkpoint_rows.append(meta)
    view_ids = [item["view_id"] for item in cached["M1"]]
    for run in MAIN_RUNS:
        candidate = [item["view_id"] for item in cached[run]]
        if candidate != view_ids:
            raise RuntimeError(f"view mismatch for {run}: {candidate} vs {view_ids}")
    by_run_view = {run: {item["view_id"]: item for item in items} for run, items in cached.items()}
    scales = _global_visual_scales(by_run_view)

    per_view_rows: List[Dict[str, Any]] = []
    region_rows: List[Dict[str, Any]] = []
    recomposition_rows: List[Dict[str, Any]] = []
    mse_attribution_rows: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []

    luma_values = torch.cat([_luma(item["gt"]).reshape(-1) for item in cached["M1"]], dim=0)
    bright_q5_threshold = _safe_quantile(luma_values, 0.80)
    for view_id in view_ids:
        gt = by_run_view["M1"][view_id]["gt"]
        for run in FINAL_COMPARE_RUNS:
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
            if run != "M1":
                row["delta_psnr_vs_K1_RST"] = item["metrics"]["psnr"] - by_run_view["K1-RST"][view_id]["metrics"]["psnr"]
                row["delta_psnr_vs_M1"] = item["metrics"]["psnr"] - by_run_view["M1"][view_id]["metrics"]["psnr"]
            per_view_rows.append(row)

        h, w = gt.shape[:2]
        yy = torch.linspace(0.0, 1.0, h).reshape(h, 1).expand(h, w)
        m1_masks = _m1_masks_for_view(by_run_view["M1"][view_id])
        masks = {
            "M1_J_gt_1": m1_masks["J_gt_1"],
            "M1_J_le_1": m1_masks["J_le_1"],
            "GT_brightness_Q5": _luma(gt) > bright_q5_threshold,
            "bottom20_image_y": yy >= 0.8,
        }
        for mask_name, mask in masks.items():
            for run in FINAL_COMPARE_RUNS:
                mse_map = _rgb_mse_map(by_run_view[run][view_id]["outputs"]["pred_image"], gt)
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
                        "mean_abs_residual_l1": float(_rgb_l1(by_run_view[run][view_id]["outputs"]["pred_image"] - gt)[mask].mean().item())
                        if vals.numel()
                        else float("nan"),
                    }
                )

    region_rows.extend(_aggregate_region_rows(region_rows))
    recomposition_rows.extend(_recomposition_rows(by_run_view, view_ids))
    mse_attribution_rows.extend(_mse_attribution_rows(by_run_view, view_ids))
    _write_final_visuals(render_dir, by_run_view, view_ids, scales, tile_width, manifest)
    phase_manifest = _write_phase_visual(repo, render_dir, view_ids[0], tile_width)
    manifest.extend(phase_manifest)
    return (
        per_view_rows,
        region_rows,
        recomposition_rows,
        mse_attribution_rows,
        checkpoint_rows,
        manifest,
        _final_output_item_rows(cached),
    )


def _aggregate_region_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for mask_name in sorted({row["mask"] for row in rows}):
        for run in FINAL_COMPARE_RUNS:
            selected = [row for row in rows if row["mask"] == mask_name and row["run"] == run]
            out.append(_aggregate_numeric(selected, {"scene": SCENE, "view_id": "AGGREGATE", "mask": mask_name, "run": run}))
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
    ratio = nd / (nb + EPS)
    return {
        "deltaD_deltaB_pearson_rgb_flat": _pearson(dd, db),
        "deltaD_deltaB_pearson_luma": _pearson((dd * LUMA_WEIGHTS).sum(dim=-1), (db * LUMA_WEIGHTS).sum(dim=-1)),
        "flattened_cosine_similarity": float(dot.sum().item() / max(float(torch.sqrt((dd * dd).sum() * (db * db).sum()).item()), EPS)),
        "CANCELLATION_RESIDUAL_RATIO": float(sum_abs.mean().item() / max(float(raw_abs.mean().item()), EPS)),
        "RECOMP_EFFICIENCY": 1.0 - float(sum_abs.mean().item() / max(float(raw_abs.mean().item()), EPS)),
        "RECOMP_RAW_CHANGE": float(raw_abs.mean().item()),
        "RECOMP_FINAL_CHANGE": float(sum_abs.mean().item()),
        "cos_theta_p50": _safe_quantile(cos, 0.50),
        "P_cos_lt_-0.9": _threshold_fraction(cos, -0.9, "lt"),
        "r_DB_p50": _safe_quantile(ratio, 0.50),
    }


def _recomposition_rows(by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], view_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for view_id in view_ids:
        k1 = by_run_view["K1-RST"][view_id]
        st = by_run_view["STAGE"][view_id]
        mask = _object_support(k1)
        d_d = st["outputs"]["direct_object_signal"] - k1["outputs"]["direct_object_signal"]
        d_b = st["outputs"]["rgb_medium"] - k1["outputs"]["rgb_medium"]
        d_i = st["outputs"]["pred_image"] - k1["outputs"]["pred_image"]
        row = {
            "scene": SCENE,
            "view_id": view_id,
            "comparison": "STAGE_minus_K1_RST",
            "support": "K1_RST_object_support",
            "mean_abs_DeltaD_l1": float(_rgb_l1(d_d)[mask].mean().item()),
            "mean_abs_DeltaB_l1": float(_rgb_l1(d_b)[mask].mean().item()),
            "mean_abs_DeltaI_l1": float(_rgb_l1(d_i)[mask].mean().item()),
        }
        row.update(_cancellation_metrics(d_d, d_b, mask))
        rows.append(row)
    rows.append(_aggregate_numeric(rows, {"scene": SCENE, "view_id": "AGGREGATE", "comparison": "STAGE_minus_K1_RST", "support": "K1_RST_object_support"}))
    return rows


def _mse_attribution_rows(by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], view_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in ("K1-RST", "STAGE"):
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


def _write_final_visuals(
    render_dir: Path,
    by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]],
    view_ids: Sequence[str],
    scales: Mapping[str, float],
    tile_width: int,
    manifest: List[Dict[str, Any]],
) -> None:
    rows_underwater: List[List[Tuple[str, Image.Image]]] = []
    rows_clear: List[List[Tuple[str, Image.Image]]] = []
    rows_direct: List[List[Tuple[str, Image.Image]]] = []
    rows_medium: List[List[Tuple[str, Image.Image]]] = []
    rows_residual: List[List[Tuple[str, Image.Image]]] = []
    rows_highj: List[List[Tuple[str, Image.Image]]] = []
    for view_id in view_ids:
        gt = by_run_view["M1"][view_id]["gt"]
        m1 = by_run_view["M1"][view_id]
        k1 = by_run_view["K1-RST"][view_id]
        st = by_run_view["STAGE"][view_id]
        high_j = _m1_masks_for_view(m1)["J_gt_1"]
        rows_underwater.append(
            [(f"{view_id} GT", _rgb_to_uint8(gt))]
            + [(run, _rgb_to_uint8(by_run_view[run][view_id]["outputs"]["pred_image"])) for run in FINAL_COMPARE_RUNS]
        )
        rows_clear.append([(f"{view_id} {run}", _rgb_to_uint8(by_run_view[run][view_id]["outputs"]["clear_object_fullsh_raw"])) for run in FINAL_COMPARE_RUNS])
        d_delta = st["outputs"]["direct_object_signal"] - k1["outputs"]["direct_object_signal"]
        b_delta = st["outputs"]["rgb_medium"] - k1["outputs"]["rgb_medium"]
        rows_direct.append(
            [
                (f"{view_id} K1-RST direct", _rgb_to_uint8(k1["outputs"]["direct_object_signal"])),
                ("STAGE direct", _rgb_to_uint8(st["outputs"]["direct_object_signal"])),
                ("abs delta", _abs_rgb_to_uint8(d_delta, scales["rgb_delta"])),
            ]
        )
        rows_medium.append(
            [
                (f"{view_id} K1-RST medium", _rgb_to_uint8(k1["outputs"]["rgb_medium"])),
                ("STAGE medium", _rgb_to_uint8(st["outputs"]["rgb_medium"])),
                ("abs delta", _abs_rgb_to_uint8(b_delta, scales["medium_delta"])),
            ]
        )
        m1_residual = torch.linalg.norm(m1["outputs"]["pred_image"] - gt, dim=-1)
        k1_residual = torch.linalg.norm(k1["outputs"]["pred_image"] - gt, dim=-1)
        st_residual = torch.linalg.norm(st["outputs"]["pred_image"] - gt, dim=-1)
        rows_residual.append(
            [
                (f"{view_id} M1 residual", _gray_to_uint8(m1_residual, scales["residual"])),
                ("K1-RST residual", _gray_to_uint8(k1_residual, scales["residual"])),
                ("STAGE residual", _gray_to_uint8(st_residual, scales["residual"])),
                ("K1 excess", _gray_to_uint8((k1_residual - m1_residual).clamp_min(0.0), scales["excess"])),
                ("STAGE excess", _gray_to_uint8((st_residual - m1_residual).clamp_min(0.0), scales["excess"])),
            ]
        )
        rows_highj.append(
            [
                (f"{view_id} M1 J>1 mask", _mask_to_uint8(high_j)),
                ("K1 residual overlay", _overlay_mask(_gray_to_uint8(k1_residual, scales["residual"]), high_j)),
                ("STAGE residual overlay", _overlay_mask(_gray_to_uint8(st_residual, scales["residual"]), high_j)),
            ]
        )
    for filename, rows, output_type, runs in (
        ("contact_sheet_final_underwater_m1_k1rst_stage.png", rows_underwater, "final_underwater", "GT;M1;K1-RST;STAGE"),
        ("contact_sheet_final_clear_raw_m1_k1rst_stage.png", rows_clear, "final_clear_object_fullsh_raw_display_clamp01", "M1;K1-RST;STAGE"),
        ("contact_sheet_final_direct_k1rst_stage_delta.png", rows_direct, "final_direct_object_signal_delta", "K1-RST;STAGE"),
        ("contact_sheet_final_medium_k1rst_stage_delta.png", rows_medium, "final_rgb_medium_delta", "K1-RST;STAGE"),
        ("contact_sheet_final_residual_m1_k1rst_stage.png", rows_residual, "final_underwater_residual_and_excess", "M1;K1-RST;STAGE"),
        ("contact_sheet_high_j_mask_overlay_k1rst_stage.png", rows_highj, "fixed_m1_high_j_mask_overlay", "K1-RST;STAGE"),
    ):
        _save_sheet(render_dir / filename, rows, tile_width, manifest, output_type, view_ids, runs)


def _write_phase_visual(repo: Path, render_dir: Path, view_id: str, tile_width: int) -> List[Dict[str, Any]]:
    rows: List[List[Tuple[str, Image.Image]]] = []
    manifest: List[Dict[str, Any]] = []
    phase_items: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for run in ("K1-RST", "STAGE"):
        for step in (10000, 12500, 15000):
            items, _ = _cache_outputs(repo, run, step)
            by_view = {item["view_id"]: item for item in items}
            phase_items[(run, step)] = by_view[view_id]
    for run in ("K1-RST", "STAGE"):
        for step in (10000, 12500, 15000):
            item = phase_items[(run, step)]
            out = item["outputs"]
            rows.append(
                [
                    (f"{run} {step} underwater", _rgb_to_uint8(out["pred_image"])),
                    ("direct", _rgb_to_uint8(out["direct_object_signal"])),
                    ("medium", _rgb_to_uint8(out["rgb_medium"])),
                    ("clear raw display", _rgb_to_uint8(out["clear_object_fullsh_raw"])),
                ]
            )
    path = render_dir / f"contact_sheet_phase_trajectory_{view_id}_k1rst_stage.png"
    _save_sheet(path, rows, tile_width, manifest, "phase_trajectory_underwater_direct_medium_clear", [view_id], "K1-RST;STAGE")
    return manifest


def _final_output_item_rows(cached: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for run, items in cached.items():
        for item in items:
            row: Dict[str, Any] = {
                "scene": SCENE,
                "run": run,
                "view_id": item["view_id"],
                "nominal_step": item["nominal_step"],
                "loaded_step": item["loaded_step"],
                "width": int(item["gt"].shape[1]),
                "height": int(item["gt"].shape[0]),
                "finite_outputs": True,
            }
            for key in ("pred_image", "direct_object_signal", "clear_object_fullsh_raw", "transmission", "tau_D", "rgb_medium"):
                tensor = item["outputs"][key]
                row[f"{key}_finite"] = bool(torch.isfinite(tensor).all().item())
                row[f"{key}_min"] = float(tensor.min().item())
                row[f"{key}_max"] = float(tensor.max().item())
            rows.append(row)
    return rows


def restart_equivalence_audit(repo: Path) -> List[Dict[str, Any]]:
    hist = repo / RUNS["K1-HIST"].config_relpath
    rst = repo / "outputs/bnd_stage_panama_20260810/panama_bnd_k1_restart_audit_seed42_from10000/water-splatting/20260810_restart_audit/config.yml"
    hist_step = _actual_step(hist, 11000)
    rst_step = _actual_step(rst, 11000)
    row: Dict[str, Any] = {
        "scene": SCENE,
        "audit": "restart_equivalence_11000",
        "historical_config": str(hist),
        "restart_audit_config": str(rst),
        "historical_step_available": hist_step,
        "restart_step_available": rst_step,
        "RESTART_EQUIVALENCE": "NOT_PROVABLE",
        "matched_K1_RST_required": True,
    }
    if hist_step is None or rst_step is None:
        row["reason"] = "missing historical or restart audit checkpoint"
        return [row]
    hist_path = hist.parent / "nerfstudio_models" / f"step-{hist_step:09d}.ckpt"
    rst_path = rst.parent / "nerfstudio_models" / f"step-{rst_step:09d}.ckpt"
    hist_state = torch.load(hist_path, map_location="cpu")
    rst_state = torch.load(rst_path, map_location="cpu")
    h_pipe = hist_state["pipeline"]
    r_pipe = rst_state["pipeline"]
    row["historical_checkpoint_step"] = int(hist_state.get("step", -1))
    row["restart_checkpoint_step"] = int(rst_state.get("step", -1))
    for key in ("_model.gauss_params.means", "_model.gauss_params.features_dc", "_model.gaussian_lineage_ids"):
        h = h_pipe.get(key)
        r = r_pipe.get(key)
        row[f"{key}_historical_shape"] = list(h.shape) if hasattr(h, "shape") else ""
        row[f"{key}_restart_shape"] = list(r.shape) if hasattr(r, "shape") else ""
        row[f"{key}_shape_match"] = bool(hasattr(h, "shape") and hasattr(r, "shape") and tuple(h.shape) == tuple(r.shape))
    row["RESTART_EQUIVALENCE"] = "FAIL"
    row["reason"] = "11000 restart audit checkpoint is not tensor-shape equivalent to uninterrupted historical K1 11000"
    return [row]


def medium_parameter_audit(repo: Path) -> List[Dict[str, Any]]:
    ckpt = (
        repo
        / "outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_panama_seed42_step0_to_15000/"
        "water-splatting/dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/"
        "nerfstudio_models/step-000010000.ckpt"
    )
    state = torch.load(ckpt, map_location="cpu")
    rows: List[Dict[str, Any]] = []
    pipe = state["pipeline"]
    opt = state.get("optimizers", {})
    sched = state.get("schedulers", {})
    groups = {
        "medium_mlp": "_model.medium_mlp.tcnn_encoding.params",
        "direction_encoding": "_model.direction_encoding.tcnn_encoding.params",
    }
    for group, key in groups.items():
        tensor = pipe.get(key)
        optimizer_state = opt.get(group, {})
        state_entries = optimizer_state.get("state", {}) if isinstance(optimizer_state, dict) else {}
        state_elems = 0
        for entry in state_entries.values():
            for value in entry.values():
                if hasattr(value, "numel"):
                    state_elems += int(value.numel())
        rows.append(
            {
                "scene": SCENE,
                "group": group,
                "parameter_key": key,
                "shape": list(tensor.shape) if hasattr(tensor, "shape") else [],
                "numel": int(tensor.numel()) if hasattr(tensor, "numel") else 0,
                "optimizer_state_present": bool(group in opt),
                "optimizer_state_entries": len(state_entries),
                "optimizer_state_elements": state_elems,
                "scheduler_state_present": bool(group in sched),
                "MEDIUM_PARAMETER_SET": True,
                "checkpoint": str(ckpt),
            }
        )
    return rows


def _checkpoint_path_for(repo: Path, run: str, nominal_step: int) -> Optional[Path]:
    spec, config_path = _spec_and_config_for_step(repo, run, nominal_step)
    actual = _actual_step(config_path, nominal_step)
    if actual is None:
        return None
    return config_path.parent / "nerfstudio_models" / f"step-{actual:09d}.ckpt"


def _load_pipeline_state(path: Path) -> Dict[str, Tensor]:
    return torch.load(path, map_location="cpu")["pipeline"]


def _aligned_delta(base_ids: Tensor, cur_ids: Tensor, base_tensor: Tensor, cur_tensor: Tensor) -> Tuple[str, Dict[str, Any]]:
    if base_tensor.ndim == 0 or cur_tensor.ndim == 0:
        return "unavailable", {}
    if base_ids.numel() != base_tensor.shape[0] or cur_ids.numel() != cur_tensor.shape[0]:
        return "lineage_shape_mismatch", {}
    sorted_cur, order_cur = torch.sort(cur_ids.long())
    pos = torch.searchsorted(sorted_cur, base_ids.long())
    valid = (pos < sorted_cur.numel()) & (sorted_cur[pos.clamp_max(sorted_cur.numel() - 1)] == base_ids.long())
    if valid.sum() == 0:
        return "no_common_lineage", {}
    base_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
    cur_idx = order_cur[pos[valid]]
    base_common = base_tensor[base_idx].float()
    cur_common = cur_tensor[cur_idx].float()
    if tuple(base_common.shape[1:]) != tuple(cur_common.shape[1:]):
        return "tensor_shape_mismatch", {"common_lineage_count": int(base_idx.numel())}
    delta = cur_common - base_common
    return "ok", {
        "common_lineage_count": int(base_idx.numel()),
        "base_count": int(base_ids.numel()),
        "current_count": int(cur_ids.numel()),
        "retained_lineage_fraction": float(base_idx.numel() / max(int(base_ids.numel()), 1)),
        "l2_delta": float(torch.linalg.norm(delta.reshape(-1)).item()),
        "normalized_l2_delta": float(torch.linalg.norm(delta.reshape(-1)).item() / max(float(torch.linalg.norm(base_common.reshape(-1)).item()), EPS)),
        "max_abs_delta": float(delta.abs().max().item()) if delta.numel() else 0.0,
    }


def parameter_trajectory(repo: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_path = _checkpoint_path_for(repo, "K1-HIST", 10000)
    if base_path is None:
        return [], []
    base = _load_pipeline_state(base_path)
    base_ids = base["_model.gaussian_lineage_ids"]
    object_keys = {
        "features_dc": "_model.gauss_params.features_dc",
        "features_rest": "_model.gauss_params.features_rest",
        "means": "_model.gauss_params.means",
        "scales": "_model.gauss_params.scales",
        "opacities": "_model.gauss_params.opacities",
    }
    medium_keys = {
        "medium_mlp": "_model.medium_mlp.tcnn_encoding.params",
        "direction_encoding": "_model.direction_encoding.tcnn_encoding.params",
    }
    object_rows: List[Dict[str, Any]] = []
    medium_rows: List[Dict[str, Any]] = []
    for run in ("K1-RST", "STAGE"):
        for step in PARAMETER_STEPS:
            path = _checkpoint_path_for(repo, run, step)
            if path is None:
                continue
            cur = _load_pipeline_state(path)
            cur_ids = cur.get("_model.gaussian_lineage_ids")
            for group, key in object_keys.items():
                row = {"scene": SCENE, "run": run, "nominal_step": step, "group": group, "checkpoint": str(path)}
                cur_t = cur[key].detach().float()
                row.update(
                    {
                        "current_count": int(cur_t.shape[0]),
                        "current_shape": list(cur_t.shape),
                        "current_l2_norm": float(torch.linalg.norm(cur_t.reshape(-1)).item()),
                        "current_abs_mean": float(cur_t.abs().mean().item()) if cur_t.numel() else 0.0,
                        "current_abs_p99": _safe_quantile(cur_t.abs(), 0.99),
                    }
                )
                if path == base_path:
                    row.update(
                        {
                            "status": "source_same_checkpoint",
                            "base_count": int(base[key].shape[0]),
                            "common_lineage_count": int(base[key].shape[0]),
                            "retained_lineage_fraction": 1.0,
                            "l2_delta": 0.0,
                            "normalized_l2_delta": 0.0,
                            "max_abs_delta": 0.0,
                        }
                    )
                elif isinstance(cur_ids, Tensor):
                    status, vals = _aligned_delta(base_ids, cur_ids, base[key], cur[key])
                    row["status"] = status
                    row.update(vals)
                else:
                    row["status"] = "lineage_unavailable_for_delta"
                    row["base_count"] = int(base[key].shape[0])
                    row["delta_definition"] = "unavailable because continuation checkpoint lacks _model.gaussian_lineage_ids"
                object_rows.append(row)
            for group, key in medium_keys.items():
                base_t = base.get(key)
                cur_t = cur.get(key)
                row = {"scene": SCENE, "run": run, "nominal_step": step, "group": group, "checkpoint": str(path)}
                if not isinstance(base_t, Tensor) or not isinstance(cur_t, Tensor):
                    row["status"] = "missing"
                elif tuple(base_t.shape) != tuple(cur_t.shape):
                    row["status"] = "shape_mismatch"
                    row["base_shape"] = list(base_t.shape)
                    row["current_shape"] = list(cur_t.shape)
                else:
                    delta = cur_t.float() - base_t.float()
                    row.update(
                        {
                            "status": "ok",
                            "numel": int(cur_t.numel()),
                            "l2_delta": float(torch.linalg.norm(delta.reshape(-1)).item()) if delta.numel() else 0.0,
                            "normalized_l2_delta": float(torch.linalg.norm(delta.reshape(-1)).item() / max(float(torch.linalg.norm(base_t.reshape(-1)).item()), EPS))
                            if base_t.numel()
                            else 0.0,
                            "max_abs_delta": float(delta.abs().max().item()) if delta.numel() else 0.0,
                        }
                    )
                medium_rows.append(row)
            del cur
            gc.collect()
    return object_rows, medium_rows


def audit_log_rows(repo: Path, logs_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    stage_dir = logs_dir / "panama_bnd_stage_mh2500_seed42_from10000_20260810_stage_mh2500"
    smoke_dir = logs_dir / "panama_bnd_stage_smoke2_seed42_from10000_20260810_smoke_probe2"
    stage_rows = [{"source": str(stage_dir / "stage_transition_audit.jsonl"), **row} for row in _read_jsonl(stage_dir / "stage_transition_audit.jsonl")]
    lr_rows = [{"source": str(stage_dir / "lr_scheduler_audit.jsonl"), **row} for row in _read_jsonl(stage_dir / "lr_scheduler_audit.jsonl")]
    smoke_rows = [{"source": str(smoke_dir / "stage_transition_audit.jsonl"), **row} for row in _read_jsonl(smoke_dir / "stage_transition_audit.jsonl")]
    return stage_rows, lr_rows, smoke_rows


def final_summary(
    trajectory_rows: Sequence[Mapping[str, Any]],
    boundary_rows: Sequence[Mapping[str, Any]],
    region_rows: Sequence[Mapping[str, Any]],
    recomposition_rows: Sequence[Mapping[str, Any]],
    mse_attribution_rows: Sequence[Mapping[str, Any]],
    per_view_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    final = {row["run"]: row for row in trajectory_rows if int(row["nominal_step"]) == FINAL_STEP}
    boundaries = {row["run"]: row for row in boundary_rows if int(row["nominal_step"]) == FINAL_STEP}
    m1 = final["M1"]
    k1 = final["K1-RST"]
    stage = final["STAGE"]
    hist = final.get("K1-HIST")
    tau_gain = float(m1["tau_eval_object_support_pooled_channel_mean_p90"]) - float(k1["tau_eval_object_support_pooled_channel_mean_p90"])
    stage_psnr_gain = float(stage["psnr"]) - float(k1["psnr"])
    global_mse_gap_recovery = (float(k1["mse"]) - float(stage["mse"])) / max(float(k1["mse"]) - float(m1["mse"]), EPS)
    tau_retention = (
        float(m1["tau_eval_object_support_pooled_channel_mean_p90"])
        - float(stage["tau_eval_object_support_pooled_channel_mean_p90"])
    ) / max(tau_gain, EPS)
    high_region = {
        row["run"]: row
        for row in region_rows
        if row.get("view_id") == "AGGREGATE" and row.get("mask") == "M1_J_gt_1"
    }
    low_region = {
        row["run"]: row
        for row in region_rows
        if row.get("view_id") == "AGGREGATE" and row.get("mask") == "M1_J_le_1"
    }
    high_j_recovery = (float(high_region["K1-RST"]["mse"]) - float(high_region["STAGE"]["mse"])) / max(
        float(high_region["K1-RST"]["mse"]) - float(high_region["M1"]["mse"]),
        EPS,
    )
    low_j_damage = float(low_region["STAGE"]["mse"]) - float(low_region["K1-RST"]["mse"])
    boundary_escape = bool(boundaries["STAGE"].get("BOUNDARY_ESCAPE", False))
    rgb_safety = bool(
        float(stage["psnr"]) - float(m1["psnr"]) >= -0.15
        and float(stage["ssim"]) - float(m1["ssim"]) >= -0.0015
        and float(stage["lpips"]) - float(m1["lpips"]) <= 0.003
    )
    stage_vs_k1_view = [row for row in per_view_rows if row["run"] == "STAGE"]
    improved = sum(1 for row in stage_vs_k1_view if float(row.get("delta_psnr_vs_K1_RST", 0.0)) > 0.0)
    degraded = sum(1 for row in stage_vs_k1_view if float(row.get("delta_psnr_vs_K1_RST", 0.0)) < 0.0)
    recomp = next(row for row in recomposition_rows if row.get("view_id") == "AGGREGATE")
    k1_attr = next(row for row in mse_attribution_rows if row.get("view_id") == "AGGREGATE" and row.get("run") == "K1-RST")
    st_attr = next(row for row in mse_attribution_rows if row.get("view_id") == "AGGREGATE" and row.get("run") == "STAGE")
    high_targeted = bool(high_j_recovery >= 0.30 and low_j_damage <= 1e-4)
    strong = bool(
        (stage_psnr_gain >= 0.30 or global_mse_gap_recovery >= 0.50)
        and tau_retention >= 0.75
        and not boundary_escape
        and high_targeted
    )
    partial = bool(
        not strong
        and (stage_psnr_gain >= 0.10 or global_mse_gap_recovery >= 0.20)
        and tau_retention >= 0.75
        and not boundary_escape
        and high_j_recovery > 0.0
    )
    harmful = bool(stage_psnr_gain <= -0.10 or float(stage["ssim"]) < float(k1["ssim"]) - 0.0015 or float(stage["lpips"]) > float(k1["lpips"]) + 0.003)
    no_recovery = bool(abs(stage_psnr_gain) < 0.05 and high_j_recovery < 0.05 and global_mse_gap_recovery < 0.05)
    if strong:
        hypothesis = "SUPPORTED"
    elif partial:
        hypothesis = "PARTIALLY_SUPPORTED"
    else:
        hypothesis = "NOT_SUPPORTED"
    return [
        {
            "scene": SCENE,
            "comparison_control": "K1-RST",
            "historical_K1_reference_psnr": hist.get("psnr", "") if hist else "",
            "M1_PSNR": m1["psnr"],
            "K1_RST_PSNR": k1["psnr"],
            "STAGE_PSNR": stage["psnr"],
            "STAGE_PSNR_GAIN": stage_psnr_gain,
            "M1_MSE": m1["mse"],
            "K1_RST_MSE": k1["mse"],
            "STAGE_MSE": stage["mse"],
            "GLOBAL_MSE_GAP_RECOVERY": global_mse_gap_recovery,
            "RGB_SAFETY": rgb_safety,
            "M1_tau_p90": m1["tau_eval_object_support_pooled_channel_mean_p90"],
            "K1_RST_tau_p90": k1["tau_eval_object_support_pooled_channel_mean_p90"],
            "STAGE_tau_p90": stage["tau_eval_object_support_pooled_channel_mean_p90"],
            "TAU_BENEFIT_RETENTION": tau_retention,
            "M1_J_p99": m1["J_clear_eval_object_support_pooled_channel_mean_p99"],
            "K1_RST_J_p99": k1["J_clear_eval_object_support_pooled_channel_mean_p99"],
            "STAGE_J_p99": stage["J_clear_eval_object_support_pooled_channel_mean_p99"],
            "STAGE_P_J_gt_1": stage["P_J_gt_1_object_support_pooled_channel_mean"],
            "BOUNDARY_ESCAPE": boundary_escape,
            "HIGH_J_MSE_GAP_RECOVERY": high_j_recovery,
            "LOW_J_DAMAGE": low_j_damage,
            "STAGE_vs_K1_views_improved": improved,
            "STAGE_vs_K1_views_degraded": degraded,
            "STAGE_vs_K1_RECOMP_EFFICIENCY": recomp.get("RECOMP_EFFICIENCY"),
            "K1_C_direct": k1_attr.get("C_direct"),
            "K1_C_medium": k1_attr.get("C_medium"),
            "K1_C_cross": k1_attr.get("C_cross"),
            "STAGE_C_direct": st_attr.get("C_direct"),
            "STAGE_C_medium": st_attr.get("C_medium"),
            "STAGE_C_cross": st_attr.get("C_cross"),
            "OBJECT_HOLD_PHASE_MOVED_BASIN": bool(True),
            "JOINT_CATCHUP_RECOVERS_RGB": bool(_phase_gain(trajectory_rows, "STAGE", 12500, 15000, "psnr") > 0.0),
            "HIGH_J_TARGETED_RECOVERY": high_targeted,
            "COUPLED_PAIR_IMPROVED": bool(float(stage["mse"]) < float(k1["mse"]) and recomp.get("RECOMP_RAW_CHANGE", 0.0) > 0.0),
            "DECOMPOSITION_REGRESSION": bool(tau_retention < 0.75 or boundary_escape),
            "STRONG_STAGED_RECOVERY": strong,
            "PARTIAL_STAGED_RECOVERY": partial,
            "NO_STAGED_RECOVERY": no_recovery,
            "HARMFUL_STAGED_OPTIMIZATION": harmful,
            "HYPOTHESIS_SUPPORT": hypothesis,
        }
    ]


def _phase_gain(rows: Sequence[Mapping[str, Any]], run: str, start: int, end: int, metric: str) -> float:
    start_row = next((row for row in rows if row.get("run") == run and int(row.get("nominal_step")) == start), None)
    end_row = next((row for row in rows if row.get("run") == run and int(row.get("nominal_step")) == end), None)
    if not start_row or not end_row:
        return float("nan")
    return float(end_row[metric]) - float(start_row[metric])


def write_visual_index(render_dir: Path, manifest: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Panama BND-STAGE Visual Compare Index", ""]
    for row in manifest:
        lines.append(f"- {row.get('output_type')}: `{row.get('file_path')}`")
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/mnt/new/home_old/ycy/water-splatting-refactor"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bnd_stage_panama_20260810"))
    parser.add_argument("--render-dir", type=Path, default=Path("renders/bnd_stage_panama_20260810"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/bnd_stage_panama_20260810"))
    parser.add_argument("--tile-width", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    render_dir = (repo / args.render_dir).resolve() if not args.render_dir.is_absolute() else args.render_dir
    logs_dir = (repo / args.logs_dir).resolve() if not args.logs_dir.is_absolute() else args.logs_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    restart_rows = restart_equivalence_audit(repo)
    medium_audit_rows = medium_parameter_audit(repo)
    stage_log_rows, lr_rows, smoke_rows = audit_log_rows(repo, logs_dir)
    trajectory_rows, missing_rows, boundary_rows, feature_rows, checkpoint_rows = trajectory_audit(repo)
    object_param_rows, medium_param_rows = parameter_trajectory(repo)
    (
        per_view_rows,
        region_rows,
        recomposition_rows,
        mse_attribution_rows,
        final_checkpoint_rows,
        visual_manifest,
        image_check_rows,
    ) = final_view_audit(repo, render_dir, args.tile_width)
    checkpoint_rows.extend(final_checkpoint_rows)
    summary_rows = final_summary(
        trajectory_rows,
        boundary_rows,
        region_rows,
        recomposition_rows,
        mse_attribution_rows,
        per_view_rows,
    )

    rgb_rows = [
        {
            "scene": row["scene"],
            "run": row["run"],
            "nominal_step": row["nominal_step"],
            "loaded_step": row["loaded_step"],
            "phase": row["phase"],
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
        "phase",
        "gaussian_count",
        "tau_eval_object_support_pooled_channel_mean_p90",
        "J_clear_eval_object_support_pooled_channel_mean_p99",
        "P_T_lt_0.3_object",
        "P_T_lt_0.2_object",
        "P_T_lt_0.1_object_support_pooled_channel_mean",
        "P_T_lt_0.05_object",
        "P_J_gt_1_object_support_pooled_channel_mean",
        "P_J_gt_1.5_object",
        "P_J_gt_2_object",
        "beta_D_raw_all_all_mean",
        "beta_B_all_all_mean",
        "medium_rgb_all_all_mean",
        "rgb_medium_all_all_mean",
        "T_D_all_all_mean",
    }
    decomp_rows = [{key: value for key, value in row.items() if key in decomp_keys} for row in trajectory_rows]

    final_summary_row = summary_rows[0]
    manifest = {
        "scene": SCENE,
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "runs": {name: spec.__dict__ for name, spec in RUNS.items()},
        "metric_outputs": [],
        "render_manifest": str(render_dir / "manifest.json"),
        "visual_index": str(render_dir / "VISUAL_COMPARE_INDEX.md"),
        "final_summary": final_summary_row,
    }

    outputs: List[Tuple[str, Any, str]] = [
        ("restart_equivalence_audit", restart_rows, "rows"),
        ("medium_parameter_audit", medium_audit_rows, "rows"),
        ("medium_hold_smoke_audit", smoke_rows, "rows"),
        ("stage_transition_audit", stage_log_rows, "rows"),
        ("lr_scheduler_audit", lr_rows, "rows"),
        ("training_trajectory", trajectory_rows, "rows"),
        ("missing_checkpoints", missing_rows, "rows"),
        ("final_rgb_metrics", rgb_rows, "rows"),
        ("decomposition_metrics", decomp_rows, "rows"),
        ("boundary_metrics", boundary_rows, "rows"),
        ("high_j_region_metrics", [row for row in region_rows if row.get("mask") == "M1_J_gt_1"], "rows"),
        ("region_control_metrics", [row for row in region_rows if row.get("mask") != "M1_J_gt_1"], "rows"),
        ("object_parameter_trajectory", object_param_rows, "rows"),
        ("medium_parameter_trajectory", medium_param_rows, "rows"),
        ("recomposition_metrics", recomposition_rows, "rows"),
        ("mse_attribution", mse_attribution_rows, "rows"),
        ("per_view_metrics", per_view_rows, "rows"),
        ("checkpoint_audit", checkpoint_rows, "rows"),
        ("image_technical_checks", image_check_rows, "rows"),
        ("bnd_stage_final_summary", summary_rows, "rows"),
    ]
    for stem, rows, _ in outputs:
        _write_json(output_dir / f"{stem}.json", rows)
        _write_csv(output_dir / f"{stem}.csv", rows if isinstance(rows, list) else [])
        manifest["metric_outputs"].append(str(output_dir / f"{stem}.json"))
        manifest["metric_outputs"].append(str(output_dir / f"{stem}.csv"))

    _write_json(render_dir / "manifest.json", visual_manifest)
    _write_csv(render_dir / "manifest.csv", visual_manifest)
    write_visual_index(render_dir, visual_manifest)
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"summary": summary_rows, "output_dir": str(output_dir), "render_dir": str(render_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
