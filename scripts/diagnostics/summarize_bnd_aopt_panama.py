#!/usr/bin/env python
"""Summarize and render the Panama BND-AOPT experiment.

This diagnostic is read-only with respect to checkpoints. It loads existing M1
and BND-K1 checkpoints plus the newly trained BND-K2/K4 checkpoints, evaluates
the same Panama eval views, and writes CSV/JSON metrics plus contact sheets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw

from nerfstudio.utils.eval_utils import eval_setup
from water_splatting.water_splatting import SH2RGB, SHLogits2RGB


SCENE = "Panama"
TRAJECTORY_STEPS = (1000, 3000, 5000, 8000, 10000, 13000, 15000)
FINAL_STEP = 15000
CHANNELS = ("r", "g", "b")


@dataclass(frozen=True)
class RunSpec:
    name: str
    config_relpath: str
    parameterization: str
    appearance_lr_scale: float
    reused: bool


RUNS: Dict[str, RunSpec] = {
    "M1": RunSpec(
        name="M1",
        config_relpath=(
            "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/"
            "cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
        ),
        parameterization="legacy",
        appearance_lr_scale=1.0,
        reused=True,
    ),
    "K1": RunSpec(
        name="K1",
        config_relpath=(
            "outputs/dewater_bounded_sh3_cross_scene_20260808/"
            "dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/"
            "dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/"
            "config.yml"
        ),
        parameterization="bounded_sh3",
        appearance_lr_scale=1.0,
        reused=True,
    ),
    "K2": RunSpec(
        name="K2",
        config_relpath=(
            "outputs/bnd_aopt_equivalence_panama_20260809/bnd_aopt_panama_seed42_k2_step0_to_15000/"
            "water-splatting/20260809_bnd_aopt_k2/config.yml"
        ),
        parameterization="bounded_sh3",
        appearance_lr_scale=2.0,
        reused=False,
    ),
    "K4": RunSpec(
        name="K4",
        config_relpath=(
            "outputs/bnd_aopt_equivalence_panama_20260809/bnd_aopt_panama_seed42_k4_step0_to_15000/"
            "water-splatting/20260809_bnd_aopt_k4/config.yml"
        ),
        parameterization="bounded_sh3",
        appearance_lr_scale=4.0,
        reused=False,
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
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if value == value]
    return float(sum(values) / len(values)) if values else float("nan")


def _quantile(values: torch.Tensor, q: float) -> float:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return float("nan")
    if q <= 0:
        return float(flat.min().item())
    if q >= 1:
        return float(flat.max().item())
    k = max(1, min(flat.numel(), int(math.ceil(q * flat.numel()))))
    return float(torch.kthvalue(flat, k).values.item())


def _stats(values: torch.Tensor, prefix: str) -> Dict[str, float]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {f"{prefix}{key}": float("nan") for key in ("mean", "p01", "p05", "p10", "p50", "p90", "p95", "p99", "max")}
    return {
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}p01": _quantile(flat, 0.01),
        f"{prefix}p05": _quantile(flat, 0.05),
        f"{prefix}p10": _quantile(flat, 0.10),
        f"{prefix}p50": _quantile(flat, 0.50),
        f"{prefix}p90": _quantile(flat, 0.90),
        f"{prefix}p95": _quantile(flat, 0.95),
        f"{prefix}p99": _quantile(flat, 0.99),
        f"{prefix}max": float(flat.max().item()),
    }


def _channel_stats(values: torch.Tensor, prefix: str) -> Dict[str, float]:
    values = values.detach().float()
    out: Dict[str, float] = {}
    if values.ndim > 0 and values.shape[-1] == 3:
        for idx, channel in enumerate(CHANNELS):
            out.update(_stats(values[..., idx], f"{prefix}_{channel}_"))
        out.update(_stats(values.reshape(-1), f"{prefix}_all_"))
    else:
        out.update(_stats(values.reshape(-1), f"{prefix}_"))
    return out


def _thresholds(values: torch.Tensor, prefix: str, thresholds: Sequence[float], op: str) -> Dict[str, float]:
    flat = values.detach().float().reshape(-1)
    out: Dict[str, float] = {}
    for threshold in thresholds:
        if flat.numel() == 0:
            value = float("nan")
        elif op == "lt":
            value = float((flat < threshold).float().mean().item())
        elif op == "gt":
            value = float((flat > threshold).float().mean().item())
        elif op == "abs_gt":
            value = float((flat.abs() > threshold).float().mean().item())
        else:
            raise ValueError(op)
        out[f"{prefix}_{op}_{threshold:g}"] = value
    return out


def _available_steps(config_path: Path) -> Dict[int, Path]:
    ckpt_dir = config_path.parent / "nerfstudio_models"
    if not ckpt_dir.exists():
        return {}
    out: Dict[int, Path] = {}
    for path in ckpt_dir.glob("step-*.ckpt"):
        try:
            step = int(path.stem.split("-")[1])
        except ValueError:
            continue
        out[step] = path
    return out


def _actual_step(config_path: Path, nominal_step: int) -> Optional[int]:
    available = _available_steps(config_path)
    if nominal_step in available:
        return nominal_step
    if nominal_step == 15000 and 14999 in available:
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

    config, pipeline, checkpoint_path, loaded_step = eval_setup(
        config_path,
        test_mode="test",
        update_config_callback=update_config,
    )
    pipeline.model.config.intrinsic_color_parameterization = spec.parameterization
    pipeline.eval()
    return LoadedRun(
        run=run,
        nominal_step=nominal_step,
        loaded_step=loaded_step,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        config=config,
        pipeline=pipeline,
    )


def _release_loaded(loaded: Optional[LoadedRun]) -> None:
    if loaded is not None:
        try:
            del loaded.pipeline
        except Exception:
            pass
    torch.cuda.empty_cache()


def _view_records(loaded: LoadedRun) -> List[Tuple[int, str, Any, Mapping[str, Any]]]:
    dataset = loaded.pipeline.datamanager.eval_dataset
    image_filenames = list(getattr(dataset, "image_filenames", []))
    rows = []
    for eval_index, (camera, batch) in enumerate(loaded.pipeline.datamanager.fixed_indices_eval_dataloader):
        filename = image_filenames[eval_index] if eval_index < len(image_filenames) else Path(f"eval_{eval_index}")
        rows.append((eval_index, Path(filename).stem, camera, batch))
    return rows


def _metric_images(model: Any, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    pred = pred.detach().float().clamp(0.0, 1.0)
    gt = gt.detach().float().clamp(0.0, 1.0)
    pred_nchw = torch.moveaxis(pred, -1, 0)[None, ...]
    gt_nchw = torch.moveaxis(gt, -1, 0)[None, ...]
    return {
        "psnr": float(model.psnr(gt_nchw, pred_nchw).item()),
        "ssim": float(model.ssim(gt_nchw, pred_nchw).item()),
        "lpips": float(model.lpips(gt_nchw, pred_nchw).item()),
    }


def _safe_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu()


def _eval_view(model: Any, camera: Any, batch: Mapping[str, Any]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, float]]:
    with torch.no_grad():
        outputs = model.get_outputs_for_camera(camera)
        gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
        metrics = _metric_images(model, outputs["pred_image"], gt)
    return outputs, gt, metrics


def _rgb_to_uint8(image: torch.Tensor) -> Image.Image:
    arr = (image.detach().float().clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _scalar_to_uint8(values: torch.Tensor, scale: float) -> Image.Image:
    scale = max(float(scale), 1e-8)
    arr = (values.detach().float().clamp_min(0.0) / scale).clamp(0.0, 1.0)
    arr = (arr * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="L").convert("RGB")


def _mask_to_uint8(mask: torch.Tensor) -> Image.Image:
    arr = (mask.detach().bool().cpu().numpy().astype("uint8") * 255)
    return Image.fromarray(arr, mode="L").convert("RGB")


def _tile(image: Image.Image, label: str, width: int) -> Image.Image:
    if width > 0 and image.width != width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    label_height = 28
    out = Image.new("RGB", (image.width, image.height + label_height), "white")
    out.paste(image, (0, label_height))
    ImageDraw.Draw(out).text((6, 7), label, fill="black")
    return out


def _save_sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Image.Image]]], tile_width: int, manifest: List[Dict[str, Any]], output_type: str, view_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered_rows: List[Image.Image] = []
    for row in rows:
        tiles = [_tile(image, label, tile_width) for label, image in row]
        width = sum(tile.width for tile in tiles) + 6 * (len(tiles) - 1)
        height = max(tile.height for tile in tiles)
        canvas = Image.new("RGB", (width, height), "white")
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, 0))
            x += tile.width + 6
        rendered_rows.append(canvas)
    if not rendered_rows:
        return
    sheet_width = max(row.width for row in rendered_rows)
    sheet_height = sum(row.height for row in rendered_rows) + 6 * (len(rendered_rows) - 1)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    y = 0
    for row in rendered_rows:
        sheet.paste(row, (0, y))
        y += row.height + 6
    sheet.save(path)
    manifest.append(
        {
            "file_path": str(path),
            "scene": SCENE,
            "run": "M1;K1;K2;K4",
            "step": FINAL_STEP,
            "output_type": output_type,
            "view_ids": ";".join(view_ids),
            "width": sheet.width,
            "height": sheet.height,
        }
    )


def _aggregate_dicts(rows: Sequence[Mapping[str, Any]], keys: Optional[Sequence[str]] = None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if keys is None:
        keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (float, int))})
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and isinstance(row[key], (float, int)) and float(row[key]) == float(row[key])]
        if vals:
            out[key] = _mean(vals)
    return out


def _append_output_stats(row: Dict[str, Any], outputs: Mapping[str, torch.Tensor]) -> None:
    beta = outputs["medium_attn"]
    beta_eff = beta
    tau = outputs["tau_D"]
    transmission = outputs["transmission"]
    clear = outputs["clear_object_fullsh_raw"]
    row.update(_channel_stats(beta, "beta_D_raw"))
    row.update(_channel_stats(beta_eff, "beta_D_effective"))
    row.update(_channel_stats(tau, "tau_D"))
    row.update(_channel_stats(transmission, "T_D"))
    row.update(_thresholds(transmission, "T_D", (0.30, 0.20, 0.10, 0.05), "lt"))
    row.update(_channel_stats(clear, "J"))
    row.update(_thresholds(clear, "J", (0.95, 0.99, 1.0), "gt"))
    for key, label in (
        ("medium_rgb", "medium_rgb"),
        ("medium_bs", "beta_B"),
        ("rgb_medium", "rgb_medium"),
    ):
        if key in outputs and isinstance(outputs[key], torch.Tensor):
            row.update(_channel_stats(outputs[key], label))


def _boundary_stats(outputs: Mapping[str, torch.Tensor], run: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"run": run}
    if run == "M1":
        row["boundary_stats_available"] = False
        return row
    visible = outputs.get("gaussian_visible_mask")
    c = outputs.get("gaussian_view_rgb")
    logits = outputs.get("gaussian_view_logits")
    deriv = outputs.get("gaussian_sigmoid_derivative")
    if not isinstance(c, torch.Tensor) or not isinstance(logits, torch.Tensor):
        row["boundary_stats_available"] = False
        return row
    if isinstance(visible, torch.Tensor) and visible.numel() == c.shape[0]:
        mask = visible.detach().bool().to(c.device)
        c = c[mask]
        logits = logits[mask]
        if isinstance(deriv, torch.Tensor):
            deriv = deriv[mask]
    row["boundary_stats_available"] = True
    row.update(_channel_stats(c, "c"))
    row.update(_thresholds(c, "c", (0.01, 0.05), "lt"))
    row.update(_thresholds(c, "c", (0.95, 0.99), "gt"))
    row["saturation_mass"] = row.get("c_lt_0.01", float("nan")) + row.get("c_gt_0.99", float("nan"))
    row.update(_channel_stats(logits, "s"))
    row.update(_thresholds(logits, "s_abs", (5.0, 8.0, 10.0), "abs_gt"))
    row.update(_thresholds(logits, "s", (4.595,), "gt"))
    row.update(_thresholds(logits, "s", (-4.595,), "lt"))
    if isinstance(deriv, torch.Tensor):
        row.update(_channel_stats(deriv, "sigmoid_derivative"))
    row["BOUNDARY_ESCAPE"] = bool(row.get("c_gt_0.99", 0.0) > 0.05 or row.get("s_abs_abs_gt_5", 0.0) > 0.05)
    return row


def _feature_stats(model: Any, run: str, nominal_step: int) -> Dict[str, Any]:
    with torch.no_grad():
        rest = model.features_rest.detach().float().reshape(model.features_rest.shape[0], -1)
        dc = model.features_dc.detach().float()
        rest_norm = torch.linalg.norm(rest, dim=-1)
        dc_norm = torch.linalg.norm(dc, dim=-1)
        ratio = rest_norm / dc_norm.clamp_min(1e-8)
    row: Dict[str, Any] = {
        "scene": SCENE,
        "run": run,
        "nominal_step": nominal_step,
        "gaussian_count": int(model.num_points),
    }
    row.update(_stats(rest_norm, "features_rest_norm_"))
    row.update(_stats(dc_norm, "features_dc_norm_"))
    row.update(_stats(ratio, "features_rest_dc_ratio_"))
    return row


def _sh_capacity(outputs: Mapping[str, torch.Tensor], model: Any, run: str, view_id: str) -> Dict[str, Any]:
    full = outputs["gaussian_view_rgb"].detach().float()
    visible = outputs.get("gaussian_visible_mask")
    if run == "M1":
        dc_rgb = torch.clamp(SH2RGB(model.features_dc.detach().float()), min=0.0).to(full.device)
    else:
        dc_rgb = outputs.get("gaussian_view_dc_rgb")
        if not isinstance(dc_rgb, torch.Tensor):
            dc_rgb = SHLogits2RGB(model.features_dc.detach().float()).to(full.device)
        else:
            dc_rgb = dc_rgb.detach().float().to(full.device)
    residual = torch.linalg.norm(full - dc_rgb, dim=-1)
    row: Dict[str, Any] = {"scene": SCENE, "run": run, "view_id": view_id}
    row.update(_stats(residual, "R_SH_all_"))
    if isinstance(visible, torch.Tensor) and visible.numel() == residual.numel():
        visible_mask = visible.detach().bool().to(residual.device)
        row.update(_stats(residual[visible_mask], "R_SH_visible_"))
        row["visible_fraction"] = float(visible_mask.float().mean().item())
    else:
        row["visible_fraction"] = float("nan")
    return row


def _cache_final_outputs(repo: Path, run: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    loaded: Optional[LoadedRun] = None
    try:
        loaded = _load_run(repo, run, FINAL_STEP)
        model = loaded.model
        rows: List[Dict[str, Any]] = []
        for eval_index, view_id, camera, batch in _view_records(loaded):
            outputs, gt, metrics = _eval_view(model, camera, batch)
            item: Dict[str, Any] = {
                "eval_index": eval_index,
                "view_id": view_id,
                "metrics": metrics,
                "gt": _safe_cpu(gt),
                "sh_capacity": _sh_capacity(outputs, model, run, view_id),
            }
            for key in (
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
            ):
                if key in outputs and isinstance(outputs[key], torch.Tensor):
                    item[key] = _safe_cpu(outputs[key])
            if run != "M1":
                for key in ("gaussian_view_rgb", "gaussian_view_logits", "gaussian_sigmoid_derivative", "gaussian_visible_mask"):
                    if key in outputs and isinstance(outputs[key], torch.Tensor):
                        item[key] = _safe_cpu(outputs[key])
            rows.append(item)
        meta = {
            "scene": SCENE,
            "run": run,
            "nominal_step": FINAL_STEP,
            "loaded_step": loaded.loaded_step,
            "config_path": str(loaded.config_path),
            "checkpoint_path": str(loaded.checkpoint_path),
            "parameterization": RUNS[run].parameterization,
            "appearance_lr_scale": RUNS[run].appearance_lr_scale,
            "reused": RUNS[run].reused,
            "gaussian_count": int(model.num_points),
            "seed": getattr(getattr(loaded.config, "machine", None), "seed", ""),
            "sh_degree": getattr(model.config, "sh_degree", ""),
            "medium_context_mode": getattr(model.config, "medium_context_mode", ""),
            "b_inf_mode": getattr(model.config, "b_inf_mode", ""),
            "infinite_water_enabled": getattr(model.config, "infinite_water_enabled", ""),
        }
        return rows, meta
    finally:
        _release_loaded(loaded)


def trajectory_audit(repo: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    trajectory_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    sh_rows: List[Dict[str, Any]] = []
    for run in RUNS:
        config_path = repo / RUNS[run].config_relpath
        for nominal_step in TRAJECTORY_STEPS:
            actual_step = _actual_step(config_path, nominal_step)
            if actual_step is None:
                missing_rows.append(
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
            loaded: Optional[LoadedRun] = None
            try:
                loaded = _load_run(repo, run, nominal_step)
                model = loaded.model
                metric_rows: List[Dict[str, float]] = []
                component_rows: List[Dict[str, Any]] = []
                boundary_view_rows: List[Dict[str, Any]] = []
                sh_view_rows: List[Dict[str, Any]] = []
                for _, view_id, camera, batch in _view_records(loaded):
                    outputs, _, metrics = _eval_view(model, camera, batch)
                    metric_rows.append(metrics)
                    comp: Dict[str, Any] = {}
                    _append_output_stats(comp, outputs)
                    component_rows.append(comp)
                    boundary = _boundary_stats(outputs, run)
                    boundary.update({"scene": SCENE, "nominal_step": nominal_step, "view_id": view_id})
                    boundary_view_rows.append(boundary)
                    sh_view_rows.append(_sh_capacity(outputs, model, run, view_id))
                row: Dict[str, Any] = {
                    "scene": SCENE,
                    "run": run,
                    "nominal_step": nominal_step,
                    "loaded_step": loaded.loaded_step,
                    "checkpoint_path": str(loaded.checkpoint_path),
                    "config_path": str(loaded.config_path),
                    "parameterization": RUNS[run].parameterization,
                    "appearance_lr_scale": RUNS[run].appearance_lr_scale,
                    "num_eval_views": len(metric_rows),
                    "gaussian_count": int(model.num_points),
                }
                row.update(_aggregate_dicts(metric_rows, ("psnr", "ssim", "lpips")))
                row.update(_aggregate_dicts(component_rows))
                trajectory_rows.append(row)
                feature_rows.append(_feature_stats(model, run, nominal_step))
                boundary_agg = {"scene": SCENE, "run": run, "nominal_step": nominal_step, "view_id": "AGGREGATE"}
                boundary_agg.update(_aggregate_dicts(boundary_view_rows))
                if run != "M1":
                    boundary_agg["BOUNDARY_ESCAPE"] = bool(
                        boundary_agg.get("c_gt_0.99", 0.0) > 0.05
                        or boundary_agg.get("s_abs_abs_gt_5", 0.0) > 0.05
                    )
                else:
                    boundary_agg["BOUNDARY_ESCAPE"] = False
                boundary_rows.append(boundary_agg)
                sh_agg = {"scene": SCENE, "run": run, "nominal_step": nominal_step, "view_id": "AGGREGATE"}
                sh_agg.update(_aggregate_dicts(sh_view_rows))
                sh_rows.append(sh_agg)
            finally:
                _release_loaded(loaded)
    return trajectory_rows, missing_rows, feature_rows, boundary_rows, sh_rows


def final_view_audit(repo: Path, render_dir: Path, tile_width: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    cached: Dict[str, List[Dict[str, Any]]] = {}
    checkpoint_rows: List[Dict[str, Any]] = []
    for run in RUNS:
        rows, meta = _cache_final_outputs(repo, run)
        cached[run] = rows
        checkpoint_rows.append(meta)

    view_ids = [row["view_id"] for row in cached["M1"]]
    for run in RUNS:
        if [row["view_id"] for row in cached[run]] != view_ids:
            raise RuntimeError(f"View mismatch for {run}: {[row['view_id'] for row in cached[run]]} vs {view_ids}")

    by_run_view = {run: {row["view_id"]: row for row in rows} for run, rows in cached.items()}
    per_view_rows: List[Dict[str, Any]] = []
    residual_rows: List[Dict[str, Any]] = []
    sh_rows: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []

    residual_rgb_max = 1.0
    tau_scale = 1.0
    for view_id in view_ids:
        gt = by_run_view["M1"][view_id]["gt"]
        for run in RUNS:
            item = by_run_view[run][view_id]
            residual_rgb_max = max(residual_rgb_max, float((item["pred_image"] - gt).abs().max().item()))
            tau_scale = max(tau_scale, float(item["tau_D"].mean(dim=-1).max().item()))

    for view_id in view_ids:
        gt = by_run_view["M1"][view_id]["gt"]
        m1 = by_run_view["M1"][view_id]
        masks = _m1_masks(m1)
        m1_residual = torch.linalg.norm(m1["pred_image"] - gt, dim=-1)
        for run in RUNS:
            item = by_run_view[run][view_id]
            row = {
                "scene": SCENE,
                "view_id": view_id,
                "run": run,
                "psnr": item["metrics"]["psnr"],
                "ssim": item["metrics"]["ssim"],
                "lpips": item["metrics"]["lpips"],
            }
            if run != "M1":
                row["delta_psnr_vs_M1"] = item["metrics"]["psnr"] - by_run_view["M1"][view_id]["metrics"]["psnr"]
                row["delta_psnr_vs_K1"] = item["metrics"]["psnr"] - by_run_view["K1"][view_id]["metrics"]["psnr"]
            per_view_rows.append(row)
            sh_rows.append(dict(item["sh_capacity"]))
            if run != "M1":
                residual = torch.linalg.norm(item["pred_image"] - gt, dim=-1)
                excess = (residual - m1_residual).clamp_min(0.0)
                denom = float(excess.sum().item())
                for mask_name in ("J1", "COMP"):
                    mask = masks[mask_name]
                    area = float(mask.float().mean().item())
                    efrac = float(excess[mask].sum().item() / denom) if denom > 1e-12 else 0.0
                    residual_rows.append(
                        {
                            "scene": SCENE,
                            "view_id": view_id,
                            "run": run,
                            "mask": mask_name,
                            "mask_area": area,
                            "positive_excess_residual_fraction": efrac,
                            "residual_enrichment": efrac / area if area > 1e-12 else float("inf"),
                        }
                    )

    _write_visuals(render_dir, by_run_view, view_ids, tau_scale, residual_rgb_max, tile_width, manifest)
    return per_view_rows, residual_rows, sh_rows, checkpoint_rows, manifest


def _m1_masks(m1: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    j_scalar = m1["clear_object_fullsh_raw"].float().amax(dim=-1)
    tau_scalar = m1["tau_D"].float().mean(dim=-1)
    t_scalar = m1["transmission"].float().amin(dim=-1)
    tau90 = torch.quantile(tau_scalar.reshape(-1), 0.90)
    masks = {
        "J1": j_scalar > 1.0,
        "TAU90": tau_scalar >= tau90,
        "TLOW": t_scalar < 0.1,
    }
    masks["COMP"] = masks["J1"] | masks["TAU90"] | masks["TLOW"]
    return masks


def _write_visuals(
    render_dir: Path,
    by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]],
    view_ids: Sequence[str],
    tau_scale: float,
    residual_rgb_max: float,
    tile_width: int,
    manifest: List[Dict[str, Any]],
) -> None:
    rows_underwater: List[List[Tuple[str, Image.Image]]] = []
    rows_clear: List[List[Tuple[str, Image.Image]]] = []
    rows_direct: List[List[Tuple[str, Image.Image]]] = []
    rows_trans: List[List[Tuple[str, Image.Image]]] = []
    rows_tau: List[List[Tuple[str, Image.Image]]] = []
    rows_resid: List[List[Tuple[str, Image.Image]]] = []
    rows_sat: List[List[Tuple[str, Image.Image]]] = []
    for view_id in view_ids:
        gt = by_run_view["M1"][view_id]["gt"]
        rows_underwater.append(
            [(f"{view_id} GT", _rgb_to_uint8(gt))]
            + [(run, _rgb_to_uint8(by_run_view[run][view_id]["pred_image"])) for run in RUNS]
        )
        rows_clear.append(
            [(f"{view_id} {run}", _rgb_to_uint8(by_run_view[run][view_id]["clear_object_fullsh_raw"])) for run in RUNS]
        )
        rows_direct.append(
            [(f"{view_id} {run}", _rgb_to_uint8(by_run_view[run][view_id]["direct_object_signal"])) for run in RUNS]
        )
        rows_trans.append(
            [(f"{view_id} {run}", _rgb_to_uint8(by_run_view[run][view_id]["transmission"])) for run in RUNS]
        )
        rows_tau.append(
            [
                (f"{view_id} {run}", _scalar_to_uint8(by_run_view[run][view_id]["tau_D"].mean(dim=-1), tau_scale))
                for run in RUNS
            ]
        )
        rows_resid.append(
            [(f"{view_id} GT", _rgb_to_uint8(gt))]
            + [
                (
                    f"{run} abs residual",
                    _rgb_to_uint8(((by_run_view[run][view_id]["pred_image"] - gt).abs()) / max(residual_rgb_max, 1e-8)),
                )
                for run in RUNS
            ]
        )
        rows_sat.append(
            [
                (
                    f"{view_id} {run} J/c>0.99",
                    _mask_to_uint8(by_run_view[run][view_id]["clear_object_fullsh_raw"].amax(dim=-1) > 0.99),
                )
                for run in ("K1", "K2", "K4")
            ]
        )

    for filename, rows, output_type in (
        ("contact_sheet_underwater_m1_k1_k2_k4.png", rows_underwater, "underwater"),
        ("contact_sheet_clear_raw_m1_k1_k2_k4.png", rows_clear, "clear_object_fullsh_raw_display_clamp01"),
        ("contact_sheet_direct_object_signal_m1_k1_k2_k4.png", rows_direct, "direct_object_signal"),
        ("contact_sheet_transmission_m1_k1_k2_k4.png", rows_trans, "transmission"),
        ("contact_sheet_tau_d_m1_k1_k2_k4.png", rows_tau, "tau_D_common_scale"),
        ("contact_sheet_underwater_abs_residual_m1_k1_k2_k4.png", rows_resid, "underwater_abs_rgb_residual_common_scale"),
        ("contact_sheet_saturation_mask_k1_k2_k4.png", rows_sat, "J_or_c_gt_0.99_mask"),
    ):
        _save_sheet(render_dir / filename, rows, tile_width, manifest, output_type, view_ids)


def lr_and_update_audit(repo: Path, logs_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    lr_rows: List[Dict[str, Any]] = []
    update_rows: List[Dict[str, Any]] = []
    k1_steps = (0, 1, 1000, 3000, 5000, 8000, 10000, 13000, 14999)
    for step in k1_steps:
        lr_rows.append(
            {
                "scene": SCENE,
                "run": "K1",
                "step": step,
                "appearance_lr_scale": 1.0,
                "features_dc_lr": 0.0025,
                "features_rest_lr": 0.000125,
                "source": "synthetic_reference_from_unscaled_constant_appearance_scheduler",
            }
        )
    for run, tag in (("K2", "k2"), ("K4", "k4")):
        run_log_dir = logs_dir / f"bnd_aopt_panama_seed42_{tag}_step0_to_15000_20260809_bnd_aopt_{tag}"
        for row in _read_jsonl(run_log_dir / "aopt_lr_trajectory.jsonl"):
            item = {"scene": SCENE, "run": run, "source": str(run_log_dir / "aopt_lr_trajectory.jsonl")}
            item.update(row)
            lr_rows.append(item)
        for row in _read_jsonl(run_log_dir / "aopt_parameter_updates.jsonl"):
            item = {"scene": SCENE, "run": run, "source": str(run_log_dir / "aopt_parameter_updates.jsonl")}
            item.update(row)
            update_rows.append(item)
    update_rows.append(
        {
            "scene": SCENE,
            "run": "K1",
            "status": "unavailable",
            "reason": "historical reused K1 checkpoint was trained before AOPT update-audit JSONL logging existed",
        }
    )
    return lr_rows, update_rows


def final_summary(
    trajectory_rows: Sequence[Mapping[str, Any]],
    boundary_rows: Sequence[Mapping[str, Any]],
    sh_rows: Sequence[Mapping[str, Any]],
    per_view_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    final = {row["run"]: row for row in trajectory_rows if int(row["nominal_step"]) == FINAL_STEP}
    boundary = {row["run"]: row for row in boundary_rows if int(row["nominal_step"]) == FINAL_STEP}
    sh = {row["run"]: row for row in sh_rows if int(row["nominal_step"]) == FINAL_STEP}
    m1 = final["M1"]
    k1 = final["K1"]
    m1_gap = float(m1["psnr"]) - float(k1["psnr"])
    tau_gain = float(m1["tau_D_all_p90"]) - float(k1["tau_D_all_p90"])
    j_gain = float(m1["J_all_p99"]) - float(k1["J_all_p99"])
    r_m1 = float(sh["M1"].get("R_SH_visible_p50", float("nan")))
    r_k1 = float(sh["K1"].get("R_SH_visible_p50", float("nan")))
    rows = []
    for run in ("M1", "K1", "K2", "K4"):
        item = final[run]
        row: Dict[str, Any] = {
            "scene": SCENE,
            "run": run,
            "appearance_lr_scale": RUNS[run].appearance_lr_scale,
            "PSNR": item["psnr"],
            "SSIM": item["ssim"],
            "LPIPS": item["lpips"],
            "delta_PSNR_vs_M1": float(item["psnr"]) - float(m1["psnr"]),
            "delta_SSIM_vs_M1": float(item["ssim"]) - float(m1["ssim"]),
            "delta_LPIPS_vs_M1": float(item["lpips"]) - float(m1["lpips"]),
            "RGB_SAFETY_PASS": bool(
                float(item["psnr"]) - float(m1["psnr"]) >= -0.15
                and float(item["ssim"]) - float(m1["ssim"]) >= -0.0015
                and float(item["lpips"]) - float(m1["lpips"]) <= 0.003
            ),
            "beta_D_raw_mean": item.get("beta_D_raw_all_mean", float("nan")),
            "beta_D_effective_mean": item.get("beta_D_effective_all_mean", float("nan")),
            "tau_p90": item.get("tau_D_all_p90", float("nan")),
            "T_mean": item.get("T_D_all_mean", float("nan")),
            "P_T_lt_0.1": item.get("T_D_lt_0.1", float("nan")),
            "P_T_lt_0.05": item.get("T_D_lt_0.05", float("nan")),
            "J_p99": item.get("J_all_p99", float("nan")),
            "P_J_gt_1": item.get("J_gt_1", float("nan")),
            "gaussian_count": item.get("gaussian_count", ""),
            "R_SH_visible_p50": sh.get(run, {}).get("R_SH_visible_p50", float("nan")),
            "SH_CAPACITY_RATIO_vs_M1": float(sh.get(run, {}).get("R_SH_visible_p50", float("nan"))) / max(r_m1, 1e-12),
            "SH_RECOVERY_OVER_K1": float(sh.get(run, {}).get("R_SH_visible_p50", float("nan"))) / max(r_k1, 1e-12),
            "BOUNDARY_ESCAPE": bool(boundary.get(run, {}).get("BOUNDARY_ESCAPE", False)),
        }
        if run in ("K2", "K4"):
            row["RGB_GAIN_OVER_K1"] = float(item["psnr"]) - float(k1["psnr"])
            row["GAP_RECOVERY_FRACTION"] = (float(item["psnr"]) - float(k1["psnr"])) / max(m1_gap, 1e-12)
            row["TAU_BENEFIT_RETENTION"] = (float(m1["tau_D_all_p90"]) - float(item["tau_D_all_p90"])) / max(tau_gain, 1e-12)
            row["J_P99_BENEFIT_RETENTION"] = (float(m1["J_all_p99"]) - float(item["J_all_p99"])) / max(j_gain, 1e-12)
            row["STRONG_PARAMETERIZATION_RECOVERY"] = bool(
                row["GAP_RECOVERY_FRACTION"] >= 0.75
                and row["RGB_SAFETY_PASS"]
                and row["TAU_BENEFIT_RETENTION"] >= 0.75
                and not row["BOUNDARY_ESCAPE"]
                and row["SH_CAPACITY_RATIO_vs_M1"] >= 1.25 * (r_k1 / max(r_m1, 1e-12))
            )
            row["PARTIAL_PARAMETERIZATION_RECOVERY"] = bool(
                row["GAP_RECOVERY_FRACTION"] >= 0.30
                and row["TAU_BENEFIT_RETENTION"] >= 0.75
                and not row["BOUNDARY_ESCAPE"]
                and row["SH_RECOVERY_OVER_K1"] >= 1.15
            )
            row["OPTIMIZER_OVERDRIVE"] = bool(
                (float(item["psnr"]) - float(k1["psnr"]) > 0.0)
                and (row["TAU_BENEFIT_RETENTION"] < 0.50 or row["BOUNDARY_ESCAPE"])
            )
        rows.append(row)

    candidate_rows = [row for row in rows if row["run"] in ("K2", "K4")]
    no_recovery = all(
        float(row.get("RGB_GAIN_OVER_K1", 0.0)) < 0.10 and float(row.get("SH_RECOVERY_OVER_K1", 1.0)) < 1.10
        for row in candidate_rows
    )
    for row in candidate_rows:
        row["NO_PARAMETERIZATION_RECOVERY_GLOBAL"] = bool(no_recovery)

    by_run = {row["run"]: row for row in rows}
    if "K4" in by_run and "K2" in by_run and "K1" in by_run:
        k4_over_optimized = bool(
            float(by_run["K4"]["PSNR"]) < float(by_run["K2"]["PSNR"])
            or float(by_run["K4"]["PSNR"]) < float(by_run["K1"]["PSNR"])
            or float(by_run["K4"]["SH_RECOVERY_OVER_K1"]) > float(by_run["K2"]["SH_RECOVERY_OVER_K1"]) * 1.20
        )
        by_run["K4"]["K4_OVER_OPTIMIZED"] = k4_over_optimized
        by_run["K2"]["K4_OVER_OPTIMIZED"] = False

    safe_candidates = [
        row
        for row in candidate_rows
        if (not row["BOUNDARY_ESCAPE"]) and float(row.get("TAU_BENEFIT_RETENTION", -1.0)) >= 0.75
    ]
    best = max(safe_candidates, key=lambda row: (float(row["PSNR"]), float(row["SH_CAPACITY_RATIO_vs_M1"]))) if safe_candidates else None
    for row in rows:
        row["BEST_SAFE_AOPT_CANDIDATE"] = bool(best is not None and row["run"] == best["run"])

    view_counts: Dict[str, Dict[str, int]] = {}
    for run in ("K2", "K4"):
        view_rows = [row for row in per_view_rows if row.get("run") == run]
        view_counts[run] = {
            "views_psnr_improved_vs_K1": sum(1 for row in view_rows if float(row.get("delta_psnr_vs_K1", 0.0)) > 0.0),
            "views_psnr_declined_vs_K1": sum(1 for row in view_rows if float(row.get("delta_psnr_vs_K1", 0.0)) < 0.0),
        }
    for row in rows:
        row.update(view_counts.get(row["run"], {}))
    return rows


def write_visual_index(render_dir: Path, manifest: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Panama BND-AOPT Visual Compare Index", ""]
    for row in manifest:
        lines.append(f"- {row.get('output_type')}: `{row.get('file_path')}`")
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/mnt/new/home_old/ycy/water-splatting-refactor"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bnd_aopt_equivalence_panama_20260809"))
    parser.add_argument("--render-dir", type=Path, default=Path("renders/bnd_aopt_equivalence_panama_20260809"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/bnd_aopt_equivalence_panama_20260809"))
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

    trajectory_rows, missing_rows, feature_rows, boundary_rows, sh_traj_rows = trajectory_audit(repo)
    per_view_rows, residual_rows, sh_final_rows, checkpoint_rows, manifest_rows = final_view_audit(
        repo, render_dir, args.tile_width
    )
    lr_rows, update_rows = lr_and_update_audit(repo, logs_dir)
    summary_rows = final_summary(trajectory_rows, boundary_rows, sh_traj_rows, per_view_rows)

    rgb_rows = [
        {
            "scene": row["scene"],
            "run": row["run"],
            "nominal_step": row["nominal_step"],
            "loaded_step": row["loaded_step"],
            "psnr": row.get("psnr", float("nan")),
            "ssim": row.get("ssim", float("nan")),
            "lpips": row.get("lpips", float("nan")),
        }
        for row in trajectory_rows
    ]
    decomp_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            in {
                "scene",
                "run",
                "nominal_step",
                "loaded_step",
                "gaussian_count",
                "beta_D_raw_all_mean",
                "beta_D_effective_all_mean",
                "tau_D_all_mean",
                "tau_D_all_p50",
                "tau_D_all_p90",
                "tau_D_all_p95",
                "tau_D_all_p99",
                "T_D_all_mean",
                "T_D_all_p01",
                "T_D_all_p05",
                "T_D_all_p10",
                "T_D_all_p50",
                "T_D_all_p90",
                "T_D_lt_0.3",
                "T_D_lt_0.2",
                "T_D_lt_0.1",
                "T_D_lt_0.05",
                "J_all_mean",
                "J_all_p90",
                "J_all_p95",
                "J_all_p99",
                "J_all_max",
                "J_gt_0.95",
                "J_gt_0.99",
                "J_gt_1",
            }
        }
        for row in trajectory_rows
    ]
    medium_rows = [
        {
            key: value
            for key, value in row.items()
            if key.startswith(("medium_rgb_", "beta_B_", "rgb_medium_")) or key in {"scene", "run", "nominal_step", "loaded_step"}
        }
        for row in trajectory_rows
    ]

    _write_json(output_dir / "aopt_training_trajectory.json", {"rows": trajectory_rows, "missing": missing_rows})
    _write_csv(output_dir / "aopt_training_trajectory.csv", trajectory_rows)
    _write_csv(output_dir / "aopt_missing_checkpoints.csv", missing_rows)
    _write_json(output_dir / "aopt_rgb_metrics.json", rgb_rows)
    _write_csv(output_dir / "aopt_rgb_metrics.csv", rgb_rows)
    _write_json(output_dir / "aopt_decomposition_metrics.json", decomp_rows)
    _write_csv(output_dir / "aopt_decomposition_metrics.csv", decomp_rows)
    _write_json(output_dir / "aopt_sh_capacity.json", {"trajectory": sh_traj_rows, "final_per_view": sh_final_rows})
    _write_csv(output_dir / "aopt_sh_capacity.csv", sh_traj_rows)
    _write_csv(output_dir / "aopt_sh_capacity_final_per_view.csv", sh_final_rows)
    _write_json(output_dir / "aopt_boundary_saturation.json", boundary_rows)
    _write_csv(output_dir / "aopt_boundary_saturation.csv", boundary_rows)
    _write_json(output_dir / "aopt_parameter_features.json", feature_rows)
    _write_csv(output_dir / "aopt_parameter_features.csv", feature_rows)
    _write_json(output_dir / "aopt_lr_trajectory.json", lr_rows)
    _write_csv(output_dir / "aopt_lr_trajectory.csv", lr_rows)
    _write_json(output_dir / "aopt_parameter_updates.json", update_rows)
    _write_csv(output_dir / "aopt_parameter_updates.csv", update_rows)
    _write_json(output_dir / "aopt_medium_redistribution.json", medium_rows)
    _write_csv(output_dir / "aopt_medium_redistribution.csv", medium_rows)
    _write_json(output_dir / "aopt_per_view_metrics.json", per_view_rows)
    _write_csv(output_dir / "aopt_per_view_metrics.csv", per_view_rows)
    _write_json(output_dir / "aopt_residual_enrichment.json", residual_rows)
    _write_csv(output_dir / "aopt_residual_enrichment.csv", residual_rows)
    _write_json(output_dir / "aopt_checkpoint_audit.json", checkpoint_rows)
    _write_csv(output_dir / "aopt_checkpoint_audit.csv", checkpoint_rows)
    _write_json(output_dir / "aopt_final_summary.json", summary_rows)
    _write_csv(output_dir / "aopt_final_summary.csv", summary_rows)
    _write_json(render_dir / "manifest.json", manifest_rows)
    _write_csv(render_dir / "manifest.csv", manifest_rows)
    write_visual_index(render_dir, manifest_rows)
    _write_json(
        output_dir / "manifest.json",
        {
            "scene": SCENE,
            "runs": {name: spec.__dict__ for name, spec in RUNS.items()},
            "metric_outputs": sorted(str(path) for path in output_dir.glob("aopt_*") if path.is_file()),
            "render_manifest": str(render_dir / "manifest.json"),
            "visual_index": str(render_dir / "VISUAL_COMPARE_INDEX.md"),
        },
    )
    print(json.dumps({"summary": summary_rows, "output_dir": str(output_dir), "render_dir": str(render_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
