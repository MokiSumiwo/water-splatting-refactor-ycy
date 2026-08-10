#!/usr/bin/env python
"""Read-only BND object-medium recomposition diagnostic."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor

from nerfstudio.utils.eval_utils import eval_setup


CHANNELS = ("r", "g", "b")
SCENES = ("Curasao", "JapaneseGradens", "IUI3", "Panama")
FINAL_NOMINAL_STEP = 15000
EPS = 1e-8
LUMA_WEIGHTS = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)


@dataclass(frozen=True)
class RunSpec:
    scene: str
    run: str
    config_relpath: str
    parameterization: str
    role: str = "primary"
    nominal_step: int = FINAL_NOMINAL_STEP
    appearance_lr_scale: float = 1.0


RUN_SPECS: Dict[Tuple[str, str], RunSpec] = {
    ("Curasao", "M1"): RunSpec(
        "Curasao",
        "M1",
        "outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/"
        "cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml",
        "legacy",
    ),
    ("Curasao", "BND-K1"): RunSpec(
        "Curasao",
        "BND-K1",
        "outputs/dewater_bounded_sh3_scratch_20260808/"
        "dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000/water-splatting/"
        "dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000_20260808_bounded_sh3_scratch_full_bnd-scratch_g1p00/"
        "config.yml",
        "bounded_sh3",
    ),
    ("JapaneseGradens", "M1"): RunSpec(
        "JapaneseGradens",
        "M1",
        "outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/"
        "cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml",
        "legacy",
    ),
    ("JapaneseGradens", "BND-K1"): RunSpec(
        "JapaneseGradens",
        "BND-K1",
        "outputs/dewater_bounded_sh3_cross_scene_20260808/"
        "dewater_bnd_cross_scene_japanesegradens_seed42_step0_to_15000/water-splatting/"
        "dewater_bnd_cross_scene_japanesegradens_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_japanesegradens_bnd_g1p00/"
        "config.yml",
        "bounded_sh3",
    ),
    ("IUI3", "M1"): RunSpec(
        "IUI3",
        "M1",
        "outputs/gmvc_v3_four_scene_iui3_m1_seed42_15000/water-splatting/"
        "gmvc_v3_four_scene_iui3_m1_seed42_15000_20260806_gmvc_four_scene_p30_mhold_15k_m1_bootstrap/config.yml",
        "legacy",
    ),
    ("IUI3", "BND-K1"): RunSpec(
        "IUI3",
        "BND-K1",
        "outputs/dewater_bounded_sh3_cross_scene_20260808/"
        "dewater_bnd_cross_scene_iui3_seed42_step0_to_15000/water-splatting/"
        "dewater_bnd_cross_scene_iui3_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_iui3_bnd_g1p00/"
        "config.yml",
        "bounded_sh3",
    ),
    ("Panama", "M1"): RunSpec(
        "Panama",
        "M1",
        "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/"
        "cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml",
        "legacy",
    ),
    ("Panama", "BND-K1"): RunSpec(
        "Panama",
        "BND-K1",
        "outputs/dewater_bounded_sh3_cross_scene_20260808/"
        "dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/"
        "dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/"
        "config.yml",
        "bounded_sh3",
    ),
    ("Panama", "K2"): RunSpec(
        "Panama",
        "K2",
        "outputs/bnd_aopt_equivalence_panama_20260809/bnd_aopt_panama_seed42_k2_step0_to_15000/"
        "water-splatting/20260809_bnd_aopt_k2/config.yml",
        "bounded_sh3",
        role="secondary",
        appearance_lr_scale=2.0,
    ),
    ("Panama", "K4"): RunSpec(
        "Panama",
        "K4",
        "outputs/bnd_aopt_equivalence_panama_20260809/bnd_aopt_panama_seed42_k4_step0_to_15000/"
        "water-splatting/20260809_bnd_aopt_k4/config.yml",
        "bounded_sh3",
        role="secondary",
        appearance_lr_scale=4.0,
    ),
}


@dataclass
class LoadedRun:
    spec: RunSpec
    config_path: Path
    checkpoint_path: Path
    loaded_step: int
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
    if q <= 0:
        return float(flat.min().item())
    if q >= 1:
        return float(flat.max().item())
    rank = max(1, min(flat.numel(), int(math.ceil(q * flat.numel()))))
    return float(torch.kthvalue(flat, rank).values.item())


def _stats(values: Tensor, prefix: str) -> Dict[str, float]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {f"{prefix}{k}": float("nan") for k in ("count", "mean", "mean_abs", "p05", "p10", "p50", "p90", "p95", "p99", "max")}
    return {
        f"{prefix}count": int(flat.numel()),
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}mean_abs": float(flat.abs().mean().item()),
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
    raise ValueError(op)


def _pearson_torch(a: Tensor, b: Tensor) -> float:
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


def _available_steps(config_path: Path) -> Dict[int, Path]:
    ckpt_dir = config_path.parent / "nerfstudio_models"
    if not ckpt_dir.exists():
        return {}
    out: Dict[int, Path] = {}
    for path in ckpt_dir.glob("step-*.ckpt"):
        try:
            out[int(path.stem.split("-")[1])] = path
        except Exception:
            continue
    return out


def _actual_step(config_path: Path, nominal_step: int) -> Optional[int]:
    steps = _available_steps(config_path)
    if nominal_step in steps:
        return nominal_step
    if nominal_step == 15000 and 14999 in steps:
        return 14999
    return None


def _load_run(repo: Path, spec: RunSpec) -> LoadedRun:
    config_path = repo / spec.config_relpath
    actual_step = _actual_step(config_path, spec.nominal_step)
    if actual_step is None:
        raise FileNotFoundError(f"missing checkpoint for {spec.scene} {spec.run}: {config_path}")

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
    return LoadedRun(spec, config_path, checkpoint_path, loaded_step, config, pipeline)


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


def _load_outputs(loaded: LoadedRun) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model = loaded.model
    items = []
    for eval_index, view_id, camera, batch in _view_records(loaded):
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera)
            gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
            metrics = _metric_images(model, outputs["pred_image"], gt)
        keep = (
            "pred_image",
            "direct_object_signal",
            "rgb_object",
            "rgb_medium",
            "rgb_medium_finite",
            "rgb_tail",
            "clear_object_fullsh_raw",
            "transmission",
            "tau_D",
            "depth",
            "accumulation",
            "medium_rgb",
            "medium_bs",
            "medium_attn",
            "b_inf",
        )
        tensors = {key: _safe_cpu(outputs[key]) for key in keep if key in outputs and isinstance(outputs[key], Tensor)}
        camera_center = camera.camera_to_worlds[0, :3, 3].detach().float().cpu()
        camera_dir = camera.camera_to_worlds[0, :3, 2].detach().float().cpu()
        items.append(
            {
                "eval_index": eval_index,
                "view_id": view_id,
                "camera_id": eval_index,
                "gt": _safe_cpu(gt),
                "outputs": tensors,
                "metrics": metrics,
                "camera_center_x": float(camera_center[0].item()),
                "camera_center_y": float(camera_center[1].item()),
                "camera_center_z": float(camera_center[2].item()),
                "camera_dir_x": float(camera_dir[0].item()),
                "camera_dir_y": float(camera_dir[1].item()),
                "camera_dir_z": float(camera_dir[2].item()),
            }
        )
    meta = {
        "scene": loaded.spec.scene,
        "run": loaded.spec.run,
        "role": loaded.spec.role,
        "config_path": str(loaded.config_path),
        "checkpoint_path": str(loaded.checkpoint_path),
        "nominal_step": loaded.spec.nominal_step,
        "loaded_step": int(loaded.loaded_step),
        "seed": getattr(getattr(loaded.config, "machine", None), "seed", ""),
        "intrinsic_color_parameterization": loaded.spec.parameterization,
        "sh_degree": getattr(model.config, "sh_degree", ""),
        "medium_context_mode": getattr(model.config, "medium_context_mode", ""),
        "b_inf_mode": getattr(model.config, "b_inf_mode", ""),
        "infinite_water_enabled": getattr(model.config, "infinite_water_enabled", ""),
        "appearance_lr_scale": loaded.spec.appearance_lr_scale,
        "gaussian_count": int(model.num_points),
        "num_eval_views": len(items),
        "view_ids": ";".join(item["view_id"] for item in items),
    }
    return items, meta


def _component_semantics() -> Dict[str, Any]:
    return {
        "code_facts": {
            "forward_path": "water_splatting/water_splatting.py::WaterSplattingModel.get_outputs",
            "renderer_path": "water_splatting/rendering/underwater_rasterizer.py::UnderwaterRasterizer.rasterize",
            "I_PRED": "outputs['pred_image'] / outputs['rgb']; after tied B_inf recomposition when b_inf_mode='tied'",
            "D_DIRECT": "outputs['direct_object_signal'] = outputs['rgb_object'] = render.rgb_object",
            "B_MEDIUM": "outputs['rgb_medium']; when b_inf_mode='tied', this equals finite renderer medium contribution with original medium tail removed and tail_weight * b_inf added",
            "B_MEDIUM_FINITE": "outputs['rgb_medium_finite']; available when tied B_inf branch recomposes medium",
            "B_TAIL": "outputs['rgb_tail']; tied B_inf tail term tail_weight * b_inf",
            "J_CLEAR": "outputs['clear_object_fullsh_raw'] = render.j_raw; image-space alpha-composited clear-object full-SH proxy, not clear GT",
            "T_DIRECT": "outputs['transmission'] = exp(-outputs['tau_D'].clamp_min(0)).clamp(0,1)",
            "TAU_DIRECT": "outputs['tau_D'] = outputs['medium_attn'] * render.depth",
            "closure": "For b_inf_mode='tied', code sets rgb = render.rgb_object + recomposed rgb_medium and returns pred_image=rgb.",
        },
        "component_namespace": {
            "I": "I_PRED",
            "D": "D_DIRECT",
            "B": "B_MEDIUM",
            "J": "J_CLEAR",
            "T": "T_DIRECT",
            "tau": "TAU_DIRECT",
        },
        "counterfactual_policy": {
            "direct_medium_hybrids": "IMAGE_SPACE_COUNTERFACTUALS after D+B closure audit passes",
            "JT_hybrids": "Require D_DIRECT vs J_CLEAR*T_DIRECT closure; otherwise unavailable",
            "cross_run_gaussian_color_swap": "not performed; from-scratch runs do not have Gaussian index correspondence",
        },
    }


def _object_mask(item: Mapping[str, Any]) -> Tensor:
    return item["outputs"]["accumulation"].float()[..., 0] > 0.01


def _luma(rgb: Tensor) -> Tensor:
    return (rgb.float() * LUMA_WEIGHTS).sum(dim=-1)


def _rgb_l1_per_pixel(rgb: Tensor) -> Tensor:
    return rgb.detach().float().abs().sum(dim=-1)


def _rgb_l2_per_pixel(rgb: Tensor) -> Tensor:
    return torch.linalg.norm(rgb.detach().float(), dim=-1)


def _masked(values: Tensor, mask: Tensor) -> Tensor:
    if values.ndim == mask.ndim:
        return values[mask]
    while mask.ndim < values.ndim:
        mask = mask[..., None].expand(*values.shape)
    return values[mask]


def _closure_rows(scene: str, run: str, items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in items:
        out = item["outputs"]
        err = (out["pred_image"] - (out["direct_object_signal"] + out["rgb_medium"])).abs()
        flat = err.reshape(-1)
        row = {
            "scene": scene,
            "run": run,
            "view_id": item["view_id"],
            "closure_definition": "pred_image - (direct_object_signal + rgb_medium)",
            "mean_abs": float(flat.mean().item()),
            "p95_abs": _safe_quantile(flat, 0.95),
            "p99_abs": _safe_quantile(flat, 0.99),
            "max_abs": float(flat.max().item()),
            "PASS": bool(float(flat.max().item()) < 1e-5),
        }
        rows.append(row)
    rows.append(
        {
            "scene": scene,
            "run": run,
            "view_id": "AGGREGATE",
            "closure_definition": "pred_image - (direct_object_signal + rgb_medium)",
            "mean_abs": _mean(row["mean_abs"] for row in rows),
            "p95_abs": max(row["p95_abs"] for row in rows),
            "p99_abs": max(row["p99_abs"] for row in rows),
            "max_abs": max(row["max_abs"] for row in rows),
            "PASS": all(bool(row["PASS"]) for row in rows),
        }
    )
    return rows


def _pair_by_view(m1_items: Sequence[Mapping[str, Any]], bnd_items: Sequence[Mapping[str, Any]]) -> List[Tuple[Mapping[str, Any], Mapping[str, Any]]]:
    bnd_by_view = {item["view_id"]: item for item in bnd_items}
    pairs = []
    for item in m1_items:
        bnd = bnd_by_view.get(item["view_id"])
        if bnd is None:
            raise RuntimeError(f"missing BND view {item['view_id']}")
        pairs.append((item, bnd))
    return pairs


def _component_tensors(m1: Mapping[str, Any], bnd: Mapping[str, Any]) -> Dict[str, Tensor]:
    o0 = m1["outputs"]
    o1 = bnd["outputs"]
    return {
        "DeltaJ": o1["clear_object_fullsh_raw"] - o0["clear_object_fullsh_raw"],
        "DeltaT": o1["transmission"] - o0["transmission"],
        "DeltaD": o1["direct_object_signal"] - o0["direct_object_signal"],
        "DeltaB": o1["rgb_medium"] - o0["rgb_medium"],
        "DeltaI": o1["pred_image"] - o0["pred_image"],
    }


def _component_delta_rows(scene: str, pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    components = ("DeltaJ", "DeltaT", "DeltaD", "DeltaB", "DeltaI")
    for support in ("all", "object_support"):
        accum: Dict[str, List[Tensor]] = {name: [] for name in components}
        luma_accum: Dict[str, List[Tensor]] = {name: [] for name in components}
        for m1, bnd in pairs:
            mask = _object_mask(m1) if support == "object_support" else torch.ones_like(m1["outputs"]["depth"][..., 0], dtype=torch.bool)
            tensors = _component_tensors(m1, bnd)
            for name, tensor in tensors.items():
                accum[name].append(_masked(tensor, mask).reshape(-1, 3) if tensor.shape[-1] == 3 else _masked(tensor, mask).reshape(-1, 1).expand(-1, 3))
                luma_accum[name].append(_luma(tensor)[mask].reshape(-1))
        for name in components:
            rgb = torch.cat(accum[name], dim=0)
            lum = torch.cat(luma_accum[name], dim=0)
            row = {"scene": scene, "support": support, "component_delta": name, "domain": "rgb_pooled"}
            row.update(_stats(rgb.reshape(-1), ""))
            row["positive_fraction"] = _threshold_fraction(rgb, 0.0, "gt")
            row["negative_fraction"] = _threshold_fraction(rgb, 0.0, "lt")
            rows.append(row)
            lrow = {"scene": scene, "support": support, "component_delta": name, "domain": "luma"}
            lrow.update(_stats(lum, ""))
            lrow["positive_fraction"] = _threshold_fraction(lum, 0.0, "gt")
            lrow["negative_fraction"] = _threshold_fraction(lum, 0.0, "lt")
            rows.append(lrow)
    return rows


def _cancellation_metrics(delta_d: Tensor, delta_b: Tensor, mask: Tensor) -> Dict[str, Any]:
    dd = delta_d[mask].reshape(-1, 3).float()
    db = delta_b[mask].reshape(-1, 3).float()
    if dd.numel() == 0:
        return {}
    sum_abs = (dd + db).abs().sum(dim=-1)
    raw_abs = dd.abs().sum(dim=-1) + db.abs().sum(dim=-1)
    residual_ratio = float(sum_abs.mean().item() / max(float(raw_abs.mean().item()), EPS))
    dot = (dd * db).sum(dim=-1)
    nd = torch.linalg.norm(dd, dim=-1)
    nb = torch.linalg.norm(db, dim=-1)
    cos = dot / (nd * nb + EPS)
    ratio = nd / (nb + EPS)
    out = {
        "deltaD_deltaB_pearson_rgb_flat": _pearson_torch(dd, db),
        "deltaD_deltaB_pearson_luma": _pearson_torch((dd * LUMA_WEIGHTS).sum(dim=-1), (db * LUMA_WEIGHTS).sum(dim=-1)),
        "flattened_cosine_similarity": float(dot.sum().item() / max(float(torch.sqrt((dd * dd).sum() * (db * db).sum()).item()), EPS)),
        "CANCELLATION_RESIDUAL_RATIO": residual_ratio,
        "CANCELLATION_EFFICIENCY": 1.0 - residual_ratio,
        "RECOMP_RAW_CHANGE": float(raw_abs.mean().item()),
        "RECOMP_FINAL_CHANGE": float(sum_abs.mean().item()),
        "cos_theta_mean": float(cos.mean().item()),
        "cos_theta_p10": _safe_quantile(cos, 0.10),
        "cos_theta_p50": _safe_quantile(cos, 0.50),
        "cos_theta_p90": _safe_quantile(cos, 0.90),
        "P_cos_lt_-0.9": _threshold_fraction(cos, -0.9, "lt"),
        "P_cos_lt_-0.5": _threshold_fraction(cos, -0.5, "lt"),
        "P_cos_gt_0": _threshold_fraction(cos, 0.0, "gt"),
        "r_DB_p10": _safe_quantile(ratio, 0.10),
        "r_DB_p50": _safe_quantile(ratio, 0.50),
        "r_DB_p90": _safe_quantile(ratio, 0.90),
    }
    for idx, channel in enumerate(CHANNELS):
        out[f"deltaD_deltaB_pearson_{channel}"] = _pearson_torch(dd[:, idx], db[:, idx])
    return out


def _direct_medium_cancellation_rows(scene: str, pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for m1, bnd in pairs:
        tensors = _component_tensors(m1, bnd)
        mask = _object_mask(m1)
        row = {"scene": scene, "view_id": m1["view_id"], "support": "object_support"}
        row.update(_cancellation_metrics(tensors["DeltaD"], tensors["DeltaB"], mask))
        rows.append(row)
    agg = _aggregate_numeric_rows(rows, {"scene": scene, "view_id": "AGGREGATE", "support": "object_support"})
    rows.append(agg)
    return rows


def _aggregate_numeric_rows(rows: Sequence[Mapping[str, Any]], id_fields: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(id_fields)
    keys = sorted({key for row in rows for key in row if key not in {"scene", "view_id", "support", "bin", "stratification"}})
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float)) and float(row[key]) == float(row[key])]
        if vals:
            out[key] = _mean(vals)
    return out


def _mse_attribution_rows(scene: str, pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for m1, bnd in pairs:
        gt = m1["gt"].float()
        o0 = m1["outputs"]
        o1 = bnd["outputs"]
        dD = o1["direct_object_signal"] - o0["direct_object_signal"]
        dB = o1["rgb_medium"] - o0["rgb_medium"]
        e0 = o0["pred_image"] - gt
        e1 = o1["pred_image"] - gt
        delta_mse = float((e1.square().mean() - e0.square().mean()).item())
        c_direct = float((2.0 * e0 * dD + dD.square()).mean().item())
        c_medium = float((2.0 * e0 * dB + dB.square()).mean().item())
        c_cross = float((2.0 * dD * dB).mean().item())
        closure = delta_mse - (c_direct + c_medium + c_cross)
        rows.append(
            {
                "scene": scene,
                "view_id": m1["view_id"],
                "DeltaMSE_actual": delta_mse,
                "C_direct": c_direct,
                "C_medium": c_medium,
                "C_cross": c_cross,
                "component_sum": c_direct + c_medium + c_cross,
                "absolute_closure_error": abs(closure),
                "relative_closure_error": abs(closure) / max(abs(delta_mse), EPS),
            }
        )
    rows.append(_aggregate_numeric_rows(rows, {"scene": scene, "view_id": "AGGREGATE"}))
    return rows


def _strat_row(scene: str, stratification: str, bin_name: str, masks_and_pairs: Sequence[Tuple[Tensor, Mapping[str, Any], Mapping[str, Any]]]) -> Dict[str, Any]:
    dd_vals = []
    db_vals = []
    di_vals = []
    raw_vals = []
    final_vals = []
    delta_mse_vals = []
    excess_vals = []
    j_vals = []
    tau_vals = []
    t_vals = []
    pixel_count = 0
    for mask, m1, bnd in masks_and_pairs:
        if not bool(mask.any()):
            continue
        tensors = _component_tensors(m1, bnd)
        gt = m1["gt"].float()
        e0 = torch.linalg.norm(m1["outputs"]["pred_image"] - gt, dim=-1)
        e1 = torch.linalg.norm(bnd["outputs"]["pred_image"] - gt, dim=-1)
        delta_mse_map = (bnd["outputs"]["pred_image"] - gt).square().mean(dim=-1) - (m1["outputs"]["pred_image"] - gt).square().mean(dim=-1)
        dd = _rgb_l1_per_pixel(tensors["DeltaD"])[mask]
        db = _rgb_l1_per_pixel(tensors["DeltaB"])[mask]
        di = _rgb_l1_per_pixel(tensors["DeltaI"])[mask]
        dd_vals.append(dd)
        db_vals.append(db)
        di_vals.append(di)
        raw_vals.append(dd + db)
        final_vals.append(di)
        delta_mse_vals.append(delta_mse_map[mask])
        excess_vals.append((e1 - e0).clamp_min(0.0)[mask])
        j_vals.append(m1["outputs"]["clear_object_fullsh_raw"].amax(dim=-1)[mask])
        tau_vals.append(m1["outputs"]["tau_D"].mean(dim=-1)[mask])
        t_vals.append(m1["outputs"]["transmission"].amin(dim=-1)[mask])
        pixel_count += int(mask.sum().item())
    row = {"scene": scene, "stratification": stratification, "bin": bin_name, "pixel_count": pixel_count}
    if pixel_count == 0:
        return row
    dd_all = torch.cat(dd_vals)
    db_all = torch.cat(db_vals)
    di_all = torch.cat(di_vals)
    raw_all = torch.cat(raw_vals)
    final_all = torch.cat(final_vals)
    row.update(
        {
            "mean_abs_DeltaD_l1": float(dd_all.mean().item()),
            "mean_abs_DeltaB_l1": float(db_all.mean().item()),
            "mean_abs_DeltaI_l1": float(di_all.mean().item()),
            "recomposition_efficiency": 1.0 - float(final_all.mean().item()) / max(float(raw_all.mean().item()), EPS),
            "DeltaMSE": float(torch.cat(delta_mse_vals).mean().item()),
            "BND_excess_residual": float(torch.cat(excess_vals).mean().item()),
            "M1_J_scalar_mean": float(torch.cat(j_vals).mean().item()),
            "M1_tau_mean": float(torch.cat(tau_vals).mean().item()),
            "M1_T_min_mean": float(torch.cat(t_vals).mean().item()),
        }
    )
    return row


def _pooled_thresholds(pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]], key_fn: Any, mask_fn: Any, qs: Sequence[float]) -> List[float]:
    vals = []
    for m1, _ in pairs:
        mask = mask_fn(m1)
        vals.append(key_fn(m1)[mask].reshape(-1))
    flat = torch.cat(vals) if vals else torch.empty(0)
    return [_safe_quantile(flat, q) for q in qs]


def _quintile_stratification(scene: str, pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]], name: str, key_fn: Any, mask_fn: Any) -> List[Dict[str, Any]]:
    qs = _pooled_thresholds(pairs, key_fn, mask_fn, (0.2, 0.4, 0.6, 0.8))
    rows = []
    for bin_idx in range(5):
        selected = []
        for m1, bnd in pairs:
            support = mask_fn(m1)
            value = key_fn(m1)
            if bin_idx == 0:
                mask = support & (value <= qs[0])
            elif bin_idx == 4:
                mask = support & (value > qs[3])
            else:
                mask = support & (value > qs[bin_idx - 1]) & (value <= qs[bin_idx])
            selected.append((mask, m1, bnd))
        rows.append(_strat_row(scene, name, f"Q{bin_idx + 1}", selected))
    return rows


def _stratification_rows(scene: str, pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    support_fn = lambda item: _object_mask(item)
    depth_rows = _quintile_stratification(scene, pairs, "M1_depth_quintile", lambda item: item["outputs"]["depth"][..., 0].float(), support_fn)
    j_rows = _quintile_stratification(scene, pairs, "M1_J_max_quintile", lambda item: item["outputs"]["clear_object_fullsh_raw"].amax(dim=-1), support_fn)
    tau_rows = _quintile_stratification(scene, pairs, "M1_tau_mean_quintile", lambda item: item["outputs"]["tau_D"].mean(dim=-1), support_fn)
    bright_rows = _quintile_stratification(scene, pairs, "GT_luminance_quintile", lambda item: _luma(item["gt"]), lambda item: torch.ones_like(item["outputs"]["depth"][..., 0], dtype=torch.bool))

    extra_j = []
    extra_tau = []
    extra_pos = []
    for mask_name, mask_builder in (
        ("J_gt_1", lambda m1: support_fn(m1) & (m1["outputs"]["clear_object_fullsh_raw"].amax(dim=-1) > 1.0)),
        ("J_le_1", lambda m1: support_fn(m1) & (m1["outputs"]["clear_object_fullsh_raw"].amax(dim=-1) <= 1.0)),
    ):
        extra_j.append(_strat_row(scene, "M1_J_masks", mask_name, [(mask_builder(m1), m1, bnd) for m1, bnd in pairs]))
    for mask_name, mask_builder in (
        ("tau_top10", _tau_top_mask),
        ("T_lt_0.3", lambda m1: support_fn(m1) & (m1["outputs"]["transmission"].amin(dim=-1) < 0.3)),
        ("T_lt_0.2", lambda m1: support_fn(m1) & (m1["outputs"]["transmission"].amin(dim=-1) < 0.2)),
        ("T_lt_0.1", lambda m1: support_fn(m1) & (m1["outputs"]["transmission"].amin(dim=-1) < 0.1)),
    ):
        extra_tau.append(_strat_row(scene, "M1_tau_T_masks", mask_name, [(mask_builder(m1), m1, bnd) for m1, bnd in pairs]))

    for name, ymask in (("top20", (0.0, 0.2)), ("middle60", (0.2, 0.8)), ("bottom20", (0.8, 1.0))):
        selected = []
        for m1, bnd in pairs:
            h, w = m1["outputs"]["depth"].shape[:2]
            yy = torch.linspace(0.0, 1.0, h).reshape(h, 1).expand(h, w)
            mask = (yy >= ymask[0]) & (yy < ymask[1])
            selected.append((mask, m1, bnd))
        extra_pos.append(_strat_row(scene, "image_y_position", name, selected))
    return depth_rows, j_rows + extra_j, tau_rows + extra_tau, bright_rows + extra_pos


def _tau_top_mask(m1: Mapping[str, Any]) -> Tensor:
    support = _object_mask(m1)
    tau = m1["outputs"]["tau_D"].mean(dim=-1)
    thresh = _safe_quantile(tau[support], 0.90)
    return support & (tau >= thresh)


def _per_view_rows(scene: str, pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for m1, bnd in pairs:
        tensors = _component_tensors(m1, bnd)
        mask = _object_mask(m1)
        cancel = _cancellation_metrics(tensors["DeltaD"], tensors["DeltaB"], mask)
        gt = m1["gt"]
        e0 = m1["outputs"]["pred_image"] - gt
        e1 = bnd["outputs"]["pred_image"] - gt
        row = {
            "scene": scene,
            "view_id": m1["view_id"],
            "PSNR_M1": m1["metrics"]["psnr"],
            "PSNR_BND": bnd["metrics"]["psnr"],
            "Delta_PSNR": bnd["metrics"]["psnr"] - m1["metrics"]["psnr"],
            "SSIM_M1": m1["metrics"]["ssim"],
            "SSIM_BND": bnd["metrics"]["ssim"],
            "LPIPS_M1": m1["metrics"]["lpips"],
            "LPIPS_BND": bnd["metrics"]["lpips"],
            "mean_abs_DeltaD_l1": float(_rgb_l1_per_pixel(tensors["DeltaD"])[mask].mean().item()),
            "mean_abs_DeltaB_l1": float(_rgb_l1_per_pixel(tensors["DeltaB"])[mask].mean().item()),
            "DeltaMSE": float((e1.square().mean() - e0.square().mean()).item()),
        }
        row.update(cancel)
        rows.append(row)
    delta = torch.tensor([row["Delta_PSNR"] for row in rows], dtype=torch.float32)
    eff = torch.tensor([row["CANCELLATION_EFFICIENCY"] for row in rows], dtype=torch.float32)
    rows.append(
        {
            "scene": scene,
            "view_id": "AGGREGATE_CORRELATION",
            "DeltaPSNR_vs_cancellation_efficiency_pearson": _pearson_torch(delta, eff),
            "view_count": len(rows),
        }
    )
    return rows


def _hybrid_rows(scene: str, pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]], metric_model: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = []
    recovery_rows = []
    for m1, bnd in pairs:
        gt = m1["gt"]
        o0 = m1["outputs"]
        o1 = bnd["outputs"]
        preds = {
            "M1": o0["pred_image"],
            "BND": o1["pred_image"],
            "H_D1_B0": o1["direct_object_signal"] + o0["rgb_medium"],
            "H_D0_B1": o0["direct_object_signal"] + o1["rgb_medium"],
        }
        metrics = {name: _metric_images(metric_model, pred, gt) for name, pred in preds.items()}
        gap = metrics["BND"]["mse"] - metrics["M1"]["mse"]
        for name, m in metrics.items():
            row = {
                "scene": scene,
                "view_id": m1["view_id"],
                "hybrid": name,
                "counterfactual_type": "IMAGE_SPACE_COUNTERFACTUAL" if name.startswith("H_") else "RENDERER_OUTPUT",
                "PSNR": m["psnr"],
                "SSIM": m["ssim"],
                "LPIPS": m["lpips"],
                "MSE": m["mse"],
            }
            if name.startswith("H_"):
                row["HYBRID_GAP_RECOVERY_FRACTION"] = (metrics["BND"]["mse"] - m["mse"]) / gap if abs(gap) > EPS else float("nan")
            rows.append(row)
        recovery_rows.append(
            {
                "scene": scene,
                "view_id": m1["view_id"],
                "RGB_GAP_MSE": gap,
                "H_D1_B0_RECOVERY": (metrics["BND"]["mse"] - metrics["H_D1_B0"]["mse"]) / gap if abs(gap) > EPS else float("nan"),
                "H_D0_B1_RECOVERY": (metrics["BND"]["mse"] - metrics["H_D0_B1"]["mse"]) / gap if abs(gap) > EPS else float("nan"),
            }
        )
    for hybrid in ("M1", "BND", "H_D1_B0", "H_D0_B1"):
        selected = [row for row in rows if row["hybrid"] == hybrid]
        rows.append(_aggregate_numeric_rows(selected, {"scene": scene, "view_id": "AGGREGATE", "hybrid": hybrid, "counterfactual_type": selected[0]["counterfactual_type"]}))
    recovery_rows.append(_aggregate_numeric_rows(recovery_rows, {"scene": scene, "view_id": "AGGREGATE"}))
    return rows, recovery_rows


def _jt_hybrid_audit(scene: str, pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for run_name, idx in (("M1", 0), ("BND-K1", 1)):
        diffs = []
        for pair in pairs:
            item = pair[idx]
            out = item["outputs"]
            reconstructed = out["clear_object_fullsh_raw"] * out["transmission"]
            diff = (out["direct_object_signal"] - reconstructed).abs()
            diffs.append(diff.reshape(-1))
        flat = torch.cat(diffs)
        rows.append(
            {
                "scene": scene,
                "run": run_name,
                "D_reconstruction": "clear_object_fullsh_raw * transmission",
                "mean_abs_diff": float(flat.mean().item()),
                "p95_abs_diff": _safe_quantile(flat, 0.95),
                "p99_abs_diff": _safe_quantile(flat, 0.99),
                "max_abs_diff": float(flat.max().item()),
                "JT_HYBRID_SEMANTICALLY_VALID": bool(float(flat.max().item()) < 1e-5),
                "status": "OK" if float(flat.max().item()) < 1e-5 else "JT_HYBRID_NOT_SEMANTICALLY_VALID",
                "reason_if_unavailable": "" if float(flat.max().item()) < 1e-5 else "image-space J_CLEAR*T_DIRECT does not reconstruct rasterized direct_object_signal; direct branch includes alpha/raster compositing semantics",
            }
        )
    return rows


def _required_residual_rows(scene: str, pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for run_name, pair_index in (("M1", 0), ("BND-K1", 1)):
        for m1, bnd in pairs:
            item = (m1, bnd)[pair_index]
            out = item["outputs"]
            gt = item["gt"]
            d = out["direct_object_signal"]
            b = out["rgb_medium"]
            medium_target_error = _rgb_l2_per_pixel(b - (gt - d))
            direct_target_error = _rgb_l2_per_pixel(d - (gt - b))
            correction = gt - d - b
            h, w = correction.shape[:2]
            yy, xx = torch.meshgrid(torch.linspace(-1.0, 1.0, h), torch.linspace(-1.0, 1.0, w), indexing="ij")
            corr_luma = _luma(correction)
            row = {
                "scene": scene,
                "run": run_name,
                "view_id": item["view_id"],
                "medium_target_error_mean": float(medium_target_error.mean().item()),
                "direct_target_error_mean": float(direct_target_error.mean().item()),
                "required_correction_luma_mean": float(corr_luma.mean().item()),
                "required_correction_luma_mean_abs": float(corr_luma.abs().mean().item()),
                "corr_required_luma_image_x": _pearson_torch(corr_luma, xx),
                "corr_required_luma_image_y": _pearson_torch(corr_luma, yy),
                "corr_required_luma_depth": _pearson_torch(corr_luma, out["depth"][..., 0]),
                "corr_required_luma_medium_luma": _pearson_torch(corr_luma, _luma(out["rgb_medium"])),
                "corr_required_luma_tau": _pearson_torch(corr_luma, out["tau_D"].mean(dim=-1)),
                "corr_required_luma_J": _pearson_torch(corr_luma, out["clear_object_fullsh_raw"].amax(dim=-1)),
                "camera_center_x": item["camera_center_x"],
                "camera_center_y": item["camera_center_y"],
                "camera_center_z": item["camera_center_z"],
                "camera_dir_x": item["camera_dir_x"],
                "camera_dir_y": item["camera_dir_y"],
                "camera_dir_z": item["camera_dir_z"],
            }
            rows.append(row)
    for scene_run in ((scene, "M1"), (scene, "BND-K1")):
        selected = [row for row in rows if row["scene"] == scene_run[0] and row["run"] == scene_run[1]]
        rows.append(_aggregate_numeric_rows(selected, {"scene": scene, "run": scene_run[1], "view_id": "AGGREGATE"}))
    return rows


def _gaussian_kernel1d(sigma: float, dtype: torch.dtype = torch.float32) -> Tensor:
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=dtype)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _gaussian_blur(image: Tensor, sigma: float) -> Tensor:
    img = image.detach().float().permute(2, 0, 1)[None, ...]
    c = img.shape[1]
    k = _gaussian_kernel1d(sigma).to(img.device)
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


def _frequency_and_edge_rows(scene: str, pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    freq_rows = []
    edge_rows = []
    for m1, bnd in pairs:
        residual = bnd["gt"] - bnd["outputs"]["pred_image"]
        for sigma in (3.0, 9.0):
            low = _gaussian_blur(residual, sigma)
            high = residual - low
            low_e = float(low.square().sum().item())
            high_e = float(high.square().sum().item())
            freq_rows.append(
                {
                    "scene": scene,
                    "view_id": bnd["view_id"],
                    "signal": "BND_residual_GT_minus_I",
                    "sigma_px": sigma,
                    "LOW_FREQ_ENERGY_FRACTION": low_e / max(low_e + high_e, EPS),
                    "HIGH_FREQ_ENERGY_FRACTION": high_e / max(low_e + high_e, EPS),
                    "low_energy": low_e,
                    "high_energy": high_e,
                }
            )
        edge = _gradient_magnitude_luma(bnd["gt"])
        resid_mag = _rgb_l2_per_pixel(residual)
        edge_thresh = _safe_quantile(edge.reshape(-1), 0.80)
        edge_mask = edge >= edge_thresh
        total_e = float(resid_mag.square().sum().item())
        edge_rows.append(
            {
                "scene": scene,
                "view_id": bnd["view_id"],
                "edge_definition": "top20 percent GT luminance gradient magnitude",
                "residual_edge_pearson": _pearson_torch(resid_mag, edge),
                "top20_edge_pixel_fraction": float(edge_mask.float().mean().item()),
                "residual_energy_fraction_top20_edge": float(resid_mag.square()[edge_mask].sum().item() / max(total_e, EPS)),
                "residual_energy_fraction_non_edge": float(resid_mag.square()[~edge_mask].sum().item() / max(total_e, EPS)),
            }
        )
    freq_rows.extend(_aggregate_frequency(freq_rows, scene))
    edge_rows.append(_aggregate_numeric_rows(edge_rows, {"scene": scene, "view_id": "AGGREGATE"}))
    return freq_rows, edge_rows


def _aggregate_frequency(rows: Sequence[Mapping[str, Any]], scene: str) -> List[Dict[str, Any]]:
    out = []
    for sigma in (3.0, 9.0):
        selected = [row for row in rows if float(row["sigma_px"]) == sigma]
        out.append(_aggregate_numeric_rows(selected, {"scene": scene, "view_id": "AGGREGATE", "signal": "BND_residual_GT_minus_I", "sigma_px": sigma}))
    return out


def _rgb_to_uint8(image: Tensor) -> Image.Image:
    arr = (image.detach().float().clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _scalar_to_uint8(values: Tensor, scale: float) -> Image.Image:
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


def _tile(image: Image.Image, label: str, width: int) -> Image.Image:
    if image.width != width:
        h = max(1, round(image.height * width / image.width))
        image = image.resize((width, h), Image.Resampling.BILINEAR)
    label_h = 28
    out = Image.new("RGB", (image.width, image.height + label_h), "white")
    out.paste(image, (0, label_h))
    ImageDraw.Draw(out).text((6, 7), label, fill="black")
    return out


def _save_sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Image.Image]]], tile_width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = []
    for row in rows:
        tiles = [_tile(img, label, tile_width) for label, img in row]
        w = sum(t.width for t in tiles) + 6 * (len(tiles) - 1)
        h = max(t.height for t in tiles)
        canvas = Image.new("RGB", (w, h), "white")
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, 0))
            x += tile.width + 6
        rendered.append(canvas)
    if not rendered:
        return
    w = max(r.width for r in rendered)
    h = sum(r.height for r in rendered) + 6 * (len(rendered) - 1)
    sheet = Image.new("RGB", (w, h), "white")
    y = 0
    for row in rendered:
        sheet.paste(row, (0, y))
        y += row.height + 6
    sheet.save(path)


def _manifest_image(manifest: List[Dict[str, Any]], path: Path, scene: str, output_type: str, view_ids: Sequence[str]) -> None:
    with Image.open(path) as img:
        w, h = img.size
    manifest.append({"file_path": str(path), "scene": scene, "output_type": output_type, "view_ids": ";".join(view_ids), "width": w, "height": h})


def _write_visuals(scene: str, pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]], render_dir: Path, tile_width: int, manifest: List[Dict[str, Any]]) -> None:
    view_ids = [m1["view_id"] for m1, _ in pairs]
    delta_scale = 0.1
    mag_scale = 0.1
    residual_scale = 0.1
    map_scale = 1.0
    for m1, bnd in pairs:
        tensors = _component_tensors(m1, bnd)
        delta_scale = max(delta_scale, float(_luma(tensors["DeltaD"]).abs().max().item()), float(_luma(tensors["DeltaB"]).abs().max().item()))
        mag_scale = max(mag_scale, float(_rgb_l2_per_pixel(tensors["DeltaD"]).max().item()), float(_rgb_l2_per_pixel(tensors["DeltaB"]).max().item()), float(_rgb_l2_per_pixel(tensors["DeltaI"]).max().item()))
        residual_scale = max(
            residual_scale,
            float(_rgb_l2_per_pixel(m1["outputs"]["pred_image"] - m1["gt"]).max().item()),
            float(_rgb_l2_per_pixel(bnd["outputs"]["pred_image"] - bnd["gt"]).max().item()),
        )
        map_scale = max(map_scale, float(m1["outputs"]["depth"].max().item()), float(m1["outputs"]["tau_D"].mean(dim=-1).max().item()))

    rows_component = []
    rows_cancel = []
    rows_residual = []
    rows_strat = []
    rows_hybrid = []
    for m1, bnd in pairs:
        view_id = m1["view_id"]
        tensors = _component_tensors(m1, bnd)
        rows_component.append(
            [
                (f"{view_id} GT", _rgb_to_uint8(m1["gt"])),
                ("M1 pred", _rgb_to_uint8(m1["outputs"]["pred_image"])),
                ("BND pred", _rgb_to_uint8(bnd["outputs"]["pred_image"])),
                ("M1 direct", _rgb_to_uint8(m1["outputs"]["direct_object_signal"])),
                ("BND direct", _rgb_to_uint8(bnd["outputs"]["direct_object_signal"])),
                ("Delta direct luma", _signed_to_rgb(_luma(tensors["DeltaD"]), delta_scale)),
                ("M1 medium", _rgb_to_uint8(m1["outputs"]["rgb_medium"])),
                ("BND medium", _rgb_to_uint8(bnd["outputs"]["rgb_medium"])),
                ("Delta medium luma", _signed_to_rgb(_luma(tensors["DeltaB"]), delta_scale)),
            ]
        )
        dd = tensors["DeltaD"]
        db = tensors["DeltaB"]
        cos = (dd * db).sum(dim=-1) / (torch.linalg.norm(dd, dim=-1) * torch.linalg.norm(db, dim=-1) + EPS)
        rows_cancel.append(
            [
                (f"{view_id} |DeltaD|", _scalar_to_uint8(_rgb_l2_per_pixel(dd), mag_scale)),
                ("|DeltaB|", _scalar_to_uint8(_rgb_l2_per_pixel(db), mag_scale)),
                ("|DeltaD+DeltaB|", _scalar_to_uint8(_rgb_l2_per_pixel(dd + db), mag_scale)),
                ("cos(theta)", _signed_to_rgb(cos, 1.0)),
            ]
        )
        e0 = _rgb_l2_per_pixel(m1["outputs"]["pred_image"] - m1["gt"])
        e1 = _rgb_l2_per_pixel(bnd["outputs"]["pred_image"] - bnd["gt"])
        excess = (e1 - e0).clamp_min(0)
        residual = bnd["gt"] - bnd["outputs"]["pred_image"]
        low = _gaussian_blur(residual, 9.0)
        high = residual - low
        rows_residual.append(
            [
                (f"{view_id} GT", _rgb_to_uint8(m1["gt"])),
                ("M1 abs residual", _scalar_to_uint8(e0, residual_scale)),
                ("BND abs residual", _scalar_to_uint8(e1, residual_scale)),
                ("BND excess", _scalar_to_uint8(excess, residual_scale)),
                ("BND low-freq residual", _signed_to_rgb(_luma(low), delta_scale)),
                ("BND high-freq residual", _signed_to_rgb(_luma(high), delta_scale)),
            ]
        )
        if scene == "Panama":
            rows_strat.append(
                [
                    (f"{view_id} M1 depth", _scalar_to_uint8(m1["outputs"]["depth"][..., 0], map_scale)),
                    ("M1 J", _scalar_to_uint8(m1["outputs"]["clear_object_fullsh_raw"].amax(dim=-1), 2.0)),
                    ("M1 tau", _scalar_to_uint8(m1["outputs"]["tau_D"].mean(dim=-1), map_scale)),
                    ("M1 T", _rgb_to_uint8(m1["outputs"]["transmission"])),
                    ("BND excess", _scalar_to_uint8(excess, residual_scale)),
                ]
            )
        rows_hybrid.append(
            [
                (f"{view_id} GT", _rgb_to_uint8(m1["gt"])),
                ("M1", _rgb_to_uint8(m1["outputs"]["pred_image"])),
                ("BND", _rgb_to_uint8(bnd["outputs"]["pred_image"])),
                ("D_BND+B_M1", _rgb_to_uint8(bnd["outputs"]["direct_object_signal"] + m1["outputs"]["rgb_medium"])),
                ("D_M1+B_BND", _rgb_to_uint8(m1["outputs"]["direct_object_signal"] + bnd["outputs"]["rgb_medium"])),
            ]
        )

    scene_dir = render_dir / scene
    for filename, rows, output_type in (
        ("contact_sheet_component_decomposition.png", rows_component, "component_decomposition"),
        ("contact_sheet_cancellation.png", rows_cancel, "cancellation"),
        ("contact_sheet_residual_frequency.png", rows_residual, "residual_frequency"),
        ("contact_sheet_hybrid_counterfactuals.png", rows_hybrid, "hybrid_counterfactuals"),
    ):
        path = scene_dir / filename
        _save_sheet(path, rows, tile_width)
        _manifest_image(manifest, path, scene, output_type, view_ids)
    if scene == "Panama":
        path = scene_dir / "contact_sheet_panama_stratification_maps.png"
        _save_sheet(path, rows_strat, tile_width)
        _manifest_image(manifest, path, scene, "panama_stratification_maps", view_ids)


def _write_index(path: Path, manifest: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# BND Object-Medium Recomposition Visual Compare Index", ""]
    for scene in SCENES:
        items = [item for item in manifest if item.get("scene") == scene and str(item.get("file_path", "")).endswith(".png")]
        if not items:
            continue
        lines.extend([f"## {scene}", ""])
        for item in items:
            lines.append(f"- {item['output_type']}: `{item['file_path']}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _summary_row(
    scene: str,
    component_rows: Sequence[Mapping[str, Any]],
    cancellation_rows: Sequence[Mapping[str, Any]],
    mse_rows: Sequence[Mapping[str, Any]],
    hybrid_recovery_rows: Sequence[Mapping[str, Any]],
    freq_rows: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
    canonical_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    comp = {(row["component_delta"], row["domain"], row["support"]): row for row in component_rows if row.get("scene") == scene}
    cancel = next(row for row in cancellation_rows if row["scene"] == scene and row["view_id"] == "AGGREGATE")
    mse = next(row for row in mse_rows if row["scene"] == scene and row["view_id"] == "AGGREGATE")
    hybrid = next(row for row in hybrid_recovery_rows if row["scene"] == scene and row["view_id"] == "AGGREGATE")
    freq9 = next(row for row in freq_rows if row["scene"] == scene and row["view_id"] == "AGGREGATE" and float(row["sigma_px"]) == 9.0)
    edge = next(row for row in edge_rows if row["scene"] == scene and row["view_id"] == "AGGREGATE")
    can = {row["run"]: row for row in canonical_rows if row["scene"] == scene}
    m1 = can["M1"]
    bnd = can["BND-K1"]
    return {
        "scene": scene,
        "M1_PSNR": m1["PSNR"],
        "BND_PSNR": bnd["PSNR"],
        "Delta_PSNR": bnd["PSNR"] - m1["PSNR"],
        "M1_tau_p90_canonical": m1["tau_eval_object_support_pooled_channel_mean_p90"],
        "BND_tau_p90_canonical": bnd["tau_eval_object_support_pooled_channel_mean_p90"],
        "M1_J_p99_canonical": m1["J_clear_eval_object_support_pooled_channel_mean_p99"],
        "BND_J_p99_canonical": bnd["J_clear_eval_object_support_pooled_channel_mean_p99"],
        "mean_abs_DeltaJ": comp[("DeltaJ", "rgb_pooled", "object_support")]["mean_abs"],
        "mean_abs_DeltaT": comp[("DeltaT", "rgb_pooled", "object_support")]["mean_abs"],
        "mean_abs_DeltaD": comp[("DeltaD", "rgb_pooled", "object_support")]["mean_abs"],
        "mean_abs_DeltaB": comp[("DeltaB", "rgb_pooled", "object_support")]["mean_abs"],
        "mean_abs_DeltaI": comp[("DeltaI", "rgb_pooled", "object_support")]["mean_abs"],
        "CANCELLATION_EFFICIENCY": cancel["CANCELLATION_EFFICIENCY"],
        "cos_theta_p50": cancel["cos_theta_p50"],
        "P_cos_lt_-0.9": cancel["P_cos_lt_-0.9"],
        "r_DB_p50": cancel["r_DB_p50"],
        "DeltaMSE": mse["DeltaMSE_actual"],
        "C_direct": mse["C_direct"],
        "C_medium": mse["C_medium"],
        "C_cross": mse["C_cross"],
        "mse_closure_error": mse["absolute_closure_error"],
        "H_D1_B0_RECOVERY": hybrid["H_D1_B0_RECOVERY"],
        "H_D0_B1_RECOVERY": hybrid["H_D0_B1_RECOVERY"],
        "LOW_FREQ_ENERGY_FRACTION_sigma9": freq9["LOW_FREQ_ENERGY_FRACTION"],
        "HIGH_FREQ_ENERGY_FRACTION_sigma9": freq9["HIGH_FREQ_ENERGY_FRACTION"],
        "edge_residual_pearson": edge["residual_edge_pearson"],
        "edge_top20_energy_fraction": edge["residual_energy_fraction_top20_edge"],
    }


def _canonical_decomposition_rows(scene: str, run: str, items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in items:
        out = item["outputs"]
        mask = _object_mask(item)
        rows.append(
            {
                "scene": scene,
                "run": run,
                "view_id": item["view_id"],
                "PSNR": item["metrics"]["psnr"],
                "SSIM": item["metrics"]["ssim"],
                "LPIPS": item["metrics"]["lpips"],
                "tau_values": out["tau_D"][mask].reshape(-1, 3),
                "J_values": out["clear_object_fullsh_raw"][mask].reshape(-1, 3),
            }
        )
    tau = torch.cat([row.pop("tau_values") for row in rows], dim=0)
    j = torch.cat([row.pop("J_values") for row in rows], dim=0)
    agg = {
        "scene": scene,
        "run": run,
        "view_id": "AGGREGATE",
        "PSNR": _mean(row["PSNR"] for row in rows),
        "SSIM": _mean(row["SSIM"] for row in rows),
        "LPIPS": _mean(row["LPIPS"] for row in rows),
        "tau_eval_object_support_pooled_channel_mean_p90": _mean(_safe_quantile(tau[:, idx], 0.90) for idx in range(3)),
        "J_clear_eval_object_support_pooled_channel_mean_p99": _mean(_safe_quantile(j[:, idx], 0.99) for idx in range(3)),
    }
    return rows + [agg]


def _final_classification(summary_rows: Sequence[Mapping[str, Any]], per_view_rows: Sequence[Mapping[str, Any]], depth_rows: Sequence[Mapping[str, Any]], j_rows: Sequence[Mapping[str, Any]], tau_rows: Sequence[Mapping[str, Any]], freq_rows: Sequence[Mapping[str, Any]], edge_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_scene = {row["scene"]: row for row in summary_rows}
    if "Panama" not in by_scene or not {"Curasao", "IUI3"}.issubset(by_scene):
        return {
            "status": "INSUFFICIENT_SCENES_FOR_FINAL_CLASSIFICATION",
            "available_scenes": sorted(by_scene),
            "RECOMPOSITION_STRONG_CROSS_SCENE": False,
            "PANAMA_RECOMPOSITION_INCOMPLETE": False,
            "OBJECT_DOMINATED_REMAINDER": False,
            "MEDIUM_DOMINATED_REMAINDER": False,
            "COUPLED_OBJECT_MEDIUM_REMAINDER": False,
            "VIEW_CONTEXT_REMAINDER": False,
            "MIXED_RECOMPOSITION_REMAINDER": False,
            "NEXT_SINGLE_FACTOR_EXPERIMENT": "unavailable_until_four_scene_run",
        }
    panama = by_scene["Panama"]
    success_eff = _mean(by_scene[s]["CANCELLATION_EFFICIENCY"] for s in ("Curasao", "IUI3"))
    pan_eff = float(panama["CANCELLATION_EFFICIENCY"])
    pan_gap = float(panama["Delta_PSNR"])
    medium_recovery = float(panama["H_D1_B0_RECOVERY"])
    direct_recovery = float(panama["H_D0_B1_RECOVERY"])
    low9 = float(panama["LOW_FREQ_ENERGY_FRACTION_sigma9"])
    edge_frac = float(panama["edge_top20_energy_fraction"])
    object_dom = edge_frac >= 0.40 and low9 < 0.60 and direct_recovery >= 0.50
    medium_dom = low9 >= 0.60 and medium_recovery >= 0.50 and edge_frac < 0.40
    coupled = medium_recovery < 0.50 and direct_recovery < 0.50
    mixed = not (object_dom or medium_dom or coupled)
    view_rows = [row for row in per_view_rows if row["scene"] == "Panama" and row["view_id"] != "AGGREGATE_CORRELATION"]
    eff_vals = [float(row["CANCELLATION_EFFICIENCY"]) for row in view_rows]
    view_context = bool(max(eff_vals) - min(eff_vals) > 0.15) if eff_vals else False
    next_exp = "Panama BND staged object-medium optimization test" if coupled or mixed else (
        "Panama BND medium-context capacity test" if medium_dom else "Panama BND-v2 bounded-base controlled-residual test"
    )
    return {
        "RECOMPOSITION_STRONG_CROSS_SCENE": bool(success_eff >= 0.60),
        "PANAMA_RECOMPOSITION_INCOMPLETE": bool(pan_gap < -0.15 and pan_eff < success_eff - 0.05),
        "OBJECT_DOMINATED_REMAINDER": bool(object_dom),
        "MEDIUM_DOMINATED_REMAINDER": bool(medium_dom),
        "COUPLED_OBJECT_MEDIUM_REMAINDER": bool(coupled),
        "VIEW_CONTEXT_REMAINDER": bool(view_context),
        "MIXED_RECOMPOSITION_REMAINDER": bool(mixed),
        "NEXT_SINGLE_FACTOR_EXPERIMENT": next_exp,
        "numeric_basis": {
            "success_control_efficiency_mean_Curasao_IUI3": success_eff,
            "Panama_cancellation_efficiency": pan_eff,
            "Panama_delta_PSNR": pan_gap,
            "Panama_H_D1_B0_RECOVERY": medium_recovery,
            "Panama_H_D0_B1_RECOVERY": direct_recovery,
            "Panama_LOW_FREQ_ENERGY_FRACTION_sigma9": low9,
            "Panama_edge_top20_energy_fraction": edge_frac,
            "Panama_view_efficiency_range": max(eff_vals) - min(eff_vals) if eff_vals else float("nan"),
        },
    }


def run(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    render_dir = args.render_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    start_info = {
        "branch": _git(repo, "branch", "--show-current"),
        "start_head": _git(repo, "rev-parse", "HEAD"),
        "start_log_1": _git(repo, "log", "-1", "--oneline"),
        "diagnostic": "BND-RECOMP",
        "training": "disabled/read-only",
    }
    _write_json(output_dir / "run_start.json", start_info)
    _write_json(output_dir / "component_semantics_audit.json", _component_semantics())

    manifest: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    closure_rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []
    cancellation_rows: List[Dict[str, Any]] = []
    mse_rows: List[Dict[str, Any]] = []
    depth_rows: List[Dict[str, Any]] = []
    j_rows: List[Dict[str, Any]] = []
    tau_rows: List[Dict[str, Any]] = []
    brightness_rows: List[Dict[str, Any]] = []
    per_view_rows: List[Dict[str, Any]] = []
    hybrid_rows: List[Dict[str, Any]] = []
    hybrid_recovery_rows: List[Dict[str, Any]] = []
    required_rows: List[Dict[str, Any]] = []
    freq_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    jt_rows: List[Dict[str, Any]] = []
    canonical_rows: List[Dict[str, Any]] = []
    secondary_rows: List[Dict[str, Any]] = []

    for scene in args.scenes:
        print(f"[BND-RECOMP] processing {scene}", flush=True)
        cache: Dict[str, List[Dict[str, Any]]] = {}
        metric_model = None
        for run_name in ("M1", "BND-K1"):
            spec = RUN_SPECS[(scene, run_name)]
            loaded = None
            try:
                loaded = _load_run(repo, spec)
                metric_model = loaded.model
                items, meta = _load_outputs(loaded)
                cache[run_name] = items
                checkpoint_rows.append(meta)
                closure_rows.extend(_closure_rows(scene, run_name, items))
                canonical_rows.extend(_canonical_decomposition_rows(scene, run_name, items))
            finally:
                _release_loaded(loaded)
        assert metric_model is not None
        pairs = _pair_by_view(cache["M1"], cache["BND-K1"])
        component_rows.extend(_component_delta_rows(scene, pairs))
        cancellation_rows.extend(_direct_medium_cancellation_rows(scene, pairs))
        mse_rows.extend(_mse_attribution_rows(scene, pairs))
        drows, jrows, trows, brows = _stratification_rows(scene, pairs)
        depth_rows.extend(drows)
        j_rows.extend(jrows)
        tau_rows.extend(trows)
        brightness_rows.extend(brows)
        per_view_rows.extend(_per_view_rows(scene, pairs))
        hrows, hrec = _hybrid_rows(scene, pairs, metric_model)
        hybrid_rows.extend(hrows)
        hybrid_recovery_rows.extend(hrec)
        required_rows.extend(_required_residual_rows(scene, pairs))
        frows, erows = _frequency_and_edge_rows(scene, pairs)
        freq_rows.extend(frows)
        edge_rows.extend(erows)
        jt_rows.extend(_jt_hybrid_audit(scene, pairs))
        _write_visuals(scene, pairs, render_dir, args.tile_width, manifest)

        if scene == "Panama":
            for run_name in ("K2", "K4"):
                loaded = None
                try:
                    loaded = _load_run(repo, RUN_SPECS[(scene, run_name)])
                    items, meta = _load_outputs(loaded)
                    checkpoint_rows.append(meta)
                    canonical_rows.extend(_canonical_decomposition_rows(scene, run_name, items))
                    secondary_rows.append(
                        {
                            "scene": scene,
                            "run": run_name,
                            "PSNR": _mean(item["metrics"]["psnr"] for item in items),
                            "SSIM": _mean(item["metrics"]["ssim"] for item in items),
                            "LPIPS": _mean(item["metrics"]["lpips"] for item in items),
                        }
                    )
                finally:
                    _release_loaded(loaded)

    summary_rows = [
        _summary_row(scene, component_rows, cancellation_rows, mse_rows, hybrid_recovery_rows, freq_rows, edge_rows, canonical_rows)
        for scene in args.scenes
    ]
    classifications = _final_classification(summary_rows, per_view_rows, depth_rows, j_rows, tau_rows, freq_rows, edge_rows)

    outputs: Dict[str, Any] = {
        "checkpoint_manifest": checkpoint_rows,
        "forward_closure_audit": closure_rows,
        "component_delta_summary": component_rows,
        "direct_medium_cancellation": cancellation_rows,
        "mse_component_attribution": mse_rows,
        "depth_stratified_recomposition": depth_rows,
        "j_stratified_recomposition": j_rows,
        "tau_stratified_recomposition": tau_rows,
        "brightness_position_stratification": brightness_rows,
        "per_view_recomposition": per_view_rows,
        "hybrid_counterfactual_metrics": hybrid_rows,
        "hybrid_gap_recovery": hybrid_recovery_rows,
        "required_component_residual": required_rows,
        "frequency_residual_analysis": freq_rows,
        "edge_alignment_analysis": edge_rows,
        "jt_hybrid_audit": jt_rows,
        "canonical_decomposition_metrics": canonical_rows,
        "panama_secondary_controls": secondary_rows,
        "bnd_recomposition_final_summary": summary_rows,
        "classification": [classifications],
    }
    for name, rows in outputs.items():
        _write_json(output_dir / f"{name}.json", rows)
        _write_csv(output_dir / f"{name}.csv", rows if isinstance(rows, list) else [rows])
        manifest.append({"file_path": str(output_dir / f"{name}.json"), "scene": "ALL", "output_type": name})
        manifest.append({"file_path": str(output_dir / f"{name}.csv"), "scene": "ALL", "output_type": name})
    payload = {"run_start": start_info, "scene_summary": summary_rows, "classification": classifications, "visual_manifest": manifest}
    _write_json(output_dir / "bnd_recomposition_final_summary.json", payload)
    _write_csv(output_dir / "bnd_recomposition_final_summary.csv", summary_rows)
    _write_json(output_dir / "manifest.json", manifest)
    _write_csv(output_dir / "manifest.csv", manifest)
    _write_json(render_dir / "manifest.json", manifest)
    _write_csv(render_dir / "manifest.csv", manifest)
    _write_index(render_dir / "VISUAL_COMPARE_INDEX.md", manifest)
    _write_index(output_dir / "VISUAL_COMPARE_INDEX.md", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bnd_object_medium_recomposition_20260810"))
    parser.add_argument("--render-dir", type=Path, default=Path("renders/bnd_object_medium_recomposition_20260810"))
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--tile-width", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    try:
        run(parse_args())
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise


if __name__ == "__main__":
    main()
