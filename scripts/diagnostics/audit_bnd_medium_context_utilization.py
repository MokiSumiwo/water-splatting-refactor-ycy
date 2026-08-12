#!/usr/bin/env python
"""Read-only BND medium-context utilization audit for Panama.

This diagnostic loads existing Panama M1 and BND-K1 checkpoints, queries the
trained medium MLP under fixed/counterfactual context, renders selected
counterfactual medium fields through the existing rasterizer, and writes
statistics plus visual assets.  It never creates an optimizer, calls a training
step, densifies, prunes, resets opacity, or writes checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup

from water_splatting.fields import (
    compute_bounded_gaussian_colors,
    compute_gaussian_colors,
    get_medium_context_extra_dim,
)
from water_splatting.sh import spherical_harmonics

from scripts.diagnostics import audit_bnd_cdepth_setup as cdepth_setup


SCENE = "Panama"
OUTPUT_DIR = Path("outputs/bnd_medctx_panama_20260812")
RENDER_DIR = Path("renders/bnd_medctx_panama_20260812")
LOG_DIR = Path("logs/bnd_medctx_panama_20260812")
RESEARCH_NOTE = Path("research_notes/BND_MEDIUM_CONTEXT_UTILIZATION_AUDIT_2026-08-12.md")

K1_CONFIG = cdepth_setup.K1_CONFIG
M1_CONFIG = Path(
    "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/"
    "cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
)
FINAL_NOMINAL_STEP = 15000
K1_LATE_STEPS = (8000, 10000, 13000, 15000)
TRAIN_VIEWS = (
    "MTN_1538",
    "MTN_1541",
    "MTN_1540",
    "MTN_1534",
    "MTN_1535",
    "MTN_1536",
    "MTN_1533",
    "MTN_1542",
    "MTN_1537",
    "MTN_1532",
    "MTN_1546",
    "MTN_1543",
    "MTN_1544",
    "MTN_1545",
    "MTN_1548",
)
HELDOUT_VIEWS = ("MTN_1529", "MTN_1539", "MTN_1547")
REGIONS = ("WHOLE_IMAGE", "OBJECT_SUPPORT", "M1_HIGH_J", "PERSISTENT_BND_HARD", "BND_HARD_CORE")
OUTPUT_GROUPS = ("B_inf", "beta_B", "beta_D")
CHANNELS = ("r", "g", "b")
ALLOWED_GPU_IDS = frozenset({"6", "7", "8", "9"})
EPS = 1e-12


@dataclass
class LoadedRun:
    run: str
    nominal_step: int
    actual_step: int
    config_path: Path
    checkpoint_path: Path
    loaded_step: int
    config: Any
    pipeline: Any

    @property
    def model(self) -> Any:
        return self.pipeline.model


@dataclass
class QueryFeatures:
    mlp_input: Tensor
    direction_encoded: Tensor
    xy_context: Tensor
    camera_context: Tensor
    directions: Tensor
    height: int
    width: int


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


def _run_cmd(repo: Path, args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(list(args), cwd=repo, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


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
    if nominal_step == FINAL_NOMINAL_STEP and 14999 in steps:
        return 14999
    raise FileNotFoundError(f"Missing checkpoint step {nominal_step} for {config_path}; available={sorted(steps)}")


def _assert_gpu_policy() -> Dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [token.strip() for token in visible.split(",") if token.strip()]
    if torch.cuda.is_available():
        if not devices:
            raise RuntimeError("CUDA_VISIBLE_DEVICES must be set to physical GPU 6, 7, 8, or 9 for this diagnostic.")
        if not set(devices).issubset(ALLOWED_GPU_IDS):
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES must use only physical GPUs {sorted(ALLOWED_GPU_IDS)}; got {visible!r}")
    return {
        "CUDA_VISIBLE_DEVICES": visible,
        "allowed_physical_gpus": sorted(ALLOWED_GPU_IDS),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torch_visible_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() else "none",
    }


def _release(loaded: Optional[LoadedRun]) -> None:
    if loaded is None:
        return
    try:
        del loaded.pipeline
    except Exception:
        pass
    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_run(repo: Path, run: str, step: int = FINAL_NOMINAL_STEP) -> LoadedRun:
    if run == "M1":
        config_path = repo / M1_CONFIG
        parameterization = "legacy"
    elif run == "BND-K1":
        config_path = repo / K1_CONFIG
        parameterization = "bounded_sh3"
    else:
        raise ValueError(f"Unknown run: {run}")
    actual = _actual_step(config_path, step)

    def update_config(config: Any) -> Any:
        config.load_step = actual
        config.pipeline.model.intrinsic_color_parameterization = parameterization
        config.pipeline.model.rasterize_mode = "classic"
        config.pipeline.model.medium_context_mode = "dir_xy_camera"
        config.pipeline.model.b_inf_mode = "tied"
        config.pipeline.model.infinite_water_enabled = False
        config.pipeline.model.coarse_depth_supervision_enabled = False
        config.pipeline.datamanager.load_depths = False
        return config

    config, pipeline, checkpoint_path, loaded_step = eval_setup(
        config_path,
        test_mode="test",
        update_config_callback=update_config,
    )
    model = pipeline.model
    model.config.intrinsic_color_parameterization = parameterization
    model.config.rasterize_mode = "classic"
    model.config.medium_context_mode = "dir_xy_camera"
    model.config.b_inf_mode = "tied"
    model.config.infinite_water_enabled = False
    model.config.coarse_depth_supervision_enabled = False
    model.step = int(loaded_step)
    pipeline.eval()
    return LoadedRun(run, step, int(actual), config_path, Path(checkpoint_path), int(loaded_step), config, pipeline)


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _train_records(pipeline: Any) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    dataset = pipeline.datamanager.train_dataset
    filenames = list(getattr(dataset, "image_filenames", []))
    cameras = dataset.cameras.to(pipeline.model.device)
    rows: List[Tuple[int, str, Cameras, Dict[str, Any]]] = []
    for index, filename in enumerate(filenames):
        view_id = Path(filename).stem
        batch = pipeline.datamanager.cached_train[index].copy()
        rows.append((index, view_id, cameras[index : index + 1], _batch_to_device(batch, pipeline.model.device)))
    return rows


def _eval_records(pipeline: Any) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    dataset = pipeline.datamanager.eval_dataset
    filenames = list(getattr(dataset, "image_filenames", []))
    rows: List[Tuple[int, str, Cameras, Dict[str, Any]]] = []
    for eval_index, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        view_id = Path(filenames[eval_index]).stem if eval_index < len(filenames) else f"eval_{eval_index}"
        rows.append((eval_index, view_id, camera.to(pipeline.model.device), _batch_to_device(batch, pipeline.model.device)))
    return rows


def _select_records(
    records: Sequence[Tuple[int, str, Cameras, Dict[str, Any]]],
    view_ids: Sequence[str],
) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    by_id = {view_id: record for record in records for view_id in [record[1]]}
    missing = [view_id for view_id in view_ids if view_id not in by_id]
    if missing:
        raise RuntimeError(f"Missing requested views: {missing}; available={sorted(by_id)}")
    return [by_id[view_id] for view_id in view_ids]


def _metric_images(model: Any, pred: Tensor, gt: Tensor) -> Dict[str, float]:
    pred = pred.detach().float().clamp(0.0, 1.0).to(model.device)
    gt = gt.detach().float().clamp(0.0, 1.0).to(model.device)
    pred_nchw = torch.moveaxis(pred, -1, 0)[None, ...]
    gt_nchw = torch.moveaxis(gt, -1, 0)[None, ...]
    mse = float(((pred - gt) ** 2).mean().item())
    return {
        "PSNR": float(model.psnr(gt_nchw, pred_nchw).item()),
        "SSIM": float(model.ssim(gt_nchw, pred_nchw).item()),
        "LPIPS": float(model.lpips(gt_nchw, pred_nchw).item()),
        "MSE": mse,
    }


def _safe_cpu_outputs(outputs: Mapping[str, Any]) -> Dict[str, Tensor]:
    keys = (
        "pred_image",
        "background",
        "accumulation",
        "direct_object_signal",
        "rgb_object",
        "rgb_medium",
        "rgb_medium_finite",
        "rgb_tail",
        "b_inf",
        "medium_rgb",
        "medium_bs",
        "medium_attn",
        "transmission",
        "tau_D",
        "depth",
        "clear_object_fullsh_raw",
        "rgb_clear",
        "rgb_clear_clamp",
    )
    return {
        key: value.detach().float().cpu()
        for key, value in outputs.items()
        if key in keys and isinstance(value, Tensor)
    }


def _get_gt(model: Any, batch: Mapping[str, Any], background: Tensor) -> Tensor:
    return model.composite_with_background(model.get_gt_img(batch["image"]), background.to(model.device)).detach().float().cpu()


def _render_records(
    pipeline: Any,
    records: Sequence[Tuple[int, str, Cameras, Dict[str, Any]]],
) -> Dict[str, Dict[str, Tensor]]:
    model = pipeline.model
    out: Dict[str, Dict[str, Tensor]] = {}
    model.eval()
    for _idx, view_id, camera, batch in records:
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera.to(model.device))
            gt = _get_gt(model, batch, outputs["background"])
            metrics = _metric_images(model, outputs["pred_image"], gt)
        safe = _safe_cpu_outputs(outputs)
        safe["gt"] = gt
        safe["metrics"] = torch.tensor(
            [metrics["PSNR"], metrics["SSIM"], metrics["LPIPS"], metrics["MSE"]],
            dtype=torch.float32,
        )
        safe["residual_mse"] = (safe["pred_image"].clamp(0.0, 1.0) - gt.clamp(0.0, 1.0)).square().mean(dim=-1)
        out[view_id] = safe
    return out


def _quantile(values: Tensor, q: float) -> float:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    if q <= 0:
        return float(flat.min().item())
    if q >= 1:
        return float(flat.max().item())
    return float(torch.quantile(flat, q).item())


def _stats(values: Tensor, prefix: str = "") -> Dict[str, Any]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    names = ("count", "mean", "std", "p01", "p05", "p10", "p50", "p90", "p95", "p99", "min", "max")
    if flat.numel() == 0:
        out = {f"{prefix}{name}": float("nan") for name in names}
        out[f"{prefix}count"] = 0
        return out
    return {
        f"{prefix}count": int(flat.numel()),
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        f"{prefix}p01": _quantile(flat, 0.01),
        f"{prefix}p05": _quantile(flat, 0.05),
        f"{prefix}p10": _quantile(flat, 0.10),
        f"{prefix}p50": _quantile(flat, 0.50),
        f"{prefix}p90": _quantile(flat, 0.90),
        f"{prefix}p95": _quantile(flat, 0.95),
        f"{prefix}p99": _quantile(flat, 0.99),
        f"{prefix}min": float(flat.min().item()),
        f"{prefix}max": float(flat.max().item()),
    }


def _channel_stats(values: Tensor, prefix: str) -> Dict[str, Any]:
    values = values.detach().float()
    out: Dict[str, Any] = {}
    if values.ndim >= 1 and values.shape[-1] == 3:
        for index, channel in enumerate(CHANNELS):
            out.update(_stats(values[..., index], f"{prefix}_{channel}_"))
        out.update(_stats(values.reshape(-1), f"{prefix}_all_"))
    else:
        out.update(_stats(values.reshape(-1), f"{prefix}_"))
    return out


def _flatten_finite_pair(a: Tensor, b: Tensor) -> Tuple[np.ndarray, np.ndarray]:
    aa = a.detach().float().cpu().reshape(-1).numpy()
    bb = b.detach().float().cpu().reshape(-1).numpy()
    finite = np.isfinite(aa) & np.isfinite(bb)
    return aa[finite], bb[finite]


def _rank_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    if values.size > 1:
        ranks = ranks / float(values.size - 1)
    return ranks


def _spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    ra = _rank_np(a)
    rb = _rank_np(b)
    if np.std(ra) < EPS or np.std(rb) < EPS:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) < EPS or np.std(b) < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _top_mask(values: Tensor, domain: Tensor, fraction: float) -> Tensor:
    mask = domain.detach().bool().cpu()
    out = torch.zeros_like(mask, dtype=torch.bool)
    vals = values.detach().float().cpu()[mask]
    finite = torch.isfinite(vals)
    if vals.numel() == 0 or int(finite.sum().item()) == 0:
        return out
    clean = vals.clone()
    clean[~finite] = clean[finite].min()
    k = max(1, int(math.ceil(float(fraction) * clean.numel())))
    indices = torch.topk(clean, k, largest=True).indices
    flat = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
    out.reshape(-1)[flat[indices]] = True
    return out


def _sample_indices(height: int, width: int, max_samples: int, seed: int, mask: Optional[Tensor] = None) -> Tensor:
    total = height * width
    if mask is not None:
        flat = torch.nonzero(mask.detach().bool().reshape(-1), as_tuple=False).reshape(-1).cpu()
        if flat.numel() == 0:
            flat = torch.arange(total, dtype=torch.long)
    else:
        flat = torch.arange(total, dtype=torch.long)
    if flat.numel() <= max_samples:
        return flat
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(flat.numel(), generator=generator)[:max_samples]
    return flat[perm]


def _camera_rotation_world_from_camera(model: Any, camera: Cameras) -> Tuple[Tensor, Tensor, float, float, int, int]:
    camera = camera.to(model.device)
    camera_downscale = model._get_downscale_factor()
    camera.rescale_output_resolution(1 / camera_downscale)
    try:
        rotation = camera.camera_to_worlds[0, :3, :3]
        r_edit = torch.diag(torch.tensor([1, -1, -1], device=model.device, dtype=rotation.dtype))
        rotation = rotation @ r_edit
        cx = float(camera.cx.item())
        cy = float(camera.cy.item())
        width = int(camera.width.item())
        height = int(camera.height.item())
    finally:
        camera.rescale_output_resolution(camera_downscale)
    return camera, rotation, cx, cy, height, width


def _build_medium_features(model: Any, camera: Cameras, require_grad: bool = False) -> QueryFeatures:
    camera, rotation, cx, cy, height, width = _camera_rotation_world_from_camera(model, camera)
    dtype = rotation.dtype
    device = rotation.device
    y = torch.linspace(0.0, height, height, device=device, dtype=dtype)
    x = torch.linspace(0.0, width, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yy_cam = (yy - cy) / camera.fy.item()
    xx_cam = (xx - cx) / camera.fx.item()
    directions = torch.stack([xx_cam, yy_cam, torch.ones_like(xx_cam)], dim=-1)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True).clamp_min(1e-12)
    directions = directions @ rotation.T
    direction_encoded = model.direction_encoding(directions.reshape(-1, 3))
    image_y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    image_x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    image_yy, image_xx = torch.meshgrid(image_y, image_x, indexing="ij")
    radius = torch.sqrt(image_xx.square() + image_yy.square())
    xy_context = torch.stack([image_xx, image_yy, radius], dim=-1).reshape(-1, 3)
    scene_center, scene_scale = model._get_scene_normalization(dtype=dtype, device=device)
    camera_feature = camera.camera_to_worlds[0, :3, 3].to(device=device, dtype=dtype)
    camera_feature = (camera_feature - scene_center) / (scene_scale + 1e-6)
    camera_feature = camera_feature * float(getattr(model.config, "medium_camera_context_scale", 1.0))
    camera_context = camera_feature.reshape(1, 3).expand(height * width, 3)
    mlp_input = torch.cat([direction_encoded, xy_context, camera_context], dim=-1)
    if require_grad:
        mlp_input = mlp_input.detach().clone().requires_grad_(True)
        ddir = direction_encoded.shape[-1]
        direction_encoded = mlp_input[:, :ddir]
        xy_context = mlp_input[:, ddir : ddir + 3]
        camera_context = mlp_input[:, ddir + 3 : ddir + 6]
    return QueryFeatures(
        mlp_input=mlp_input,
        direction_encoded=direction_encoded,
        xy_context=xy_context,
        camera_context=camera_context,
        directions=directions,
        height=height,
        width=width,
    )


def _medium_from_input(model: Any, mlp_input: Tensor) -> Dict[str, Tensor]:
    if model.config.mlp_type == "tcnn":
        raw = model.medium_mlp(mlp_input)
    else:
        raw = model.medium_mlp(mlp_input.float())
    density_bias = float(getattr(model, "medium_density_bias", 0.0))
    medium_rgb = torch.sigmoid(raw[..., :3]).float()
    medium_bs = F.softplus(raw[..., 3:6] + density_bias).float()
    medium_attn = F.softplus(raw[..., 6:9] + density_bias).float()
    return {"B_inf": medium_rgb, "medium_rgb": medium_rgb, "beta_B": medium_bs, "medium_bs": medium_bs, "beta_D": medium_attn, "medium_attn": medium_attn}


def _chunked_medium_from_input(model: Any, mlp_input: Tensor, chunk: int = 262144) -> Dict[str, Tensor]:
    outs: Dict[str, List[Tensor]] = {"B_inf": [], "beta_B": [], "beta_D": []}
    for start in range(0, mlp_input.shape[0], chunk):
        part = _medium_from_input(model, mlp_input[start : start + chunk])
        outs["B_inf"].append(part["B_inf"].detach())
        outs["beta_B"].append(part["beta_B"].detach())
        outs["beta_D"].append(part["beta_D"].detach())
    return {key: torch.cat(values, dim=0) for key, values in outs.items()}


def _context_reference(features_by_view: Mapping[str, QueryFeatures], view_ids: Sequence[str], max_per_view: int = 4096) -> Dict[str, Tensor]:
    xy_parts = []
    cam_parts = []
    dir_parts = []
    for idx, view_id in enumerate(view_ids):
        features = features_by_view[view_id]
        inds = _sample_indices(features.height, features.width, max_per_view, 1000 + idx)
        xy_parts.append(features.xy_context[inds].detach().cpu())
        cam_parts.append(features.camera_context[inds].detach().cpu())
        dir_parts.append(features.direction_encoded[inds].detach().cpu())
    xy_all = torch.cat(xy_parts, dim=0)
    cam_all = torch.cat(cam_parts, dim=0)
    dir_all = torch.cat(dir_parts, dim=0)
    return {
        "xy_mean": xy_all.mean(dim=0),
        "camera_mean": cam_all.mean(dim=0),
        "dir_std_rms": torch.sqrt(dir_all.var(dim=0, unbiased=False).mean()).clamp_min(0.0),
        "xy_std_rms": torch.sqrt(xy_all.var(dim=0, unbiased=False).mean()).clamp_min(0.0),
        "camera_std_rms": torch.sqrt(cam_all.var(dim=0, unbiased=False).mean()).clamp_min(0.0),
        "direction_encoded_dim": int(dir_all.shape[-1]),
    }


def _input_distribution_rows(
    run: str,
    features_by_view: Mapping[str, QueryFeatures],
    split_by_view: Mapping[str, str],
    max_per_view: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pooled: Dict[Tuple[str, str], List[Tensor]] = {}
    for view_idx, (view_id, features) in enumerate(features_by_view.items()):
        indices = _sample_indices(features.height, features.width, max_per_view, 2000 + view_idx)
        group_tensors = {
            "direction_encoded": features.direction_encoded[indices].detach().cpu(),
            "xy": features.xy_context[indices].detach().cpu(),
            "camera": features.camera_context[indices].detach().cpu(),
            "raw_direction": features.directions.reshape(-1, 3)[indices].detach().cpu(),
        }
        split = split_by_view[view_id]
        for group, tensor in group_tensors.items():
            row = {"run": run, "split": split, "view_id": view_id, "input_group": group, "sample_count": int(tensor.shape[0])}
            row.update(_stats(tensor, ""))
            rows.append(row)
            pooled.setdefault((split, group), []).append(tensor)
            pooled.setdefault(("all", group), []).append(tensor)
    summary: Dict[str, Any] = {}
    for (split, group), tensors in pooled.items():
        values = torch.cat(tensors, dim=0)
        row = {"run": run, "split": split, "view_id": "ALL", "input_group": group, "sample_count": int(values.shape[0])}
        row.update(_stats(values, ""))
        rows.append(row)
        summary[f"{split}_{group}"] = row
    return rows, summary


def _baseline_medium_rows(
    run: str,
    features_by_view: Mapping[str, QueryFeatures],
    split_by_view: Mapping[str, str],
    model: Any,
    max_samples_per_view: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pooled: Dict[Tuple[str, str], List[Tensor]] = {}
    model.eval()
    with torch.no_grad():
        for view_index, (view_id, features) in enumerate(features_by_view.items()):
            indices = _sample_indices(features.height, features.width, max_samples_per_view, 5000 + view_index).to(model.device)
            outputs = _medium_from_input(model, features.mlp_input[indices])
            split = split_by_view[view_id]
            for group in OUTPUT_GROUPS:
                values = outputs[group].detach().cpu()
                row = {
                    "run": run,
                    "split": split,
                    "view_id": view_id,
                    "output_group": group,
                    "sampling_rule": f"deterministic pixel sample, max_samples_per_view={max_samples_per_view}",
                    "sample_count": int(values.shape[0]),
                }
                row.update(_channel_stats(values, group))
                rows.append(row)
                pooled.setdefault((split, group), []).append(values.reshape(-1, 3))
                pooled.setdefault(("all", group), []).append(values.reshape(-1, 3))
    summary: Dict[str, Any] = {}
    for (split, group), tensors in pooled.items():
        values = torch.cat(tensors, dim=0)
        row = {"run": run, "split": split, "view_id": "ALL", "output_group": group}
        row.update(_channel_stats(values, group))
        rows.append(row)
        summary[f"{run}_{split}_{group}"] = row
    return rows, summary


def _jacobian_rows(
    run: str,
    model: Any,
    features_by_view: Mapping[str, QueryFeatures],
    split_by_view: Mapping[str, str],
    reference: Mapping[str, Tensor],
    max_per_view: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ddir = int(reference["direction_encoded_dim"])
    scales = {
        "direction": float(reference["dir_std_rms"].item()),
        "xy": float(reference["xy_std_rms"].item()),
        "camera": float(reference["camera_std_rms"].item()),
    }
    slices = {
        "direction": slice(0, ddir),
        "xy": slice(ddir, ddir + 3),
        "camera": slice(ddir + 3, ddir + 6),
    }
    for view_index, (view_id, base_features) in enumerate(features_by_view.items()):
        sample_indices = _sample_indices(base_features.height, base_features.width, max_per_view, 3000 + view_index).to(model.device)
        inputs = base_features.mlp_input[sample_indices].detach().clone().requires_grad_(True)
        outputs = _medium_from_input(model, inputs)
        for group_name in OUTPUT_GROUPS:
            grads_sq = {input_group: torch.zeros(inputs.shape[0], device=model.device) for input_group in slices}
            for channel in range(3):
                grad = torch.autograd.grad(
                    outputs[group_name][:, channel].sum(),
                    inputs,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                for input_group, slc in slices.items():
                    grads_sq[input_group] = grads_sq[input_group] + grad[:, slc].square().sum(dim=-1)
            for input_group, values_sq in grads_sq.items():
                raw = torch.sqrt(values_sq.clamp_min(0.0)).detach().cpu()
                row = {
                    "run": run,
                    "split": split_by_view[view_id],
                    "view_id": view_id,
                    "output_group": group_name,
                    "input_group": input_group,
                    "sample_count": int(raw.numel()),
                    "scale_rms_std": scales[input_group],
                }
                row.update(_stats(raw, "raw_jacobian_norm_"))
                normed = raw * scales[input_group]
                row.update(_stats(normed, "scale_normalized_sensitivity_"))
                rows.append(row)
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_rows: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["split"], row["output_group"], row["input_group"])
        summary_rows.setdefault(key, []).append(row)
    summary: Dict[str, Any] = {"run": run, "scale_definition": "raw local Frobenius norm multiplied by training-development RMS std for the input group"}
    for (split, out_group, in_group), group_rows in summary_rows.items():
        means = torch.tensor([float(row["scale_normalized_sensitivity_mean"]) for row in group_rows], dtype=torch.float32)
        raw_means = torch.tensor([float(row["raw_jacobian_norm_mean"]) for row in group_rows], dtype=torch.float32)
        summary[f"{split}_{out_group}_{in_group}"] = {
            "view_count": len(group_rows),
            "raw_jacobian_norm_mean_over_views": float(raw_means.mean().item()),
            "scale_normalized_sensitivity_mean_over_views": float(means.mean().item()),
        }
    return rows, summary


def _context_input(
    features: QueryFeatures,
    *,
    ddir: int,
    camera_context: Optional[Tensor] = None,
    xy_context: Optional[Tensor] = None,
) -> Tensor:
    direction = features.direction_encoded
    xy = features.xy_context if xy_context is None else xy_context.to(device=features.mlp_input.device, dtype=features.mlp_input.dtype)
    cam = features.camera_context if camera_context is None else camera_context.to(device=features.mlp_input.device, dtype=features.mlp_input.dtype)
    if xy.ndim == 1:
        xy = xy.reshape(1, 3).expand(features.height * features.width, 3)
    if cam.ndim == 1:
        cam = cam.reshape(1, 3).expand(features.height * features.width, 3)
    return torch.cat([direction[:, :ddir], xy, cam], dim=-1)


def _context_response_maps(
    model: Any,
    features_by_view: Mapping[str, QueryFeatures],
    records: Sequence[Tuple[int, str, Cameras, Dict[str, Any]]],
    split_by_view: Mapping[str, str],
    reference: Mapping[str, Tensor],
    max_store_views: Sequence[str],
    labels: Optional[Mapping[str, Mapping[str, Tensor]]] = None,
    baseline_outputs: Optional[Mapping[str, Mapping[str, Tensor]]] = None,
    max_samples_per_view: int = 4096,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Dict[str, Dict[str, Tensor]]]]:
    ddir = int(reference["direction_encoded_dim"])
    rows: List[Dict[str, Any]] = []
    stored_maps: Dict[str, Dict[str, Dict[str, Tensor]]] = {}
    record_by_view = {view_id: record for _idx, view_id, _cam, _batch in records for record in [(view_id, _cam, _batch)]}
    view_order = list(features_by_view)
    with torch.no_grad():
        for view_idx, view_id in enumerate(view_order):
            features = features_by_view[view_id]
            next_view = view_order[(view_idx + 1) % len(view_order)]
            foreign_cam = features_by_view[next_view].camera_context[0]
            inds = _sample_indices(features.height, features.width, max_samples_per_view, 6000 + view_idx).to(model.device)
            reverse_xy_all = torch.flip(features.xy_context, dims=[0])
            cam_swap_input = _context_input(features, ddir=ddir, camera_context=foreign_cam)
            xy_swap_input = torch.cat([features.direction_encoded[:, :ddir], reverse_xy_all, features.camera_context], dim=-1)
            cam_mean_input = _context_input(features, ddir=ddir, camera_context=reference["camera_mean"])
            extra_mean_input = _context_input(features, ddir=ddir, camera_context=reference["camera_mean"], xy_context=reference["xy_mean"])

            def sampled_outputs(sample_indices: Tensor) -> Tuple[Dict[str, Tensor], Dict[str, Dict[str, Tensor]]]:
                base = _medium_from_input(model, features.mlp_input[sample_indices])
                cfs = {
                    "camera_swap": _medium_from_input(model, cam_swap_input[sample_indices]),
                    "xy_swap": _medium_from_input(model, xy_swap_input[sample_indices]),
                    "CAM_CONTEXT_FIXED_CF": _medium_from_input(model, cam_mean_input[sample_indices]),
                    "EXTRA_CONTEXT_FIXED_CF": _medium_from_input(model, extra_mean_input[sample_indices]),
                }
                return base, cfs

            full, comparisons = sampled_outputs(inds)
            store = view_id in max_store_views
            if store:
                stored_maps[view_id] = {}
                full_all = _chunked_medium_from_input(model, features.mlp_input)
                cam_swap_all = _chunked_medium_from_input(model, cam_swap_input)
                xy_swap_all = _chunked_medium_from_input(model, xy_swap_input)
                cam_mean_all = _chunked_medium_from_input(model, cam_mean_input)
                extra_mean_all = _chunked_medium_from_input(model, extra_mean_input)
                stored_full = full_all
                stored_comparisons = {
                    "camera_swap": cam_swap_all,
                    "xy_swap": xy_swap_all,
                    "CAM_CONTEXT_FIXED_CF": cam_mean_all,
                    "EXTRA_CONTEXT_FIXED_CF": extra_mean_all,
                }
            for cf_name, cf_out in comparisons.items():
                for group in OUTPUT_GROUPS:
                    diff = (cf_out[group] - full[group]).abs().detach().cpu()
                    base_row = {
                        "run": getattr(model, "_medctx_run_name", ""),
                        "split": split_by_view[view_id],
                        "view_id": view_id,
                        "counterfactual": cf_name,
                        "output_group": group,
                        "reference_view_id": next_view if cf_name == "camera_swap" else "",
                        "sampling_rule": f"deterministic pixel sample, max_samples_per_view={max_samples_per_view}",
                    }
                    row = {**base_row, "region": "SAMPLED_ALL", "sample_count": int(diff.shape[0])}
                    row.update(_channel_stats(diff, "abs_delta"))
                    rows.append(row)
            if labels is not None and baseline_outputs is not None and view_id in labels and view_id in baseline_outputs:
                regions = _regions_for_view(labels[view_id], baseline_outputs[view_id])
                region_diffs: Dict[str, Dict[str, Dict[str, Tensor]]] = {}
                object_reference: Dict[Tuple[str, str], float] = {}
                for region_idx, region_name in enumerate(REGIONS):
                    mask_full = regions[region_name]
                    region_inds = _sample_indices(
                        features.height,
                        features.width,
                        max_samples_per_view,
                        6100 + view_idx * 31 + region_idx,
                        mask_full,
                    ).to(model.device)
                    region_full, region_comparisons = sampled_outputs(region_inds)
                    region_diffs[region_name] = {}
                    for cf_name, cf_out in region_comparisons.items():
                        region_diffs[region_name][cf_name] = {}
                        for group in OUTPUT_GROUPS:
                            diff = (cf_out[group] - region_full[group]).abs().detach().cpu()
                            region_diffs[region_name][cf_name][group] = diff
                            if region_name == "OBJECT_SUPPORT":
                                object_reference[(cf_name, group)] = float(diff.mean(dim=-1).mean().item()) if diff.numel() else float("nan")
                    del region_full, region_comparisons
                for region_name, cf_map in region_diffs.items():
                    for cf_name, group_map in cf_map.items():
                        for group, diff in group_map.items():
                            delta_l1 = diff.mean(dim=-1)
                            object_mean = object_reference.get((cf_name, group), float("nan"))
                            rrow = {
                                "run": getattr(model, "_medctx_run_name", ""),
                                "split": split_by_view[view_id],
                                "view_id": view_id,
                                "counterfactual": cf_name,
                                "output_group": group,
                                "reference_view_id": next_view if cf_name == "camera_swap" else "",
                                "sampling_rule": f"deterministic per-region pixel sample, max_samples_per_region={max_samples_per_view}",
                                "region": region_name,
                                "sample_count": int(delta_l1.numel()),
                                "object_support_mean_reference": object_mean,
                            }
                            rrow.update(_stats(delta_l1, "abs_delta_l1_"))
                            rrow.update(_channel_stats(diff, "abs_delta_rgb"))
                            mean_value = float(rrow["abs_delta_l1_mean"])
                            rrow["enrichment_vs_object_support"] = (
                                mean_value / max(object_mean, EPS) if math.isfinite(mean_value) and math.isfinite(object_mean) else float("nan")
                            )
                            rows.append(rrow)
            if store:
                for group in OUTPUT_GROUPS:
                    stored_maps[view_id].setdefault("FULL", {})[group] = stored_full[group].reshape(features.height, features.width, 3).detach().cpu()
                    for cf_name, cf_out in stored_comparisons.items():
                        diff_all = (cf_out[group] - stored_full[group]).abs().reshape(features.height, features.width, 3).detach().cpu()
                        stored_maps[view_id].setdefault(cf_name, {})[group] = diff_all.mean(dim=-1)
                        stored_maps[view_id].setdefault(cf_name, {})[f"{group}_rgb_delta"] = diff_all
                del full_all, cam_swap_all, xy_swap_all, cam_mean_all, extra_mean_all
            del full, comparisons, cam_swap_input, xy_swap_input, cam_mean_input, extra_mean_input
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary: Dict[str, Any] = {"run": getattr(model, "_medctx_run_name", "")}
    for cf_name in ("camera_swap", "xy_swap", "CAM_CONTEXT_FIXED_CF", "EXTRA_CONTEXT_FIXED_CF"):
        for group in OUTPUT_GROUPS:
            selected = [
                row
                for row in rows
                if row["counterfactual"] == cf_name
                and row["output_group"] == group
                and row.get("region", "SAMPLED_ALL") == "SAMPLED_ALL"
            ]
            if selected:
                vals = torch.tensor([float(row["abs_delta_all_mean"]) for row in selected], dtype=torch.float32)
                summary[f"{cf_name}_{group}"] = {
                    "view_count": len(selected),
                    "mean_abs_delta_over_views": float(vals.mean().item()),
                    "p90_abs_delta_mean_over_views": float(torch.quantile(vals, 0.90).item()),
                }
    return rows, summary, stored_maps


def _matched_los_rows(
    run: str,
    model: Any,
    features_by_view: Mapping[str, QueryFeatures],
    view_ids: Sequence[str],
    max_samples_per_view: int,
    max_pairs: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    sample: Dict[str, Dict[str, Tensor]] = {}
    with torch.no_grad():
        for view_idx, view_id in enumerate(view_ids):
            features = features_by_view[view_id]
            inds = _sample_indices(features.height, features.width, max_samples_per_view, 4000 + view_idx).to(model.device)
            outputs = _medium_from_input(model, features.mlp_input[inds])
            sample[view_id] = {
                "indices": inds.detach().cpu(),
                "directions": features.directions.reshape(-1, 3)[inds].detach(),
                "xy": features.xy_context[inds].detach(),
                "camera": features.camera_context[inds].detach(),
                "B_inf": outputs["B_inf"].detach(),
                "beta_B": outputs["beta_B"].detach(),
                "beta_D": outputs["beta_D"].detach(),
            }
    cross_rows: List[Dict[str, Any]] = []
    within_rows: List[Dict[str, Any]] = []

    def append_pair(rows: List[Dict[str, Any]], split_name: str, a_id: str, b_id: str, ia: int, ib: int, cos_value: float) -> None:
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_value))))
        a = sample[a_id]
        b = sample[b_id]
        row: Dict[str, Any] = {
            "run": run,
            "pair_type": split_name,
            "view_a": a_id,
            "view_b": b_id,
            "pixel_index_a": int(a["indices"][ia].item()),
            "pixel_index_b": int(b["indices"][ib].item()),
            "angular_difference_deg": angle,
            "delta_xy_norm": float(torch.linalg.norm(a["xy"][ia] - b["xy"][ib]).item()),
            "delta_camera_norm": float(torch.linalg.norm(a["camera"][ia] - b["camera"][ib]).item()),
        }
        for group in OUTPUT_GROUPS:
            row[f"delta_{group}_l1"] = float((a[group][ia] - b[group][ib]).abs().mean().item())
        rows.append(row)

    all_candidate_pairs: List[Tuple[float, str, str, int, int]] = []
    for i, view_a in enumerate(view_ids):
        dirs_a = sample[view_a]["directions"]
        for j, view_b in enumerate(view_ids):
            if j <= i:
                continue
            dirs_b = sample[view_b]["directions"]
            sims = dirs_a @ dirs_b.T
            best_vals, best_idx = torch.max(sims, dim=1)
            for ia in range(best_vals.numel()):
                all_candidate_pairs.append((float(best_vals[ia].item()), view_a, view_b, ia, int(best_idx[ia].item())))
    candidate_angles = [math.degrees(math.acos(max(-1.0, min(1.0, item[0])))) for item in all_candidate_pairs]
    count_1deg = sum(angle <= 1.0 for angle in candidate_angles)
    threshold = 2.0 if count_1deg < 10000 else 1.0
    selected_cross = [item for item, angle in zip(all_candidate_pairs, candidate_angles) if angle <= threshold]
    selected_cross = sorted(selected_cross, key=lambda item: -item[0])[:max_pairs]
    for cos_value, view_a, view_b, ia, ib in selected_cross:
        append_pair(cross_rows, "cross_camera", view_a, view_b, ia, ib, cos_value)

    # Within-camera direction control uses deterministic offset pairs with a
    # target angular distance no larger than the locked cross-camera threshold.
    for view_id in view_ids:
        dirs = sample[view_id]["directions"]
        n = dirs.shape[0]
        if n < 2:
            continue
        offset = max(1, n // 17)
        sims = (dirs * torch.roll(dirs, shifts=-offset, dims=0)).sum(dim=-1)
        for ia in range(min(n, max(1, max_pairs // max(1, len(view_ids))))):
            ib = int((ia + offset) % n)
            angle = math.degrees(math.acos(max(-1.0, min(1.0, float(sims[ia].item())))))
            if angle <= threshold:
                append_pair(within_rows, "within_camera", view_id, view_id, ia, ib, float(sims[ia].item()))

    summary: Dict[str, Any] = {
        "run": run,
        "cross_camera_threshold_rule": "Use 1 degree if at least 10000 matched pairs before result inspection, else 2 degrees.",
        "cross_camera_1deg_candidate_count": int(count_1deg),
        "locked_angular_threshold_deg": float(threshold),
        "cross_camera_pair_count": len(cross_rows),
        "within_camera_pair_count": len(within_rows),
    }
    for pair_type, rows in (("cross_camera", cross_rows), ("within_camera", within_rows)):
        for group in OUTPUT_GROUPS:
            vals = torch.tensor([float(row[f"delta_{group}_l1"]) for row in rows], dtype=torch.float32)
            summary[f"{pair_type}_{group}_mean_delta_l1"] = float(vals.mean().item()) if vals.numel() else float("nan")
            summary[f"{pair_type}_{group}_p90_delta_l1"] = float(torch.quantile(vals, 0.90).item()) if vals.numel() else float("nan")
    for group in OUTPUT_GROUPS:
        c = float(summary.get(f"cross_camera_{group}_mean_delta_l1", float("nan")))
        w = float(summary.get(f"within_camera_{group}_mean_delta_l1", float("nan")))
        summary[f"cross_over_within_{group}_mean_ratio"] = c / max(w, EPS) if math.isfinite(c) and math.isfinite(w) else float("nan")
    pairs = cross_rows + within_rows
    return pairs, within_rows, cross_rows, summary


def _legacy_colors(model: Any, camera: Cameras, active_sh_degree: int) -> Tensor:
    colors = torch.cat((model.features_dc[:, None, :], model.features_rest), dim=1)
    viewdirs = model.means.detach() - camera.camera_to_worlds[..., :3, 3].detach()
    viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    if model.config.sh_degree > 0:
        rgbs = spherical_harmonics(active_sh_degree, viewdirs, colors)
        return torch.clamp(rgbs + 0.5, min=0.0)
    return torch.sigmoid(colors[:, 0, :])


def _current_gaussian_colors(model: Any, camera: Cameras, active_sh_degree: int) -> Tensor:
    parameterization = getattr(model.config, "intrinsic_color_parameterization", "legacy")
    if parameterization == "legacy":
        return _legacy_colors(model, camera, active_sh_degree)
    if parameterization == "bounded_sh3":
        return compute_bounded_gaussian_colors(
            means=model.means,
            features_dc=model.features_dc,
            features_rest=model.features_rest,
            camera_position=camera.camera_to_worlds[..., :3, 3],
            sh_degree=model.config.sh_degree,
            active_sh_degree=active_sh_degree,
        ).rgb
    raise ValueError(f"Unsupported parameterization for diagnostic render: {parameterization}")


@torch.no_grad()
def _render_with_medium_override(
    model: Any,
    camera: Cameras,
    medium_rgb: Tensor,
    medium_bs: Tensor,
    medium_attn: Tensor,
) -> Dict[str, Tensor]:
    if not isinstance(camera, Cameras):
        raise TypeError("Expected Cameras object")
    camera = camera.to(model.device)
    camera_downscale = model._get_downscale_factor()
    camera.rescale_output_resolution(1 / camera_downscale)
    try:
        rotation = camera.camera_to_worlds[0, :3, :3]
        translation = camera.camera_to_worlds[0, :3, 3:4]
        r_edit = torch.diag(torch.tensor([1, -1, -1], device=model.device, dtype=rotation.dtype))
        rotation = rotation @ r_edit
        r_inv = rotation.T
        t_inv = -r_inv @ translation
        viewmat = torch.eye(4, device=rotation.device, dtype=rotation.dtype)
        viewmat[:3, :3] = r_inv
        viewmat[:3, 3:4] = t_inv
        cx = float(camera.cx.item())
        cy = float(camera.cy.item())
        width = int(camera.width.item())
        height = int(camera.height.item())

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
            height=height,
            width=width,
            clip_thresh=model.config.clip_thresh,
        )
    finally:
        camera.rescale_output_resolution(camera_downscale)

    if radii.sum() == 0:
        rgb = medium_rgb
        depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10.0
        clear = torch.zeros_like(rgb)
        tau_d = medium_attn * depth
        transmission = torch.exp(-tau_d.clamp_min(0.0)).clamp(0.0, 1.0)
        return {
            "rgb": rgb,
            "pred_image": rgb,
            "background": medium_rgb,
            "accumulation": medium_rgb.new_zeros(*rgb.shape[:2], 1),
            "direct_object_signal": clear,
            "rgb_object": clear,
            "rgb_medium": medium_rgb,
            "rgb_medium_finite": medium_rgb,
            "rgb_tail": torch.zeros_like(medium_rgb),
            "b_inf": medium_rgb,
            "medium_rgb": medium_rgb,
            "medium_bs": medium_bs,
            "medium_attn": medium_attn,
            "transmission": transmission,
            "tau_D": tau_d,
            "depth": depth,
        }

    active_sh_degree = min(model.step // model.config.sh_degree_interval, model.config.sh_degree)
    if crop_ids is not None and crop_ids.sum() != 0:
        # This branch is kept for source equivalence even though Panama final
        # diagnostics do not set a crop box.
        old_means, old_dc, old_rest = model.means, model.features_dc, model.features_rest
        try:
            model.means = means_crop
            model.features_dc = features_dc_crop
            model.features_rest = features_rest_crop
            rgbs = _current_gaussian_colors(model, camera, active_sh_degree)
        finally:
            model.means, model.features_dc, model.features_rest = old_means, old_dc, old_rest
    else:
        rgbs = _current_gaussian_colors(model, camera, active_sh_degree)
    if model.config.rasterize_mode == "antialiased":
        opacities = torch.sigmoid(opacities_crop) * comp[:, None]
    elif model.config.rasterize_mode == "classic":
        opacities = torch.sigmoid(opacities_crop)
    else:
        raise ValueError(f"Unknown rasterize_mode: {model.config.rasterize_mode}")

    xys_grad_abs = torch.zeros_like(xys)
    render = model.underwater_rasterizer.rasterize(
        xys=xys,
        xys_grad_abs=xys_grad_abs,
        depths=depths,
        radii=radii,
        conics=conics,
        num_tiles_hit=num_tiles_hit,
        colors=rgbs,
        opacities=opacities,
        medium_rgb=medium_rgb,
        medium_bs=medium_bs,
        medium_attn=medium_attn,
        height=height,
        width=width,
        background=medium_rgb,
        step=model.step,
    )
    rgb_medium = render.rgb_medium
    rgb_medium_finite = rgb_medium
    tail_weight = render.final_transmittance * torch.exp(-medium_bs * render.last_depth)
    rgb_tail_original = tail_weight * medium_rgb
    rgb_medium_finite = rgb_medium - rgb_tail_original
    rgb_tail = tail_weight * medium_rgb
    rgb_medium = rgb_medium_finite + rgb_tail
    rgb = render.rgb_object + rgb_medium
    tau_d = medium_attn * render.depth
    transmission = torch.exp(-tau_d.clamp_min(0.0)).clamp(0.0, 1.0)
    return {
        "rgb": rgb,
        "pred_image": rgb,
        "background": medium_rgb,
        "accumulation": render.accumulation,
        "direct_object_signal": render.rgb_object,
        "rgb_object": render.rgb_object,
        "rgb_medium": rgb_medium,
        "rgb_medium_finite": rgb_medium_finite,
        "rgb_tail": rgb_tail,
        "b_inf": medium_rgb,
        "medium_rgb": medium_rgb,
        "medium_bs": medium_bs,
        "medium_attn": medium_attn,
        "transmission": transmission,
        "tau_D": tau_d,
        "depth": render.depth,
        "clear_object_fullsh_raw": render.j_raw,
        "rgb_clear": render.rgb_clear,
        "rgb_clear_clamp": render.rgb_clear_clamp,
    }


def _counterfactual_render_rows(
    run: str,
    model: Any,
    records: Sequence[Tuple[int, str, Cameras, Dict[str, Any]]],
    features_by_view: Mapping[str, QueryFeatures],
    split_by_view: Mapping[str, str],
    labels: Mapping[str, Mapping[str, Tensor]],
    reference: Mapping[str, Tensor],
    visual_view_ids: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Dict[str, Tensor]]]]:
    ddir = int(reference["direction_encoded_dim"])
    rows_metrics: List[Dict[str, Any]] = []
    rows_region: List[Dict[str, Any]] = []
    visual_outputs: Dict[str, Dict[str, Dict[str, Tensor]]] = {}
    for _idx, view_id, camera, batch in records:
        features = features_by_view[view_id]
        with torch.no_grad():
            full_out = model.get_outputs_for_camera(camera.to(model.device))
            gt = _get_gt(model, batch, full_out["background"])
            full_medium = _chunked_medium_from_input(model, features.mlp_input)
            cam_mean_medium = _chunked_medium_from_input(model, _context_input(features, ddir=ddir, camera_context=reference["camera_mean"]))
            extra_mean_medium = _chunked_medium_from_input(
                model,
                _context_input(features, ddir=ddir, camera_context=reference["camera_mean"], xy_context=reference["xy_mean"]),
            )
            cf_rendered: Dict[str, Dict[str, Tensor]] = {
                "FULL": _safe_cpu_outputs(full_out),
                "CAM_CONTEXT_FIXED_CF": _safe_cpu_outputs(
                    _render_with_medium_override(
                        model,
                        camera,
                        cam_mean_medium["B_inf"].reshape(features.height, features.width, 3),
                        cam_mean_medium["beta_B"].reshape(features.height, features.width, 3),
                        cam_mean_medium["beta_D"].reshape(features.height, features.width, 3),
                    )
                ),
                "EXTRA_CONTEXT_FIXED_CF": _safe_cpu_outputs(
                    _render_with_medium_override(
                        model,
                        camera,
                        extra_mean_medium["B_inf"].reshape(features.height, features.width, 3),
                        extra_mean_medium["beta_B"].reshape(features.height, features.width, 3),
                        extra_mean_medium["beta_D"].reshape(features.height, features.width, 3),
                    )
                ),
            }
        # Forward consistency check for the directly queried FULL medium output.
        for group, key in (("B_inf", "medium_rgb"), ("beta_B", "medium_bs"), ("beta_D", "medium_attn")):
            queried = full_medium[group].reshape(features.height, features.width, 3).detach().cpu()
            rendered = cf_rendered["FULL"][key]
            diff = (queried - rendered).abs()
            rows_metrics.append(
                {
                    "run": run,
                    "split": split_by_view[view_id],
                    "view_id": view_id,
                    "counterfactual": "FULL_MEDIUM_QUERY_FORWARD_CHECK",
                    "output_group": group,
                    "MSE": float(diff.square().mean().item()),
                    "MAE": float(diff.mean().item()),
                    "max_abs": float(diff.max().item()),
                }
            )
        for cf_name, outputs in cf_rendered.items():
            metrics = _metric_images(model, outputs["pred_image"], gt)
            row = {
                "run": run,
                "split": split_by_view[view_id],
                "view_id": view_id,
                "counterfactual": cf_name,
                "output_group": "rgb_render",
            }
            row.update(metrics)
            rows_metrics.append(row)
            if cf_name == "FULL":
                continue
            delta_rgb = (outputs["pred_image"] - cf_rendered["FULL"]["pred_image"]).abs().mean(dim=-1)
            delta_medium = (outputs["rgb_medium"] - cf_rendered["FULL"]["rgb_medium"]).abs().mean(dim=-1)
            delta_direct = (outputs["direct_object_signal"] - cf_rendered["FULL"]["direct_object_signal"]).abs().mean(dim=-1)
            region_map = _regions_for_view(labels[view_id], cf_rendered["FULL"])
            for delta_name, delta in (("delta_final_rgb", delta_rgb), ("delta_medium", delta_medium), ("delta_direct", delta_direct)):
                for region_name, mask in region_map.items():
                    vals = delta[mask]
                    row_r = {
                        "run": run,
                        "split": split_by_view[view_id],
                        "view_id": view_id,
                        "counterfactual": cf_name,
                        "response": delta_name,
                        "region": region_name,
                        "pixel_count": int(mask.sum().item()),
                    }
                    row_r.update(_stats(vals, "abs_response_"))
                    rows_region.append(row_r)
        if view_id in visual_view_ids:
            visual_outputs[view_id] = cf_rendered
        del full_medium, cam_mean_medium, extra_mean_medium, cf_rendered
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows_metrics, rows_region, visual_outputs


def _regions_for_view(label_map: Mapping[str, Tensor], outputs: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    if "accumulation" in outputs:
        support = outputs["accumulation"][..., 0].detach().float().cpu() > 0.01
    else:
        any_label = next(iter(label_map.values()))
        support = torch.ones_like(any_label, dtype=torch.bool)
    whole = torch.ones_like(support, dtype=torch.bool)
    return {
        "WHOLE_IMAGE": whole,
        "OBJECT_SUPPORT": support,
        "M1_HIGH_J": label_map["M1_HIGH_J"].detach().bool().cpu(),
        "PERSISTENT_BND_HARD": label_map["PERSISTENT_BND_HARD"].detach().bool().cpu(),
        "BND_HARD_CORE": label_map["BND_HARD_CORE"].detach().bool().cpu(),
    }


def _build_label_maps(
    repo: Path,
    all_records_final: Sequence[Tuple[int, str, Cameras, Dict[str, Any]]],
) -> Tuple[Dict[str, Dict[str, Tensor]], Dict[str, Tensor], Dict[str, Any], Dict[str, Dict[str, Tensor]], Dict[str, Dict[str, Tensor]]]:
    m1 = _load_run(repo, "M1", FINAL_NOMINAL_STEP)
    k1_final = _load_run(repo, "BND-K1", FINAL_NOMINAL_STEP)
    late_runs: Dict[int, LoadedRun] = {}
    try:
        for step in K1_LATE_STEPS:
            late_runs[step] = _load_run(repo, "BND-K1", step)
        all_view_ids = [view_id for _idx, view_id, _camera, _batch in all_records_final]
        m1_records = _select_records(_train_records(m1.pipeline) + _eval_records(m1.pipeline), all_view_ids)
        k1_records = _select_records(_train_records(k1_final.pipeline) + _eval_records(k1_final.pipeline), all_view_ids)
        m1_maps = _render_records(m1.pipeline, m1_records)
        k1_maps = _render_records(k1_final.pipeline, k1_records)
        late_maps: Dict[int, Dict[str, Dict[str, Tensor]]] = {}
        for step, loaded in late_runs.items():
            late_records = _select_records(_train_records(loaded.pipeline) + _eval_records(loaded.pipeline), all_view_ids)
            late_maps[step] = _render_records(loaded.pipeline, late_records)
        labels: Dict[str, Dict[str, Tensor]] = {}
        domains: Dict[str, Tensor] = {}
        persistent_required = int(math.ceil(0.75 * len(late_maps)))
        for view_id in all_view_ids:
            support = k1_maps[view_id]["accumulation"][..., 0] > 0.01
            domains[view_id] = support
            m1_highj = (m1_maps[view_id]["accumulation"][..., 0] > 0.01) & (m1_maps[view_id]["clear_object_fullsh_raw"].amax(dim=-1) > 1.0)
            count = torch.zeros_like(support, dtype=torch.int32)
            for maps in late_maps.values():
                count += _top_mask(maps[view_id]["residual_mse"], support, 0.10).int()
            persistent = support & (count >= persistent_required)
            labels[view_id] = {
                "M1_HIGH_J": m1_highj,
                "PERSISTENT_BND_HARD": persistent,
                "BND_HARD_CORE": persistent & m1_highj,
            }
        meta = {
            "definition": {
                "M1_HIGH_J": "M1 final accumulation > 0.01 and max RGB of M1 final clear_object_fullsh_raw > 1.0; offline oracle diagnostic label only.",
                "PERSISTENT_BND_HARD": "K1 late residual top 10 percent inside final K1 object support for at least 75 percent of available late checkpoints; offline future-outcome diagnostic label only.",
                "BND_HARD_CORE": "PERSISTENT_BND_HARD AND M1_HIGH_J.",
                "OBJECT_SUPPORT": "BND-K1 final accumulation > 0.01.",
            },
            "late_steps": sorted(late_maps),
            "persistent_required_count": persistent_required,
        }
        return labels, domains, meta, m1_maps, k1_maps
    finally:
        _release(m1)
        _release(k1_final)
        for loaded in late_runs.values():
            _release(loaded)


def _region_enrichment_rows(
    run: str,
    response_maps: Mapping[str, Dict[str, Dict[str, Tensor]]],
    labels: Mapping[str, Dict[str, Tensor]],
    baseline_outputs: Mapping[str, Dict[str, Tensor]],
    split_by_view: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for view_id, cf_map in response_maps.items():
        regions = _regions_for_view(labels[view_id], baseline_outputs[view_id])
        object_mask = regions["OBJECT_SUPPORT"]
        for cf_name, groups in cf_map.items():
            if cf_name == "FULL":
                continue
            for group in OUTPUT_GROUPS:
                if group not in groups:
                    continue
                delta = groups[group].detach().float().cpu()
                obj_mean = float(delta[object_mask].mean().item()) if int(object_mask.sum().item()) else float("nan")
                for region_name, mask in regions.items():
                    vals = delta[mask]
                    row = {
                        "run": run,
                        "split": split_by_view[view_id],
                        "view_id": view_id,
                        "counterfactual": cf_name,
                        "output_group": group,
                        "region": region_name,
                        "pixel_count": int(mask.sum().item()),
                        "object_support_mean_reference": obj_mean,
                    }
                    row.update(_stats(vals, "abs_delta_"))
                    mean_value = float(row["abs_delta_mean"])
                    row["enrichment_vs_object_support"] = mean_value / max(obj_mean, EPS) if math.isfinite(mean_value) and math.isfinite(obj_mean) else float("nan")
                    rows.append(row)
    summary: Dict[str, Any] = {"run": run}
    for cf_name in ("camera_swap", "xy_swap", "CAM_CONTEXT_FIXED_CF", "EXTRA_CONTEXT_FIXED_CF"):
        for group in OUTPUT_GROUPS:
            for region_name in REGIONS:
                selected = [row for row in rows if row["counterfactual"] == cf_name and row["output_group"] == group and row["region"] == region_name]
                if selected:
                    vals = torch.tensor([float(row["enrichment_vs_object_support"]) for row in selected if math.isfinite(float(row["enrichment_vs_object_support"]))])
                    summary[f"{cf_name}_{group}_{region_name}"] = {
                        "view_count": len(selected),
                        "mean_enrichment_vs_object_support": float(vals.mean().item()) if vals.numel() else float("nan"),
                        "views_enrichment_ge_1p5": int((vals >= 1.5).sum().item()) if vals.numel() else 0,
                    }
    return rows, summary


def _context_region_summary(rows: Sequence[Mapping[str, Any]], run: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"run": run, "source": "sampled context response rows"}
    for cf_name in ("camera_swap", "xy_swap", "CAM_CONTEXT_FIXED_CF", "EXTRA_CONTEXT_FIXED_CF"):
        for group in OUTPUT_GROUPS:
            for region_name in REGIONS:
                selected = [
                    row
                    for row in rows
                    if row.get("counterfactual") == cf_name
                    and row.get("output_group") == group
                    and row.get("region") == region_name
                ]
                vals = [
                    float(row.get("enrichment_vs_object_support", float("nan")))
                    for row in selected
                    if math.isfinite(float(row.get("enrichment_vs_object_support", float("nan"))))
                ]
                means = [
                    float(row.get("abs_delta_l1_mean", float("nan")))
                    for row in selected
                    if math.isfinite(float(row.get("abs_delta_l1_mean", float("nan"))))
                ]
                summary[f"{cf_name}_{group}_{region_name}"] = {
                    "view_count": len(selected),
                    "mean_abs_delta_l1": float(torch.tensor(means).mean().item()) if means else float("nan"),
                    "mean_enrichment_vs_object_support": float(torch.tensor(vals).mean().item()) if vals else float("nan"),
                    "views_enrichment_ge_1p5": int(sum(v >= 1.5 for v in vals)),
                }
    return summary


def _correlation_rows(
    run: str,
    response_maps: Mapping[str, Dict[str, Dict[str, Tensor]]],
    labels: Mapping[str, Dict[str, Tensor]],
    baseline_outputs: Mapping[str, Dict[str, Tensor]],
    split_by_view: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for view_id, cf_map in response_maps.items():
        residual = baseline_outputs[view_id]["residual_mse"].detach().float().cpu()
        hard_score = labels[view_id]["BND_HARD_CORE"].float()
        object_mask = (baseline_outputs[view_id]["accumulation"][..., 0] > 0.01).detach().bool().cpu()
        for cf_name, groups in cf_map.items():
            if cf_name == "FULL":
                continue
            for group in OUTPUT_GROUPS:
                if group not in groups:
                    continue
                context = groups[group].detach().float().cpu()
                for region_name, domain in (("WHOLE_IMAGE", torch.ones_like(object_mask, dtype=torch.bool)), ("OBJECT_SUPPORT", object_mask)):
                    a_res, b_res = _flatten_finite_pair(context[domain], residual[domain])
                    a_hard, b_hard = _flatten_finite_pair(context[domain], hard_score[domain])
                    rows.append(
                        {
                            "run": run,
                            "split": split_by_view[view_id],
                            "view_id": view_id,
                            "counterfactual": cf_name,
                            "output_group": group,
                            "domain": region_name,
                            "pixel_count": int(domain.sum().item()),
                            "pearson_context_vs_rgb_mse": _pearson_np(a_res, b_res),
                            "spearman_context_vs_rgb_mse": _spearman_np(a_res, b_res),
                            "pearson_context_vs_bnd_hard_core": _pearson_np(a_hard, b_hard),
                            "spearman_context_vs_bnd_hard_core": _spearman_np(a_hard, b_hard),
                        }
                    )
    summary: Dict[str, Any] = {"run": run}
    for cf_name in ("camera_swap", "xy_swap", "CAM_CONTEXT_FIXED_CF", "EXTRA_CONTEXT_FIXED_CF"):
        for group in OUTPUT_GROUPS:
            selected = [
                row
                for row in rows
                if row["counterfactual"] == cf_name and row["output_group"] == group and row["domain"] == "OBJECT_SUPPORT"
            ]
            if selected:
                vals_mse = torch.tensor([float(row["spearman_context_vs_rgb_mse"]) for row in selected if math.isfinite(float(row["spearman_context_vs_rgb_mse"]))])
                vals_hard = torch.tensor([float(row["spearman_context_vs_bnd_hard_core"]) for row in selected if math.isfinite(float(row["spearman_context_vs_bnd_hard_core"]))])
                summary[f"{cf_name}_{group}_object_support"] = {
                    "mean_spearman_context_vs_rgb_mse": float(vals_mse.mean().item()) if vals_mse.numel() else float("nan"),
                    "mean_spearman_context_vs_bnd_hard_core": float(vals_hard.mean().item()) if vals_hard.numel() else float("nan"),
                }
    return rows, summary


def _sampled_context_error_correlation_rows(
    run: str,
    model: Any,
    features_by_view: Mapping[str, QueryFeatures],
    split_by_view: Mapping[str, str],
    labels: Mapping[str, Dict[str, Tensor]],
    baseline_outputs: Mapping[str, Dict[str, Tensor]],
    reference: Mapping[str, Tensor],
    max_samples_per_view: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ddir = int(reference["direction_encoded_dim"])
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for view_idx, (view_id, features) in enumerate(features_by_view.items()):
            object_mask_full = baseline_outputs[view_id]["accumulation"][..., 0] > 0.01
            inds = _sample_indices(features.height, features.width, max_samples_per_view, 7000 + view_idx, object_mask_full).to(model.device)
            next_view = list(features_by_view)[(view_idx + 1) % len(features_by_view)]
            foreign_cam = features_by_view[next_view].camera_context[0]
            reverse_xy_all = torch.flip(features.xy_context, dims=[0])
            full = _medium_from_input(model, features.mlp_input[inds])
            cfs = {
                "camera_swap": _medium_from_input(model, _context_input(features, ddir=ddir, camera_context=foreign_cam)[inds]),
                "xy_swap": _medium_from_input(model, torch.cat([features.direction_encoded[:, :ddir], reverse_xy_all, features.camera_context], dim=-1)[inds]),
                "CAM_CONTEXT_FIXED_CF": _medium_from_input(model, _context_input(features, ddir=ddir, camera_context=reference["camera_mean"])[inds]),
                "EXTRA_CONTEXT_FIXED_CF": _medium_from_input(
                    model,
                    _context_input(features, ddir=ddir, camera_context=reference["camera_mean"], xy_context=reference["xy_mean"])[inds],
                ),
            }
            inds_cpu = inds.detach().cpu()
            residual = baseline_outputs[view_id]["residual_mse"].reshape(-1)[inds_cpu]
            hard_core = labels[view_id]["BND_HARD_CORE"].float().reshape(-1)[inds_cpu]
            for cf_name, cf_out in cfs.items():
                for group in OUTPUT_GROUPS:
                    context = (cf_out[group] - full[group]).abs().mean(dim=-1).detach().cpu()
                    a_res, b_res = _flatten_finite_pair(context, residual)
                    a_hard, b_hard = _flatten_finite_pair(context, hard_core)
                    rows.append(
                        {
                            "run": run,
                            "split": split_by_view[view_id],
                            "view_id": view_id,
                            "counterfactual": cf_name,
                            "output_group": group,
                            "domain": "OBJECT_SUPPORT_SAMPLED",
                            "sample_count": int(context.numel()),
                            "pearson_context_vs_rgb_mse": _pearson_np(a_res, b_res),
                            "spearman_context_vs_rgb_mse": _spearman_np(a_res, b_res),
                            "pearson_context_vs_bnd_hard_core": _pearson_np(a_hard, b_hard),
                            "spearman_context_vs_bnd_hard_core": _spearman_np(a_hard, b_hard),
                        }
                    )
            del full, cfs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary: Dict[str, Any] = {"run": run, "source": "sampled object-support pixels"}
    for cf_name in ("camera_swap", "xy_swap", "CAM_CONTEXT_FIXED_CF", "EXTRA_CONTEXT_FIXED_CF"):
        for group in OUTPUT_GROUPS:
            selected = [row for row in rows if row["counterfactual"] == cf_name and row["output_group"] == group]
            vals_mse = [float(row["spearman_context_vs_rgb_mse"]) for row in selected if math.isfinite(float(row["spearman_context_vs_rgb_mse"]))]
            vals_hard = [float(row["spearman_context_vs_bnd_hard_core"]) for row in selected if math.isfinite(float(row["spearman_context_vs_bnd_hard_core"]))]
            summary[f"{cf_name}_{group}_object_support"] = {
                "mean_spearman_context_vs_rgb_mse": float(torch.tensor(vals_mse).mean().item()) if vals_mse else float("nan"),
                "mean_spearman_context_vs_bnd_hard_core": float(torch.tensor(vals_hard).mean().item()) if vals_hard else float("nan"),
            }
    return rows, summary


def _rgb_to_img(image: Tensor) -> Image.Image:
    arr = (torch.nan_to_num(image.detach().float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _gray_to_img(values: Tensor, scale: float = 1.0) -> Image.Image:
    vals = torch.nan_to_num(values.detach().float().cpu(), nan=0.0, posinf=float(scale), neginf=0.0)
    arr = (vals / max(float(scale), EPS)).clamp(0.0, 1.0)
    arr_np = (arr * 255.0).round().byte().numpy()
    return Image.fromarray(arr_np, mode="L").convert("RGB")


def _heat_to_img(values: Tensor, scale: Optional[float] = None) -> Image.Image:
    vals = torch.nan_to_num(values.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    if scale is None:
        scale = float(torch.quantile(vals.reshape(-1), 0.99).item()) if vals.numel() else 1.0
    scaled = (vals / max(float(scale), EPS)).clamp(0.0, 1.0).numpy()
    cmap = plt.get_cmap("magma")
    arr = (cmap(scaled)[..., :3] * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _mask_img(mask: Tensor, color: Tuple[int, int, int]) -> Image.Image:
    mask_np = mask.detach().bool().cpu().numpy()
    arr = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    arr[mask_np] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _overlay_mask(base: Tensor, mask: Tensor, color: Tuple[int, int, int]) -> Image.Image:
    if base.detach().ndim == 2:
        img = _gray_to_img(base, 1.0).convert("RGB")
    else:
        img = _rgb_to_img(base).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    mask_np = mask.detach().bool().cpu().numpy()
    arr[mask_np] = 0.45 * arr[mask_np] + 0.55 * np.asarray(color, dtype=np.float32)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="RGB")


def _tile(image: Image.Image, label: str, width: int) -> Image.Image:
    if image.width != width:
        height = max(1, int(round(image.height * width / max(image.width, 1))))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    label_h = 28
    canvas = Image.new("RGB", (image.width, image.height + label_h), "white")
    canvas.paste(image, (0, label_h))
    ImageDraw.Draw(canvas).text((6, 8), label, fill=(0, 0, 0))
    return canvas


def _save_sheet(
    path: Path,
    rows: Sequence[Sequence[Tuple[str, Image.Image]]],
    manifest: List[Dict[str, Any]],
    output_type: str,
    view_ids: Sequence[str],
    tile_width: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered_rows: List[Image.Image] = []
    for row in rows:
        tiles = [_tile(image, label, tile_width) for label, image in row]
        width = sum(tile.width for tile in tiles) + 6 * max(0, len(tiles) - 1)
        height = max(tile.height for tile in tiles)
        canvas = Image.new("RGB", (width, height), "white")
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, 0))
            x += tile.width + 6
        rendered_rows.append(canvas)
    if not rendered_rows:
        return
    sheet = Image.new(
        "RGB",
        (max(row.width for row in rendered_rows), sum(row.height for row in rendered_rows) + 6 * max(0, len(rendered_rows) - 1)),
        "white",
    )
    y = 0
    for row in rendered_rows:
        sheet.paste(row, (0, y))
        y += row.height + 6
    sheet.save(path)
    manifest.append(
        {
            "file_path": str(path),
            "scene": SCENE,
            "output_type": output_type,
            "view_ids": ";".join(view_ids),
            "width": sheet.width,
            "height": sheet.height,
        }
    )


def _save_bar_plot(
    path: Path,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    manifest: List[Dict[str, Any]],
    output_type: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, len(labels) * 0.55), 4))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=35, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    manifest.append({"file_path": str(path), "scene": SCENE, "output_type": output_type})


def _save_text_image(
    path: Path,
    title: str,
    lines: Sequence[str],
    manifest: List[Dict[str, Any]],
    output_type: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1500
    height = max(380, 52 + 22 * len(lines))
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((18, 18), title, fill=(0, 0, 0))
    y = 54
    for line in lines:
        draw.text((18, y), line, fill=(0, 0, 0))
        y += 22
    img.save(path)
    manifest.append({"file_path": str(path), "scene": SCENE, "output_type": output_type, "width": width, "height": height})


def _make_visuals(
    render_dir: Path,
    render_manifest: List[Dict[str, Any]],
    k1_outputs: Mapping[str, Dict[str, Tensor]],
    labels: Mapping[str, Dict[str, Tensor]],
    bnd_context_maps: Mapping[str, Dict[str, Dict[str, Tensor]]],
    cf_visual_outputs: Mapping[str, Dict[str, Dict[str, Tensor]]],
    enrichment_summary: Mapping[str, Any],
    m1_vs_bnd_summary: Mapping[str, Any],
    classification: Mapping[str, Any],
    visual_view_ids: Sequence[str],
    tile_width: int,
) -> None:
    render_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for view_id in visual_view_ids:
        out = k1_outputs[view_id]
        rows.append(
            [
                (f"{view_id} B_inf", _rgb_to_img(out["b_inf"])),
                ("beta_B", _rgb_to_img(out["medium_bs"] / (out["medium_bs"] + 1.0))),
                ("beta_D", _rgb_to_img(out["medium_attn"] / (out["medium_attn"] + 1.0))),
            ]
        )
    _save_sheet(render_dir / "contact_sheet_baseline_medium_components.png", rows, render_manifest, "baseline_medium_components", visual_view_ids, tile_width)

    rows = []
    cam_values: List[Tensor] = []
    xy_values: List[Tensor] = []
    for view_id in visual_view_ids:
        cam_map = bnd_context_maps[view_id]["camera_swap"]["beta_D"]
        xy_map = bnd_context_maps[view_id]["xy_swap"]["beta_D"]
        total_map = bnd_context_maps[view_id]["EXTRA_CONTEXT_FIXED_CF"]["beta_D"]
        cam_values.append(cam_map.reshape(-1))
        xy_values.append(xy_map.reshape(-1))
        rows.append(
            [
                (f"{view_id} camera", _heat_to_img(cam_map)),
                ("XY", _heat_to_img(xy_map)),
                ("total extra", _heat_to_img(total_map)),
            ]
        )
    _save_sheet(render_dir / "contact_sheet_context_sensitivity_maps.png", rows, render_manifest, "context_sensitivity_maps", visual_view_ids, tile_width)

    rows = []
    for view_id in visual_view_ids:
        full = cf_visual_outputs[view_id]["FULL"]
        cam_cf = cf_visual_outputs[view_id]["CAM_CONTEXT_FIXED_CF"]
        extra_cf = cf_visual_outputs[view_id]["EXTRA_CONTEXT_FIXED_CF"]
        rows.append(
            [
                (f"{view_id} FULL", _rgb_to_img(full["pred_image"])),
                ("CAM_CONTEXT_FIXED_CF", _rgb_to_img(cam_cf["pred_image"])),
                ("EXTRA_CONTEXT_FIXED_CF", _rgb_to_img(extra_cf["pred_image"])),
                ("|FULL-CAM_CF|", _rgb_to_img((full["pred_image"] - cam_cf["pred_image"]).abs() / 0.1)),
                ("|FULL-EXTRA_CF|", _rgb_to_img((full["pred_image"] - extra_cf["pred_image"]).abs() / 0.1)),
            ]
        )
    _save_sheet(render_dir / "contact_sheet_counterfactual_rgb_response.png", rows, render_manifest, "counterfactual_rgb_response", visual_view_ids, tile_width)

    rows = []
    for view_id in visual_view_ids:
        rows.append(
            [
                (f"{view_id} GT", _rgb_to_img(k1_outputs[view_id]["gt"])),
                ("M1_HIGH_J", _overlay_mask(k1_outputs[view_id]["gt"], labels[view_id]["M1_HIGH_J"], (60, 140, 255))),
                ("BND_HARD_CORE", _overlay_mask(k1_outputs[view_id]["gt"], labels[view_id]["BND_HARD_CORE"], (255, 200, 40))),
                ("persistent", _overlay_mask(k1_outputs[view_id]["gt"], labels[view_id]["PERSISTENT_BND_HARD"], (255, 80, 40))),
            ]
        )
    _save_sheet(render_dir / "contact_sheet_hard_region_overlays.png", rows, render_manifest, "hard_region_overlays", visual_view_ids, tile_width)

    rows = []
    for view_id in visual_view_ids:
        cam = bnd_context_maps[view_id]["camera_swap"]["beta_D"]
        xy = bnd_context_maps[view_id]["xy_swap"]["beta_D"]
        rows.append(
            [
                (f"{view_id} cam beta_D", _heat_to_img(cam)),
                ("cam + core", _overlay_mask(cam / max(float(cam.max().item()), EPS), labels[view_id]["BND_HARD_CORE"], (255, 220, 40))),
                ("XY beta_D", _heat_to_img(xy)),
                ("XY + core", _overlay_mask(xy / max(float(xy.max().item()), EPS), labels[view_id]["BND_HARD_CORE"], (255, 220, 40))),
            ]
        )
    _save_sheet(render_dir / "contact_sheet_sensitivity_vs_hard_region.png", rows, render_manifest, "sensitivity_vs_hard_region", visual_view_ids, tile_width)

    if cam_values:
        _save_bar_plot(
            render_dir / "plot_per_view_camera_sensitivity.png",
            list(visual_view_ids),
            [float(v.mean().item()) for v in cam_values],
            "Per-view camera-context beta_D response",
            "mean |delta beta_D|",
            render_manifest,
            "per_view_camera_sensitivity_plot",
        )
    if xy_values:
        _save_bar_plot(
            render_dir / "plot_per_view_xy_sensitivity.png",
            list(visual_view_ids),
            [float(v.mean().item()) for v in xy_values],
            "Per-view XY-context beta_D response",
            "mean |delta beta_D|",
            render_manifest,
            "per_view_xy_sensitivity_plot",
        )

    lines = [
        f"classification: {classification.get('classification')}",
        f"extra_context_used: {classification.get('extra_context_used')}",
        f"hard_region_association: {classification.get('hard_region_association')}",
        f"bound_compatible_escalation: {classification.get('bound_compatible_escalation')}",
        f"camera_vs_xy_stronger: {classification.get('camera_vs_xy_stronger')}",
        f"train core camera beta_D enrichment: {classification.get('train_core_camera_beta_D_enrichment')}",
        f"heldout core camera beta_D views >=1.5: {classification.get('heldout_core_camera_beta_D_views_ge_1p5')}",
        f"BND/M1 camera beta_D ratio: {classification.get('bnd_m1_camera_beta_D_ratio')}",
        f"BND/M1 xy beta_D ratio: {classification.get('bnd_m1_xy_beta_D_ratio')}",
    ]
    _save_text_image(render_dir / "scorecard_final_medctx.png", "BND-MEDCTX Scorecard", lines, render_manifest, "final_scorecard")

    m1_labels = []
    m1_vals = []
    for key in ("camera_swap_beta_D_ratio_BND_over_M1", "xy_swap_beta_D_ratio_BND_over_M1", "matched_los_beta_D_ratio_BND_over_M1"):
        if key in m1_vs_bnd_summary:
            m1_labels.append(key.replace("_ratio_BND_over_M1", ""))
            m1_vals.append(float(m1_vs_bnd_summary[key]))
    if m1_vals:
        _save_bar_plot(render_dir / "plot_bnd_vs_m1_context_response.png", m1_labels, m1_vals, "BND/M1 context response", "ratio", render_manifest, "bnd_vs_m1_context_plot")

    heldout_lines = []
    for key, value in enrichment_summary.items():
        if "BND_HARD_CORE" in key and ("camera_swap_beta_D" in key or "xy_swap_beta_D" in key):
            heldout_lines.append(f"{key}: {value}")
    _save_text_image(render_dir / "heldout_summary_medctx.png", "Held-out Context Audit", heldout_lines[:28], render_manifest, "heldout_summary")


def _plot_matched_los(path: Path, summaries: Mapping[str, Any], manifest: List[Dict[str, Any]]) -> None:
    labels = []
    cross_vals = []
    within_vals = []
    for group in OUTPUT_GROUPS:
        labels.append(group)
        cross_vals.append(float(summaries.get(f"cross_camera_{group}_mean_delta_l1", float("nan"))))
        within_vals.append(float(summaries.get(f"within_camera_{group}_mean_delta_l1", float("nan"))))
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, cross_vals, width, label="cross camera")
    plt.bar(x + width / 2, within_vals, width, label="within camera")
    plt.xticks(x, labels)
    plt.ylabel("mean matched-LOS delta L1")
    plt.title("Matched-LOS variation")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    manifest.append({"file_path": str(path), "scene": SCENE, "output_type": "matched_los_variation_plot"})


def _source_audit(model: Any) -> Dict[str, Any]:
    direction_dim = int(model.direction_encoding.get_out_dim())
    mode = getattr(model.config, "medium_context_mode", "dir_only")
    total_dim = direction_dim + get_medium_context_extra_dim(mode)
    return {
        "medium_context_mode": mode,
        "direction_feature": {
            "raw_direction": "Per-pixel camera ray [x,y,1] normalized in camera frame, then rotated by camera_to_worlds[:3,:3] with gsplat y/z flip.",
            "encoding": "SHEncoding(levels=4, implementation='tcnn') output passed to medium MLP.",
            "encoded_dimension": direction_dim,
            "normalization": "Raw direction is unit-normalized before SH encoding.",
        },
        "xy_feature": {
            "definition": "image_x and image_y are linspace(-1,1,width/height), r=sqrt(x^2+y^2).",
            "dimension": 3 if mode in ("dir_xy", "dir_xy_camera") else 0,
            "range": "image_x/image_y in [-1,1]; r in [0,sqrt(2)].",
            "notes": "XY is appended after encoded direction and may be partially redundant with ray direction through intrinsics.",
        },
        "camera_context": {
            "definition": "camera_center = camera.camera_to_worlds[0,:3,3], scene normalized by (camera_center-scene_center)/(scene_scale+1e-6), then multiplied by medium_camera_context_scale.",
            "dimension": 3 if mode == "dir_xy_camera" else 0,
            "scale": float(getattr(model.config, "medium_camera_context_scale", 1.0)),
            "dropout": float(getattr(model.config, "medium_camera_context_dropout", 0.0)),
            "coordinate_frame": "Scene-box normalized world coordinates.",
        },
        "medium_mlp": {
            "input_dimension": total_dim,
            "num_layers_medium": int(getattr(model.config, "num_layers_medium", 0)),
            "hidden_dim_medium": int(getattr(model.config, "hidden_dim_medium", 0)),
            "output_dimension": 9,
            "internal_activation": "nn.Sigmoid inside nerfstudio MLP hidden layers for num_layers_medium > 1; no out_activation.",
            "output_slices": {
                "0:3": "medium_rgb / B_inf through sigmoid",
                "3:6": "medium_bs / beta_B through softplus(raw + medium_density_bias)",
                "6:9": "medium_attn / beta_D through softplus(raw + medium_density_bias)",
            },
            "medium_density_bias": float(getattr(model, "medium_density_bias", 0.0)),
            "mlp_type": getattr(model.config, "mlp_type", ""),
        },
    }


def _write_source_audit_files(output_dir: Path, audit: Mapping[str, Any]) -> None:
    _write_json(output_dir / "medium_context_source_audit.json", audit)
    lines = [
        "# BND-MEDCTX Medium Context Source Audit",
        "",
        "CODE FACT: `dir_only` uses only the SH-encoded per-pixel world ray direction.",
        "CODE FACT: `dir_xy` appends normalized image coordinates `(image_x, image_y, r)`.",
        "CODE FACT: `dir_xy_camera` appends `(image_x, image_y, r)` plus scene-box normalized camera-center context.",
        f"CONFIG FACT: Active mode is `{audit['medium_context_mode']}`.",
        f"CONFIG FACT: Direction encoded dimension is `{audit['direction_feature']['encoded_dimension']}`.",
        f"CONFIG FACT: Total medium MLP input dimension is `{audit['medium_mlp']['input_dimension']}`.",
        "CODE FACT: Shared medium MLP output channels 0:3 feed sigmoid `medium_rgb`, channels 3:6 feed softplus `medium_bs`, channels 6:9 feed softplus `medium_attn`.",
        "CODE FACT: With `b_inf_mode=tied`, `B_inf = medium_rgb`.",
        "",
        "This audit treats zero/mean/swap context renders as counterfactual sensitivity probes, not as trained `dir_only` model outputs.",
    ]
    (output_dir / "medium_context_source_audit.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def _split_by_view() -> Dict[str, str]:
    return {**{view: "train" for view in TRAIN_VIEWS}, **{view: "heldout" for view in HELDOUT_VIEWS}}


def _summary_by_split(rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str], value_field: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    groups: Dict[Tuple[Any, ...], List[float]] = {}
    for row in rows:
        try:
            value = float(row[value_field])
        except Exception:
            continue
        if not math.isfinite(value):
            continue
        key = tuple(row.get(field, "") for field in key_fields)
        groups.setdefault(key, []).append(value)
    for key, values in groups.items():
        tensor = torch.tensor(values, dtype=torch.float32)
        out_key = "__".join(str(part) for part in key)
        out[out_key] = {
            "count": len(values),
            "mean": float(tensor.mean().item()),
            "p50": float(torch.quantile(tensor, 0.50).item()),
            "p90": float(torch.quantile(tensor, 0.90).item()),
        }
    return out


def _m1_vs_bnd_summary(
    bnd_camera_summary: Mapping[str, Any],
    m1_camera_summary: Mapping[str, Any],
    bnd_los_summary: Mapping[str, Any],
    m1_los_summary: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    for cf_name in ("camera_swap", "xy_swap", "CAM_CONTEXT_FIXED_CF", "EXTRA_CONTEXT_FIXED_CF"):
        for group in OUTPUT_GROUPS:
            key = f"{cf_name}_{group}"
            b = bnd_camera_summary.get(key, {})
            m = m1_camera_summary.get(key, {})
            bval = float(b.get("mean_abs_delta_over_views", float("nan"))) if isinstance(b, Mapping) else float("nan")
            mval = float(m.get("mean_abs_delta_over_views", float("nan"))) if isinstance(m, Mapping) else float("nan")
            ratio = bval / max(mval, EPS) if math.isfinite(bval) and math.isfinite(mval) else float("nan")
            rows.append({"comparison": "BND_over_M1", "counterfactual": cf_name, "output_group": group, "BND_mean_abs_delta": bval, "M1_mean_abs_delta": mval, "ratio": ratio})
            summary[f"{cf_name}_{group}_ratio_BND_over_M1"] = ratio
    for group in OUTPUT_GROUPS:
        bval = float(bnd_los_summary.get(f"cross_camera_{group}_mean_delta_l1", float("nan")))
        mval = float(m1_los_summary.get(f"cross_camera_{group}_mean_delta_l1", float("nan")))
        ratio = bval / max(mval, EPS) if math.isfinite(bval) and math.isfinite(mval) else float("nan")
        rows.append({"comparison": "BND_over_M1", "counterfactual": "matched_los_cross_camera", "output_group": group, "BND_mean_abs_delta": bval, "M1_mean_abs_delta": mval, "ratio": ratio})
        summary[f"matched_los_{group}_ratio_BND_over_M1"] = ratio
    return rows, summary


def _classify(
    bnd_context_summary: Mapping[str, Any],
    bnd_los_summary: Mapping[str, Any],
    enrichment_summary: Mapping[str, Any],
    correlation_summary: Mapping[str, Any],
    m1_vs_bnd: Mapping[str, Any],
) -> Dict[str, Any]:
    cam_beta = float(bnd_context_summary.get("camera_swap_beta_D", {}).get("mean_abs_delta_over_views", float("nan")))
    xy_beta = float(bnd_context_summary.get("xy_swap_beta_D", {}).get("mean_abs_delta_over_views", float("nan")))
    cam_a = float(bnd_context_summary.get("camera_swap_B_inf", {}).get("mean_abs_delta_over_views", float("nan")))
    xy_a = float(bnd_context_summary.get("xy_swap_B_inf", {}).get("mean_abs_delta_over_views", float("nan")))
    extra_used = any(math.isfinite(v) and v >= 1e-3 for v in (cam_beta, xy_beta, cam_a, xy_a))
    camera_vs_xy = "camera" if cam_beta >= xy_beta else "xy"
    cross_ratio_beta = float(bnd_los_summary.get("cross_over_within_beta_D_mean_ratio", float("nan")))
    matched_los_variation = bool(math.isfinite(cross_ratio_beta) and cross_ratio_beta >= 1.25)
    train_core_cam = enrichment_summary.get("camera_swap_beta_D_BND_HARD_CORE", {})
    train_core_xy = enrichment_summary.get("xy_swap_beta_D_BND_HARD_CORE", {})
    core_cam_enrich = float(train_core_cam.get("mean_enrichment_vs_object_support", float("nan"))) if isinstance(train_core_cam, Mapping) else float("nan")
    core_xy_enrich = float(train_core_xy.get("mean_enrichment_vs_object_support", float("nan"))) if isinstance(train_core_xy, Mapping) else float("nan")
    heldout_cam_views = int(train_core_cam.get("views_enrichment_ge_1p5", 0)) if isinstance(train_core_cam, Mapping) else 0
    heldout_xy_views = int(train_core_xy.get("views_enrichment_ge_1p5", 0)) if isinstance(train_core_xy, Mapping) else 0
    corr_entries = [
        value.get("mean_spearman_context_vs_rgb_mse", float("nan"))
        for key, value in correlation_summary.items()
        if isinstance(value, Mapping) and "beta_D_object_support" in key
    ]
    positive_corr = any(math.isfinite(float(v)) and float(v) > 0.10 for v in corr_entries)
    hard_association = bool(
        (math.isfinite(core_cam_enrich) and core_cam_enrich >= 1.5 and heldout_cam_views >= 2)
        or (math.isfinite(core_xy_enrich) and core_xy_enrich >= 1.5 and heldout_xy_views >= 2)
        or positive_corr
    )
    bnd_m1_cam = float(m1_vs_bnd.get("camera_swap_beta_D_ratio_BND_over_M1", float("nan")))
    bnd_m1_xy = float(m1_vs_bnd.get("xy_swap_beta_D_ratio_BND_over_M1", float("nan")))
    escalation = bool(hard_association and ((math.isfinite(bnd_m1_cam) and bnd_m1_cam > 1.1) or (math.isfinite(bnd_m1_xy) and bnd_m1_xy > 1.1)))
    if not extra_used:
        cls = "EXTRA_CONTEXT_MINIMALLY_USED"
    elif escalation:
        cls = "BOUND_COMPATIBLE_CONTEXT_ESCALATION"
    elif hard_association:
        cls = "HARD_REGION_CONTEXT_ASSOCIATION"
    elif extra_used:
        cls = "EXTRA_CONTEXT_USED_WITHOUT_HARD_REGION_ASSOCIATION"
    else:
        cls = "NOT_EVALUABLE"
    return {
        "classification": cls,
        "allowed_classifications": [
            "EXTRA_CONTEXT_MINIMALLY_USED",
            "EXTRA_CONTEXT_USED_WITHOUT_HARD_REGION_ASSOCIATION",
            "HARD_REGION_CONTEXT_ASSOCIATION",
            "BOUND_COMPATIBLE_CONTEXT_ESCALATION",
            "NOT_EVALUABLE",
        ],
        "extra_context_used": extra_used,
        "camera_vs_xy_stronger": camera_vs_xy,
        "matched_los_cross_over_within_beta_D_ratio": cross_ratio_beta,
        "matched_los_variation": matched_los_variation,
        "hard_region_association": hard_association,
        "bound_compatible_escalation": escalation,
        "train_core_camera_beta_D_enrichment": core_cam_enrich,
        "train_core_xy_beta_D_enrichment": core_xy_enrich,
        "heldout_core_camera_beta_D_views_ge_1p5": heldout_cam_views,
        "heldout_core_xy_beta_D_views_ge_1p5": heldout_xy_views,
        "bnd_m1_camera_beta_D_ratio": bnd_m1_cam,
        "bnd_m1_xy_beta_D_ratio": bnd_m1_xy,
        "gate_notes": {
            "extra_context_used_rule": "Any camera/XY swap mean abs delta for beta_D or B_inf >= 1e-3.",
            "hard_region_association_rule": "Core enrichment >=1.5 with at least 2 views, or positive object-support Spearman >0.10.",
            "escalation_rule": "Hard-region association plus BND/M1 context beta_D ratio >1.1.",
        },
    }


def _write_manifest(output_dir: Path, render_dir: Path, render_manifest: Sequence[Mapping[str, Any]]) -> None:
    output_files = [
        {"file_path": str(path), "kind": "output_file", "size_bytes": path.stat().st_size}
        for path in sorted(output_dir.glob("*"))
        if path.is_file()
    ]
    render_files = [
        {"file_path": str(path), "kind": "render_file", "size_bytes": path.stat().st_size}
        for path in sorted(render_dir.glob("*"))
        if path.is_file()
    ]
    _write_json(output_dir / "manifest.json", {"outputs": output_files, "renders": render_files, "render_manifest": list(render_manifest)})
    _write_json(render_dir / "manifest.json", {"renders": render_files, "render_manifest": list(render_manifest)})
    lines = ["# BND-MEDCTX Visual Compare Index", ""]
    for row in render_manifest:
        lines.append(f"- {row.get('output_type')}: `{row.get('file_path')}`")
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_research_note(
    repo: Path,
    repo_manifest: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    input_summary: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    jac_summary: Mapping[str, Any],
    los_summary: Mapping[str, Any],
    context_summary: Mapping[str, Any],
    enrichment_summary: Mapping[str, Any],
    m1_vs_bnd: Mapping[str, Any],
    cf_metric_summary: Mapping[str, Any],
    classification: Mapping[str, Any],
    render_manifest: Sequence[Mapping[str, Any]],
) -> None:
    RESEARCH_NOTE.parent.mkdir(parents=True, exist_ok=True)
    cls = classification.get("classification")
    lines = [
        "# BND Medium Context Utilization Audit",
        "",
        "## 1. Motivation",
        "HYPOTHESIS: Extra XY/camera conditioning may either represent legitimate medium variation or provide context-specific fitting freedom.",
        "CODE FACT: This audit is read-only and does not call optimizer/scheduler steps, densification, pruning, opacity reset, or checkpoint writes.",
        "",
        "## 2. Current formal M1 context",
        f"CONFIG FACT: Branch `{repo_manifest.get('branch')}`, HEAD `{repo_manifest.get('HEAD')}`.",
        "CONFIG FACT: Formal M1/BND medium context mode is `dir_xy_camera`, `b_inf_mode=tied`, and `B_inf=medium_rgb`.",
        "",
        "## 3. Why extra context is not automatically a flaw",
        "INFERENCE: Extra context use is representational flexibility by itself; it is not evidence of compensation unless associated with hard regions or error structure.",
        "",
        "## 4. Source audit",
        f"CODE FACT: Direction feature dimension `{source_audit['direction_feature']['encoded_dimension']}`; total medium MLP input dimension `{source_audit['medium_mlp']['input_dimension']}`.",
        "CODE FACT: XY context is `(image_x, image_y, r)` with normalized image coordinates; camera context is scene-box normalized camera center.",
        "",
        "## 5. Input semantics",
        f"QUANTITATIVE RESULT: Input summary keys: `{sorted(input_summary)[:8]}` ...",
        "",
        "## 6. Jacobian sensitivity",
        f"QUANTITATIVE RESULT: Jacobian sensitivity summary written to `{OUTPUT_DIR / 'jacobian_sensitivity_summary.json'}`.",
        f"QUANTITATIVE RESULT: Summary keys include `{sorted(jac_summary)[:8]}` ...",
        "",
        "## 7. Matched-LOS analysis",
        f"QUANTITATIVE RESULT: BND matched-LOS summary `{los_summary}`.",
        "",
        "## 8. Camera swap response",
        f"QUANTITATIVE RESULT: BND camera beta_D mean abs delta `{context_summary.get('camera_swap_beta_D', {}).get('mean_abs_delta_over_views', 'NA')}`.",
        "",
        "## 9. XY swap response",
        f"QUANTITATIVE RESULT: BND XY beta_D mean abs delta `{context_summary.get('xy_swap_beta_D', {}).get('mean_abs_delta_over_views', 'NA')}`.",
        "",
        "## 10. Counterfactual rendering",
        f"EXPERIMENTAL FACT: FULL, CAM_CONTEXT_FIXED_CF, and EXTRA_CONTEXT_FIXED_CF render metrics are saved at `{OUTPUT_DIR / 'counterfactual_rgb_metrics.csv'}`.",
        f"QUANTITATIVE RESULT: Counterfactual metric summary keys include `{sorted(cf_metric_summary)[:8]}` ...",
        "",
        "## 11. Hard-region enrichment",
        f"QUANTITATIVE RESULT: Core camera beta_D enrichment `{classification.get('train_core_camera_beta_D_enrichment')}`; core XY beta_D enrichment `{classification.get('train_core_xy_beta_D_enrichment')}`.",
        "",
        "## 12. M1 vs BND comparison",
        f"QUANTITATIVE RESULT: BND/M1 camera beta_D response ratio `{classification.get('bnd_m1_camera_beta_D_ratio')}`; XY ratio `{classification.get('bnd_m1_xy_beta_D_ratio')}`.",
        "",
        "## 13. Held-out validation",
        f"QUANTITATIVE RESULT: Held-out/core views with enrichment >=1.5: camera `{classification.get('heldout_core_camera_beta_D_views_ge_1p5')}`, XY `{classification.get('heldout_core_xy_beta_D_views_ge_1p5')}`.",
        "",
        "## 14. Classification",
        f"QUANTITATIVE RESULT: MEDCTX classification `{cls}`.",
        "INFERENCE: The classification is a diagnostic category, not a causal training-result claim.",
        "",
        "## 15. Scientific interpretation",
        "INFERENCE: The audit answers whether extra context is used and whether that use is associated with bounded-hard regions under fixed offline labels.",
        "HYPOTHESIS: Any causal claim would require a later single-factor context-reduction training experiment.",
        "",
        "## 16. Decision",
        f"INFERENCE: Context-reduced M1 training support is `{classification.get('supports_context_reduced_m1_training')}` under this audit's gate.",
        "",
        "## 17. New-mechanism roadmap",
        "HYPOTHESIS: Candidate A is OceanSplat-style object multi-view consistency focused on bounded intrinsic/direct-object consistency.",
        "HYPOTHESIS: Candidate B is a SeaFree CB-Loss split into foreground inverse-intensity weighting and background-water anchoring.",
        "HYPOTHESIS: Candidate C is cross-scene BG-anchor readiness, especially Curasao and IUI3.",
        "",
        "## 18. OceanSplat vs GMVC",
        "INFERENCE: GMVC historically targets geometry-anchored multi-view medium calibration and object/medium responsibility, while OceanSplat-style losses target virtual-view object/full-render consistency and Gaussian spatial optimization.",
        "",
        "## 19. SeaFree CB-Loss candidates",
        "INFERENCE: CB-FG and CB-BG should be tested as separate factors. Panama current locked background mask did not support CB-BG, but that does not invalidate CB-BG for other scenes.",
        "",
        "## 20. Cross-scene BG-anchor candidate",
        "HYPOTHESIS: Curasao/IUI3 should be read-only audited before any cross-scene BG-anchor training.",
        "",
        "## Output Paths",
        f"EXPERIMENTAL FACT: Output directory `{OUTPUT_DIR}`.",
        f"EXPERIMENTAL FACT: Render directory `{RENDER_DIR}`.",
        f"EXPERIMENTAL FACT: Visual index `{RENDER_DIR / 'VISUAL_COMPARE_INDEX.md'}`.",
    ]
    for row in render_manifest:
        lines.append(f"EXPERIMENTAL FACT: Visual `{row.get('output_type')}` saved at `{row.get('file_path')}`.")
    RESEARCH_NOTE.write_text("\n".join(lines) + "\n", encoding="utf8")


def _final_summary_rows(classification: Mapping[str, Any], summaries: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = [{"key": key, "value": value} for key, value in classification.items() if not isinstance(value, (dict, list))]
    for name, summary in summaries.items():
        rows.append({"key": name, "value": json.dumps(summary, default=_json_default, sort_keys=True)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/new/home_old/ycy/water-splatting-refactor"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--render-dir", type=Path, default=RENDER_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--input-samples-per-view", type=int, default=4096)
    parser.add_argument("--jacobian-samples-per-view", type=int, default=384)
    parser.add_argument("--los-samples-per-view", type=int, default=512)
    parser.add_argument("--max-los-pairs", type=int, default=20000)
    parser.add_argument("--tile-width", type=int, default=260)
    parser.add_argument("--light", action="store_true", help="Use smaller samples for a fast structural check.")
    args = parser.parse_args()

    repo = args.repo.resolve()
    output_dir = args.output_dir
    render_dir = args.render_dir
    log_dir = args.log_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.light:
        args.input_samples_per_view = min(args.input_samples_per_view, 256)
        args.jacobian_samples_per_view = min(args.jacobian_samples_per_view, 64)
        args.los_samples_per_view = min(args.los_samples_per_view, 128)
        args.max_los_pairs = min(args.max_los_pairs, 2000)

    gpu_manifest = _assert_gpu_policy()
    repo_manifest = {
        "repo": str(repo),
        "START_HEAD": _git(repo, "rev-parse", "HEAD"),
        "HEAD": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "branch", "--show-current"),
        "log_10": _git(repo, "log", "-10", "--oneline"),
        "initial_status": _git(repo, "status", "--short"),
        "diff_stat": _git(repo, "diff", "--stat"),
        "diff_check": _git(repo, "diff", "--check"),
        "python": sys.executable,
        "gpu_policy": gpu_manifest,
        "historical_untracked_protected": [
            "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py",
            "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py",
        ],
        "read_only_no_training": True,
        "train_views": list(TRAIN_VIEWS),
        "heldout_views": list(HELDOUT_VIEWS),
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)

    split_by_view = _split_by_view()
    bnd = _load_run(repo, "BND-K1", FINAL_NOMINAL_STEP)
    m1: Optional[LoadedRun] = None
    render_manifest: List[Dict[str, Any]] = []
    try:
        model = bnd.model
        model._medctx_run_name = "BND-K1"
        source_audit = _source_audit(model)
        _write_source_audit_files(output_dir, source_audit)
        all_records = _select_records(_train_records(bnd.pipeline) + _eval_records(bnd.pipeline), [*TRAIN_VIEWS, *HELDOUT_VIEWS])
        visual_view_ids = list(HELDOUT_VIEWS)

        features_by_view = {view_id: _build_medium_features(model, camera) for _idx, view_id, camera, _batch in all_records}
        reference = _context_reference(features_by_view, TRAIN_VIEWS, max_per_view=args.input_samples_per_view)
        input_rows, input_summary = _input_distribution_rows("BND-K1", features_by_view, split_by_view, args.input_samples_per_view)
        _write_csv(output_dir / "medium_input_distribution.csv", input_rows)
        _write_json(output_dir / "medium_input_distribution.json", {"rows": input_rows, "summary": input_summary})

        baseline_rows, baseline_summary = _baseline_medium_rows("BND-K1", features_by_view, split_by_view, model, args.input_samples_per_view)
        _write_csv(output_dir / "medium_output_baseline.csv", baseline_rows)
        _write_json(output_dir / "medium_output_baseline.json", {"rows": baseline_rows, "summary": baseline_summary})

        jac_rows, jac_summary = _jacobian_rows("BND-K1", model, features_by_view, split_by_view, reference, args.jacobian_samples_per_view)
        _write_csv(output_dir / "jacobian_sensitivity_per_view.csv", jac_rows)
        _write_json(output_dir / "jacobian_sensitivity_summary.json", jac_summary)

        pairs, within_rows, cross_rows, los_summary = _matched_los_rows(
            "BND-K1",
            model,
            {view: features_by_view[view] for view in TRAIN_VIEWS},
            TRAIN_VIEWS,
            args.los_samples_per_view,
            args.max_los_pairs,
        )
        _write_csv(output_dir / "matched_los_pairs.csv", pairs)
        _write_json(output_dir / "matched_los_summary.json", los_summary)
        _write_csv(output_dir / "within_camera_direction_control.csv", within_rows)
        _write_csv(output_dir / "cross_camera_direction_variation.csv", cross_rows)
        _plot_matched_los(render_dir / "plot_matched_los_cross_camera_variation.png", los_summary, render_manifest)

        labels, domains, label_meta, m1_label_maps, k1_outputs = _build_label_maps(repo, all_records)
        label_rows: List[Dict[str, Any]] = []
        for view_id, label_map in labels.items():
            region_masks = _regions_for_view(label_map, k1_outputs[view_id])
            for name, mask in region_masks.items():
                label_rows.append(
                    {
                        "split": split_by_view[view_id],
                        "view_id": view_id,
                        "region": name,
                        "pixel_count": int(mask.sum().item()),
                        "total_pixels": int(mask.numel()),
                        "coverage": float(mask.float().mean().item()),
                    }
                )
        _write_json(output_dir / "offline_region_labels.json", {"meta": label_meta, "rows": label_rows})

        context_rows, context_summary, response_maps_sparse = _context_response_maps(
            model,
            features_by_view,
            all_records,
            split_by_view,
            reference,
            max_store_views=visual_view_ids,
            labels=labels,
            baseline_outputs=k1_outputs,
            max_samples_per_view=args.input_samples_per_view,
        )
        camera_rows = [row for row in context_rows if row["counterfactual"] == "camera_swap"]
        xy_rows = [row for row in context_rows if row["counterfactual"] == "xy_swap"]
        extra_rows = [row for row in context_rows if row["counterfactual"] in ("CAM_CONTEXT_FIXED_CF", "EXTRA_CONTEXT_FIXED_CF")]
        _write_csv(output_dir / "camera_swap_response.csv", camera_rows)
        _write_json(output_dir / "camera_swap_response.json", {"rows": camera_rows, "summary": {k: v for k, v in context_summary.items() if k.startswith("camera_swap")}})
        _write_csv(output_dir / "xy_swap_response.csv", xy_rows)
        _write_json(output_dir / "xy_swap_response.json", {"rows": xy_rows, "summary": {k: v for k, v in context_summary.items() if k.startswith("xy_swap")}})
        _write_csv(output_dir / "extra_context_counterfactual.csv", extra_rows)
        _write_json(output_dir / "extra_context_counterfactual.json", {"rows": extra_rows, "summary": context_summary})

        region_rows = [row for row in context_rows if row.get("region") in REGIONS]
        region_summary = _context_region_summary(region_rows, "BND-K1")
        _write_csv(output_dir / "region_context_enrichment.csv", region_rows)
        _write_json(output_dir / "region_context_enrichment.json", {"rows": region_rows, "summary": region_summary, "label_meta": label_meta})

        corr_rows, corr_summary = _sampled_context_error_correlation_rows(
            "BND-K1",
            model,
            features_by_view,
            split_by_view,
            labels,
            k1_outputs,
            reference,
            args.input_samples_per_view,
        )
        _write_csv(output_dir / "context_error_correlation.csv", corr_rows)
        _write_json(output_dir / "context_error_correlation.json", {"rows": corr_rows, "summary": corr_summary})

        cf_metric_rows, cf_region_rows, cf_visual_outputs = _counterfactual_render_rows(
            "BND-K1",
            model,
            _select_records(all_records, visual_view_ids),
            features_by_view,
            split_by_view,
            labels,
            reference,
            visual_view_ids,
        )
        cf_metric_summary = _summary_by_split(cf_metric_rows, ("counterfactual", "output_group"), "MSE")
        _write_csv(output_dir / "counterfactual_rgb_metrics.csv", cf_metric_rows)
        _write_json(output_dir / "counterfactual_rgb_metrics.json", {"rows": cf_metric_rows, "summary": cf_metric_summary})
        _write_csv(output_dir / "counterfactual_region_response.csv", cf_region_rows)
        _write_json(output_dir / "counterfactual_region_response.json", {"rows": cf_region_rows})

        m1 = _load_run(repo, "M1", FINAL_NOMINAL_STEP)
        m1.model._medctx_run_name = "M1"
        m1_records = _select_records(_train_records(m1.pipeline) + _eval_records(m1.pipeline), [*TRAIN_VIEWS, *HELDOUT_VIEWS])
        m1_features = {view_id: _build_medium_features(m1.model, camera) for _idx, view_id, camera, _batch in m1_records}
        m1_reference = _context_reference(m1_features, TRAIN_VIEWS, max_per_view=args.input_samples_per_view)
        m1_context_rows, m1_context_summary, _ = _context_response_maps(
            m1.model,
            m1_features,
            m1_records,
            split_by_view,
            m1_reference,
            max_store_views=[],
            labels=None,
            baseline_outputs=None,
            max_samples_per_view=args.input_samples_per_view,
        )
        m1_pairs, _m1_within, _m1_cross, m1_los_summary = _matched_los_rows(
            "M1",
            m1.model,
            {view: m1_features[view] for view in TRAIN_VIEWS},
            TRAIN_VIEWS,
            args.los_samples_per_view,
            args.max_los_pairs,
        )
        m1bnd_rows, m1bnd_summary = _m1_vs_bnd_summary(context_summary, m1_context_summary, los_summary, m1_los_summary)
        _write_csv(output_dir / "m1_vs_bnd_context_response.csv", [*m1bnd_rows, *m1_context_rows])
        _write_json(
            output_dir / "m1_vs_bnd_context_response.json",
            {
                "rows": m1bnd_rows,
                "BND_context_summary": context_summary,
                "M1_context_summary": m1_context_summary,
                "BND_matched_los_summary": los_summary,
                "M1_matched_los_summary": m1_los_summary,
                "summary": m1bnd_summary,
            },
        )
        _write_json(output_dir / "m1_matched_los_summary.json", m1_los_summary)
        _write_csv(output_dir / "m1_matched_los_pairs.csv", m1_pairs)

        heldout_rows = [
            row
            for row in [*region_rows, *corr_rows, *context_rows]
            if row.get("split") == "heldout"
        ]
        heldout_summary = {
            "region_enrichment": {k: v for k, v in region_summary.items() if isinstance(v, Mapping)},
            "context_error_correlation": {k: v for k, v in corr_summary.items() if isinstance(v, Mapping)},
        }
        _write_csv(output_dir / "heldout_context_audit.csv", heldout_rows)
        _write_json(output_dir / "heldout_context_audit.json", {"rows": heldout_rows, "summary": heldout_summary})

        classification = _classify(context_summary, los_summary, region_summary, corr_summary, m1bnd_summary)
        classification["supports_context_reduced_m1_training"] = bool(
            classification["classification"] in ("HARD_REGION_CONTEXT_ASSOCIATION", "BOUND_COMPATIBLE_CONTEXT_ESCALATION")
        )
        classification["close_medctx_concern"] = bool(
            classification["classification"] in ("EXTRA_CONTEXT_MINIMALLY_USED", "EXTRA_CONTEXT_USED_WITHOUT_HARD_REGION_ASSOCIATION")
        )
        _write_json(output_dir / "medctx_classification.json", classification)

        _make_visuals(
            render_dir,
            render_manifest,
            {view: k1_outputs[view] for view in visual_view_ids},
            labels,
            response_maps_sparse,
            cf_visual_outputs,
            region_summary,
            m1bnd_summary,
            classification,
            visual_view_ids,
            args.tile_width,
        )
        _write_csv(output_dir / "bnd_medctx_final_summary.csv", _final_summary_rows(classification, {"los": los_summary, "context": context_summary, "m1_vs_bnd": m1bnd_summary}))
        _write_json(
            output_dir / "bnd_medctx_final_summary.json",
            {
                "classification": classification,
                "source_audit": source_audit,
                "input_summary": input_summary,
                "baseline_summary": baseline_summary,
                "jacobian_summary": jac_summary,
                "matched_los_summary": los_summary,
                "context_summary": context_summary,
                "region_enrichment_summary": region_summary,
                "context_error_correlation_summary": corr_summary,
                "m1_vs_bnd_summary": m1bnd_summary,
                "counterfactual_rgb_metric_summary": cf_metric_summary,
                "render_manifest": render_manifest,
            },
        )
        _write_research_note(
            repo,
            repo_manifest,
            source_audit,
            input_summary,
            baseline_summary,
            jac_summary,
            los_summary,
            context_summary,
            region_summary,
            m1bnd_summary,
            cf_metric_summary,
            classification,
            render_manifest,
        )
        _write_manifest(output_dir, render_dir, render_manifest)
    finally:
        _release(m1)
        _release(bnd)


if __name__ == "__main__":
    main()
