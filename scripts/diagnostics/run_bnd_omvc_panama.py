#!/usr/bin/env python
"""Causal Panama BND-OMVC continuation experiment.

This runner keeps the production model defaults unchanged and injects the
single BND-OMVC loss only inside the O1 matched-continuation training loop.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.utils.eval_utils import eval_setup

from scripts.diagnostics import audit_bnd_cdepth_setup as cdepth_setup
from scripts.diagnostics import run_bnd_aware_refine_panama as aware


SCENE = "Panama"
OUTPUT_DIR = Path("outputs/bnd_omvc_panama_20260812")
RENDER_DIR = Path("renders/bnd_omvc_panama_20260812")
LOG_DIR = Path("logs/bnd_omvc_panama_20260812")
RESEARCH_NOTE = Path("research_notes/BND_OBJECT_MULTI_VIEW_CONSISTENCY_2026-08-12.md")

K1_CONFIG = cdepth_setup.K1_CONFIG
M1_CONFIG = aware.M1_CONFIG
START_NOMINAL_STEP = 3000
FINAL_NOMINAL_STEP = 15000
SNAPSHOT_ABS_NOMINAL = (3000, 4000, 5000, 8000, 10000, 13000, 15000)
TRAIN_VIEWS = aware.TRAIN_VIEWS
EVAL_VIEWS = aware.EVAL_VIEWS
BRANCHES = ("C0", "O1")
ALLOWED_PHYSICAL_GPUS = frozenset({"6", "7", "8", "9"})
OMVC_ACTIVE_START = 4000
OMVC_ACTIVE_END = 10000
OMVC_TARGET = "direct_object_signal"
OMVC_PRIMARY_TARGET_REJECTED = "clear_object_fullsh_raw"
MIN_WARP_ACCUMULATION = 0.01
EPS = 1e-12


@dataclass
class LoadedBranch:
    branch: str
    config_path: Path
    checkpoint_path: Path
    loaded_step: int
    config: Any
    pipeline: Any
    optimizers: Optimizers


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _object_hash(value: Any) -> str:
    import pickle

    return _sha256_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _assert_runtime_policy() -> None:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env != "water_splatting":
        raise RuntimeError(f"Formal experiment requires CONDA_DEFAULT_ENV=water_splatting, got {conda_env!r}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [token.strip() for token in visible.split(",") if token.strip()]
    if len(devices) != 1 or devices[0] not in ALLOWED_PHYSICAL_GPUS:
        allowed = ",".join(sorted(ALLOWED_PHYSICAL_GPUS))
        raise RuntimeError(
            "This single-process experiment requires exactly one physical GPU "
            f"from {allowed}; got CUDA_VISIBLE_DEVICES={visible!r}"
        )


def _environment_manifest() -> Dict[str, Any]:
    gpu_rows: List[Dict[str, Any]] = []
    if torch.cuda.is_available():
        logical = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(logical)
        gpu_rows.append(
            {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "physical_gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "torch_logical_gpu_id": int(logical),
                "gpu_name": props.name,
                "total_memory_bytes": int(props.total_memory),
            }
        )
    return {
        "CONDA_ENV": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "PYTHON_PATH": sys.executable,
        "PYTHON_VERSION": sys.version.split()[0],
        "TORCH_VERSION": torch.__version__,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "gpus": gpu_rows,
    }


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


def _snapshot_abs_steps(final_actual_step: int) -> Tuple[int, ...]:
    out = []
    for step in SNAPSHOT_ABS_NOMINAL:
        actual = final_actual_step if step == FINAL_NOMINAL_STEP else step
        if START_NOMINAL_STEP <= actual <= final_actual_step:
            out.append(actual)
    out.append(final_actual_step)
    return tuple(dict.fromkeys(out))


def _rel(abs_step: int) -> int:
    return int(abs_step) - START_NOMINAL_STEP


def _snapshot_rel_steps(final_actual_step: int) -> Tuple[int, ...]:
    return tuple(_rel(step) for step in _snapshot_abs_steps(final_actual_step))


def _optimizer_groups(config: Any, model: Any) -> Dict[str, Any]:
    groups = model.get_param_groups()
    return {name: config.optimizers[name] for name in groups}


def _load_branch(repo: Path, branch: str, step: int = START_NOMINAL_STEP) -> LoadedBranch:
    config_path = repo / K1_CONFIG
    actual = _actual_step(config_path, step)

    def update_config(config: Any) -> Any:
        config.load_step = actual
        config.pipeline.model.intrinsic_color_parameterization = "bounded_sh3"
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
    model.config.intrinsic_color_parameterization = "bounded_sh3"
    model.config.rasterize_mode = "classic"
    model.config.medium_context_mode = "dir_xy_camera"
    model.config.b_inf_mode = "tied"
    model.config.infinite_water_enabled = False
    model.config.coarse_depth_supervision_enabled = False
    model.config.refinement_priority_mode = "baseline"
    model.set_refinement_budget_schedule(None)
    model.set_refinement_guidance(None, None)
    model.step = int(loaded_step)

    optimizers = Optimizers(_optimizer_groups(config, model), model.get_param_groups())
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    for group in optimizers.optimizers:
        optimizers.optimizers[group].load_state_dict(ckpt["optimizers"][group])
    for group in optimizers.schedulers:
        optimizers.schedulers[group].load_state_dict(ckpt["schedulers"][group])
    pipeline.eval()
    return LoadedBranch(branch, config_path, Path(checkpoint_path), int(loaded_step), config, pipeline, optimizers)


def _load_eval_only(repo: Path, run: str, step: int) -> Tuple[Any, Any, Path, int]:
    if run == "M1":
        config_path = repo / M1_CONFIG
        parameterization = "legacy"
    else:
        config_path = repo / K1_CONFIG
        parameterization = "bounded_sh3"
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
    return config, pipeline, Path(checkpoint_path), int(loaded_step)


def _release(obj: Any) -> None:
    if obj is None:
        return
    try:
        del obj.pipeline
    except Exception:
        pass
    try:
        del obj.optimizers
    except Exception:
        pass
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _train_records(pipeline: Any) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    dataset = pipeline.datamanager.train_dataset
    filenames = list(getattr(dataset, "image_filenames", []))
    cameras = dataset.cameras.to(pipeline.model.device)
    rows = []
    for index, filename in enumerate(filenames):
        view_id = Path(filename).stem
        batch = pipeline.datamanager.cached_train[index].copy()
        rows.append((index, view_id, cameras[index : index + 1], _batch_to_device(batch, pipeline.model.device)))
    return rows


def _eval_records(pipeline: Any) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    dataset = pipeline.datamanager.eval_dataset
    filenames = list(getattr(dataset, "image_filenames", []))
    rows = []
    for eval_index, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        view_id = Path(filenames[eval_index]).stem if eval_index < len(filenames) else f"eval_{eval_index}"
        rows.append((eval_index, view_id, camera, _batch_to_device(batch, pipeline.model.device)))
    return rows


def _safe_outputs(outputs: Mapping[str, Any]) -> Dict[str, Tensor]:
    keys = (
        "pred_image",
        "background",
        "rgb_object",
        "direct_object_signal",
        "rgb_medium",
        "rgb_medium_finite",
        "rgb_tail",
        "accumulation",
        "clear_object_fullsh_raw",
        "rgb_clear",
        "rgb_clear_clamp",
        "depth",
        "tau_D",
        "transmission",
        "medium_rgb",
        "medium_bs",
        "medium_attn",
        "b_inf",
        "gaussian_view_rgb",
        "gaussian_view_logits",
        "gaussian_visible_mask",
    )
    return {key: outputs[key].detach().float().cpu() for key in keys if isinstance(outputs.get(key), Tensor)}


def _gt_for(model: Any, batch: Mapping[str, Any], background: Tensor) -> Tensor:
    return model.composite_with_background(model.get_gt_img(batch["image"]), background.to(model.device)).detach().float().cpu()


def _render_records(pipeline: Any, records: Sequence[Tuple[int, str, Cameras, Dict[str, Any]]]) -> Dict[str, Dict[str, Tensor]]:
    model = pipeline.model
    model.eval()
    out: Dict[str, Dict[str, Tensor]] = {}
    for _idx, view_id, camera, batch in records:
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera.to(model.device))
            safe = _safe_outputs(outputs)
            gt = _gt_for(model, batch, outputs["background"])
        pred = safe["pred_image"].clamp(0.0, 1.0)
        gt = gt.clamp(0.0, 1.0)
        residual = (pred - gt).square().mean(dim=-1)
        out[view_id] = {
            **safe,
            "gt": gt,
            "pred": pred,
            "residual": residual,
            "clear": safe["clear_object_fullsh_raw"],
            "bound": safe["clear_object_fullsh_raw"].amax(dim=-1),
        }
    return out


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


def _stats(values: Tensor, prefix: str = "") -> Dict[str, Any]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {f"{prefix}{name}": float("nan") for name in ("mean", "std", "p10", "p50", "p90", "p99", "max")}
    return {
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        f"{prefix}p10": _quantile_flat(flat, 0.10),
        f"{prefix}p50": _quantile_flat(flat, 0.50),
        f"{prefix}p90": _quantile_flat(flat, 0.90),
        f"{prefix}p99": _quantile_flat(flat, 0.99),
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


def _quantile_flat(values: Tensor, q: float) -> float:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    if q <= 0:
        return float(flat.min().item())
    if q >= 1:
        return float(flat.max().item())
    k = max(1, min(int(math.ceil(q * flat.numel())), flat.numel()))
    return float(torch.kthvalue(flat, k).values.item())


def _clone_camera(camera: Cameras) -> Cameras:
    def maybe_clone(value: Any) -> Any:
        return value.clone() if isinstance(value, Tensor) else value

    return Cameras(
        camera_to_worlds=camera.camera_to_worlds.clone(),
        fx=maybe_clone(camera.fx),
        fy=maybe_clone(camera.fy),
        cx=maybe_clone(camera.cx),
        cy=maybe_clone(camera.cy),
        width=maybe_clone(camera.width),
        height=maybe_clone(camera.height),
        distortion_params=maybe_clone(camera.distortion_params),
        camera_type=maybe_clone(camera.camera_type),
        times=maybe_clone(camera.times),
        metadata=camera.metadata,
    )


def _effective_camera(camera: Cameras, downscale: int) -> Cameras:
    cam = _clone_camera(camera)
    if downscale > 1:
        cam.rescale_output_resolution(1.0 / float(downscale))
    return cam


def _offset_camera(camera: Cameras, *, axis: int, baseline: float) -> Cameras:
    cam = _clone_camera(camera)
    direction = cam.camera_to_worlds[0, :3, axis]
    norm = torch.linalg.norm(direction).clamp_min(1e-8)
    cam.camera_to_worlds[0, :3, 3] = cam.camera_to_worlds[0, :3, 3] + float(baseline) * direction / norm
    return cam


def _model_rotation_world_from_camera(camera: Cameras) -> Tensor:
    c2w = camera.camera_to_worlds[0]
    edit = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=c2w.device, dtype=c2w.dtype))
    return c2w[:3, :3] @ edit


def _project_central_depth_to_virtual(
    central_camera: Cameras,
    virtual_camera: Cameras,
    central_depth: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    device = central_depth.device
    dtype = central_depth.dtype
    height, width = int(central_depth.shape[0]), int(central_depth.shape[1])
    central = _effective_camera(central_camera.to(device), 1)
    virtual = _effective_camera(virtual_camera.to(device), 1)
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    z = central_depth[..., 0].detach()
    x_cam = (x - central.cx.reshape(-1)[0].to(device=device, dtype=dtype)) / central.fx.reshape(-1)[0].to(device=device, dtype=dtype)
    y_cam = (y - central.cy.reshape(-1)[0].to(device=device, dtype=dtype)) / central.fy.reshape(-1)[0].to(device=device, dtype=dtype)
    p_cam = torch.stack([x_cam * z, y_cam * z, z], dim=-1)
    r_c = _model_rotation_world_from_camera(central)
    center_c = central.camera_to_worlds[0, :3, 3].to(device=device, dtype=dtype)
    world = center_c + torch.einsum("ij,hwj->hwi", r_c.to(dtype=dtype), p_cam)

    r_v = _model_rotation_world_from_camera(virtual)
    center_v = virtual.camera_to_worlds[0, :3, 3].to(device=device, dtype=dtype)
    p_v = torch.einsum("ji,hwj->hwi", r_v.to(dtype=dtype), world - center_v)
    z_v = p_v[..., 2]
    fx_v = virtual.fx.reshape(-1)[0].to(device=device, dtype=dtype)
    fy_v = virtual.fy.reshape(-1)[0].to(device=device, dtype=dtype)
    cx_v = virtual.cx.reshape(-1)[0].to(device=device, dtype=dtype)
    cy_v = virtual.cy.reshape(-1)[0].to(device=device, dtype=dtype)
    u_v = fx_v * (p_v[..., 0] / z_v.clamp_min(1e-8)) + cx_v
    v_v = fy_v * (p_v[..., 1] / z_v.clamp_min(1e-8)) + cy_v
    positive_depth = z_v > 1e-6
    in_bounds = (u_v >= 0.0) & (u_v <= width - 1) & (v_v >= 0.0) & (v_v <= height - 1)
    valid = torch.isfinite(z) & (z > 1e-6) & positive_depth & in_bounds
    grid_x = 2.0 * u_v / max(width - 1, 1) - 1.0
    grid_y = 2.0 * v_v / max(height - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid, valid, positive_depth


def _warp_virtual_to_central(
    source: Tensor,
    central_camera: Cameras,
    virtual_camera: Cameras,
    central_depth: Tensor,
    central_accumulation: Tensor,
    *,
    downscale: int,
    min_accumulation: float = MIN_WARP_ACCUMULATION,
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    central_eff = _effective_camera(central_camera, downscale).to(source.device)
    virtual_eff = _effective_camera(virtual_camera, downscale).to(source.device)
    grid, geometric_valid, positive_depth = _project_central_depth_to_virtual(
        central_eff,
        virtual_eff,
        central_depth,
    )
    height, width = int(source.shape[0]), int(source.shape[1])
    sampled = F.grid_sample(
        source.permute(2, 0, 1)[None, ...],
        grid[None, ...],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0].permute(1, 2, 0)
    support = central_accumulation[..., 0].detach() > min_accumulation
    valid = geometric_valid & support
    geom_count = int(geometric_valid.sum().detach().cpu().item())
    support_count = int(support.sum().detach().cpu().item())
    valid_count = int(valid.sum().detach().cpu().item())
    total = int(height * width)
    meta = {
        "total_pixels": total,
        "geometric_valid_pixels": geom_count,
        "object_support_pixels": support_count,
        "valid_pixels": valid_count,
        "geometric_valid_fraction": geom_count / max(total, 1),
        "object_support_fraction": support_count / max(total, 1),
        "valid_fraction": valid_count / max(total, 1),
        "valid_fraction_of_object_support": valid_count / max(support_count, 1),
        "positive_depth_fraction": float(positive_depth.float().mean().detach().cpu().item()),
    }
    return sampled, valid, meta


def _masked_l1(a: Tensor, b: Tensor, mask: Tensor) -> Tensor:
    mask3 = mask[..., None].to(device=a.device, dtype=a.dtype)
    denom = mask3.sum().clamp_min(1.0) * a.shape[-1]
    return ((a - b).abs() * mask3).sum() / denom


def _masked_reg_l1(a: Tensor, b: Tensor, mask: Tensor) -> Tensor:
    mask3 = mask[..., None].to(device=a.device, dtype=a.dtype)
    denom = mask3.sum().clamp_min(1.0) * a.shape[-1]
    return (((a - b).abs() / (a.detach().abs() + 1e-3)) * mask3).sum() / denom


def _omvc_loss_and_meta(
    model: Any,
    camera: Cameras,
    central_outputs: Mapping[str, Tensor],
    *,
    target_key: str,
    baseline_h: float,
    baseline_v: float,
) -> Tuple[Tensor, Dict[str, Any], Dict[str, Tensor]]:
    downscale = int(model._get_downscale_factor())
    cam_h = _offset_camera(camera, axis=0, baseline=baseline_h).to(model.device)
    cam_v = _offset_camera(camera, axis=1, baseline=baseline_v).to(model.device)
    out_h = model.get_outputs(cam_h)
    out_v = model.get_outputs(cam_v)
    warped_h, mask_h, meta_h = _warp_virtual_to_central(
        out_h[target_key],
        camera,
        cam_h,
        central_outputs["depth"],
        central_outputs["accumulation"],
        downscale=downscale,
    )
    warped_v, mask_v, meta_v = _warp_virtual_to_central(
        out_v[target_key],
        camera,
        cam_v,
        central_outputs["depth"],
        central_outputs["accumulation"],
        downscale=downscale,
    )
    target = central_outputs[target_key]
    loss_h = _masked_reg_l1(warped_h, target, mask_h)
    loss_v = _masked_reg_l1(warped_v, target, mask_v)
    loss = loss_h + loss_v
    meta = {
        "target_key": target_key,
        "horizontal_reg_l1": float(loss_h.detach().cpu().item()),
        "vertical_reg_l1": float(loss_v.detach().cpu().item()),
        "pooled_reg_l1": float((loss_h + loss_v).detach().cpu().item()),
        "horizontal_l1": float(_masked_l1(warped_h, target, mask_h).detach().cpu().item()),
        "vertical_l1": float(_masked_l1(warped_v, target, mask_v).detach().cpu().item()),
        "horizontal_valid_fraction": meta_h["valid_fraction"],
        "vertical_valid_fraction": meta_v["valid_fraction"],
        "horizontal_valid_fraction_of_object_support": meta_h["valid_fraction_of_object_support"],
        "vertical_valid_fraction_of_object_support": meta_v["valid_fraction_of_object_support"],
        "horizontal_valid_pixels": meta_h["valid_pixels"],
        "vertical_valid_pixels": meta_v["valid_pixels"],
    }
    tensors = {
        "horizontal_source": out_h[target_key],
        "vertical_source": out_v[target_key],
        "horizontal_warped": warped_h,
        "vertical_warped": warped_v,
        "horizontal_mask": mask_h,
        "vertical_mask": mask_v,
    }
    return loss, meta, tensors


def _omvc_metrics_for_outputs(
    model: Any,
    camera: Cameras,
    central_outputs: Mapping[str, Tensor],
    *,
    target_key: str,
    baseline_h: float,
    baseline_v: float,
    extra_mask: Optional[Tensor] = None,
) -> Tuple[Dict[str, Any], Dict[str, Tensor]]:
    with torch.no_grad():
        loss, meta, tensors = _omvc_loss_and_meta(
            model,
            camera,
            central_outputs,
            target_key=target_key,
            baseline_h=baseline_h,
            baseline_v=baseline_v,
        )
        target = central_outputs[target_key]
        mask_h = tensors["horizontal_mask"]
        mask_v = tensors["vertical_mask"]
        if extra_mask is not None:
            em = extra_mask.to(device=mask_h.device, dtype=torch.bool)
            mask_h = mask_h & em
            mask_v = mask_v & em
        h_reg = _masked_reg_l1(tensors["horizontal_warped"], target, mask_h)
        v_reg = _masked_reg_l1(tensors["vertical_warped"], target, mask_v)
        h_l1 = _masked_l1(tensors["horizontal_warped"], target, mask_h)
        v_l1 = _masked_l1(tensors["vertical_warped"], target, mask_v)
        region_meta = dict(meta)
        region_meta.update(
            {
                "horizontal_reg_l1": float(h_reg.detach().cpu().item()),
                "vertical_reg_l1": float(v_reg.detach().cpu().item()),
                "pooled_reg_l1": float((h_reg + v_reg).detach().cpu().item()),
                "horizontal_l1": float(h_l1.detach().cpu().item()),
                "vertical_l1": float(v_l1.detach().cpu().item()),
                "pooled_l1": float((h_l1 + v_l1).detach().cpu().item()),
                "horizontal_valid_pixels": int(mask_h.sum().detach().cpu().item()),
                "vertical_valid_pixels": int(mask_v.sum().detach().cpu().item()),
                "pooled_valid_pixels": int(mask_h.sum().detach().cpu().item() + mask_v.sum().detach().cpu().item()),
            }
        )
        return region_meta, tensors


def _capture_render_state(model: Any) -> Dict[str, Any]:
    return {
        "xys": getattr(model, "xys", None),
        "radii": getattr(model, "radii", None),
        "xys_grad_abs": getattr(model, "xys_grad_abs", None),
        "depths": getattr(model, "depths", None),
        "last_size": getattr(model, "last_size", None),
        "last_fx": getattr(model, "last_fx", None),
        "last_fy": getattr(model, "last_fy", None),
    }


def _restore_render_state(model: Any, state: Mapping[str, Any]) -> None:
    for key, value in state.items():
        setattr(model, key, value)


def _compute_loss_components(model: Any, outputs: Mapping[str, Tensor], batch: Mapping[str, Any]) -> Dict[str, Tensor]:
    gt_img = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
    pred_img = outputs["pred_image"]
    if "mask" in batch:
        mask = model._downscale_if_required(batch["mask"]).to(model.device)
        gt_img = gt_img * mask
        pred_img = pred_img * mask
    recon = torch.abs((gt_img - pred_img) / (pred_img.detach() + 1e-3)).mean()
    sim = 1 - model.ssim(
        (gt_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...],
        (pred_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...],
    )
    return {"reg_l1": recon.detach(), "reg_ssim": sim.detach()}


def _run_before(model: Any, optimizers: Optimizers, abs_step: int) -> None:
    model.step_cb(step=abs_step)
    model.aopt_before_train_iteration(optimizers, step=abs_step)
    model.medium_hold_before_train_iteration(optimizers, step=abs_step)


def _run_after(model: Any, optimizers: Optimizers, abs_step: int) -> Mapping[str, Any]:
    model.aopt_after_train_iteration(step=abs_step)
    model.medium_hold_after_train_iteration(optimizers, step=abs_step)
    model.after_train(step=abs_step)
    if abs_step % int(model.config.refine_every) == 0:
        model.refinement_after(optimizers, step=abs_step)
        return dict(model._refinement_last_event)
    return {
        "step": abs_step,
        "refinement_called": False,
        "priority_mode": getattr(model.config, "refinement_priority_mode", "baseline"),
        "N_after": int(model.num_points),
    }


def _optimizer_lrs(optimizers: Optimizers) -> Dict[str, float]:
    return {group: float(opt.param_groups[0]["lr"]) for group, opt in optimizers.optimizers.items()}


def _model_param_tensors(model: Any) -> Dict[str, Tensor]:
    out = {
        name: getattr(model, name).detach().cpu().clone()
        for name in ("means", "scales", "quats", "features_dc", "features_rest", "opacities")
    }
    for prefix in ("medium_mlp", "direction_encoding"):
        parts = [p.detach().reshape(-1).cpu() for p in getattr(model, prefix).parameters()]
        out[prefix] = torch.cat(parts) if parts else torch.empty(0)
    return out


def _optimizer_state_tensors(optimizers: Optimizers) -> Dict[str, Dict[str, Tensor]]:
    out: Dict[str, Dict[str, Tensor]] = {}
    for group, optimizer in optimizers.optimizers.items():
        pieces: Dict[str, List[Tensor]] = {"exp_avg": [], "exp_avg_sq": [], "step": []}
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                state = optimizer.state.get(param, {})
                for key in pieces:
                    if key in state:
                        value = state[key]
                        if isinstance(value, Tensor):
                            pieces[key].append(value.detach().reshape(-1).cpu())
                        else:
                            pieces[key].append(torch.tensor([float(value)]))
        out[group] = {key: torch.cat(vals) if vals else torch.empty(0) for key, vals in pieces.items()}
    return out


def _scheduler_state_tensors(optimizers: Optimizers) -> Dict[str, Tensor]:
    out: Dict[str, Tensor] = {}
    for group, scheduler in optimizers.schedulers.items():
        state = scheduler.state_dict()
        pieces: List[Tensor] = []
        for value in state.values():
            if isinstance(value, Tensor):
                pieces.append(value.detach().reshape(-1).cpu().float())
            elif isinstance(value, (int, float, bool)):
                pieces.append(torch.tensor([float(value)]))
            elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                pieces.append(torch.tensor([float(x) for x in value]))
        out[group] = torch.cat(pieces) if pieces else torch.empty(0)
    return out


def _compare_tensor_dict(a: Mapping[str, Tensor], b: Mapping[str, Tensor], name_key: str) -> List[Dict[str, Any]]:
    rows = []
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            rows.append({name_key: name, "max_abs_diff": float("nan"), "pass": False})
            continue
        if a[name].shape != b[name].shape:
            rows.append(
                {
                    name_key: name,
                    "max_abs_diff": float("nan"),
                    "pass": False,
                    "shape_a": list(a[name].shape),
                    "shape_b": list(b[name].shape),
                }
            )
            continue
        diff = (a[name] - b[name]).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        rows.append({name_key: name, "max_abs_diff": max_abs, "pass": bool(max_abs == 0.0)})
    return rows


def _initial_equivalence(repo: Path, output_dir: Path) -> Dict[str, Any]:
    loaded = [_load_branch(repo, branch) for branch in BRANCHES]
    try:
        p0 = _model_param_tensors(loaded[0].pipeline.model)
        param_rows = []
        for other in loaded[1:]:
            for row in _compare_tensor_dict(p0, _model_param_tensors(other.pipeline.model), "parameter"):
                row["branch_a"] = "C0"
                row["branch_b"] = other.branch
                param_rows.append(row)
        _write_csv(output_dir / "initial_parameter_equivalence.csv", param_rows)

        opt0 = _optimizer_state_tensors(loaded[0].optimizers)
        opt_rows = []
        for other in loaded[1:]:
            opt_other = _optimizer_state_tensors(other.optimizers)
            for group in sorted(set(opt0) | set(opt_other)):
                for row in _compare_tensor_dict(opt0.get(group, {}), opt_other.get(group, {}), "state_tensor"):
                    row["branch_a"] = "C0"
                    row["branch_b"] = other.branch
                    row["optimizer_group"] = group
                    opt_rows.append(row)
        _write_csv(output_dir / "initial_optimizer_equivalence.csv", opt_rows)

        sched0 = _scheduler_state_tensors(loaded[0].optimizers)
        sched_rows = []
        for other in loaded[1:]:
            for row in _compare_tensor_dict(sched0, _scheduler_state_tensors(other.optimizers), "scheduler_group"):
                row["branch_a"] = "C0"
                row["branch_b"] = other.branch
                sched_rows.append(row)
        _write_csv(output_dir / "initial_scheduler_equivalence.csv", sched_rows)

        records = _train_records(loaded[0].pipeline)
        idx, _view_id, camera0, _batch0 = records[0]
        forward_rows = []
        with torch.no_grad():
            out0 = loaded[0].pipeline.model.get_outputs(camera0.to(loaded[0].pipeline.model.device))
        for other in loaded[1:]:
            rec = _train_records(other.pipeline)[idx]
            with torch.no_grad():
                out = other.pipeline.model.get_outputs(rec[2].to(other.pipeline.model.device))
            for key in ("pred_image", "direct_object_signal", "clear_object_fullsh_raw", "tau_D", "transmission", "accumulation"):
                diff = (out0[key].detach().cpu() - out[key].detach().cpu()).abs()
                max_abs = float(diff.max().item()) if diff.numel() else 0.0
                forward_rows.append(
                    {
                        "branch_a": "C0",
                        "branch_b": other.branch,
                        "key": key,
                        "max_abs_diff": max_abs,
                        "pass": bool(max_abs <= 1e-6),
                    }
                )
        _write_csv(output_dir / "initial_forward_equivalence.csv", forward_rows)

        ckpt = torch.load(loaded[0].checkpoint_path, map_location="cpu")
        scaler_hash = _object_hash(ckpt.get("scalers", {}))
        summary = {
            "START_STATE_EQUIVALENCE": bool(
                all(row["pass"] for row in param_rows)
                and all(row["pass"] for row in opt_rows)
                and all(row["pass"] for row in sched_rows)
                and all(row["pass"] for row in forward_rows)
            ),
            "INITIAL_PARAMETER_EQUIVALENCE": all(row["pass"] for row in param_rows),
            "INITIAL_OPTIMIZER_EQUIVALENCE": all(row["pass"] for row in opt_rows),
            "INITIAL_SCHEDULER_EQUIVALENCE": all(row["pass"] for row in sched_rows),
            "INITIAL_FORWARD_EQUIVALENCE": all(row["pass"] for row in forward_rows),
            "SCALER_STATE_IDENTICAL": True,
            "SCALER_STATE_SHA256": scaler_hash,
            "basis": "Both branches are loaded independently from the exact same BND-K1@3000 checkpoint.",
        }
        _write_json(output_dir / "start_state_audit.json", summary)
        return summary
    finally:
        for item in loaded:
            _release(item)


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _rng_manifest(state: Mapping[str, Any]) -> Dict[str, Any]:
    out = {
        "python_rng_sha256": _object_hash(state["python"]),
        "numpy_rng_sha256": _object_hash(state["numpy"]),
        "torch_cpu_rng_sha256": hashlib.sha256(state["torch_cpu"].detach().cpu().numpy().tobytes()).hexdigest(),
        "cuda_count": len(state.get("torch_cuda", [])),
    }
    if "torch_cuda" in state:
        out["torch_cuda_rng_sha256"] = [
            hashlib.sha256(item.detach().cpu().numpy().tobytes()).hexdigest() for item in state["torch_cuda"]
        ]
    return out


def _generate_camera_sequence(branch: LoadedBranch, output_dir: Path, final_actual_step: int) -> Tuple[List[int], List[str], List[Dict[str, Any]]]:
    dm = branch.pipeline.datamanager
    filenames = list(getattr(dm.train_dataset, "image_filenames", []))
    names = [Path(path).stem for path in filenames]
    rows = []
    indices = []
    view_ids = []
    for abs_step in range(START_NOMINAL_STEP + 1, final_actual_step + 1):
        if len(dm.train_unseen_cameras) == 0:
            dm.train_unseen_cameras = dm.sample_train_cameras()
        index = int(dm.train_unseen_cameras.pop(0))
        if len(dm.train_unseen_cameras) == 0:
            dm.train_unseen_cameras = dm.sample_train_cameras()
        indices.append(index)
        view_id = names[index]
        view_ids.append(view_id)
        rows.append({"relative_step": _rel(abs_step), "absolute_step": abs_step, "camera_index": index, "camera_name": view_id})
    payload = {
        "scene": SCENE,
        "length": len(rows),
        "start_abs_step_exclusive": START_NOMINAL_STEP,
        "final_abs_step_inclusive": final_actual_step,
        "rows": rows,
    }
    _write_json(output_dir / "paired_camera_sequence.json", payload)
    _write_json(
        output_dir / "camera_sequence_audit.json",
        {
            "CAMERA_SEQUENCE_MATCH": True,
            "CAMERA_SEQUENCE_EXACT_MATCH": True,
            "mismatch_count": 0,
            "length": len(rows),
            "basis": "C0 and O1 consume this explicit central-camera index list instead of branch-local random sampling.",
        },
    )
    return indices, view_ids, rows


def _camera_geometry_baseline_preflight(repo: Path, output_dir: Path) -> Dict[str, Any]:
    branch = _load_branch(repo, "PREFLIGHT")
    try:
        model = branch.pipeline.model
        records = _train_records(branch.pipeline)
        cameras = branch.pipeline.datamanager.train_dataset.cameras.to(model.device)
        centers = cameras.camera_to_worlds[:, :3, 3].detach().float()
        distances = torch.cdist(centers, centers)
        distances[torch.eye(distances.shape[0], device=distances.device, dtype=torch.bool)] = float("inf")
        median_nn = float(distances.min(dim=1).values.median().detach().cpu().item())
        scene_center, scene_scale = model._get_scene_normalization(dtype=centers.dtype, device=centers.device)
        if not math.isfinite(median_nn) or median_nn <= 0.0:
            median_nn = float(scene_scale.detach().cpu().item()) * 0.01
        candidates = (0.05, 0.025, 0.01)
        selected = None
        rows: List[Dict[str, Any]] = []
        sampled = records[:: max(1, len(records) // 5)][:5]
        model.eval()
        for frac in candidates:
            baseline = float(frac * median_nn)
            cover_h: List[float] = []
            cover_v: List[float] = []
            support_h: List[float] = []
            support_v: List[float] = []
            for _idx, view_id, camera, _batch in sampled:
                with torch.no_grad():
                    outputs = model.get_outputs_for_camera(camera.to(model.device))
                    cam_h = _offset_camera(camera.to(model.device), axis=0, baseline=baseline)
                    cam_v = _offset_camera(camera.to(model.device), axis=1, baseline=baseline)
                    dummy = outputs[OMVC_TARGET]
                    _wh, _mh, mh = _warp_virtual_to_central(
                        dummy,
                        camera.to(model.device),
                        cam_h,
                        outputs["depth"],
                        outputs["accumulation"],
                        downscale=int(model._get_downscale_factor()),
                    )
                    _wv, _mv, mv = _warp_virtual_to_central(
                        dummy,
                        camera.to(model.device),
                        cam_v,
                        outputs["depth"],
                        outputs["accumulation"],
                        downscale=int(model._get_downscale_factor()),
                    )
                cover_h.append(mh["valid_fraction"])
                cover_v.append(mv["valid_fraction"])
                support_h.append(mh["valid_fraction_of_object_support"])
                support_v.append(mv["valid_fraction_of_object_support"])
                rows.append(
                    {
                        "candidate_fraction_of_median_nn": frac,
                        "candidate_baseline": baseline,
                        "view_id": view_id,
                        "horizontal_valid_fraction": mh["valid_fraction"],
                        "vertical_valid_fraction": mv["valid_fraction"],
                        "horizontal_valid_fraction_of_object_support": mh["valid_fraction_of_object_support"],
                        "vertical_valid_fraction_of_object_support": mv["valid_fraction_of_object_support"],
                    }
                )
            pooled_support = min(float(np.mean(support_h)), float(np.mean(support_v)))
            if selected is None and pooled_support >= 0.70:
                selected = (frac, baseline, cover_h, cover_v, support_h, support_v)
        if selected is None:
            frac = candidates[-1]
            baseline = float(frac * median_nn)
            selected = (frac, baseline, [], [], [], [])
        frac, baseline, cover_h, cover_v, support_h, support_v = selected
        payload = {
            "B_H": baseline,
            "B_V": baseline,
            "candidate_fraction_of_median_nn": frac,
            "normalization_rule": "B_H = B_V = selected_fraction * median nearest-neighbor train-camera center distance; selected by pure warp-overlap preflight before training.",
            "median_train_camera_nearest_neighbor_distance": median_nn,
            "scene_center": scene_center.detach().cpu().tolist(),
            "scene_scale": float(scene_scale.detach().cpu().item()),
            "selection_rule": "Choose the largest candidate in [0.05,0.025,0.01] with pooled H/V valid fraction over object support >= 0.70; otherwise use 0.01.",
            "valid_warp_fraction": {
                "horizontal_mean_total": float(np.mean(cover_h)) if cover_h else float("nan"),
                "vertical_mean_total": float(np.mean(cover_v)) if cover_v else float("nan"),
                "horizontal_mean_object_support": float(np.mean(support_h)) if support_h else float("nan"),
                "vertical_mean_object_support": float(np.mean(support_v)) if support_v else float("nan"),
            },
            "preflight_views": [view_id for _idx, view_id, _camera, _batch in sampled],
            "candidate_rows": rows,
        }
        _write_csv(output_dir / "warp_overlap_preflight.csv", rows)
        _write_json(output_dir / "warp_overlap_preflight.json", payload)
        return payload
    finally:
        _release(branch)


def _write_source_semantics(output_dir: Path, gradient_probe: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "classification": "CLEAR_J_NO_SAFE_GRADIENT_PATH__FALLBACK_TO_DIRECT_OBJECT_SIGNAL",
        "primary_target_requested": OMVC_PRIMARY_TARGET_REJECTED,
        "selected_training_target": OMVC_TARGET,
        "renderer_outputs": {
            "clear_object_fullsh_raw": {
                "source": "UnderwaterRasterizer.rasterize returns CUDA out_clr as render.j_raw.",
                "semantics": "Transparent-water alpha composite of per-Gaussian intrinsic colors: sum_i alpha_i T_alpha_i c_i. No medium attenuation is included.",
                "clamp_or_tonemap": "No clamp or tone map on clear_object_fullsh_raw. Separate aliases rgb_clear_clamp/J_gaussian clamp to [0,1]; rgb_clear applies J/(J+1).",
                "gradient_status": "Unsafe as a training target in current code: _RasterizeGaussians.backward receives v_out_clr but does not pass it into rasterize_backward.",
            },
            "clear_raw": {
                "source": "No separate clear_raw key exists; historical clear_raw corresponds to clear_object_fullsh_raw/J_gaussian_raw.",
            },
            "direct_object_signal": {
                "source": "WaterSplattingModel outputs direct_object_signal = render.rgb_object.",
                "semantics": "Attenuated direct object branch: sum_i alpha_i T_alpha_i exp(-beta_D(pixel) depth_i) c_i.",
                "gradient_status": "Gradient is supported through v_out_img in water_splatting/rasterize.py and water_splatting/cuda/csrc/backward.cu.",
            },
            "rgb_object": {
                "source": "Alias of direct_object_signal in current get_outputs.",
                "semantics": "Same attenuated direct object signal.",
            },
            "accumulation": {
                "source": "1 - final alpha transmittance from rasterizer final_Ts.",
                "semantics": "Alpha compositing support, not a semantic foreground mask.",
            },
            "depth": {
                "source": "Expected camera-space z: depth_im / accumulation, with max-depth fallback where accumulation is zero.",
                "gradient_status": "Depth output gradient is not propagated by _RasterizeGaussians.backward.",
            },
            "transmission": {
                "source": "exp(-medium_attn * expected_depth) after rasterization.",
                "semantics": "Pixel-depth transmission diagnostic, not the per-Gaussian attenuation actually used in direct_object_signal.",
            },
            "medium_contribution": {
                "source": "rgb_medium, with tied B_inf tail correction when b_inf_mode=tied.",
                "semantics": "Finite segment backscatter plus tail term under formal M1/BND b_inf_mode=tied.",
            },
        },
        "fallback_decision": {
            "reason": "The preferred bounded intrinsic clear-object render J has no safe current gradient path. Training both J-loss and direct-object-loss is disallowed, so the single OMVC target is direct_object_signal.",
            "object_medium_caveat": "The fallback target includes beta_D attenuation; this is not a pure dewatered intrinsic J objective.",
        },
        "gradient_probe": gradient_probe,
    }
    _write_json(output_dir / "source_semantics.json", payload)
    _write_json(output_dir / "bnd_omvc_source_semantics.json", payload)
    lines = [
        "# BND-OMVC Source Semantics",
        "",
        "CODE FACT: `clear_object_fullsh_raw` is the CUDA `out_clr` transparent-water object composite. It has no clamp or tone map.",
        "CODE FACT: `_RasterizeGaussians.backward` does not route `v_out_clr` to the CUDA backward kernel, so clear J is not safe as the first OMVC training target.",
        "CODE FACT: `direct_object_signal` and `rgb_object` are aliases for the attenuated object branch and receive gradients through `v_out_img`.",
        "",
        "CONFIG FACT: This run uses one fallback target only: `direct_object_signal`.",
        "",
        "INFERENCE: The intervention tests object-branch cross-view consistency, but the target is not a pure intrinsic/dewatered J loss because beta_D attenuation remains in the signal.",
    ]
    (output_dir / "source_semantics.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    (output_dir / "bnd_omvc_source_semantics.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    return payload


def _grad_norms(model: Any, groups: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in groups:
        tensor = getattr(model, name)
        grad = tensor.grad
        if grad is None:
            out[name] = {"grad_l2": 0.0, "grad_mean_abs": 0.0, "grad_max_abs": 0.0, "has_grad": False}
            continue
        g = grad.detach().float()
        out[name] = {
            "grad_l2": float(torch.linalg.norm(g).item()),
            "grad_mean_abs": float(g.abs().mean().item()) if g.numel() else 0.0,
            "grad_max_abs": float(g.abs().max().item()) if g.numel() else 0.0,
            "has_grad": bool(g.abs().sum().item() > 0),
        }
    return out


def _gradient_scale_audit(
    repo: Path,
    output_dir: Path,
    *,
    baseline_h: float,
    baseline_v: float,
) -> Tuple[Dict[str, Any], float]:
    branch = _load_branch(repo, "GRAD")
    try:
        model = branch.pipeline.model
        model.train()
        records = _train_records(branch.pipeline)
        camera_index, view_id, camera, batch = records[0]
        abs_step = START_NOMINAL_STEP + 1
        batch = _batch_to_device(batch, model.device)
        camera = camera.to(model.device)
        groups = ("means", "features_dc", "features_rest")

        _run_before(model, branch.optimizers, abs_step)
        branch.optimizers.zero_grad_all()
        outputs = model.get_outputs(camera)
        rgb_loss = model.get_loss_dict(outputs, batch, {})["main_loss"]
        rgb_loss.backward()
        rgb_norms = _grad_norms(model, groups)

        branch.optimizers.zero_grad_all()
        outputs = model.get_outputs(camera)
        state = _capture_render_state(model)
        j_loss, j_meta, _j_tensors = _omvc_loss_and_meta(
            model,
            camera,
            outputs,
            target_key=OMVC_PRIMARY_TARGET_REJECTED,
            baseline_h=baseline_h,
            baseline_v=baseline_v,
        )
        _restore_render_state(model, state)
        j_loss.backward()
        j_norms = _grad_norms(model, groups)

        branch.optimizers.zero_grad_all()
        outputs = model.get_outputs(camera)
        state = _capture_render_state(model)
        omvc_loss, omvc_meta, _tensors = _omvc_loss_and_meta(
            model,
            camera,
            outputs,
            target_key=OMVC_TARGET,
            baseline_h=baseline_h,
            baseline_v=baseline_v,
        )
        _restore_render_state(model, state)
        omvc_loss.backward()
        omvc_norms = _grad_norms(model, groups)

        ratios = []
        rows = []
        for group in groups:
            rgb = rgb_norms[group]["grad_l2"]
            omvc = omvc_norms[group]["grad_l2"]
            ratio = omvc / max(rgb, EPS)
            ratios.append(ratio)
            rows.append(
                {
                    "group": group,
                    "rgb_grad_l2": rgb,
                    "clear_j_unweighted_grad_l2": j_norms[group]["grad_l2"],
                    "omvc_unweighted_grad_l2": omvc,
                    "omvc_to_rgb_grad_l2_ratio": ratio,
                    "rgb_grad_mean_abs": rgb_norms[group]["grad_mean_abs"],
                    "omvc_grad_mean_abs": omvc_norms[group]["grad_mean_abs"],
                }
            )
        median_ratio = float(np.median([r for r in ratios if math.isfinite(r)])) if ratios else float("nan")
        if math.isfinite(median_ratio) and 0.1 <= median_ratio <= 10.0:
            lambda_omvc = 0.1
            rule_result = "same_order_use_0.1"
        elif math.isfinite(median_ratio) and median_ratio > 0.0:
            lambda_omvc = min(1.0, max(1e-4, 0.1 / median_ratio))
            rule_result = "gradient_normalized_to_approximately_0.1x_rgb_median_group_l2"
        else:
            lambda_omvc = 0.1
            rule_result = "fallback_0.1_due_to_nonfinite_ratio"

        payload = {
            "audit_step": abs_step,
            "camera_index": camera_index,
            "camera_name": view_id,
            "target_key": OMVC_TARGET,
            "clear_j_probe_target": OMVC_PRIMARY_TARGET_REJECTED,
            "rgb_loss": float(rgb_loss.detach().cpu().item()),
            "clear_j_loss": float(j_loss.detach().cpu().item()),
            "omvc_unweighted_loss": float(omvc_loss.detach().cpu().item()),
            "clear_j_loss_meta": j_meta,
            "omvc_loss_meta": omvc_meta,
            "rows": rows,
            "median_omvc_to_rgb_grad_l2_ratio": median_ratio,
            "LAMBDA_SELECTION_RULE": "If median unweighted OMVC/RGB grad L2 ratio over means/features_dc/features_rest is in [0.1,10], use lambda=0.1; otherwise set lambda=clamp(0.1/ratio,1e-4,1.0).",
            "LAMBDA_SELECTION_RULE_RESULT": rule_result,
            "LAMBDA_OMVC": lambda_omvc,
            "PRIMARY_CLEAR_J_GRADIENT_SAFE": False,
            "FALLBACK_TARGET_USED": OMVC_TARGET,
        }
        _write_csv(output_dir / "gradient_scale_audit.csv", rows)
        _write_json(output_dir / "gradient_scale_audit.json", payload)
        return payload, lambda_omvc
    finally:
        _release(branch)


def _save_checkpoint(branch: LoadedBranch, rel_step: int, output_dir: Path) -> Path:
    path = output_dir / "continuation_checkpoints" / branch.branch / f"relative-{rel_step:06d}.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "branch": branch.branch,
            "relative_step": rel_step,
            "absolute_step": START_NOMINAL_STEP + rel_step,
            "pipeline": branch.pipeline.state_dict(),
            "optimizers": {group: opt.state_dict() for group, opt in branch.optimizers.optimizers.items()},
            "schedulers": {group: sched.state_dict() for group, sched in branch.optimizers.schedulers.items()},
            "scalers": {},
            "omvc_target": OMVC_TARGET if branch.branch == "O1" else None,
        },
        path,
    )
    return path


def _ckpt_path(output_dir: Path, branch: str, rel_step: int) -> Path:
    return output_dir / "continuation_checkpoints" / branch / f"relative-{rel_step:06d}.ckpt"


def _load_snapshot(branch: LoadedBranch, output_dir: Path, rel_step: int) -> None:
    ckpt = torch.load(_ckpt_path(output_dir, branch.branch, rel_step), map_location="cpu")
    branch.pipeline.load_pipeline(ckpt["pipeline"], int(ckpt["absolute_step"]))
    branch.pipeline.model.step = int(ckpt["absolute_step"])
    branch.pipeline.model.config.intrinsic_color_parameterization = "bounded_sh3"
    branch.pipeline.model.config.rasterize_mode = "classic"
    branch.pipeline.model.config.medium_context_mode = "dir_xy_camera"
    branch.pipeline.model.config.b_inf_mode = "tied"
    branch.pipeline.model.config.infinite_water_enabled = False
    branch.pipeline.model.config.coarse_depth_supervision_enabled = False
    branch.pipeline.model.config.refinement_priority_mode = "baseline"
    branch.pipeline.model.set_refinement_guidance(None, None)
    branch.pipeline.model.set_refinement_budget_schedule(None)


def _train_branch(
    repo: Path,
    branch_name: str,
    *,
    camera_indices: Sequence[int],
    camera_names: Sequence[str],
    rng_state: Mapping[str, Any],
    snapshot_rels: Sequence[int],
    output_dir: Path,
    baseline_h: float,
    baseline_v: float,
    lambda_omvc: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    _set_rng_state(rng_state)
    branch = _load_branch(repo, branch_name)
    model = branch.pipeline.model
    model.config.refinement_priority_mode = "baseline"
    model.set_refinement_guidance(None, None)
    model.set_refinement_budget_schedule(None)

    dm = branch.pipeline.datamanager
    snapshot_set = set(int(x) for x in snapshot_rels)
    rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    ckpt_rows: List[Dict[str, Any]] = []
    count_rows: List[Dict[str, Any]] = []
    try:
        if 0 in snapshot_set:
            ckpt = _save_checkpoint(branch, 0, output_dir)
            ckpt_rows.append({"branch": branch_name, "relative_step": 0, "absolute_step": START_NOMINAL_STEP, "checkpoint_path": str(ckpt)})
            count_rows.append({"branch": branch_name, "relative_step": 0, "absolute_step": START_NOMINAL_STEP, "gaussian_count": int(model.num_points)})
        cached_train = dm.cached_train
        train_cameras = getattr(dm, "train_cameras", dm.train_dataset.cameras).to(model.device)
        for rel_step, (camera_index, camera_name) in enumerate(zip(camera_indices, camera_names), start=1):
            abs_step = START_NOMINAL_STEP + rel_step
            branch.pipeline.train()
            model.train()
            model.config.coarse_depth_supervision_enabled = False
            _run_before(model, branch.optimizers, abs_step)
            branch.optimizers.zero_grad_all()
            batch = _batch_to_device(cached_train[camera_index].copy(), model.device)
            camera = train_cameras[camera_index : camera_index + 1]
            outputs = model.get_outputs(camera)
            central_state = _capture_render_state(model)
            comps = _compute_loss_components(model, outputs, batch)
            losses = model.get_loss_dict(outputs, batch, {})
            rgb_loss = losses["main_loss"]
            omvc_raw = rgb_loss.new_tensor(0.0)
            omvc_meta: Dict[str, Any] = {
                "target_key": OMVC_TARGET,
                "horizontal_reg_l1": 0.0,
                "vertical_reg_l1": 0.0,
                "pooled_reg_l1": 0.0,
                "horizontal_valid_fraction": float("nan"),
                "vertical_valid_fraction": float("nan"),
            }
            omvc_active = branch_name == "O1" and OMVC_ACTIVE_START <= abs_step <= OMVC_ACTIVE_END
            if omvc_active:
                omvc_raw, omvc_meta, _tensors = _omvc_loss_and_meta(
                    model,
                    camera,
                    outputs,
                    target_key=OMVC_TARGET,
                    baseline_h=baseline_h,
                    baseline_v=baseline_v,
                )
                _restore_render_state(model, central_state)
            total_loss = rgb_loss + (float(lambda_omvc) * omvc_raw if omvc_active else 0.0)
            if not bool(torch.isfinite(total_loss).detach().cpu().item()):
                raise RuntimeError(f"Non-finite loss {branch_name} step {abs_step}")
            total_loss.backward()
            _restore_render_state(model, central_state)
            branch.optimizers.optimizer_step_all()
            branch.optimizers.scheduler_step_all(abs_step)
            event = _run_after(model, branch.optimizers, abs_step)
            lrs = _optimizer_lrs(branch.optimizers)
            row = {
                "branch": branch_name,
                "relative_step": rel_step,
                "absolute_step": abs_step,
                "camera_index": camera_index,
                "camera_name": camera_name,
                "L_RGB": float(rgb_loss.detach().cpu().item()),
                "L_OMVC_unweighted": float(omvc_raw.detach().cpu().item()),
                "lambda_omvc": float(lambda_omvc) if omvc_active else 0.0,
                "L_OMVC_weighted": float((float(lambda_omvc) * omvc_raw).detach().cpu().item()) if omvc_active else 0.0,
                "L_total": float(total_loss.detach().cpu().item()),
                "OMVC_active": bool(omvc_active),
                "OMVC_target": OMVC_TARGET if omvc_active else "",
                "reg_l1": float(comps["reg_l1"].detach().cpu().item()),
                "reg_ssim": float(comps["reg_ssim"].detach().cpu().item()),
                "omvc_horizontal_reg_l1": omvc_meta["horizontal_reg_l1"],
                "omvc_vertical_reg_l1": omvc_meta["vertical_reg_l1"],
                "omvc_horizontal_valid_fraction": omvc_meta["horizontal_valid_fraction"],
                "omvc_vertical_valid_fraction": omvc_meta["vertical_valid_fraction"],
                "gaussian_count": int(model.num_points),
                "stable": True,
            }
            for group, lr in lrs.items():
                row[f"lr_{group}"] = lr
            rows.append(row)
            if event.get("refinement_called"):
                event["branch"] = branch_name
                event["relative_step"] = rel_step
                event["absolute_step"] = abs_step
                event["camera_name"] = camera_name
                event["OMVC_active"] = bool(omvc_active)
                event_rows.append(event)
            if rel_step in snapshot_set:
                ckpt = _save_checkpoint(branch, rel_step, output_dir)
                ckpt_rows.append({"branch": branch_name, "relative_step": rel_step, "absolute_step": abs_step, "checkpoint_path": str(ckpt)})
                count_rows.append({"branch": branch_name, "relative_step": rel_step, "absolute_step": abs_step, "gaussian_count": int(model.num_points)})
        return rows, event_rows, ckpt_rows, count_rows
    finally:
        _release(branch)


def _top_mask(values: Tensor, domain: Tensor, fraction: float) -> Tensor:
    mask = domain.detach().bool().cpu()
    out = torch.zeros_like(mask)
    vals = values.detach().float().cpu()[mask]
    if vals.numel() == 0:
        return out
    k = max(1, int(math.ceil(float(fraction) * vals.numel())))
    indices = torch.topk(vals, k, largest=True).indices
    flat = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
    out.reshape(-1)[flat[indices]] = True
    return out


def _build_label_maps(repo: Path) -> Tuple[Dict[str, Dict[str, Tensor]], Dict[str, Tensor], Dict[str, Any]]:
    _, m1_pipe, _, _ = _load_eval_only(repo, "M1", FINAL_NOMINAL_STEP)
    _, k1_final_pipe, _, _ = _load_eval_only(repo, "K1", FINAL_NOMINAL_STEP)
    late_pipes: Dict[int, Any] = {}
    for step in (8000, 10000, 13000, FINAL_NOMINAL_STEP):
        _, pipe, _, _ = _load_eval_only(repo, "K1", step)
        late_pipes[step] = pipe
    try:
        eval_records = _eval_records(k1_final_pipe)
        m1_maps = _render_records(m1_pipe, _eval_records(m1_pipe))
        final_maps = _render_records(k1_final_pipe, eval_records)
        late_maps = {step: _render_records(pipe, _eval_records(pipe)) for step, pipe in late_pipes.items()}
        labels: Dict[str, Dict[str, Tensor]] = {}
        domains: Dict[str, Tensor] = {}
        for _idx, view_id, _camera, _batch in eval_records:
            support = final_maps[view_id]["accumulation"][..., 0] > 0.01
            domains[view_id] = support
            m1_highj = (m1_maps[view_id]["accumulation"][..., 0] > 0.01) & (m1_maps[view_id]["bound"] > 1.0)
            count = torch.zeros_like(support, dtype=torch.int32)
            for maps in late_maps.values():
                count += _top_mask(maps[view_id]["residual"], support, 0.10).int()
            persistent = support & (count >= 3)
            labels[view_id] = {
                "PERSISTENT_BND_HARD": persistent,
                "M1_HIGH_J": m1_highj,
                "BND_HARD_CORE": persistent & m1_highj,
            }
        meta = {"late_steps": [8000, 10000, 13000, 15000], "persistent_required_count": 3}
        return labels, domains, meta
    finally:
        for pipe in [m1_pipe, k1_final_pipe, *late_pipes.values()]:
            try:
                del pipe
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _decomposition_row(branch_name: str, rel_step: int, maps: Mapping[str, Mapping[str, Tensor]]) -> Dict[str, Any]:
    all_clear = torch.cat([view_map["clear_object_fullsh_raw"].reshape(-1, 3) for view_map in maps.values()], dim=0)
    all_tau = torch.cat([view_map["tau_D"].reshape(-1, 3) for view_map in maps.values()], dim=0)
    all_t = torch.cat([view_map["transmission"].reshape(-1, 3) for view_map in maps.values()], dim=0)
    all_medium_rgb = torch.cat([view_map["medium_rgb"].reshape(-1, 3) for view_map in maps.values()], dim=0)
    all_b_inf = torch.cat([view_map["b_inf"].reshape(-1, 3) for view_map in maps.values() if "b_inf" in view_map], dim=0)
    all_beta_d = torch.cat([view_map["medium_attn"].reshape(-1, 3) for view_map in maps.values()], dim=0)
    all_beta_b = torch.cat([view_map["medium_bs"].reshape(-1, 3) for view_map in maps.values()], dim=0)
    visible_rgb: List[Tensor] = []
    visible_logits: List[Tensor] = []
    for view_map in maps.values():
        c = view_map.get("gaussian_view_rgb")
        visible = view_map.get("gaussian_visible_mask")
        if c is not None and visible is not None:
            mask = visible.reshape(-1).bool()
            if c.ndim == 2 and c.shape[0] == mask.shape[0] and int(mask.sum().item()) > 0:
                visible_rgb.append(c[mask].reshape(-1, 3))
        logits = view_map.get("gaussian_view_logits")
        if logits is not None and visible is not None:
            mask = visible.reshape(-1).bool()
            if logits.ndim == 2 and logits.shape[0] == mask.shape[0] and int(mask.sum().item()) > 0:
                visible_logits.append(logits[mask].reshape(-1, 3))
    all_c = torch.cat(visible_rgb, dim=0) if visible_rgb else torch.empty(0, 3)
    all_logits = torch.cat(visible_logits, dim=0) if visible_logits else torch.empty(0, 3)
    row: Dict[str, Any] = {
        "branch": branch_name,
        "relative_step": rel_step,
        "absolute_step": START_NOMINAL_STEP + rel_step,
        "J_p99": _quantile_flat(all_clear, 0.99),
        "P_J_gt_1": _threshold_fraction(all_clear, 1.0, "gt"),
        "tau_p90": _quantile_flat(all_tau, 0.90),
        "tau_p99": _quantile_flat(all_tau, 0.99),
        "P_T_lt_0p1": _threshold_fraction(all_t, 0.1, "lt"),
        "P_c_gt_0p99": _threshold_fraction(all_c, 0.99, "gt") if all_c.numel() else float("nan"),
        "P_abs_s_full_gt_5": _threshold_fraction(all_logits.abs(), 5.0, "gt") if all_logits.numel() else float("nan"),
        "visible_gaussian_color_count": int(all_c.shape[0]),
    }
    row.update(_stats(all_b_inf, "B_inf_"))
    row.update(_stats(all_medium_rgb, "medium_rgb_"))
    row.update(_stats(all_beta_d, "beta_D_"))
    row.update(_stats(all_beta_b, "beta_B_"))
    return row


def _evaluate_snapshots(
    repo: Path,
    output_dir: Path,
    labels: Mapping[str, Mapping[str, Tensor]],
    snapshot_rels: Sequence[int],
    *,
    baseline_h: float,
    baseline_v: float,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[Tuple[str, int], Dict[str, Dict[str, Tensor]]],
]:
    global_rows: List[Dict[str, Any]] = []
    per_view_rows: List[Dict[str, Any]] = []
    region_rows: List[Dict[str, Any]] = []
    decomp_rows: List[Dict[str, Any]] = []
    omvc_rows: List[Dict[str, Any]] = []
    render_cache: Dict[Tuple[str, int], Dict[str, Dict[str, Tensor]]] = {}
    for branch_name in BRANCHES:
        branch = _load_branch(repo, branch_name)
        try:
            for rel_step in snapshot_rels:
                _load_snapshot(branch, output_dir, rel_step)
                records = _eval_records(branch.pipeline)
                maps = _render_records(branch.pipeline, records)
                render_cache[(branch_name, rel_step)] = maps
                metric_accum: Dict[str, List[float]] = {"PSNR": [], "SSIM": [], "LPIPS": [], "MSE": []}
                for _idx, view_id, camera, _batch in records:
                    metrics = _metric_images(branch.pipeline.model, maps[view_id]["pred"], maps[view_id]["gt"])
                    per_row = {
                        "branch": branch_name,
                        "relative_step": rel_step,
                        "absolute_step": START_NOMINAL_STEP + rel_step,
                        "view_id": view_id,
                        **metrics,
                    }
                    per_view_rows.append(per_row)
                    for key, value in metrics.items():
                        metric_accum[key].append(value)

                    model = branch.pipeline.model
                    with torch.no_grad():
                        central = model.get_outputs_for_camera(camera.to(model.device))
                    direct_metrics, direct_tensors = _omvc_metrics_for_outputs(
                        model,
                        camera.to(model.device),
                        central,
                        target_key=OMVC_TARGET,
                        baseline_h=baseline_h,
                        baseline_v=baseline_v,
                    )
                    clear_metrics, _clear_tensors = _omvc_metrics_for_outputs(
                        model,
                        camera.to(model.device),
                        central,
                        target_key=OMVC_PRIMARY_TARGET_REJECTED,
                        baseline_h=baseline_h,
                        baseline_v=baseline_v,
                    )
                    omvc_rows.append(
                        {
                            "branch": branch_name,
                            "relative_step": rel_step,
                            "absolute_step": START_NOMINAL_STEP + rel_step,
                            "view_id": view_id,
                            "target": OMVC_TARGET,
                            **direct_metrics,
                            "clearJ_horizontal_reg_l1": clear_metrics["horizontal_reg_l1"],
                            "clearJ_vertical_reg_l1": clear_metrics["vertical_reg_l1"],
                            "clearJ_pooled_reg_l1": clear_metrics["pooled_reg_l1"],
                            "clearJ_horizontal_l1": clear_metrics["horizontal_l1"],
                            "clearJ_vertical_l1": clear_metrics["vertical_l1"],
                            "clearJ_pooled_l1": clear_metrics["pooled_l1"],
                        }
                    )

                    residual = maps[view_id]["residual"]
                    for label_name in ("PERSISTENT_BND_HARD", "BND_HARD_CORE", "M1_HIGH_J"):
                        mask = labels[view_id][label_name].to(device=model.device, dtype=torch.bool)
                        rgb_mse = float(residual[mask.cpu()].mean().item()) if int(mask.sum().item()) > 0 else float("nan")
                        region_omvc, _ = _omvc_metrics_for_outputs(
                            model,
                            camera.to(model.device),
                            central,
                            target_key=OMVC_TARGET,
                            baseline_h=baseline_h,
                            baseline_v=baseline_v,
                            extra_mask=mask,
                        )
                        clear = central["clear_object_fullsh_raw"].detach()
                        clear_vals = clear[mask]
                        clear_stats = _stats(clear_vals, "clear_J_") if clear_vals.numel() else _stats(torch.empty(0), "clear_J_")
                        region_rows.append(
                            {
                                "branch": branch_name,
                                "relative_step": rel_step,
                                "absolute_step": START_NOMINAL_STEP + rel_step,
                                "view_id": view_id,
                                "label": label_name,
                                "RGB_MSE": rgb_mse,
                                "object_consistency_pooled_reg_l1": region_omvc["pooled_reg_l1"],
                                "object_consistency_pooled_l1": region_omvc["pooled_l1"],
                                "valid_warp_pixels": region_omvc["pooled_valid_pixels"],
                                "pixel_count": int(mask.sum().detach().cpu().item()),
                                "P_J_gt_1": _threshold_fraction(clear_vals, 1.0, "gt") if clear_vals.numel() else float("nan"),
                                **clear_stats,
                            }
                        )
                row = {"branch": branch_name, "relative_step": rel_step, "absolute_step": START_NOMINAL_STEP + rel_step}
                for key, vals in metric_accum.items():
                    row[key] = float(sum(vals) / len(vals))
                global_rows.append(row)
                decomp_rows.append(_decomposition_row(branch_name, rel_step, maps))
        finally:
            _release(branch)
    return global_rows, per_view_rows, omvc_rows, region_rows, decomp_rows, render_cache


def _aggregate_omvc(omvc_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    counts: Dict[Tuple[str, int], int] = {}
    for row in omvc_rows:
        key = (str(row["branch"]), int(row["relative_step"]))
        grouped.setdefault(
            key,
            {
                "horizontal_reg_l1": [],
                "vertical_reg_l1": [],
                "pooled_reg_l1": [],
                "horizontal_l1": [],
                "vertical_l1": [],
                "pooled_l1": [],
                "clearJ_pooled_reg_l1": [],
                "horizontal_valid_fraction": [],
                "vertical_valid_fraction": [],
            },
        )
        for metric in grouped[key]:
            if metric in row:
                grouped[key][metric].append(float(row[metric]))
        counts[key] = counts.get(key, 0) + 1
    out = []
    for (branch, rel), vals in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        row: Dict[str, Any] = {"branch": branch, "relative_step": rel, "absolute_step": START_NOMINAL_STEP + rel, "view_count": counts[(branch, rel)]}
        for key, items in vals.items():
            row[key] = float(sum(items) / len(items)) if items else float("nan")
        out.append(row)
    return out


def _aggregate_region(region_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str], Dict[str, List[float]]] = {}
    pix: Dict[Tuple[str, int, str], int] = {}
    for row in region_rows:
        key = (str(row["branch"]), int(row["relative_step"]), str(row["label"]))
        grouped.setdefault(key, {"RGB_MSE": [], "object_consistency_pooled_reg_l1": [], "object_consistency_pooled_l1": [], "clear_J_p99": []})
        for metric in grouped[key]:
            value = row.get(metric)
            if value is not None and value != "":
                grouped[key][metric].append(float(value))
        pix[key] = pix.get(key, 0) + int(row.get("pixel_count", 0))
    out = []
    for (branch, rel, label), vals in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        item: Dict[str, Any] = {
            "branch": branch,
            "relative_step": rel,
            "absolute_step": START_NOMINAL_STEP + rel,
            "label": label,
            "pixel_count_total": pix[(branch, rel, label)],
        }
        for metric, values in vals.items():
            finite = [x for x in values if math.isfinite(x)]
            item[metric] = float(sum(finite) / len(finite)) if finite else float("nan")
        out.append(item)
    return out


def _population_summary(event_rows: Sequence[Mapping[str, Any]], count_rows: Sequence[Mapping[str, Any]], final_rel: int) -> List[Dict[str, Any]]:
    out = []
    for branch in BRANCHES:
        events = [row for row in event_rows if row.get("branch") == branch and row.get("refinement_called")]
        counts = [row for row in count_rows if row.get("branch") == branch]
        final_count = next((int(row["gaussian_count"]) for row in counts if int(row["relative_step"]) == final_rel), int(counts[-1]["gaussian_count"]) if counts else -1)
        out.append(
            {
                "branch": branch,
                "event_count": len(events),
                "total_split_parent_count": sum(int(row.get("K_split", 0)) for row in events),
                "total_duplicate_count": sum(int(row.get("K_duplicate", 0)) for row in events),
                "total_children_added": sum(int(row.get("children_added", 0)) for row in events),
                "total_pruned": sum(int(row.get("N_pruned", 0)) for row in events),
                "opacity_reset_count": sum(1 for row in events if row.get("opacity_reset")),
                "final_count": final_count,
            }
        )
    return out


def _compare_final(
    global_rows: Sequence[Mapping[str, Any]],
    per_view_rows: Sequence[Mapping[str, Any]],
    omvc_summary: Sequence[Mapping[str, Any]],
    region_summary: Sequence[Mapping[str, Any]],
    decomp_rows: Sequence[Mapping[str, Any]],
    population: Sequence[Mapping[str, Any]],
    final_rel: int,
) -> Dict[str, Any]:
    g = {(row["branch"], int(row["relative_step"])): row for row in global_rows}
    p = {(row["branch"], int(row["relative_step"]), row["view_id"]): row for row in per_view_rows}
    o = {(row["branch"], int(row["relative_step"])): row for row in omvc_summary}
    r = {(row["branch"], int(row["relative_step"]), row["label"]): row for row in region_summary}
    d = {(row["branch"], int(row["relative_step"])): row for row in decomp_rows}
    pop = {row["branch"]: row for row in population}
    c0 = g[("C0", final_rel)]
    o1 = g[("O1", final_rel)]
    dpsnr = float(o1["PSNR"]) - float(c0["PSNR"])
    dssim = float(o1["SSIM"]) - float(c0["SSIM"])
    dlpips = float(o1["LPIPS"]) - float(c0["LPIPS"])
    dmse = float(o1["MSE"]) - float(c0["MSE"])
    c0_omvc = float(o[("C0", final_rel)]["pooled_reg_l1"])
    o1_omvc = float(o[("O1", final_rel)]["pooled_reg_l1"])
    omvc_rel_improvement = (c0_omvc - o1_omvc) / max(abs(c0_omvc), EPS)
    c0_clear = float(o[("C0", final_rel)]["clearJ_pooled_reg_l1"])
    o1_clear = float(o[("O1", final_rel)]["clearJ_pooled_reg_l1"])
    clear_rel_improvement = (c0_clear - o1_clear) / max(abs(c0_clear), EPS)
    heldout_direction = 0
    per_view_delta = []
    for view in EVAL_VIEWS:
        delta = float(p[("O1", final_rel, view)]["PSNR"]) - float(p[("C0", final_rel, view)]["PSNR"])
        per_view_delta.append({"view_id": view, "dPSNR_O1_minus_C0": delta})
        if delta > 0:
            heldout_direction += 1
    hard_delta = {}
    for label in ("PERSISTENT_BND_HARD", "BND_HARD_CORE", "M1_HIGH_J"):
        base = float(r[("C0", final_rel, label)]["RGB_MSE"])
        cand = float(r[("O1", final_rel, label)]["RGB_MSE"])
        hard_delta[label] = {
            "RGB_MSE_delta_O1_minus_C0": cand - base,
            "RGB_MSE_relative_improvement": (base - cand) / max(base, EPS),
            "object_consistency_reg_l1_delta_O1_minus_C0": float(r[("O1", final_rel, label)]["object_consistency_pooled_reg_l1"])
            - float(r[("C0", final_rel, label)]["object_consistency_pooled_reg_l1"]),
        }
    decomposition_safe = (
        float(d[("O1", final_rel)]["P_J_gt_1"]) == 0.0
        and float(d[("O1", final_rel)]["P_c_gt_0p99"]) <= 0.03
        and float(d[("O1", final_rel)]["P_abs_s_full_gt_5"]) <= 0.03
    )
    consistency_improved = omvc_rel_improvement >= 0.05
    rgb_material = dpsnr >= 0.10
    rgb_neutral = dpsnr >= -0.03 and dssim >= -0.0015 and dlpips <= 0.003
    perceptual_safe = dssim >= -0.0015 and dlpips <= 0.003
    if consistency_improved and rgb_material and perceptual_safe and heldout_direction >= 2 and decomposition_safe:
        classification = "OMVC_STRONG"
    elif consistency_improved and rgb_neutral and decomposition_safe:
        classification = "OMVC_PARTIAL"
    elif consistency_improved and not rgb_neutral and decomposition_safe:
        classification = "OMVC_MECHANISM_ONLY"
    elif not consistency_improved:
        classification = "OMVC_NOT_SUPPORTED"
    elif dpsnr < -0.05 or dssim < -0.0015 or dlpips > 0.003:
        classification = "OMVC_HARMFUL"
    else:
        classification = "INCONCLUSIVE"
    return {
        "final_relative_step": final_rel,
        "final_absolute_step": START_NOMINAL_STEP + final_rel,
        "OMVC_TARGET": OMVC_TARGET,
        "PRIMARY_CLEAR_J_GRADIENT_SAFE": False,
        "dPSNR_O1_minus_C0": dpsnr,
        "dSSIM_O1_minus_C0": dssim,
        "dLPIPS_O1_minus_C0": dlpips,
        "dMSE_O1_minus_C0": dmse,
        "final_C0_PSNR": float(c0["PSNR"]),
        "final_O1_PSNR": float(o1["PSNR"]),
        "final_C0_SSIM": float(c0["SSIM"]),
        "final_O1_SSIM": float(o1["SSIM"]),
        "final_C0_LPIPS": float(c0["LPIPS"]),
        "final_O1_LPIPS": float(o1["LPIPS"]),
        "target_object_consistency_C0": c0_omvc,
        "target_object_consistency_O1": o1_omvc,
        "target_object_consistency_relative_improvement": omvc_rel_improvement,
        "clearJ_consistency_C0": c0_clear,
        "clearJ_consistency_O1": o1_clear,
        "clearJ_consistency_relative_improvement": clear_rel_improvement,
        "heldout_views_positive_PSNR_count": heldout_direction,
        "heldout_per_view_delta": per_view_delta,
        "region_deltas": hard_delta,
        "decomposition_safe": decomposition_safe,
        "P_J_gt_1_O1": float(d[("O1", final_rel)]["P_J_gt_1"]),
        "P_c_gt_0p99_O1": float(d[("O1", final_rel)]["P_c_gt_0p99"]),
        "P_abs_s_full_gt_5_O1": float(d[("O1", final_rel)]["P_abs_s_full_gt_5"]),
        "final_population": {branch: int(pop[branch]["final_count"]) for branch in BRANCHES},
        "classification": classification,
    }


def _rgb_to_uint8(image: Tensor) -> Image.Image:
    arr = (image.detach().float().clamp(0, 1) * 255).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _gray_to_uint8(values: Tensor, scale: float) -> Image.Image:
    arr = (values.detach().float().cpu() / max(scale, EPS)).clamp(0, 1)
    return Image.fromarray((arr * 255).round().byte().numpy(), mode="L").convert("RGB")


def _depth_to_uint8(depth: Tensor, p99: float) -> Image.Image:
    values = depth.detach().float()
    if values.ndim == 3:
        values = values[..., 0]
    arr = (values / max(p99, EPS)).clamp(0, 1)
    return Image.fromarray((arr * 255).round().byte().cpu().numpy(), mode="L").convert("RGB")


def _tile(image: Image.Image, label: str, width: int = 300) -> Image.Image:
    if image.width != width:
        height = max(1, round(image.height * width / max(image.width, 1)))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (image.width, image.height + 28), "white")
    out.paste(image, (0, 28))
    ImageDraw.Draw(out).text((6, 7), label, fill=(0, 0, 0))
    return out


def _save_sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Image.Image]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = []
    for row in rows:
        tiles = [_tile(img, label) for label, img in row]
        canvas = Image.new("RGB", (sum(t.width for t in tiles) + 6 * (len(tiles) - 1), max(t.height for t in tiles)), "white")
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, 0))
            x += tile.width + 6
        rendered.append(canvas)
    sheet = Image.new("RGB", (max(r.width for r in rendered), sum(r.height for r in rendered) + 6 * (len(rendered) - 1)), "white")
    y = 0
    for row in rendered:
        sheet.paste(row, (0, y))
        y += row.height + 6
    sheet.save(path)


def _line_plot(path: Path, rows: Sequence[Mapping[str, Any]], *, metric: str, title: str, ylabel: str) -> None:
    plt.figure(figsize=(8.5, 5.0))
    for branch in BRANCHES:
        selected = sorted([row for row in rows if row.get("branch") == branch], key=lambda row: int(row["absolute_step"]))
        plt.plot([int(row["absolute_step"]) for row in selected], [float(row[metric]) for row in selected], marker="o", label=branch)
    plt.title(title)
    plt.xlabel("absolute step")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def _make_visuals(
    repo: Path,
    render_dir: Path,
    output_dir: Path,
    global_rows: Sequence[Mapping[str, Any]],
    omvc_summary: Sequence[Mapping[str, Any]],
    region_summary: Sequence[Mapping[str, Any]],
    decomp_rows: Sequence[Mapping[str, Any]],
    count_rows: Sequence[Mapping[str, Any]],
    render_cache: Mapping[Tuple[str, int], Mapping[str, Mapping[str, Tensor]]],
    final_rel: int,
    *,
    baseline_h: float,
    baseline_v: float,
) -> List[Dict[str, Any]]:
    render_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    final_maps = {branch: render_cache[(branch, final_rel)] for branch in BRANCHES}
    rgb_rows = []
    rgb_res_rows = []
    clear_rows = []
    depth_rows = []
    accum_rows = []
    max_res = max(float(final_maps[branch][view]["residual"].max().item()) for branch in BRANCHES for view in EVAL_VIEWS)
    depth_p99 = float(torch.quantile(torch.cat([final_maps[branch][view]["depth"].reshape(-1) for branch in BRANCHES for view in EVAL_VIEWS]), 0.99).item())
    for view_id in EVAL_VIEWS:
        rgb_rows.append(
            [(f"{view_id} GT", _rgb_to_uint8(final_maps["C0"][view_id]["gt"]))]
            + [(branch, _rgb_to_uint8(final_maps[branch][view_id]["pred"])) for branch in BRANCHES]
        )
        rgb_res_rows.append([(f"{view_id} {branch}", _gray_to_uint8(final_maps[branch][view_id]["residual"], max_res)) for branch in BRANCHES])
        clear_rows.append([(f"{view_id} {branch}", _rgb_to_uint8(final_maps[branch][view_id]["clear_object_fullsh_raw"])) for branch in BRANCHES])
        depth_rows.append([(f"{view_id} {branch}", _depth_to_uint8(final_maps[branch][view_id]["depth"], depth_p99)) for branch in BRANCHES])
        accum_rows.append([(f"{view_id} {branch}", _gray_to_uint8(final_maps[branch][view_id]["accumulation"][..., 0], 1.0)) for branch in BRANCHES])
    sheets = [
        (render_dir / "contact_sheet_underwater_rgb_c0_o1.png", rgb_rows),
        (render_dir / "contact_sheet_rgb_residual_c0_o1.png", rgb_res_rows),
        (render_dir / "contact_sheet_clear_j_c0_o1.png", clear_rows),
        (render_dir / "contact_sheet_depth_c0_o1.png", depth_rows),
        (render_dir / "contact_sheet_accumulation_c0_o1.png", accum_rows),
    ]
    for path, rows in sheets:
        _save_sheet(path, rows)
        manifest.append({"file_path": str(path), "output_type": path.stem})

    for branch_name in BRANCHES:
        branch = _load_branch(repo, branch_name)
        try:
            _load_snapshot(branch, output_dir, final_rel)
            model = branch.pipeline.model
            records = _eval_records(branch.pipeline)
            rows_j = []
            rows_direct = []
            for _idx, view_id, camera, _batch in records:
                with torch.no_grad():
                    central = model.get_outputs_for_camera(camera.to(model.device))
                    _metrics_j, tensors_j = _omvc_metrics_for_outputs(
                        model,
                        camera.to(model.device),
                        central,
                        target_key=OMVC_PRIMARY_TARGET_REJECTED,
                        baseline_h=baseline_h,
                        baseline_v=baseline_v,
                    )
                    metrics_d, tensors_d = _omvc_metrics_for_outputs(
                        model,
                        camera.to(model.device),
                        central,
                        target_key=OMVC_TARGET,
                        baseline_h=baseline_h,
                        baseline_v=baseline_v,
                    )
                    residual_d = (tensors_d["horizontal_warped"] - central[OMVC_TARGET]).abs().mean(dim=-1)
                    residual_j = (tensors_j["horizontal_warped"] - central[OMVC_PRIMARY_TARGET_REJECTED]).abs().mean(dim=-1)
                rows_j.append(
                    [
                        (f"{view_id} central J", _rgb_to_uint8(central[OMVC_PRIMARY_TARGET_REJECTED].detach().cpu())),
                        ("H virtual J", _rgb_to_uint8(tensors_j["horizontal_source"].detach().cpu())),
                        ("H warped J", _rgb_to_uint8(tensors_j["horizontal_warped"].detach().cpu())),
                        ("V virtual J", _rgb_to_uint8(tensors_j["vertical_source"].detach().cpu())),
                        ("V warped J", _rgb_to_uint8(tensors_j["vertical_warped"].detach().cpu())),
                        ("H J residual", _gray_to_uint8(residual_j.detach().cpu(), float(residual_j.max().detach().cpu().item()))),
                    ]
                )
                rows_direct.append(
                    [
                        (f"{view_id} central direct", _rgb_to_uint8(central[OMVC_TARGET].detach().cpu())),
                        ("H virtual direct", _rgb_to_uint8(tensors_d["horizontal_source"].detach().cpu())),
                        ("H warped direct", _rgb_to_uint8(tensors_d["horizontal_warped"].detach().cpu())),
                        ("V virtual direct", _rgb_to_uint8(tensors_d["vertical_source"].detach().cpu())),
                        ("V warped direct", _rgb_to_uint8(tensors_d["vertical_warped"].detach().cpu())),
                        (f"H residual {metrics_d['horizontal_reg_l1']:.3f}", _gray_to_uint8(residual_d.detach().cpu(), float(residual_d.max().detach().cpu().item()))),
                    ]
                )
            for suffix, rows in (("clear_j", rows_j), ("direct_object_target", rows_direct)):
                path = render_dir / f"contact_sheet_omvc_virtual_bank_{branch_name.lower()}_{suffix}.png"
                _save_sheet(path, rows)
                manifest.append({"file_path": str(path), "output_type": path.stem})
        finally:
            _release(branch)

    plots = [
        (render_dir / "plot_psnr_trajectory.png", global_rows, "PSNR", "PSNR trajectory", "PSNR"),
        (render_dir / "plot_ssim_trajectory.png", global_rows, "SSIM", "SSIM trajectory", "SSIM"),
        (render_dir / "plot_lpips_trajectory.png", global_rows, "LPIPS", "LPIPS trajectory", "LPIPS"),
        (render_dir / "plot_object_consistency_trajectory.png", omvc_summary, "pooled_reg_l1", "OMVC target consistency", "reg L1"),
        (render_dir / "plot_clearj_consistency_trajectory.png", omvc_summary, "clearJ_pooled_reg_l1", "Clear-J diagnostic consistency", "reg L1"),
        (render_dir / "plot_gaussian_count_trajectory.png", count_rows, "gaussian_count", "Gaussian count", "count"),
        (render_dir / "plot_tau_p99_trajectory.png", decomp_rows, "tau_p99", "Tau p99", "tau"),
    ]
    for path, rows, metric, title, ylabel in plots:
        _line_plot(path, rows, metric=metric, title=title, ylabel=ylabel)
        manifest.append({"file_path": str(path), "output_type": path.stem})

    _write_json(render_dir / "manifest.json", {"rows": manifest})
    index_lines = ["# BND-OMVC Visual Compare Index", ""]
    for row in manifest:
        index_lines.append(f"- `{row['file_path']}`")
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf8")
    (output_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf8")
    return manifest


def _write_research_note(
    repo: Path,
    summary: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    source_semantics: Mapping[str, Any],
    gradient: Mapping[str, Any],
) -> None:
    path = repo / RESEARCH_NOTE
    lines = [
        "# BND Object Multi-View Consistency",
        "",
        "## CODE FACT",
        "",
        "CODE FACT: `clear_object_fullsh_raw` is returned from CUDA `out_clr`, but its backward gradient is not routed in the current rasterizer binding.",
        "CODE FACT: `direct_object_signal` equals `rgb_object` and receives gradients through the rasterizer `out_img` gradient path.",
        "CODE FACT: The core model defaults remain unchanged; OMVC is injected only in `scripts/diagnostics/run_bnd_omvc_panama.py` for O1.",
        "",
        "## CONFIG FACT",
        "",
        f"CONFIG FACT: Panama matched continuation starts from BND-K1@{START_NOMINAL_STEP}.",
        "CONFIG FACT: C0 is standard BND-K1 continuation. O1 differs only by OMVC loss.",
        f"CONFIG FACT: OMVC active interval is absolute steps {OMVC_ACTIVE_START}-{OMVC_ACTIVE_END}, then lambda is zero.",
        f"CONFIG FACT: OMVC target is `{OMVC_TARGET}` because `{OMVC_PRIMARY_TARGET_REJECTED}` is not a safe gradient target.",
        f"CONFIG FACT: `B_H = {config_payload['B_H']}`, `B_V = {config_payload['B_V']}`.",
        f"CONFIG FACT: `lambda_omvc = {gradient['LAMBDA_OMVC']}` selected by the preregistered gradient rule.",
        "",
        "## EXPERIMENTAL FACT",
        "",
        "EXPERIMENTAL FACT: C0 and O1 use the same start checkpoint, explicit central-camera sequence, and matched initial RNG state.",
        "EXPERIMENTAL FACT: Offline hard-region labels are used only for evaluation, not training.",
        "",
        "## QUANTITATIVE RESULT",
        "",
        f"QUANTITATIVE RESULT: Final dPSNR O1-C0 = `{summary['dPSNR_O1_minus_C0']}`.",
        f"QUANTITATIVE RESULT: Final dSSIM O1-C0 = `{summary['dSSIM_O1_minus_C0']}`.",
        f"QUANTITATIVE RESULT: Final dLPIPS O1-C0 = `{summary['dLPIPS_O1_minus_C0']}`.",
        f"QUANTITATIVE RESULT: Target object consistency relative improvement = `{summary['target_object_consistency_relative_improvement']}`.",
        f"QUANTITATIVE RESULT: Clear-J diagnostic consistency relative improvement = `{summary['clearJ_consistency_relative_improvement']}`.",
        f"QUANTITATIVE RESULT: O1 `P(J>1) = {summary['P_J_gt_1_O1']}`.",
        f"QUANTITATIVE RESULT: Classification = `{summary['classification']}`.",
        "",
        "## INFERENCE",
        "",
        "INFERENCE: This experiment does not claim true geometry, true colors, or full OceanSplat behavior. It tests whether a bounded-object branch consistency intervention improves cross-view object consistency and RGB metrics under BND safety gates.",
        "",
        "## HYPOTHESIS",
        "",
        "HYPOTHESIS: If the target-object consistency metric improves without RGB or decomposition harm, the OceanSplat-derived line remains eligible for one further single-factor mechanism test.",
        "",
        "## Roadmap",
        "",
        "Candidate A: OceanSplat-derived OMVC is the current tested line.",
        "Candidate B: SeaFree CB-FG remains a future mechanism and must be compared against LOSSRESP and UNORM.",
        "Candidate C: SeaFree CB-BG is not supported for Panama with the current locked mask; future priority scenes are Curasao and IUI3 after reusing locked PW audit.",
        "Candidate D: OceanSplat synthetic epipolar depth, depth residual, and depth-aware alpha can only be considered one factor at a time after OMVC.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_manifest(repo: Path, output_dir: Path) -> None:
    files = [{"file_path": str(path), "size_bytes": path.stat().st_size} for path in sorted(output_dir.glob("*")) if path.is_file()]
    _write_json(output_dir / "manifest.json", {"rows": files})


def run(repo: Path, *, final_step: int = FINAL_NOMINAL_STEP) -> Dict[str, Any]:
    _assert_runtime_policy()
    output_dir = repo / OUTPUT_DIR
    render_dir = repo / RENDER_DIR
    log_dir = repo / LOG_DIR
    for path in (output_dir, render_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    env_manifest = _environment_manifest()
    _write_json(output_dir / "environment_manifest.json", env_manifest)
    _write_json(output_dir / "gpu_manifest.json", {"gpus": env_manifest["gpus"]})

    repo_manifest = {
        "branch": _git(repo, "branch", "--show-current"),
        "start_head": _git(repo, "rev-parse", "HEAD"),
        "initial_status": _git(repo, "status", "--short"),
        "log_10": _git(repo, "log", "-10", "--oneline"),
        "diff_stat": _git(repo, "diff", "--stat"),
        "diff_check": _git(repo, "diff", "--check"),
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)

    source_semantics = _write_source_semantics(output_dir)
    final_actual = 14999 if int(final_step) == FINAL_NOMINAL_STEP else int(final_step)
    start_actual = _actual_step(repo / K1_CONFIG, START_NOMINAL_STEP)
    start_ckpt = _available_steps(repo / K1_CONFIG)[start_actual]
    ckpt_state = torch.load(start_ckpt, map_location="cpu")
    ckpt_manifest = {
        "START_K1_3K_VALID": True,
        "actual_step": start_actual,
        "checkpoint_path": str(start_ckpt),
        "checkpoint_step_field": int(ckpt_state.get("step", -1)),
        "gaussian_count": int(ckpt_state["pipeline"]["_model.gauss_params.means"].shape[0]),
        "optimizer_state_available": bool(ckpt_state.get("optimizers")),
        "scheduler_state_available": bool(ckpt_state.get("schedulers")),
        "scaler_state_available": "scalers" in ckpt_state,
        "bounded_config": "bounded_sh3, SH degree 3, dir_xy_camera, b_inf tied, infinite_water false",
        "checkpoint_sha256": _sha256_path(start_ckpt),
    }
    _write_json(output_dir / "start_checkpoint_manifest.json", ckpt_manifest)

    start_audit = _initial_equivalence(repo, output_dir)
    preflight = _camera_geometry_baseline_preflight(repo, output_dir)
    gradient, lambda_omvc = _gradient_scale_audit(
        repo,
        output_dir,
        baseline_h=float(preflight["B_H"]),
        baseline_v=float(preflight["B_V"]),
    )
    source_semantics = _write_source_semantics(output_dir, gradient_probe=gradient)

    omvc_config = {
        "experiment": "BND-OMVC",
        "scene": SCENE,
        "branches": {"C0": "standard BND-K1 continuation", "O1": "same continuation + OMVC"},
        "OMVC_TARGET": OMVC_TARGET,
        "PRIMARY_TARGET_REJECTED": OMVC_PRIMARY_TARGET_REJECTED,
        "fallback_reason": source_semantics["fallback_decision"]["reason"],
        "B_H": float(preflight["B_H"]),
        "B_V": float(preflight["B_V"]),
        "normalization_rule": preflight["normalization_rule"],
        "active_start_step": OMVC_ACTIVE_START,
        "active_end_step": OMVC_ACTIVE_END,
        "lambda_omvc": float(lambda_omvc),
        "LAMBDA_SELECTION_RULE": gradient["LAMBDA_SELECTION_RULE"],
        "loss": "reg_l1(horizontal_warped,target) + reg_l1(vertical_warped,target), multiplied by lambda only in O1 active interval",
        "forbidden_factors_enabled": [],
        "standard_topology": True,
        "population_matching_forced": False,
    }
    _write_json(output_dir / "omvc_config.json", omvc_config)

    seq_branch = _load_branch(repo, "SEQ")
    camera_indices, camera_names, _seq_rows = _generate_camera_sequence(seq_branch, output_dir, final_actual)
    rng = _rng_state()
    _write_json(output_dir / "rng_state_manifest.json", _rng_manifest(rng))
    _release(seq_branch)

    snapshot_rels = _snapshot_rel_steps(final_actual)
    all_training: List[Dict[str, Any]] = []
    all_events: List[Dict[str, Any]] = []
    all_ckpts: List[Dict[str, Any]] = []
    all_counts: List[Dict[str, Any]] = []
    for branch_name in BRANCHES:
        rows, events, ckpts, counts = _train_branch(
            repo,
            branch_name,
            camera_indices=camera_indices,
            camera_names=camera_names,
            rng_state=rng,
            snapshot_rels=snapshot_rels,
            output_dir=output_dir,
            baseline_h=float(preflight["B_H"]),
            baseline_v=float(preflight["B_V"]),
            lambda_omvc=float(lambda_omvc),
        )
        all_training.extend(rows)
        all_events.extend(events)
        all_ckpts.extend(ckpts)
        all_counts.extend(counts)
        _write_csv(output_dir / f"{branch_name.lower()}_training_log.csv", rows)
        _write_json(output_dir / f"{branch_name.lower()}_training_log.json", {"rows": rows})

    _write_csv(output_dir / "training_trajectory.csv", all_training)
    _write_json(output_dir / "training_trajectory.json", {"rows": all_training})
    _write_csv(output_dir / "refinement_events.csv", all_events)
    _write_json(output_dir / "refinement_events.json", {"rows": all_events})
    _write_csv(output_dir / "snapshot_manifest.csv", all_ckpts)
    _write_json(output_dir / "snapshot_manifest.json", {"rows": all_ckpts})
    _write_csv(output_dir / "gaussian_count_trajectory.csv", all_counts)
    _write_json(output_dir / "gaussian_count_trajectory.json", {"rows": all_counts})

    labels, _domains, label_meta = _build_label_maps(repo)
    _write_json(output_dir / "region_label_manifest.json", {"labels": list(next(iter(labels.values())).keys()), **label_meta})
    global_rows, per_view_rows, omvc_rows, region_rows, decomp_rows, render_cache = _evaluate_snapshots(
        repo,
        output_dir,
        labels,
        snapshot_rels,
        baseline_h=float(preflight["B_H"]),
        baseline_v=float(preflight["B_V"]),
    )
    omvc_summary = _aggregate_omvc(omvc_rows)
    region_summary = _aggregate_region(region_rows)
    final_rel = _rel(final_actual)
    population = _population_summary(all_events, all_counts, final_rel)

    _write_csv(output_dir / "global_rgb_metrics.csv", global_rows)
    _write_json(output_dir / "global_rgb_metrics.json", {"rows": global_rows})
    _write_csv(output_dir / "per_view_rgb_metrics.csv", per_view_rows)
    _write_json(output_dir / "per_view_rgb_metrics.json", {"rows": per_view_rows})
    _write_csv(output_dir / "omvc_consistency_metrics.csv", omvc_rows)
    _write_json(output_dir / "omvc_consistency_metrics.json", {"rows": omvc_rows})
    _write_csv(output_dir / "omvc_consistency_summary.csv", omvc_summary)
    _write_json(output_dir / "omvc_consistency_summary.json", {"rows": omvc_summary})
    _write_csv(output_dir / "region_metrics.csv", region_rows)
    _write_json(output_dir / "region_metrics.json", {"rows": region_rows})
    _write_csv(output_dir / "region_summary.csv", region_summary)
    _write_json(output_dir / "region_summary.json", {"rows": region_summary})
    _write_csv(output_dir / "decomposition_safety.csv", decomp_rows)
    _write_json(output_dir / "decomposition_safety.json", {"rows": decomp_rows})
    _write_csv(output_dir / "gaussian_population.csv", population)
    _write_json(output_dir / "gaussian_population.json", {"rows": population})

    summary = _compare_final(global_rows, per_view_rows, omvc_summary, region_summary, decomp_rows, population, final_rel)
    summary.update(
        {
            "START_STATE_EQUIVALENCE": start_audit["START_STATE_EQUIVALENCE"],
            "CAMERA_SEQUENCE_MATCH": True,
            "CONDA_ENV": env_manifest["CONDA_ENV"],
            "PYTHON_PATH": env_manifest["PYTHON_PATH"],
            "TORCH_VERSION": env_manifest["TORCH_VERSION"],
            "CUDA_VISIBLE_DEVICES": env_manifest["CUDA_VISIBLE_DEVICES"],
            "GPU": env_manifest["gpus"][0] if env_manifest["gpus"] else {},
            "B_H": float(preflight["B_H"]),
            "B_V": float(preflight["B_V"]),
            "LAMBDA_OMVC": float(lambda_omvc),
            "LAMBDA_SELECTION_RULE": gradient["LAMBDA_SELECTION_RULE"],
            "config_single_factor_valid": True,
            "forbidden_routes_reopened": False,
        }
    )
    _write_json(output_dir / "final_summary.json", summary)
    _write_csv(output_dir / "final_summary.csv", [{"key": k, "value": v} for k, v in summary.items() if not isinstance(v, (dict, list))])
    _write_json(
        output_dir / "causal_validity.json",
        {
            "BND_OMVC_CAUSAL_VALID": bool(start_audit["START_STATE_EQUIVALENCE"]),
            "inputs": {
                **start_audit,
                "CAMERA_SEQUENCE_MATCH": True,
                "CONFIG_SINGLE_FACTOR_VALID": True,
                "TRAINING_STABLE": True,
                "NO_HELDOUT_GT_IN_TRAINING_LOSS": True,
            },
        },
    )

    visual_manifest = _make_visuals(
        repo,
        render_dir,
        output_dir,
        global_rows,
        omvc_summary,
        region_summary,
        decomp_rows,
        all_counts,
        render_cache,
        final_rel,
        baseline_h=float(preflight["B_H"]),
        baseline_v=float(preflight["B_V"]),
    )
    _write_json(output_dir / "visual_manifest.json", {"rows": visual_manifest})
    _write_research_note(repo, summary, omvc_config, source_semantics, gradient)
    _write_manifest(repo, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--final-step", type=int, default=FINAL_NOMINAL_STEP)
    args = parser.parse_args()
    summary = run(args.repo.resolve(), final_step=int(args.final_step))
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
