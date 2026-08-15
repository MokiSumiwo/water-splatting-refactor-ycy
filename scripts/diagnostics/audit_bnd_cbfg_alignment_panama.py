#!/usr/bin/env python
"""Zero-training BND CB-FG/FAW alignment audit for Panama.

This diagnostic reproduces only SeaFree foreground-aware inverse-intensity
weighting semantics. It performs no optimizer step, writes no checkpoints, and
does not enable CB-BG, coarse depth, OMVC, CDEPTH, or any training loss.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup

from scripts.diagnostics import audit_bnd_cdepth_setup as cdepth_setup
from scripts.diagnostics import run_bnd_aware_refine_panama as aware


SCENE = "Panama"
OUTPUT_DIR = Path("outputs/bnd_cbfg_alignment_audit_20260815")
RESEARCH_NOTE = Path("research_notes/BND_CBFG_ALIGNMENT_AUDIT_2026-08-15.md")

K1_CONFIG = cdepth_setup.K1_CONFIG
M1_CONFIG = aware.M1_CONFIG
DEPTHS_PATH = cdepth_setup.DEPTHS_PATH
TRAIN_VIEWS = aware.TRAIN_VIEWS
EVAL_VIEWS = aware.EVAL_VIEWS

FINAL_NOMINAL_STEP = 15000
START_NOMINAL_STEP = 3000
LATE_STEPS = (8000, 10000, 13000, FINAL_NOMINAL_STEP)
ALLOWED_PHYSICAL_GPUS = frozenset({"6", "7", "8", "9"})
EPS = 1e-12
CBFG_THRESHOLD = 1e-2
TOP_FRACTIONS = (0.10, 0.20, 0.30)
LABELS = ("M1_HIGH_J", "PERSISTENT_BND_HARD", "BND_HARD_CORE")
OBJECT_GROUPS = ("means", "features_dc", "features_rest", "opacities", "scales", "quats")
MEDIUM_GROUPS = ("medium_mlp", "direction_encoding")


@dataclass
class LoadedRun:
    run: str
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
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def _assert_runtime_policy() -> None:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env != "water_splatting":
        raise RuntimeError(f"CONDA_DEFAULT_ENV must be water_splatting, got {conda_env!r}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [token.strip() for token in visible.split(",") if token.strip()]
    if len(devices) != 1 or devices[0] not in ALLOWED_PHYSICAL_GPUS:
        allowed = ",".join(sorted(ALLOWED_PHYSICAL_GPUS))
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be exactly one physical GPU in {allowed}; got {visible!r}")


def _environment_manifest() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    if torch.cuda.is_available():
        logical = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(logical)
        rows.append(
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
        "torch_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpus": rows,
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


def _load_run(repo: Path, run: str, step: int, *, load_depths: bool) -> LoadedRun:
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
    model.config.intrinsic_color_parameterization = parameterization
    model.config.rasterize_mode = "classic"
    model.config.medium_context_mode = "dir_xy_camera"
    model.config.b_inf_mode = "tied"
    model.config.infinite_water_enabled = False
    model.config.coarse_depth_supervision_enabled = False
    model.step = int(loaded_step)
    pipeline.eval()
    return LoadedRun(run, config_path, Path(checkpoint_path), int(loaded_step), config, pipeline)


def _release(obj: Any) -> None:
    if obj is None:
        return
    try:
        del obj.pipeline
    except Exception:
        pass
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _records(pipeline: Any, split: str) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    if split == "train":
        dataset = pipeline.datamanager.train_dataset
        filenames = list(getattr(dataset, "image_filenames", []))
        cameras = dataset.cameras.to(pipeline.model.device)
        rows = []
        for index, filename in enumerate(filenames):
            batch = pipeline.datamanager.cached_train[index].copy()
            rows.append((index, Path(filename).stem, cameras[index : index + 1], _batch_to_device(batch, pipeline.model.device)))
        return rows
    if split == "eval":
        dataset = pipeline.datamanager.eval_dataset
        filenames = list(getattr(dataset, "image_filenames", []))
        rows = []
        for eval_index, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            view_id = Path(filenames[eval_index]).stem if eval_index < len(filenames) else f"eval_{eval_index}"
            rows.append((eval_index, view_id, camera, _batch_to_device(batch, pipeline.model.device)))
        return rows
    raise ValueError(split)


def _gt_pred(model: Any, outputs: Mapping[str, Tensor], batch: Mapping[str, Any]) -> Tuple[Tensor, Tensor]:
    gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
    pred = outputs["pred_image"]
    if "mask" in batch:
        mask = model._downscale_if_required(batch["mask"]).to(model.device)
        gt = gt * mask
        pred = pred * mask
    return gt, pred


def _render_no_grad(model: Any, camera: Cameras, batch: Mapping[str, Any]) -> Dict[str, Tensor]:
    model.eval()
    with torch.no_grad():
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        gt, pred = _gt_pred(model, outputs, batch)
    pred_raw = pred.detach().float().cpu()
    pred_c = pred_raw.clamp(0.0, 1.0)
    gt_c = gt.detach().float().clamp(0.0, 1.0).cpu()
    item: Dict[str, Tensor] = {
        "pred": pred_c,
        "pred_raw": pred_raw,
        "gt": gt_c,
        "err": (pred_c - gt_c).square().mean(dim=-1),
    }
    for key in ("accumulation", "clear_object_fullsh_raw", "tau_D", "transmission", "depth"):
        if key in outputs and isinstance(outputs[key], Tensor):
            item[key] = outputs[key].detach().float().cpu()
    return item


def _safe_quantile_np(values: np.ndarray, q: float) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q))


def _stats_np(values: np.ndarray, prefix: str = "") -> Dict[str, Any]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return {
            f"{prefix}count": 0,
            f"{prefix}mean": float("nan"),
            f"{prefix}median": float("nan"),
            f"{prefix}p10": float("nan"),
            f"{prefix}p90": float("nan"),
            f"{prefix}p99": float("nan"),
            f"{prefix}max": float("nan"),
        }
    return {
        f"{prefix}count": int(vals.size),
        f"{prefix}mean": float(vals.mean()),
        f"{prefix}median": float(np.median(vals)),
        f"{prefix}p10": _safe_quantile_np(vals, 0.10),
        f"{prefix}p90": _safe_quantile_np(vals, 0.90),
        f"{prefix}p99": _safe_quantile_np(vals, 0.99),
        f"{prefix}max": float(vals.max()),
    }


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def _spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    finite = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[finite]
    bb = bb[finite]
    if aa.size < 2:
        return float("nan")
    ra = _rankdata_average(aa)
    rb = _rankdata_average(bb)
    if float(np.std(ra)) < EPS or float(np.std(rb)) < EPS:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _top_mask_np(values: np.ndarray, domain: np.ndarray, fraction: float) -> np.ndarray:
    domain = domain.astype(bool)
    out = np.zeros_like(domain, dtype=bool)
    idx = np.flatnonzero(domain.reshape(-1))
    if idx.size == 0:
        return out
    vals = values.reshape(-1)[idx]
    finite = np.isfinite(vals)
    idx = idx[finite]
    vals = vals[finite]
    if idx.size == 0:
        return out
    k = max(1, int(math.ceil(float(fraction) * idx.size)))
    selected = idx[np.argpartition(vals, -k)[-k:]]
    out.reshape(-1)[selected] = True
    return out


def _seafree_foreground_from_batch(model: Any, batch: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if "depth_image" not in batch:
        raise RuntimeError("CBFG_PREFLIGHT_BLOCKED_NO_LOCKED_PSEUDODEPTH: batch lacks depth_image")
    pseudo = model._downscale_if_required(batch["depth_image"]).to(model.device).detach().float()
    max_value = float(pseudo.max().detach().cpu().item())
    if not math.isfinite(max_value) or max_value <= 0.0:
        raise RuntimeError("CBFG_PREFLIGHT_BLOCKED_NO_LOCKED_PSEUDODEPTH: pseudo-depth max is nonpositive")
    pseudo = pseudo / max_value
    pseudo_np = pseudo.squeeze().cpu().numpy().astype(np.float32)
    mask_1e_2_copy = (pseudo_np < CBFG_THRESHOLD).astype(np.uint8) * 255
    _, binary_image = cv2.threshold(mask_1e_2_copy, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    foreground_mask_array = np.zeros_like(binary_image)
    largest_contour_area = 0.0
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        largest_contour_area = float(cv2.contourArea(max_contour))
        cv2.drawContours(foreground_mask_array, [max_contour], -1, (255,), thickness=cv2.FILLED)
    foreground = foreground_mask_array.astype(np.float32) / 255.0
    foreground[foreground < CBFG_THRESHOLD] = 0.0
    foreground[foreground > 0.0] = 1.0
    fg_bool = foreground.astype(bool)
    meta = {
        "status": "OK",
        "pseudo_depth_max_before_normalization": max_value,
        "pseudo_depth_min_normalized": float(np.nanmin(pseudo_np)),
        "pseudo_depth_max_normalized": float(np.nanmax(pseudo_np)),
        "foreground_pixels": int(fg_bool.sum()),
        "total_pixels": int(fg_bool.size),
        "foreground_fraction": float(fg_bool.mean()),
        "background_fraction": float(1.0 - fg_bool.mean()),
        "contour_count": int(len(contours)),
        "largest_contour_area": largest_contour_area,
        "threshold": CBFG_THRESHOLD,
        "mask_semantics": "SeaFree normalized pseudo-depth threshold 1e-2, THRESH_BINARY_INV, largest external contour fill, binarized foreground.",
    }
    return pseudo_np, fg_bool, meta


def _cbfg_weight_scalar(pred_raw_cpu: Tensor, foreground: np.ndarray) -> np.ndarray:
    pred = pred_raw_cpu.detach().float().cpu().numpy()
    w = 1.0 / (pred + 1e-3)
    w = np.where(foreground[..., None], w, np.ones_like(w))
    return w.mean(axis=-1).astype(np.float32)


def _scalar_map(tensor: Tensor, mode: str) -> np.ndarray:
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        if mode == "mean":
            arr = arr.mean(axis=-1)
        elif mode == "min":
            arr = arr.min(axis=-1)
        elif mode == "max":
            arr = arr.max(axis=-1)
        else:
            raise ValueError(mode)
    return arr.astype(np.float32)


def _aggregate_arrays(view_data: Mapping[str, Mapping[str, np.ndarray]], keys: Sequence[str], domain_key: Optional[str] = None) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if domain_key is None:
        masks = {view_id: np.ones_like(next(iter(data.values())), dtype=bool) for view_id, data in view_data.items()}
    else:
        masks = {view_id: data[domain_key].astype(bool) for view_id, data in view_data.items()}
    for key in keys:
        pieces = []
        for view_id, data in view_data.items():
            mask = masks[view_id]
            pieces.append(data[key][mask].reshape(-1))
        out[key] = np.concatenate(pieces, axis=0) if pieces else np.asarray([], dtype=np.float32)
    return out


def _parameter_snapshot(model: Any) -> Dict[str, Tensor]:
    snap = {}
    for name, params in model.get_param_groups().items():
        for idx, param in enumerate(params):
            snap[f"{name}.{idx}"] = param.detach().cpu().clone()
    return snap


def _parameter_delta(before: Mapping[str, Tensor], model: Any) -> Dict[str, Any]:
    rows = []
    max_abs = 0.0
    after_groups = model.get_param_groups()
    after = {}
    for name, params in after_groups.items():
        for idx, param in enumerate(params):
            after[f"{name}.{idx}"] = param.detach().cpu()
    for key, old in before.items():
        diff = (after[key] - old).abs()
        val = float(diff.max().item()) if diff.numel() else 0.0
        max_abs = max(max_abs, val)
        rows.append({"parameter": key, "max_abs_delta": val, "mean_abs_delta": float(diff.mean().item()) if diff.numel() else 0.0})
    return {"max_abs_delta": max_abs, "rows": rows}


def _zero_grad(model: Any) -> None:
    model.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.grad = None


def _split_ssim(model: Any, img1: Tensor, img2: Tensor) -> Tensor:
    height, width = img1.shape[-2], img1.shape[-1]
    if height <= 800 and width <= 800:
        return model.ssim(img1, img2)
    h_half, w_half = height // 2, width // 2
    blocks1 = [
        img1[..., :h_half, :w_half],
        img1[..., :h_half, w_half:],
        img1[..., h_half:, :w_half],
        img1[..., h_half:, w_half:],
    ]
    blocks2 = [
        img2[..., :h_half, :w_half],
        img2[..., :h_half, w_half:],
        img2[..., h_half:, :w_half],
        img2[..., h_half:, w_half:],
    ]
    return torch.mean(torch.stack([model.ssim(a, b) for a, b in zip(blocks1, blocks2)]))


def _cbfg_loss(model: Any, outputs: Mapping[str, Tensor], batch: Mapping[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
    gt, pred = _gt_pred(model, outputs, batch)
    pseudo_depth = model._downscale_if_required(batch["depth_image"]).to(model.device).float()
    pseudo_depth = pseudo_depth / pseudo_depth.max().clamp_min(EPS)
    pseudo_np = pseudo_depth.squeeze().detach().cpu().numpy().astype(np.float32)
    mask_1e_2_copy = (pseudo_np < CBFG_THRESHOLD).astype(np.uint8) * 255
    _, binary_image = cv2.threshold(mask_1e_2_copy, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg_arr = np.zeros_like(binary_image)
    if len(contours) > 0:
        cv2.drawContours(fg_arr, [max(contours, key=cv2.contourArea)], -1, (255,), thickness=cv2.FILLED)
    fg_arr = fg_arr.astype(np.float32) / 255.0
    fg_arr[fg_arr < CBFG_THRESHOLD] = 0.0
    fg_arr[fg_arr > 0.0] = 1.0
    fg = torch.from_numpy(fg_arr).to(device=model.device, dtype=pred.dtype)
    weight = 1.0 / (pred.detach() + 1e-3)
    weight = torch.where(fg[..., None] < 0.5, torch.ones_like(weight), weight)
    weighted_l1 = torch.abs((gt - pred) * weight).mean()
    w_chw = weight.permute(2, 0, 1)[None, ...]
    gt_chw = gt.permute(2, 0, 1)[None, ...] * w_chw
    pred_chw = pred.permute(2, 0, 1)[None, ...] * w_chw
    weighted_dssim = 1.0 - _split_ssim(model, gt_chw, pred_chw)
    loss = (1.0 - float(model.config.ssim_lambda)) * weighted_l1 + float(model.config.ssim_lambda) * weighted_dssim
    return loss, {
        "weighted_l1": float(weighted_l1.detach().cpu().item()),
        "weighted_dssim": float(weighted_dssim.detach().cpu().item()),
        "foreground_fraction": float(fg.detach().float().mean().cpu().item()),
    }


def _grad_norms_for_loss(model: Any, camera: Cameras, batch: Mapping[str, Any], loss_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    _zero_grad(model)
    outputs = model.get_outputs(camera.to(model.device))
    if loss_name == "BASE":
        loss = model.get_loss_dict(outputs, batch, {})["main_loss"]
        meta = {"loss_name": "BASE", "loss_value": float(loss.detach().cpu().item()), "loss_semantics": "WaterSplatting formal main_loss"}
    elif loss_name == "CBFG":
        loss, cb_meta = _cbfg_loss(model, outputs, batch)
        meta = {"loss_name": "CBFG", "loss_value": float(loss.detach().cpu().item()), **cb_meta}
    else:
        raise ValueError(loss_name)
    groups = model.get_param_groups()
    params = [param for plist in groups.values() for param in plist]
    grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
    rows: Dict[str, Any] = {}
    offset = 0
    for group_name, plist in groups.items():
        sq = 0.0
        elems = 0
        for grad in grads[offset : offset + len(plist)]:
            offset += 1
            if grad is None:
                continue
            gd = grad.detach().float()
            sq += float((gd * gd).sum().cpu().item())
            elems += int(gd.numel())
        rows[group_name] = {"grad_l2": math.sqrt(sq), "grad_elements": elems}
    del outputs, loss
    _zero_grad(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, meta


def _gradient_audit(repo: Path, output_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    loaded = None
    try:
        loaded = _load_run(repo, "K1", FINAL_NOMINAL_STEP, load_depths=True)
        model = loaded.pipeline.model
        model.train()
        model.step = loaded.loaded_step
        record = _records(loaded.pipeline, "train")[0]
        _, view_id, camera, batch = record
        before = _parameter_snapshot(model)
        base, base_meta = _grad_norms_for_loss(model, camera, batch, "BASE")
        cbfg, cbfg_meta = _grad_norms_for_loss(model, camera, batch, "CBFG")
        delta = _parameter_delta(before, model)
        rows = []
        for group in sorted(set(base) | set(cbfg)):
            b = float(base.get(group, {}).get("grad_l2", 0.0))
            c = float(cbfg.get(group, {}).get("grad_l2", 0.0))
            rows.append(
                {
                    "view_id": view_id,
                    "parameter_group": group,
                    "grad_BASE": b,
                    "grad_CBFG": c,
                    "ratio_CBFG_BASE": c / max(b, EPS),
                    "base_elements": base.get(group, {}).get("grad_elements", 0),
                    "cbfg_elements": cbfg.get(group, {}).get("grad_elements", 0),
                }
            )
        def group_norm(source: Mapping[str, Mapping[str, Any]], names: Sequence[str]) -> float:
            return math.sqrt(sum(float(source.get(name, {}).get("grad_l2", 0.0)) ** 2 for name in names))

        object_base = group_norm(base, OBJECT_GROUPS)
        object_cbfg = group_norm(cbfg, OBJECT_GROUPS)
        medium_base = group_norm(base, MEDIUM_GROUPS)
        medium_cbfg = group_norm(cbfg, MEDIUM_GROUPS)
        rows.append(
            {
                "view_id": view_id,
                "parameter_group": "OBJECT_GROUPS_AGG",
                "grad_BASE": object_base,
                "grad_CBFG": object_cbfg,
                "ratio_CBFG_BASE": object_cbfg / max(object_base, EPS),
            }
        )
        rows.append(
            {
                "view_id": view_id,
                "parameter_group": "MEDIUM_GROUPS_AGG",
                "grad_BASE": medium_base,
                "grad_CBFG": medium_cbfg,
                "ratio_CBFG_BASE": medium_cbfg / max(medium_base, EPS),
            }
        )
        summary = {
            "view_id": view_id,
            "loaded_step": loaded.loaded_step,
            "BASE": base_meta,
            "CBFG": cbfg_meta,
            "object_group_ratio_CBFG_BASE": object_cbfg / max(object_base, EPS),
            "medium_group_ratio_CBFG_BASE": medium_cbfg / max(medium_base, EPS),
            "medium_to_object_ratio_BASE": medium_base / max(object_base, EPS),
            "medium_to_object_ratio_CBFG": medium_cbfg / max(object_cbfg, EPS),
            "parameter_delta_max_abs": delta["max_abs_delta"],
            "NO_PARAMETER_UPDATE": bool(delta["max_abs_delta"] == 0.0),
        }
        _write_csv(output_dir / "gradient_responsibility.csv", rows)
        _write_json(output_dir / "gradient_responsibility.json", {"summary": summary, "rows": rows, "parameter_delta_rows": delta["rows"]})
        return rows, summary
    finally:
        _release(loaded)


def _omvc_trajectory_closure(output_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = Path("outputs/bnd_omvc_panama_20260812/omvc_consistency_summary.json")
    data = json.loads(path.read_text())
    by_key = {(row["branch"], int(row["absolute_step"])): row for row in data["rows"]}
    steps = sorted({int(row["absolute_step"]) for row in data["rows"]})
    rows = []
    active_improved = False
    inactive_improved = False
    for step in steps:
        c0 = by_key.get(("C0", step))
        o1 = by_key.get(("O1", step))
        if not c0 or not o1:
            continue
        active = 4000 <= step <= 10000
        direct_delta = float(o1["pooled_reg_l1"]) - float(c0["pooled_reg_l1"])
        clear_delta = float(o1["clearJ_pooled_reg_l1"]) - float(c0["clearJ_pooled_reg_l1"])
        row = {
            "absolute_step": step,
            "relative_step": int(c0["relative_step"]),
            "OMVC_active": active,
            "C0_registered_direct_object_signal_consistency": float(c0["pooled_reg_l1"]),
            "O1_registered_direct_object_signal_consistency": float(o1["pooled_reg_l1"]),
            "O1_minus_C0_registered_direct": direct_delta,
            "relative_O1_C0_registered_direct": -direct_delta / max(float(c0["pooled_reg_l1"]), EPS),
            "C0_clearJ_diagnostic_consistency": float(c0["clearJ_pooled_reg_l1"]),
            "O1_clearJ_diagnostic_consistency": float(o1["clearJ_pooled_reg_l1"]),
            "O1_minus_C0_clearJ": clear_delta,
            "relative_O1_C0_clearJ": -clear_delta / max(float(c0["clearJ_pooled_reg_l1"]), EPS),
        }
        rows.append(row)
        if active and direct_delta < 0:
            active_improved = True
        if (not active) and direct_delta < 0:
            inactive_improved = True
    final = rows[-1] if rows else {}
    if active_improved and float(final.get("O1_minus_C0_registered_direct", 0.0)) >= 0:
        closure = "OMVC_NONPERSISTENT_AFTER_DEACTIVATION"
    elif active_improved:
        closure = "OMVC_ACTIVE_IMPROVEMENT_PRESENT_BUT_FINAL_CLOSED"
    else:
        closure = "OMVC_MECHANISM_NOT_ENFORCED"
    summary = {
        "source": str(path),
        "closure_classification": closure,
        "active_registered_improvement_seen": active_improved,
        "inactive_registered_improvement_seen": inactive_improved,
        "BND_OMVC_DIRECT_OBJECT_SIGNAL": "CLOSED",
        "pure_intrinsic_J_OMVC": "UNTESTED_due_current_backward_limitation",
        "final_registered_O1_minus_C0": final.get("O1_minus_C0_registered_direct"),
        "final_clearJ_O1_minus_C0": final.get("O1_minus_C0_clearJ"),
    }
    _write_csv(output_dir / "omvc_trajectory_closure.csv", rows)
    _write_json(output_dir / "omvc_trajectory_closure.json", {"summary": summary, "rows": rows})
    return rows, summary


def _recover_historical(output_dir: Path) -> Dict[str, Any]:
    lossresp_summary = json.loads(Path("outputs/lossresp_audit_20260810/lossresp_final_summary.json").read_text())
    lossresp = {row["key"]: row["value"] for row in lossresp_summary}
    unorm_rows = json.loads(Path("outputs/bnd_unorm_panama_20260810/bnd_unorm_final_summary.json").read_text())
    unorm = unorm_rows[0] if unorm_rows else {}
    recovered = {
        "LOSSRESP": {
            "CODE FACT": {
                "formal_loss": "relative_pred_detached: 0.8*reg_l1 + 0.2*reg_ssim",
                "seafree_reference": "inverse-intensity weight 1/(pred.detach()+1e-3); prior audit lacked full pseudo-depth mask and used intensity-only reference",
            },
            "EXPERIMENTAL FACT": {
                "scene": "Panama",
                "checkpoint": "formal BND-K1 final eval views",
                "read_only": True,
            },
            "QUANTITATIVE RESULT": lossresp,
            "INFERENCE": {
                "overall": lossresp.get("OVERALL_HYPOTHESIS", "NOT RECOVERED"),
                "seafree_specific": lossresp.get("SEAFREE_SPECIFIC_HYPOTHESIS", "NOT RECOVERED"),
            },
        },
        "UNORM": {
            "CODE FACT": {
                "weighting_definition": "absolute photometric normalization: 0.8*mean(abs(GT-pred)) + 0.2*(1-SSIM(GT,pred))",
                "excluded": "no foreground masks, residual weighting, pseudo-depth, SeaFree CB-BG, or depth loss",
            },
            "CONFIG FACT": {
                "scene": "Panama",
                "seed": 42,
                "single_factor": "photometric_normalization_mode relative_pred_detached -> absolute",
            },
            "QUANTITATIVE RESULT": unorm,
            "INFERENCE": {
                "formal_conclusion": unorm.get("HYPOTHESIS_SUPPORT", "NOT RECOVERED"),
                "local_highj_recovery": unorm.get("HIGH_J_TARGETED_RECOVERY", "NOT RECOVERED"),
                "global_rgb_safety": unorm.get("RGB_SAFETY", "NOT RECOVERED"),
            },
        },
    }
    _write_json(output_dir / "historical_lossresp_unorm_recovery.json", recovered)
    return recovered


def _pseudo_depth_file_audit(repo: Path) -> Dict[str, Any]:
    image_dir = repo / "undistorted_data/undistorted_Panama/images/ColorImage"
    depth_dir = repo / "undistorted_data/undistorted_Panama" / DEPTHS_PATH
    image_files = sorted(image_dir.glob("*.png"))
    depth_files = sorted(depth_dir.glob("*.png"))
    depth_stems = {path.stem for path in depth_files}
    train_missing = [view for view in TRAIN_VIEWS if view not in depth_stems]
    eval_missing = [view for view in EVAL_VIEWS if view not in depth_stems]
    return {
        "image_dir": str(image_dir),
        "depth_dir": str(depth_dir),
        "num_images": len(image_files),
        "num_depths": len(depth_files),
        "train_views": list(TRAIN_VIEWS),
        "eval_views": list(EVAL_VIEWS),
        "missing_train_depths": train_missing,
        "missing_eval_depths": eval_missing,
        "LOCKED_PSEUDODEPTH_AVAILABLE": bool(len(depth_files) > 0 and not train_missing and not eval_missing),
        "pseudo_depth_source": "undistorted_data/undistorted_Panama/depthAnything_u16",
    }


def _collect_m1_final(repo: Path, split: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    loaded = None
    try:
        loaded = _load_run(repo, "M1", FINAL_NOMINAL_STEP, load_depths=False)
        records = _records(loaded.pipeline, split)
        out: Dict[str, Dict[str, np.ndarray]] = {}
        rows = []
        for _idx, view_id, camera, batch in records:
            item = _render_no_grad(loaded.pipeline.model, camera, batch)
            acc = _scalar_map(item["accumulation"], "mean")
            jmax = _scalar_map(item["clear_object_fullsh_raw"], "max")
            highj = (acc > 0.01) & (jmax > 1.0)
            out[view_id] = {
                "m1_err": item["err"].numpy().astype(np.float32),
                "m1_highj": highj,
            }
            rows.append({"split": split, "view_id": view_id, "M1_HIGH_J_pixels": int(highj.sum()), "total_pixels": int(highj.size), "M1_HIGH_J_fraction": float(highj.mean())})
            del item
            torch.cuda.empty_cache()
        return out, {"loaded_step": loaded.loaded_step, "rows": rows}
    finally:
        _release(loaded)


def _collect_bnd_final(repo: Path, split: str, m1_data: Mapping[str, Mapping[str, np.ndarray]], output_dir: Path) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any], List[Dict[str, Any]]]:
    loaded = None
    try:
        loaded = _load_run(repo, "K1", FINAL_NOMINAL_STEP, load_depths=True)
        model = loaded.pipeline.model
        records = _records(loaded.pipeline, split)
        view_data: Dict[str, Dict[str, np.ndarray]] = {}
        pseudo_rows: List[Dict[str, Any]] = []
        for _idx, view_id, camera, batch in records:
            item = _render_no_grad(model, camera, batch)
            raw_pred = item["pred_raw"]
            pseudo, fg, meta = _seafree_foreground_from_batch(model, batch)
            faw = _cbfg_weight_scalar(raw_pred, fg)
            pred_intensity = raw_pred.numpy().mean(axis=-1).astype(np.float32)
            m1_err = m1_data[view_id]["m1_err"]
            bnd_err = item["err"].numpy().astype(np.float32)
            delta = bnd_err - m1_err
            jmax = _scalar_map(item["clear_object_fullsh_raw"], "max")
            tau = _scalar_map(item.get("tau_D", torch.zeros_like(item["err"])), "mean")
            transmission = _scalar_map(item.get("transmission", torch.zeros_like(item["err"])), "mean")
            depth = _scalar_map(item.get("depth", torch.zeros_like(item["err"])), "mean")
            support = _scalar_map(item["accumulation"], "mean") > 0.01
            valid = (
                np.isfinite(faw)
                & np.isfinite(pred_intensity)
                & np.isfinite(bnd_err)
                & np.isfinite(m1_err)
                & np.isfinite(delta)
                & np.isfinite(pseudo)
            )
            view_data[view_id] = {
                "faw": faw,
                "raw_darkness": (1.0 - pred_intensity).astype(np.float32),
                "pred_intensity": pred_intensity,
                "bnd_err": bnd_err,
                "m1_err": m1_err.astype(np.float32),
                "delta": delta.astype(np.float32),
                "positive_delta": np.maximum(delta, 0.0).astype(np.float32),
                "pseudo_depth": pseudo.astype(np.float32),
                "foreground": fg,
                "tau": tau.astype(np.float32),
                "transmission": transmission.astype(np.float32),
                "jmax": jmax.astype(np.float32),
                "depth": depth.astype(np.float32),
                "support": support,
                "valid": valid,
                "M1_HIGH_J": m1_data[view_id]["m1_highj"].astype(bool),
            }
            pseudo_rows.append({"split": split, "view_id": view_id, **meta})
            del item
            torch.cuda.empty_cache()
        summary = {"loaded_step": loaded.loaded_step, "view_count": len(view_data)}
        _write_csv(output_dir / f"pseudo_depth_foreground_audit_{split}.csv", pseudo_rows)
        return view_data, summary, pseudo_rows
    finally:
        _release(loaded)


def _add_persistent_labels(repo: Path, split: str, view_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    counts = {view_id: np.zeros_like(data["support"], dtype=np.uint8) for view_id, data in view_data.items()}
    late_loaded = []
    for nominal in LATE_STEPS:
        loaded = None
        try:
            loaded = _load_run(repo, "K1", nominal, load_depths=False)
            late_loaded.append(loaded.loaded_step)
            model = loaded.pipeline.model
            records = _records(loaded.pipeline, split)
            for _idx, view_id, camera, batch in records:
                if view_id not in view_data:
                    continue
                item = _render_no_grad(model, camera, batch)
                support = view_data[view_id]["support"]
                top = _top_mask_np(item["err"].numpy().astype(np.float32), support, 0.10)
                counts[view_id] += top.astype(np.uint8)
                del item
                torch.cuda.empty_cache()
        finally:
            _release(loaded)
    for view_id, data in view_data.items():
        persistent = data["support"] & (counts[view_id] >= 3)
        data["PERSISTENT_BND_HARD"] = persistent
        data["BND_HARD_CORE"] = persistent & data["M1_HIGH_J"]
    return {
        "split": split,
        "late_steps_requested": list(LATE_STEPS),
        "late_steps_loaded": late_loaded,
        "persistent_required_count": 3,
        "definition": "Per-view K1 late residual top 10 percent inside final K1 object support for at least 3 of 4 late checkpoints.",
    }


def _correlation_rows(view_data: Mapping[str, Mapping[str, np.ndarray]], split: str) -> List[Dict[str, Any]]:
    rows = []
    for domain_name, domain_expr in (
        ("all_valid", None),
        ("valid_foreground", "foreground"),
    ):
        keys = ("faw", "raw_darkness", "bnd_err", "positive_delta", "delta", "tau", "transmission", "pred_intensity", "jmax", "depth")
        if domain_expr is None:
            domain_data = {}
            for view_id, data in view_data.items():
                domain_data[view_id] = {**data, "_domain": data["valid"].astype(bool)}
            arr = _aggregate_arrays(domain_data, keys, "_domain")
        else:
            domain_data = {}
            for view_id, data in view_data.items():
                domain_data[view_id] = {**data, "_domain": data["valid"].astype(bool) & data[domain_expr].astype(bool)}
            arr = _aggregate_arrays(domain_data, keys, "_domain")
        rows.extend(
            [
                {"split": split, "domain": domain_name, "association": "Spearman(FAW,BND_RGB_error)", "spearman": _spearman_np(arr["faw"], arr["bnd_err"]), "n": int(arr["faw"].size)},
                {"split": split, "domain": domain_name, "association": "Spearman(FAW,positive_delta_e_BND)", "spearman": _spearman_np(arr["faw"], arr["positive_delta"]), "n": int(arr["faw"].size)},
                {"split": split, "domain": domain_name, "association": "Spearman(raw_darkness,positive_delta_e_BND)", "spearman": _spearman_np(arr["raw_darkness"], arr["positive_delta"]), "n": int(arr["faw"].size)},
                {"split": split, "domain": domain_name, "association": "Spearman(FAW,tau)", "spearman": _spearman_np(arr["faw"], arr["tau"]), "n": int(arr["faw"].size)},
                {"split": split, "domain": domain_name, "association": "Spearman(FAW,transmission_T)", "spearman": _spearman_np(arr["faw"], arr["transmission"]), "n": int(arr["faw"].size)},
                {"split": split, "domain": domain_name, "association": "Spearman(FAW,underwater_RGB_intensity)", "spearman": _spearman_np(arr["faw"], arr["pred_intensity"]), "n": int(arr["faw"].size)},
                {"split": split, "domain": domain_name, "association": "Spearman(FAW,bounded_Jmax)", "spearman": _spearman_np(arr["faw"], arr["jmax"]), "n": int(arr["faw"].size)},
                {"split": split, "domain": domain_name, "association": "Spearman(FAW,depth)", "spearman": _spearman_np(arr["faw"], arr["depth"]), "n": int(arr["faw"].size)},
            ]
        )
    return rows


def _enrichment_rows(view_data: Mapping[str, Mapping[str, np.ndarray]], split: str) -> List[Dict[str, Any]]:
    domain_data = {}
    for view_id, data in view_data.items():
        domain_data[view_id] = {**data, "_domain": data["valid"].astype(bool) & data["foreground"].astype(bool)}
    arr = _aggregate_arrays(domain_data, ("faw", "raw_darkness", "bnd_err", "delta", "positive_delta"), "_domain")
    rows = []
    if arr["faw"].size == 0:
        return rows
    base_pos_frac = float((arr["delta"] > 0).mean())
    base_pos_sum = float(arr["positive_delta"].sum())
    for score_name in ("faw", "raw_darkness"):
        order = np.argsort(arr[score_name])
        n = arr[score_name].size
        for frac in TOP_FRACTIONS:
            k = max(1, int(math.ceil(frac * n)))
            selected = order[-k:]
            pos_sum = float(arr["positive_delta"][selected].sum())
            pos_frac = float((arr["delta"][selected] > 0).mean())
            pixel_share = k / max(n, 1)
            rows.append(
                {
                    "split": split,
                    "domain": "valid_foreground",
                    "ranking_signal": score_name,
                    "top_fraction": frac,
                    "pixel_fraction": pixel_share,
                    "mean_BND_RGB_error": float(arr["bnd_err"][selected].mean()),
                    "median_BND_RGB_error": float(np.median(arr["bnd_err"][selected])),
                    "mean_delta_e_BND": float(arr["delta"][selected].mean()),
                    "median_delta_e_BND": float(np.median(arr["delta"][selected])),
                    "fraction_delta_e_BND_gt_0": pos_frac,
                    "share_total_positive_BND_excess_MSE_captured": pos_sum / max(base_pos_sum, EPS),
                    "positive_regression_enrichment": pos_frac / max(base_pos_frac, EPS),
                    "positive_excess_error_concentration": (pos_sum / max(base_pos_sum, EPS)) / max(pixel_share, EPS),
                    "foreground_population_positive_delta_fraction": base_pos_frac,
                    "foreground_population_positive_delta_sum": base_pos_sum,
                }
            )
    return rows


def _hard_region_rows(view_data: Mapping[str, Mapping[str, np.ndarray]], split: str) -> List[Dict[str, Any]]:
    rows = []
    fg_arrays = _aggregate_arrays(
        {v: {**d, "_domain": d["valid"].astype(bool) & d["foreground"].astype(bool)} for v, d in view_data.items()},
        ("faw", "positive_delta"),
        "_domain",
    )
    fg_mean_faw = float(fg_arrays["faw"].mean()) if fg_arrays["faw"].size else float("nan")
    total_positive = float(
        sum(float(np.maximum(data["delta"][data["valid"].astype(bool)], 0.0).sum()) for data in view_data.values())
    )
    top_masks: Dict[Tuple[str, float], Dict[str, np.ndarray]] = {}
    for frac in TOP_FRACTIONS:
        top_masks[("faw", frac)] = {
            view_id: _top_mask_np(data["faw"], data["valid"].astype(bool) & data["foreground"].astype(bool), frac)
            for view_id, data in view_data.items()
        }
    for label in LABELS:
        all_vals = []
        region_faw = []
        region_delta = []
        region_bnd = []
        region_fg_faw = []
        region_pixels = 0
        total_pixels = 0
        positive_sum = 0.0
        top_hits = {frac: 0 for frac in TOP_FRACTIONS}
        for view_id, data in view_data.items():
            valid = data["valid"].astype(bool)
            region = data[label].astype(bool) & valid
            region_pixels += int(region.sum())
            total_pixels += int(valid.sum())
            if int(region.sum()) == 0:
                continue
            all_vals.append(region)
            region_faw.append(data["faw"][region])
            region_delta.append(data["delta"][region])
            region_bnd.append(data["bnd_err"][region])
            fg_region = region & data["foreground"].astype(bool)
            if int(fg_region.sum()) > 0:
                region_fg_faw.append(data["faw"][fg_region])
            positive_sum += float(np.maximum(data["delta"][region], 0.0).sum())
            for frac in TOP_FRACTIONS:
                top_hits[frac] += int((region & top_masks[("faw", frac)][view_id]).sum())
        faw_vals = np.concatenate(region_faw) if region_faw else np.asarray([], dtype=np.float32)
        delta_vals = np.concatenate(region_delta) if region_delta else np.asarray([], dtype=np.float32)
        bnd_vals = np.concatenate(region_bnd) if region_bnd else np.asarray([], dtype=np.float32)
        fg_faw_vals = np.concatenate(region_fg_faw) if region_fg_faw else np.asarray([], dtype=np.float32)
        row = {
            "split": split,
            "label": label,
            "pixels": region_pixels,
            "valid_pixels": total_pixels,
            "coverage": region_pixels / max(total_pixels, 1),
            "FAW_mean": float(faw_vals.mean()) if faw_vals.size else float("nan"),
            "FAW_median": float(np.median(faw_vals)) if faw_vals.size else float("nan"),
            "FAW_p90": _safe_quantile_np(faw_vals, 0.90),
            "FAW_foreground_mean": float(fg_faw_vals.mean()) if fg_faw_vals.size else float("nan"),
            "FAW_enrichment_relative_to_valid_foreground": (float(fg_faw_vals.mean()) / max(fg_mean_faw, EPS)) if fg_faw_vals.size else float("nan"),
            "BND_RGB_MSE": float(bnd_vals.mean()) if bnd_vals.size else float("nan"),
            "delta_e_BND_mean": float(delta_vals.mean()) if delta_vals.size else float("nan"),
            "delta_e_BND_median": float(np.median(delta_vals)) if delta_vals.size else float("nan"),
            "fraction_delta_e_BND_gt_0": float((delta_vals > 0).mean()) if delta_vals.size else float("nan"),
            "positive_excess_MSE_share": positive_sum / max(total_positive, EPS),
        }
        for frac in TOP_FRACTIONS:
            row[f"fraction_region_inside_FAW_top_{int(frac*100)}"] = top_hits[frac] / max(region_pixels, 1)
        rows.append(row)
    return rows


def _classification(
    corr_rows: Sequence[Mapping[str, Any]],
    enrich_rows: Sequence[Mapping[str, Any]],
    hard_rows: Sequence[Mapping[str, Any]],
    grad_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    def find_corr(name: str, domain: str) -> float:
        for row in corr_rows:
            if row.get("association") == name and row.get("domain") == domain and row.get("split") == "train":
                return float(row.get("spearman", float("nan")))
        return float("nan")

    faw_delta = find_corr("Spearman(FAW,positive_delta_e_BND)", "valid_foreground")
    dark_delta = find_corr("Spearman(raw_darkness,positive_delta_e_BND)", "valid_foreground")
    faw_top10 = next(
        (row for row in enrich_rows if row.get("split") == "train" and row.get("ranking_signal") == "faw" and abs(float(row.get("top_fraction", 0.0)) - 0.10) < 1e-9),
        {},
    )
    dark_top10 = next(
        (row for row in enrich_rows if row.get("split") == "train" and row.get("ranking_signal") == "raw_darkness" and abs(float(row.get("top_fraction", 0.0)) - 0.10) < 1e-9),
        {},
    )
    faw_enrichment = float(faw_top10.get("positive_regression_enrichment", float("nan")))
    concentration = float(faw_top10.get("positive_excess_error_concentration", float("nan")))
    darkness_enrichment = float(dark_top10.get("positive_regression_enrichment", float("nan")))
    hard_best = max(
        [
            float(row.get("FAW_enrichment_relative_to_valid_foreground", float("nan")))
            for row in hard_rows
            if row.get("split") == "train" and math.isfinite(float(row.get("FAW_enrichment_relative_to_valid_foreground", float("nan"))))
        ]
        or [float("nan")]
    )
    medium_ratio = float(grad_summary.get("medium_to_object_ratio_CBFG", float("nan")))
    object_ratio = float(grad_summary.get("object_group_ratio_CBFG_BASE", float("nan")))
    medium_pathological = math.isfinite(medium_ratio) and medium_ratio > 1.0
    meaningful = (
        math.isfinite(faw_enrichment)
        and faw_enrichment >= 1.25
        and math.isfinite(concentration)
        and concentration >= 1.25
        and math.isfinite(hard_best)
        and hard_best >= 1.10
    )
    stronger_than_dark = math.isfinite(faw_enrichment) and math.isfinite(darkness_enrichment) and faw_enrichment > darkness_enrichment * 1.05
    if meaningful and stronger_than_dark and not medium_pathological:
        label = "CBFG_READY"
    elif meaningful or (math.isfinite(faw_delta) and faw_delta > 0 and not stronger_than_dark):
        label = "CBFG_WEAK_ALIGNMENT"
    else:
        label = "CBFG_NOT_SUPPORTED"
    return {
        "classification": label,
        "decision_rule": "READY requires positive BND-regression enrichment, positive-excess concentration, registered hard-population signal not explained by darkness, and no medium-dominated gradient pattern.",
        "train_foreground_spearman_faw_positive_delta": faw_delta,
        "train_foreground_spearman_darkness_positive_delta": dark_delta,
        "faw_top10_positive_regression_enrichment": faw_enrichment,
        "darkness_top10_positive_regression_enrichment": darkness_enrichment,
        "faw_top10_positive_excess_concentration": concentration,
        "best_hard_region_faw_enrichment_vs_foreground": hard_best,
        "object_group_ratio_CBFG_BASE": object_ratio,
        "medium_to_object_ratio_CBFG": medium_ratio,
        "medium_pathological": medium_pathological,
        "recommendation": "Do not launch CB-FG training in this task. Next single experiment is BND + CB-FG-only only if classification is CBFG_READY; otherwise run one read-only alternative responsibility-signal audit.",
    }


def _comparison_table(historical: Mapping[str, Any], classification: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "Mechanism": "LOSSRESP",
            "Weighting signal": "diagnostic responsibility accounting; SeaFree reference was inverse intensity",
            "Target population": "M1_HIGH_J / bright localized BND RGB errors",
            "Gradient pathway": "image-space and parameter-space no-step audit",
            "Historical local effect": f"high-J MSE share {historical['LOSSRESP']['QUANTITATIVE RESULT'].get('highj_error_share_mse', 'NOT RECOVERED')}, gradient share {historical['LOSSRESP']['QUANTITATIVE RESULT'].get('highj_total_grad_share', 'NOT RECOVERED')}",
            "Historical global effect": "no training run; SeaFree-specific hypothesis not supported",
            "Alignment with BND regression": historical["LOSSRESP"]["INFERENCE"].get("seafree_specific", "NOT RECOVERED"),
        },
        {
            "Mechanism": "UNORM",
            "Weighting signal": "absolute photometric loss, removes inverse-prediction normalization",
            "Target population": "bright / legacy M1_HIGH_J responsibility",
            "Gradient pathway": "all RGB photometric paths, object and medium",
            "Historical local effect": f"HIGH_J_MSE_GAP_RECOVERY={historical['UNORM']['QUANTITATIVE RESULT'].get('HIGH_J_MSE_GAP_RECOVERY', 'NOT RECOVERED')}",
            "Historical global effect": f"PSNR gain={historical['UNORM']['QUANTITATIVE RESULT'].get('STAGE_PSNR_GAIN', 'NOT RECOVERED')}, RGB_SAFETY={historical['UNORM']['QUANTITATIVE RESULT'].get('RGB_SAFETY', 'NOT RECOVERED')}",
            "Alignment with BND regression": "local high-J recovery, global / perceptual failure",
        },
        {
            "Mechanism": "SeaFree CB-FG / FAW",
            "Weighting signal": "foreground 1/(rendered_underwater_rgb.detach()+1e-3), scalar diagnostic mean RGB weight",
            "Target population": "audited against positive delta_e_BND = BND MSE - M1 MSE",
            "Gradient pathway": "weighted photometric no-step gradient through WaterSplatting RGB output",
            "Historical local effect": "this audit only",
            "Historical global effect": "no training run",
            "Alignment with BND regression": classification.get("classification", "NOT RECOVERED"),
        },
    ]


def _write_research_note(
    repo_manifest: Mapping[str, Any],
    env: Mapping[str, Any],
    omvc: Mapping[str, Any],
    historical: Mapping[str, Any],
    pseudo: Mapping[str, Any],
    corr_rows: Sequence[Mapping[str, Any]],
    enrich_rows: Sequence[Mapping[str, Any]],
    hard_rows: Sequence[Mapping[str, Any]],
    grad: Mapping[str, Any],
    classification: Mapping[str, Any],
) -> None:
    def row_lookup(rows: Sequence[Mapping[str, Any]], **kwargs: Any) -> Mapping[str, Any]:
        for row in rows:
            if all(row.get(k) == v for k, v in kwargs.items()):
                return row
        return {}

    faw_corr = row_lookup(corr_rows, split="train", domain="valid_foreground", association="Spearman(FAW,positive_delta_e_BND)")
    dark_corr = row_lookup(corr_rows, split="train", domain="valid_foreground", association="Spearman(raw_darkness,positive_delta_e_BND)")
    faw_top10 = next((r for r in enrich_rows if r.get("split") == "train" and r.get("ranking_signal") == "faw" and abs(float(r.get("top_fraction", 0)) - 0.10) < 1e-9), {})
    lines = [
        "# BND-CBFG Alignment Audit - 2026-08-15",
        "",
        "## Repo",
        f"EXPERIMENTAL FACT: Branch `{repo_manifest.get('branch')}`, HEAD `{repo_manifest.get('head')}`.",
        "",
        "## Environment",
        f"EXPERIMENTAL FACT: CONDA_ENV `{env.get('CONDA_ENV')}`, PYTHON_PATH `{env.get('PYTHON_PATH')}`, TORCH_VERSION `{env.get('TORCH_VERSION')}`.",
        f"EXPERIMENTAL FACT: CUDA_VISIBLE_DEVICES `{env.get('CUDA_VISIBLE_DEVICES')}` maps torch logical cuda:0 to physical GPU `{env.get('CUDA_VISIBLE_DEVICES')}`.",
        "",
        "## OMVC Trajectory Closure",
        f"QUANTITATIVE RESULT: closure `{omvc.get('closure_classification')}`; final registered O1-C0 `{omvc.get('final_registered_O1_minus_C0')}`; final clear-J O1-C0 `{omvc.get('final_clearJ_O1_minus_C0')}`.",
        "INFERENCE: BND-OMVC-DIRECT_OBJECT_SIGNAL remains CLOSED. Pure intrinsic-J OMVC remains untested due current backward limitation.",
        "",
        "## Historical Recovery",
        f"QUANTITATIVE RESULT: LOSSRESP SeaFree-specific hypothesis `{historical['LOSSRESP']['INFERENCE'].get('seafree_specific')}`; high-J MSE share `{historical['LOSSRESP']['QUANTITATIVE RESULT'].get('highj_error_share_mse')}`; high-J gradient share `{historical['LOSSRESP']['QUANTITATIVE RESULT'].get('highj_total_grad_share')}`.",
        f"QUANTITATIVE RESULT: UNORM PSNR gain `{historical['UNORM']['QUANTITATIVE RESULT'].get('STAGE_PSNR_GAIN')}`, HIGH_J_MSE_GAP_RECOVERY `{historical['UNORM']['QUANTITATIVE RESULT'].get('HIGH_J_MSE_GAP_RECOVERY')}`, RGB_SAFETY `{historical['UNORM']['QUANTITATIVE RESULT'].get('RGB_SAFETY')}`.",
        "",
        "## SeaFree CB-FG Semantics",
        "CODE FACT: This audit reproduces only foreground-aware reconstruction weighting: normalized pseudo-depth, threshold 1e-2, THRESH_BINARY_INV, largest-contour foreground, W=1/(rendered_underwater_rgb.detach()+1e-3) on foreground and W=1 on background.",
        "CONFIG FACT: CB-BG, coarse-depth loss, OMVC, CDEPTH, depth residuals, depth-aware alpha, and training are excluded.",
        "",
        "## Pseudo-Depth / Foreground",
        f"EXPERIMENTAL FACT: locked pseudo-depth source `{pseudo.get('pseudo_depth_source')}`; availability `{pseudo.get('LOCKED_PSEUDODEPTH_AVAILABLE')}`.",
        "",
        "## FAW Alignment",
        f"QUANTITATIVE RESULT: train valid-foreground Spearman(FAW, positive delta_e_BND) `{faw_corr.get('spearman')}`; Spearman(raw darkness, positive delta_e_BND) `{dark_corr.get('spearman')}`.",
        f"QUANTITATIVE RESULT: FAW top-10 positive-regression enrichment `{faw_top10.get('positive_regression_enrichment')}`; positive-excess concentration `{faw_top10.get('positive_excess_error_concentration')}`.",
        "",
        "## Hard Regions",
    ]
    for row in hard_rows:
        if row.get("split") == "train":
            lines.append(
                f"QUANTITATIVE RESULT: {row.get('label')} coverage `{row.get('coverage')}`, FAW enrichment vs foreground `{row.get('FAW_enrichment_relative_to_valid_foreground')}`, region in FAW top10 `{row.get('fraction_region_inside_FAW_top_10')}`, delta_e_BND mean `{row.get('delta_e_BND_mean')}`."
            )
    lines.extend(
        [
            "",
            "## Gradient Responsibility",
            f"QUANTITATIVE RESULT: object CBFG/BASE ratio `{grad.get('object_group_ratio_CBFG_BASE')}`, medium CBFG/BASE ratio `{grad.get('medium_group_ratio_CBFG_BASE')}`, CBFG medium/object ratio `{grad.get('medium_to_object_ratio_CBFG')}`, no parameter update `{grad.get('NO_PARAMETER_UPDATE')}`.",
            "INFERENCE: Gradient magnitude alone is not interpreted as disentanglement improvement.",
            "",
            "## Classification",
            f"INFERENCE: Final classification `{classification.get('classification')}`.",
            f"INFERENCE: {classification.get('recommendation')}",
            "",
            "## Next Single Experiment",
            "RECOMMENDATION: If the classification is not CBFG_READY, do not run CB-FG training. The next single experiment should be one read-only alternative responsibility-signal audit; if future evidence reaches CBFG_READY, the only next training experiment should be BND + CB-FG-only.",
            "",
            "## Outputs",
            "- `outputs/bnd_cbfg_alignment_audit_20260815/`",
        ]
    )
    RESEARCH_NOTE.write_text("\n".join(lines) + "\n", encoding="utf8")


def main() -> None:
    _assert_runtime_policy()
    repo = REPO_ROOT
    output_dir = repo / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_manifest = {
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "status_short": _git(repo, "status", "--short"),
        "log_10": _git(repo, "log", "-10", "--oneline"),
        "diff_stat": _git(repo, "diff", "--stat"),
    }
    env = _environment_manifest()
    gpu = {"allowed_physical_gpus": sorted(ALLOWED_PHYSICAL_GPUS), "selected": env.get("gpus", [])}
    _write_json(output_dir / "repo_manifest.json", repo_manifest)
    _write_json(output_dir / "environment_manifest.json", env)
    _write_json(output_dir / "gpu_manifest.json", gpu)

    omvc_rows, omvc_summary = _omvc_trajectory_closure(output_dir)
    historical = _recover_historical(output_dir)

    pseudo_file_audit = _pseudo_depth_file_audit(repo)
    _write_json(output_dir / "pseudo_depth_source_preflight.json", pseudo_file_audit)
    if not pseudo_file_audit["LOCKED_PSEUDODEPTH_AVAILABLE"]:
        final = {"classification": "CBFG_PREFLIGHT_BLOCKED_NO_LOCKED_PSEUDODEPTH", "pseudo_depth_source_preflight": pseudo_file_audit}
        _write_json(output_dir / "final_summary.json", final)
        raise RuntimeError("CBFG_PREFLIGHT_BLOCKED_NO_LOCKED_PSEUDODEPTH")

    checkpoint_availability = {
        "M1_available_steps": sorted(_available_steps(repo / M1_CONFIG)),
        "K1_available_steps": sorted(_available_steps(repo / K1_CONFIG)),
        "start_state_delta_e_BND": "NOT_RECOVERED_M1_STEP_3000_UNAVAILABLE"
        if START_NOMINAL_STEP not in _available_steps(repo / M1_CONFIG)
        else "AVAILABLE",
        "primary_alignment_state": "final/late matched M1 and BND-K1",
    }
    _write_json(output_dir / "checkpoint_state_availability.json", checkpoint_availability)

    all_corr_rows: List[Dict[str, Any]] = []
    all_enrich_rows: List[Dict[str, Any]] = []
    all_hard_rows: List[Dict[str, Any]] = []
    all_pseudo_rows: List[Dict[str, Any]] = []
    split_summaries: Dict[str, Any] = {}
    for split in ("train", "eval"):
        m1_data, m1_summary = _collect_m1_final(repo, split)
        bnd_data, bnd_summary, pseudo_rows = _collect_bnd_final(repo, split, m1_data, output_dir)
        persistent_meta = _add_persistent_labels(repo, split, bnd_data)
        corr_rows = _correlation_rows(bnd_data, split)
        enrich_rows = _enrichment_rows(bnd_data, split)
        hard_rows = _hard_region_rows(bnd_data, split)
        all_corr_rows.extend(corr_rows)
        all_enrich_rows.extend(enrich_rows)
        all_hard_rows.extend(hard_rows)
        all_pseudo_rows.extend(pseudo_rows)
        split_summaries[split] = {"M1": m1_summary, "BND": bnd_summary, "persistent_labels": persistent_meta}
        del m1_data, bnd_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_csv(output_dir / "faw_correlation.csv", all_corr_rows)
    _write_json(output_dir / "faw_correlation.json", {"rows": all_corr_rows})
    _write_csv(output_dir / "enrichment_curves.csv", all_enrich_rows)
    _write_json(output_dir / "enrichment_curves.json", {"rows": all_enrich_rows})
    _write_csv(output_dir / "hard_region_alignment.csv", all_hard_rows)
    _write_json(output_dir / "hard_region_alignment.json", {"rows": all_hard_rows})
    _write_csv(output_dir / "pseudo_depth_foreground_audit.csv", all_pseudo_rows)
    _write_json(output_dir / "pseudo_depth_foreground_audit.json", {"source_preflight": pseudo_file_audit, "rows": all_pseudo_rows})
    _write_csv(output_dir / "decomposition_alignment.csv", [row for row in all_corr_rows if any(token in row["association"] for token in ("tau", "transmission", "intensity", "Jmax", "depth"))])
    _write_json(output_dir / "decomposition_alignment.json", {"rows": [row for row in all_corr_rows if any(token in row["association"] for token in ("tau", "transmission", "intensity", "Jmax", "depth"))]})

    grad_rows, grad_summary = _gradient_audit(repo, output_dir)
    classification = _classification(all_corr_rows, all_enrich_rows, all_hard_rows, grad_summary)
    comparison_rows = _comparison_table(historical, classification)
    _write_csv(output_dir / "comparison_lossresp_unorm_cbfg.csv", comparison_rows)
    _write_json(output_dir / "comparison_lossresp_unorm_cbfg.json", {"rows": comparison_rows})
    _write_json(
        output_dir / "cbfg_semantics.json",
        {
            "CODE FACT": {
                "pseudo_depth": "batch['depth_image'] downscaled if required, normalized by per-image max",
                "foreground_mask": "pseudo_depth < 1e-2 -> uint8 mask, cv2 THRESH_BINARY_INV, largest external contour fill, binarize",
                "weight": "W = 1/(rendered_underwater_rgb.detach()+1e-3) on foreground, W=1 on background",
                "scalar_FAWS": "mean RGB channel weight, used only for diagnostics",
                "weighted_reconstruction": "same detached W multiplies GT and prediction for L1 and DSSIM",
            },
            "CONFIG FACT": {
                "excluded": ["CB-BG", "CGD", "synthetic epipolar depth", "depth residual", "depth-aware alpha", "CDEPTH", "OMVC", "MEDCTX modification", "training"],
            },
        },
    )

    final_summary = {
        "repo": repo_manifest,
        "environment": env,
        "omvc_trajectory_closure": omvc_summary,
        "historical_recovery": {
            "LOSSRESP_SEAFREE_SPECIFIC_HYPOTHESIS": historical["LOSSRESP"]["INFERENCE"].get("seafree_specific"),
            "UNORM_HYPOTHESIS_SUPPORT": historical["UNORM"]["INFERENCE"].get("formal_conclusion"),
        },
        "pseudo_depth_source_preflight": pseudo_file_audit,
        "checkpoint_availability": checkpoint_availability,
        "split_summaries": split_summaries,
        "gradient_summary": grad_summary,
        "classification": classification,
        "outputs": sorted(str(path.relative_to(repo)) for path in output_dir.glob("*")),
    }
    _write_json(output_dir / "final_summary.json", final_summary)
    _write_research_note(repo_manifest, env, omvc_summary, historical, pseudo_file_audit, all_corr_rows, all_enrich_rows, all_hard_rows, grad_summary, classification)


if __name__ == "__main__":
    main()
