#!/usr/bin/env python
"""Read-only Panama BND bounded appearance pressure audit.

This diagnostic tests the single pre-registered BAP score:

    BAP(p) = R_PLUS(p) * J_MAX(p)

It performs no optimizer step, writes no checkpoint, and does not introduce a
new training loss. Historical AA/CDEPTH checkpoints are used only as read-only
retrospective references when available.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
OUTPUT_DIR = Path("outputs/bnd_appearance_pressure_audit_20260816")
RESEARCH_NOTE = Path("research_notes/BND_APPEARANCE_PRESSURE_AUDIT_2026-08-16.md")

M1_CONFIG = aware.M1_CONFIG
K1_CONFIG = cdepth_setup.K1_CONFIG
AA_CONFIG = Path("outputs/bnd_aa_panama_20260810/panama_bnd_aa_seed42_step0_to_15000/water-splatting/20260810_bnd_aa/config.yml")
CDEPTH_CONFIG = Path("outputs/bnd_cdepth_panama_20260811/panama_bnd_cdepth_seed42_step0_to_15000/water-splatting/20260811_bnd_cdepth/config.yml")
DEPTHS_PATH = cdepth_setup.DEPTHS_PATH

FINAL_NOMINAL_STEP = 15000
LATE_STEPS = (8000, 10000, 13000, FINAL_NOMINAL_STEP)
TRAIN_VIEWS = aware.TRAIN_VIEWS
EVAL_VIEWS = aware.EVAL_VIEWS
ALLOWED_PHYSICAL_GPUS = frozenset({"6", "7", "8", "9"})
LABELS = ("M1_HIGH_J", "PERSISTENT_BND_HARD", "BND_HARD_CORE")
SIGNALS = ("BAP", "R_PLUS", "J_MAX", "FAW", "darkness")
TOP_FRACTIONS = (0.10, 0.20, 0.30)
EPS = 1e-12
SEAFREE_THRESHOLD = 1e-2
OBJECT_APPEARANCE_GROUPS = ("features_dc", "features_rest")
OBJECT_GEOMETRY_GROUPS = ("means", "scales", "quats", "opacities")
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
    gpu_rows = []
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
        "torch_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
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


def _run_config(run: str) -> Tuple[Path, str, str]:
    if run == "M1":
        return M1_CONFIG, "legacy", "classic"
    if run == "K1":
        return K1_CONFIG, "bounded_sh3", "classic"
    if run == "AA":
        return AA_CONFIG, "bounded_sh3", "antialiased"
    if run == "CDEPTH":
        return CDEPTH_CONFIG, "bounded_sh3", "classic"
    raise ValueError(run)


def _load_run(repo: Path, run: str, step: int = FINAL_NOMINAL_STEP, *, load_depths: bool = False) -> LoadedRun:
    config_rel, parameterization, rasterize_mode = _run_config(run)
    config_path = repo / config_rel
    actual = _actual_step(config_path, step)

    def update_config(config: Any) -> Any:
        config.load_step = actual
        config.pipeline.model.intrinsic_color_parameterization = parameterization
        config.pipeline.model.rasterize_mode = rasterize_mode
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
    model.config.rasterize_mode = rasterize_mode
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


def _render_no_grad(model: Any, camera: Cameras, batch: Mapping[str, Any], *, keep_bnd_maps: bool = False) -> Dict[str, Tensor]:
    model.eval()
    with torch.no_grad():
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        gt, pred = _gt_pred(model, outputs, batch)
    pred_cpu = pred.detach().float().cpu()
    gt_cpu = gt.detach().float().cpu()
    item: Dict[str, Tensor] = {
        "pred": pred_cpu,
        "gt": gt_cpu,
        "err": (pred_cpu - gt_cpu).square().mean(dim=-1),
    }
    if keep_bnd_maps:
        for key in ("accumulation", "clear_object_fullsh_raw", "depth", "transmission", "tau_D"):
            if key in outputs and isinstance(outputs[key], Tensor):
                item[key] = outputs[key].detach().float().cpu()
    return item


def _scalar_map(tensor: Tensor, mode: str) -> np.ndarray:
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        if mode == "mean":
            arr = arr.mean(axis=-1)
        elif mode == "max":
            arr = arr.max(axis=-1)
        elif mode == "min":
            arr = arr.min(axis=-1)
        else:
            raise ValueError(mode)
    return arr.astype(np.float32)


def _safe_quantile_np(values: np.ndarray, q: float) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q))


def _stats_np(values: np.ndarray, prefix: str = "") -> Dict[str, Any]:
    vals = np.asarray(values).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {f"{prefix}{key}": float("nan") for key in ("mean", "median", "p10", "p90", "p99", "max")}
    return {
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


def _top_mask_np(values: np.ndarray, domain: np.ndarray, fraction: float, *, largest: bool = True) -> np.ndarray:
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
    if largest:
        selected = idx[np.argpartition(vals, -k)[-k:]]
    else:
        selected = idx[np.argpartition(vals, k - 1)[:k]]
    out.reshape(-1)[selected] = True
    return out


def _seafree_foreground_from_batch(model: Any, batch: Mapping[str, Any]) -> np.ndarray:
    pseudo = model._downscale_if_required(batch["depth_image"]).to(model.device).detach().float()
    pseudo = pseudo / pseudo.max().clamp_min(EPS)
    pseudo_np = pseudo.squeeze().cpu().numpy().astype(np.float32)
    mask_1e_2_copy = (pseudo_np < SEAFREE_THRESHOLD).astype(np.uint8) * 255
    _, binary_image = cv2.threshold(mask_1e_2_copy, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    foreground_mask_array = np.zeros_like(binary_image)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(foreground_mask_array, [max_contour], -1, (255,), thickness=cv2.FILLED)
    foreground_mask_array = foreground_mask_array.astype(np.float32) / 255.0
    foreground_mask_array[foreground_mask_array < SEAFREE_THRESHOLD] = 0.0
    foreground_mask_array[foreground_mask_array > 0.0] = 1.0
    return foreground_mask_array.astype(bool)


def _faw_scalar(pred: Tensor, foreground: np.ndarray) -> np.ndarray:
    pred_np = pred.detach().float().cpu().numpy()
    weight = 1.0 / (pred_np + 1e-3)
    weight = np.where(foreground[..., None], weight, np.ones_like(weight))
    return weight.mean(axis=-1).astype(np.float32)


def _aggregate(view_data: Mapping[str, Mapping[str, np.ndarray]], keys: Sequence[str], domain_key: str = "valid") -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key in keys:
        pieces = []
        for data in view_data.values():
            mask = data[domain_key].astype(bool)
            pieces.append(data[key][mask].reshape(-1))
        out[key] = np.concatenate(pieces, axis=0) if pieces else np.asarray([], dtype=np.float32)
    return out


def _collect_m1(repo: Path, split: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    loaded = None
    try:
        loaded = _load_run(repo, "M1", load_depths=False)
        rows = []
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for _idx, view_id, camera, batch in _records(loaded.pipeline, split):
            item = _render_no_grad(loaded.pipeline.model, camera, batch, keep_bnd_maps=True)
            acc = _scalar_map(item["accumulation"], "mean")
            jmax = _scalar_map(item["clear_object_fullsh_raw"], "max")
            highj = (acc > 0.01) & (jmax > 1.0)
            out[view_id] = {"err": item["err"].numpy().astype(np.float32), "M1_HIGH_J": highj.astype(bool)}
            rows.append({"split": split, "view_id": view_id, "M1_HIGH_J_pixels": int(highj.sum()), "total_pixels": int(highj.size), "M1_HIGH_J_fraction": float(highj.mean())})
            del item
            torch.cuda.empty_cache()
        return out, {"loaded_step": loaded.loaded_step, "rows": rows}
    finally:
        _release(loaded)


def _collect_bnd(repo: Path, split: str, m1_data: Mapping[str, Mapping[str, np.ndarray]]) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    loaded = None
    try:
        loaded = _load_run(repo, "K1", load_depths=True)
        model = loaded.pipeline.model
        out: Dict[str, Dict[str, np.ndarray]] = {}
        foreground_pixels = 0
        total_pixels = 0
        for _idx, view_id, camera, batch in _records(loaded.pipeline, split):
            item = _render_no_grad(model, camera, batch, keep_bnd_maps=True)
            gt = item["gt"]
            pred = item["pred"]
            residual = gt - pred
            r_plus = torch.clamp(residual, min=0.0).mean(dim=-1).numpy().astype(np.float32)
            r_minus = torch.clamp(-residual, min=0.0).mean(dim=-1).numpy().astype(np.float32)
            jmax = _scalar_map(item["clear_object_fullsh_raw"], "max")
            foreground = _seafree_foreground_from_batch(model, batch)
            foreground_pixels += int(foreground.sum())
            total_pixels += int(foreground.size)
            faw = _faw_scalar(pred, foreground)
            pred_intensity = pred.numpy().mean(axis=-1).astype(np.float32)
            bnd_err = item["err"].numpy().astype(np.float32)
            delta = bnd_err - m1_data[view_id]["err"]
            valid = (
                np.isfinite(bnd_err)
                & np.isfinite(delta)
                & np.isfinite(r_plus)
                & np.isfinite(r_minus)
                & np.isfinite(jmax)
                & np.isfinite(faw)
                & np.isfinite(pred_intensity)
            )
            out[view_id] = {
                "bnd_err": bnd_err,
                "m1_err": m1_data[view_id]["err"].astype(np.float32),
                "delta": delta.astype(np.float32),
                "positive_delta": np.maximum(delta, 0.0).astype(np.float32),
                "R_PLUS": r_plus,
                "R_MINUS": r_minus,
                "J_MAX": jmax.astype(np.float32),
                "BAP": (r_plus * jmax).astype(np.float32),
                "FAW": faw.astype(np.float32),
                "darkness": (1.0 - pred_intensity).astype(np.float32),
                "valid": valid.astype(bool),
                "support": (_scalar_map(item["accumulation"], "mean") > 0.01),
                "M1_HIGH_J": m1_data[view_id]["M1_HIGH_J"].astype(bool),
            }
            del item
            torch.cuda.empty_cache()
        return out, {
            "loaded_step": loaded.loaded_step,
            "view_count": len(out),
            "SeaFree_foreground_pixels": foreground_pixels,
            "total_pixels": total_pixels,
            "SeaFree_foreground_fraction": foreground_pixels / max(total_pixels, 1),
        }
    finally:
        _release(loaded)


def _add_persistent_labels(repo: Path, split: str, view_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    counts = {view_id: np.zeros_like(data["support"], dtype=np.uint8) for view_id, data in view_data.items()}
    loaded_steps = []
    for nominal in LATE_STEPS:
        loaded = None
        try:
            loaded = _load_run(repo, "K1", step=nominal, load_depths=False)
            loaded_steps.append(loaded.loaded_step)
            for _idx, view_id, camera, batch in _records(loaded.pipeline, split):
                if view_id not in view_data:
                    continue
                item = _render_no_grad(loaded.pipeline.model, camera, batch)
                top = _top_mask_np(item["err"].numpy().astype(np.float32), view_data[view_id]["support"], 0.10)
                counts[view_id] += top.astype(np.uint8)
                del item
                torch.cuda.empty_cache()
        finally:
            _release(loaded)
    for view_id, data in view_data.items():
        persistent = data["support"] & (counts[view_id] >= 3)
        data["PERSISTENT_BND_HARD"] = persistent
        data["BND_HARD_CORE"] = persistent & data["M1_HIGH_J"]
    return {"split": split, "late_steps_requested": list(LATE_STEPS), "late_steps_loaded": loaded_steps, "persistent_required_count": 3}


def _load_intervention_errors(repo: Path, split: str, run: str) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
    loaded = None
    try:
        loaded = _load_run(repo, run, load_depths=False)
        errors: Dict[str, np.ndarray] = {}
        for _idx, view_id, camera, batch in _records(loaded.pipeline, split):
            item = _render_no_grad(loaded.pipeline.model, camera, batch)
            errors[view_id] = item["err"].numpy().astype(np.float32)
            del item
            torch.cuda.empty_cache()
        return errors, {"status": "OK", "run": run, "split": split, "loaded_step": loaded.loaded_step, "config_path": str(loaded.config_path), "checkpoint_path": str(loaded.checkpoint_path)}
    except Exception as exc:
        return None, {"status": "NOT_RECOVERED", "run": run, "split": split, "reason": str(exc)}
    finally:
        _release(loaded)


def _signed_residual_rows(view_data: Mapping[str, Mapping[str, np.ndarray]], split: str) -> List[Dict[str, Any]]:
    rows = []
    arr = _aggregate(view_data, ("R_PLUS", "R_MINUS", "positive_delta", "delta"))
    pos = arr["delta"] > 0.0
    for name in ("R_PLUS", "R_MINUS"):
        vals = arr[name][pos]
        rows.append(
            {
                "split": split,
                "population": "delta_e_BND_gt_0",
                "quantity": name,
                "pixels": int(vals.size),
                **_stats_np(vals),
                "spearman_with_positive_delta_e_BND": _spearman_np(arr[name], arr["positive_delta"]),
            }
        )
    for score_name in ("R_PLUS", "R_MINUS"):
        rows.extend(_top_rows_for_signal(arr, split, score_name, table="signed_residual_direction"))
    return rows


def _top_rows_for_signal(arr: Mapping[str, np.ndarray], split: str, score_name: str, *, table: str) -> List[Dict[str, Any]]:
    rows = []
    score = arr[score_name]
    delta = arr["delta"]
    positive = arr["positive_delta"]
    n = score.size
    if n == 0:
        return rows
    order = np.argsort(score)
    base_pos_frac = float((delta > 0.0).mean())
    base_pos_sum = float(positive.sum())
    for frac in TOP_FRACTIONS:
        k = max(1, int(math.ceil(frac * n)))
        selected = order[-k:]
        pixel_share = k / max(n, 1)
        pos_sum = float(positive[selected].sum())
        pos_frac = float((delta[selected] > 0.0).mean())
        rows.append(
            {
                "split": split,
                "table": table,
                "signal": score_name,
                "top_fraction": frac,
                "pixel_fraction": pixel_share,
                "mean_delta_e_BND": float(delta[selected].mean()),
                "median_delta_e_BND": float(np.median(delta[selected])),
                "P_delta_e_BND_gt_0": pos_frac,
                "positive_regression_enrichment": pos_frac / max(base_pos_frac, EPS),
                "share_total_positive_BND_excess_MSE": pos_sum / max(base_pos_sum, EPS),
                "positive_excess_error_concentration": (pos_sum / max(base_pos_sum, EPS)) / max(pixel_share, EPS),
                "mean_score": float(score[selected].mean()),
                "median_score": float(np.median(score[selected])),
            }
        )
    return rows


def _correlation_rows(view_data: Mapping[str, Mapping[str, np.ndarray]], split: str) -> List[Dict[str, Any]]:
    keys = ("BAP", "R_PLUS", "J_MAX", "FAW", "darkness", "positive_delta")
    arr = _aggregate(view_data, keys)
    rows = []
    for signal in SIGNALS:
        rows.append(
            {
                "split": split,
                "domain": "all_valid",
                "signal": signal,
                "target": "positive_delta_e_BND",
                "spearman": _spearman_np(arr[signal], arr["positive_delta"]),
                "n": int(arr["positive_delta"].size),
            }
        )
    return rows


def _component_rows(view_data: Mapping[str, Mapping[str, np.ndarray]], split: str) -> List[Dict[str, Any]]:
    arr = _aggregate(view_data, ("BAP", "R_PLUS", "J_MAX", "FAW", "darkness", "delta", "positive_delta"))
    rows = []
    for signal in SIGNALS:
        rows.extend(_top_rows_for_signal(arr, split, signal, table="component_ablation"))
    return rows


def _hard_region_rows(view_data: Mapping[str, Mapping[str, np.ndarray]], split: str) -> List[Dict[str, Any]]:
    rows = []
    valid_pixels = int(sum(int(data["valid"].sum()) for data in view_data.values()))
    total_positive = float(sum(float(data["positive_delta"][data["valid"]].sum()) for data in view_data.values()))
    signal_global_means = {}
    for signal in SIGNALS:
        vals = _aggregate(view_data, (signal,))[signal]
        signal_global_means[signal] = float(vals.mean()) if vals.size else float("nan")
    top_masks: Dict[Tuple[str, float], Dict[str, np.ndarray]] = {}
    for signal in SIGNALS:
        for frac in TOP_FRACTIONS:
            top_masks[(signal, frac)] = {
                view_id: _top_mask_np(data[signal], data["valid"], frac)
                for view_id, data in view_data.items()
            }
    for label in LABELS:
        for signal in SIGNALS:
            sig_vals = []
            delta_vals = []
            positive_sum = 0.0
            pixels = 0
            top_hits = {frac: 0 for frac in TOP_FRACTIONS}
            for view_id, data in view_data.items():
                region = data[label].astype(bool) & data["valid"].astype(bool)
                pixels += int(region.sum())
                if int(region.sum()) == 0:
                    continue
                sig_vals.append(data[signal][region])
                delta_vals.append(data["delta"][region])
                positive_sum += float(data["positive_delta"][region].sum())
                for frac in TOP_FRACTIONS:
                    top_hits[frac] += int((region & top_masks[(signal, frac)][view_id]).sum())
            sig_arr = np.concatenate(sig_vals) if sig_vals else np.asarray([], dtype=np.float32)
            delta_arr = np.concatenate(delta_vals) if delta_vals else np.asarray([], dtype=np.float32)
            row = {
                "split": split,
                "label": label,
                "signal": signal,
                "pixels": pixels,
                "valid_pixels": valid_pixels,
                "coverage": pixels / max(valid_pixels, 1),
                "signal_mean": float(sig_arr.mean()) if sig_arr.size else float("nan"),
                "signal_median": float(np.median(sig_arr)) if sig_arr.size else float("nan"),
                "signal_p90": _safe_quantile_np(sig_arr, 0.90),
                "signal_enrichment_vs_all_valid": (float(sig_arr.mean()) / max(signal_global_means[signal], EPS)) if sig_arr.size else float("nan"),
                "mean_delta_e_BND": float(delta_arr.mean()) if delta_arr.size else float("nan"),
                "positive_excess_MSE_share": positive_sum / max(total_positive, EPS),
            }
            for frac in TOP_FRACTIONS:
                row[f"fraction_region_inside_{signal}_top_{int(frac * 100)}"] = top_hits[frac] / max(pixels, 1)
            rows.append(row)
    return rows


def _recoverability_rows(
    view_data: Mapping[str, Mapping[str, np.ndarray]],
    intervention_errors: Mapping[str, np.ndarray],
    split: str,
    run: str,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, np.ndarray]] = {}
    for view_id, data in view_data.items():
        if view_id not in intervention_errors:
            continue
        recovery = data["bnd_err"] - intervention_errors[view_id]
        merged[view_id] = {**data, "recovery": recovery.astype(np.float32), "positive_recovery": np.maximum(recovery, 0.0).astype(np.float32)}
    if not merged:
        return [{"split": split, "run": run, "status": "NOT_RECOVERED"}]
    arr = _aggregate(merged, ("BAP", "R_PLUS", "J_MAX", "FAW", "darkness", "recovery", "positive_recovery"))
    rows = []
    for signal in SIGNALS:
        rows.append(
            {
                "split": split,
                "run": run,
                "signal": signal,
                "metric": "Spearman(signal,positive_recovery_X)",
                "spearman": _spearman_np(arr[signal], arr["positive_recovery"]),
                "n": int(arr["recovery"].size),
            }
        )
        score = arr[signal]
        order = np.argsort(score)
        total_pos = float(arr["positive_recovery"].sum())
        n = score.size
        for frac in TOP_FRACTIONS:
            k = max(1, int(math.ceil(frac * n)))
            selected = order[-k:]
            share = float(arr["positive_recovery"][selected].sum()) / max(total_pos, EPS)
            rows.append(
                {
                    "split": split,
                    "run": run,
                    "signal": signal,
                    "metric": "top_recovery",
                    "top_fraction": frac,
                    "mean_recovery_X": float(arr["recovery"][selected].mean()),
                    "median_recovery_X": float(np.median(arr["recovery"][selected])),
                    "fraction_positive_recovery_X": float((arr["recovery"][selected] > 0.0).mean()),
                    "share_total_positive_recovery_X": share,
                    "positive_recovery_concentration": share / max(k / max(n, 1), EPS),
                }
            )
    return rows


def _snapshot_params(model: Any) -> Dict[str, Tensor]:
    out = {}
    for group, params in model.get_param_groups().items():
        for idx, param in enumerate(params):
            out[f"{group}.{idx}"] = param.detach().cpu().clone()
    return out


def _parameter_delta(before: Mapping[str, Tensor], model: Any) -> Dict[str, Any]:
    after = {}
    for group, params in model.get_param_groups().items():
        for idx, param in enumerate(params):
            after[f"{group}.{idx}"] = param.detach().cpu()
    rows = []
    max_abs = 0.0
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


def _masked_l1_loss(model: Any, outputs: Mapping[str, Tensor], batch: Mapping[str, Any], mask_cpu: np.ndarray) -> Tensor:
    gt, pred = _gt_pred(model, outputs, batch)
    mask = torch.from_numpy(mask_cpu.astype(np.float32)).to(device=model.device, dtype=pred.dtype)
    denom = mask.sum().clamp_min(1.0) * pred.shape[-1]
    return ((gt - pred).abs() * mask[..., None]).sum() / denom


def _grad_probe_for_mask(model: Any, camera: Cameras, batch: Mapping[str, Any], mask_cpu: np.ndarray, population: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    _zero_grad(model)
    outputs = model.get_outputs(camera.to(model.device))
    logits = outputs.get("gaussian_view_logits")
    if not isinstance(logits, Tensor):
        raise RuntimeError("gaussian_view_logits unavailable for bounded sigmoid probe")
    loss = _masked_l1_loss(model, outputs, batch, mask_cpu)
    groups = model.get_param_groups()
    group_items = list(groups.items())
    params = [param for _name, plist in group_items for param in plist]
    grads = torch.autograd.grad(loss, params + [logits], retain_graph=False, allow_unused=True)
    param_grads = grads[:-1]
    grad_s = grads[-1]
    if grad_s is None:
        grad_s = torch.zeros_like(logits)
    c = torch.sigmoid(logits.detach())
    sig_deriv = (c * (1.0 - c)).clamp_min(1e-12)
    grad_c = grad_s.detach() / sig_deriv
    active = grad_s.detach().abs().sum(dim=-1) > 0.0
    if int(active.sum().item()) == 0:
        active = torch.isfinite(logits.detach()).all(dim=-1)
    rows = []
    offset = 0
    group_norms: Dict[str, float] = {}
    for group_name, plist in group_items:
        sq = 0.0
        elems = 0
        for grad in param_grads[offset : offset + len(plist)]:
            offset += 1
            if grad is None:
                continue
            gd = grad.detach().float()
            sq += float((gd * gd).sum().cpu().item())
            elems += int(gd.numel())
        group_norms[group_name] = math.sqrt(sq)
        rows.append({"population": population, "parameter_group": group_name, "grad_l2": group_norms[group_name], "grad_elements": elems})

    def norm_for(names: Sequence[str]) -> float:
        return math.sqrt(sum(group_norms.get(name, 0.0) ** 2 for name in names))

    rows.extend(
        [
            {"population": population, "parameter_group": "OBJECT_APPEARANCE_AGG", "grad_l2": norm_for(OBJECT_APPEARANCE_GROUPS)},
            {"population": population, "parameter_group": "OBJECT_GEOMETRY_AGG", "grad_l2": norm_for(OBJECT_GEOMETRY_GROUPS)},
            {"population": population, "parameter_group": "MEDIUM_AGG", "grad_l2": norm_for(MEDIUM_GROUPS)},
        ]
    )
    c_active = c[active].detach().float().reshape(-1).cpu().numpy()
    deriv_active = sig_deriv[active].detach().float().reshape(-1).cpu().numpy()
    grad_s_active = grad_s.detach().float()[active]
    grad_c_active = grad_c.detach().float()[active]
    ds_norm = float(torch.linalg.norm(grad_s_active).cpu().item()) if grad_s_active.numel() else 0.0
    dc_norm = float(torch.linalg.norm(grad_c_active).cpu().item()) if grad_c_active.numel() else 0.0
    sigmoid = {
        "population": population,
        "mask_pixels": int(mask_cpu.astype(bool).sum()),
        "loss_value": float(loss.detach().cpu().item()),
        "active_gaussian_channels": int(c_active.size),
        "active_gaussians": int(active.sum().detach().cpu().item()),
        "dL_ds_full_l2": ds_norm,
        "dL_dc_chain_rule_l2": dc_norm,
        "ratio_dL_ds_full_over_dL_dc": ds_norm / max(dc_norm, EPS),
        "c_mean": float(c_active.mean()) if c_active.size else float("nan"),
        "c_median": float(np.median(c_active)) if c_active.size else float("nan"),
        "c_p90": _safe_quantile_np(c_active, 0.90),
        "sigmoid_derivative_mean": float(deriv_active.mean()) if deriv_active.size else float("nan"),
        "sigmoid_derivative_median": float(np.median(deriv_active)) if deriv_active.size else float("nan"),
        "sigmoid_derivative_p10": _safe_quantile_np(deriv_active, 0.10),
        "fraction_c_gt_0p90": float((c_active > 0.90).mean()) if c_active.size else float("nan"),
        "fraction_c_gt_0p99": float((c_active > 0.99).mean()) if c_active.size else float("nan"),
    }
    del outputs, logits, loss
    _zero_grad(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, sigmoid


def _fixed_camera_probes(repo: Path, train_data: Mapping[str, Mapping[str, np.ndarray]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    loaded = None
    try:
        loaded = _load_run(repo, "K1", load_depths=True)
        model = loaded.pipeline.model
        model.train()
        model.step = loaded.loaded_step
        records = _records(loaded.pipeline, "train")
        _idx, view_id, camera, batch = records[0]
        data = train_data[view_id]
        domain = data["valid"].astype(bool)
        high = _top_mask_np(data["BAP"], domain, 0.20, largest=True)
        control = _top_mask_np(data["BAP"], domain & (~high), float(high.sum()) / max(float((domain & (~high)).sum()), 1.0), largest=False)
        # The expression above selects an equal-count lowest-BAP control from non-high pixels.
        if int(control.sum()) != int(high.sum()):
            flat = np.flatnonzero((domain & (~high)).reshape(-1))
            vals = data["BAP"].reshape(-1)[flat]
            k = int(high.sum())
            selected = flat[np.argpartition(vals, min(k, vals.size) - 1)[:k]]
            control = np.zeros_like(domain, dtype=bool)
            control.reshape(-1)[selected] = True
        before = _snapshot_params(model)
        rows_high, sig_high = _grad_probe_for_mask(model, camera, batch, high, "BAP_TOP20")
        rows_control, sig_control = _grad_probe_for_mask(model, camera, batch, control, "BAP_LOW_CONTROL_EQCOUNT")
        delta = _parameter_delta(before, model)
        grad_rows = rows_high + rows_control
        sigmoid_rows = [sig_high, sig_control]
        summary = {
            "SIGMOID_CAPACITY_PROBE": "PARTIAL_CHAIN_RULE_FROM_EXPOSED_LOGITS",
            "view_id": view_id,
            "loaded_step": loaded.loaded_step,
            "loss": "normalized masked absolute RGB L1 over fixed pixels",
            "BAP_high_selection": "top 20 percent of valid pixels by pre-registered BAP",
            "control_selection": "equal-count lowest-BAP valid pixels excluding BAP top20",
            "direct_s_full_access": "outputs['gaussian_view_logits'] is non-detached and participates in the underwater RGB forward path",
            "direct_c_access": "non-detached c tensor is not directly exposed; dL/dc is exact chain-rule reconstruction from dL/ds_full and c(1-c)",
            "parameter_delta_max_abs": delta["max_abs_delta"],
            "NO_PARAMETER_UPDATE": bool(delta["max_abs_delta"] == 0.0),
        }
        return grad_rows, sigmoid_rows, {"summary": summary, "parameter_delta_rows": delta["rows"]}
    finally:
        _release(loaded)


def _recover_historical(output_dir: Path) -> Dict[str, Any]:
    lossresp_list = json.loads(Path("outputs/lossresp_audit_20260810/lossresp_final_summary.json").read_text())
    lossresp = {row["key"]: row["value"] for row in lossresp_list}
    unorm_list = json.loads(Path("outputs/bnd_unorm_panama_20260810/bnd_unorm_final_summary.json").read_text())
    unorm = unorm_list[0] if unorm_list else {}
    aa_list = json.loads(Path("outputs/bnd_aa_panama_20260810/aa_final_summary.json").read_text())
    aa = aa_list[0] if aa_list else {}
    cdepth = json.loads(Path("outputs/bnd_cdepth_panama_20260811/bnd_cdepth_final_summary.json").read_text())
    recovered = {
        "LOSSRESP": {
            "QUANTITATIVE RESULT": lossresp,
            "INFERENCE": {"SeaFree-specific hypothesis": lossresp.get("SEAFREE_SPECIFIC_HYPOTHESIS", "NOT_RECOVERED")},
        },
        "UNORM": {
            "QUANTITATIVE RESULT": unorm,
            "CONFIG FACT": "absolute photometric normalization, no foreground/depth/background supervision",
            "INFERENCE": {"formal conclusion": unorm.get("HYPOTHESIS_SUPPORT", "NOT_RECOVERED")},
        },
        "AA": {
            "CONFIG FACT": "bounded_sh3 with rasterize_mode=antialiased; no SeaFree CB/depth/OMVC mechanism",
            "QUANTITATIVE RESULT": aa,
            "INFERENCE": {"mechanism_support": aa.get("MECHANISM_SUPPORT", "NOT_RECOVERED")},
        },
        "CDEPTH": {
            "CONFIG FACT": "bounded_sh3 plus SeaFree-style coarse pseudo-depth loss only; no CB-FG/BG/OMVC",
            "QUANTITATIVE RESULT": cdepth,
            "INFERENCE": {"hypothesis": cdepth.get("Hypothesis", "NOT_RECOVERED")},
        },
    }
    _write_json(output_dir / "recovered_historical_references.json", recovered)
    return recovered


def _classification(
    signed_rows: Sequence[Mapping[str, Any]],
    corr_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    hard_rows: Sequence[Mapping[str, Any]],
    sigmoid_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    train_corr = {row["signal"]: float(row["spearman"]) for row in corr_rows if row.get("split") == "train"}
    bap_rows = [row for row in component_rows if row.get("split") == "train" and row.get("signal") == "BAP"]
    rplus_rows = [row for row in component_rows if row.get("split") == "train" and row.get("signal") == "R_PLUS"]
    jmax_rows = [row for row in component_rows if row.get("split") == "train" and row.get("signal") == "J_MAX"]
    faw_rows = [row for row in component_rows if row.get("split") == "train" and row.get("signal") == "FAW"]
    dark_rows = [row for row in component_rows if row.get("split") == "train" and row.get("signal") == "darkness"]
    bap_enriched_count = sum(float(row.get("positive_regression_enrichment", 0.0)) > 1.0 for row in bap_rows)
    bap_conc_count = sum(float(row.get("positive_excess_error_concentration", 0.0)) > 1.0 for row in bap_rows)
    def mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        vals = [float(row.get(key, float("nan"))) for row in rows]
        vals = [v for v in vals if math.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    bap_mean_enrich = mean_metric(bap_rows, "positive_regression_enrichment")
    rplus_mean_enrich = mean_metric(rplus_rows, "positive_regression_enrichment")
    jmax_mean_enrich = mean_metric(jmax_rows, "positive_regression_enrichment")
    faw_mean_enrich = mean_metric(faw_rows, "positive_regression_enrichment")
    dark_mean_enrich = mean_metric(dark_rows, "positive_regression_enrichment")
    better_than_bad_controls = bap_mean_enrich > max(faw_mean_enrich, dark_mean_enrich)
    adds_beyond_components = bap_mean_enrich > max(rplus_mean_enrich, jmax_mean_enrich) * 1.02
    hard_bap_enriched = any(
        row.get("split") == "train"
        and row.get("signal") == "BAP"
        and math.isfinite(float(row.get("signal_enrichment_vs_all_valid", float("nan"))))
        and float(row.get("signal_enrichment_vs_all_valid", 0.0)) > 1.0
        for row in hard_rows
    )
    rplus_directional = train_corr.get("R_PLUS", float("nan")) > 0.0
    sig_by_pop = {row.get("population"): row for row in sigmoid_rows}
    high = sig_by_pop.get("BAP_TOP20", {})
    ctrl = sig_by_pop.get("BAP_LOW_CONTROL_EQCOUNT", {})
    sigmoid_distinct = False
    if high and ctrl:
        high_deriv = float(high.get("sigmoid_derivative_median", float("nan")))
        ctrl_deriv = float(ctrl.get("sigmoid_derivative_median", float("nan")))
        high_c90 = float(high.get("fraction_c_gt_0p90", float("nan")))
        ctrl_c90 = float(ctrl.get("fraction_c_gt_0p90", float("nan")))
        sigmoid_distinct = (math.isfinite(high_deriv) and math.isfinite(ctrl_deriv) and high_deriv < ctrl_deriv) or (
            math.isfinite(high_c90) and math.isfinite(ctrl_c90) and high_c90 > ctrl_c90
        )
    if (
        rplus_directional
        and bap_enriched_count >= 2
        and bap_conc_count >= 2
        and better_than_bad_controls
        and adds_beyond_components
        and hard_bap_enriched
        and sigmoid_distinct
    ):
        label = "BAP_READY"
    elif rplus_directional and bap_enriched_count >= 1:
        label = "BAP_WEAK_ALIGNMENT"
    else:
        label = "BAP_NOT_SUPPORTED"
    next_step = "BND + BAP-guided responsibility" if label == "BAP_READY" else "CLOSE PANAMA LOSS-RESPONSIBILITY LINE"
    return {
        "classification": label,
        "rplus_directional_association": rplus_directional,
        "BAP_enriched_quantile_count": bap_enriched_count,
        "BAP_positive_excess_concentration_count": bap_conc_count,
        "BAP_mean_positive_regression_enrichment": bap_mean_enrich,
        "R_PLUS_mean_positive_regression_enrichment": rplus_mean_enrich,
        "J_MAX_mean_positive_regression_enrichment": jmax_mean_enrich,
        "FAW_mean_positive_regression_enrichment": faw_mean_enrich,
        "darkness_mean_positive_regression_enrichment": dark_mean_enrich,
        "BAP_better_than_FAW_darkness": better_than_bad_controls,
        "BAP_adds_beyond_R_PLUS_J_MAX": adds_beyond_components,
        "hard_population_BAP_enriched": hard_bap_enriched,
        "sigmoid_probe_distinct": sigmoid_distinct,
        "next_single_experiment": next_step,
    }


def _write_research_note(
    repo_manifest: Mapping[str, Any],
    env: Mapping[str, Any],
    historical: Mapping[str, Any],
    signed_rows: Sequence[Mapping[str, Any]],
    corr_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    hard_rows: Sequence[Mapping[str, Any]],
    sigmoid_rows: Sequence[Mapping[str, Any]],
    grad_rows: Sequence[Mapping[str, Any]],
    grad_summary: Mapping[str, Any],
    recovery_rows: Sequence[Mapping[str, Any]],
    classification: Mapping[str, Any],
) -> None:
    def corr(signal: str) -> Any:
        for row in corr_rows:
            if row.get("split") == "train" and row.get("signal") == signal:
                return row.get("spearman")
        return "NOT_RECOVERED"

    def top(signal: str, frac: float) -> Mapping[str, Any]:
        for row in component_rows:
            if row.get("split") == "train" and row.get("signal") == signal and abs(float(row.get("top_fraction", 0.0)) - frac) < 1e-9:
                return row
        return {}

    def signed(quantity: str) -> Mapping[str, Any]:
        for row in signed_rows:
            if row.get("split") == "train" and row.get("population") == "delta_e_BND_gt_0" and row.get("quantity") == quantity:
                return row
        return {}

    def grad(population: str, group: str) -> Any:
        for row in grad_rows:
            if row.get("population") == population and row.get("parameter_group") == group:
                return row.get("grad_l2")
        return "NOT_RECOVERED"

    def recovery(run: str, split: str, signal: str) -> Any:
        for row in recovery_rows:
            if (
                row.get("split") == split
                and row.get("run") == run
                and row.get("signal") == signal
                and row.get("metric") == "Spearman(signal,positive_recovery_X)"
            ):
                return row.get("spearman")
        return "NOT_RECOVERED"

    lines = [
        "# BND Appearance Pressure Audit - 2026-08-16",
        "",
        "## Scope",
        "CONFIG FACT: This is a read-only, zero-training audit. No optimizer step, checkpoint write, CDEPTH/AA training, CB-FG training, OMVC work, or CUDA backward modification is performed.",
        "HYPOTHESIS: Some BND-specific RGB regression pixels may combine positive observed-radiance underfit with high bounded intrinsic appearance pressure.",
        "",
        "## Repo",
        f"EXPERIMENTAL FACT: Branch `{repo_manifest.get('branch')}`, HEAD `{repo_manifest.get('head')}`.",
        "",
        "## Environment",
        f"EXPERIMENTAL FACT: CONDA_ENV `{env.get('CONDA_ENV')}`, PYTHON_PATH `{env.get('PYTHON_PATH')}`, TORCH_VERSION `{env.get('TORCH_VERSION')}`.",
        f"EXPERIMENTAL FACT: CUDA_VISIBLE_DEVICES `{env.get('CUDA_VISIBLE_DEVICES')}` maps torch logical cuda:0 to physical GPU `{env.get('CUDA_VISIBLE_DEVICES')}`.",
        "",
        "## Recovered Historical References",
        f"QUANTITATIVE RESULT: LOSSRESP high-J MSE share `{historical['LOSSRESP']['QUANTITATIVE RESULT'].get('highj_error_share_mse')}`, high-J gradient share `{historical['LOSSRESP']['QUANTITATIVE RESULT'].get('highj_total_grad_share')}`, SeaFree-specific `{historical['LOSSRESP']['INFERENCE'].get('SeaFree-specific hypothesis')}`.",
        f"QUANTITATIVE RESULT: UNORM PSNR gain `{historical['UNORM']['QUANTITATIVE RESULT'].get('STAGE_PSNR_GAIN')}`, HIGH_J_MSE_GAP_RECOVERY `{historical['UNORM']['QUANTITATIVE RESULT'].get('HIGH_J_MSE_GAP_RECOVERY')}`, RGB_SAFETY `{historical['UNORM']['QUANTITATIVE RESULT'].get('RGB_SAFETY')}`.",
        f"QUANTITATIVE RESULT: AA PSNR gain `{historical['AA']['QUANTITATIVE RESULT'].get('AA_PSNR_GAIN')}`, HIGH_J_MSE_GAP_RECOVERY `{historical['AA']['QUANTITATIVE RESULT'].get('HIGH_J_MSE_GAP_RECOVERY')}`, mechanism support `{historical['AA']['INFERENCE'].get('mechanism_support')}`.",
        f"QUANTITATIVE RESULT: CDEPTH PSNR gain `{historical['CDEPTH']['QUANTITATIVE RESULT'].get('CDEPTH_PSNR_GAIN')}`, HIGH_J_MSE_GAP_RECOVERY `{historical['CDEPTH']['QUANTITATIVE RESULT'].get('HIGH_J_MSE_GAP_RECOVERY')}`, hypothesis `{historical['CDEPTH']['INFERENCE'].get('hypothesis')}`.",
        "",
        "## Signed BND Residual Direction",
        f"QUANTITATIVE RESULT: On positive delta_e_BND train pixels, R_PLUS mean `{signed('R_PLUS').get('mean')}`, median `{signed('R_PLUS').get('median')}`; R_MINUS mean `{signed('R_MINUS').get('mean')}`, median `{signed('R_MINUS').get('median')}`.",
        f"QUANTITATIVE RESULT: Spearman R_PLUS vs positive_delta `{signed('R_PLUS').get('spearman_with_positive_delta_e_BND')}`, R_MINUS `{signed('R_MINUS').get('spearman_with_positive_delta_e_BND')}`.",
        "INFERENCE: The formal train population does not support positive-radiance underfit as the primary signed direction of BND-specific regression.",
        "",
        "## BAP Definition",
        "CODE FACT: `BAP(p) = R_PLUS(p) * J_MAX(p)`, where `R_PLUS=mean_rgb(max(I_GT-I_BND,0))` and `J_MAX=max_rgb(clear_object_fullsh_raw)`.",
        "CODE FACT: `J_MAX` is a bounded intrinsic / dewatered proxy diagnostic, not true color.",
        "",
        "## BAP vs BND Regression",
        f"QUANTITATIVE RESULT: train Spearman BAP `{corr('BAP')}`, R_PLUS `{corr('R_PLUS')}`, J_MAX `{corr('J_MAX')}`, FAW `{corr('FAW')}`, darkness `{corr('darkness')}`.",
        f"QUANTITATIVE RESULT: BAP top-10 enrichment `{top('BAP', 0.10).get('positive_regression_enrichment')}`, positive-excess concentration `{top('BAP', 0.10).get('positive_excess_error_concentration')}`.",
        "",
        "## Control-Signal Comparison",
        f"QUANTITATIVE RESULT: Mean top-quantile positive-regression enrichment: BAP `{classification.get('BAP_mean_positive_regression_enrichment')}`, R_PLUS `{classification.get('R_PLUS_mean_positive_regression_enrichment')}`, J_MAX `{classification.get('J_MAX_mean_positive_regression_enrichment')}`, FAW `{classification.get('FAW_mean_positive_regression_enrichment')}`, darkness `{classification.get('darkness_mean_positive_regression_enrichment')}`.",
        f"INFERENCE: BAP better than FAW/darkness `{classification.get('BAP_better_than_FAW_darkness')}`; BAP adds beyond R_PLUS/J_MAX `{classification.get('BAP_adds_beyond_R_PLUS_J_MAX')}`.",
        "",
        "## Hard Regions",
    ]
    for row in hard_rows:
        if row.get("split") == "train" and row.get("signal") == "BAP":
            lines.append(
                f"QUANTITATIVE RESULT: {row.get('label')} BAP enrichment `{row.get('signal_enrichment_vs_all_valid')}`, top10 overlap `{row.get('fraction_region_inside_BAP_top_10')}`, mean delta_e_BND `{row.get('mean_delta_e_BND')}`."
            )
    lines.extend(
        [
            "",
            "## Sigmoid Capacity Probe",
        ]
    )
    for row in sigmoid_rows:
        lines.append(
            f"QUANTITATIVE RESULT: {row.get('population')} median c `{row.get('c_median')}`, median c(1-c) `{row.get('sigmoid_derivative_median')}`, P(c>0.9) `{row.get('fraction_c_gt_0p90')}`, ratio ||dL/ds||/||dL/dc|| `{row.get('ratio_dL_ds_full_over_dL_dc')}`."
        )
    lines.extend(
        [
            f"EXPERIMENTAL FACT: Sigmoid probe status `{grad_summary.get('SIGMOID_CAPACITY_PROBE')}`; parameter delta max `{grad_summary.get('parameter_delta_max_abs')}`.",
            "",
            "## Gradient Responsibility",
            f"QUANTITATIVE RESULT: BAP_TOP20 aggregate grad L2 appearance `{grad('BAP_TOP20', 'OBJECT_APPEARANCE_AGG')}`, geometry `{grad('BAP_TOP20', 'OBJECT_GEOMETRY_AGG')}`, medium `{grad('BAP_TOP20', 'MEDIUM_AGG')}`.",
            f"QUANTITATIVE RESULT: BAP_LOW_CONTROL_EQCOUNT aggregate grad L2 appearance `{grad('BAP_LOW_CONTROL_EQCOUNT', 'OBJECT_APPEARANCE_AGG')}`, geometry `{grad('BAP_LOW_CONTROL_EQCOUNT', 'OBJECT_GEOMETRY_AGG')}`, medium `{grad('BAP_LOW_CONTROL_EQCOUNT', 'MEDIUM_AGG')}`.",
            "INFERENCE: The no-step probe does not show a clean BAP-high shift toward bounded object appearance responsibility.",
            "",
            "## Recoverability",
        ]
    )
    for run in ("AA", "CDEPTH"):
        lines.append(
            f"QUANTITATIVE RESULT: train/eval Spearman(BAP, positive recovery_{run}) `{recovery(run, 'train', 'BAP')}` / `{recovery(run, 'eval', 'BAP')}`; eval R_PLUS `{recovery(run, 'eval', 'R_PLUS')}`, eval J_MAX `{recovery(run, 'eval', 'J_MAX')}`."
        )
    lines.extend(
        [
            "",
            "## Classification",
            f"INFERENCE: Final classification `{classification.get('classification')}`.",
            f"RECOMMENDATION: `{classification.get('next_single_experiment')}`.",
            "",
            "## Outputs",
            "- `outputs/bnd_appearance_pressure_audit_20260816/`",
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
    gpu_manifest = {"allowed_physical_gpus": sorted(ALLOWED_PHYSICAL_GPUS), "selected": env.get("gpus", [])}
    _write_json(output_dir / "repo_manifest.json", repo_manifest)
    _write_json(output_dir / "environment_manifest.json", env)
    _write_json(output_dir / "gpu_manifest.json", gpu_manifest)
    _write_json(
        output_dir / "bap_definition.json",
        {
            "CODE FACT": {
                "R_PLUS": "mean_rgb(max(I_GT - I_BND, 0)) from BND underwater RGB forward path",
                "R_MINUS": "mean_rgb(max(I_BND - I_GT, 0)) from BND underwater RGB forward path",
                "J_MAX": "max_rgb(outputs['clear_object_fullsh_raw']) bounded intrinsic/dewatered proxy diagnostic",
                "BAP": "R_PLUS * J_MAX",
                "controls": list(SIGNALS[1:]),
            }
        },
    )

    historical = _recover_historical(output_dir)

    all_signed_rows: List[Dict[str, Any]] = []
    all_corr_rows: List[Dict[str, Any]] = []
    all_component_rows: List[Dict[str, Any]] = []
    all_hard_rows: List[Dict[str, Any]] = []
    all_recovery_rows: List[Dict[str, Any]] = []
    split_summaries: Dict[str, Any] = {}
    train_data_for_probe: Optional[Dict[str, Dict[str, np.ndarray]]] = None

    for split in ("train", "eval"):
        m1_data, m1_summary = _collect_m1(repo, split)
        bnd_data, bnd_summary = _collect_bnd(repo, split, m1_data)
        label_summary = _add_persistent_labels(repo, split, bnd_data)
        signed_rows = _signed_residual_rows(bnd_data, split)
        corr_rows = _correlation_rows(bnd_data, split)
        component_rows = _component_rows(bnd_data, split)
        hard_rows = _hard_region_rows(bnd_data, split)
        all_signed_rows.extend(signed_rows)
        all_corr_rows.extend(corr_rows)
        all_component_rows.extend(component_rows)
        all_hard_rows.extend(hard_rows)
        recovery_meta = []
        for run in ("AA", "CDEPTH"):
            errors, meta = _load_intervention_errors(repo, split, run)
            recovery_meta.append(meta)
            if errors is not None:
                all_recovery_rows.extend(_recoverability_rows(bnd_data, errors, split, run))
        split_summaries[split] = {"M1": m1_summary, "BND": bnd_summary, "labels": label_summary, "recoverability": recovery_meta}
        if split == "train":
            train_data_for_probe = bnd_data
        else:
            del bnd_data
        del m1_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if train_data_for_probe is None:
        raise RuntimeError("Missing train data for fixed-camera probes")
    grad_rows, sigmoid_rows, probe_payload = _fixed_camera_probes(repo, train_data_for_probe)
    grad_summary = probe_payload["summary"]
    del train_data_for_probe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    classification = _classification(all_signed_rows, all_corr_rows, all_component_rows, all_hard_rows, sigmoid_rows)

    _write_csv(output_dir / "signed_residual_direction.csv", all_signed_rows)
    _write_json(output_dir / "signed_residual_direction.json", {"rows": all_signed_rows})
    _write_csv(output_dir / "bap_correlation.csv", all_corr_rows)
    _write_json(output_dir / "bap_correlation.json", {"rows": all_corr_rows})
    _write_csv(output_dir / "component_ablation_comparison.csv", all_component_rows)
    _write_json(output_dir / "component_ablation_comparison.json", {"rows": all_component_rows})
    _write_csv(output_dir / "hard_region_alignment.csv", all_hard_rows)
    _write_json(output_dir / "hard_region_alignment.json", {"rows": all_hard_rows})
    _write_csv(output_dir / "gradient_responsibility.csv", grad_rows)
    _write_json(output_dir / "gradient_responsibility.json", {"summary": grad_summary, "rows": grad_rows, "parameter_delta_rows": probe_payload["parameter_delta_rows"]})
    _write_csv(output_dir / "sigmoid_capacity_probe.csv", sigmoid_rows)
    _write_json(output_dir / "sigmoid_capacity_probe.json", {"summary": grad_summary, "rows": sigmoid_rows})
    _write_csv(output_dir / "recoverability_alignment.csv", all_recovery_rows)
    _write_json(output_dir / "recoverability_alignment.json", {"rows": all_recovery_rows})

    final_summary = {
        "repo": repo_manifest,
        "environment": env,
        "historical_references": {
            "LOSSRESP_SEAFREE_SPECIFIC_HYPOTHESIS": historical["LOSSRESP"]["INFERENCE"].get("SeaFree-specific hypothesis"),
            "UNORM_HYPOTHESIS": historical["UNORM"]["INFERENCE"].get("formal conclusion"),
            "AA_MECHANISM_SUPPORT": historical["AA"]["INFERENCE"].get("mechanism_support"),
            "CDEPTH_HYPOTHESIS": historical["CDEPTH"]["INFERENCE"].get("hypothesis"),
        },
        "split_summaries": split_summaries,
        "sigmoid_capacity_probe": grad_summary,
        "classification": classification,
        "outputs": sorted(str(path.relative_to(repo)) for path in output_dir.glob("*")),
    }
    _write_json(output_dir / "final_summary.json", final_summary)
    _write_research_note(
        repo_manifest,
        env,
        historical,
        all_signed_rows,
        all_corr_rows,
        all_component_rows,
        all_hard_rows,
        sigmoid_rows,
        grad_rows,
        grad_summary,
        all_recovery_rows,
        classification,
    )


if __name__ == "__main__":
    main()
