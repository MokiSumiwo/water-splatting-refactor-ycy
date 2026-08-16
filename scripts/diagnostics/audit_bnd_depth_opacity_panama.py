#!/usr/bin/env python
"""Read-only Panama BND depth-opacity contamination audit.

This diagnostic tests whether projected Gaussian depth inconsistency coexists
with meaningful opacity burden and whether that burden aligns with the
remaining BND RGB trade-off. It performs no optimizer step, writes no
checkpoint, and does not introduce any training loss.
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

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup

from scripts.diagnostics import audit_bnd_cdepth_setup as cdepth_setup
from scripts.diagnostics import run_bnd_aware_refine_panama as aware


SCENE = "Panama"
OUTPUT_DIR = Path("outputs/bnd_depth_opacity_audit_20260816")
RESEARCH_NOTE = Path("research_notes/BND_DEPTH_OPACITY_AUDIT_2026-08-16.md")

M1_CONFIG = aware.M1_CONFIG
K1_CONFIG = cdepth_setup.K1_CONFIG
REQUESTED_STEPS = (3000, 5000, 8000, 10000, 15000)
LATE_LABEL_STEPS = (8000, 10000, 13000, 15000)
TRAIN_VIEWS = aware.TRAIN_VIEWS
EVAL_VIEWS = aware.EVAL_VIEWS
ALLOWED_PHYSICAL_GPUS = frozenset({"6", "7", "8", "9"})
SPLITS = ("train", "eval")
RUNS = ("M1", "BND-K1")
LABELS = ("M1_HIGH_J", "PERSISTENT_BND_HARD", "BND_HARD_CORE")
TOP_FRACTIONS = (0.10, 0.20, 0.30)
EPS = 1e-6
SPEARMAN_MAX_N = 1_000_000


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


def _actual_step(config_path: Path, nominal_step: int) -> Optional[int]:
    steps = _available_steps(config_path)
    if nominal_step in steps:
        return nominal_step
    if nominal_step == 15000 and 14999 in steps:
        return 14999
    return None


def _run_spec(run: str) -> Tuple[Path, str, str]:
    if run == "M1":
        return M1_CONFIG, "legacy", "classic"
    if run == "BND-K1":
        return K1_CONFIG, "bounded_sh3", "classic"
    raise ValueError(run)


def _load_run(repo: Path, run: str, nominal_step: int) -> LoadedRun:
    config_rel, parameterization, rasterize_mode = _run_spec(run)
    config_path = repo / config_rel
    actual = _actual_step(config_path, nominal_step)
    if actual is None:
        raise FileNotFoundError(f"Missing {run} checkpoint for nominal step {nominal_step}: {config_path}")

    def update_config(config: Any) -> Any:
        config.load_step = actual
        config.pipeline.model.intrinsic_color_parameterization = parameterization
        config.pipeline.model.rasterize_mode = rasterize_mode
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
    model.config.rasterize_mode = rasterize_mode
    model.config.medium_context_mode = "dir_xy_camera"
    model.config.b_inf_mode = "tied"
    model.config.infinite_water_enabled = False
    model.config.coarse_depth_supervision_enabled = False
    model.step = int(loaded_step)
    pipeline.eval()
    model.eval()
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


def _scalar_map(tensor: Tensor, mode: str = "mean") -> Tensor:
    arr = tensor.detach().float()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        if mode == "mean":
            return arr.mean(dim=-1)
        if mode == "max":
            return arr.amax(dim=-1)
        if mode == "min":
            return arr.amin(dim=-1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Cannot scalarize shape {tuple(arr.shape)} with mode {mode}")


def _safe_np(values: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def _safe_stats(values: np.ndarray, prefix: str = "") -> Dict[str, Any]:
    vals = np.asarray(values).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {f"{prefix}{key}": float("nan") for key in ("mean", "median", "p10", "p90", "p99", "max")}
    return {
        f"{prefix}mean": float(vals.mean()),
        f"{prefix}median": float(np.median(vals)),
        f"{prefix}p10": float(np.quantile(vals, 0.10)),
        f"{prefix}p90": float(np.quantile(vals, 0.90)),
        f"{prefix}p99": float(np.quantile(vals, 0.99)),
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


def _subsample_pair(a: np.ndarray, b: np.ndarray, max_n: int = SPEARMAN_MAX_N) -> Tuple[np.ndarray, np.ndarray, int]:
    finite = np.isfinite(a) & np.isfinite(b)
    aa = a[finite].reshape(-1)
    bb = b[finite].reshape(-1)
    if aa.size > max_n:
        idx = np.linspace(0, aa.size - 1, max_n, dtype=np.int64)
        aa = aa[idx]
        bb = bb[idx]
    return aa, bb, int(aa.size)


def _spearman_np(a: np.ndarray, b: np.ndarray) -> Tuple[float, int]:
    aa, bb, n_used = _subsample_pair(np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1))
    if aa.size < 2:
        return float("nan"), n_used
    ra = _rankdata_average(aa)
    rb = _rankdata_average(bb)
    if float(np.std(ra)) < EPS or float(np.std(rb)) < EPS:
        return float("nan"), n_used
    return float(np.corrcoef(ra, rb)[0, 1]), n_used


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


def _sample_scalar_at_xys(scalar: Tensor, xys: Tensor, chunk: int = 500_000) -> Tensor:
    if scalar.ndim != 2:
        raise ValueError(f"Expected scalar HxW map, got {tuple(scalar.shape)}")
    h, w = scalar.shape
    image = scalar[None, None, :, :].float()
    vals = []
    for start in range(0, xys.shape[0], chunk):
        pts = xys[start : start + chunk].float()
        gx = (pts[:, 0] / max(w - 1, 1)) * 2.0 - 1.0
        gy = (pts[:, 1] / max(h - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
        sampled = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        vals.append(sampled.view(-1))
    return torch.cat(vals, dim=0) if vals else scalar.new_zeros(0)


def _view_direction_row(camera: Cameras) -> Dict[str, Any]:
    c2w = camera.camera_to_worlds[0].detach().float().cpu().numpy()
    center = c2w[:3, 3]
    z_col = c2w[:3, 2]
    return {
        "camera_center_x": float(center[0]),
        "camera_center_y": float(center[1]),
        "camera_center_z": float(center[2]),
        "camera_view_zcol_x": float(z_col[0]),
        "camera_view_zcol_y": float(z_col[1]),
        "camera_view_zcol_z": float(z_col[2]),
    }


def _render_projection_view(model: Any, camera: Cameras, batch: Mapping[str, Any]) -> Dict[str, Any]:
    with torch.no_grad():
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        gt, pred = _gt_pred(model, outputs, batch)
        err = (pred - gt).square().mean(dim=-1).detach().float().cpu().numpy().astype(np.float32)

        depth_map = _scalar_map(outputs["depth"], "mean")
        tau_map = _scalar_map(outputs["tau_D"], "mean") if "tau_D" in outputs else torch.zeros_like(depth_map)
        trans_map = _scalar_map(outputs["transmission"], "mean") if "transmission" in outputs else torch.ones_like(depth_map)
        acc_map = _scalar_map(outputs["accumulation"], "mean") if "accumulation" in outputs else torch.zeros_like(depth_map)
        h, w = depth_map.shape

        xys = model.xys.detach()
        depths = model.depths.detach().float().reshape(-1)
        radii = model.radii.detach().float().reshape(-1)
        opacity = torch.sigmoid(model.opacities.detach().float()).reshape(-1)
        valid = (
            torch.isfinite(xys).all(dim=-1)
            & torch.isfinite(depths)
            & torch.isfinite(radii)
            & (radii > 0)
            & (depths > 0)
            & (xys[:, 0] >= 0)
            & (xys[:, 0] <= (w - 1))
            & (xys[:, 1] >= 0)
            & (xys[:, 1] <= (h - 1))
        )

        xys_v = xys[valid]
        depths_v = depths[valid]
        radii_v = radii[valid]
        opacity_v = opacity[valid]
        sampled_depth = _sample_scalar_at_xys(depth_map, xys_v)
        sampled_tau = _sample_scalar_at_xys(tau_map, xys_v)
        sampled_trans = _sample_scalar_at_xys(trans_map, xys_v)
        sampled_acc = _sample_scalar_at_xys(acc_map, xys_v)

        positive_sample = torch.isfinite(sampled_depth) & (sampled_depth > 0)
        xys_v = xys_v[positive_sample]
        depths_v = depths_v[positive_sample]
        radii_v = radii_v[positive_sample]
        opacity_v = opacity_v[positive_sample]
        sampled_depth = sampled_depth[positive_sample]
        sampled_tau = sampled_tau[positive_sample]
        sampled_trans = sampled_trans[positive_sample]
        sampled_acc = sampled_acc[positive_sample]

        rz_abs = (sampled_depth - depths_v).abs()
        rz_rel = rz_abs / sampled_depth.clamp_min(EPS)
        footprint_area = math.pi * radii_v.square()
        opacity_footprint = opacity_v * footprint_area
        joint = rz_rel * opacity_v
        joint_footprint = joint * footprint_area

        xi = torch.round(xys_v[:, 0]).long().clamp(0, w - 1).detach().cpu().numpy()
        yi = torch.round(xys_v[:, 1]).long().clamp(0, h - 1).detach().cpu().numpy()
        flat = yi * w + xi
        minlength = h * w
        rz_np = rz_rel.detach().float().cpu().numpy().astype(np.float32)
        op_np = opacity_v.detach().float().cpu().numpy().astype(np.float32)
        opfp_np = opacity_footprint.detach().float().cpu().numpy().astype(np.float32)
        joint_np = joint.detach().float().cpu().numpy().astype(np.float32)
        jointfp_np = joint_footprint.detach().float().cpu().numpy().astype(np.float32)
        count = np.bincount(flat, minlength=minlength).astype(np.float32).reshape(h, w)
        rz_sum = np.bincount(flat, weights=rz_np, minlength=minlength).astype(np.float32).reshape(h, w)
        op_sum = np.bincount(flat, weights=op_np, minlength=minlength).astype(np.float32).reshape(h, w)
        opfp_sum = np.bincount(flat, weights=opfp_np, minlength=minlength).astype(np.float32).reshape(h, w)
        joint_sum = np.bincount(flat, weights=joint_np, minlength=minlength).astype(np.float32).reshape(h, w)
        jointfp_sum = np.bincount(flat, weights=jointfp_np, minlength=minlength).astype(np.float32).reshape(h, w)
        rz_mean = np.divide(rz_sum, np.maximum(count, 1.0), out=np.zeros_like(rz_sum), where=count > 0)

        pair = {
            "RZ_ABS": rz_abs.detach().float().cpu().numpy().astype(np.float32),
            "RZ_REL": rz_np,
            "opacity": op_np,
            "radius": radii_v.detach().float().cpu().numpy().astype(np.float32),
            "footprint_area": footprint_area.detach().float().cpu().numpy().astype(np.float32),
            "opacity_footprint": opfp_np,
            "joint_depth_opacity": joint_np,
            "joint_depth_opacity_footprint": jointfp_np,
            "tau": sampled_tau.detach().float().cpu().numpy().astype(np.float32),
            "transmission": sampled_trans.detach().float().cpu().numpy().astype(np.float32),
            "accumulation_sampled": sampled_acc.detach().float().cpu().numpy().astype(np.float32),
            "z_camera": depths_v.detach().float().cpu().numpy().astype(np.float32),
        }
        pixel = {
            "err": err,
            "valid": np.isfinite(err),
            "pixel_projected_count": count,
            "pixel_rz_rel_mean": rz_mean,
            "pixel_opacity_sum": op_sum,
            "pixel_opacity_footprint_sum": opfp_sum,
            "pixel_joint_depth_opacity_sum": joint_sum,
            "pixel_joint_depth_opacity_footprint_sum": jointfp_sum,
            "depth": depth_map.detach().float().cpu().numpy().astype(np.float32),
            "tau": tau_map.detach().float().cpu().numpy().astype(np.float32),
            "transmission": trans_map.detach().float().cpu().numpy().astype(np.float32),
            "accumulation": acc_map.detach().float().cpu().numpy().astype(np.float32),
        }
        if "clear_object_fullsh_raw" in outputs:
            pixel["J_MAX"] = _scalar_map(outputs["clear_object_fullsh_raw"], "max").detach().float().cpu().numpy().astype(np.float32)
        if "gaussian_view_logits" in outputs:
            logits = outputs["gaussian_view_logits"].detach().float()
            visible = outputs.get("gaussian_visible_mask")
            if isinstance(visible, Tensor):
                logits = logits[visible.detach().bool()]
            pixel["gaussian_logits_visible"] = logits.detach().float().cpu().numpy().astype(np.float32)
    return {"pair": pair, "pixel": pixel, "image_shape": (h, w)}


def _concat_pair(pair_views: Sequence[Mapping[str, np.ndarray]], keys: Sequence[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key in keys:
        pieces = [view[key].reshape(-1) for view in pair_views if key in view and view[key].size > 0]
        out[key] = np.concatenate(pieces, axis=0) if pieces else np.asarray([], dtype=np.float32)
    return out


def _pair_distribution_rows(pair_views: Sequence[Mapping[str, np.ndarray]], run: str, nominal_step: int, actual_step: int, split: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    keys = (
        "RZ_ABS",
        "RZ_REL",
        "opacity",
        "radius",
        "footprint_area",
        "opacity_footprint",
        "joint_depth_opacity",
        "joint_depth_opacity_footprint",
        "tau",
        "transmission",
        "z_camera",
    )
    arr = _concat_pair(pair_views, keys)
    n = int(arr["RZ_REL"].size)
    base = {"run": run, "nominal_step": nominal_step, "actual_step": actual_step, "split": split, "pair_count": n}
    rows = [{**base, **_safe_stats(arr["RZ_REL"], "RZ_REL_"), **_safe_stats(arr["opacity"], "opacity_"), **_safe_stats(arr["joint_depth_opacity"], "joint_") }]
    if n:
        rows[0].update(
            {
                "depth_inconsistent_fraction_RZ_REL_gt_0p5": float((arr["RZ_REL"] > 0.5).mean()),
                "depth_inconsistent_fraction_RZ_REL_gt_1p0": float((arr["RZ_REL"] > 1.0).mean()),
                "total_opacity_mass": float(arr["opacity"].sum()),
                "total_opacity_footprint_mass": float(arr["opacity_footprint"].sum()),
                "PER_GAUSSIAN_CONTRIBUTION": "IMPLEMENTATION_BLOCKED",
            }
        )
        sp_op, sp_n = _spearman_np(arr["RZ_REL"], arr["opacity"])
        sp_opfp, _ = _spearman_np(arr["RZ_REL"], arr["opacity_footprint"])
        rows[0].update(
            {
                "spearman_RZ_REL_opacity": sp_op,
                "spearman_RZ_REL_opacity_footprint_proxy": sp_opfp,
                "spearman_sample_n": sp_n,
            }
        )

    top_rows: List[Dict[str, Any]] = []
    joint_rows: List[Dict[str, Any]] = []
    if n:
        rz = arr["RZ_REL"]
        opacity = arr["opacity"]
        opfp = arr["opacity_footprint"]
        joint = arr["joint_depth_opacity"]
        total_op = float(opacity.sum())
        total_opfp = float(opfp.sum())
        order_rz = np.argsort(rz)
        order_op = np.argsort(opacity)
        for frac in TOP_FRACTIONS:
            k = max(1, int(math.ceil(frac * n)))
            rz_sel = order_rz[-k:]
            top_rows.append(
                {
                    **base,
                    "top_by": "RZ_REL",
                    "top_fraction": frac,
                    "selected_pairs": int(k),
                    "pair_fraction": float(k / max(n, 1)),
                    "RZ_REL_mean": float(rz[rz_sel].mean()),
                    "RZ_REL_median": float(np.median(rz[rz_sel])),
                    "opacity_mean": float(opacity[rz_sel].mean()),
                    "opacity_median": float(np.median(opacity[rz_sel])),
                    "opacity_p90": float(np.quantile(opacity[rz_sel], 0.90)),
                    "opacity_mass_share": float(opacity[rz_sel].sum() / max(total_op, EPS)),
                    "opacity_footprint_proxy_mean": float(opfp[rz_sel].mean()),
                    "opacity_footprint_proxy_p90": float(np.quantile(opfp[rz_sel], 0.90)),
                    "opacity_footprint_proxy_mass_share": float(opfp[rz_sel].sum() / max(total_opfp, EPS)),
                    "contribution_status": "IMPLEMENTATION_BLOCKED",
                }
            )
            rz_mask = np.zeros(n, dtype=bool)
            op_mask = np.zeros(n, dtype=bool)
            rz_mask[order_rz[-k:]] = True
            op_mask[order_op[-k:]] = True
            sel = rz_mask & op_mask
            joint_rows.append(
                {
                    **base,
                    "joint_definition": "top RZ_REL fraction AND top opacity fraction",
                    "top_fraction": frac,
                    "selected_pairs": int(sel.sum()),
                    "pair_fraction": float(sel.mean()),
                    "opacity_mass_share": float(opacity[sel].sum() / max(total_op, EPS)) if int(sel.sum()) else 0.0,
                    "opacity_footprint_proxy_mass_share": float(opfp[sel].sum() / max(total_opfp, EPS)) if int(sel.sum()) else 0.0,
                    "joint_depth_opacity_mean": float(joint[sel].mean()) if int(sel.sum()) else float("nan"),
                    "joint_depth_opacity_p90": float(np.quantile(joint[sel], 0.90)) if int(sel.sum()) else float("nan"),
                    "contribution_status": "IMPLEMENTATION_BLOCKED",
                }
            )
    return rows, top_rows, joint_rows


def _medium_association_rows(pair_views: Sequence[Mapping[str, np.ndarray]], camera_rows: Sequence[Mapping[str, Any]], run: str, nominal_step: int, actual_step: int, split: str) -> List[Dict[str, Any]]:
    arr = _concat_pair(pair_views, ("RZ_REL", "opacity", "joint_depth_opacity", "tau", "transmission"))
    rows: List[Dict[str, Any]] = []
    for signal in ("RZ_REL", "opacity", "joint_depth_opacity"):
        for target in ("tau", "transmission"):
            sp, n_used = _spearman_np(arr[signal], arr[target])
            rows.append(
                {
                    "run": run,
                    "nominal_step": nominal_step,
                    "actual_step": actual_step,
                    "split": split,
                    "scope": "projected_gaussian_pairs",
                    "signal": signal,
                    "target": target,
                    "spearman": sp,
                    "n": int(arr[signal].size),
                    "sample_n": n_used,
                }
            )
    if len(camera_rows) >= 3:
        for signal in ("RZ_REL_mean", "joint_depth_opacity_mean", "opacity_mean"):
            sig = np.asarray([float(row.get(signal, float("nan"))) for row in camera_rows], dtype=np.float64)
            for target in ("tau_mean", "transmission_mean", "camera_view_zcol_x", "camera_view_zcol_y", "camera_view_zcol_z"):
                tar = np.asarray([float(row.get(target, float("nan"))) for row in camera_rows], dtype=np.float64)
                sp, n_used = _spearman_np(sig, tar)
                rows.append(
                    {
                        "run": run,
                        "nominal_step": nominal_step,
                        "actual_step": actual_step,
                        "split": split,
                        "scope": "per_camera",
                        "signal": signal,
                        "target": target,
                        "spearman": sp,
                        "n": len(camera_rows),
                        "sample_n": n_used,
                    }
                )
    return rows


def _camera_summary(pair: Mapping[str, np.ndarray], run: str, nominal_step: int, actual_step: int, split: str, view_id: str, camera: Cameras) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "run": run,
        "nominal_step": nominal_step,
        "actual_step": actual_step,
        "split": split,
        "view_id": view_id,
        "pair_count": int(pair["RZ_REL"].size),
    }
    row.update(_view_direction_row(camera))
    for key in ("RZ_REL", "opacity", "joint_depth_opacity", "tau", "transmission"):
        vals = pair[key]
        row[f"{key}_mean"] = float(vals.mean()) if vals.size else float("nan")
        row[f"{key}_median"] = float(np.median(vals)) if vals.size else float("nan")
    return row


def _collect_run_step(repo: Path, run: str, nominal_step: int) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    loaded = None
    try:
        loaded = _load_run(repo, run, nominal_step)
        model = loaded.pipeline.model
        model.eval()
        split_views: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        rows: Dict[str, List[Dict[str, Any]]] = {
            "distribution": [],
            "rz_top": [],
            "joint": [],
            "medium": [],
            "camera": [],
            "decomposition": [],
        }
        for split in SPLITS:
            pair_views = []
            camera_rows = []
            view_maps: Dict[str, Dict[str, np.ndarray]] = {}
            j_values = []
            tau_values = []
            trans_values = []
            logit_values = []
            for _idx, view_id, camera, batch in _records(loaded.pipeline, split):
                item = _render_projection_view(model, camera, batch)
                pair = item["pair"]
                pixel = item["pixel"]
                view_maps[view_id] = pixel
                pair_views.append(pair)
                cam_row = _camera_summary(pair, run, nominal_step, loaded.loaded_step, split, view_id, camera)
                camera_rows.append(cam_row)
                rows["camera"].append(cam_row)
                if run == "BND-K1":
                    if "J_MAX" in pixel:
                        j_values.append(pixel["J_MAX"].reshape(-1))
                    tau_values.append(pixel["tau"].reshape(-1))
                    trans_values.append(pixel["transmission"].reshape(-1))
                    if "gaussian_logits_visible" in pixel:
                        logit_values.append(pixel["gaussian_logits_visible"].reshape(-1))
                torch.cuda.empty_cache()
            dist, rz_top, joint = _pair_distribution_rows(pair_views, run, nominal_step, loaded.loaded_step, split)
            rows["distribution"].extend(dist)
            rows["rz_top"].extend(rz_top)
            rows["joint"].extend(joint)
            rows["medium"].extend(_medium_association_rows(pair_views, camera_rows, run, nominal_step, loaded.loaded_step, split))
            if run == "BND-K1":
                j_all = np.concatenate(j_values) if j_values else np.asarray([], dtype=np.float32)
                tau_all = np.concatenate(tau_values) if tau_values else np.asarray([], dtype=np.float32)
                t_all = np.concatenate(trans_values) if trans_values else np.asarray([], dtype=np.float32)
                logits_all = np.concatenate(logit_values) if logit_values else np.asarray([], dtype=np.float32)
                c_all = 1.0 / (1.0 + np.exp(-logits_all)) if logits_all.size else np.asarray([], dtype=np.float32)
                decomp = {
                    "run": run,
                    "nominal_step": nominal_step,
                    "actual_step": loaded.loaded_step,
                    "split": split,
                    "J_p99": float(np.quantile(j_all[np.isfinite(j_all)], 0.99)) if j_all.size else float("nan"),
                    "P_J_gt_1": float((j_all > 1.0).mean()) if j_all.size else float("nan"),
                    "tau_p90": float(np.quantile(tau_all[np.isfinite(tau_all)], 0.90)) if tau_all.size else float("nan"),
                    "tau_p99": float(np.quantile(tau_all[np.isfinite(tau_all)], 0.99)) if tau_all.size else float("nan"),
                    "P_T_lt_0p1": float((t_all < 0.1).mean()) if t_all.size else float("nan"),
                    "P_c_gt_0p99": float((c_all > 0.99).mean()) if c_all.size else float("nan"),
                    "P_abs_s_full_gt_5": float((np.abs(logits_all) > 5.0).mean()) if logits_all.size else float("nan"),
                    "visible_logit_channels": int(logits_all.size),
                }
                rows["decomposition"].append(decomp)
            split_views[split] = view_maps
        manifest = {
            "run": run,
            "nominal_step": nominal_step,
            "actual_step": loaded.loaded_step,
            "config_path": str(loaded.config_path),
            "checkpoint_path": str(loaded.checkpoint_path),
            "num_points": int(model.num_points),
        }
        return split_views, manifest, rows
    finally:
        _release(loaded)


def _collect_hard_labels(repo: Path) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, Any]]:
    labels: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {split: {} for split in SPLITS}
    manifest: Dict[str, Any] = {"definitions": {
        "M1_HIGH_J": "final M1 accumulation > 0.01 and final M1 clear_object_fullsh_raw max RGB > 1.0",
        "PERSISTENT_BND_HARD": "inside final BND support, RGB residual top 10% for at least 3 of late BND checkpoints",
        "BND_HARD_CORE": "M1_HIGH_J AND PERSISTENT_BND_HARD",
    }}
    final_m1 = None
    try:
        final_m1 = _load_run(repo, "M1", 15000)
        manifest["final_m1_actual_step"] = final_m1.loaded_step
        for split in SPLITS:
            for _idx, view_id, camera, batch in _records(final_m1.pipeline, split):
                with torch.no_grad():
                    out = final_m1.pipeline.model.get_outputs_for_camera(camera.to(final_m1.pipeline.model.device))
                acc = _scalar_map(out["accumulation"], "mean").detach().float().cpu().numpy()
                jmax = _scalar_map(out["clear_object_fullsh_raw"], "max").detach().float().cpu().numpy()
                highj = (acc > 0.01) & (jmax > 1.0)
                labels[split].setdefault(view_id, {})["M1_HIGH_J"] = highj
    finally:
        _release(final_m1)

    final_bnd = None
    try:
        final_bnd = _load_run(repo, "BND-K1", 15000)
        manifest["final_bnd_actual_step"] = final_bnd.loaded_step
        for split in SPLITS:
            for _idx, view_id, camera, batch in _records(final_bnd.pipeline, split):
                with torch.no_grad():
                    out = final_bnd.pipeline.model.get_outputs_for_camera(camera.to(final_bnd.pipeline.model.device))
                    gt, pred = _gt_pred(final_bnd.pipeline.model, out, batch)
                err = (pred - gt).square().mean(dim=-1).detach().float().cpu().numpy()
                sup = (_scalar_map(out["accumulation"], "mean").detach().float().cpu().numpy() > 0.01)
                labels[split].setdefault(view_id, {})["final_bnd_err"] = err
                labels[split].setdefault(view_id, {})["BND_FINAL_SUPPORT"] = sup
        manifest["late_steps_requested"] = list(LATE_LABEL_STEPS)
    finally:
        _release(final_bnd)
    loaded_late = []
    for nominal in LATE_LABEL_STEPS:
        actual = _actual_step(REPO_ROOT / K1_CONFIG, nominal)
        if actual is None:
            continue
        loaded_late.append(actual)
        loaded = None
        try:
            loaded = _load_run(repo, "BND-K1", nominal)
            for split in SPLITS:
                for _idx, view_id, camera, batch in _records(loaded.pipeline, split):
                    if view_id not in labels[split]:
                        continue
                    with torch.no_grad():
                        out = loaded.pipeline.model.get_outputs_for_camera(camera.to(loaded.pipeline.model.device))
                        gt, pred = _gt_pred(loaded.pipeline.model, out, batch)
                    err = (pred - gt).square().mean(dim=-1).detach().float().cpu().numpy()
                    domain = labels[split][view_id]["BND_FINAL_SUPPORT"] & np.isfinite(err)
                    top = _top_mask_np(err, domain, 0.10)
                    key = "_persistent_count"
                    if key not in labels[split][view_id]:
                        labels[split][view_id][key] = np.zeros_like(err, dtype=np.uint8)
                    labels[split][view_id][key] += top.astype(np.uint8)
        finally:
            _release(loaded)
    for split in SPLITS:
        for view_id, data in labels[split].items():
            persistent = data.get("_persistent_count", np.zeros_like(data["M1_HIGH_J"], dtype=np.uint8)) >= 3
            data["PERSISTENT_BND_HARD"] = persistent
            data["BND_HARD_CORE"] = persistent & data["M1_HIGH_J"]
            data.pop("_persistent_count", None)
            data.pop("final_bnd_err", None)
            data.pop("BND_FINAL_SUPPORT", None)
    manifest["late_steps_loaded"] = loaded_late
    manifest["persistent_required_count"] = 3
    return labels, manifest


def _alignment_rows(
    bnd_views: Mapping[str, Mapping[str, np.ndarray]],
    m1_views: Mapping[str, Mapping[str, np.ndarray]],
    nominal_step: int,
    actual_step: int,
    split: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    signals = ("pixel_rz_rel_mean", "pixel_opacity_sum", "pixel_joint_depth_opacity_sum", "pixel_joint_depth_opacity_footprint_sum")
    pieces: Dict[str, List[np.ndarray]] = {signal: [] for signal in signals}
    delta_pieces: List[np.ndarray] = []
    positive_pieces: List[np.ndarray] = []
    valid_pieces: List[np.ndarray] = []
    for view_id, bnd in bnd_views.items():
        if view_id not in m1_views:
            continue
        valid = bnd["valid"] & m1_views[view_id]["valid"] & np.isfinite(bnd["err"]) & np.isfinite(m1_views[view_id]["err"])
        delta = bnd["err"] - m1_views[view_id]["err"]
        delta_pieces.append(delta[valid].reshape(-1))
        positive_pieces.append(np.maximum(delta[valid], 0.0).reshape(-1))
        valid_pieces.append(valid.reshape(-1))
        for signal in signals:
            pieces[signal].append(bnd[signal][valid].reshape(-1))
    if not delta_pieces:
        return rows
    delta_all = np.concatenate(delta_pieces)
    pos_all = np.concatenate(positive_pieces)
    base_pos_frac = float((delta_all > 0.0).mean())
    total_pos = float(pos_all.sum())
    for signal in signals:
        vals = np.concatenate(pieces[signal])
        sp, n_used = _spearman_np(vals, pos_all)
        rows.append(
            {
                "nominal_step": nominal_step,
                "actual_step": actual_step,
                "split": split,
                "signal": signal,
                "metric": "Spearman(signal,positive_delta_e_BND)",
                "spearman": sp,
                "n": int(vals.size),
                "sample_n": n_used,
                "base_P_delta_e_BND_gt_0": base_pos_frac,
            }
        )
        order = np.argsort(vals)
        n = vals.size
        for frac in TOP_FRACTIONS:
            k = max(1, int(math.ceil(frac * n)))
            selected = order[-k:]
            pos_share = float(pos_all[selected].sum() / max(total_pos, EPS))
            p_pos = float((delta_all[selected] > 0.0).mean())
            rows.append(
                {
                    "nominal_step": nominal_step,
                    "actual_step": actual_step,
                    "split": split,
                    "signal": signal,
                    "metric": "top_enrichment",
                    "top_fraction": frac,
                    "pixel_fraction": float(k / max(n, 1)),
                    "mean_delta_e_BND": float(delta_all[selected].mean()),
                    "median_delta_e_BND": float(np.median(delta_all[selected])),
                    "P_delta_e_BND_gt_0": p_pos,
                    "positive_regression_enrichment": p_pos / max(base_pos_frac, EPS),
                    "share_total_positive_BND_excess_MSE": pos_share,
                    "positive_excess_error_concentration": pos_share / max(k / max(n, 1), EPS),
                    "mean_signal": float(vals[selected].mean()),
                }
            )
    pos_mask = delta_all > 0.0
    for signal in signals:
        vals = np.concatenate(pieces[signal])
        rows.append(
            {
                "nominal_step": nominal_step,
                "actual_step": actual_step,
                "split": split,
                "signal": signal,
                "metric": "population_contrast",
                "positive_delta_mean": float(vals[pos_mask].mean()) if pos_mask.any() else float("nan"),
                "nonpositive_delta_mean": float(vals[~pos_mask].mean()) if (~pos_mask).any() else float("nan"),
                "positive_delta_median": float(np.median(vals[pos_mask])) if pos_mask.any() else float("nan"),
                "nonpositive_delta_median": float(np.median(vals[~pos_mask])) if (~pos_mask).any() else float("nan"),
            }
        )
    return rows


def _hard_region_rows(
    view_maps: Mapping[str, Mapping[str, np.ndarray]],
    labels: Mapping[str, Mapping[str, np.ndarray]],
    run: str,
    nominal_step: int,
    actual_step: int,
    split: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    signals = ("pixel_rz_rel_mean", "pixel_opacity_sum", "pixel_joint_depth_opacity_sum", "pixel_joint_depth_opacity_footprint_sum")
    valid_total = int(sum(int(view["valid"].sum()) for view in view_maps.values()))
    top_masks: Dict[Tuple[str, float], Dict[str, np.ndarray]] = {}
    for signal in signals:
        for frac in TOP_FRACTIONS:
            top_masks[(signal, frac)] = {view_id: _top_mask_np(view[signal], view["valid"], frac) for view_id, view in view_maps.items()}
    global_means = {}
    for signal in signals:
        vals = [view[signal][view["valid"]].reshape(-1) for view in view_maps.values()]
        all_vals = np.concatenate(vals) if vals else np.asarray([], dtype=np.float32)
        global_means[signal] = float(all_vals.mean()) if all_vals.size else float("nan")
    for label in LABELS:
        for signal in signals:
            sig_vals = []
            pixels = 0
            overlaps = {frac: 0 for frac in TOP_FRACTIONS}
            for view_id, view in view_maps.items():
                if view_id not in labels:
                    continue
                region = labels[view_id][label] & view["valid"]
                pixels += int(region.sum())
                if int(region.sum()) == 0:
                    continue
                sig_vals.append(view[signal][region].reshape(-1))
                for frac in TOP_FRACTIONS:
                    overlaps[frac] += int((region & top_masks[(signal, frac)][view_id]).sum())
            arr = np.concatenate(sig_vals) if sig_vals else np.asarray([], dtype=np.float32)
            row = {
                "run": run,
                "nominal_step": nominal_step,
                "actual_step": actual_step,
                "split": split,
                "label": label,
                "signal": signal,
                "pixels": pixels,
                "valid_pixels": valid_total,
                "coverage": pixels / max(valid_total, 1),
                "signal_mean": float(arr.mean()) if arr.size else float("nan"),
                "signal_median": float(np.median(arr)) if arr.size else float("nan"),
                "signal_p90": float(np.quantile(arr, 0.90)) if arr.size else float("nan"),
                "signal_enrichment_vs_all_valid": float(arr.mean()) / max(global_means.get(signal, float("nan")), EPS) if arr.size else float("nan"),
            }
            for frac in TOP_FRACTIONS:
                row[f"fraction_region_inside_{signal}_top_{int(frac * 100)}"] = overlaps[frac] / max(pixels, 1)
            rows.append(row)
    return rows


def _m1_bnd_delta_rows(distribution_rows: Sequence[Mapping[str, Any]], joint_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    keyed = {}
    for row in distribution_rows:
        keyed[(row["nominal_step"], row["split"], row["run"])] = row
    for (step, split, run), bnd in keyed.items():
        if run != "BND-K1":
            continue
        m1 = keyed.get((step, split, "M1"))
        if not m1:
            continue
        for key in ("RZ_REL_mean", "RZ_REL_p90", "RZ_REL_p99", "opacity_mean", "joint_mean", "depth_inconsistent_fraction_RZ_REL_gt_1p0"):
            rows.append(
                {
                    "nominal_step": step,
                    "split": split,
                    "quantity": key,
                    "M1": m1.get(key),
                    "BND_K1": bnd.get(key),
                    "BND_minus_M1": float(bnd.get(key, float("nan"))) - float(m1.get(key, float("nan"))),
                }
            )
    return rows


def _recover_cdepth_semantics(output_dir: Path) -> Dict[str, Any]:
    source_path = Path("outputs/bnd_cdepth_panama_20260811/seafree_cdepth_source_audit.json")
    semantic_path = Path("outputs/bnd_cdepth_panama_20260811/depth_semantics_alignment.json")
    summary_path = Path("outputs/bnd_cdepth_panama_20260811/bnd_cdepth_final_summary.json")
    source = json.loads(source_path.read_text()) if source_path.exists() else {}
    semantics = json.loads(semantic_path.read_text()) if semantic_path.exists() else {}
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    recovered = {
        "CDEPTH": {
            "supervision_source": source.get("pseudo_depth_input", "NOT_RECOVERED"),
            "pseudo_depth_source": "undistorted_data/undistorted_Panama/depthAnything_u16",
            "target_variable": "outputs['depth'] transformed to approximate rendered disparity",
            "loss_definition": source.get("loss_formula", "0.1 * (1 - pearson_corrcoef(pseudo_depth, 1/(10*depth+1)))"),
            "gradient_pathway": source.get("gradient_target", "rendered_depth branch; geometry/opacity paths according to autograd"),
            "active_schedule": source.get("activation_period", "enabled whenever coarse_depth_supervision_enabled is true; no step cutoff in source"),
            "formal_conclusion": summary.get("Hypothesis", "NOT_RECOVERED"),
            "known_outcome": {
                "CDEPTH_PSNR_GAIN": summary.get("CDEPTH_PSNR_GAIN"),
                "HIGH_J_MSE_GAP_RECOVERY": summary.get("HIGH_J_MSE_GAP_RECOVERY"),
                "RGB_SAFETY": summary.get("RGB_SAFETY"),
            },
            "source_audit": source,
            "depth_semantics_alignment": semantics,
        },
        "future_depth_aware_alpha_difference": {
            "supervision_source": "No new pseudo-depth supervision is implemented in this audit; a future alpha mechanism would condition opacity modulation/pruning on depth/view diagnostics rather than fitting pseudo-depth disparity.",
            "geometry_semantics": "CDEPTH optimizes rendered-depth agreement with a coarse pseudo-depth cue; depth-aware alpha would act on opacity/topology of floating-artifact candidates and would not make pseudo-depth ground truth.",
            "gradient_pathway": "CDEPTH gradients enter through outputs['depth']; a depth-aware alpha-only intervention would route through opacity modulation/pruning, not through an added rendered-depth residual loss.",
        },
    }
    _write_json(output_dir / "historical_cdepth_semantics.json", recovered)
    return recovered


def _checkpoint_availability(repo: Path, output_dir: Path) -> Tuple[List[int], Dict[str, Any]]:
    availability: Dict[str, Any] = {"requested_steps": list(REQUESTED_STEPS), "runs": {}}
    common: Optional[set] = None
    for run in RUNS:
        config_rel, _parameterization, _rasterize = _run_spec(run)
        config_path = repo / config_rel
        available = sorted(_available_steps(config_path))
        requested_to_actual = {str(step): _actual_step(config_path, step) for step in REQUESTED_STEPS}
        actual_set = {actual for actual in requested_to_actual.values() if actual is not None}
        availability["runs"][run] = {
            "config_path": str(config_path),
            "available_steps": available,
            "requested_to_actual_step": requested_to_actual,
        }
        common = actual_set if common is None else common & actual_set
    matched = sorted(common or set())
    availability["matched_actual_steps"] = matched
    availability["matched_nominal_steps"] = [15000 if step == 14999 else step for step in matched]
    _write_json(output_dir / "checkpoint_availability.json", availability)
    return availability["matched_nominal_steps"], availability


def _classification(
    distribution_rows: Sequence[Mapping[str, Any]],
    alignment_rows: Sequence[Mapping[str, Any]],
    hard_rows: Sequence[Mapping[str, Any]],
    medium_rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    bnd_train = [row for row in distribution_rows if row.get("run") == "BND-K1" and row.get("split") == "train"]
    early_bnd = [row for row in bnd_train if int(row.get("nominal_step", 0)) in (5000, 8000, 10000)]
    max_frac_rz1 = max((float(row.get("depth_inconsistent_fraction_RZ_REL_gt_1p0", 0.0)) for row in bnd_train), default=0.0)
    early_frac_rz1 = max((float(row.get("depth_inconsistent_fraction_RZ_REL_gt_1p0", 0.0)) for row in early_bnd), default=0.0)
    mean_opacity = max((float(row.get("opacity_mean", 0.0)) for row in bnd_train), default=0.0)
    reg_rows = [
        row
        for row in alignment_rows
        if row.get("split") == "train"
        and row.get("signal") == "pixel_joint_depth_opacity_sum"
        and row.get("metric") == "Spearman(signal,positive_delta_e_BND)"
    ]
    max_reg_spearman = max((float(row.get("spearman", 0.0)) for row in reg_rows if math.isfinite(float(row.get("spearman", float("nan"))))), default=float("nan"))
    enrich_rows = [
        row
        for row in alignment_rows
        if row.get("split") == "train"
        and row.get("signal") == "pixel_joint_depth_opacity_sum"
        and row.get("metric") == "top_enrichment"
        and abs(float(row.get("top_fraction", 0.0)) - 0.10) < 1e-9
    ]
    max_reg_enrich = max((float(row.get("positive_regression_enrichment", 0.0)) for row in enrich_rows), default=0.0)
    hard_joint = [
        row
        for row in hard_rows
        if row.get("run") == "BND-K1"
        and row.get("split") == "train"
        and row.get("signal") == "pixel_joint_depth_opacity_sum"
    ]
    max_hard_enrich = max((float(row.get("signal_enrichment_vs_all_valid", 0.0)) for row in hard_joint), default=0.0)
    medium_joint = [
        row
        for row in medium_rows
        if row.get("run") == "BND-K1"
        and row.get("split") == "train"
        and row.get("scope") == "projected_gaussian_pairs"
        and row.get("signal") == "joint_depth_opacity"
        and row.get("target") in ("tau", "transmission")
    ]
    max_medium_abs = max((abs(float(row.get("spearman", 0.0))) for row in medium_joint if math.isfinite(float(row.get("spearman", float("nan"))))), default=0.0)
    bnd_delta_joint = [
        row
        for row in delta_rows
        if row.get("split") == "train" and row.get("quantity") in ("joint_mean", "depth_inconsistent_fraction_RZ_REL_gt_1p0")
    ]
    bnd_stronger_count = sum(float(row.get("BND_minus_M1", 0.0)) > 0.0 for row in bnd_delta_joint)

    depth_population = max_frac_rz1 > 0.01
    opacity_non_negligible = mean_opacity > 0.01
    regression_alignment = (math.isfinite(max_reg_spearman) and max_reg_spearman > 0.05) or max_reg_enrich > 1.10
    hard_alignment = max_hard_enrich > 1.25
    early_visible = early_frac_rz1 > 0.01
    medium_assoc = max_medium_abs > 0.10
    bnd_relevant = bnd_stronger_count > 0

    if depth_population and opacity_non_negligible and regression_alignment and hard_alignment and early_visible and medium_assoc and bnd_relevant:
        label = "DEPTH_OPACITY_READY"
    elif depth_population and opacity_non_negligible and (regression_alignment or hard_alignment) and early_visible:
        label = "DEPTH_OPACITY_WEAK"
    else:
        label = "DEPTH_OPACITY_NOT_SUPPORTED"
    next_step = "BND + DEPTH-AWARE-ALPHA-ONLY" if label == "DEPTH_OPACITY_READY" else "DO NOT train depth-aware alpha; close this alpha mechanism line"
    return {
        "classification": label,
        "depth_population_RZ_REL_gt_1": depth_population,
        "max_BND_train_fraction_RZ_REL_gt_1": max_frac_rz1,
        "early_BND_train_fraction_RZ_REL_gt_1": early_frac_rz1,
        "opacity_non_negligible": opacity_non_negligible,
        "max_BND_train_opacity_mean": mean_opacity,
        "regression_alignment": regression_alignment,
        "max_regression_spearman_joint": max_reg_spearman,
        "max_regression_top10_enrichment_joint": max_reg_enrich,
        "hard_alignment": hard_alignment,
        "max_hard_joint_enrichment": max_hard_enrich,
        "early_visible": early_visible,
        "medium_association": medium_assoc,
        "max_abs_medium_spearman_joint": max_medium_abs,
        "BND_relevance_distributional": bnd_relevant,
        "BND_stronger_count": bnd_stronger_count,
        "next_single_experiment": next_step,
    }


def _write_research_note(
    repo_manifest: Mapping[str, Any],
    env: Mapping[str, Any],
    cdepth: Mapping[str, Any],
    availability: Mapping[str, Any],
    contribution: Mapping[str, Any],
    distribution_rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
    alignment_rows: Sequence[Mapping[str, Any]],
    hard_rows: Sequence[Mapping[str, Any]],
    medium_rows: Sequence[Mapping[str, Any]],
    temporal_rows: Sequence[Mapping[str, Any]],
    decomp_rows: Sequence[Mapping[str, Any]],
    classification: Mapping[str, Any],
) -> None:
    def first(rows: Sequence[Mapping[str, Any]], **kwargs: Any) -> Mapping[str, Any]:
        for row in rows:
            if all(row.get(k) == v for k, v in kwargs.items()):
                return row
        return {}

    bnd_final = first(distribution_rows, run="BND-K1", split="train", nominal_step=15000)
    bnd_5k = first(distribution_rows, run="BND-K1", split="train", nominal_step=5000)
    m1_final = first(distribution_rows, run="M1", split="train", nominal_step=15000)
    reg_final = first(
        alignment_rows,
        split="train",
        nominal_step=15000,
        signal="pixel_joint_depth_opacity_sum",
        metric="Spearman(signal,positive_delta_e_BND)",
    )
    med_final_tau = first(
        medium_rows,
        run="BND-K1",
        split="train",
        nominal_step=15000,
        scope="projected_gaussian_pairs",
        signal="joint_depth_opacity",
        target="tau",
    )
    med_final_t = first(
        medium_rows,
        run="BND-K1",
        split="train",
        nominal_step=15000,
        scope="projected_gaussian_pairs",
        signal="joint_depth_opacity",
        target="transmission",
    )
    decomp_final = first(decomp_rows, run="BND-K1", split="train", nominal_step=15000)

    lines = [
        "# BND Depth-Opacity Audit - 2026-08-16",
        "",
        "## Scope",
        "CONFIG FACT: This is a read-only, zero-training mechanism audit. No optimizer step, checkpoint write, CDEPTH/OMVC/CB-FG/BAP training, depth residual loss, synthetic epipolar depth, or depth-aware alpha implementation is performed.",
        "",
        "## Repo",
        f"EXPERIMENTAL FACT: Branch `{repo_manifest.get('branch')}`, HEAD `{repo_manifest.get('head')}`.",
        "",
        "## Environment",
        f"EXPERIMENTAL FACT: `CONDA_ENV={env.get('CONDA_ENV')}`, `PYTHON_PATH={env.get('PYTHON_PATH')}`, `TORCH_VERSION={env.get('TORCH_VERSION')}`.",
        f"EXPERIMENTAL FACT: `CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}` maps torch logical `cuda:0` to physical GPU `{env.get('CUDA_VISIBLE_DEVICES')}`.",
        "",
        "## Historical CDEPTH Semantics",
        f"CODE FACT: CDEPTH supervision source `{cdepth['CDEPTH'].get('supervision_source')}`.",
        f"CODE FACT: Loss `{cdepth['CDEPTH'].get('loss_definition')}`.",
        f"CODE FACT: Gradient pathway `{cdepth['CDEPTH'].get('gradient_pathway')}`.",
        f"INFERENCE: Formal CDEPTH conclusion `{cdepth['CDEPTH'].get('formal_conclusion')}`.",
        "INFERENCE: A future depth-aware-alpha-only mechanism would differ from CDEPTH by operating through opacity modulation/pruning rather than a pseudo-depth rendered-depth loss.",
        "",
        "## Available Matched Checkpoints",
        f"EXPERIMENTAL FACT: Matched nominal checkpoints used `{availability.get('matched_nominal_steps')}`; requested-to-actual map stored in output manifest.",
        "",
        "## Depth-Inconsistency Definition",
        "CODE FACT: `RZ_ABS(i,v)=abs(D_v(x_i,v)-z_i,v)` and `RZ_REL=RZ_ABS/(D_v(x_i,v)+epsilon)` with `epsilon=1e-6`.",
        "CODE FACT: `D_v` is normal rendered alpha-blended expected depth; this is pseudo/structure consistency evidence, not ground-truth geometry.",
        "",
        "## Opacity / Contribution Availability",
        f"CODE FACT: `{contribution.get('PER_GAUSSIAN_CONTRIBUTION')}`; safe proxies are `{contribution.get('safe_proxy')}`.",
        "",
        "## M1 vs BND Depth-Opacity Distributions",
        f"QUANTITATIVE RESULT: Final train M1 RZ_REL mean `{m1_final.get('RZ_REL_mean')}`, fraction RZ_REL>1 `{m1_final.get('depth_inconsistent_fraction_RZ_REL_gt_1p0')}`.",
        f"QUANTITATIVE RESULT: Final train BND RZ_REL mean `{bnd_final.get('RZ_REL_mean')}`, fraction RZ_REL>1 `{bnd_final.get('depth_inconsistent_fraction_RZ_REL_gt_1p0')}`, opacity mean `{bnd_final.get('opacity_mean')}`, joint mean `{bnd_final.get('joint_mean')}`.",
        "",
        "## BND Regression Alignment",
        f"QUANTITATIVE RESULT: Final train Spearman(pixel_joint_depth_opacity_sum, positive_delta_e_BND) `{reg_final.get('spearman')}`.",
        "",
        "## Hard-Region Alignment",
    ]
    for label in LABELS:
        row = first(
            hard_rows,
            run="BND-K1",
            split="train",
            nominal_step=15000,
            label=label,
            signal="pixel_joint_depth_opacity_sum",
        )
        lines.append(
            f"QUANTITATIVE RESULT: Final train {label} joint enrichment `{row.get('signal_enrichment_vs_all_valid')}`, top10 overlap `{row.get('fraction_region_inside_pixel_joint_depth_opacity_sum_top_10')}`."
        )
    lines.extend(
        [
            "",
            "## Medium / Direction Alignment",
            f"QUANTITATIVE RESULT: Final train Spearman(joint_depth_opacity,tau) `{med_final_tau.get('spearman')}`; Spearman(joint_depth_opacity,transmission) `{med_final_t.get('spearman')}`.",
            "",
            "## Temporal Emergence",
            f"QUANTITATIVE RESULT: BND train 5k fraction RZ_REL>1 `{bnd_5k.get('depth_inconsistent_fraction_RZ_REL_gt_1p0')}`, final `{bnd_final.get('depth_inconsistent_fraction_RZ_REL_gt_1p0')}`.",
            "",
            "## Decomposition Context",
            f"QUANTITATIVE RESULT: Final train BND `J_p99={decomp_final.get('J_p99')}`, `P_J_gt_1={decomp_final.get('P_J_gt_1')}`, `tau_p90={decomp_final.get('tau_p90')}`, `tau_p99={decomp_final.get('tau_p99')}`, `P_T_lt_0p1={decomp_final.get('P_T_lt_0p1')}`, `P_c_gt_0p99={decomp_final.get('P_c_gt_0p99')}`, `P_abs_s_full_gt_5={decomp_final.get('P_abs_s_full_gt_5')}`.",
            "",
            "## Classification",
            f"INFERENCE: `{classification.get('classification')}`.",
            f"RECOMMENDATION: `{classification.get('next_single_experiment')}`.",
            "",
            "## Outputs",
            "- `outputs/bnd_depth_opacity_audit_20260816/`",
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
    contribution = {
        "PER_GAUSSIAN_CONTRIBUTION": "IMPLEMENTATION_BLOCKED",
        "reason": "UnderwaterRasterizer returns image-level rgb/depth/alpha/final transmittance but does not expose per-Gaussian compositing weights without CUDA changes.",
        "safe_proxy": "opacity alpha_i=sigmoid(opacity_logit), screen-space radius/footprint, opacity_footprint=alpha_i*pi*r^2, joint_depth_opacity=RZ_REL*alpha_i, and deterministic nearest-pixel sums.",
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)
    _write_json(output_dir / "environment_manifest.json", env)
    _write_json(output_dir / "gpu_manifest.json", gpu_manifest)
    _write_json(output_dir / "contribution_availability.json", contribution)
    _write_json(
        output_dir / "depth_inconsistency_definition.json",
        {
            "CODE FACT": {
                "RZ_ABS": "abs(outputs['depth'] sampled bilinearly at projected Gaussian center - projected Gaussian camera-space z)",
                "RZ_REL": "RZ_ABS / (sampled rendered depth + 1e-6)",
                "rendered_depth": "normal WaterSplatting alpha-blended expected depth output; not GT geometry",
                "pixel_aggregation": "nearest projected pixel; sum opacity, opacity_footprint, RZ_REL*opacity, and RZ_REL*opacity*footprint; mean RZ_REL divides by projected pair count",
            }
        },
    )

    cdepth = _recover_cdepth_semantics(output_dir)
    matched_nominal_steps, availability = _checkpoint_availability(repo, output_dir)
    labels, labels_manifest = _collect_hard_labels(repo)
    _write_json(output_dir / "hard_region_labels_manifest.json", labels_manifest)

    all_distribution_rows: List[Dict[str, Any]] = []
    all_rz_top_rows: List[Dict[str, Any]] = []
    all_joint_rows: List[Dict[str, Any]] = []
    all_medium_rows: List[Dict[str, Any]] = []
    all_camera_rows: List[Dict[str, Any]] = []
    all_decomp_rows: List[Dict[str, Any]] = []
    all_alignment_rows: List[Dict[str, Any]] = []
    all_hard_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []

    for nominal_step in matched_nominal_steps:
        step_maps: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]] = {}
        step_manifests: Dict[str, Any] = {}
        for run in RUNS:
            split_maps, manifest, rows = _collect_run_step(repo, run, nominal_step)
            step_maps[run] = split_maps
            step_manifests[run] = manifest
            checkpoint_rows.append(manifest)
            all_distribution_rows.extend(rows["distribution"])
            all_rz_top_rows.extend(rows["rz_top"])
            all_joint_rows.extend(rows["joint"])
            all_medium_rows.extend(rows["medium"])
            all_camera_rows.extend(rows["camera"])
            all_decomp_rows.extend(rows["decomposition"])
            for split in SPLITS:
                all_hard_rows.extend(_hard_region_rows(split_maps[split], labels[split], run, nominal_step, manifest["actual_step"], split))
        actual_step = step_manifests["BND-K1"]["actual_step"]
        for split in SPLITS:
            all_alignment_rows.extend(_alignment_rows(step_maps["BND-K1"][split], step_maps["M1"][split], nominal_step, actual_step, split))
        del step_maps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    delta_rows = _m1_bnd_delta_rows(all_distribution_rows, all_joint_rows)
    classification = _classification(all_distribution_rows, all_alignment_rows, all_hard_rows, all_medium_rows, delta_rows)

    # Temporal emergence is a compact join of the main train diagnostics.
    temporal_rows: List[Dict[str, Any]] = []
    for row in all_distribution_rows:
        if row.get("split") != "train":
            continue
        run = row["run"]
        step = row["nominal_step"]
        temporal = {
            "run": run,
            "nominal_step": step,
            "actual_step": row["actual_step"],
            "depth_inconsistent_fraction_RZ_REL_gt_1p0": row.get("depth_inconsistent_fraction_RZ_REL_gt_1p0"),
            "joint_depth_opacity_mean": row.get("joint_mean"),
            "opacity_mean": row.get("opacity_mean"),
            "RZ_REL_mean": row.get("RZ_REL_mean"),
        }
        reg = next(
            (
                r
                for r in all_alignment_rows
                if r.get("split") == "train"
                and r.get("nominal_step") == step
                and r.get("signal") == "pixel_joint_depth_opacity_sum"
                and r.get("metric") == "Spearman(signal,positive_delta_e_BND)"
            ),
            {},
        )
        med = next(
            (
                r
                for r in all_medium_rows
                if r.get("split") == "train"
                and r.get("run") == run
                and r.get("nominal_step") == step
                and r.get("scope") == "projected_gaussian_pairs"
                and r.get("signal") == "joint_depth_opacity"
                and r.get("target") == "tau"
            ),
            {},
        )
        temporal["BND_regression_joint_spearman"] = reg.get("spearman") if run == "BND-K1" else ""
        temporal["joint_tau_spearman"] = med.get("spearman")
        temporal_rows.append(temporal)

    _write_csv(output_dir / "checkpoint_manifest.csv", checkpoint_rows)
    _write_json(output_dir / "checkpoint_manifest.json", {"rows": checkpoint_rows})
    _write_csv(output_dir / "depth_opacity_distribution.csv", all_distribution_rows)
    _write_json(output_dir / "depth_opacity_distribution.json", {"rows": all_distribution_rows})
    _write_csv(output_dir / "rz_top_quantiles.csv", all_rz_top_rows)
    _write_json(output_dir / "rz_top_quantiles.json", {"rows": all_rz_top_rows})
    _write_csv(output_dir / "joint_depth_opacity_quantiles.csv", all_joint_rows)
    _write_json(output_dir / "joint_depth_opacity_quantiles.json", {"rows": all_joint_rows})
    _write_csv(output_dir / "m1_bnd_distribution_delta.csv", delta_rows)
    _write_json(output_dir / "m1_bnd_distribution_delta.json", {"rows": delta_rows})
    _write_csv(output_dir / "bnd_regression_alignment.csv", all_alignment_rows)
    _write_json(output_dir / "bnd_regression_alignment.json", {"rows": all_alignment_rows})
    _write_csv(output_dir / "hard_region_alignment.csv", all_hard_rows)
    _write_json(output_dir / "hard_region_alignment.json", {"rows": all_hard_rows})
    _write_csv(output_dir / "medium_direction_alignment.csv", all_medium_rows)
    _write_json(output_dir / "medium_direction_alignment.json", {"rows": all_medium_rows})
    _write_csv(output_dir / "per_camera_direction_summary.csv", all_camera_rows)
    _write_json(output_dir / "per_camera_direction_summary.json", {"rows": all_camera_rows})
    _write_csv(output_dir / "temporal_emergence.csv", temporal_rows)
    _write_json(output_dir / "temporal_emergence.json", {"rows": temporal_rows})
    _write_csv(output_dir / "decomposition_context.csv", all_decomp_rows)
    _write_json(output_dir / "decomposition_context.json", {"rows": all_decomp_rows})
    final_summary = {
        "repo": repo_manifest,
        "environment": env,
        "checkpoint_availability": availability,
        "historical_cdepth": {
            "formal_conclusion": cdepth["CDEPTH"].get("formal_conclusion"),
            "known_outcome": cdepth["CDEPTH"].get("known_outcome"),
        },
        "contribution_availability": contribution,
        "classification": classification,
        "outputs": sorted(str(path.relative_to(repo)) for path in output_dir.glob("*")),
    }
    _write_json(output_dir / "final_summary.json", final_summary)
    _write_research_note(
        repo_manifest,
        env,
        cdepth,
        availability,
        contribution,
        all_distribution_rows,
        delta_rows,
        all_alignment_rows,
        all_hard_rows,
        all_medium_rows,
        temporal_rows,
        all_decomp_rows,
        classification,
    )


if __name__ == "__main__":
    main()
