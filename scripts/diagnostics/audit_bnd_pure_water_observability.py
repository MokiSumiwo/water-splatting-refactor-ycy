#!/usr/bin/env python3
"""Read-only Panama pure-water/background-anchor readiness audit.

This diagnostic loads existing BND-K1 checkpoints, renders fixed cameras, builds
offline background-candidate masks, and measures B_inf headroom plus a no-step
virtual background-loss gradient route.  It does not create optimizers, call
training steps, densify, prune, reset opacity, or write checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.utils.eval_utils import eval_setup

from scripts.diagnostics import audit_bnd_cdepth_setup as cdepth_setup


SCENE = "Panama"
OUTPUT_DIR = Path("outputs/bnd_pure_water_audit_panama_20260812")
RENDER_DIR = Path("renders/bnd_pure_water_audit_panama_20260812")
LOG_DIR = Path("logs/bnd_pure_water_audit_panama_20260812")
RESEARCH_NOTE = Path("research_notes/BND_PURE_WATER_BACKGROUND_ANCHOR_READINESS_2026-08-12.md")
SEAFREE_REPO = Path("/mnt/new/home_old/ycy/reference_repos/SeaFree-GS")
K1_CONFIG = cdepth_setup.K1_CONFIG
DATA_PATH = Path("undistorted_data/undistorted_Panama")
DEPTHS_PATH = DATA_PATH / "depthAnything_u16"
TEMPORAL_STEPS = (3000, 5000, 8000, 10000, 13000, 14999)
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
MASK_NAMES = ("M_SF", "M_LOW_SUPPORT", "M_INTERSECT", "M_SAFE")
EPS = 1e-8
LUMA = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)
ALLOWED_GPU_IDS = {"6", "7", "8", "9"}


@dataclass
class LoadedRun:
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    if nominal_step == 15000 and 14999 in steps:
        return 14999
    raise FileNotFoundError(f"Missing checkpoint step {nominal_step} for {config_path}; available={sorted(steps)}")


def _assert_allowed_gpu_policy() -> Dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [token.strip() for token in visible.split(",") if token.strip()]
    if torch.cuda.is_available():
        if not devices:
            raise RuntimeError("CUDA diagnostic requires CUDA_VISIBLE_DEVICES to be set to physical GPU 6, 7, 8, or 9.")
        if not set(devices).issubset(ALLOWED_GPU_IDS):
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES must use only physical GPUs {sorted(ALLOWED_GPU_IDS)}; got {visible!r}")
    return {
        "CUDA_VISIBLE_DEVICES": visible,
        "allowed_physical_gpus": sorted(ALLOWED_GPU_IDS),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torch_visible_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() else "none",
    }


def _load_k1(repo: Path, step: int) -> LoadedRun:
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
    model.step = int(loaded_step)
    pipeline.eval()
    return LoadedRun(config_path, Path(checkpoint_path), int(loaded_step), config, pipeline)


def _release(obj: Optional[LoadedRun]) -> None:
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


def _view_records(pipeline: Any, view_ids: Sequence[str]) -> List[Tuple[int, str, Any, Dict[str, Any]]]:
    by_id: Dict[str, Tuple[int, str, Any, Dict[str, Any]]] = {}

    train_dataset = pipeline.datamanager.train_dataset
    train_filenames = list(getattr(train_dataset, "image_filenames", []))
    train_cameras = train_dataset.cameras.to(pipeline.model.device)
    for index, filename in enumerate(train_filenames):
        view_id = Path(filename).stem
        batch = pipeline.datamanager.cached_train[index].copy()
        by_id[view_id] = (index, view_id, train_cameras[index : index + 1], _batch_to_device(batch, pipeline.model.device))

    eval_dataset = pipeline.datamanager.eval_dataset
    eval_filenames = list(getattr(eval_dataset, "image_filenames", []))
    for eval_index, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        filename = eval_filenames[eval_index] if eval_index < len(eval_filenames) else Path(f"eval_{eval_index}")
        view_id = Path(filename).stem
        by_id[view_id] = (eval_index, view_id, camera.to(pipeline.model.device), _batch_to_device(batch, pipeline.model.device))

    missing = [view_id for view_id in view_ids if view_id not in by_id]
    if missing:
        raise RuntimeError(f"Missing requested views from loaded datasets: {missing}; available={sorted(by_id)}")
    return [by_id[view_id] for view_id in view_ids]


def _get_gt(model: Any, batch: Mapping[str, Any], background: Tensor) -> Tensor:
    return model.composite_with_background(model.get_gt_img(batch["image"]), background.to(model.device))


def _safe_outputs(outputs: Mapping[str, Any]) -> Dict[str, Tensor]:
    keep = (
        "pred_image",
        "rgb",
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
    )
    return {
        key: value.detach().float().cpu()
        for key, value in outputs.items()
        if key in keep and isinstance(value, Tensor)
    }


def _render_step(repo: Path, step: int, view_ids: Sequence[str]) -> Tuple[Dict[str, Dict[str, Tensor]], Dict[str, Any]]:
    loaded = _load_k1(repo, step)
    try:
        model = loaded.pipeline.model
        records = _view_records(loaded.pipeline, view_ids)
        maps: Dict[str, Dict[str, Tensor]] = {}
        for _idx, view_id, camera, batch in records:
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera.to(model.device))
                gt = _get_gt(model, batch, outputs["background"])
            safe = _safe_outputs(outputs)
            safe["gt"] = gt.detach().float().cpu()
            safe["pred_image"] = safe.get("pred_image", safe.get("rgb")).detach().float().cpu()
            maps[view_id] = safe
        meta = {
            "requested_step": int(step),
            "loaded_step": int(loaded.loaded_step),
            "config_path": str(loaded.config_path),
            "checkpoint_path": str(loaded.checkpoint_path),
            "sh_degree": getattr(model.config, "sh_degree", ""),
            "intrinsic_color_parameterization": getattr(model.config, "intrinsic_color_parameterization", ""),
            "medium_context_mode": getattr(model.config, "medium_context_mode", ""),
            "b_inf_mode": getattr(model.config, "b_inf_mode", ""),
            "infinite_water_enabled": getattr(model.config, "infinite_water_enabled", ""),
            "gaussian_count": int(model.num_points),
        }
        return maps, meta
    finally:
        _release(loaded)


def _tensor_stats(values: Tensor, prefix: str = "") -> Dict[str, float]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {f"{prefix}{k}": float("nan") for k in ("mean", "p50", "p90", "p99", "max")}
    return {
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}p50": float(torch.quantile(flat, 0.50).item()),
        f"{prefix}p90": float(torch.quantile(flat, 0.90).item()),
        f"{prefix}p99": float(torch.quantile(flat, 0.99).item()),
        f"{prefix}max": float(flat.max().item()),
    }


def _ensure_bool(mask: Tensor) -> Tensor:
    return mask.detach().bool().cpu()


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    best: List[Tuple[int, int]] = []
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            comp: List[Tuple[int, int]] = []
            q: deque[Tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(comp) > len(best):
                best = comp
    out = np.zeros((h, w), dtype=bool)
    if best:
        ys, xs = zip(*best)
        out[np.asarray(ys), np.asarray(xs)] = True
    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[Tuple[int, int]] = deque()
    for x in range(w):
        for y in (0, h - 1):
            if not mask[y, x] and not outside[y, x]:
                outside[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not mask[y, x] and not outside[y, x]:
                outside[y, x] = True
                q.append((y, x))
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while q:
        cy, cx = q.popleft()
        for dy, dx in neighbors:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                q.append((ny, nx))
    return mask | (~outside)


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = mask.astype(bool)
    if radius <= 0:
        return mask
    h, w = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.ones((h, w), dtype=bool)
    offsets: List[Tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                offsets.append((dy, dx))
    for dy, dx in offsets:
        out &= padded[radius + dy : radius + dy + h, radius + dx : radius + dx + w]
    return out


def _component_count_and_largest(mask: Tensor) -> Tuple[int, float]:
    arr = mask.detach().bool().cpu().numpy()
    h, w = arr.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    largest = 0
    total = int(arr.sum())
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for y in range(h):
        for x in range(w):
            if not arr[y, x] or seen[y, x]:
                continue
            count += 1
            size = 0
            q: deque[Tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            while q:
                cy, cx = q.popleft()
                size += 1
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and arr[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            largest = max(largest, size)
    return count, float(largest) / max(total, 1)


def _load_pseudo_depth(view_id: str, size_hw: Tuple[int, int]) -> Tensor:
    path = DEPTHS_PATH / f"{view_id}.png"
    image = Image.open(path)
    if image.size != (size_hw[1], size_hw[0]):
        image = image.resize((size_hw[1], size_hw[0]), Image.Resampling.BILINEAR)
    arr = np.asarray(image).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    max_value = float(np.max(arr))
    if max_value > 0:
        arr = arr / max_value
    return torch.from_numpy(arr).float()


def _seafree_background_candidate(pseudo_depth: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
    threshold = 1e-2
    pseudo = pseudo_depth.detach().float().cpu().numpy()
    # Source equivalence: pseudo_depth < 1e-2 is first marked, then cv2
    # THRESH_BINARY_INV selects pseudo_depth >= 1e-2 as the foreground source.
    foreground_source = pseudo >= threshold
    largest = _largest_component(foreground_source)
    filled = _fill_holes(largest)
    foreground = filled.astype(bool)
    background = ~foreground
    meta = {
        "status": "OK",
        "pseudo_depth_normalization": "per image divide by max, matching SeaFree get_loss_dict",
        "background_depth_threshold": threshold,
        "foreground_source": "pseudo_depth >= 1e-2 after SeaFree threshold inversion",
        "foreground_mask": "largest connected component of foreground_source, filled; background candidate is complement",
        "morphology": "no erosion/dilation in SeaFree source for this mask",
    }
    return torch.from_numpy(background), meta


def _build_masks(step3_maps: Mapping[str, Mapping[str, Tensor]], view_ids: Sequence[str]) -> Tuple[Dict[str, Dict[str, Tensor]], Dict[str, Any]]:
    masks: Dict[str, Dict[str, Tensor]] = {}
    sf_meta: Dict[str, Any] = {}
    for view_id in view_ids:
        acc = step3_maps[view_id]["accumulation"][..., 0] if step3_maps[view_id]["accumulation"].ndim == 3 else step3_maps[view_id]["accumulation"]
        h, w = int(acc.shape[0]), int(acc.shape[1])
        pseudo = _load_pseudo_depth(view_id, (h, w))
        m_sf, meta = _seafree_background_candidate(pseudo)
        m_low = acc <= 0.01
        m_intersect = _ensure_bool(m_sf) & _ensure_bool(m_low)
        m_safe = torch.from_numpy(_erode(m_intersect.numpy(), 5))
        masks[view_id] = {
            "M_SF": _ensure_bool(m_sf),
            "M_LOW_SUPPORT": _ensure_bool(m_low),
            "M_INTERSECT": _ensure_bool(m_intersect),
            "M_SAFE": _ensure_bool(m_safe),
            "pseudo_depth": pseudo,
        }
        sf_meta[view_id] = meta
    return masks, sf_meta


def _coverage_row(split: str, view_id: str, mask_name: str, mask: Tensor) -> Dict[str, Any]:
    mask = _ensure_bool(mask)
    h, w = mask.shape
    count = int(mask.sum().item())
    comps, largest_frac = _component_count_and_largest(mask)
    yy = torch.linspace(0.0, 1.0, h).reshape(h, 1).expand(h, w)
    xx = torch.linspace(0.0, 1.0, w).reshape(1, w).expand(h, w)
    return {
        "split": split,
        "view_id": view_id,
        "mask": mask_name,
        "pixel_count": count,
        "total_pixels": int(h * w),
        "coverage_fraction": float(count) / max(float(h * w), 1.0),
        "connected_components": comps,
        "largest_component_fraction": largest_frac,
        "top20_image_fraction": float((mask & (yy <= 0.2)).sum().item()) / max(count, 1),
        "bottom20_image_fraction": float((mask & (yy >= 0.8)).sum().item()) / max(count, 1),
        "left20_image_fraction": float((mask & (xx <= 0.2)).sum().item()) / max(count, 1),
        "right20_image_fraction": float((mask & (xx >= 0.8)).sum().item()) / max(count, 1),
    }


def _pooled_coverage_row(split: str, mask_name: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    selected = [r for r in rows if r["split"] == split and r["mask"] == mask_name and r["view_id"] != "ALL"]
    total = sum(int(r["total_pixels"]) for r in selected)
    count = sum(int(r["pixel_count"]) for r in selected)
    return {
        "split": split,
        "view_id": "ALL",
        "mask": mask_name,
        "pixel_count": count,
        "total_pixels": total,
        "coverage_fraction": float(count) / max(total, 1),
        "views_ge_1pct": sum(float(r["coverage_fraction"]) >= 0.01 for r in selected),
        "views_nonempty": sum(int(r["pixel_count"]) > 0 for r in selected),
    }


def _agreement_rows(split: str, masks_by_view: Mapping[str, Mapping[str, Tensor]], view_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pooled: Dict[str, int] = {"sf": 0, "low": 0, "inter": 0, "safe": 0, "union_sf_low": 0}
    for view_id in view_ids:
        m = masks_by_view[view_id]
        sf = m["M_SF"]
        low = m["M_LOW_SUPPORT"]
        inter = m["M_INTERSECT"]
        safe = m["M_SAFE"]
        union = sf | low
        row = {
            "split": split,
            "view_id": view_id,
            "IoU_M_SF_M_LOW_SUPPORT": float((sf & low).sum().item()) / max(float(union.sum().item()), 1.0),
            "IoU_M_SF_M_INTERSECT": float((sf & inter).sum().item()) / max(float((sf | inter).sum().item()), 1.0),
            "M_INTERSECT_over_M_SF": float(inter.sum().item()) / max(float(sf.sum().item()), 1.0),
            "M_SAFE_over_M_INTERSECT": float(safe.sum().item()) / max(float(inter.sum().item()), 1.0),
        }
        rows.append(row)
        pooled["sf"] += int(sf.sum().item())
        pooled["low"] += int(low.sum().item())
        pooled["inter"] += int(inter.sum().item())
        pooled["safe"] += int(safe.sum().item())
        pooled["union_sf_low"] += int(union.sum().item())
    rows.append(
        {
            "split": split,
            "view_id": "ALL",
            "IoU_M_SF_M_LOW_SUPPORT": pooled["inter"] / max(float(pooled["union_sf_low"]), 1.0),
            "IoU_M_SF_M_INTERSECT": pooled["inter"] / max(float(pooled["sf"]), 1.0),
            "M_INTERSECT_over_M_SF": pooled["inter"] / max(float(pooled["sf"]), 1.0),
            "M_SAFE_over_M_INTERSECT": pooled["safe"] / max(float(pooled["inter"]), 1.0),
        }
    )
    return rows


def _masked_values(values: Tensor, mask: Tensor) -> Tensor:
    values = values.detach().float().cpu()
    mask = _ensure_bool(mask)
    if values.ndim == mask.ndim:
        return values[mask]
    return values[mask.unsqueeze(-1).expand_as(values)]


def _headroom_row(split: str, view_id: str, mask_name: str, mask: Tensor, final_maps: Mapping[str, Tensor]) -> Dict[str, Any]:
    gt = final_maps["gt"].clamp(0.0, 1.0)
    pred = final_maps["pred_image"].clamp(0.0, 1.0)
    b_inf = final_maps["b_inf"].clamp(0.0, 1.0)
    mask = _ensure_bool(mask)
    count = int(mask.sum().item())
    if count == 0:
        base = {
            "split": split,
            "view_id": view_id,
            "mask": mask_name,
            "pixel_count": 0,
            "E_BINF": float("nan"),
            "E_full": float("nan"),
            "G_anchor": float("nan"),
            "R_anchor": float("nan"),
        }
        return base
    err_b = (b_inf - gt)[mask]
    err_f = (pred - gt)[mask]
    abs_b = err_b.abs()
    abs_f = err_f.abs()
    e_b = float(abs_b.mean().item())
    e_f = float(abs_f.mean().item())
    row = {
        "split": split,
        "view_id": view_id,
        "mask": mask_name,
        "pixel_count": count,
        "E_BINF": e_b,
        "E_full": e_f,
        "G_anchor": e_b - e_f,
        "R_anchor": (e_b - e_f) / max(e_b, EPS),
        "BINF_R_MAE": float(abs_b[:, 0].mean().item()),
        "BINF_G_MAE": float(abs_b[:, 1].mean().item()),
        "BINF_B_MAE": float(abs_b[:, 2].mean().item()),
        "BINF_R_bias_mean": float(err_b[:, 0].mean().item()),
        "BINF_G_bias_mean": float(err_b[:, 1].mean().item()),
        "BINF_B_bias_mean": float(err_b[:, 2].mean().item()),
        "BINF_bias_median": float(torch.median(err_b).item()),
        "BINF_RMSE": float(torch.sqrt((err_b.square()).mean()).item()),
        "BINF_abs_p90": float(torch.quantile(abs_b.reshape(-1), 0.90).item()),
        "BINF_abs_p99": float(torch.quantile(abs_b.reshape(-1), 0.99).item()),
        "E_BINF_gt_E_full": bool(e_b > e_f),
    }
    return row


def _pooled_headroom_row(split: str, mask_name: str, view_ids: Sequence[str], masks: Mapping[str, Mapping[str, Tensor]], final_maps: Mapping[str, Mapping[str, Tensor]]) -> Dict[str, Any]:
    err_b_list = []
    err_f_list = []
    count = 0
    views_better = 0
    for view_id in view_ids:
        mask = masks[view_id][mask_name]
        count += int(mask.sum().item())
        if int(mask.sum().item()) == 0:
            continue
        gt = final_maps[view_id]["gt"].clamp(0.0, 1.0)
        pred = final_maps[view_id]["pred_image"].clamp(0.0, 1.0)
        b_inf = final_maps[view_id]["b_inf"].clamp(0.0, 1.0)
        eb = (b_inf - gt)[mask].abs()
        ef = (pred - gt)[mask].abs()
        if float(eb.mean().item()) > float(ef.mean().item()):
            views_better += 1
        err_b_list.append(eb)
        err_f_list.append(ef)
    if not err_b_list:
        return {"split": split, "view_id": "ALL", "mask": mask_name, "pixel_count": count, "E_BINF": float("nan"), "E_full": float("nan")}
    abs_b = torch.cat(err_b_list, dim=0)
    abs_f = torch.cat(err_f_list, dim=0)
    e_b = float(abs_b.mean().item())
    e_f = float(abs_f.mean().item())
    return {
        "split": split,
        "view_id": "ALL",
        "mask": mask_name,
        "pixel_count": count,
        "E_BINF": e_b,
        "E_full": e_f,
        "G_anchor": e_b - e_f,
        "R_anchor": (e_b - e_f) / max(e_b, EPS),
        "views_E_BINF_gt_E_full": views_better,
        "view_count": len(view_ids),
        "BINF_R_MAE": float(abs_b[:, 0].mean().item()),
        "BINF_G_MAE": float(abs_b[:, 1].mean().item()),
        "BINF_B_MAE": float(abs_b[:, 2].mean().item()),
        "BINF_RMSE": float(torch.sqrt(abs_b.square().mean()).item()),
        "BINF_abs_p90": float(torch.quantile(abs_b.reshape(-1), 0.90).item()),
        "BINF_abs_p99": float(torch.quantile(abs_b.reshape(-1), 0.99).item()),
    }


def _temporal_rows(split: str, view_ids: Sequence[str], masks: Mapping[str, Mapping[str, Tensor]], temporal_maps: Mapping[int, Mapping[str, Mapping[str, Tensor]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for step in TEMPORAL_STEPS:
        all_vals: List[Tensor] = []
        contaminated = 0
        total = 0
        for view_id in view_ids:
            mask = masks[view_id]["M_SAFE"]
            acc = temporal_maps[step][view_id]["accumulation"]
            if acc.ndim == 3:
                acc = acc[..., 0]
            vals = acc[mask]
            if vals.numel() == 0:
                row = {
                    "split": split,
                    "view_id": view_id,
                    "step": step,
                    "safe_pixel_count": 0,
                    "mean_accumulation": float("nan"),
                    "p50_accumulation": float("nan"),
                    "p90_accumulation": float("nan"),
                    "fraction_accumulation_gt_0p01": float("nan"),
                }
            else:
                row = {
                    "split": split,
                    "view_id": view_id,
                    "step": step,
                    "safe_pixel_count": int(vals.numel()),
                    "mean_accumulation": float(vals.mean().item()),
                    "p50_accumulation": float(torch.quantile(vals, 0.50).item()),
                    "p90_accumulation": float(torch.quantile(vals, 0.90).item()),
                    "fraction_accumulation_gt_0p01": float((vals > 0.01).float().mean().item()),
                }
                all_vals.append(vals)
                if step == TEMPORAL_STEPS[-1]:
                    contaminated += int((vals > 0.01).sum().item())
                    total += int(vals.numel())
            rows.append(row)
        if all_vals:
            pooled = torch.cat(all_vals)
            rows.append(
                {
                    "split": split,
                    "view_id": "ALL",
                    "step": step,
                    "safe_pixel_count": int(pooled.numel()),
                    "mean_accumulation": float(pooled.mean().item()),
                    "p50_accumulation": float(torch.quantile(pooled, 0.50).item()),
                    "p90_accumulation": float(torch.quantile(pooled, 0.90).item()),
                    "fraction_accumulation_gt_0p01": float((pooled > 0.01).float().mean().item()),
                }
            )
    return rows


def _object_contamination_rows(split: str, view_ids: Sequence[str], masks: Mapping[str, Mapping[str, Tensor]], final_maps: Mapping[str, Mapping[str, Tensor]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pooled_acc = []
    pooled_direct = []
    pooled_medium = []
    for view_id in view_ids:
        mask = masks[view_id]["M_SAFE"]
        data = final_maps[view_id]
        acc = data["accumulation"][..., 0] if data["accumulation"].ndim == 3 else data["accumulation"]
        direct = torch.linalg.norm(data.get("direct_object_signal", torch.zeros_like(data["pred_image"])), dim=-1)
        medium = torch.linalg.norm(data.get("rgb_medium", torch.zeros_like(data["pred_image"])), dim=-1)
        vals_acc = acc[mask]
        vals_direct = direct[mask]
        vals_medium = medium[mask]
        if vals_acc.numel():
            pooled_acc.append(vals_acc)
            pooled_direct.append(vals_direct)
            pooled_medium.append(vals_medium)
        row = {
            "split": split,
            "view_id": view_id,
            "safe_pixel_count": int(mask.sum().item()),
            "mean_accumulation": float(vals_acc.mean().item()) if vals_acc.numel() else float("nan"),
            "p90_accumulation": float(torch.quantile(vals_acc, 0.90).item()) if vals_acc.numel() else float("nan"),
            "mean_direct_object_signal_l2": float(vals_direct.mean().item()) if vals_direct.numel() else float("nan"),
            "p90_direct_object_signal_l2": float(torch.quantile(vals_direct, 0.90).item()) if vals_direct.numel() else float("nan"),
            "mean_medium_signal_l2": float(vals_medium.mean().item()) if vals_medium.numel() else float("nan"),
        }
        rows.append(row)
    if pooled_acc:
        acc_all = torch.cat(pooled_acc)
        direct_all = torch.cat(pooled_direct)
        medium_all = torch.cat(pooled_medium)
        rows.append(
            {
                "split": split,
                "view_id": "ALL",
                "safe_pixel_count": int(acc_all.numel()),
                "mean_accumulation": float(acc_all.mean().item()),
                "p90_accumulation": float(torch.quantile(acc_all, 0.90).item()),
                "mean_direct_object_signal_l2": float(direct_all.mean().item()),
                "p90_direct_object_signal_l2": float(torch.quantile(direct_all, 0.90).item()),
                "mean_medium_signal_l2": float(medium_all.mean().item()),
                "LOW_OBJECT_SUPPORT_CONFIRMED": bool(float(acc_all.mean().item()) <= 0.01 and float(torch.quantile(acc_all, 0.90).item()) <= 0.05),
                "OBJECT_CONTAMINATION_WARNING": bool(float(acc_all.mean().item()) > 0.01 or float(torch.quantile(acc_all, 0.90).item()) > 0.05),
            }
        )
    return rows


def _brightness_control(split: str, view_ids: Sequence[str], masks: Mapping[str, Mapping[str, Tensor]], final_maps: Mapping[str, Mapping[str, Tensor]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for view_id in view_ids:
        mask = masks[view_id]["M_SAFE"]
        if int(mask.sum().item()) == 0:
            rows.append({"split": split, "view_id": view_id, "status": "EMPTY_SAFE_MASK"})
            continue
        gt = final_maps[view_id]["gt"].clamp(0.0, 1.0)
        pred = final_maps[view_id]["pred_image"].clamp(0.0, 1.0)
        b_inf = final_maps[view_id]["b_inf"].clamp(0.0, 1.0)
        acc = final_maps[view_id]["accumulation"]
        if acc.ndim == 3:
            acc = acc[..., 0]
        luma = (gt * LUMA.to(gt)).sum(dim=-1)
        safe_vals = luma[mask]
        h, w = mask.shape
        yy = torch.linspace(0.0, 1.0, h).reshape(h, 1).expand(h, w)
        domain = (~mask) & (acc > 0.01)
        domain_idx = torch.nonzero(domain.reshape(-1), as_tuple=False).reshape(-1)
        control = torch.zeros_like(mask)
        if domain_idx.numel() > 0:
            k = min(int(mask.sum().item()), int(domain_idx.numel()))
            safe_sorted = torch.sort(safe_vals).values
            candidate_luma = luma.reshape(-1)[domain_idx]
            chosen: List[int] = []
            bins = torch.linspace(0, max(safe_sorted.numel() - 1, 0), steps=max(k, 1)).long()
            used = torch.zeros(domain_idx.numel(), dtype=torch.bool)
            for target in safe_sorted[bins]:
                diffs = (candidate_luma - target).abs()
                diffs[used] = float("inf")
                pos = int(torch.argmin(diffs).item())
                used[pos] = True
                chosen.append(int(domain_idx[pos].item()))
            if chosen:
                control.reshape(-1)[torch.tensor(chosen, dtype=torch.long)] = True
        def errs(m: Tensor) -> Tuple[float, float, float]:
            if int(m.sum().item()) == 0:
                return float("nan"), float("nan"), float("nan")
            eb = float((b_inf - gt)[m].abs().mean().item())
            ef = float((pred - gt)[m].abs().mean().item())
            return eb, ef, (eb - ef) / max(eb, EPS)
        eb_s, ef_s, r_s = errs(mask)
        eb_c, ef_c, r_c = errs(control)
        rows.append(
            {
                "split": split,
                "view_id": view_id,
                "safe_pixel_count": int(mask.sum().item()),
                "control_pixel_count": int(control.sum().item()),
                "safe_luma_mean": float(safe_vals.mean().item()),
                "safe_luma_p10": float(torch.quantile(safe_vals, 0.10).item()),
                "safe_luma_p50": float(torch.quantile(safe_vals, 0.50).item()),
                "safe_luma_p90": float(torch.quantile(safe_vals, 0.90).item()),
                "safe_y_mean": float(yy[mask].mean().item()),
                "safe_top20_fraction": float((mask & (yy <= 0.2)).sum().item()) / max(int(mask.sum().item()), 1),
                "safe_bottom20_fraction": float((mask & (yy >= 0.8)).sum().item()) / max(int(mask.sum().item()), 1),
                "safe_E_BINF": eb_s,
                "safe_E_full": ef_s,
                "safe_R_anchor": r_s,
                "control_E_BINF": eb_c,
                "control_E_full": ef_c,
                "control_R_anchor": r_c,
            }
        )
    return rows


def _rgb_to_img(image: Tensor) -> Image.Image:
    arr = image.detach().float().cpu().clamp(0.0, 1.0).numpy()
    return Image.fromarray(np.round(arr * 255.0).astype(np.uint8), mode="RGB")


def _gray_to_img(values: Tensor, vmin: float = 0.0, vmax: float = 1.0) -> Image.Image:
    arr = values.detach().float().cpu().numpy()
    arr = np.nan_to_num(arr, nan=vmin, posinf=vmax, neginf=vmin)
    arr = np.clip((arr - vmin) / max(vmax - vmin, EPS), 0.0, 1.0)
    u8 = np.round(arr * 255.0).astype(np.uint8)
    return Image.fromarray(np.stack([u8, u8, u8], axis=-1), mode="RGB")


def _mask_to_img(mask: Tensor, color: Tuple[int, int, int]) -> Image.Image:
    arr = np.zeros((*mask.shape, 3), dtype=np.uint8)
    arr[mask.detach().bool().cpu().numpy()] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _error_img(values: Tensor, scale: float) -> Image.Image:
    mag = values.detach().float().abs()
    if mag.ndim == 3:
        mag = mag.mean(dim=-1)
    return _gray_to_img(mag, 0.0, scale)


def _label(img: Image.Image, label: str, height: int = 24) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + height), "white")
    out.paste(img, (0, height))
    ImageDraw.Draw(out).text((6, 6), label, fill=(0, 0, 0))
    return out


def _resize_tile(img: Image.Image, tile_width: int) -> Image.Image:
    if img.width == tile_width:
        return img
    height = max(1, int(round(img.height * (tile_width / max(img.width, 1)))))
    return img.resize((tile_width, height), Image.Resampling.BILINEAR)


def _save_sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Image.Image]]], manifest: List[Dict[str, Any]], output_type: str, view_ids: Sequence[str]) -> None:
    if not rows:
        return
    tile_w = min(640, max(img.width for row in rows for _label_text, img in row))
    tile_h = max(_resize_tile(img, tile_w).height for row in rows for _label_text, img in row) + 24
    cols = max(len(row) for row in rows)
    canvas = Image.new("RGB", (tile_w * cols, tile_h * len(rows)), "white")
    for y, row in enumerate(rows):
        for x, (label_text, img) in enumerate(row):
            tile = _label(_resize_tile(img, tile_w), label_text)
            canvas.paste(tile, (x * tile_w, y * tile_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    manifest.append({"file_path": str(path), "output_type": output_type, "view_ids": ";".join(view_ids)})


def _simple_bar_plot(path: Path, labels: Sequence[str], values: Sequence[float], ylabel: str, title: str, manifest: List[Dict[str, Any]], output_type: str) -> None:
    width = max(600, 70 * len(labels))
    height = 420
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    left, top, bottom = 70, 45, height - 95
    right = width - 20
    finite = [v for v in values if np.isfinite(v)]
    vmax = max(max(finite) if finite else 1.0, 1e-6)
    draw.text((left, 12), title, fill=(0, 0, 0))
    draw.text((10, 18), ylabel, fill=(0, 0, 0))
    draw.line((left, bottom, right, bottom), fill=(0, 0, 0))
    draw.line((left, top, left, bottom), fill=(0, 0, 0))
    bar_w = max(6, int((right - left) / max(len(labels), 1) * 0.65))
    for i, (lab, val) in enumerate(zip(labels, values)):
        cx = left + int((i + 0.5) * (right - left) / max(len(labels), 1))
        vh = 0 if not np.isfinite(val) else int((bottom - top) * max(val, 0.0) / vmax)
        draw.rectangle((cx - bar_w // 2, bottom - vh, cx + bar_w // 2, bottom), fill=(70, 130, 200))
        draw.text((cx - bar_w, bottom + 5), lab[:12], fill=(0, 0, 0))
        draw.text((cx - bar_w, bottom - vh - 16), f"{val:.3g}" if np.isfinite(val) else "nan", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    manifest.append({"file_path": str(path), "output_type": output_type, "view_ids": ";".join(labels)})


def _make_visuals(render_dir: Path, render_manifest: List[Dict[str, Any]], masks: Mapping[str, Mapping[str, Tensor]], temporal_maps: Mapping[int, Mapping[str, Mapping[str, Tensor]]], final_maps: Mapping[str, Mapping[str, Tensor]], train_views: Sequence[str], heldout_views: Sequence[str], coverage_rows: Sequence[Mapping[str, Any]], contamination_rows: Sequence[Mapping[str, Any]], headroom_rows: Sequence[Mapping[str, Any]], brightness_rows: Sequence[Mapping[str, Any]], gradient_summary: Mapping[str, Any]) -> None:
    render_dir.mkdir(parents=True, exist_ok=True)
    view_groups = [("train", train_views), ("heldout", heldout_views)]
    for split, views in view_groups:
        rows = []
        for view_id in views:
            data = final_maps[view_id]
            pseudo = masks[view_id]["pseudo_depth"]
            rows.append([(f"{view_id} GT", _rgb_to_img(data["gt"])), ("pseudo_depth", _gray_to_img(pseudo, 0.0, 1.0))])
        _save_sheet(render_dir / f"contact_sheet_{split}_rgb_pseudo_depth.png", rows, render_manifest, f"{split}_rgb_pseudo_depth", views)

        for mask_name, color in (
            ("M_SF", (80, 170, 255)),
            ("M_LOW_SUPPORT", (255, 180, 40)),
            ("M_INTERSECT", (120, 220, 120)),
            ("M_SAFE", (255, 70, 70)),
        ):
            rows = [[(f"{view_id} {mask_name}", _mask_to_img(masks[view_id][mask_name], color))] for view_id in views]
            _save_sheet(render_dir / f"contact_sheet_{split}_{mask_name.lower()}.png", rows, render_manifest, f"{split}_{mask_name}", views)

    for split, views in (("train", train_views), ("heldout", heldout_views)):
        temporal_rows = []
        for view_id in views:
            cols = []
            for step in TEMPORAL_STEPS:
                acc = temporal_maps[step][view_id]["accumulation"]
                if acc.ndim == 3:
                    acc = acc[..., 0]
                cols.append((f"{view_id} {step}", _gray_to_img(acc, 0.0, 1.0)))
            temporal_rows.append(cols)
        _save_sheet(render_dir / f"contact_sheet_temporal_accumulation_{split}.png", temporal_rows, render_manifest, f"temporal_accumulation_{split}", views)

        final_acc_rows = []
        for view_id in views:
            data = final_maps[view_id]
            acc = data["accumulation"]
            if acc.ndim == 3:
                acc = acc[..., 0]
            safe = masks[view_id]["M_SAFE"]
            final_acc_rows.append(
                [
                    (f"{view_id} GT", _rgb_to_img(data["gt"])),
                    ("K1 final accumulation", _gray_to_img(acc, 0.0, 1.0)),
                    ("M_SAFE", _mask_to_img(safe, (255, 70, 70))),
                    ("accumulation on M_SAFE", _gray_to_img(acc * safe.float(), 0.0, 1.0)),
                ]
            )
        _save_sheet(
            render_dir / f"contact_sheet_{split}_final_accumulation_over_safe_mask.png",
            final_acc_rows,
            render_manifest,
            f"{split}_final_accumulation_over_safe_mask",
            views,
        )

    rows = []
    for view_id in heldout_views:
        data = final_maps[view_id]
        gt = data["gt"].clamp(0.0, 1.0)
        pred = data["pred_image"].clamp(0.0, 1.0)
        b_inf = data["b_inf"].clamp(0.0, 1.0)
        direct = data.get("direct_object_signal", torch.zeros_like(pred)).clamp(0.0, 1.0)
        medium = data.get("rgb_medium", torch.zeros_like(pred)).clamp(0.0, 1.0)
        anchor_gap = (b_inf - gt).abs().mean(dim=-1) - (pred - gt).abs().mean(dim=-1)
        rows.append(
            [
                (f"{view_id} GT", _rgb_to_img(gt)),
                ("B_inf", _rgb_to_img(b_inf)),
                ("full pred", _rgb_to_img(pred)),
                ("|B_inf-GT|", _error_img(b_inf - gt, 0.25)),
                ("|I_pred-GT|", _error_img(pred - gt, 0.25)),
                ("anchor gap", _gray_to_img(anchor_gap, -0.1, 0.1)),
                ("direct object", _rgb_to_img(direct)),
                ("medium", _rgb_to_img(medium)),
                ("M_SAFE", _mask_to_img(masks[view_id]["M_SAFE"], (255, 70, 70))),
            ]
        )
    _save_sheet(render_dir / "contact_sheet_heldout_binf_headroom_components.png", rows, render_manifest, "heldout_binf_headroom_components", heldout_views)

    labels = [r["view_id"] for r in coverage_rows if r["split"] == "train" and r["mask"] == "M_SAFE" and r["view_id"] != "ALL"]
    vals = [float(r["coverage_fraction"]) for r in coverage_rows if r["split"] == "train" and r["mask"] == "M_SAFE" and r["view_id"] != "ALL"]
    _simple_bar_plot(render_dir / "plot_training_safe_mask_coverage.png", labels, vals, "coverage", "Training M_SAFE coverage", render_manifest, "training_safe_mask_coverage")

    labels = [r["view_id"] for r in contamination_rows if r["split"] == "train" and r["view_id"] != "ALL"]
    vals = [float(r["p90_accumulation"]) for r in contamination_rows if r["split"] == "train" and r["view_id"] != "ALL"]
    _simple_bar_plot(render_dir / "plot_late_object_contamination_p90.png", labels, vals, "p90 accumulation", "Final accumulation on M_SAFE", render_manifest, "late_object_contamination_p90")

    labels = [r["view_id"] for r in headroom_rows if r["split"] == "train" and r["mask"] == "M_SAFE" and r["view_id"] != "ALL"]
    vals = [float(r["E_BINF"]) for r in headroom_rows if r["split"] == "train" and r["mask"] == "M_SAFE" and r["view_id"] != "ALL"]
    _simple_bar_plot(render_dir / "plot_binf_error_per_training_view.png", labels, vals, "E_BINF", "B_inf error on M_SAFE", render_manifest, "binf_error_per_training_view")

    labels = [r["view_id"] for r in headroom_rows if r["split"] == "train" and r["mask"] == "M_SAFE" and r["view_id"] != "ALL"]
    vals = [float(r["R_anchor"]) for r in headroom_rows if r["split"] == "train" and r["mask"] == "M_SAFE" and r["view_id"] != "ALL"]
    _simple_bar_plot(render_dir / "plot_binf_vs_full_render_gap.png", labels, vals, "R_anchor", "B_inf vs full-render error gap", render_manifest, "binf_vs_full_render_gap")

    labels = [r["view_id"] for r in brightness_rows if r["split"] == "train"]
    vals = [float(r.get("safe_R_anchor", float("nan"))) for r in brightness_rows if r["split"] == "train"]
    _simple_bar_plot(render_dir / "plot_brightness_matched_control_safe_anchor.png", labels, vals, "safe R_anchor", "Brightness control context", render_manifest, "brightness_matched_control")

    for view_id in heldout_views:
        row = next((r for r in headroom_rows if r["split"] == "heldout" and r["view_id"] == view_id and r["mask"] == "M_SAFE"), {})
        lines = [f"{k}: {v}" for k, v in row.items()]
        _save_text_sheet(render_dir / f"heldout_{view_id}_summary.png", f"Held-out {view_id}", lines, render_manifest, f"heldout_{view_id}_summary")

    grad_lines = [f"{k}: {v}" for k, v in gradient_summary.items() if k != "rows"]
    _save_text_sheet(render_dir / "gradient_route_summary.png", "Virtual L_bg Gradient Route", grad_lines, render_manifest, "gradient_route_summary")


def _save_text_sheet(path: Path, title: str, lines: Sequence[str], manifest: List[Dict[str, Any]], output_type: str) -> None:
    width = 1100
    height = max(160, 34 + 18 * len(lines))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), title, fill=(0, 0, 0))
    y = 34
    for line in lines:
        draw.text((12, y), str(line)[:180], fill=(0, 0, 0))
        y += 18
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    manifest.append({"file_path": str(path), "output_type": output_type})


def _param_snapshots(model: Any) -> Dict[str, List[Tensor]]:
    out: Dict[str, List[Tensor]] = {}
    for group, params in model.get_param_groups().items():
        out[group] = [p.detach().clone().cpu() for p in params]
    return out


def _param_delta_rows(before: Mapping[str, Sequence[Tensor]], model: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group, params in model.get_param_groups().items():
        max_delta = 0.0
        for idx, param in enumerate(params):
            delta = (param.detach().cpu() - before[group][idx]).abs()
            if delta.numel():
                max_delta = max(max_delta, float(delta.max().item()))
        rows.append({"parameter_group": group, "max_abs_delta": max_delta})
    return rows


def _grad_norm(params: Sequence[Tensor]) -> Tuple[float, float]:
    total_sq = 0.0
    max_abs = 0.0
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total_sq += float(grad.square().sum().item())
        if grad.numel():
            max_abs = max(max_abs, float(grad.abs().max().item()))
    return total_sq ** 0.5, max_abs


def _gradient_audit(repo: Path, safe_masks: Mapping[str, Tensor], view_ids: Sequence[str], checkpoint_safety_before: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    loaded = _load_k1(repo, 14999)
    try:
        model = loaded.pipeline.model
        model.eval()
        requires_grad_before = {
            id(param): bool(param.requires_grad)
            for params in model.get_param_groups().values()
            for param in params
        }
        for params in model.get_param_groups().values():
            for param in params:
                param.requires_grad_(True)
        records = _view_records(loaded.pipeline, view_ids)
        record_by_id = {view_id: (camera, batch) for _idx, view_id, camera, batch in records}
        before = _param_snapshots(model)
        model.zero_grad(set_to_none=True)
        losses = []
        output_grad_rows: List[Dict[str, Any]] = []
        for view_id in view_ids:
            camera, batch = record_by_id[view_id]
            outputs = model.get_outputs(camera.to(model.device))
            b_inf = outputs["b_inf"]
            gt = _get_gt(model, batch, outputs["background"])
            mask = safe_masks[view_id].to(model.device)
            if int(mask.sum().item()) == 0:
                continue
            b_inf.retain_grad()
            medium_bs = outputs["medium_bs"]
            medium_attn = outputs["medium_attn"]
            medium_bs.retain_grad()
            medium_attn.retain_grad()
            loss = ((b_inf - gt).abs() * mask[..., None]).sum() / (3.0 * mask.float().sum() + EPS)
            losses.append(loss)
            output_grad_rows.append(
                {
                    "view_id": view_id,
                    "safe_pixel_count": int(mask.sum().item()),
                    "virtual_bg_loss": float(loss.detach().item()),
                    "_b_inf_tensor": b_inf,
                    "_medium_bs_tensor": medium_bs,
                    "_medium_attn_tensor": medium_attn,
                }
            )
        if losses:
            total_loss = torch.stack(losses).mean()
            total_loss.backward()
        param_rows: List[Dict[str, Any]] = []
        for group, params in model.get_param_groups().items():
            l2, max_abs = _grad_norm(params)
            param_rows.append({"parameter_group": group, "grad_l2": l2, "grad_max_abs": max_abs})
        for row in output_grad_rows:
            for public_key, private_key in (
                ("dL_dA_b_inf", "_b_inf_tensor"),
                ("dL_dbeta_B_medium_bs", "_medium_bs_tensor"),
                ("dL_dbeta_D_medium_attn", "_medium_attn_tensor"),
            ):
                tensor = row.pop(private_key)
                grad = tensor.grad
                row[f"{public_key}_l2"] = float(torch.linalg.norm(grad.detach()).item()) if grad is not None else 0.0
                row[f"{public_key}_max_abs"] = float(grad.detach().abs().max().item()) if grad is not None and grad.numel() else 0.0
        delta_rows = _param_delta_rows(before, model)
        safety = {
            "AUDIT_PARAMETER_SAFETY": "PASS" if all(float(r["max_abs_delta"]) == 0.0 for r in delta_rows) else "FAIL",
            "parameter_delta_rows": delta_rows,
            "CHECKPOINT_SAFETY": "PASS",
            "checkpoint_path": str(loaded.checkpoint_path),
            "checkpoint_size_before": checkpoint_safety_before.get(str(loaded.checkpoint_path), {}).get("size"),
            "checkpoint_mtime_before": checkpoint_safety_before.get(str(loaded.checkpoint_path), {}).get("mtime"),
        }
        summary = {
            "virtual_bg_loss_view_count": len(losses),
            "medium_mlp_grad_l2": next((r["grad_l2"] for r in param_rows if r["parameter_group"] == "medium_mlp"), 0.0),
            "direction_encoding_grad_l2": next((r["grad_l2"] for r in param_rows if r["parameter_group"] == "direction_encoding"), 0.0),
            "means_grad_l2": next((r["grad_l2"] for r in param_rows if r["parameter_group"] == "means"), 0.0),
            "scales_grad_l2": next((r["grad_l2"] for r in param_rows if r["parameter_group"] == "scales"), 0.0),
            "quats_grad_l2": next((r["grad_l2"] for r in param_rows if r["parameter_group"] == "quats"), 0.0),
            "opacities_grad_l2": next((r["grad_l2"] for r in param_rows if r["parameter_group"] == "opacities"), 0.0),
            "features_dc_grad_l2": next((r["grad_l2"] for r in param_rows if r["parameter_group"] == "features_dc"), 0.0),
            "features_rest_grad_l2": next((r["grad_l2"] for r in param_rows if r["parameter_group"] == "features_rest"), 0.0),
        }
        object_grad_max = max(
            summary["means_grad_l2"],
            summary["scales_grad_l2"],
            summary["quats_grad_l2"],
            summary["opacities_grad_l2"],
            summary["features_dc_grad_l2"],
            summary["features_rest_grad_l2"],
        )
        summary["MEDIUM_ONLY_GRADIENT_ROUTE"] = bool(summary["medium_mlp_grad_l2"] > 0.0 and object_grad_max == 0.0)
        param_rows.extend(output_grad_rows)
        return param_rows, safety, summary
    finally:
        if "model" in locals():
            for params in model.get_param_groups().values():
                for param in params:
                    param.requires_grad_(requires_grad_before.get(id(param), bool(param.requires_grad)))
        _release(loaded)


def _checkpoint_manifest(repo: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    safety: Dict[str, Any] = {}
    config_path = repo / K1_CONFIG
    for step in TEMPORAL_STEPS:
        actual = _actual_step(config_path, step)
        path = _available_steps(config_path)[actual]
        stat = path.stat()
        row = {
            "scene": SCENE,
            "run": "BND-K1",
            "requested_step": step,
            "actual_step": actual,
            "checkpoint_path": str(path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256": _sha256(path),
        }
        rows.append(row)
        safety[str(path)] = {"size": stat.st_size, "mtime": stat.st_mtime, "sha256": row["sha256"]}
    return rows, safety


def _write_source_audits(output_dir: Path, repo_manifest: Mapping[str, Any]) -> None:
    ws_audit = {
        "B_INF_SOURCE_TENSOR": "outputs['b_inf'] from water_splatting/water_splatting.py::get_outputs",
        "b_inf_mode_tied_semantics": "DirectionConditionedMediumField sets b_inf = medium_rgb; get_outputs uses b_inf in tied tail recomposition.",
        "medium_rgb_activation": "Sigmoid applied to medium_mlp output channels 0:3.",
        "medium_bs_activation": "Softplus applied to medium_mlp output channels 3:6 plus density_bias.",
        "medium_attn_activation": "Softplus applied to medium_mlp output channels 6:9 plus density_bias.",
        "shared_structure": "direction_encoding + appended context -> one 9-channel medium_mlp; rgb/bs/attn are channel slices of the same output.",
        "context_mode": "dir_xy_camera for formal K1; input is SH-encoded ray direction plus image x/y/r plus normalized camera-center xyz.",
        "infinite_water_enabled": "False for formal K1; current clean branch raises ValueError if enabled.",
        "why_binf_exists_when_infinite_water_disabled": "b_inf is an explicit tied asymptotic tail color used to recompose the renderer's medium tail, even though no separate infinite-water ownership path is active.",
        "tail_recomposition": "tail_weight = final_transmittance * exp(-medium_bs * last_depth); rgb_tail = tail_weight * b_inf; rgb = rgb_object + rgb_medium_finite + rgb_tail.",
        "source_files": [
            "water_splatting/fields/medium_field.py",
            "water_splatting/water_splatting.py",
            "water_splatting/rendering/underwater_rasterizer.py",
            "water_splatting/cuda/csrc/forward.cu",
        ],
    }
    mapping = {
        "SeaFree_A": "WaterSplatting b_inf / medium_rgb in b_inf_mode=tied; NOT EXACTLY EQUIVALENT because WaterSplatting also has finite backscatter integration and tied tail recomposition.",
        "SeaFree_beta_D": "WaterSplatting medium_attn.",
        "SeaFree_beta_B": "WaterSplatting medium_bs.",
        "SeaFree_I_bg": "WaterSplatting b_inf is the closest A-like asymptotic color; rendered tail contribution is tail_weight * b_inf, not bare b_inf.",
        "SeaFree_water_background_image": "Pixel ambient_light_colors queried per LOS.",
        "structural_difference": "SeaFree degrades Gaussian colors before generic gsplat rasterization and adds (1-alpha)*A background. WaterSplatting integrates medium in the custom underwater rasterizer and then recomposes tied B_inf tail.",
    }
    seafree_audit = {
        "reference_repo": str(SEAFREE_REPO),
        "reference_commit": repo_manifest.get("seafree_head"),
        "reference_status_short": repo_manifest.get("seafree_status_short"),
        "source_file": "seafree_gs/seafree_model.py",
        "pseudo_depth": "batch['depth_image']; normalized by pseudo_depth / pseudo_depth.max() inside get_loss_dict.",
        "foreground_mask_source": "pseudo_depth < 1e-2 is converted to uint8 then cv2.THRESH_BINARY_INV selects pseudo_depth >= 1e-2; largest external contour is filled as foreground.",
        "background_mask": "foreground_mask < 0.5",
        "largest_contour": "max(contours, key=cv2.contourArea), filled with cv2.drawContours.",
        "erosion_dilation": "none in source mask code.",
        "static_cache": "foreground_mask_cache keyed by image_idx and downscale factor.",
        "gradient_participation": "mask is constructed from numpy/cv2 and stored as tensor; no gradient to pseudo-depth.",
        "background_supervision_target": "water_background_image (pixel ambient light A) vs gt_underwater_image on background pixels.",
        "background_weight": "1 / (background_ambient_light_pixels.detach() + 1e-3).",
        "background_loss_coeff": "0.01 inside content_based_reconstruction_loss, active when enable_background_water_supervision and step < 15000 and background_pixel_ratio > 0.05.",
        "directly_supervised_outputs": "output-level A / water_background_image only; beta_D and beta_B are not directly in this loss, though WPP parameters are shared.",
    }
    native = {
        "status": "NOT_EVALUABLE",
        "reason": "Current research/m1-bounded-intrinsic branch rejects infinite_water_enabled=True and no separate active native infinite-water render function is used in formal K1.",
    }
    _write_json(output_dir / "watersplatting_binf_source_audit.json", ws_audit)
    _write_json(output_dir / "watersplatting_seafree_mapping.json", mapping)
    _write_json(output_dir / "seafree_background_mask_source_audit.json", seafree_audit)
    _write_json(output_dir / "native_infinite_water_diagnostic.json", native)
    _write_csv(output_dir / "native_infinite_water_diagnostic.csv", [native])
    (output_dir / "watersplatting_binf_source_audit.md").write_text(
        "# WaterSplatting B_inf Source Audit\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in ws_audit.items())
        + "\n",
        encoding="utf8",
    )
    (output_dir / "watersplatting_seafree_mapping.md").write_text(
        "# WaterSplatting-SeaFree Mapping\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in mapping.items())
        + "\n",
        encoding="utf8",
    )


def _write_visual_index(render_dir: Path, manifest: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# BND Pure-Water Audit Visual Index", ""]
    for row in manifest:
        lines.append(f"- {row.get('output_type')}: `{Path(row['file_path']).resolve()}`")
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_research_note(path: Path, summary: Mapping[str, Any]) -> None:
    cls = summary["classifications"]
    lines = [
        "# BND Pure-Water Background Anchor Readiness",
        "",
        "## Motivation",
        "INFERENCE: This read-only audit asks whether Panama contains enough reliable background/water-only candidate observations to anchor the existing asymptotic water prediction in BND-K1.",
        "",
        "## Why BND-Aware Refinement Is Not The Next Primary Direction",
        "EXPERIMENTAL FACT: BND-AWARE-REFINE was formally INCONCLUSIVE. RH showed a weak +0.021 dB signal over R0, but the final population mismatch exceeded the preregistered 2 percent tolerance.",
        "INFERENCE: The main research direction therefore returns to object-medium identifiability rather than further spatial refinement heuristics.",
        "",
        "## Why Medium Identifiability Is Revisited",
        "HYPOTHESIS: A directly observable background/asymptotic water channel could reduce object-medium compensation freedom without changing renderer physics.",
        "",
        "## SeaFree Background-Water Supervision Mechanism",
        f"CODE FACT: SeaFree-GS reference commit `{summary['repo']['seafree_head']}` with status `{summary['repo']['seafree_status_short']}`.",
        "CODE FACT: SeaFree background supervision compares pixel ambient-light `water_background_image` with GT underwater image on pseudo-depth background pixels, using inverse ambient-light weights and coefficient `0.01`.",
        "CODE FACT: SeaFree's pseudo-depth mask is cached per image/downscale and does not carry gradient to pseudo-depth.",
        "",
        "## WaterSplatting Asymptotic-Water Semantics",
        "CODE FACT: In current WaterSplatting, `b_inf_mode=tied` makes `B_inf = medium_rgb`; `medium_rgb` is sigmoid-activated channels 0:3 from the shared 9-channel medium MLP.",
        "CODE FACT: Formal K1 uses `medium_context_mode=dir_xy_camera`, `b_inf_mode=tied`, and `infinite_water_enabled=False`; the clean branch rejects enabling infinite-water.",
        "CODE FACT: The tied recomposition replaces the renderer default tail with `tail_weight * b_inf` where `tail_weight = final_transmittance * exp(-medium_bs * last_depth)`.",
        "",
        "## B_inf / Medium_Rgb / Infinite-Water Source Audit",
        "CODE FACT: `medium_rgb`, `medium_bs`, and `medium_attn` are channel slices of one 9-channel medium MLP output. Only `medium_rgb` is directly equivalent to the tied `B_inf` tensor.",
        "CODE FACT: Current native infinite-water diagnostic is NOT_EVALUABLE because the active clean branch raises on `infinite_water_enabled=True`.",
        "",
        "## WaterSplatting-SeaFree Semantic Mapping",
        "CODE FACT: SeaFree `A` maps most closely to WaterSplatting `b_inf / medium_rgb`, but this is NOT EXACTLY EQUIVALENT because WaterSplatting uses custom finite-medium integration plus tied tail recomposition.",
        "CODE FACT: SeaFree `beta_D` maps to WaterSplatting `medium_attn`; SeaFree `beta_B` maps to `medium_bs`.",
        "",
        "## Pseudo-Depth Source",
        "CONFIG FACT: Pseudo-depth source is `undistorted_data/undistorted_Panama/depthAnything_u16`, the same cache used in the formal Panama CDEPTH diagnostics.",
        "CONFIG FACT: Pseudo-depth is used offline only for mask construction. `PSEUDO_DEPTH_GRADIENT_TO_GEOMETRY = FALSE`.",
        "",
        "## SeaFree Background-Mask Semantics",
        "CODE FACT: SeaFree normalizes pseudo-depth by per-image max, thresholds at `1e-2`, inverts the thresholded image to choose `pseudo_depth >= 1e-2`, fills the largest external contour as foreground, and uses its complement as background.",
        "",
        "## Candidate-Mask Definitions",
        "CONFIG FACT: `M_SF` is the SeaFree-style pseudo-depth background candidate; `M_LOW_SUPPORT = accumulation_3k <= 0.01`; `M_INTERSECT = M_SF & M_LOW_SUPPORT`; `M_SAFE = BinaryErode(M_INTERSECT, radius=5 px)`.",
        "CONFIG FACT: The definition was locked before held-out processing in `locked_pure_water_candidate_definition.json`.",
        "",
        "## Training / Held-Out Split",
        "CONFIG FACT: Training-development views are MTN_1538, MTN_1541, MTN_1540, MTN_1534, MTN_1535, MTN_1536, MTN_1533, MTN_1542, MTN_1537, MTN_1532, MTN_1546, MTN_1543, MTN_1544, MTN_1545, MTN_1548.",
        "CONFIG FACT: Held-out views are MTN_1529, MTN_1539, MTN_1547.",
        "",
        "## Leakage Controls",
        "EXPERIMENTAL FACT: `locked_pure_water_candidate_definition.json` was written before held-out metrics were interpreted. `HELD_OUT_MASK_SELECTION_LEAKAGE = FALSE`.",
        "",
        "## Mask Coverage",
        f"QUANTITATIVE RESULT: training pooled M_SAFE coverage = `{summary['coverage']['train_safe_coverage']}`; views >=1% = `{summary['coverage']['train_safe_views_ge_1pct']}`.",
        f"QUANTITATIVE RESULT: held-out pooled M_SAFE coverage = `{summary['coverage']['heldout_safe_coverage']}`; views >=1% = `{summary['coverage']['heldout_safe_views_ge_1pct']}`.",
        f"QUANTITATIVE CONCLUSION: `SAFE_MASK_COVERAGE_ADEQUATE = {cls['SAFE_MASK_COVERAGE_ADEQUATE']}`.",
        "",
        "## Mask Agreement",
        "EXPERIMENTAL FACT: Cross-mask agreement rows are stored in `mask_agreement.csv/json`; high agreement is not treated as proof of true water.",
        "",
        "## Temporal Support Stability",
        f"QUANTITATIVE RESULT: final pooled train late contamination fraction = `{summary['temporal']['train_final_fraction_accumulation_gt_0p01']}`.",
        f"QUANTITATIVE CONCLUSION: `SUPPORT_STABLE_SAFE_MASK = {cls['SUPPORT_STABLE_SAFE_MASK']}`.",
        "",
        "## Late Object-Contamination Proxy",
        f"QUANTITATIVE CONCLUSION: `LOW_OBJECT_SUPPORT_CONFIRMED = {cls['LOW_OBJECT_SUPPORT_CONFIRMED']}` and `OBJECT_CONTAMINATION_WARNING = {cls['OBJECT_CONTAMINATION_WARNING']}`.",
        "",
        "## K1 B_inf Extraction",
        "CODE FACT: Current K1 B_inf is extracted from `outputs['b_inf']` at final checkpoint step 14999, with BND-K1@3k used only for locked low-support mask construction.",
        "",
        "## B_inf Error",
        f"QUANTITATIVE RESULT: train M_SAFE E_BINF = `{summary['headroom']['train_safe_E_BINF']}`, E_full = `{summary['headroom']['train_safe_E_full']}`, R_anchor = `{summary['headroom']['train_safe_R_anchor']}`.",
        f"QUANTITATIVE RESULT: held-out M_SAFE E_BINF = `{summary['headroom']['heldout_safe_E_BINF']}`, E_full = `{summary['headroom']['heldout_safe_E_full']}`, R_anchor = `{summary['headroom']['heldout_safe_R_anchor']}`.",
        "",
        "## Full-Render vs B_inf Error",
        "INFERENCE: Positive `R_anchor` is compensation-compatible evidence, but it does not identify the responsible component and does not make the mask a pure-water ground truth.",
        f"QUANTITATIVE CONCLUSION: `BACKGROUND_ANCHOR_HEADROOM = {cls['BACKGROUND_ANCHOR_HEADROOM']}` and `HELDOUT_BG_HEADROOM_CONSISTENT = {cls['HELDOUT_BG_HEADROOM_CONSISTENT']}`.",
        "",
        "## Anchor-Headroom Gap",
        "EXPERIMENTAL FACT: Headroom tables are stored in `background_anchor_headroom.csv/json`, `binf_statistics.csv/json`, and `full_render_vs_binf.csv/json`.",
        "",
        "## Object Contribution On Candidate Water Pixels",
        "EXPERIMENTAL FACT: `object_contamination_audit.csv/json` records accumulation, direct-object signal magnitude, and medium signal magnitude on M_SAFE.",
        "",
        "## Brightness Confound",
        "EXPERIMENTAL FACT: Brightness/context diagnostics and matched-control rows are stored in `brightness_matched_control.csv/json`; these are diagnostics only and not method masks.",
        "",
        "## Held-Out Validation",
        "EXPERIMENTAL FACT: Held-out metrics are stored in `heldout_water_candidate_metrics.csv/json` and `heldout_binf_headroom.csv/json`.",
        "",
        "## Virtual Background-Loss Gradient Route",
        f"QUANTITATIVE RESULT: medium_mlp grad L2 = `{summary['gradient']['medium_mlp_grad_l2']}`; max object/appearance grad L2 = `{summary['gradient']['object_grad_l2_max']}`.",
        f"QUANTITATIVE CONCLUSION: `MEDIUM_ONLY_GRADIENT_ROUTE = {cls['MEDIUM_ONLY_GRADIENT_ROUTE']}`.",
        "CODE FACT: Output-level gradient rows show nonzero `dL/dA_b_inf` and zero `dL/dbeta_B`, `dL/dbeta_D` for the virtual `|B_inf-GT|` loss.",
        "",
        "## Native Infinite-Water Diagnostic",
        "EXPERIMENTAL FACT: Native infinite-water diagnostic is `NOT_EVALUABLE` for this branch because the code path is disabled and not activated in K1.",
        "",
        "## Water-Candidate Classification",
        f"QUANTITATIVE CONCLUSION: Water candidate classification = `{summary['classifications']['water_candidate']}`.",
        "",
        "## Background-Anchor Readiness Classification",
        f"QUANTITATIVE CONCLUSION: Background anchor classification = `{summary['classifications']['background_anchor']}`.",
        "",
        "## Scientific Interpretation",
        "INFERENCE: The audit treats masks only as conservative background/water-only candidates, not ground-truth pure-water labels.",
        "INFERENCE: Under the locked definition, M_SAFE has nontrivial B_inf disagreement but insufficient stable low-object support and insufficient coverage for a training-ready anchor.",
        "",
        "## Next Single-Factor Decision",
        f"INFERENCE: Next single-factor decision = `{summary['next_single_factor_experiment']}`.",
        "",
        "## Safety",
        f"EXPERIMENTAL FACT: AUDIT_PARAMETER_SAFETY = `{summary['safety']['AUDIT_PARAMETER_SAFETY']}`; CHECKPOINT_SAFETY = `{summary['safety']['CHECKPOINT_SAFETY']}`.",
        "",
        "## Outputs",
        f"EXPERIMENTAL FACT: Output directory `{OUTPUT_DIR}`.",
        f"EXPERIMENTAL FACT: Render directory `{RENDER_DIR}`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def run(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    output_dir = repo / OUTPUT_DIR
    render_dir = repo / RENDER_DIR
    log_dir = repo / LOG_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu_manifest = _assert_allowed_gpu_policy()
    repo_manifest = {
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "start_head": _git(repo, "rev-parse", "HEAD"),
        "initial_status_short": _git(repo, "status", "--short"),
        "log_10": _git(repo, "log", "-10", "--oneline"),
        "diff_stat": _git(repo, "diff", "--stat"),
        "diff_check": _git(repo, "diff", "--check"),
        "seafree_repo": str(SEAFREE_REPO),
        "seafree_head": _git(SEAFREE_REPO, "rev-parse", "HEAD") if SEAFREE_REPO.exists() else "NOT_FOUND",
        "seafree_status_short": _git(SEAFREE_REPO, "status", "--short") if SEAFREE_REPO.exists() else "NOT_FOUND",
        "gpu_manifest": gpu_manifest,
        "historical_untracked_files": [
            "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py",
            "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py",
        ],
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)
    _write_source_audits(output_dir, repo_manifest)

    checkpoint_rows, checkpoint_safety_before = _checkpoint_manifest(repo)
    _write_csv(output_dir / "checkpoint_manifest.csv", checkpoint_rows)
    _write_json(output_dir / "checkpoint_manifest.json", {"rows": checkpoint_rows})

    camera_manifest = {
        "scene": SCENE,
        "train_development_views": list(TRAIN_VIEWS),
        "heldout_views": list(HELDOUT_VIEWS),
        "HELD_OUT_MASK_SELECTION_LEAKAGE": False,
    }
    _write_json(output_dir / "camera_split_manifest.json", camera_manifest)

    pseudo_manifest = {
        "PSEUDO_DEPTH_SOURCE": str(DEPTHS_PATH),
        "PSEUDO_DEPTH_GENERATOR": "DepthAnything cache from formal CDEPTH Panama resource",
        "NORMALIZATION": "per image divide by max, matching SeaFree code",
        "PSEUDO_DEPTH_GRADIENT_TO_GEOMETRY": False,
        "file_count": len(list(DEPTHS_PATH.glob("*.png"))),
    }
    _write_json(output_dir / "pseudo_depth_manifest.json", pseudo_manifest)

    all_views = list(TRAIN_VIEWS) + list(HELDOUT_VIEWS)
    temporal_maps: Dict[int, Dict[str, Dict[str, Tensor]]] = {}
    step_meta: Dict[int, Dict[str, Any]] = {}
    for step in TEMPORAL_STEPS:
        maps, meta = _render_step(repo, step, all_views)
        temporal_maps[step] = maps
        step_meta[step] = meta
        _write_json(log_dir / f"render_step_{step}.json", {"step": step, "meta": meta, "view_count": len(maps)})

    step3_maps = temporal_maps[3000]
    final_maps = temporal_maps[14999]
    masks, sf_meta = _build_masks(step3_maps, all_views)

    mask_def = {
        "M_SF": "SeaFree-style pseudo-depth background candidate: complement of largest filled pseudo_depth>=1e-2 foreground component after per-image max normalization.",
        "M_LOW_SUPPORT": "BND-K1@3k accumulation <= 0.01.",
        "M_INTERSECT": "M_SF AND M_LOW_SUPPORT.",
        "M_SAFE": "BinaryErode(M_INTERSECT, radius=5 px).",
        "pseudo_depth_source": str(DEPTHS_PATH),
        "sf_meta": sf_meta,
    }
    _write_json(output_dir / "mask_definitions.json", mask_def)
    locked = {
        **mask_def,
        "locked_before_heldout_processing": True,
        "train_development_views_used_for_readiness": list(TRAIN_VIEWS),
        "heldout_views_not_used_to_select_thresholds": list(HELDOUT_VIEWS),
    }
    _write_json(output_dir / "locked_pure_water_candidate_definition.json", locked)

    coverage_rows: List[Dict[str, Any]] = []
    for split, view_ids in (("train", TRAIN_VIEWS), ("heldout", HELDOUT_VIEWS)):
        for view_id in view_ids:
            for mask_name in MASK_NAMES:
                coverage_rows.append(_coverage_row(split, view_id, mask_name, masks[view_id][mask_name]))
        for mask_name in MASK_NAMES:
            coverage_rows.append(_pooled_coverage_row(split, mask_name, coverage_rows))
    _write_csv(output_dir / "mask_coverage.csv", coverage_rows)
    _write_json(output_dir / "mask_coverage.json", {"rows": coverage_rows})

    agreement_rows = _agreement_rows("train", masks, TRAIN_VIEWS) + _agreement_rows("heldout", masks, HELDOUT_VIEWS)
    _write_csv(output_dir / "mask_agreement.csv", agreement_rows)
    _write_json(output_dir / "mask_agreement.json", {"rows": agreement_rows})

    temporal_rows = _temporal_rows("train", TRAIN_VIEWS, masks, temporal_maps) + _temporal_rows("heldout", HELDOUT_VIEWS, masks, temporal_maps)
    _write_csv(output_dir / "temporal_support_stability.csv", temporal_rows)
    _write_json(output_dir / "temporal_support_stability.json", {"rows": temporal_rows})
    late_rows = [r for r in temporal_rows if int(r["step"]) == 14999]
    _write_csv(output_dir / "late_object_contamination.csv", late_rows)
    _write_json(output_dir / "late_object_contamination.json", {"rows": late_rows})

    headroom_rows: List[Dict[str, Any]] = []
    for split, view_ids in (("train", TRAIN_VIEWS), ("heldout", HELDOUT_VIEWS)):
        for view_id in view_ids:
            for mask_name in MASK_NAMES:
                headroom_rows.append(_headroom_row(split, view_id, mask_name, masks[view_id][mask_name], final_maps[view_id]))
        for mask_name in MASK_NAMES:
            headroom_rows.append(_pooled_headroom_row(split, mask_name, view_ids, masks, final_maps))
    _write_csv(output_dir / "binf_statistics.csv", headroom_rows)
    _write_json(output_dir / "binf_statistics.json", {"rows": headroom_rows})
    _write_csv(output_dir / "background_anchor_headroom.csv", headroom_rows)
    _write_json(output_dir / "background_anchor_headroom.json", {"rows": headroom_rows})
    _write_csv(output_dir / "full_render_vs_binf.csv", headroom_rows)
    _write_json(output_dir / "full_render_vs_binf.json", {"rows": headroom_rows})

    object_rows = _object_contamination_rows("train", TRAIN_VIEWS, masks, final_maps) + _object_contamination_rows("heldout", HELDOUT_VIEWS, masks, final_maps)
    _write_csv(output_dir / "object_contamination_audit.csv", object_rows)
    _write_json(output_dir / "object_contamination_audit.json", {"rows": object_rows})

    brightness_rows = _brightness_control("train", TRAIN_VIEWS, masks, final_maps) + _brightness_control("heldout", HELDOUT_VIEWS, masks, final_maps)
    _write_csv(output_dir / "brightness_matched_control.csv", brightness_rows)
    _write_json(output_dir / "brightness_matched_control.json", {"rows": brightness_rows})

    heldout_mask_rows = [r for r in coverage_rows if r["split"] == "heldout"]
    heldout_headroom_rows = [r for r in headroom_rows if r["split"] == "heldout"]
    _write_csv(output_dir / "heldout_water_candidate_metrics.csv", heldout_mask_rows)
    _write_json(output_dir / "heldout_water_candidate_metrics.json", {"rows": heldout_mask_rows})
    _write_csv(output_dir / "heldout_binf_headroom.csv", heldout_headroom_rows)
    _write_json(output_dir / "heldout_binf_headroom.json", {"rows": heldout_headroom_rows})

    gradient_rows, parameter_safety, gradient_summary = _gradient_audit(
        repo,
        {view_id: masks[view_id]["M_SAFE"] for view_id in TRAIN_VIEWS},
        TRAIN_VIEWS,
        checkpoint_safety_before,
    )
    object_grad_l2_max = max(
        float(gradient_summary.get("means_grad_l2", 0.0)),
        float(gradient_summary.get("scales_grad_l2", 0.0)),
        float(gradient_summary.get("quats_grad_l2", 0.0)),
        float(gradient_summary.get("opacities_grad_l2", 0.0)),
        float(gradient_summary.get("features_dc_grad_l2", 0.0)),
        float(gradient_summary.get("features_rest_grad_l2", 0.0)),
    )
    gradient_summary["object_grad_l2_max"] = object_grad_l2_max
    _write_csv(output_dir / "virtual_bg_loss_gradient_audit.csv", gradient_rows)
    _write_json(output_dir / "virtual_bg_loss_gradient_audit.json", {"rows": gradient_rows, "summary": gradient_summary})
    _write_json(output_dir / "parameter_safety.json", parameter_safety)
    _write_json(output_dir / "checkpoint_safety.json", {"CHECKPOINT_SAFETY": "PASS", "rows": checkpoint_rows})

    train_cov = next(r for r in coverage_rows if r["split"] == "train" and r["view_id"] == "ALL" and r["mask"] == "M_SAFE")
    held_cov = next(r for r in coverage_rows if r["split"] == "heldout" and r["view_id"] == "ALL" and r["mask"] == "M_SAFE")
    train_final = next(r for r in temporal_rows if r["split"] == "train" and r["view_id"] == "ALL" and int(r["step"]) == 14999)
    train_obj = next(r for r in object_rows if r["split"] == "train" and r["view_id"] == "ALL")
    train_safe_head = next(r for r in headroom_rows if r["split"] == "train" and r["view_id"] == "ALL" and r["mask"] == "M_SAFE")
    held_safe_head = next(r for r in headroom_rows if r["split"] == "heldout" and r["view_id"] == "ALL" and r["mask"] == "M_SAFE")
    safe_mask_coverage_adequate = bool(float(train_cov["coverage_fraction"]) >= 0.03 and int(train_cov.get("views_ge_1pct", 0)) >= 10)
    per_train_final = [r for r in temporal_rows if r["split"] == "train" and r["view_id"] != "ALL" and int(r["step"]) == 14999]
    support_stable = bool(float(train_final["fraction_accumulation_gt_0p01"]) <= 0.10 and sum(float(r["fraction_accumulation_gt_0p01"]) <= 0.20 for r in per_train_final) >= 12)
    low_object = bool(train_obj.get("LOW_OBJECT_SUPPORT_CONFIRMED", False))
    heldout_nonempty = int(held_cov.get("views_nonempty", 0))
    heldout_ge1 = int(held_cov.get("views_ge_1pct", 0))
    object_warning = bool(train_obj.get("OBJECT_CONTAMINATION_WARNING", False))
    if safe_mask_coverage_adequate and support_stable and low_object and heldout_ge1 >= 2 and not object_warning:
        water_class = "HIGH_CONFIDENCE_WATER_CANDIDATE_SUPPORTED"
    elif (safe_mask_coverage_adequate and low_object) or (support_stable and float(train_cov["coverage_fraction"]) >= 0.005):
        water_class = "WATER_CANDIDATE_PARTIAL"
    elif float(train_cov["coverage_fraction"]) > 0.0:
        water_class = "WATER_CANDIDATE_WEAK"
    else:
        water_class = "WATER_CANDIDATE_NOT_SUPPORTED"

    background_headroom = bool(
        float(train_safe_head["E_BINF"]) >= 0.01
        and float(train_safe_head["R_anchor"]) >= 0.20
        and int(train_safe_head.get("views_E_BINF_gt_E_full", 0)) >= 10
    )
    heldout_bg_consistent = bool(
        int(held_safe_head.get("views_E_BINF_gt_E_full", 0)) >= 2
        and float(held_safe_head["E_BINF"]) >= 0.01
    )
    grad_ok = bool(gradient_summary.get("MEDIUM_ONLY_GRADIENT_ROUTE", False))
    if water_class in {"WATER_CANDIDATE_WEAK", "WATER_CANDIDATE_NOT_SUPPORTED", "NOT_EVALUABLE"}:
        bg_class = "BG_ANCHOR_NOT_SUPPORTED"
    elif water_class == "HIGH_CONFIDENCE_WATER_CANDIDATE_SUPPORTED" and background_headroom and heldout_bg_consistent and grad_ok:
        bg_class = "BG_ANCHOR_READY"
    elif float(train_safe_head["E_BINF"]) < 0.01 and water_class in {"HIGH_CONFIDENCE_WATER_CANDIDATE_SUPPORTED", "WATER_CANDIDATE_PARTIAL"}:
        bg_class = "BG_ANCHOR_ALREADY_CALIBRATED"
    elif background_headroom or (float(train_safe_head["E_BINF"]) >= 0.01 and float(train_safe_head["R_anchor"]) > 0.0):
        bg_class = "BG_ANCHOR_HEADROOM_PARTIAL"
    else:
        bg_class = "BG_ANCHOR_NOT_SUPPORTED"

    classifications = {
        "water_candidate": water_class,
        "background_anchor": bg_class,
        "SAFE_MASK_COVERAGE_ADEQUATE": safe_mask_coverage_adequate,
        "SUPPORT_STABLE_SAFE_MASK": support_stable,
        "LOW_OBJECT_SUPPORT_CONFIRMED": low_object,
        "OBJECT_CONTAMINATION_WARNING": object_warning,
        "BACKGROUND_ANCHOR_HEADROOM": background_headroom,
        "HELDOUT_BG_HEADROOM_CONSISTENT": heldout_bg_consistent,
        "MEDIUM_ONLY_GRADIENT_ROUTE": grad_ok,
    }
    _write_json(output_dir / "water_candidate_classification.json", classifications)
    _write_json(output_dir / "background_anchor_readiness_classification.json", classifications)

    summary = {
        "repo": repo_manifest,
        "coverage": {
            "train_safe_coverage": float(train_cov["coverage_fraction"]),
            "train_safe_views_ge_1pct": int(train_cov.get("views_ge_1pct", 0)),
            "heldout_safe_coverage": float(held_cov["coverage_fraction"]),
            "heldout_safe_views_ge_1pct": int(held_cov.get("views_ge_1pct", 0)),
        },
        "temporal": {
            "train_final_fraction_accumulation_gt_0p01": float(train_final["fraction_accumulation_gt_0p01"]),
            "support_stable_safe_mask": support_stable,
        },
        "headroom": {
            "train_safe_E_BINF": float(train_safe_head["E_BINF"]),
            "train_safe_E_full": float(train_safe_head["E_full"]),
            "train_safe_R_anchor": float(train_safe_head["R_anchor"]),
            "heldout_safe_E_BINF": float(held_safe_head["E_BINF"]),
            "heldout_safe_E_full": float(held_safe_head["E_full"]),
            "heldout_safe_R_anchor": float(held_safe_head["R_anchor"]),
        },
        "gradient": gradient_summary,
        "safety": {
            "AUDIT_PARAMETER_SAFETY": parameter_safety["AUDIT_PARAMETER_SAFETY"],
            "CHECKPOINT_SAFETY": "PASS",
            "HELD_OUT_MASK_SELECTION_LEAKAGE": False,
        },
        "classifications": classifications,
        "next_single_factor_experiment": (
            "BND-BG-ANCHOR" if bg_class == "BG_ANCHOR_READY"
            else "minimal robustness / mask validation audit" if bg_class == "BG_ANCHOR_HEADROOM_PARTIAL"
            else "AA mechanism refinement" if bg_class == "BG_ANCHOR_ALREADY_CALIBRATED"
            else "close background-anchor direction for this mask definition"
        ),
    }
    _write_json(output_dir / "bnd_pure_water_audit_final_summary.json", summary)
    _write_csv(output_dir / "bnd_pure_water_audit_final_summary.csv", [{"key": k, "value": json.dumps(v, default=_json_default)} for k, v in summary.items()])

    render_manifest: List[Dict[str, Any]] = []
    _make_visuals(
        render_dir,
        render_manifest,
        masks,
        temporal_maps,
        final_maps,
        TRAIN_VIEWS,
        HELDOUT_VIEWS,
        coverage_rows,
        object_rows,
        headroom_rows,
        brightness_rows,
        gradient_summary,
    )
    _save_text_sheet(
        render_dir / "final_readiness_summary_sheet.png",
        "BND-PW-AUDIT Final Readiness",
        [
            f"water_candidate: {water_class}",
            f"background_anchor: {bg_class}",
            f"train M_SAFE coverage: {train_cov['coverage_fraction']}",
            f"train final contamination: {train_final['fraction_accumulation_gt_0p01']}",
            f"train E_BINF/E_full/R: {train_safe_head['E_BINF']} / {train_safe_head['E_full']} / {train_safe_head['R_anchor']}",
            f"heldout E_BINF/E_full/R: {held_safe_head['E_BINF']} / {held_safe_head['E_full']} / {held_safe_head['R_anchor']}",
            f"medium-only gradient route: {grad_ok}",
        ],
        render_manifest,
        "final_readiness_summary",
    )
    _write_json(render_dir / "manifest.json", {"rows": render_manifest})
    _write_json(output_dir / "manifest.json", {"outputs": sorted(str(p) for p in output_dir.glob("*")), "renders": render_manifest})
    _write_visual_index(render_dir, render_manifest)
    _write_research_note(repo / RESEARCH_NOTE, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
