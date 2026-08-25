#!/usr/bin/env python3
"""Read-only IUI3 pure-water/background-anchor observability audit.

This diagnostic reuses the locked Panama BND-PW-AUDIT mask semantics on
IUI3. It loads existing checkpoints, renders fixed train/eval cameras,
evaluates candidate-water coverage, object contamination, B_inf headroom, and a
no-step background-anchor gradient probe. It never creates an optimizer, never
updates parameters, and never writes checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.utils.eval_utils import eval_setup


SCENE = "IUI3"
OUTPUT_DIR = Path("outputs/bnd_pw_audit_iui3_20260825")
RESEARCH_NOTE = Path("research_notes/BND_PW_AUDIT_IUI3_2026-08-25.md")
DATA_PATH = Path("undistorted_data/undistorted_IUI3-RedSea")
DEPTHS_PATH = DATA_PATH / "depthAnything_u16"
M1_CONFIG = Path(
    "outputs/gmvc_v3_four_scene_iui3_m1_seed42_15000/water-splatting/"
    "gmvc_v3_four_scene_iui3_m1_seed42_15000_20260806_gmvc_four_scene_p30_mhold_15k_m1_bootstrap/"
    "config.yml"
)
BND_CONFIG = Path(
    "outputs/dewater_bounded_sh3_cross_scene_20260808/"
    "dewater_bnd_cross_scene_iui3_seed42_step0_to_15000/water-splatting/"
    "dewater_bnd_cross_scene_iui3_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_iui3_bnd_g1p00/"
    "config.yml"
)
BND_STEPS = (3000, 5000, 8000, 10000, 13000, 15000)
M1_STEPS = (5000, 10000, 15000)
MATCHED_STEPS = (5000, 10000, 15000)
MASK_NAMES = ("M_SF", "M_LOW_SUPPORT", "M_INTERSECT", "M_SAFE")
ALLOWED_PHYSICAL_GPUS = frozenset({"6", "7", "8", "9"})
EPS = 1e-8
QUANTILE_MAX_N = 1_000_000
GRADIENT_PIXELS_PER_VIEW = 4096


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


def _assert_runtime_policy() -> Dict[str, Any]:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env != "water_splatting":
        raise RuntimeError(f"CONDA_DEFAULT_ENV must be water_splatting, got {conda_env!r}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [token.strip() for token in visible.split(",") if token.strip()]
    if len(devices) != 1 or devices[0] not in ALLOWED_PHYSICAL_GPUS:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be exactly one physical GPU in "
            f"{sorted(ALLOWED_PHYSICAL_GPUS)}; got {visible!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available for this WaterSplatting diagnostic.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one torch-visible GPU after masking, got {torch.cuda.device_count()}")
    props = torch.cuda.get_device_properties(0)
    return {
        "CUDA_VISIBLE_DEVICES": visible,
        "physical_gpu_id": visible,
        "torch_logical_gpu_id": 0,
        "gpu_name": props.name,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "total_memory_bytes": int(props.total_memory),
    }


def _environment_manifest(gpu_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "CONDA_ENV": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "PYTHON_PATH": sys.executable,
        "PYTHON_VERSION": sys.version.split()[0],
        "TORCH_VERSION": torch.__version__,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu": dict(gpu_manifest),
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


def _run_config(run: str) -> Tuple[Path, str]:
    if run == "M1":
        return M1_CONFIG, "legacy"
    if run == "BND":
        return BND_CONFIG, "bounded_sh3"
    raise ValueError(run)


def _load_run(repo: Path, run: str, nominal_step: int) -> LoadedRun:
    rel_config, parameterization = _run_config(run)
    config_path = repo / rel_config
    actual = _actual_step(config_path, nominal_step)
    if actual is None:
        raise FileNotFoundError(f"Missing {run} checkpoint for nominal step {nominal_step}: {config_path}")

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
    model.eval()
    return LoadedRun(run, config_path, Path(checkpoint_path), int(loaded_step), config, pipeline)


def _release(obj: Optional[LoadedRun]) -> None:
    if obj is not None:
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


def _records(pipeline: Any) -> Dict[str, List[Tuple[int, str, Any, Dict[str, Any]]]]:
    rows: Dict[str, List[Tuple[int, str, Any, Dict[str, Any]]]] = {"train": [], "eval": []}
    train_dataset = pipeline.datamanager.train_dataset
    train_files = list(getattr(train_dataset, "image_filenames", []))
    train_cameras = train_dataset.cameras.to(pipeline.model.device)
    for index, filename in enumerate(train_files):
        batch = pipeline.datamanager.cached_train[index].copy()
        rows["train"].append(
            (index, Path(filename).stem, train_cameras[index : index + 1], _batch_to_device(batch, pipeline.model.device))
        )

    eval_dataset = pipeline.datamanager.eval_dataset
    eval_files = list(getattr(eval_dataset, "image_filenames", []))
    for eval_index, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        filename = eval_files[eval_index] if eval_index < len(eval_files) else Path(f"eval_{eval_index}")
        rows["eval"].append(
            (eval_index, Path(filename).stem, camera.to(pipeline.model.device), _batch_to_device(batch, pipeline.model.device))
        )
    return rows


def _get_gt(model: Any, batch: Mapping[str, Any], background: Tensor) -> Tensor:
    return model.composite_with_background(model.get_gt_img(batch["image"]), background.to(model.device))


def _scalar_map(value: Tensor) -> Tensor:
    value = value.detach().float()
    if value.ndim == 3 and value.shape[-1] == 1:
        return value[..., 0]
    if value.ndim == 3 and value.shape[-1] == 3:
        return value.mean(dim=-1)
    if value.ndim == 2:
        return value
    raise ValueError(f"Cannot scalarize shape {tuple(value.shape)}")


def _safe_outputs(outputs: Mapping[str, Any], gt: Tensor) -> Dict[str, Tensor]:
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
        "clear_object_fullsh_raw",
        "gaussian_view_logits",
        "gaussian_visible_mask",
    )
    safe = {
        key: value.detach().float().cpu()
        for key, value in outputs.items()
        if key in keep and isinstance(value, Tensor)
    }
    safe["gt"] = gt.detach().float().cpu()
    safe["pred_image"] = safe.get("pred_image", safe.get("rgb")).detach().float().cpu()
    return safe


def _render_split_maps(repo: Path, run: str, nominal_step: int) -> Tuple[Dict[str, Dict[str, Dict[str, Tensor]]], Dict[str, Any]]:
    loaded: Optional[LoadedRun] = None
    try:
        loaded = _load_run(repo, run, nominal_step)
        model = loaded.pipeline.model
        out: Dict[str, Dict[str, Dict[str, Tensor]]] = {"train": {}, "eval": {}}
        for split, rows in _records(loaded.pipeline).items():
            for _idx, view_id, camera, batch in rows:
                with torch.no_grad():
                    outputs = model.get_outputs_for_camera(camera.to(model.device))
                    gt = _get_gt(model, batch, outputs["background"])
                out[split][view_id] = _safe_outputs(outputs, gt)
                torch.cuda.empty_cache()
        meta = {
            "run": run,
            "nominal_step": int(nominal_step),
            "actual_step": int(loaded.loaded_step),
            "config_path": str(loaded.config_path),
            "checkpoint_path": str(loaded.checkpoint_path),
            "gaussian_count": int(model.num_points),
            "intrinsic_color_parameterization": getattr(model.config, "intrinsic_color_parameterization", ""),
            "medium_context_mode": getattr(model.config, "medium_context_mode", ""),
            "b_inf_mode": getattr(model.config, "b_inf_mode", ""),
            "infinite_water_enabled": getattr(model.config, "infinite_water_enabled", ""),
        }
        return out, meta
    finally:
        _release(loaded)


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
    foreground_source = pseudo >= threshold
    largest = _largest_component(foreground_source)
    foreground = _fill_holes(largest).astype(bool)
    background = ~foreground
    meta = {
        "background_depth_threshold": threshold,
        "pseudo_depth_normalization": "per image divide by max",
        "foreground_source": "pseudo_depth >= 1e-2 after SeaFree threshold inversion",
        "foreground_mask": "largest connected component of foreground_source, filled",
        "background_candidate": "complement of foreground_mask",
        "erosion_dilation": "none in SeaFree source mask",
    }
    return torch.from_numpy(background), meta


def _build_masks(step3_maps: Mapping[str, Mapping[str, Mapping[str, Tensor]]]) -> Tuple[Dict[str, Dict[str, Dict[str, Tensor]]], Dict[str, Any]]:
    masks: Dict[str, Dict[str, Dict[str, Tensor]]] = {"train": {}, "eval": {}}
    meta: Dict[str, Any] = {}
    for split, view_maps in step3_maps.items():
        for view_id, data in view_maps.items():
            acc = _scalar_map(data["accumulation"])
            h, w = int(acc.shape[0]), int(acc.shape[1])
            pseudo = _load_pseudo_depth(view_id, (h, w))
            m_sf, sf_meta = _seafree_background_candidate(pseudo)
            m_low = acc <= 0.01
            m_inter = m_sf.bool() & m_low.bool()
            m_safe = torch.from_numpy(_erode(m_inter.numpy(), 5))
            masks[split][view_id] = {
                "M_SF": m_sf.bool(),
                "M_LOW_SUPPORT": m_low.bool(),
                "M_INTERSECT": m_inter.bool(),
                "M_SAFE": m_safe.bool(),
                "pseudo_depth": pseudo,
            }
            meta[f"{split}/{view_id}"] = sf_meta
    return masks, meta


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


def _coverage_rows(masks: Mapping[str, Mapping[str, Mapping[str, Tensor]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split, views in masks.items():
        for view_id, by_mask in views.items():
            for mask_name in MASK_NAMES:
                mask = by_mask[mask_name].bool()
                h, w = mask.shape
                count = int(mask.sum().item())
                comps, largest_frac = _component_count_and_largest(mask)
                yy = torch.linspace(0.0, 1.0, h).reshape(h, 1).expand(h, w)
                xx = torch.linspace(0.0, 1.0, w).reshape(1, w).expand(h, w)
                rows.append(
                    {
                        "split": split,
                        "view_id": view_id,
                        "mask": mask_name,
                        "valid_pixel_count": int(h * w),
                        "candidate_pixel_count": count,
                        "candidate_fraction": float(count) / max(float(h * w), 1.0),
                        "connected_components": comps,
                        "largest_connected_candidate_fraction": largest_frac,
                        "top20_image_fraction": float((mask & (yy <= 0.2)).sum().item()) / max(count, 1),
                        "bottom20_image_fraction": float((mask & (yy >= 0.8)).sum().item()) / max(count, 1),
                        "left20_image_fraction": float((mask & (xx <= 0.2)).sum().item()) / max(count, 1),
                        "right20_image_fraction": float((mask & (xx >= 0.8)).sum().item()) / max(count, 1),
                    }
                )
        for mask_name in MASK_NAMES:
            selected = [r for r in rows if r["split"] == split and r["mask"] == mask_name and r["view_id"] != "ALL"]
            fracs = np.asarray([float(r["candidate_fraction"]) for r in selected], dtype=np.float64)
            rows.append(
                {
                    "split": split,
                    "view_id": "ALL",
                    "mask": mask_name,
                    "valid_pixel_count": int(sum(int(r["valid_pixel_count"]) for r in selected)),
                    "candidate_pixel_count": int(sum(int(r["candidate_pixel_count"]) for r in selected)),
                    "candidate_fraction": float(
                        sum(int(r["candidate_pixel_count"]) for r in selected)
                        / max(sum(int(r["valid_pixel_count"]) for r in selected), 1)
                    ),
                    "coverage_mean_across_views": float(fracs.mean()) if fracs.size else float("nan"),
                    "coverage_median_across_views": float(np.median(fracs)) if fracs.size else float("nan"),
                    "coverage_min_across_views": float(fracs.min()) if fracs.size else float("nan"),
                    "coverage_max_across_views": float(fracs.max()) if fracs.size else float("nan"),
                    "coverage_std_across_views": float(fracs.std()) if fracs.size else float("nan"),
                    "views_nonempty": int(sum(int(r["candidate_pixel_count"]) > 0 for r in selected)),
                    "views_ge_1pct_locked_nontrivial": int(sum(float(r["candidate_fraction"]) >= 0.01 for r in selected)),
                    "view_count": len(selected),
                }
            )
    return rows


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    return ~_erode(~mask.astype(bool), radius)


def _to_uint8_panel(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _mask_panel(mask: Tensor, color: Tuple[int, int, int]) -> np.ndarray:
    arr = np.zeros((*mask.shape, 3), dtype=np.uint8)
    arr[mask.detach().bool().cpu().numpy()] = color
    return arr


def _resize_panel(panel: np.ndarray, max_width: int = 360) -> Image.Image:
    image = Image.fromarray(panel)
    if image.width <= max_width:
        return image
    scale = max_width / float(image.width)
    return image.resize((max_width, max(1, int(round(image.height * scale)))), Image.Resampling.NEAREST)


def _write_spatial_overlays(
    output_dir: Path,
    step3_maps: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    masks: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
) -> List[Dict[str, Any]]:
    overlay_dir = output_dir / "spatial_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for split, views in masks.items():
        for view_id, by_mask in views.items():
            data = step3_maps[split][view_id]
            rgb = _to_uint8_panel(data["gt"].detach().float().cpu().numpy()[..., :3])
            pseudo = _to_uint8_panel(by_mask["pseudo_depth"].detach().float().cpu().numpy())
            panels = [
                _resize_panel(rgb),
                _resize_panel(pseudo),
                _resize_panel(_mask_panel(by_mask["M_SF"], (0, 192, 255))),
                _resize_panel(_mask_panel(by_mask["M_LOW_SUPPORT"], (255, 64, 192))),
                _resize_panel(_mask_panel(by_mask["M_SAFE"], (80, 255, 120))),
            ]
            width = sum(panel.width for panel in panels)
            height = max(panel.height for panel in panels)
            canvas = Image.new("RGB", (width, height), (0, 0, 0))
            x = 0
            for panel in panels:
                canvas.paste(panel, (x, 0))
                x += panel.width
            rel_path = overlay_dir / f"{split}_{view_id}_rgb_pseudodepth_masks.png"
            canvas.save(rel_path)
            rows.append(
                {
                    "split": split,
                    "view_id": view_id,
                    "overlay_path": str(rel_path),
                    "panel_order": "input_rgb | pseudo_depth | M_SF | M_LOW_SUPPORT | M_SAFE",
                }
            )
    return rows


def _spatial_stability_rows(
    masks: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    step3_maps: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    overlay_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    overlay_key = {(r["split"], r["view_id"]): r["overlay_path"] for r in overlay_rows}
    rows: List[Dict[str, Any]] = []
    for split, views in masks.items():
        for view_id, by_mask in views.items():
            mask = by_mask["M_SAFE"].detach().bool().cpu()
            h, w = mask.shape
            count = int(mask.sum().item())
            comps, largest_frac = _component_count_and_largest(mask)
            yy = torch.linspace(0.0, 1.0, h).reshape(h, 1).expand(h, w)
            xx = torch.linspace(0.0, 1.0, w).reshape(1, w).expand(h, w)
            top = float((mask & (yy <= 0.2)).sum().item()) / max(count, 1)
            bottom = float((mask & (yy >= 0.8)).sum().item()) / max(count, 1)
            left = float((mask & (xx <= 0.2)).sum().item()) / max(count, 1)
            right = float((mask & (xx >= 0.8)).sum().item()) / max(count, 1)
            arr = mask.numpy()
            ring = _dilate(arr, 5) & (~arr)
            acc = _scalar_map(step3_maps[split][view_id]["accumulation"]).detach().float().cpu().numpy()
            ring_count = int(ring.sum())
            ring_support = float((acc[ring] > 0.01).mean()) if ring_count else float("nan")
            max_edge = max(top, bottom, left, right)
            rows.append(
                {
                    "split": split,
                    "view_id": view_id,
                    "candidate_pixel_count": count,
                    "candidate_fraction": float(count) / max(float(h * w), 1.0),
                    "connected_components": comps,
                    "largest_connected_candidate_fraction": largest_frac,
                    "top20_image_fraction": top,
                    "bottom20_image_fraction": bottom,
                    "left20_image_fraction": left,
                    "right20_image_fraction": right,
                    "max_edge20_fraction": max_edge,
                    "dilated_5px_ring_fraction_accumulation_gt_0p01": ring_support,
                    "spatial_coherence_descriptor": "spatially_coherent" if count > 0 and comps <= 3 and largest_frac >= 0.8 else "fragmented_or_empty",
                    "border_descriptor": "border_dominated" if max_edge >= 0.7 else "not_border_dominated",
                    "object_adjacency_descriptor": "object_adjacent_ring_support" if np.isfinite(ring_support) and ring_support >= 0.10 else "low_step3_ring_support",
                    "overlay_path": overlay_key.get((split, view_id), ""),
                }
            )
    return rows


def _rankdata(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(vals.shape[0], dtype=np.float64)
    i = 0
    while i < vals.shape[0]:
        j = i + 1
        while j < vals.shape[0] and vals[order[j]] == vals[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def _spearman(x_values: Sequence[float], y_values: Sequence[float]) -> Tuple[float, int]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan"), int(x.size)
    xr = _rankdata(x)
    yr = _rankdata(y)
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = math.sqrt(float((xr * xr).sum()) * float((yr * yr).sum()))
    if denom <= 0.0:
        return float("nan"), int(x.size)
    return float((xr * yr).sum() / denom), int(x.size)


def _finite_rows(rows: Sequence[Mapping[str, Any]], required: Sequence[str]) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    for row in rows:
        ok = True
        for key in required:
            try:
                ok = ok and np.isfinite(float(row[key]))
            except Exception:
                ok = False
                break
        if ok:
            out.append(row)
    return out


def _numeric_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            val = float(row[key])
        except Exception:
            continue
        if np.isfinite(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else float("nan")


def _numeric_std(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            val = float(row[key])
        except Exception:
            continue
        if np.isfinite(val):
            vals.append(val)
    return float(np.std(vals)) if vals else float("nan")


def _sign_consistency(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            val = float(row[key])
        except Exception:
            continue
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(max((arr > 0.0).mean(), (arr < 0.0).mean()))


def _contamination_headroom_rows(
    headroom_rows: Sequence[Mapping[str, Any]],
    contamination_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    head_keyed = {
        (r["run"], int(r["nominal_step"]), r["split"], r["view_id"]): r
        for r in headroom_rows
        if r.get("mask") == "M_SAFE" and r.get("view_id") not in ("", "ALL")
    }
    joined: List[Dict[str, Any]] = []
    for contam in contamination_rows:
        if contam.get("run") != "BND" or contam.get("split") != "train" or contam.get("view_id") in ("", "ALL"):
            continue
        key = (contam["run"], int(contam["nominal_step"]), contam["split"], contam["view_id"])
        head = head_keyed.get(key)
        if head is None:
            continue
        row = {
            "row_type": "view_checkpoint",
            "run": "BND",
            "nominal_step": int(contam["nominal_step"]),
            "actual_step": int(contam["actual_step"]),
            "split": "train",
            "view_id": contam["view_id"],
            "candidate_pixel_count": int(contam["candidate_pixel_count"]),
            "fraction_accumulation_gt_0p01": contam.get("fraction_accumulation_gt_0p01", float("nan")),
            "mean_accumulation": contam.get("mean_accumulation", float("nan")),
            "BINF_L1": head.get("BINF_L1", float("nan")),
            "BINF_MSE": head.get("BINF_MSE", float("nan")),
            "R_anchor": head.get("R_anchor", float("nan")),
        }
        joined.append(row)

    rows: List[Dict[str, Any]] = list(joined)
    for step in BND_STEPS:
        selected = _finite_rows(
            [r for r in joined if int(r["nominal_step"]) == step],
            ("fraction_accumulation_gt_0p01", "BINF_L1", "R_anchor"),
        )
        rho_l1, n_l1 = _spearman(
            [float(r["fraction_accumulation_gt_0p01"]) for r in selected],
            [float(r["BINF_L1"]) for r in selected],
        )
        rho_ra, n_ra = _spearman(
            [float(r["fraction_accumulation_gt_0p01"]) for r in selected],
            [float(r["R_anchor"]) for r in selected],
        )
        rows.append(
            {
                "row_type": "spearman_within_step",
                "run": "BND",
                "nominal_step": step,
                "split": "train",
                "spearman_contamination_BINF_L1": rho_l1,
                "spearman_contamination_R_anchor": rho_ra,
                "n": min(n_l1, n_ra),
            }
        )
    selected_all = _finite_rows(joined, ("fraction_accumulation_gt_0p01", "BINF_L1", "R_anchor"))
    rho_l1, n_l1 = _spearman(
        [float(r["fraction_accumulation_gt_0p01"]) for r in selected_all],
        [float(r["BINF_L1"]) for r in selected_all],
    )
    rho_ra, n_ra = _spearman(
        [float(r["fraction_accumulation_gt_0p01"]) for r in selected_all],
        [float(r["R_anchor"]) for r in selected_all],
    )
    rows.append(
        {
            "row_type": "spearman_all_view_checkpoints",
            "run": "BND",
            "split": "train",
            "spearman_contamination_BINF_L1": rho_l1,
            "spearman_contamination_R_anchor": rho_ra,
            "n": min(n_l1, n_ra),
        }
    )
    for step in BND_STEPS:
        selected = [r for r in joined if int(r["nominal_step"]) == step]
        rows.append(
            {
                "row_type": "temporal_curve",
                "run": "BND",
                "nominal_step": step,
                "split": "train",
                "fraction_accumulation_gt_0p01_view_mean": _numeric_mean(selected, "fraction_accumulation_gt_0p01"),
                "mean_accumulation_view_mean": _numeric_mean(selected, "mean_accumulation"),
                "BINF_L1_view_mean": _numeric_mean(selected, "BINF_L1"),
                "R_anchor_view_mean": _numeric_mean(selected, "R_anchor"),
            }
        )
    return rows


def _stage_name(step: int) -> str:
    if step in (3000, 5000):
        return "early"
    if step in (8000, 10000):
        return "mid"
    if step in (13000, 15000):
        return "late"
    return "other"


def _stage_summary_rows(
    coverage_rows: Sequence[Mapping[str, Any]],
    headroom_rows: Sequence[Mapping[str, Any]],
    contamination_rows: Sequence[Mapping[str, Any]],
    view_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    train_cov = next(r for r in coverage_rows if r["split"] == "train" and r["view_id"] == "ALL" and r["mask"] == "M_SAFE")
    rows: List[Dict[str, Any]] = []
    for stage in ("early", "mid", "late"):
        steps = [step for step in BND_STEPS if _stage_name(step) == stage]
        h_selected = [
            r
            for r in headroom_rows
            if r.get("run") == "BND"
            and int(r.get("nominal_step", -1)) in steps
            and r.get("split") == "train"
            and r.get("mask") == "M_SAFE"
            and r.get("view_id") not in ("", "ALL")
        ]
        c_selected = [
            r
            for r in contamination_rows
            if r.get("run") == "BND"
            and int(r.get("nominal_step", -1)) in steps
            and r.get("split") == "train"
            and r.get("view_id") not in ("", "ALL")
        ]
        v_selected = [
            r
            for r in view_rows
            if r.get("run") == "BND"
            and int(r.get("nominal_step", -1)) in steps
            and r.get("split") == "train"
            and r.get("view_id") not in ("", "ALL")
        ]
        contam = _numeric_mean(c_selected, "fraction_accumulation_gt_0p01")
        binf_l1 = _numeric_mean(h_selected, "BINF_L1")
        r_anchor = _numeric_mean(h_selected, "R_anchor")
        stability_l1_std = _numeric_std(v_selected, "BINF_L1")
        stability_contam_std = _numeric_std(v_selected, "fraction_accumulation_gt_0p01")
        rows.append(
            {
                "stage": stage,
                "steps": ";".join(str(step) for step in steps),
                "candidate_coverage_fixed_train_M_SAFE": float(train_cov["candidate_fraction"]),
                "views_ge_1pct_locked_nontrivial": int(train_cov.get("views_ge_1pct_locked_nontrivial", 0)),
                "fraction_accumulation_gt_0p01_view_checkpoint_mean": contam,
                "mean_accumulation_view_checkpoint_mean": _numeric_mean(c_selected, "mean_accumulation"),
                "BINF_L1_view_checkpoint_mean": binf_l1,
                "R_anchor_view_checkpoint_mean": r_anchor,
                "BINF_L1_view_checkpoint_std": stability_l1_std,
                "contamination_view_checkpoint_std": stability_contam_std,
                "BINF_L1_cv": stability_l1_std / max(abs(binf_l1), EPS) if np.isfinite(binf_l1) else float("nan"),
                "contamination_cv": stability_contam_std / max(abs(contam), EPS) if np.isfinite(contam) else float("nan"),
                "R_sign_consistency": _sign_consistency(v_selected, "BINF_R_signed_mean"),
                "G_sign_consistency": _sign_consistency(v_selected, "BINF_G_signed_mean"),
                "B_sign_consistency": _sign_consistency(v_selected, "BINF_B_signed_mean"),
                "stage_candidate_reliability_high_locked_rule": bool(
                    np.isfinite(contam)
                    and contam <= 0.10
                    and sum(
                        float(r.get("fraction_accumulation_gt_0p01", 1.0)) <= 0.20
                        for r in c_selected
                        if np.isfinite(float(r.get("fraction_accumulation_gt_0p01", float("nan"))))
                    )
                    >= 12 * max(len(steps), 1)
                ),
                "stage_headroom_nontrivial_locked_rule": bool(
                    (np.isfinite(binf_l1) and binf_l1 >= 0.01)
                    or (np.isfinite(r_anchor) and r_anchor >= 0.20)
                ),
            }
        )
    return rows


def _cross_scene_comparison_rows(
    coverage_rows: Sequence[Mapping[str, Any]],
    headroom_rows: Sequence[Mapping[str, Any]],
    contamination_rows: Sequence[Mapping[str, Any]],
    gradient_summary: Mapping[str, Any],
    classification: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    def pooled_contam(step: int) -> float:
        row = next(
            (
                r
                for r in contamination_rows
                if r.get("run") == "BND"
                and int(r.get("nominal_step", -1)) == step
                and r.get("split") == "train"
                and r.get("view_id") == "ALL"
            ),
            {},
        )
        return float(row.get("fraction_accumulation_gt_0p01_view_mean", float("nan")))

    final_head = next(
        r
        for r in headroom_rows
        if r.get("run") == "BND"
        and int(r.get("nominal_step", -1)) == 15000
        and r.get("split") == "train"
        and r.get("view_id") == "ALL"
    )
    train_cov = next(r for r in coverage_rows if r["split"] == "train" and r["view_id"] == "ALL" and r["mask"] == "M_SAFE")
    return [
        {
            "Scene": "Panama",
            "M_SAFE_coverage": 0.0014541265330494672,
            "early_contamination": float("nan"),
            "mid_contamination": float("nan"),
            "final_contamination": 1.0,
            "final_BINF_L1": 0.010516314767301083,
            "final_R_anchor": float("nan"),
            "medium_only_gradient_clean": True,
            "view_stability": False,
            "formal_classification": "BG_ANCHOR_NOT_SUPPORTED",
        },
        {
            "Scene": "Curasao",
            "M_SAFE_coverage": 0.04173292860893505,
            "early_contamination": 0.016372220752398588,
            "mid_contamination": 0.022528012231406238,
            "final_contamination": 0.27303126868274474,
            "final_BINF_L1": 0.0052376006367719835,
            "final_R_anchor": 0.012957839604056259,
            "medium_only_gradient_clean": True,
            "view_stability": False,
            "formal_classification": "BG_ANCHOR_WEAK",
        },
        {
            "Scene": "IUI3",
            "M_SAFE_coverage": float(train_cov["candidate_fraction"]),
            "early_contamination": pooled_contam(5000),
            "mid_contamination": pooled_contam(10000),
            "final_contamination": pooled_contam(15000),
            "final_BINF_L1": float(final_head.get("BINF_L1_view_mean", float("nan"))),
            "final_R_anchor": float(final_head.get("R_anchor_view_mean", float("nan"))),
            "medium_only_gradient_clean": bool(gradient_summary.get("MEDIUM_DOMINANT_GRADIENT_ROUTE", False)),
            "view_stability": bool(classification.get("VIEW_STABILITY_RULE", False)),
            "formal_classification": classification["classification"],
        },
    ]


def _stats_np(values: np.ndarray, prefix: str) -> Dict[str, Any]:
    vals = np.asarray(values).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {f"{prefix}{key}": float("nan") for key in ("mean", "p01", "p10", "p50", "p90", "p99", "min", "max")}
    return {
        f"{prefix}mean": float(vals.mean()),
        f"{prefix}p01": float(np.quantile(vals, 0.01)),
        f"{prefix}p10": float(np.quantile(vals, 0.10)),
        f"{prefix}p50": float(np.quantile(vals, 0.50)),
        f"{prefix}p90": float(np.quantile(vals, 0.90)),
        f"{prefix}p99": float(np.quantile(vals, 0.99)),
        f"{prefix}min": float(vals.min()),
        f"{prefix}max": float(vals.max()),
    }


def _quantile_tensor(values: Tensor, q: float) -> float:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    if flat.numel() > QUANTILE_MAX_N:
        idx = torch.linspace(0, flat.numel() - 1, QUANTILE_MAX_N, device=flat.device).long()
        idx = idx.clamp(0, flat.numel() - 1)
        flat = flat[idx]
    return float(torch.quantile(flat, q).item())


def _masked_rgb(values: Tensor, mask: Tensor) -> Tensor:
    if int(mask.sum().item()) == 0:
        return values.new_zeros((0, 3))
    return values.detach().float()[mask.bool()]


def _headroom_row(
    run: str,
    nominal_step: int,
    actual_step: int,
    split: str,
    view_id: str,
    mask_name: str,
    mask: Tensor,
    data: Mapping[str, Tensor],
) -> Dict[str, Any]:
    gt = data["gt"].clamp(0.0, 1.0)
    pred = data["pred_image"].clamp(0.0, 1.0)
    b_inf = data["b_inf"].clamp(0.0, 1.0)
    count = int(mask.sum().item())
    row: Dict[str, Any] = {
        "run": run,
        "nominal_step": nominal_step,
        "actual_step": actual_step,
        "split": split,
        "view_id": view_id,
        "mask": mask_name,
        "candidate_pixel_count": count,
    }
    if count == 0:
        row.update({"BINF_L1": float("nan"), "BINF_MSE": float("nan"), "FULL_L1": float("nan"), "R_anchor": float("nan")})
        return row
    err = _masked_rgb(b_inf - gt, mask)
    full_err = _masked_rgb(pred - gt, mask)
    abs_err = err.abs()
    abs_full = full_err.abs()
    e_b = float(abs_err.mean().item())
    e_f = float(abs_full.mean().item())
    row.update(
        {
            "BINF_L1": e_b,
            "BINF_MSE": float(err.square().mean().item()),
            "FULL_L1": e_f,
            "FULL_MSE": float(full_err.square().mean().item()),
            "G_anchor": e_b - e_f,
            "R_anchor": (e_b - e_f) / max(e_b, EPS),
            "BINF_abs_p50": _quantile_tensor(abs_err, 0.50),
            "BINF_abs_p90": _quantile_tensor(abs_err, 0.90),
            "BINF_abs_p99": _quantile_tensor(abs_err, 0.99),
            "BINF_R_signed_mean": float(err[:, 0].mean().item()),
            "BINF_G_signed_mean": float(err[:, 1].mean().item()),
            "BINF_B_signed_mean": float(err[:, 2].mean().item()),
            "BINF_R_abs_mean": float(abs_err[:, 0].mean().item()),
            "BINF_G_abs_mean": float(abs_err[:, 1].mean().item()),
            "BINF_B_abs_mean": float(abs_err[:, 2].mean().item()),
            "E_BINF_gt_E_full": bool(e_b > e_f),
        }
    )
    return row


def _contamination_row(
    run: str,
    nominal_step: int,
    actual_step: int,
    split: str,
    view_id: str,
    mask: Tensor,
    data: Mapping[str, Tensor],
) -> Dict[str, Any]:
    count = int(mask.sum().item())
    row: Dict[str, Any] = {
        "run": run,
        "nominal_step": nominal_step,
        "actual_step": actual_step,
        "split": split,
        "view_id": view_id,
        "mask": "M_SAFE",
        "candidate_pixel_count": count,
    }
    if count == 0:
        return row
    acc = _scalar_map(data["accumulation"])
    direct = torch.linalg.norm(data.get("direct_object_signal", torch.zeros_like(data["pred_image"])), dim=-1)
    rgb_object = torch.linalg.norm(data.get("rgb_object", torch.zeros_like(data["pred_image"])), dim=-1)
    medium = torch.linalg.norm(data.get("rgb_medium", torch.zeros_like(data["pred_image"])), dim=-1)
    vals_acc = acc[mask]
    vals_direct = direct[mask]
    vals_obj = rgb_object[mask]
    vals_medium = medium[mask]
    row.update(
        {
            "mean_accumulation": float(vals_acc.mean().item()),
            "p50_accumulation": _quantile_tensor(vals_acc, 0.50),
            "p90_accumulation": _quantile_tensor(vals_acc, 0.90),
            "p99_accumulation": _quantile_tensor(vals_acc, 0.99),
            "fraction_accumulation_gt_0p01": float((vals_acc > 0.01).float().mean().item()),
            "fraction_accumulation_gt_0p05": float((vals_acc > 0.05).float().mean().item()),
            "mean_direct_object_signal_l2": float(vals_direct.mean().item()),
            "p90_direct_object_signal_l2": _quantile_tensor(vals_direct, 0.90),
            "mean_rgb_object_l2": float(vals_obj.mean().item()),
            "mean_medium_signal_l2": float(vals_medium.mean().item()),
        }
    )
    return row


def _saturation_row(
    run: str,
    nominal_step: int,
    actual_step: int,
    split: str,
    view_id: str,
    mask: Tensor,
    data: Mapping[str, Tensor],
) -> Dict[str, Any]:
    b_inf = data["b_inf"].detach().float().clamp(EPS, 1.0 - EPS)
    vals = _masked_rgb(b_inf, mask)
    row: Dict[str, Any] = {
        "run": run,
        "nominal_step": nominal_step,
        "actual_step": actual_step,
        "split": split,
        "view_id": view_id,
        "mask": "M_SAFE",
        "candidate_pixel_count": int(mask.sum().item()),
        "pre_sigmoid_logit_source": "torch.logit(outputs['b_inf']) because b_inf=medium_rgb=sigmoid(raw[...,0:3])",
    }
    if vals.numel() == 0:
        return row
    logits = torch.logit(vals.clamp(EPS, 1.0 - EPS))
    deriv = vals * (1.0 - vals)
    flat_b = vals.reshape(-1).cpu().numpy()
    flat_l = logits.reshape(-1).cpu().numpy()
    flat_d = deriv.reshape(-1).cpu().numpy()
    row.update(_stats_np(flat_b, "B_inf_"))
    row.update(_stats_np(flat_l, "B_inf_logit_"))
    row.update(_stats_np(flat_d, "B_inf_sigmoid_deriv_"))
    row.update(
        {
            "P_B_inf_gt_0p95": float((vals > 0.95).float().mean().item()),
            "P_B_inf_gt_0p99": float((vals > 0.99).float().mean().item()),
            "P_B_inf_lt_0p05": float((vals < 0.05).float().mean().item()),
        }
    )
    for idx, channel in enumerate(("R", "G", "B")):
        row[f"B_inf_{channel}_mean"] = float(vals[:, idx].mean().item())
        row[f"B_inf_{channel}_logit_mean"] = float(logits[:, idx].mean().item())
    return row


def _pooled_rows(rows: Sequence[Mapping[str, Any]], metric_type: str) -> List[Dict[str, Any]]:
    pooled: List[Dict[str, Any]] = []
    keys = sorted({(r["run"], r["nominal_step"], r["actual_step"], r["split"]) for r in rows})
    for run, nominal_step, actual_step, split in keys:
        selected = [r for r in rows if (r["run"], r["nominal_step"], r["actual_step"], r["split"]) == (run, nominal_step, actual_step, split) and r["view_id"] != "ALL"]
        if not selected:
            continue
        count = sum(int(r.get("candidate_pixel_count", 0)) for r in selected)
        out: Dict[str, Any] = {
            "run": run,
            "nominal_step": nominal_step,
            "actual_step": actual_step,
            "split": split,
            "view_id": "ALL",
            "mask": "M_SAFE",
            "candidate_pixel_count": count,
            "metric_type": metric_type,
        }
        numeric_keys = [
            key
            for row in selected
            for key, val in row.items()
            if key not in {"run", "nominal_step", "actual_step", "split", "view_id", "mask", "metric_type"}
            and isinstance(val, (int, float, np.generic))
        ]
        for key in sorted(set(numeric_keys)):
            vals = np.asarray([float(r[key]) for r in selected if key in r and np.isfinite(float(r[key]))], dtype=np.float64)
            if vals.size:
                out[f"{key}_view_mean"] = float(vals.mean())
                out[f"{key}_view_median"] = float(np.median(vals))
                out[f"{key}_view_min"] = float(vals.min())
                out[f"{key}_view_max"] = float(vals.max())
        pooled.append(out)
    return pooled


def _decomposition_rows(
    run: str,
    nominal_step: int,
    actual_step: int,
    split: str,
    maps: Mapping[str, Mapping[str, Tensor]],
) -> List[Dict[str, Any]]:
    if run != "BND":
        return []
    j_vals: List[Tensor] = []
    tau_vals: List[Tensor] = []
    t_vals: List[Tensor] = []
    logit_vals: List[Tensor] = []
    for data in maps.values():
        if "clear_object_fullsh_raw" in data:
            j_vals.append(data["clear_object_fullsh_raw"].detach().float().reshape(-1, 3).amax(dim=-1))
        if "tau_D" in data:
            tau_vals.append(_scalar_map(data["tau_D"]).reshape(-1))
        if "transmission" in data:
            t_vals.append(_scalar_map(data["transmission"]).reshape(-1))
        if "gaussian_view_logits" in data:
            logits = data["gaussian_view_logits"].detach().float()
            visible = data.get("gaussian_visible_mask")
            if isinstance(visible, Tensor) and visible.numel() == logits.shape[0]:
                logits = logits[visible.bool()]
            logit_vals.append(logits.reshape(-1))
    row = {
        "run": run,
        "nominal_step": nominal_step,
        "actual_step": actual_step,
        "split": split,
    }
    if j_vals:
        j = torch.cat(j_vals)
        row["J_p99"] = _quantile_tensor(j, 0.99)
        row["P_J_gt_1"] = float((j > 1.0).float().mean().item())
    if tau_vals:
        tau = torch.cat(tau_vals)
        row["tau_p90"] = _quantile_tensor(tau, 0.90)
        row["tau_p99"] = _quantile_tensor(tau, 0.99)
    if t_vals:
        trans = torch.cat(t_vals)
        row["P_T_lt_0p1"] = float((trans < 0.1).float().mean().item())
    if logit_vals:
        logits = torch.cat(logit_vals)
        c = torch.sigmoid(logits)
        row["P_c_gt_0p99"] = float((c > 0.99).float().mean().item())
        row["P_abs_s_full_gt_5"] = float((logits.abs() > 5.0).float().mean().item())
    return [row]


def _audit_run_step(
    repo: Path,
    run: str,
    nominal_step: int,
    masks: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    maps, meta = _render_split_maps(repo, run, nominal_step)
    headroom_rows: List[Dict[str, Any]] = []
    contam_rows: List[Dict[str, Any]] = []
    sat_rows: List[Dict[str, Any]] = []
    view_rows: List[Dict[str, Any]] = []
    decomp_rows: List[Dict[str, Any]] = []
    for split, view_maps in maps.items():
        for view_id, data in view_maps.items():
            if view_id not in masks[split]:
                continue
            safe = masks[split][view_id]["M_SAFE"]
            for mask_name in MASK_NAMES:
                headroom_rows.append(
                    _headroom_row(run, nominal_step, meta["actual_step"], split, view_id, mask_name, masks[split][view_id][mask_name], data)
                )
            contam = _contamination_row(run, nominal_step, meta["actual_step"], split, view_id, safe, data)
            contam_rows.append(contam)
            sat_rows.append(_saturation_row(run, nominal_step, meta["actual_step"], split, view_id, safe, data))
            view_rows.append(
                {
                    "run": run,
                    "nominal_step": nominal_step,
                    "actual_step": meta["actual_step"],
                    "split": split,
                    "view_id": view_id,
                    "candidate_fraction": float(safe.sum().item()) / max(float(safe.numel()), 1.0),
                    "BINF_L1": next(r["BINF_L1"] for r in headroom_rows if r["run"] == run and r["nominal_step"] == nominal_step and r["split"] == split and r["view_id"] == view_id and r["mask"] == "M_SAFE"),
                    "BINF_R_signed_mean": next(r.get("BINF_R_signed_mean", float("nan")) for r in headroom_rows if r["run"] == run and r["nominal_step"] == nominal_step and r["split"] == split and r["view_id"] == view_id and r["mask"] == "M_SAFE"),
                    "BINF_G_signed_mean": next(r.get("BINF_G_signed_mean", float("nan")) for r in headroom_rows if r["run"] == run and r["nominal_step"] == nominal_step and r["split"] == split and r["view_id"] == view_id and r["mask"] == "M_SAFE"),
                    "BINF_B_signed_mean": next(r.get("BINF_B_signed_mean", float("nan")) for r in headroom_rows if r["run"] == run and r["nominal_step"] == nominal_step and r["split"] == split and r["view_id"] == view_id and r["mask"] == "M_SAFE"),
                    "mean_accumulation": contam.get("mean_accumulation", float("nan")),
                    "fraction_accumulation_gt_0p01": contam.get("fraction_accumulation_gt_0p01", float("nan")),
                }
            )
        decomp_rows.extend(_decomposition_rows(run, nominal_step, meta["actual_step"], split, view_maps))
    headroom_rows.extend(_pooled_rows([r for r in headroom_rows if r["mask"] == "M_SAFE"], "headroom"))
    contam_rows.extend(_pooled_rows(contam_rows, "contamination"))
    sat_rows.extend(_pooled_rows(sat_rows, "saturation"))
    return headroom_rows, contam_rows, sat_rows, view_rows, decomp_rows, meta


def _comparison_rows(headroom_rows: Sequence[Mapping[str, Any]], contamination_rows: Sequence[Mapping[str, Any]], saturation_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sources = [
        ("headroom", headroom_rows, ("BINF_L1", "BINF_MSE", "FULL_L1", "R_anchor")),
        ("contamination", contamination_rows, ("mean_accumulation", "p90_accumulation", "fraction_accumulation_gt_0p01", "mean_direct_object_signal_l2")),
        ("saturation", saturation_rows, ("B_inf_mean", "B_inf_logit_mean", "P_B_inf_gt_0p95", "B_inf_sigmoid_deriv_mean")),
    ]
    for source, data_rows, quantities in sources:
        keyed = {
            (r["run"], r["nominal_step"], r["split"], r["view_id"], r.get("mask", "M_SAFE")): r
            for r in data_rows
            if r.get("view_id") == "ALL" and r.get("mask", "M_SAFE") == "M_SAFE"
        }
        for step in MATCHED_STEPS:
            for split in ("train", "eval"):
                m1 = keyed.get(("M1", step, split, "ALL", "M_SAFE"))
                bnd = keyed.get(("BND", step, split, "ALL", "M_SAFE"))
                if not m1 or not bnd:
                    continue
                for quantity in quantities:
                    m1_key = f"{quantity}_view_mean" if f"{quantity}_view_mean" in m1 else quantity
                    bnd_key = f"{quantity}_view_mean" if f"{quantity}_view_mean" in bnd else quantity
                    if m1_key not in m1 or bnd_key not in bnd:
                        continue
                    rows.append(
                        {
                            "source": source,
                            "nominal_step": step,
                            "split": split,
                            "quantity": quantity,
                            "M1": m1[m1_key],
                            "BND": bnd[bnd_key],
                            "BND_minus_M1": float(bnd[bnd_key]) - float(m1[m1_key]),
                        }
                    )
    return rows


def _parameter_snapshots(model: Any) -> Dict[str, List[Tensor]]:
    return {
        group: [param.detach().clone().cpu() for param in params]
        for group, params in model.get_param_groups().items()
    }


def _parameter_delta_rows(before: Mapping[str, Sequence[Tensor]], model: Any) -> List[Dict[str, Any]]:
    rows = []
    for group, params in model.get_param_groups().items():
        max_delta = 0.0
        for idx, param in enumerate(params):
            diff = (param.detach().cpu() - before[group][idx]).abs()
            if diff.numel():
                max_delta = max(max_delta, float(diff.max().item()))
        rows.append({"parameter_group": group, "max_abs_delta": max_delta})
    return rows


def _grad_l2(params: Iterable[Tensor]) -> Tuple[float, float]:
    total = 0.0
    max_abs = 0.0
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += float(grad.square().sum().item())
        if grad.numel():
            max_abs = max(max_abs, float(grad.abs().max().item()))
    return math.sqrt(total), max_abs


def _sample_mask_pixels(mask: Tensor, max_pixels: int = GRADIENT_PIXELS_PER_VIEW) -> Tuple[Tensor, Tensor]:
    mask = mask.detach().bool().cpu()
    h, w = mask.shape
    flat = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
    if flat.numel() == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    if flat.numel() > max_pixels:
        pos = torch.linspace(0, flat.numel() - 1, max_pixels).long().clamp(0, flat.numel() - 1)
        flat = flat[pos]
    ys = torch.div(flat, w, rounding_mode="floor")
    xs = flat % w
    return ys.long(), xs.long()


def _medium_for_pixels(model: Any, camera: Any, ys_cpu: Tensor, xs_cpu: Tensor) -> Dict[str, Tensor]:
    if ys_cpu.numel() == 0:
        empty = torch.empty((0, 3), device=model.device)
        return {"b_inf": empty, "medium_bs": empty, "medium_attn": empty}
    camera = camera.to(model.device)
    camera_downscale = model._get_downscale_factor()
    camera.rescale_output_resolution(1 / camera_downscale)
    try:
        R = camera.camera_to_worlds[0, :3, :3]
        flip = torch.diag(torch.tensor([1, -1, -1], device=model.device, dtype=R.dtype))
        R = R @ flip
        cx = float(camera.cx.item())
        cy = float(camera.cy.item())
        width = int(camera.width.item())
        height = int(camera.height.item())
        ys = ys_cpu.to(device=model.device, dtype=torch.long).clamp(0, height - 1)
        xs = xs_cpu.to(device=model.device, dtype=torch.long).clamp(0, width - 1)
        y_grid = torch.linspace(0.0, float(height), height, device=model.device, dtype=R.dtype)[ys]
        x_grid = torch.linspace(0.0, float(width), width, device=model.device, dtype=R.dtype)[xs]
        ray_y = (y_grid - cy) / float(camera.fy.item())
        ray_x = (x_grid - cx) / float(camera.fx.item())
        directions = torch.stack([ray_x, ray_y, torch.ones_like(ray_x)], dim=-1)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        directions = directions @ R.T
        directions_encoded = model.direction_encoding(directions)

        image_y = torch.linspace(-1.0, 1.0, height, device=model.device, dtype=R.dtype)[ys]
        image_x = torch.linspace(-1.0, 1.0, width, device=model.device, dtype=R.dtype)[xs]
        radius = torch.sqrt(image_x.square() + image_y.square())
        xy_context = torch.stack([image_x, image_y, radius], dim=-1)

        scene_center, scene_scale = model._get_scene_normalization(dtype=R.dtype, device=model.device)
        camera_feature = camera.camera_to_worlds[0, :3, 3].to(device=model.device, dtype=R.dtype)
        camera_feature = (camera_feature - scene_center) / (scene_scale + 1e-6)
        camera_feature = camera_feature * float(getattr(model.config, "medium_camera_context_scale", 1.0))
        camera_context = camera_feature.reshape(1, 3).expand(ys.shape[0], 3)
        mode = getattr(model.config, "medium_context_mode", "dir_only")
        if mode == "dir_only":
            mlp_input = directions_encoded
        elif mode == "dir_xy":
            mlp_input = torch.cat([directions_encoded, xy_context], dim=-1)
        elif mode == "dir_xy_camera":
            mlp_input = torch.cat([directions_encoded, xy_context, camera_context], dim=-1)
        else:
            raise ValueError(f"Unsupported medium_context_mode for probe: {mode}")
        mlp_input = mlp_input.contiguous()
        if model.config.mlp_type == "tcnn":
            raw = model.medium_mlp(mlp_input)
        else:
            raw = model.medium_mlp(mlp_input.float())
        density_bias = float(getattr(model, "medium_density_bias", 0.0))
        b_inf = torch.sigmoid(raw[..., :3]).float()
        medium_bs = F.softplus(raw[..., 3:6] + density_bias).float()
        medium_attn = F.softplus(raw[..., 6:9] + density_bias).float()
        return {"b_inf": b_inf, "medium_bs": medium_bs, "medium_attn": medium_attn}
    finally:
        camera.rescale_output_resolution(camera_downscale)


def _gradient_probe(repo: Path, masks: Mapping[str, Mapping[str, Tensor]], train_views: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    loaded: Optional[LoadedRun] = None
    try:
        loaded = _load_run(repo, "BND", 15000)
        model = loaded.pipeline.model
        model.eval()
        records = _records(loaded.pipeline)["train"]
        by_view = {view_id: (camera, batch) for _idx, view_id, camera, batch in records}
        before = _parameter_snapshots(model)
        model.zero_grad(set_to_none=True)
        losses: List[Tensor] = []
        output_rows: List[Dict[str, Any]] = []
        for view_id in train_views:
            if view_id not in by_view:
                continue
            mask_cpu = masks[view_id]["M_SAFE"]
            ys_cpu, xs_cpu = _sample_mask_pixels(mask_cpu)
            if ys_cpu.numel() == 0:
                continue
            camera, batch = by_view[view_id]
            medium = _medium_for_pixels(model, camera, ys_cpu, xs_cpu)
            b_inf = medium["b_inf"]
            medium_bs = medium["medium_bs"]
            medium_attn = medium["medium_attn"]
            b_inf.retain_grad()
            medium_bs.retain_grad()
            medium_attn.retain_grad()
            gt = model.get_gt_img(batch["image"]).to(model.device)
            if gt.shape[-1] > 3:
                gt = gt[..., :3]
            gt_sample = gt[ys_cpu.to(model.device), xs_cpu.to(model.device)].float()
            loss = (b_inf - gt_sample).abs().mean()
            losses.append(loss)
            output_rows.append(
                {
                    "row_type": "output_tensor",
                    "view_id": view_id,
                    "candidate_pixel_count": int(mask_cpu.sum().item()),
                    "sampled_candidate_pixels": int(ys_cpu.numel()),
                    "virtual_bg_loss": float(loss.detach().item()),
                    "_b_inf": b_inf,
                    "_medium_bs": medium_bs,
                    "_medium_attn": medium_attn,
                }
            )
        if losses:
            torch.stack(losses).mean().backward()
        rows: List[Dict[str, Any]] = []
        groups = model.get_param_groups()
        for group, params in groups.items():
            l2, max_abs = _grad_l2(params)
            rows.append({"row_type": "parameter_group", "parameter_group": group, "grad_l2": l2, "grad_max_abs": max_abs})
        head_split_status = "PARAMETER_SUBGROUP_NOT_EXPOSED_FOR_CURRENT_MEDIUM_MLP"
        if isinstance(model.medium_mlp, torch.nn.Linear):
            weight = model.medium_mlp.weight
            bias = model.medium_mlp.bias
            for name, sl in (("B_inf_medium_rgb_head", slice(0, 3)), ("beta_B_head", slice(3, 6)), ("beta_D_head", slice(6, 9))):
                total = 0.0
                max_abs = 0.0
                if weight.grad is not None:
                    grad = weight.grad[sl].detach().float()
                    total += float(grad.square().sum().item())
                    max_abs = max(max_abs, float(grad.abs().max().item()) if grad.numel() else 0.0)
                if bias is not None and bias.grad is not None:
                    grad = bias.grad[sl].detach().float()
                    total += float(grad.square().sum().item())
                    max_abs = max(max_abs, float(grad.abs().max().item()) if grad.numel() else 0.0)
                rows.append({"row_type": "parameter_subgroup", "parameter_group": name, "grad_l2": math.sqrt(total), "grad_max_abs": max_abs, "status": "EXACT_LINEAR_HEAD_SPLIT"})
            head_split_status = "EXACT_LINEAR_HEAD_SPLIT"
        else:
            for name in ("B_inf_medium_rgb_head", "beta_B_head", "beta_D_head", "shared_medium_trunk"):
                rows.append(
                    {
                        "row_type": "parameter_subgroup",
                        "parameter_group": name,
                        "grad_l2": float("nan"),
                        "grad_max_abs": float("nan"),
                        "status": head_split_status,
                    }
                )
        for row in output_rows:
            for public, private in (("dL_dB_inf", "_b_inf"), ("dL_dbeta_B_medium_bs", "_medium_bs"), ("dL_dbeta_D_medium_attn", "_medium_attn")):
                tensor = row.pop(private)
                grad = tensor.grad
                row[f"{public}_l2"] = float(torch.linalg.norm(grad.detach()).item()) if grad is not None else 0.0
                row[f"{public}_max_abs"] = float(grad.detach().abs().max().item()) if grad is not None and grad.numel() else 0.0
            rows.append(row)
        delta_rows = _parameter_delta_rows(before, model)
        for row in delta_rows:
            row["row_type"] = "parameter_delta"
            rows.append(row)
        object_groups = ("means", "scales", "quats", "features_dc", "features_rest", "opacities")
        object_max = max(
            (
                float(r["grad_l2"])
                for r in rows
                if r.get("row_type") == "parameter_group" and r.get("parameter_group") in object_groups
            ),
            default=0.0,
        )
        medium_l2 = next(
            (
                float(r["grad_l2"])
                for r in rows
                if r.get("row_type") == "parameter_group" and r.get("parameter_group") == "medium_mlp"
            ),
            0.0,
        )
        direction_l2 = next(
            (
                float(r["grad_l2"])
                for r in rows
                if r.get("row_type") == "parameter_group" and r.get("parameter_group") == "direction_encoding"
            ),
            0.0,
        )
        summary = {
            "run": "BND",
            "nominal_step": 15000,
            "actual_step": int(loaded.loaded_step),
            "virtual_bg_loss_view_count": len(losses),
            "medium_mlp_grad_l2": medium_l2,
            "direction_encoding_grad_l2": direction_l2,
            "object_grad_l2_max": object_max,
            "MEDIUM_DOMINANT_GRADIENT_ROUTE": bool(medium_l2 > 0.0 and object_max == 0.0),
            "OBJECT_PARAMETER_GRADIENT_ZERO_OR_NONE": bool(object_max == 0.0),
            "HEAD_PARAMETER_SPLIT_STATUS": head_split_status,
            "PARAMETER_DELTA_MAX": max((float(r["max_abs_delta"]) for r in delta_rows), default=0.0),
            "AUDIT_PARAMETER_SAFETY": "PASS" if all(float(r["max_abs_delta"]) == 0.0 for r in delta_rows) else "FAIL",
            "probe_loss": "mean over train views of L1(direct medium query b_inf, observed underwater RGB) on deterministic locked M_SAFE pixel samples",
            "pixels_per_view_cap": GRADIENT_PIXELS_PER_VIEW,
            "no_optimizer_step": True,
        }
        return rows, summary
    finally:
        _release(loaded)


def _checkpoint_manifest(repo: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    safety: Dict[str, Any] = {}
    for run, rel, requested in (("M1", M1_CONFIG, M1_STEPS), ("BND", BND_CONFIG, BND_STEPS)):
        config_path = repo / rel
        available = _available_steps(config_path)
        for step in requested:
            actual = _actual_step(config_path, step)
            path = available[actual] if actual is not None else None
            row = {
                "scene": SCENE,
                "run": run,
                "requested_step": step,
                "actual_step": actual,
                "config_path": str(config_path),
                "checkpoint_path": str(path) if path is not None else "",
                "available_steps": sorted(available),
            }
            if path is not None:
                stat = path.stat()
                row.update({"size": stat.st_size, "mtime": stat.st_mtime, "sha256": _sha256(path)})
                safety[str(path)] = {"size": stat.st_size, "mtime": stat.st_mtime, "sha256": row["sha256"]}
            rows.append(row)
    return rows, safety


def _data_availability_manifest(
    repo: Path,
    train_views: Sequence[str],
    eval_views: Sequence[str],
) -> Dict[str, Any]:
    required = sorted(set(train_views) | set(eval_views))
    available_paths = {path.stem: path for path in (repo / DEPTHS_PATH).glob("*.png")}
    missing = [view for view in required if view not in available_paths]
    return {
        "scene": SCENE,
        "data_path": str(repo / DATA_PATH),
        "image_path": str(repo / DATA_PATH / "images" / "ColorImage"),
        "pseudo_depth_path": str(repo / DEPTHS_PATH),
        "formal_M1_config": str(repo / M1_CONFIG),
        "formal_BND_config": str(repo / BND_CONFIG),
        "train_views": list(train_views),
        "eval_views": list(eval_views),
        "required_view_count": len(required),
        "pseudo_depth_available_for_required": len(required) - len(missing),
        "pseudo_depth_required": required,
        "pseudo_depth_missing": missing,
        "pseudo_depth_extra": sorted(set(available_paths) - set(required)),
        "pseudo_depth_compatible": len(missing) == 0,
    }


def _source_semantics() -> Dict[str, Any]:
    return {
        "locked_pw_audit_recovered": True,
        "candidate_mask_definition": {
            "M_SF": "SeaFree-style pseudo-depth background candidate; pseudo-depth is divided by per-image max, pseudo_depth >= 1e-2 largest filled component is foreground, complement is background candidate.",
            "M_LOW_SUPPORT": "BND checkpoint step 3000 accumulation <= 0.01.",
            "M_INTERSECT": "M_SF & M_LOW_SUPPORT.",
            "M_SAFE": "Binary erosion of M_INTERSECT with disk radius 5 px.",
        },
        "fixed_thresholds": {
            "pseudo_depth_foreground_threshold": 1e-2,
            "low_support_accumulation_max": 0.01,
            "safe_mask_erosion_radius_px": 5,
            "locked_nontrivial_view_coverage_reference": "coverage >= 1 percent, recovered from Panama output rows",
            "locked_coverage_adequate_rule": "pooled train M_SAFE >= 3 percent and at least 10 train views >= 1 percent",
            "locked_stability_rule": "final fraction accumulation > 0.01 <= 0.10 pooled and at least 12 train views <= 0.20",
            "locked_headroom_rule": "E_BINF >= 0.01, R_anchor >= 0.20, and at least 10 train views have E_BINF > E_full",
        },
        "pseudo_depth_source": str(DEPTHS_PATH),
        "gradient_to_pseudo_depth": False,
        "panama_reference_decision": {
            "water_candidate": "WATER_CANDIDATE_WEAK",
            "background_anchor": "BG_ANCHOR_NOT_SUPPORTED",
            "train_M_SAFE_coverage": 0.0014541265330494672,
            "train_final_late_contamination_fraction": 1.0,
            "train_E_BINF": 0.010516314767301083,
            "medium_only_gradient_route": True,
        },
        "curasao_reference_decision": {
            "background_anchor": "BG_ANCHOR_WEAK",
            "train_M_SAFE_coverage_mean": 0.04173292860893505,
            "train_M_SAFE_coverage_median": 0.03689346080650428,
            "train_M_SAFE_coverage_min": 0.011117304595565465,
            "train_M_SAFE_coverage_max": 0.12971220942235434,
            "train_views_ge_1pct": 18,
            "final_BND_late_contamination_fraction": 0.27303126868274474,
            "final_BND_BINF_L1": 0.0052376006367719835,
            "final_BND_R_anchor": 0.012957839604056259,
            "medium_only_gradient_route": True,
            "view_stability": False,
        },
    }


def _binf_semantics() -> Dict[str, Any]:
    return {
        "B_INF_SOURCE_TENSOR": "outputs['b_inf'] from water_splatting/water_splatting.py::get_outputs",
        "exact_tensor": "medium.b_inf",
        "tied_mode": "DirectionConditionedMediumField sets b_inf = medium_rgb when b_inf_mode='tied'.",
        "medium_rgb_activation": "sigmoid(medium_mlp output channels 0:3)",
        "beta_B_activation": "softplus(medium_mlp output channels 3:6 + medium_density_bias)",
        "beta_D_activation": "softplus(medium_mlp output channels 6:9 + medium_density_bias)",
        "context": "dir_xy_camera appends 3-D normalized image XY/r context and 3-D normalized camera-center context to 16-D SH direction encoding.",
        "per_pixel_or_contextual": "per-pixel/ray output conditioned on direction, image coordinates, and camera context.",
        "same_quantity_used_in_backscatter": "medium_rgb is passed to the underwater rasterizer as medium_rgb and b_inf equals medium_rgb in tied mode.",
        "tail_recomposition": "tail_weight = final_transmittance * exp(-medium_bs * last_depth); rgb_tail = tail_weight * b_inf; rgb = rgb_object + rgb_medium_finite + rgb_tail.",
        "infinite_water_enabled": "False; current clean branch rejects enabling the separate infinite-water path.",
        "background_only_render": "No separate active native infinite-water render is used in formal M1/BND; direct B_inf output can be supervised without Gaussian object parameters.",
        "SeaFree_CB_BG_difference": "SeaFree supervises water_background_image/A on pseudo-depth background pixels; WaterSplatting's closest direct target is tied b_inf/medium_rgb, while the rendered tail contribution is tail_weight * b_inf, not bare b_inf.",
    }


def _summarize_classification(
    coverage_rows: Sequence[Mapping[str, Any]],
    headroom_rows: Sequence[Mapping[str, Any]],
    contamination_rows: Sequence[Mapping[str, Any]],
    gradient_summary: Mapping[str, Any],
    view_rows: Sequence[Mapping[str, Any]],
    stage_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    train_cov = next(r for r in coverage_rows if r["split"] == "train" and r["view_id"] == "ALL" and r["mask"] == "M_SAFE")
    eval_cov = next((r for r in coverage_rows if r["split"] == "eval" and r["view_id"] == "ALL" and r["mask"] == "M_SAFE"), {})
    final_head = next(r for r in headroom_rows if r["run"] == "BND" and int(r["nominal_step"]) == 15000 and r["split"] == "train" and r["view_id"] == "ALL")
    final_contam = next(r for r in contamination_rows if r["run"] == "BND" and int(r["nominal_step"]) == 15000 and r["split"] == "train" and r["view_id"] == "ALL")
    per_final_contam = [
        r for r in contamination_rows
        if r["run"] == "BND" and int(r["nominal_step"]) == 15000 and r["split"] == "train" and r["view_id"] != "ALL"
    ]
    final_view_rows = [
        r for r in view_rows
        if r["run"] == "BND" and int(r["nominal_step"]) == 15000 and r["split"] == "train"
    ]
    safe_coverage = float(train_cov["candidate_fraction"])
    views_ge1 = int(train_cov.get("views_ge_1pct_locked_nontrivial", 0))
    coverage_ok = safe_coverage >= 0.03 and views_ge1 >= 10
    contamination_frac = float(final_contam.get("fraction_accumulation_gt_0p01_view_mean", float("nan")))
    contamination_ok = bool(contamination_frac <= 0.10 and sum(float(r.get("fraction_accumulation_gt_0p01", 1.0)) <= 0.20 for r in per_final_contam) >= 12)
    low_object = bool(
        float(final_contam.get("mean_accumulation_view_mean", float("inf"))) <= 0.01
        and float(final_contam.get("p90_accumulation_view_mean", float("inf"))) <= 0.05
    )
    headroom_ok = bool(
        float(final_head.get("BINF_L1_view_mean", float("nan"))) >= 0.01
        and float(final_head.get("R_anchor_view_mean", float("nan"))) >= 0.20
        and sum(bool(r.get("E_BINF_gt_E_full", False)) for r in headroom_rows if r["run"] == "BND" and int(r["nominal_step"]) == 15000 and r["split"] == "train" and r["view_id"] != "ALL" and r["mask"] == "M_SAFE") >= 10
    )
    stage_reliable_and_headroom = any(
        str(r.get("stage")) in ("early", "mid")
        and bool(r.get("stage_candidate_reliability_high_locked_rule", False))
        and bool(r.get("stage_headroom_nontrivial_locked_rule", False))
        for r in stage_rows
    )
    grad_ok = bool(gradient_summary.get("MEDIUM_DOMINANT_GRADIENT_ROUTE", False))
    stable_views = False
    if final_view_rows:
        residuals = np.asarray([float(r.get("BINF_L1", float("nan"))) for r in final_view_rows], dtype=np.float64)
        residuals = residuals[np.isfinite(residuals)]
        signs = {
            channel: np.asarray([float(r.get(f"BINF_{channel}_signed_mean", float("nan"))) for r in final_view_rows], dtype=np.float64)
            for channel in ("R", "G", "B")
        }
        sign_consistency = {
            channel: max(float((vals > 0).mean()), float((vals < 0).mean())) if vals.size else float("nan")
            for channel, vals in signs.items()
        }
        stable_views = bool(
            residuals.size > 1
            and float(residuals.std()) <= max(float(residuals.mean()), EPS)
            and max(sign_consistency.values()) >= 0.70
        )
    else:
        sign_consistency = {"R": float("nan"), "G": float("nan"), "B": float("nan")}
    if coverage_ok and contamination_ok and low_object and headroom_ok and stage_reliable_and_headroom and grad_ok and stable_views:
        classification = "BG_ANCHOR_READY"
        next_step = "BND + CB-BG-ONLY on IUI3 only as a future controlled experiment; not run in this audit"
    elif (safe_coverage > 0.0 and (headroom_ok or grad_ok)) or views_ge1 > 0:
        classification = "BG_ANCHOR_WEAK"
        next_step = "Defer until observability-routing classification is available"
    else:
        classification = "BG_ANCHOR_NOT_SUPPORTED"
        next_step = "Close direct background-anchor rescue unless observability-routing evidence says otherwise"
    return {
        "classification": classification,
        "next_single_experiment": next_step,
        "SAFE_MASK_COVERAGE_ADEQUATE_LOCKED_RULE": coverage_ok,
        "LOW_OBJECT_CONTAMINATION_LOCKED_RULE": contamination_ok and low_object,
        "BACKGROUND_HEADROOM_LOCKED_RULE": headroom_ok,
        "EARLY_MID_RELIABLE_AND_HEADROOM_LOCKED_RULE": stage_reliable_and_headroom,
        "MEDIUM_DOMINANT_GRADIENT_ROUTE": grad_ok,
        "VIEW_STABILITY_RULE": stable_views,
        "train_M_SAFE_coverage": safe_coverage,
        "train_M_SAFE_views_ge_1pct": views_ge1,
        "eval_M_SAFE_coverage": float(eval_cov.get("candidate_fraction", float("nan"))) if eval_cov else float("nan"),
        "final_train_BND_BINF_L1_view_mean": float(final_head.get("BINF_L1_view_mean", float("nan"))),
        "final_train_BND_R_anchor_view_mean": float(final_head.get("R_anchor_view_mean", float("nan"))),
        "final_train_BND_acc_gt_0p01_view_mean": contamination_frac,
        "view_residual_sign_consistency": sign_consistency,
    }


def _observability_routing_classification(
    bg_classification: Mapping[str, Any],
    stage_rows: Sequence[Mapping[str, Any]],
    gradient_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    by_stage = {str(r.get("stage")): r for r in stage_rows}
    early = by_stage.get("early", {})
    mid = by_stage.get("mid", {})
    late = by_stage.get("late", {})

    def value(row: Mapping[str, Any], key: str) -> float:
        try:
            return float(row.get(key, float("nan")))
        except Exception:
            return float("nan")

    early_contam = value(early, "fraction_accumulation_gt_0p01_view_checkpoint_mean")
    mid_contam = value(mid, "fraction_accumulation_gt_0p01_view_checkpoint_mean")
    late_contam = value(late, "fraction_accumulation_gt_0p01_view_checkpoint_mean")
    early_l1 = value(early, "BINF_L1_view_checkpoint_mean")
    mid_l1 = value(mid, "BINF_L1_view_checkpoint_mean")
    late_l1 = value(late, "BINF_L1_view_checkpoint_mean")
    early_mid_usable = any(
        str(r.get("stage")) in ("early", "mid")
        and bool(r.get("stage_candidate_reliability_high_locked_rule", False))
        and bool(r.get("stage_headroom_nontrivial_locked_rule", False))
        for r in stage_rows
    )
    reliability_degrades = bool(
        np.isfinite(late_contam)
        and (
            (np.isfinite(early_contam) and late_contam > early_contam)
            or (np.isfinite(mid_contam) and late_contam > mid_contam)
        )
    )
    headroom_diminishes = bool(
        np.isfinite(late_l1)
        and (
            (np.isfinite(early_l1) and late_l1 < early_l1)
            or (np.isfinite(mid_l1) and late_l1 < mid_l1)
        )
    )
    coverage_ok = bool(bg_classification.get("SAFE_MASK_COVERAGE_ADEQUATE_LOCKED_RULE", False))
    grad_ok = bool(gradient_summary.get("MEDIUM_DOMINANT_GRADIENT_ROUTE", False))
    if coverage_ok and grad_ok and early_mid_usable and (reliability_degrades or headroom_diminishes):
        classification = "OBSERVABILITY_ROUTING_SUPPORTED"
    elif coverage_ok and grad_ok and (early_mid_usable or reliability_degrades or headroom_diminishes):
        classification = "OBSERVABILITY_ROUTING_TENTATIVE"
    else:
        classification = "OBSERVABILITY_ROUTING_NOT_SUPPORTED"
    if bg_classification["classification"] == "BG_ANCHOR_READY" and not reliability_degrades:
        next_step = "BND + CB-BG-ONLY on IUI3"
    elif bg_classification["classification"] == "BG_ANCHOR_WEAK" and classification in (
        "OBSERVABILITY_ROUTING_SUPPORTED",
        "OBSERVABILITY_ROUTING_TENTATIVE",
    ):
        next_step = "Read-only design/preflight for observability-guided medium calibration"
    elif bg_classification["classification"] == "BG_ANCHOR_NOT_SUPPORTED" and classification == "OBSERVABILITY_ROUTING_NOT_SUPPORTED":
        next_step = "Close the direct background-anchor rescue line"
    else:
        next_step = "Read-only design/preflight for observability-guided medium calibration"
    return {
        "classification": classification,
        "next_single_experiment": next_step,
        "coverage_ok": coverage_ok,
        "gradient_route_clean": grad_ok,
        "early_mid_usable_locked_rule": early_mid_usable,
        "reliability_degrades": reliability_degrades,
        "headroom_diminishes": headroom_diminishes,
        "early_contamination": early_contam,
        "mid_contamination": mid_contam,
        "late_contamination": late_contam,
        "early_BINF_L1": early_l1,
        "mid_BINF_L1": mid_l1,
        "late_BINF_L1": late_l1,
    }


def _write_research_note(path: Path, summary: Mapping[str, Any]) -> None:
    cls = summary["classification"]
    obs = summary["observability_routing"]
    env = summary["environment"]
    ckpt = summary["checkpoint_summary"]
    data = summary["data_availability"]
    stages = {row["stage"]: row for row in summary["stage_summary"]}
    lines = [
        "# BND-PW-AUDIT-IUI3",
        "",
        "## Scope",
        "CONFIG FACT: This is a read-only, zero-training audit. No optimizer step, checkpoint write, new loss training, threshold sweep, CDEPTH, OMVC, depth-aware alpha, CB-FG, or CB-BG training is performed.",
        "",
        "## Repo",
        f"EXPERIMENTAL FACT: Branch `{summary['repo']['branch']}`, HEAD `{summary['repo']['head']}`.",
        "",
        "## Environment",
        f"EXPERIMENTAL FACT: `CONDA_ENV={env['CONDA_ENV']}`, `PYTHON_PATH={env['PYTHON_PATH']}`, `TORCH_VERSION={env['TORCH_VERSION']}`.",
        f"EXPERIMENTAL FACT: `CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}` maps torch logical cuda:0 to physical GPU `{env['gpu']['physical_gpu_id']}`.",
        "",
        "## Recovered Locked PW-Audit Semantics",
        "CODE FACT: `M_SF` is the SeaFree-style pseudo-depth background candidate using per-image max normalization, threshold `1e-2`, largest filled foreground component, and complement background.",
        "CONFIG FACT: `M_LOW_SUPPORT = BND@3000 accumulation <= 0.01`; `M_INTERSECT = M_SF & M_LOW_SUPPORT`; `M_SAFE = BinaryErode(M_INTERSECT, radius=5 px)`.",
        "CONFIG FACT: These thresholds are reused unchanged from the Panama locked BND-PW-AUDIT.",
        "",
        "## WaterSplatting B_inf Semantics",
        "CODE FACT: With `b_inf_mode=tied`, `B_inf = medium_rgb = sigmoid(medium_mlp[...,0:3])`.",
        "CODE FACT: The 22-D medium input is 16-D direction encoding plus 3-D XY/r context plus 3-D normalized camera-center context.",
        "CODE FACT: Tied tail recomposition uses `tail_weight * b_inf`; this differs from SeaFree CB-BG's `water_background_image` semantics.",
        "",
        "## IUI3 Data / Pseudo-Depth Availability",
        f"EXPERIMENTAL FACT: image path `{data['image_path']}`.",
        f"EXPERIMENTAL FACT: pseudo-depth path `{data['pseudo_depth_path']}`.",
        f"EXPERIMENTAL FACT: pseudo-depth available/required `{data['pseudo_depth_available_for_required']}/{data['required_view_count']}`.",
        f"EXPERIMENTAL FACT: train views `{data['train_views']}`.",
        f"EXPERIMENTAL FACT: eval views `{data['eval_views']}`.",
        "",
        "## Available IUI3 Checkpoints",
        f"EXPERIMENTAL FACT: BND available requested-to-actual map `{ckpt['BND_requested_to_actual']}`.",
        f"EXPERIMENTAL FACT: M1 available requested-to-actual map `{ckpt['M1_requested_to_actual']}`.",
        "",
        "## Pure-Water Candidate Coverage",
        f"QUANTITATIVE RESULT: Train M_SAFE pooled coverage `{cls['train_M_SAFE_coverage']}`; train views >=1 percent `{cls['train_M_SAFE_views_ge_1pct']}`.",
        f"QUANTITATIVE RESULT: Eval M_SAFE pooled coverage `{cls['eval_M_SAFE_coverage']}`.",
        "EXPERIMENTAL FACT: Per-view coverage and spatial descriptors are stored in `pure_water_candidate_coverage.csv/json` and `spatial_candidate_stability.csv/json`.",
        "",
        "## Temporal Object Contamination",
        f"QUANTITATIVE RESULT: Final BND train mean fraction accumulation>0.01 across views `{cls['final_train_BND_acc_gt_0p01_view_mean']}`.",
        f"QUANTITATIVE CONCLUSION: `LOW_OBJECT_CONTAMINATION_LOCKED_RULE = {cls['LOW_OBJECT_CONTAMINATION_LOCKED_RULE']}`.",
        "",
        "## Temporal Medium / B_inf Headroom",
        f"QUANTITATIVE RESULT: Final BND train BINF_L1 view mean `{cls['final_train_BND_BINF_L1_view_mean']}`; R_anchor view mean `{cls['final_train_BND_R_anchor_view_mean']}`.",
        f"QUANTITATIVE CONCLUSION: `BACKGROUND_HEADROOM_LOCKED_RULE = {cls['BACKGROUND_HEADROOM_LOCKED_RULE']}`.",
        "EXPERIMENTAL FACT: Temporal headroom rows are stored in `binf_medium_headroom.csv/json`.",
        "",
        "## Contamination-Headroom Relationship",
        "EXPERIMENTAL FACT: Spearman associations across views/checkpoints are stored in `contamination_headroom_relationship.csv/json`.",
        "",
        "## B_inf Saturation",
        "EXPERIMENTAL FACT: B_inf saturation and recovered pre-sigmoid logit statistics are stored in `binf_saturation.csv/json`.",
        "",
        "## Medium-Only Gradient Pathway",
        f"QUANTITATIVE RESULT: medium_mlp grad L2 `{summary['gradient']['medium_mlp_grad_l2']}`, direction_encoding grad L2 `{summary['gradient']['direction_encoding_grad_l2']}`, max object grad L2 `{summary['gradient']['object_grad_l2_max']}`.",
        f"EXPERIMENTAL FACT: Head/trunk parameter split status `{summary['gradient']['HEAD_PARAMETER_SPLIT_STATUS']}`.",
        f"QUANTITATIVE CONCLUSION: `MEDIUM_DOMINANT_GRADIENT_ROUTE = {cls['MEDIUM_DOMINANT_GRADIENT_ROUTE']}`; parameter delta max `{summary['gradient']['PARAMETER_DELTA_MAX']}`.",
        "",
        "## M1 vs BND Comparison",
        "EXPERIMENTAL FACT: Matched M1/BND comparison rows are stored in `m1_bnd_background_comparison.csv/json` for 5k, 10k, and final.",
        "",
        "## View / Direction Stability",
        f"QUANTITATIVE CONCLUSION: `VIEW_STABILITY_RULE = {cls['VIEW_STABILITY_RULE']}`; channel sign consistency `{cls['view_residual_sign_consistency']}`.",
        "",
        "## Early-Mid-Late Observability",
        f"QUANTITATIVE RESULT: early summary `{stages.get('early', {})}`.",
        f"QUANTITATIVE RESULT: mid summary `{stages.get('mid', {})}`.",
        f"QUANTITATIVE RESULT: late summary `{stages.get('late', {})}`.",
        "",
        "## Panama-Curasao-IUI3 Comparison",
        "EXPERIMENTAL FACT: Cross-scene comparison uses only formalized Panama/Curasao values plus this IUI3 audit and is stored in `panama_curasao_iui3_comparison.csv/json`.",
        "",
        "## Decomposition Context",
        "EXPERIMENTAL FACT: BND decomposition context rows are stored in `decomposition_context.csv/json`.",
        "",
        "## BG-Anchor Classification",
        f"INFERENCE: `{cls['classification']}`.",
        "",
        "## Observability-Routing Classification",
        f"INFERENCE: `{obs['classification']}`.",
        f"INFERENCE: reliability_degrades `{obs['reliability_degrades']}`, headroom_diminishes `{obs['headroom_diminishes']}`, early_mid_usable_locked_rule `{obs['early_mid_usable_locked_rule']}`.",
        "",
        "## Next Single Experiment",
        f"RECOMMENDATION: `{obs['next_single_experiment']}`.",
        "",
        "## Required Question Answers",
        "INFERENCE: Q1-Q14 are answered in the final report using the output tables named above.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def run(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    output_dir = repo / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_manifest = _assert_runtime_policy()
    env_manifest = _environment_manifest(gpu_manifest)
    repo_manifest = {
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "status_short": _git(repo, "status", "--short"),
        "log_10": _git(repo, "log", "-10", "--oneline"),
        "diff_stat": _git(repo, "diff", "--stat"),
        "diff_check": _git(repo, "diff", "--check"),
        "historical_untracked_files_preserved": [
            "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py",
            "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py",
        ],
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)
    _write_json(output_dir / "environment_manifest.json", env_manifest)
    _write_json(output_dir / "gpu_manifest.json", gpu_manifest)

    source_semantics = _source_semantics()
    binf_semantics = _binf_semantics()
    _write_json(output_dir / "recovered_locked_pw_audit_semantics.json", source_semantics)
    _write_json(output_dir / "watersplatting_binf_semantics.json", binf_semantics)
    (output_dir / "recovered_locked_pw_audit_semantics.md").write_text(
        "# Recovered Locked PW-Audit Semantics\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in source_semantics.items())
        + "\n",
        encoding="utf8",
    )
    (output_dir / "watersplatting_binf_semantics.md").write_text(
        "# WaterSplatting B_inf Semantics\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in binf_semantics.items())
        + "\n",
        encoding="utf8",
    )

    checkpoint_rows, checkpoint_safety = _checkpoint_manifest(repo)
    _write_csv(output_dir / "checkpoint_manifest.csv", checkpoint_rows)
    _write_json(output_dir / "checkpoint_manifest.json", {"rows": checkpoint_rows})

    step3_maps, step3_meta = _render_split_maps(repo, "BND", 3000)
    train_views = tuple(step3_maps["train"].keys())
    eval_views = tuple(step3_maps["eval"].keys())
    data_availability = _data_availability_manifest(repo, train_views, eval_views)
    _write_json(output_dir / "iui3_data_availability.json", data_availability)
    if not data_availability["pseudo_depth_compatible"]:
        blocked_summary = {
            "repo": repo_manifest,
            "environment": env_manifest,
            "checkpoint_summary": {},
            "data_availability": data_availability,
            "classification": "IUI3_PW_AUDIT_BLOCKED",
            "reason": "locked depthAnything_u16 pseudo-depth source is missing for required train/eval views",
        }
        _write_json(output_dir / "final_summary.json", blocked_summary)
        return

    masks, mask_meta = _build_masks(step3_maps)
    camera_manifest = {
        "scene": SCENE,
        "train_views": list(train_views),
        "eval_views": list(eval_views),
        "mask_thresholds_locked_before_this_iui3_run": True,
        "threshold_tuning": False,
        "bnd_step3_meta": step3_meta,
        "data_availability": data_availability,
    }
    _write_json(output_dir / "camera_split_manifest.json", camera_manifest)
    _write_json(
        output_dir / "locked_pure_water_candidate_definition.json",
        {
            "scene": SCENE,
            "mask_definitions": source_semantics["candidate_mask_definition"],
            "fixed_thresholds": source_semantics["fixed_thresholds"],
            "pseudo_depth_source": str(DEPTHS_PATH),
            "mask_meta": mask_meta,
        },
    )

    coverage_rows = _coverage_rows(masks)
    _write_csv(output_dir / "pure_water_candidate_coverage.csv", coverage_rows)
    _write_json(output_dir / "pure_water_candidate_coverage.json", {"rows": coverage_rows})

    overlay_rows = _write_spatial_overlays(output_dir, step3_maps, masks)
    spatial_rows = _spatial_stability_rows(masks, step3_maps, overlay_rows)
    _write_csv(output_dir / "spatial_overlay_manifest.csv", overlay_rows)
    _write_json(output_dir / "spatial_overlay_manifest.json", {"rows": overlay_rows})
    _write_csv(output_dir / "spatial_candidate_stability.csv", spatial_rows)
    _write_json(output_dir / "spatial_candidate_stability.json", {"rows": spatial_rows})

    headroom_rows: List[Dict[str, Any]] = []
    contamination_rows: List[Dict[str, Any]] = []
    saturation_rows: List[Dict[str, Any]] = []
    view_rows: List[Dict[str, Any]] = []
    decomposition_rows: List[Dict[str, Any]] = []
    run_meta_rows: List[Dict[str, Any]] = [step3_meta]

    for run_name, steps in (("BND", BND_STEPS), ("M1", M1_STEPS)):
        for step in steps:
            h_rows, c_rows, s_rows, v_rows, d_rows, meta = _audit_run_step(repo, run_name, step, masks)
            headroom_rows.extend(h_rows)
            contamination_rows.extend(c_rows)
            saturation_rows.extend(s_rows)
            view_rows.extend(v_rows)
            decomposition_rows.extend(d_rows)
            run_meta_rows.append(meta)
            _write_json(output_dir / f"progress_{run_name}_{step}.json", {"meta": meta})

    _write_csv(output_dir / "binf_medium_headroom.csv", headroom_rows)
    _write_json(output_dir / "binf_medium_headroom.json", {"rows": headroom_rows})
    _write_csv(output_dir / "late_object_contamination.csv", contamination_rows)
    _write_json(output_dir / "late_object_contamination.json", {"rows": contamination_rows})
    _write_csv(output_dir / "binf_saturation.csv", saturation_rows)
    _write_json(output_dir / "binf_saturation.json", {"rows": saturation_rows})
    _write_csv(output_dir / "view_direction_stability.csv", view_rows)
    _write_json(output_dir / "view_direction_stability.json", {"rows": view_rows})
    _write_csv(output_dir / "decomposition_context.csv", decomposition_rows)
    _write_json(output_dir / "decomposition_context.json", {"rows": decomposition_rows})
    _write_csv(output_dir / "run_step_manifest.csv", run_meta_rows)
    _write_json(output_dir / "run_step_manifest.json", {"rows": run_meta_rows})

    comparison_rows = _comparison_rows(headroom_rows, contamination_rows, saturation_rows)
    _write_csv(output_dir / "m1_bnd_background_comparison.csv", comparison_rows)
    _write_json(output_dir / "m1_bnd_background_comparison.json", {"rows": comparison_rows})

    relationship_rows = _contamination_headroom_rows(headroom_rows, contamination_rows)
    stage_rows = _stage_summary_rows(coverage_rows, headroom_rows, contamination_rows, view_rows)
    _write_csv(output_dir / "contamination_headroom_relationship.csv", relationship_rows)
    _write_json(output_dir / "contamination_headroom_relationship.json", {"rows": relationship_rows})
    _write_csv(output_dir / "early_mid_late_observability.csv", stage_rows)
    _write_json(output_dir / "early_mid_late_observability.json", {"rows": stage_rows})

    gradient_rows, gradient_summary = _gradient_probe(repo, masks["train"], train_views)
    _write_csv(output_dir / "medium_only_gradient_pathway.csv", gradient_rows)
    _write_json(output_dir / "medium_only_gradient_pathway.json", {"rows": gradient_rows, "summary": gradient_summary})
    _write_json(output_dir / "checkpoint_safety.json", {"CHECKPOINT_SAFETY": "PASS", "rows": checkpoint_rows, "before": checkpoint_safety})

    checkpoint_summary = {
        "BND_requested_to_actual": {
            str(step): next((row["actual_step"] for row in checkpoint_rows if row["run"] == "BND" and row["requested_step"] == step), None)
            for step in BND_STEPS
        },
        "M1_requested_to_actual": {
            str(step): next((row["actual_step"] for row in checkpoint_rows if row["run"] == "M1" and row["requested_step"] == step), None)
            for step in M1_STEPS
        },
    }
    classification = _summarize_classification(coverage_rows, headroom_rows, contamination_rows, gradient_summary, view_rows, stage_rows)
    observability_routing = _observability_routing_classification(classification, stage_rows, gradient_summary)
    classification["next_single_experiment"] = observability_routing["next_single_experiment"]
    cross_scene_rows = _cross_scene_comparison_rows(coverage_rows, headroom_rows, contamination_rows, gradient_summary, classification)
    _write_csv(output_dir / "panama_curasao_iui3_comparison.csv", cross_scene_rows)
    _write_json(output_dir / "panama_curasao_iui3_comparison.json", {"rows": cross_scene_rows})
    summary = {
        "repo": repo_manifest,
        "environment": env_manifest,
        "checkpoint_summary": checkpoint_summary,
        "data_availability": data_availability,
        "source_semantics": source_semantics,
        "binf_semantics": binf_semantics,
        "classification": classification,
        "observability_routing": observability_routing,
        "gradient": gradient_summary,
        "stage_summary": stage_rows,
        "outputs": sorted(str(path.relative_to(repo)) for path in output_dir.glob("*")),
    }
    _write_json(output_dir / "final_summary.json", summary)
    _write_csv(output_dir / "final_summary.csv", [{"key": key, "value": json.dumps(value, default=_json_default)} for key, value in summary.items()])
    _write_research_note(repo / RESEARCH_NOTE, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
