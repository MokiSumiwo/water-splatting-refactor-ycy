#!/usr/bin/env python3
"""Read-only IUI3 late candidate-region object-occupation drift audit.

This diagnostic loads existing WaterSplatting checkpoints, reuses the locked
IUI3 PW-audit candidate mask semantics, and measures whether late increases in
screen-space Gaussian accumulation are associated with RGB or decomposition
harm. It never constructs an optimizer, never updates model parameters, never
writes checkpoints, and never changes CUDA or training code.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_pw_helpers() -> Any:
    helper_path = REPO_ROOT / "scripts/diagnostics/audit_bnd_pw_iui3.py"
    spec = importlib.util.spec_from_file_location("audit_bnd_pw_iui3_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PW = _load_pw_helpers()

SCENE = "IUI3"
OUTPUT_DIR = Path("outputs/bnd_responsibility_drift_audit_iui3_20260825")
RESEARCH_NOTE = Path("research_notes/BND_RESPONSIBILITY_DRIFT_AUDIT_IUI3_2026-08-25.md")
BND_STEPS = (3000, 5000, 8000, 10000, 13000, 15000)
M1_STEPS = (5000, 10000, 15000)
LATE_TRANSITIONS = ((10000, 13000), (13000, 15000), (10000, 15000))
ALL_TRANSITIONS = ((3000, 5000), (5000, 8000), (8000, 10000), *LATE_TRANSITIONS)
ALLOWED_PHYSICAL_GPUS = frozenset({"6", "7", "8", "9"})
EPS = 1e-8
TOP_FRACTIONS = (0.10, 0.20, 0.30)
SPEARMAN_MAX_N = 2_000_000
VISUAL_VIEW_LIMIT = 6


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


def _shell_capture(repo: Path, args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(list(args), cwd=repo, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _sha256(path: Path) -> str:
    h = __import__("hashlib").sha256()
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


def _repo_manifest(repo: Path) -> Dict[str, Any]:
    return {
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "status_short": _git(repo, "status", "--short"),
        "log_10": _git(repo, "log", "-10", "--oneline"),
        "diff_stat": _git(repo, "diff", "--stat"),
        "diff_check": _git(repo, "diff", "--check"),
        "historical_untracked_files_preserved_not_modified_or_committed": [
            "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py",
            "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py",
        ],
    }


def _accumulation_semantics() -> Dict[str, Any]:
    return {
        "CODE_FACT": True,
        "outputs_accumulation": "outputs['accumulation'] is render.accumulation from UnderwaterRasterizer.",
        "renderer_source": "water_splatting/rendering/underwater_rasterizer.py:92-122 calls rasterize_gaussians(..., return_alpha=True, return_hit_stats=True).",
        "python_binding_source": "water_splatting/rasterize.py:249-263 sets out_alpha = 1 - final_Ts.",
        "cuda_source": "water_splatting/cuda/csrc/forward.cu:439-445 computes alpha=min(0.999, opacity_i*exp(-sigma)) and next_T=T*(1-alpha).",
        "screen_space_semantics": "per-pixel accumulated Gaussian alpha/transmittance complement, 1 - final transmittance after compositing all rasterized Gaussians that pass the CUDA contribution test.",
        "per_pixel": True,
        "differentiable": "The rasterize binding saves tensors and implements backward; accumulation is returned from the differentiable rasterizer path, though this read-only script uses no gradients.",
        "includes_all_gaussians": "All projected/rasterized Gaussians passing visibility/contribution tests may affect final_Ts; the output is not a per-Gaussian contribution map.",
        "medium_contribution": "The medium field contributes to scattering/color and appears in the CUDA skip condition via attenuation, but medium itself is not accumulated as alpha.",
        "distinctions": {
            "gaussian_opacity_parameter_alpha_i": "torch.sigmoid(model.opacities) per Gaussian, used inside alpha_i*exp(-sigma).",
            "screen_space_accumulation": "per-pixel 1-final_Ts after sorted Gaussian alpha compositing.",
            "actual_per_gaussian_compositing_contribution": "not exposed by the current renderer without CUDA changes; this audit does not modify CUDA.",
        },
    }


def _checkpoint_topology_semantics(repo: Path) -> Dict[str, Any]:
    config_path = repo / PW.BND_CONFIG
    config_hits = _shell_capture(
        repo,
        [
            "rg",
            "-n",
            "stop_split_at|stop_screen_size_at|refine_every|reset_alpha_every|reset_alpha_thresh|densify_grad_thresh|densify_size_thresh|split_screen_size|continue_cull_post_densification|n_split_samples",
            str(config_path),
        ],
    )
    source_hits = _shell_capture(
        repo,
        [
            "rg",
            "-n",
            "def refinement_after|def split_gaussians|def dup_gaussians|def cull_gaussians|reset_alpha|stop_split_at|continue_cull_post_densification",
            "water_splatting/water_splatting.py",
        ],
    )
    return {
        "CODE_FACT": True,
        "stored_gaussian_parameter_tensors": [
            "_model.gauss_params.means",
            "_model.gauss_params.scales",
            "_model.gauss_params.quats",
            "_model.gauss_params.features_dc",
            "_model.gauss_params.features_rest",
            "_model.gauss_params.opacities",
            "_model.medium_mlp.*",
            "_model.direction_encoding.*",
        ],
        "densification_schedule": "refinement_after runs every refine_every callback after warmup; densification only when step < stop_split_at and reset-interval visibility condition is satisfied.",
        "split_semantics": "split_gaussians samples children around selected means, copies appearance/opacities/quats, divides selected scales by 1.6, appends children, then culls original split parents via extra_cull_mask.",
        "duplicate_semantics": "dup_gaussians appends exact parameter copies for selected small/high-gradient Gaussians.",
        "pruning_semantics": "cull_gaussians removes sigmoid(opacity) below cull_alpha_thresh before stop_split_at, cull_alpha_thresh_post after stop_split_at, and huge-scale Gaussians after reset_alpha_every*refine_every.",
        "opacity_reset_semantics": "when step < stop_split_at and step % (reset_alpha_every*refine_every) == refine_every, opacities are clamped to logit(reset_alpha_thresh) and opacity optimizer moments are reset.",
        "scale_updates": "scales are trainable Gaussian parameters and are also reduced for split parents/children during split.",
        "position_updates": "means are trainable Gaussian parameters; split children get sampled offsets around parent means.",
        "formal_config_source_hits": config_hits,
        "topology_source_hits": source_hits,
    }


def _scalar_map(value: Tensor) -> Tensor:
    value = value.float()
    if value.ndim == 3 and value.shape[-1] == 1:
        return value[..., 0]
    if value.ndim == 3 and value.shape[-1] == 3:
        return value.mean(dim=-1)
    if value.ndim == 2:
        return value
    raise ValueError(f"Cannot scalarize tensor with shape {tuple(value.shape)}")


def _np_stats(values: np.ndarray, prefix: str = "") -> Dict[str, Any]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            f"{prefix}mean": float("nan"),
            f"{prefix}median": float("nan"),
            f"{prefix}p10": float("nan"),
            f"{prefix}p50": float("nan"),
            f"{prefix}p90": float("nan"),
            f"{prefix}p99": float("nan"),
            f"{prefix}min": float("nan"),
            f"{prefix}max": float("nan"),
        }
    return {
        f"{prefix}mean": float(vals.mean()),
        f"{prefix}median": float(np.median(vals)),
        f"{prefix}p10": float(np.quantile(vals, 0.10)),
        f"{prefix}p50": float(np.quantile(vals, 0.50)),
        f"{prefix}p90": float(np.quantile(vals, 0.90)),
        f"{prefix}p99": float(np.quantile(vals, 0.99)),
        f"{prefix}min": float(vals.min()),
        f"{prefix}max": float(vals.max()),
    }


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


def _spearman_np(x_values: np.ndarray, y_values: np.ndarray) -> Tuple[float, int, bool]:
    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y = np.asarray(y_values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    sampled = False
    if x.size > SPEARMAN_MAX_N:
        idx = np.linspace(0, x.size - 1, SPEARMAN_MAX_N).astype(np.int64)
        x = x[idx]
        y = y[idx]
        sampled = True
    if x.size < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan"), int(x.size), sampled
    xr = _rankdata(x)
    yr = _rankdata(y)
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = math.sqrt(float((xr * xr).sum()) * float((yr * yr).sum()))
    if denom <= 0.0:
        return float("nan"), int(x.size), sampled
    return float((xr * yr).sum() / denom), int(x.size), sampled


def _safe_values(tensor: Tensor, mask: Tensor) -> np.ndarray:
    if int(mask.sum().item()) == 0:
        return np.empty((0,), dtype=np.float32)
    return tensor.detach().float()[mask].cpu().numpy().astype(np.float32, copy=False)


def _safe_rgb_values(tensor: Tensor, mask: Tensor) -> np.ndarray:
    if int(mask.sum().item()) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return tensor.detach().float()[mask].cpu().numpy().astype(np.float32, copy=False)


def _top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    vals = np.asarray(values).reshape(-1)
    if vals.size == 0:
        return np.zeros((0,), dtype=bool)
    k = max(1, int(math.ceil(vals.size * fraction)))
    if k >= vals.size:
        return np.ones(vals.shape, dtype=bool)
    idx = np.argpartition(vals, vals.size - k)[vals.size - k :]
    out = np.zeros(vals.shape, dtype=bool)
    out[idx] = True
    return out


def _mean_or_nan(values: Iterable[float]) -> float:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    return float(vals.mean()) if vals.size else float("nan")


def _sum_or_nan(values: Iterable[float]) -> float:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    return float(vals.sum()) if vals.size else float("nan")


def _available_checkpoint_path(repo: Path, run: str, nominal_step: int) -> Tuple[Optional[int], Optional[Path]]:
    rel_config, _parameterization = PW._run_config(run)
    config_path = repo / rel_config
    available = PW._available_steps(config_path)
    actual = PW._actual_step(config_path, nominal_step)
    if actual is None:
        return None, None
    return actual, available.get(actual)


def _checkpoint_manifest(repo: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    safety: Dict[str, Any] = {}
    for run, requested in (("BND", BND_STEPS), ("M1", M1_STEPS)):
        rel_config, _ = PW._run_config(run)
        config_path = repo / rel_config
        available = PW._available_steps(config_path)
        for step in requested:
            actual, path = _available_checkpoint_path(repo, run, step)
            row = {
                "scene": SCENE,
                "run": run,
                "requested_step": step,
                "actual_step": actual,
                "config_path": str(config_path),
                "checkpoint_path": str(path) if path else "",
                "available_steps": sorted(available),
            }
            if path is not None:
                stat = path.stat()
                row.update({"size": stat.st_size, "mtime": stat.st_mtime, "sha256": _sha256(path)})
                safety[str(path)] = {"size": stat.st_size, "mtime": stat.st_mtime, "sha256": row["sha256"]}
            rows.append(row)
    return rows, safety


def _checkpoint_storage_rows(repo: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run, steps in (("BND", BND_STEPS), ("M1", M1_STEPS)):
        for nominal in steps:
            actual, path = _available_checkpoint_path(repo, run, nominal)
            if path is None:
                rows.append({"run": run, "nominal_step": nominal, "actual_step": actual, "status": "MISSING"})
                continue
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt.get("pipeline", ckpt)
            means = state.get("_model.gauss_params.means")
            row: Dict[str, Any] = {
                "run": run,
                "nominal_step": nominal,
                "actual_step": actual,
                "checkpoint_path": str(path),
                "status": "AVAILABLE",
                "checkpoint_top_keys": ";".join(sorted(ckpt.keys())),
                "gaussian_count": int(means.shape[0]) if isinstance(means, Tensor) else None,
                "gaussian_parameter_tensors": ";".join(
                    key
                    for key in sorted(state.keys())
                    if key.startswith("_model.gauss_params.")
                    or key.startswith("_model.medium_mlp")
                    or key.startswith("_model.direction_encoding")
                ),
            }
            lineage = state.get("_model.gaussian_lineage_ids")
            if isinstance(lineage, Tensor):
                unique = torch.unique(lineage)
                row.update(
                    {
                        "gaussian_lineage_ids_present": True,
                        "lineage_shape": list(lineage.shape),
                        "lineage_dtype": str(lineage.dtype),
                        "lineage_unique_count": int(unique.numel()),
                        "lineage_duplicate_count": int(lineage.numel() - unique.numel()),
                        "lineage_min": int(lineage.min().item()),
                        "lineage_max": int(lineage.max().item()),
                        "lineage_unique_equals_gaussian_count": bool(unique.numel() == lineage.numel()),
                    }
                )
            else:
                row.update(
                    {
                        "gaussian_lineage_ids_present": False,
                        "lineage_unique_count": 0,
                        "lineage_duplicate_count": None,
                        "lineage_unique_equals_gaussian_count": False,
                    }
                )
            rows.append(row)
            del ckpt, state
            gc.collect()
    lineage_hits = _shell_capture(repo, ["rg", "-n", "gaussian_lineage_ids", "water_splatting", "scripts"])
    bnd_rows = [r for r in rows if r.get("run") == "BND" and r.get("status") == "AVAILABLE"]
    all_present = all(bool(r.get("gaussian_lineage_ids_present")) for r in bnd_rows)
    all_unique = all(bool(r.get("lineage_unique_equals_gaussian_count")) for r in bnd_rows)
    if all_present and all_unique and "pop(\"gaussian_lineage_ids\"" not in lineage_hits:
        classification = "GAUSSIAN_LINEAGE_EXACT"
    elif all_present:
        classification = "GAUSSIAN_LINEAGE_PARTIAL"
    else:
        classification = "GAUSSIAN_LINEAGE_NOT_AVAILABLE"
    summary = {
        "classification": classification,
        "exact_identity_recoverable": classification == "GAUSSIAN_LINEAGE_EXACT",
        "lineage_tensor_present_in_bnd_checkpoints": all_present,
        "lineage_unique_per_gaussian_in_all_bnd_checkpoints": all_unique,
        "lineage_source_search": lineage_hits,
        "lineage_interpretation": (
            "Lineage tensors exist but have many duplicate IDs and model.load_state_dict pops gaussian_lineage_ids; "
            "row-wise identity and post-10k birth identity are not safe for counterfactual matching."
        )
        if classification == "GAUSSIAN_LINEAGE_PARTIAL"
        else "",
        "LATE_BIRTH_COUNTERFACTUAL": "NOT_RUN" if classification != "GAUSSIAN_LINEAGE_EXACT" else "ELIGIBLE",
    }
    return rows, summary


def _gaussian_parameter_population_row(model: Any, run: str, nominal_step: int, actual_step: int) -> Dict[str, Any]:
    opacities = torch.sigmoid(model.opacities.detach()).reshape(-1).float().cpu().numpy()
    scales = torch.exp(model.scales.detach()).float()
    scale_max = scales.max(dim=-1).values.cpu().numpy()
    scale_mean = scales.mean(dim=-1).cpu().numpy()
    row: Dict[str, Any] = {
        "run": run,
        "nominal_step": nominal_step,
        "actual_step": actual_step,
        "gaussian_count": int(model.num_points),
        "parameter_scope": "all_gaussians",
    }
    row.update(_np_stats(opacities, "opacity_"))
    row.update(_np_stats(scale_max, "scale_max_"))
    row.update(_np_stats(scale_mean, "scale_mean_"))
    return row


def _snapshot_gaussian_param_groups(model: Any) -> Dict[str, List[Tensor]]:
    return {
        name: [param.detach().clone().cpu() for param in params]
        for name, params in model.get_gaussian_param_groups().items()
    }


def _gaussian_param_delta_max(before: Mapping[str, Sequence[Tensor]], model: Any) -> float:
    max_delta = 0.0
    for name, params in model.get_gaussian_param_groups().items():
        previous = before.get(name, [])
        for idx, param in enumerate(params):
            if idx >= len(previous):
                max_delta = float("inf")
                continue
            after = param.detach().cpu()
            if after.shape != previous[idx].shape:
                max_delta = float("inf")
                continue
            if after.numel():
                max_delta = max(max_delta, float((after - previous[idx]).abs().max().item()))
    return max_delta


def _projection_group_row(
    *,
    run: str,
    nominal_step: int,
    actual_step: int,
    split: str,
    view_id: str,
    group: str,
    selected: np.ndarray,
    visible: np.ndarray,
    in_bounds: np.ndarray,
    safe_pixels: int,
    opacities: np.ndarray,
    scale_max: np.ndarray,
    radii: np.ndarray,
    depths: np.ndarray,
) -> Dict[str, Any]:
    count = int(selected.sum())
    visible_count = int(visible.sum())
    in_bounds_count = int(in_bounds.sum())
    row: Dict[str, Any] = {
        "run": run,
        "nominal_step": nominal_step,
        "actual_step": actual_step,
        "split": split,
        "view_id": view_id,
        "support_group": group,
        "safe_pixel_count": safe_pixels,
        "gaussian_count_selected": count,
        "visible_gaussian_count": visible_count,
        "in_bounds_visible_gaussian_count": in_bounds_count,
        "fraction_of_visible": float(count / max(visible_count, 1)),
        "fraction_of_in_bounds_visible": float(count / max(in_bounds_count, 1)),
        "center_density_per_M_SAFE_pixel": float(count / max(safe_pixels, 1)),
        "footprint_proxy_semantics": "in-bounds projected-center proxy; footprint overlap uses distance_transform_edt(~M_SAFE) <= projected radius and is not exact compositing contribution",
        "center_pixel_rule": "floor(projected x/y) after requiring projected center in image bounds",
    }
    if count == 0:
        return row
    opacity_sel = opacities[selected]
    radius_sel = radii[selected]
    scale_sel = scale_max[selected]
    depth_sel = depths[selected]
    opacity_footprint = opacity_sel * math.pi * np.square(np.maximum(radius_sel, 0.0))
    row.update(
        {
            "opacity_mass": float(opacity_sel.sum()),
            "opacity_footprint_mass": float(opacity_footprint.sum()),
            "mean_opacity_times_radius": float((opacity_sel * np.maximum(radius_sel, 0.0)).mean()),
        }
    )
    row.update(_np_stats(opacity_sel, "opacity_"))
    row.update(_np_stats(scale_sel, "scale_max_"))
    row.update(_np_stats(radius_sel, "projected_radius_"))
    row.update(_np_stats(depth_sel, "projected_depth_"))
    return row


def _projection_proxy_rows(
    *,
    model: Any,
    run: str,
    nominal_step: int,
    actual_step: int,
    split: str,
    view_id: str,
    mask: Tensor,
    distance_to_safe: np.ndarray,
) -> List[Dict[str, Any]]:
    xys = model.xys.detach().float().cpu().numpy()
    radii = model.radii.detach().float().reshape(-1).cpu().numpy()
    depths = model.depths.detach().float().reshape(-1).cpu().numpy()
    opacities = torch.sigmoid(model.opacities.detach()).reshape(-1).float().cpu().numpy()
    scale_max = torch.exp(model.scales.detach()).amax(dim=-1).float().cpu().numpy()
    if xys.shape[0] != opacities.shape[0]:
        raise RuntimeError(f"Projection tensor length {xys.shape[0]} does not match Gaussian count {opacities.shape[0]}")
    mask_np = mask.detach().bool().cpu().numpy()
    h, w = mask_np.shape
    finite_xy = np.isfinite(xys).all(axis=1)
    visible = finite_xy & np.isfinite(radii) & (radii > 0)
    in_bounds = visible & (xys[:, 0] >= 0.0) & (xys[:, 0] < float(w)) & (xys[:, 1] >= 0.0) & (xys[:, 1] < float(h))
    xi = np.zeros(xys.shape[0], dtype=np.int64)
    yi = np.zeros(xys.shape[0], dtype=np.int64)
    idx_in = np.where(in_bounds)[0]
    if idx_in.size:
        xi[idx_in] = np.floor(xys[idx_in, 0]).astype(np.int64).clip(0, w - 1)
        yi[idx_in] = np.floor(xys[idx_in, 1]).astype(np.int64).clip(0, h - 1)
    center_in_safe = np.zeros_like(visible, dtype=bool)
    footprint_overlap = np.zeros_like(visible, dtype=bool)
    if idx_in.size:
        center_in_safe[idx_in] = mask_np[yi[idx_in], xi[idx_in]]
        distances = distance_to_safe[yi[idx_in], xi[idx_in]]
        footprint_overlap[idx_in] = distances <= np.maximum(radii[idx_in], 0.0)
    outside_footprint = footprint_overlap & ~center_in_safe
    groups = {
        "ALL_VISIBLE": visible,
        "IN_BOUNDS_VISIBLE": in_bounds,
        "CENTER_IN_SAFE": center_in_safe,
        "ANY_FOOTPRINT_OVERLAP": footprint_overlap,
        "CENTER_OUTSIDE_BUT_FOOTPRINT_OVERLAP": outside_footprint,
    }
    return [
        _projection_group_row(
            run=run,
            nominal_step=nominal_step,
            actual_step=actual_step,
            split=split,
            view_id=view_id,
            group=group,
            selected=selected,
            visible=visible,
            in_bounds=in_bounds,
            safe_pixels=int(mask_np.sum()),
            opacities=opacities,
            scale_max=scale_max,
            radii=radii,
            depths=depths,
        )
        for group, selected in groups.items()
    ]


def _append_pooled_projection_rows(rows: List[Dict[str, Any]]) -> None:
    base = [r for r in rows if r.get("view_id") != "ALL" and r.get("split") == "train"]
    keys = sorted({(r["run"], r["nominal_step"], r["actual_step"], r["support_group"]) for r in base})
    numeric_sum_keys = {
        "safe_pixel_count",
        "gaussian_count_selected",
        "visible_gaussian_count",
        "in_bounds_visible_gaussian_count",
        "opacity_mass",
        "opacity_footprint_mass",
    }
    numeric_mean_keys = {
        "fraction_of_visible",
        "fraction_of_in_bounds_visible",
        "center_density_per_M_SAFE_pixel",
        "opacity_mean",
        "opacity_median",
        "opacity_p90",
        "opacity_p99",
        "scale_max_mean",
        "scale_max_p90",
        "scale_max_p99",
        "projected_radius_mean",
        "projected_radius_p90",
        "projected_radius_p99",
        "projected_depth_mean",
    }
    for run, nominal, actual, group in keys:
        selected = [r for r in base if (r["run"], r["nominal_step"], r["actual_step"], r["support_group"]) == (run, nominal, actual, group)]
        row: Dict[str, Any] = {
            "run": run,
            "nominal_step": nominal,
            "actual_step": actual,
            "split": "train",
            "view_id": "ALL",
            "support_group": group,
            "pooled_semantics": "sum for mass/count fields; arithmetic view mean for distribution summary fields",
        }
        for key in numeric_sum_keys:
            row[key] = _sum_or_nan(float(r.get(key, float("nan"))) for r in selected)
        for key in numeric_mean_keys:
            row[f"{key}_view_mean"] = _mean_or_nan(float(r.get(key, float("nan"))) for r in selected)
        rows.append(row)


def _candidate_summary_row(
    *,
    run: str,
    nominal_step: int,
    actual_step: int,
    split: str,
    view_id: str,
    mask: Tensor,
    outputs: Mapping[str, Tensor],
    gt: Tensor,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    mask = mask.to(gt.device).bool()
    pred = outputs["pred_image"].float().clamp(0.0, 1.0)
    gt = gt.float().clamp(0.0, 1.0)
    residual = pred - gt
    rgb_mse = residual.square().mean(dim=-1)
    rgb_abs = residual.abs().mean(dim=-1)
    acc = _scalar_map(outputs["accumulation"])
    b_inf = outputs.get("b_inf")
    if isinstance(b_inf, Tensor):
        binf_res = b_inf.float() - gt
        binf_l1 = binf_res.abs().mean(dim=-1)
        binf_signed = binf_res.mean(dim=-1)
    else:
        binf_l1 = torch.full_like(rgb_mse, float("nan"))
        binf_signed = torch.full_like(rgb_mse, float("nan"))
    direct = torch.linalg.norm(outputs.get("direct_object_signal", torch.zeros_like(pred)).float(), dim=-1)
    tau = _scalar_map(outputs.get("tau_D", torch.zeros_like(acc))).float()
    transmission = _scalar_map(outputs.get("transmission", torch.zeros_like(acc))).float()
    j = outputs.get("clear_object_fullsh_raw", torch.zeros_like(pred)).float().amax(dim=-1)
    safe_count = int(mask.sum().item())
    series = {
        "acc": _safe_values(acc, mask),
        "mse": _safe_values(rgb_mse, mask),
        "rgb_abs": _safe_values(rgb_abs, mask),
        "residual_mean": _safe_values(residual.mean(dim=-1), mask),
        "residual_r": _safe_values(residual[..., 0], mask),
        "residual_g": _safe_values(residual[..., 1], mask),
        "residual_b": _safe_values(residual[..., 2], mask),
        "binf_l1": _safe_values(binf_l1, mask),
        "binf_signed": _safe_values(binf_signed, mask),
        "direct_l2": _safe_values(direct, mask),
        "tau": _safe_values(tau, mask),
        "transmission": _safe_values(transmission, mask),
        "j_max": _safe_values(j, mask),
    }
    row: Dict[str, Any] = {
        "run": run,
        "nominal_step": nominal_step,
        "actual_step": actual_step,
        "split": split,
        "view_id": view_id,
        "mask": "M_SAFE",
        "candidate_pixel_count": safe_count,
        "valid_pixel_count": int(mask.numel()),
        "candidate_fraction": float(safe_count / max(mask.numel(), 1)),
    }
    if safe_count == 0:
        return row, series
    row.update(
        {
            "mean_accumulation": float(series["acc"].mean()),
            "p50_accumulation": float(np.quantile(series["acc"], 0.50)),
            "p90_accumulation": float(np.quantile(series["acc"], 0.90)),
            "p99_accumulation": float(np.quantile(series["acc"], 0.99)),
            "fraction_accumulation_gt_0p01": float((series["acc"] > 0.01).mean()),
            "rgb_mse": float(series["mse"].mean()),
            "rgb_psnr_contribution": float(-10.0 * math.log10(max(float(series["mse"].mean()), EPS))),
            "rgb_abs_residual": float(series["rgb_abs"].mean()),
            "rgb_signed_residual_mean": float(series["residual_mean"].mean()),
            "rgb_signed_R_mean": float(series["residual_r"].mean()),
            "rgb_signed_G_mean": float(series["residual_g"].mean()),
            "rgb_signed_B_mean": float(series["residual_b"].mean()),
            "BINF_L1": float(series["binf_l1"].mean()),
            "BINF_signed_residual_mean": float(series["binf_signed"].mean()),
            "direct_object_signal_l2_mean": float(series["direct_l2"].mean()),
            "direct_object_signal_l2_p90": float(np.quantile(series["direct_l2"], 0.90)),
            "tau_mean": float(series["tau"].mean()),
            "tau_p90": float(np.quantile(series["tau"], 0.90)),
            "transmission_mean": float(series["transmission"].mean()),
            "P_transmission_lt_0p1": float((series["transmission"] < 0.1).mean()),
            "J_max_mean": float(series["j_max"].mean()),
            "J_max_p99": float(np.quantile(series["j_max"], 0.99)),
            "P_J_gt_1": float((series["j_max"] > 1.0).mean()),
        }
    )
    return row, series


def _global_rgb_row(
    *,
    model: Any,
    run: str,
    nominal_step: int,
    actual_step: int,
    split: str,
    view_id: str,
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Any],
    gt: Tensor,
) -> Dict[str, Any]:
    pred = outputs["pred_image"].float().clamp(0.0, 1.0)
    gt = gt.float().clamp(0.0, 1.0)
    mse = float((pred - gt).square().mean().item())
    row: Dict[str, Any] = {
        "run": run,
        "nominal_step": nominal_step,
        "actual_step": actual_step,
        "split": split,
        "view_id": view_id,
        "valid_pixel_count": int(pred.shape[0] * pred.shape[1]),
        "mse": mse,
        "psnr_manual": float(-10.0 * math.log10(max(mse, EPS))),
    }
    if split == "eval":
        metrics, _images = model.get_image_metrics_and_images(dict(outputs), dict(batch))
        row.update(
            {
                "psnr": float(metrics.get("psnr", float("nan"))),
                "ssim": float(metrics.get("ssim", float("nan"))),
                "lpips": float(metrics.get("lpips", float("nan"))),
            }
        )
    return row


def _decomposition_safety_row(
    *,
    run: str,
    nominal_step: int,
    actual_step: int,
    split: str,
    view_id: str,
    mask: Optional[Tensor],
    outputs: Mapping[str, Tensor],
) -> Dict[str, Any]:
    if run != "BND":
        return {}
    if mask is None:
        selector = torch.ones(outputs["pred_image"].shape[:2], device=outputs["pred_image"].device, dtype=torch.bool)
        scope = "ALL_PIXELS"
    else:
        selector = mask.to(outputs["pred_image"].device).bool()
        scope = "M_SAFE"
    row: Dict[str, Any] = {
        "run": run,
        "nominal_step": nominal_step,
        "actual_step": actual_step,
        "split": split,
        "view_id": view_id,
        "scope": scope,
        "pixel_count": int(selector.sum().item()),
    }
    if int(selector.sum().item()) == 0:
        return row
    j = outputs["clear_object_fullsh_raw"].float().amax(dim=-1)[selector]
    tau = _scalar_map(outputs["tau_D"]).float()[selector]
    trans = _scalar_map(outputs["transmission"]).float()[selector]
    b_inf = outputs["b_inf"].float()[selector]
    beta_d = outputs["medium_attn"].float()[selector]
    beta_b = outputs["medium_bs"].float()[selector]
    row.update(
        {
            "J_p99": float(torch.quantile(j, 0.99).item()),
            "J_max": float(j.max().item()),
            "P_J_gt_1": float((j > 1.0).float().mean().item()),
            "B_inf_mean": float(b_inf.mean().item()),
            "B_inf_p01": float(torch.quantile(b_inf.reshape(-1), 0.01).item()),
            "B_inf_p99": float(torch.quantile(b_inf.reshape(-1), 0.99).item()),
            "beta_D_mean": float(beta_d.mean().item()),
            "beta_D_p90": float(torch.quantile(beta_d.reshape(-1), 0.90).item()),
            "beta_B_mean": float(beta_b.mean().item()),
            "beta_B_p90": float(torch.quantile(beta_b.reshape(-1), 0.90).item()),
            "tau_p90": float(torch.quantile(tau, 0.90).item()),
            "tau_p99": float(torch.quantile(tau, 0.99).item()),
            "P_T_lt_0p1": float((trans < 0.1).float().mean().item()),
        }
    )
    logits = outputs.get("gaussian_view_logits")
    if isinstance(logits, Tensor) and logits.numel() > 0:
        flat = logits.detach().float().reshape(-1)
        c = torch.sigmoid(flat)
        row["P_c_gt_0p99"] = float((c > 0.99).float().mean().item())
        row["P_abs_s_full_gt_5"] = float((flat.abs() > 5.0).float().mean().item())
    return row


def _append_pooled_rows(rows: List[Dict[str, Any]], *, group_keys: Sequence[str], weighted_key: str = "candidate_pixel_count") -> None:
    base = [r for r in rows if r.get("view_id") != "ALL"]
    keys = sorted({tuple(r.get(k) for k in group_keys) for r in base})
    excluded = set(group_keys) | {"view_id", "mask", "scope"}
    for key_tuple in keys:
        selected = [r for r in base if tuple(r.get(k) for k in group_keys) == key_tuple]
        if not selected:
            continue
        out = {k: v for k, v in zip(group_keys, key_tuple)}
        out["view_id"] = "ALL"
        if "mask" in selected[0]:
            out["mask"] = selected[0].get("mask")
        if "scope" in selected[0]:
            out["scope"] = selected[0].get("scope")
        weights = np.asarray([float(r.get(weighted_key, 1.0)) for r in selected], dtype=np.float64)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            weights = np.ones(len(selected), dtype=np.float64)
        for key in sorted({k for row in selected for k in row.keys()} - excluded - {"view_id"}):
            vals = []
            val_weights = []
            for idx, row in enumerate(selected):
                if key not in row:
                    continue
                try:
                    val = float(row[key])
                except Exception:
                    continue
                if np.isfinite(val):
                    vals.append(val)
                    val_weights.append(weights[idx])
            if vals:
                vals_np = np.asarray(vals, dtype=np.float64)
                w_np = np.asarray(val_weights, dtype=np.float64)
                out[f"{key}_view_mean"] = float(vals_np.mean())
                out[f"{key}_weighted_mean"] = float(np.average(vals_np, weights=w_np))
        rows.append(out)


def _render_run_step(
    *,
    repo: Path,
    run: str,
    nominal_step: int,
    masks: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    distance_maps: Mapping[Tuple[str, str], np.ndarray],
    series_store: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    loaded = None
    try:
        loaded = PW._load_run(repo, run, nominal_step)
        model = loaded.pipeline.model
        actual_step = int(loaded.loaded_step)
        meta = {
            "run": run,
            "nominal_step": nominal_step,
            "actual_step": actual_step,
            "config_path": str(loaded.config_path),
            "checkpoint_path": str(loaded.checkpoint_path),
            "gaussian_count": int(model.num_points),
            "intrinsic_color_parameterization": getattr(model.config, "intrinsic_color_parameterization", ""),
            "medium_context_mode": getattr(model.config, "medium_context_mode", ""),
            "b_inf_mode": getattr(model.config, "b_inf_mode", ""),
            "infinite_water_enabled": getattr(model.config, "infinite_water_enabled", ""),
        }
        population_rows = [_gaussian_parameter_population_row(model, run, nominal_step, actual_step)]
        candidate_rows: List[Dict[str, Any]] = []
        rgb_rows: List[Dict[str, Any]] = []
        eval_rows: List[Dict[str, Any]] = []
        projection_rows: List[Dict[str, Any]] = []
        decomp_rows: List[Dict[str, Any]] = []
        param_before = _snapshot_gaussian_param_groups(model)
        for split, records in PW._records(loaded.pipeline).items():
            for _idx, view_id, camera, batch in records:
                if view_id not in masks[split]:
                    continue
                with torch.no_grad():
                    outputs = model.get_outputs_for_camera(camera.to(model.device))
                    gt = PW._get_gt(model, batch, outputs["background"])
                    cand_row, series = _candidate_summary_row(
                        run=run,
                        nominal_step=nominal_step,
                        actual_step=actual_step,
                        split=split,
                        view_id=view_id,
                        mask=masks[split][view_id]["M_SAFE"],
                        outputs=outputs,
                        gt=gt,
                    )
                    rgb_row = _global_rgb_row(
                        model=model,
                        run=run,
                        nominal_step=nominal_step,
                        actual_step=actual_step,
                        split=split,
                        view_id=view_id,
                        outputs=outputs,
                        batch=batch,
                        gt=gt,
                    )
                    decomp_all = _decomposition_safety_row(
                        run=run,
                        nominal_step=nominal_step,
                        actual_step=actual_step,
                        split=split,
                        view_id=view_id,
                        mask=None,
                        outputs=outputs,
                    )
                    decomp_safe = _decomposition_safety_row(
                        run=run,
                        nominal_step=nominal_step,
                        actual_step=actual_step,
                        split=split,
                        view_id=view_id,
                        mask=masks[split][view_id]["M_SAFE"],
                        outputs=outputs,
                    )
                candidate_rows.append(cand_row)
                rgb_rows.append(rgb_row)
                if split == "eval":
                    eval_row = dict(cand_row)
                    eval_row.update({f"global_{key}": value for key, value in rgb_row.items() if key in {"mse", "psnr_manual", "psnr", "ssim", "lpips"}})
                    eval_rows.append(eval_row)
                if decomp_all:
                    decomp_rows.append(decomp_all)
                if decomp_safe:
                    decomp_rows.append(decomp_safe)
                if split == "train":
                    series_store.setdefault(run, {}).setdefault(nominal_step, {})[view_id] = series
                    projection_rows.extend(
                        _projection_proxy_rows(
                            model=model,
                            run=run,
                            nominal_step=nominal_step,
                            actual_step=actual_step,
                            split=split,
                            view_id=view_id,
                            mask=masks[split][view_id]["M_SAFE"],
                            distance_to_safe=distance_maps[(split, view_id)],
                        )
                    )
                del outputs, gt
                torch.cuda.empty_cache()
        parameter_delta_max = _gaussian_param_delta_max(param_before, model)
        meta["parameter_delta_max_abs_during_readonly_render"] = parameter_delta_max
        meta["AUDIT_PARAMETER_SAFETY"] = "PASS" if parameter_delta_max == 0.0 else "FAIL"
        _append_pooled_rows(candidate_rows, group_keys=("run", "nominal_step", "actual_step", "split"))
        _append_pooled_rows(rgb_rows, group_keys=("run", "nominal_step", "actual_step", "split"), weighted_key="valid_pixel_count")
        _append_pooled_rows(eval_rows, group_keys=("run", "nominal_step", "actual_step", "split"))
        _append_pooled_rows(decomp_rows, group_keys=("run", "nominal_step", "actual_step", "split", "scope"), weighted_key="pixel_count")
        return population_rows, projection_rows, candidate_rows, rgb_rows, eval_rows, decomp_rows, meta
    finally:
        PW._release(loaded)


def _build_distance_maps(masks: Mapping[str, Mapping[str, Mapping[str, Tensor]]]) -> Dict[Tuple[str, str], np.ndarray]:
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for split, views in masks.items():
        for view_id, by_mask in views.items():
            mask_np = by_mask["M_SAFE"].detach().bool().cpu().numpy()
            out[(split, view_id)] = distance_transform_edt(~mask_np).astype(np.float32)
    return out


def _pixel_transition_metrics(
    *,
    run: str,
    view_id: str,
    step_a: int,
    step_b: int,
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    acc_a = first["acc"]
    acc_b = second["acc"]
    mse_a = first["mse"]
    mse_b = second["mse"]
    da = acc_b - acc_a
    de = mse_b - mse_a
    row: Dict[str, Any] = {
        "run": run,
        "view_id": view_id,
        "step_from": step_a,
        "step_to": step_b,
        "transition": f"{step_a}->{step_b}",
        "pixel_count": int(da.size),
        "mean_delta_A": float(da.mean()) if da.size else float("nan"),
        "median_delta_A": float(np.median(da)) if da.size else float("nan"),
        "p90_delta_A": float(np.quantile(da, 0.90)) if da.size else float("nan"),
        "p99_delta_A": float(np.quantile(da, 0.99)) if da.size else float("nan"),
        "fraction_delta_A_gt_0": float((da > 0.0).mean()) if da.size else float("nan"),
        "fraction_crossing_accumulation_0p01": float(((acc_a <= 0.01) & (acc_b > 0.01)).mean()) if da.size else float("nan"),
        "mean_delta_E": float(de.mean()) if de.size else float("nan"),
        "median_delta_E": float(np.median(de)) if de.size else float("nan"),
        "fraction_delta_E_gt_0": float((de > 0.0).mean()) if de.size else float("nan"),
    }
    rho, n, sampled = _spearman_np(da, de)
    row.update({"spearman_delta_A_delta_E": rho, "spearman_n": n, "spearman_sampled": sampled})
    association: Dict[str, Any] = {}
    for key, name in (
        ("binf_l1", "delta_BINF_L1"),
        ("direct_l2", "delta_direct_object_signal_l2"),
        ("tau", "delta_tau"),
        ("transmission", "delta_transmission"),
    ):
        delta = second[key] - first[key]
        rho_k, n_k, sampled_k = _spearman_np(da, delta)
        association[f"spearman_delta_A_{name}"] = rho_k
        association[f"spearman_delta_A_{name}_n"] = n_k
        association[f"spearman_delta_A_{name}_sampled"] = sampled_k
    j_instability = np.abs(second["j_max"] - first["j_max"])
    rho_j, n_j, sampled_j = _spearman_np(da, j_instability)
    association.update(
        {
            "spearman_delta_A_J_instability_abs_delta": rho_j,
            "spearman_delta_A_J_instability_abs_delta_n": n_j,
            "spearman_delta_A_J_instability_abs_delta_sampled": sampled_j,
            "mean_delta_BINF_L1": float((second["binf_l1"] - first["binf_l1"]).mean()) if da.size else float("nan"),
            "mean_delta_direct_object_signal_l2": float((second["direct_l2"] - first["direct_l2"]).mean()) if da.size else float("nan"),
            "mean_delta_tau": float((second["tau"] - first["tau"]).mean()) if da.size else float("nan"),
            "mean_delta_transmission": float((second["transmission"] - first["transmission"]).mean()) if da.size else float("nan"),
            "mean_J_instability_abs_delta": float(j_instability.mean()) if da.size else float("nan"),
        }
    )
    top_rows: List[Dict[str, Any]] = []
    positive_all = np.maximum(de, 0.0)
    total_positive = float(positive_all.sum())
    p_worse_all = float((de > 0.0).mean()) if de.size else float("nan")
    for fraction in TOP_FRACTIONS:
        top = _top_fraction_mask(da, fraction)
        top_de = de[top]
        p_worse_top = float((top_de > 0.0).mean()) if top_de.size else float("nan")
        top_rows.append(
            {
                "run": run,
                "view_id": view_id,
                "step_from": step_a,
                "step_to": step_b,
                "transition": f"{step_a}->{step_b}",
                "top_delta_A_fraction": fraction,
                "selected_pixel_count": int(top.sum()),
                "mean_delta_A_top": float(da[top].mean()) if top.any() else float("nan"),
                "mean_delta_E_top": float(top_de.mean()) if top_de.size else float("nan"),
                "median_delta_E_top": float(np.median(top_de)) if top_de.size else float("nan"),
                "P_delta_E_gt_0_top": p_worse_top,
                "P_delta_E_gt_0_all": p_worse_all,
                "RGB_error_worsening_enrichment": float(p_worse_top / max(p_worse_all, EPS)) if np.isfinite(p_worse_top) and np.isfinite(p_worse_all) else float("nan"),
                "share_total_positive_delta_E": float(np.maximum(top_de, 0.0).sum() / max(total_positive, EPS)) if total_positive > 0.0 else float("nan"),
            }
        )
    return row, top_rows, association


def _concat_series(items: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = items[0].keys()
    return {key: np.concatenate([item[key] for item in items], axis=0) for key in keys}


def _pixel_drift_tables(series_store: Mapping[str, Mapping[int, Mapping[str, Mapping[str, np.ndarray]]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []
    assoc_rows: List[Dict[str, Any]] = []
    bnd = series_store["BND"]
    view_ids = sorted(bnd[3000].keys())
    for step_a, step_b in ALL_TRANSITIONS:
        pooled_first = []
        pooled_second = []
        for view_id in view_ids:
            first = bnd[step_a][view_id]
            second = bnd[step_b][view_id]
            row, top, assoc = _pixel_transition_metrics(run="BND", view_id=view_id, step_a=step_a, step_b=step_b, first=first, second=second)
            rows.append(row)
            top_rows.extend(top)
            assoc_row = {"run": "BND", "view_id": view_id, "step_from": step_a, "step_to": step_b, "transition": f"{step_a}->{step_b}", "pixel_count": row["pixel_count"]}
            assoc_row.update(assoc)
            assoc_rows.append(assoc_row)
            pooled_first.append(first)
            pooled_second.append(second)
        first_all = _concat_series(pooled_first)
        second_all = _concat_series(pooled_second)
        row, top, assoc = _pixel_transition_metrics(run="BND", view_id="ALL", step_a=step_a, step_b=step_b, first=first_all, second=second_all)
        rows.append(row)
        top_rows.extend(top)
        assoc_row = {"run": "BND", "view_id": "ALL", "step_from": step_a, "step_to": step_b, "transition": f"{step_a}->{step_b}", "pixel_count": row["pixel_count"]}
        assoc_row.update(assoc)
        assoc_rows.append(assoc_row)
        del first_all, second_all
        gc.collect()
    return rows, top_rows, assoc_rows


def _contaminated_clean_tables(series_store: Mapping[str, Mapping[int, Mapping[str, Mapping[str, np.ndarray]]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    bnd = series_store["BND"]
    view_ids = sorted(bnd[15000].keys())
    for group_name in ("LATE_CONTAMINATED", "LATE_CLEAN"):
        for step in BND_STEPS:
            pooled: Dict[str, List[np.ndarray]] = {}
            for view_id in view_ids:
                final_acc = bnd[15000][view_id]["acc"]
                group_mask = final_acc > 0.01 if group_name == "LATE_CONTAMINATED" else final_acc <= 0.01
                item = bnd[step][view_id]
                row: Dict[str, Any] = {
                    "run": "BND",
                    "group": group_name,
                    "view_id": view_id,
                    "nominal_step": step,
                    "pixel_count": int(group_mask.sum()),
                    "group_fraction_within_M_SAFE": float(group_mask.mean()) if group_mask.size else float("nan"),
                }
                if group_mask.any():
                    row.update(
                        {
                            "mean_accumulation": float(item["acc"][group_mask].mean()),
                            "rgb_mse": float(item["mse"][group_mask].mean()),
                            "rgb_psnr_contribution": float(-10.0 * math.log10(max(float(item["mse"][group_mask].mean()), EPS))),
                            "rgb_abs_residual": float(item["rgb_abs"][group_mask].mean()),
                            "rgb_signed_residual_mean": float(item["residual_mean"][group_mask].mean()),
                            "rgb_signed_R_mean": float(item["residual_r"][group_mask].mean()),
                            "rgb_signed_G_mean": float(item["residual_g"][group_mask].mean()),
                            "rgb_signed_B_mean": float(item["residual_b"][group_mask].mean()),
                            "BINF_L1": float(item["binf_l1"][group_mask].mean()),
                            "tau_mean": float(item["tau"][group_mask].mean()),
                            "transmission_mean": float(item["transmission"][group_mask].mean()),
                            "direct_object_signal_l2_mean": float(item["direct_l2"][group_mask].mean()),
                            "J_max_mean": float(item["j_max"][group_mask].mean()),
                            "J_max_p99": float(np.quantile(item["j_max"][group_mask], 0.99)),
                        }
                    )
                    for key, values in item.items():
                        pooled.setdefault(key, []).append(values[group_mask])
                rows.append(row)
            pooled_row: Dict[str, Any] = {
                "run": "BND",
                "group": group_name,
                "view_id": "ALL",
                "nominal_step": step,
                "pixel_count": int(sum(arr.size for arr in pooled.get("acc", []))),
            }
            if pooled.get("acc"):
                merged = {key: np.concatenate(chunks) for key, chunks in pooled.items()}
                pooled_row.update(
                    {
                        "mean_accumulation": float(merged["acc"].mean()),
                        "rgb_mse": float(merged["mse"].mean()),
                        "rgb_psnr_contribution": float(-10.0 * math.log10(max(float(merged["mse"].mean()), EPS))),
                        "rgb_abs_residual": float(merged["rgb_abs"].mean()),
                        "rgb_signed_residual_mean": float(merged["residual_mean"].mean()),
                        "rgb_signed_R_mean": float(merged["residual_r"].mean()),
                        "rgb_signed_G_mean": float(merged["residual_g"].mean()),
                        "rgb_signed_B_mean": float(merged["residual_b"].mean()),
                        "BINF_L1": float(merged["binf_l1"].mean()),
                        "tau_mean": float(merged["tau"].mean()),
                        "transmission_mean": float(merged["transmission"].mean()),
                        "direct_object_signal_l2_mean": float(merged["direct_l2"].mean()),
                        "J_max_mean": float(merged["j_max"].mean()),
                        "J_max_p99": float(np.quantile(merged["j_max"], 0.99)),
                    }
                )
                del merged
            rows.append(pooled_row)
    key = {(r["group"], r["view_id"], r["nominal_step"]): r for r in rows}
    for group_name in ("LATE_CONTAMINATED", "LATE_CLEAN"):
        for view_id in [*view_ids, "ALL"]:
            r10 = key.get((group_name, view_id, 10000))
            r15 = key.get((group_name, view_id, 15000))
            if r10 and r15:
                delta_row = {
                    "run": "BND",
                    "group": group_name,
                    "view_id": view_id,
                    "nominal_step": "10000->15000",
                    "row_type": "delta",
                    "pixel_count": r15.get("pixel_count"),
                    "delta_mean_accumulation": float(r15.get("mean_accumulation", float("nan"))) - float(r10.get("mean_accumulation", float("nan"))),
                    "delta_rgb_mse": float(r15.get("rgb_mse", float("nan"))) - float(r10.get("rgb_mse", float("nan"))),
                    "delta_BINF_L1": float(r15.get("BINF_L1", float("nan"))) - float(r10.get("BINF_L1", float("nan"))),
                    "delta_tau_mean": float(r15.get("tau_mean", float("nan"))) - float(r10.get("tau_mean", float("nan"))),
                }
                rows.append(delta_row)
    return rows


def _m1_bnd_comparison(series_store: Mapping[str, Mapping[int, Mapping[str, Mapping[str, np.ndarray]]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    view_ids = sorted(series_store["BND"][10000].keys())
    for run in ("M1", "BND"):
        pooled10 = []
        pooled15 = []
        for view_id in view_ids:
            s10 = series_store[run][10000][view_id]
            s15 = series_store[run][15000][view_id]
            row = {
                "run": run,
                "view_id": view_id,
                "accumulation_10k": float(s10["acc"].mean()),
                "accumulation_15k": float(s15["acc"].mean()),
                "delta_A_10k_to_15k": float((s15["acc"] - s10["acc"]).mean()),
                "fraction_accumulation_gt_0p01_10k": float((s10["acc"] > 0.01).mean()),
                "fraction_accumulation_gt_0p01_15k": float((s15["acc"] > 0.01).mean()),
                "rgb_mse_10k": float(s10["mse"].mean()),
                "rgb_mse_15k": float(s15["mse"].mean()),
                "delta_E_10k_to_15k": float((s15["mse"] - s10["mse"]).mean()),
            }
            rows.append(row)
            pooled10.append(s10)
            pooled15.append(s15)
        s10 = _concat_series(pooled10)
        s15 = _concat_series(pooled15)
        rows.append(
            {
                "run": run,
                "view_id": "ALL",
                "accumulation_10k": float(s10["acc"].mean()),
                "accumulation_15k": float(s15["acc"].mean()),
                "delta_A_10k_to_15k": float((s15["acc"] - s10["acc"]).mean()),
                "fraction_accumulation_gt_0p01_10k": float((s10["acc"] > 0.01).mean()),
                "fraction_accumulation_gt_0p01_15k": float((s15["acc"] > 0.01).mean()),
                "rgb_mse_10k": float(s10["mse"].mean()),
                "rgb_mse_15k": float(s15["mse"].mean()),
                "delta_E_10k_to_15k": float((s15["mse"] - s10["mse"]).mean()),
            }
        )
        del s10, s15
    return rows


def _view_wise_consistency(series_store: Mapping[str, Mapping[int, Mapping[str, Mapping[str, np.ndarray]]]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bnd = series_store["BND"]
    view_ids = sorted(bnd[10000].keys())
    rows: List[Dict[str, Any]] = []
    for view_id in view_ids:
        s10 = bnd[10000][view_id]
        s13 = bnd[13000][view_id]
        s15 = bnd[15000][view_id]
        rows.append(
            {
                "view_id": view_id,
                "accumulation_10k": float(s10["acc"].mean()),
                "accumulation_13k": float(s13["acc"].mean()),
                "accumulation_15k": float(s15["acc"].mean()),
                "delta_A_10k_to_15k": float((s15["acc"] - s10["acc"]).mean()),
                "rgb_mse_10k": float(s10["mse"].mean()),
                "rgb_mse_15k": float(s15["mse"].mean()),
                "delta_E_10k_to_15k": float((s15["mse"] - s10["mse"]).mean()),
                "fraction_delta_A_gt_0_10k_to_15k": float(((s15["acc"] - s10["acc"]) > 0.0).mean()),
            }
        )
    rho, n, sampled = _spearman_np(
        np.asarray([r["delta_A_10k_to_15k"] for r in rows], dtype=np.float64),
        np.asarray([r["delta_E_10k_to_15k"] for r in rows], dtype=np.float64),
    )
    da_pos = np.maximum(np.asarray([r["delta_A_10k_to_15k"] for r in rows], dtype=np.float64), 0.0)
    top3_share = float(np.sort(da_pos)[-3:].sum() / max(float(da_pos.sum()), EPS)) if da_pos.size else float("nan")
    summary = {
        "spearman_across_views_delta_A_delta_E": rho,
        "spearman_n": n,
        "spearman_sampled": sampled,
        "views_positive_delta_A": int(sum(float(r["delta_A_10k_to_15k"]) > 0.0 for r in rows)),
        "view_count": len(rows),
        "top3_views_share_of_positive_delta_A": top3_share,
        "distribution_descriptor": "dominated_by_few_views" if top3_share >= 0.50 else "broadly_distributed",
    }
    rows.append({"view_id": "ALL", **summary})
    return rows, summary


def _hard_region_context(repo: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    requested = ("M1_HIGH_J", "PERSISTENT_BND_HARD", "BND_HARD_CORE")
    sidecars = sorted((repo / "outputs/bnd_rgb_tradeoff_diagnosis_20260809/raw_maps/IUI3").glob("*/residual_and_masks.pt"))
    rows: List[Dict[str, Any]] = []
    available_labels = set()
    sidecar_views = []
    for path in sidecars:
        try:
            data = torch.load(path, map_location="cpu")
            masks = data.get("masks", {}) if isinstance(data, dict) else {}
            labels = sorted(masks.keys()) if isinstance(masks, dict) else []
            available_labels.update(labels)
            sidecar_views.append(path.parent.name)
            rows.append(
                {
                    "sidecar_path": str(path),
                    "view_id": path.parent.name,
                    "available_mask_labels": ";".join(labels),
                    "requested_labels_present": all(label in labels for label in requested),
                    "compatible_with_registered_context": False,
                }
            )
        except Exception as exc:
            rows.append({"sidecar_path": str(path), "status": f"LOAD_FAILED: {exc}"})
    summary = {
        "requested_formal_labels": list(requested),
        "sidecar_count": len(sidecars),
        "sidecar_views": sidecar_views,
        "available_labels": sorted(available_labels),
        "HARD_REGION_CONTEXT": "AVAILABLE" if all(label in available_labels for label in requested) else "NOT_AVAILABLE_FOR_IUI3_COMPATIBLE_LABELS",
        "reason": "Existing IUI3 sidecars expose J1/J95/TAU90/TLOW/COMP, not the registered formal labels; this audit does not redefine them.",
    }
    return rows, summary


def _visualize_late_maps(
    output_dir: Path,
    masks: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    series_store: Mapping[str, Mapping[int, Mapping[str, Mapping[str, np.ndarray]]]],
    view_summary_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    view_rows = [r for r in view_summary_rows if r.get("view_id") != "ALL"]
    selected = sorted(view_rows, key=lambda r: float(r.get("delta_A_10k_to_15k", 0.0)), reverse=True)[:VISUAL_VIEW_LIMIT]
    manifest: List[Dict[str, Any]] = []
    for row in selected:
        view_id = str(row["view_id"])
        mask = masks["train"][view_id]["M_SAFE"].detach().bool().cpu().numpy()
        h, w = mask.shape
        s10 = series_store["BND"][10000][view_id]
        s15 = series_store["BND"][15000][view_id]
        da = np.zeros((h, w), dtype=np.float32)
        de = np.zeros((h, w), dtype=np.float32)
        acc15 = np.zeros((h, w), dtype=np.float32)
        da_vals = s15["acc"] - s10["acc"]
        de_vals = s15["mse"] - s10["mse"]
        da[mask] = da_vals
        de[mask] = de_vals
        acc15[mask] = s15["acc"]

        def gray_panel(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
            scaled = (arr - vmin) / max(vmax - vmin, EPS)
            return (np.clip(scaled, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        da_vmax = float(np.quantile(da_vals, 0.99)) if da_vals.size else 1.0
        de_abs = float(np.quantile(np.abs(de_vals), 0.99)) if de_vals.size else 1.0
        panels = [
            np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=-1),
            np.repeat(gray_panel(acc15, 0.0, max(float(np.quantile(s15["acc"], 0.99)), EPS))[..., None], 3, axis=-1),
            np.repeat(gray_panel(da, 0.0, max(da_vmax, EPS))[..., None], 3, axis=-1),
            np.stack(
                [
                    gray_panel(np.maximum(de, 0.0), 0.0, max(de_abs, EPS)),
                    np.zeros((h, w), dtype=np.uint8),
                    gray_panel(np.maximum(-de, 0.0), 0.0, max(de_abs, EPS)),
                ],
                axis=-1,
            ),
        ]
        resized = []
        for panel in panels:
            image = Image.fromarray(panel)
            scale = min(1.0, 360.0 / float(image.width))
            resized.append(image.resize((int(round(image.width * scale)), int(round(image.height * scale))), Image.Resampling.NEAREST))
        canvas = Image.new("RGB", (sum(p.width for p in resized), max(p.height for p in resized)), (0, 0, 0))
        x = 0
        for panel in resized:
            canvas.paste(panel, (x, 0))
            x += panel.width
        path = visual_dir / f"{view_id}_mask_acc15_deltaA_deltaE_10k_to_15k.png"
        canvas.save(path)
        manifest.append(
            {
                "view_id": view_id,
                "path": str(path.relative_to(output_dir.parent.parent) if output_dir.is_absolute() else path),
                "panel_order": "M_SAFE | BND 15k accumulation | BND delta_A 10k->15k | delta_E red=worse blue=better",
                "display_transform": "per-view p99 scaling for acc/delta_A and symmetric p99(abs(delta_E)) residual scaling",
            }
        )
    index = ["# BND Responsibility Drift Visual Index", ""]
    for item in manifest:
        index.append(f"- `{item['view_id']}`: `{item['path']}` ({item['panel_order']}; {item['display_transform']})")
    (output_dir / "VISUAL_INDEX.md").write_text("\n".join(index) + "\n", encoding="utf8")
    return manifest


def _extract_row(rows: Sequence[Mapping[str, Any]], **criteria: Any) -> Dict[str, Any]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return dict(row)
    return {}


def _classification_summary(
    *,
    pixel_rows: Sequence[Mapping[str, Any]],
    top_rows: Sequence[Mapping[str, Any]],
    assoc_rows: Sequence[Mapping[str, Any]],
    contaminated_rows: Sequence[Mapping[str, Any]],
    view_summary: Mapping[str, Any],
    projection_rows: Sequence[Mapping[str, Any]],
    population_rows: Sequence[Mapping[str, Any]],
    lineage_summary: Mapping[str, Any],
    m1_bnd_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    late = _extract_row(pixel_rows, run="BND", view_id="ALL", step_from=10000, step_to=15000)
    late_10_13 = _extract_row(pixel_rows, run="BND", view_id="ALL", step_from=10000, step_to=13000)
    late_13_15 = _extract_row(pixel_rows, run="BND", view_id="ALL", step_from=13000, step_to=15000)
    late_assoc = _extract_row(assoc_rows, run="BND", view_id="ALL", step_from=10000, step_to=15000)
    top10 = next(
        (
            r
            for r in top_rows
            if r.get("run") == "BND"
            and r.get("view_id") == "ALL"
            and int(r.get("step_from")) == 10000
            and int(r.get("step_to")) == 15000
            and abs(float(r.get("top_delta_A_fraction")) - 0.10) < 1e-6
        ),
        {},
    )
    contam_delta = _extract_row(contaminated_rows, run="BND", group="LATE_CONTAMINATED", view_id="ALL", nominal_step="10000->15000")
    clean_delta = _extract_row(contaminated_rows, run="BND", group="LATE_CLEAN", view_id="ALL", nominal_step="10000->15000")
    substantial = bool(float(late.get("mean_delta_A", 0.0)) > 0.01 or float(late.get("fraction_crossing_accumulation_0p01", 0.0)) > 0.10)
    rgb_assoc_harm = bool(
        float(late.get("spearman_delta_A_delta_E", 0.0)) > 0.10
        and float(top10.get("RGB_error_worsening_enrichment", 0.0)) > 1.10
        and float(top10.get("share_total_positive_delta_E", 0.0)) > float(top10.get("top_delta_A_fraction", 0.10))
    )
    contaminated_worse = bool(
        float(contam_delta.get("delta_rgb_mse", 0.0)) > float(clean_delta.get("delta_rgb_mse", 0.0))
        and float(contam_delta.get("delta_rgb_mse", 0.0)) > 0.0
    )
    decomp_harm = bool(
        float(late_assoc.get("spearman_delta_A_delta_BINF_L1", 0.0)) > 0.10
        and float(late_assoc.get("mean_delta_BINF_L1", 0.0)) > 0.0
    )
    across_views = bool(int(view_summary.get("views_positive_delta_A", 0)) >= max(3, int(0.5 * int(view_summary.get("view_count", 0)))))
    if substantial and across_views and (rgb_assoc_harm or contaminated_worse) and decomp_harm:
        harmfulness = "RESPONSIBILITY_DRIFT_HARMFUL"
    elif substantial and across_views and (rgb_assoc_harm or contaminated_worse or decomp_harm):
        harmfulness = "RESPONSIBILITY_DRIFT_MIXED"
    else:
        harmfulness = "RESPONSIBILITY_DRIFT_NOT_SUPPORTED"

    pop10 = _extract_row(population_rows, run="BND", nominal_step=10000)
    pop15 = _extract_row(population_rows, run="BND", nominal_step=15000)
    total_count_delta = float(pop15.get("gaussian_count", float("nan"))) - float(pop10.get("gaussian_count", float("nan")))

    def proj(step: int, group: str) -> Dict[str, Any]:
        return _extract_row(projection_rows, run="BND", nominal_step=step, split="train", view_id="ALL", support_group=group)

    center10 = proj(10000, "CENTER_IN_SAFE")
    center15 = proj(15000, "CENTER_IN_SAFE")
    outside10 = proj(10000, "CENTER_OUTSIDE_BUT_FOOTPRINT_OVERLAP")
    outside15 = proj(15000, "CENTER_OUTSIDE_BUT_FOOTPRINT_OVERLAP")
    any10 = proj(10000, "ANY_FOOTPRINT_OVERLAP")
    any15 = proj(15000, "ANY_FOOTPRINT_OVERLAP")
    center_count_delta = float(center15.get("gaussian_count_selected", float("nan"))) - float(center10.get("gaussian_count_selected", float("nan")))
    outside_count_delta = float(outside15.get("gaussian_count_selected", float("nan"))) - float(outside10.get("gaussian_count_selected", float("nan")))
    center_opacity_mass_delta = float(center15.get("opacity_mass", float("nan"))) - float(center10.get("opacity_mass", float("nan")))
    outside_opacity_footprint_delta = float(outside15.get("opacity_footprint_mass", float("nan"))) - float(outside10.get("opacity_footprint_mass", float("nan")))
    any_radius_delta = float(any15.get("projected_radius_mean_view_mean", float("nan"))) - float(any10.get("projected_radius_mean_view_mean", float("nan")))
    lineage_exact = bool(lineage_summary.get("exact_identity_recoverable", False))
    if lineage_exact and total_count_delta > 0 and center_count_delta > 0:
        origin = "DRIFT_ORIGIN_TOPOLOGY"
    elif not lineage_exact:
        if abs(center_opacity_mass_delta) > 0.0 or abs(outside_opacity_footprint_delta) > 0.0 or abs(center_count_delta) > 0.0:
            origin = "DRIFT_ORIGIN_UNRESOLVED"
        else:
            origin = "DRIFT_ORIGIN_UNRESOLVED"
    elif center_opacity_mass_delta > 0 and outside_opacity_footprint_delta <= 0 and abs(center_count_delta) < max(abs(float(center10.get("gaussian_count_selected", 0.0))) * 0.05, 1.0):
        origin = "DRIFT_ORIGIN_OPACITY"
    elif outside_opacity_footprint_delta > center_opacity_mass_delta and any_radius_delta > 0:
        origin = "DRIFT_ORIGIN_FOOTPRINT"
    else:
        origin = "DRIFT_ORIGIN_MIXED"

    if harmfulness == "RESPONSIBILITY_DRIFT_HARMFUL" and origin not in {"DRIFT_ORIGIN_MIXED", "DRIFT_ORIGIN_UNRESOLVED"}:
        mechanism = "RESPONSIBILITY_PRESERVATION_READY"
    elif harmfulness in {"RESPONSIBILITY_DRIFT_HARMFUL", "RESPONSIBILITY_DRIFT_MIXED"}:
        mechanism = "RESPONSIBILITY_PRESERVATION_TENTATIVE"
    else:
        mechanism = "RESPONSIBILITY_PRESERVATION_NOT_SUPPORTED"

    m1_all = _extract_row(m1_bnd_rows, run="M1", view_id="ALL")
    bnd_all = _extract_row(m1_bnd_rows, run="BND", view_id="ALL")
    if mechanism == "RESPONSIBILITY_PRESERVATION_NOT_SUPPORTED":
        next_experiment = "BND-MEDIUM-IDENTIFIABILITY-PREFLIGHT"
    elif origin == "DRIFT_ORIGIN_UNRESOLVED":
        next_experiment = "READ-ONLY footprint-vs-opacity decomposition diagnostic with exact renderer proxies, no training"
    elif origin == "DRIFT_ORIGIN_FOOTPRINT":
        next_experiment = "READ-ONLY PREFLIGHT for observability-aware scale / footprint responsibility"
    elif origin == "DRIFT_ORIGIN_OPACITY":
        next_experiment = "READ-ONLY PREFLIGHT for observability-aware opacity responsibility"
    elif origin == "DRIFT_ORIGIN_TOPOLOGY":
        next_experiment = "READ-ONLY / CONTROLLED PREFLIGHT for observability-aware densification responsibility"
    else:
        next_experiment = "READ-ONLY diagnostic resolving mixed opacity/footprint/topology evidence"

    return {
        "late_10k_to_13k": late_10_13,
        "late_13k_to_15k": late_13_15,
        "late_10k_to_15k": late,
        "late_10k_to_15k_associations": late_assoc,
        "top10_delta_A_10k_to_15k": top10,
        "late_contaminated_delta": contam_delta,
        "late_clean_delta": clean_delta,
        "late_accumulation_growth_substantial": substantial,
        "rgb_association_harm_rule": rgb_assoc_harm,
        "late_contaminated_worse_rule": contaminated_worse,
        "decomposition_harm_rule": decomp_harm,
        "across_views_rule": across_views,
        "harmfulness_classification": harmfulness,
        "origin_classification": origin,
        "responsibility_preservation_classification": mechanism,
        "origin_proxy_summary": {
            "total_gaussian_count_delta_10k_to_15k": total_count_delta,
            "center_in_safe_count_delta_10k_to_15k": center_count_delta,
            "center_in_safe_opacity_mass_delta_10k_to_15k": center_opacity_mass_delta,
            "outside_footprint_count_delta_10k_to_15k": outside_count_delta,
            "outside_footprint_opacity_footprint_mass_delta_10k_to_15k": outside_opacity_footprint_delta,
            "any_footprint_projected_radius_mean_delta_10k_to_15k": any_radius_delta,
            "lineage_exact": lineage_exact,
        },
        "m1_bnd_pooled": {
            "M1": m1_all,
            "BND": bnd_all,
            "BND_minus_M1_delta_A_10k_to_15k": float(bnd_all.get("delta_A_10k_to_15k", float("nan"))) - float(m1_all.get("delta_A_10k_to_15k", float("nan"))),
            "BND_minus_M1_final_fraction_acc_gt_0p01": float(bnd_all.get("fraction_accumulation_gt_0p01_15k", float("nan"))) - float(m1_all.get("fraction_accumulation_gt_0p01_15k", float("nan"))),
        },
        "next_single_experiment": next_experiment,
        "RESPONSIBILITY_PRESERVATION_BY_OCCUPANCY": "CLOSED" if harmfulness == "RESPONSIBILITY_DRIFT_NOT_SUPPORTED" else "OPEN_FOR_PREFLIGHT_ONLY",
    }


def _write_research_note(path: Path, summary: Mapping[str, Any]) -> None:
    cls = summary["classification"]
    env = summary["environment"]
    repo = summary["repo"]
    lineage = summary["gaussian_lineage"]
    hard = summary["hard_region_context"]
    lines = [
        "# BND-RESPONSIBILITY-DRIFT-AUDIT-IUI3",
        "",
        "## Scope",
        "CONFIG FACT: This is a read-only, zero-training diagnostic. No optimizer, parameter update, new loss, new module, threshold sweep, CUDA edit, checkpoint write, or training intervention is used.",
        "",
        "## Repo",
        f"EXPERIMENTAL FACT: Branch `{repo['branch']}`, HEAD `{repo['head']}`.",
        f"EXPERIMENTAL FACT: script-run pre-staging status was `{repo['status_short']}`.",
        "",
        "## Environment",
        f"EXPERIMENTAL FACT: `CONDA_ENV={env['CONDA_ENV']}`, `PYTHON_PATH={env['PYTHON_PATH']}`, `PYTHON_VERSION={env['PYTHON_VERSION']}`, `TORCH_VERSION={env['TORCH_VERSION']}`.",
        f"EXPERIMENTAL FACT: `CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}` maps torch logical cuda:0 to physical GPU `{env['gpu']['physical_gpu_id']}` (`{env['gpu']['gpu_name']}`).",
        "",
        "## Locked Candidate Semantics",
        "CONFIG FACT: `M_SAFE = erode_5px(M_SF & (BND@3000 accumulation <= 0.01))`; `M_SF` is the pseudo-depth background candidate from per-image max-normalized `depthAnything_u16`, threshold `1e-2`, largest filled foreground component, and complement background.",
        "CONFIG FACT: The candidate mask is fixed across all later checkpoints and is not treated as true water.",
        "",
        "## Accumulation Semantics",
        "CODE FACT: `outputs['accumulation']` is per-pixel `1 - final_Ts`, the screen-space transmittance complement from Gaussian alpha compositing.",
        "CODE FACT: It is not Gaussian opacity alpha_i and not exact per-Gaussian contribution; medium does not itself add alpha.",
        "",
        "## Gaussian Topology / Lineage",
        f"CODE FACT: Lineage classification `{lineage['classification']}`.",
        f"INFERENCE: Late-birth counterfactual `{lineage['LATE_BIRTH_COUNTERFACTUAL']}` because exact row-wise identity is not recoverable.",
        "",
        "## Quantitative Results",
        f"QUANTITATIVE RESULT: 10k->13k pooled mean delta_A `{cls['late_10k_to_13k'].get('mean_delta_A')}`, crossing fraction `{cls['late_10k_to_13k'].get('fraction_crossing_accumulation_0p01')}`.",
        f"QUANTITATIVE RESULT: 13k->15k pooled mean delta_A `{cls['late_13k_to_15k'].get('mean_delta_A')}`, crossing fraction `{cls['late_13k_to_15k'].get('fraction_crossing_accumulation_0p01')}`.",
        f"QUANTITATIVE RESULT: 10k->15k pooled Spearman(delta_A, delta_E) `{cls['late_10k_to_15k'].get('spearman_delta_A_delta_E')}`.",
        f"QUANTITATIVE RESULT: top-10% delta_A RGB worsening enrichment `{cls['top10_delta_A_10k_to_15k'].get('RGB_error_worsening_enrichment')}`, positive delta_E share `{cls['top10_delta_A_10k_to_15k'].get('share_total_positive_delta_E')}`.",
        f"QUANTITATIVE RESULT: late contaminated delta RGB MSE `{cls['late_contaminated_delta'].get('delta_rgb_mse')}`; late clean delta RGB MSE `{cls['late_clean_delta'].get('delta_rgb_mse')}`.",
        f"QUANTITATIVE RESULT: Spearman(delta_A, delta_BINF_L1) `{cls['late_10k_to_15k_associations'].get('spearman_delta_A_delta_BINF_L1')}`; Spearman(delta_A, delta_tau) `{cls['late_10k_to_15k_associations'].get('spearman_delta_A_delta_tau')}`.",
        f"QUANTITATIVE RESULT: M1/BND pooled comparison `{cls['m1_bnd_pooled']}`.",
        f"QUANTITATIVE RESULT: origin proxy summary `{cls['origin_proxy_summary']}`.",
        "",
        "## Hard-Region Context",
        f"EXPERIMENTAL FACT: `{hard['HARD_REGION_CONTEXT']}`. {hard['reason']}",
        "",
        "## Classifications",
        f"INFERENCE: Harmfulness `{cls['harmfulness_classification']}`.",
        f"INFERENCE: Origin `{cls['origin_classification']}`.",
        f"INFERENCE: Responsibility preservation `{cls['responsibility_preservation_classification']}`.",
        "",
        "## Scientific Interpretation",
        "INFERENCE: The audit evaluates late candidate-region object occupation as a hypothesis, not as confirmed misattribution. It does not claim true color, true geometry, or exact per-Gaussian responsibility.",
        "",
        "## Next Single Experiment",
        f"RECOMMENDATION: `{cls['next_single_experiment']}`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def run(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    os.chdir(repo)
    output_dir = repo / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_manifest = _assert_runtime_policy()
    env_manifest = _environment_manifest(gpu_manifest)
    repo_manifest = _repo_manifest(repo)
    _write_json(output_dir / "repo_manifest.json", repo_manifest)
    _write_json(output_dir / "environment_manifest.json", env_manifest)
    _write_json(output_dir / "gpu_manifest.json", gpu_manifest)

    accum_semantics = _accumulation_semantics()
    topology_semantics = _checkpoint_topology_semantics(repo)
    _write_json(output_dir / "accumulation_semantics.json", accum_semantics)
    _write_json(output_dir / "checkpoint_topology_semantics.json", topology_semantics)
    (output_dir / "accumulation_semantics.md").write_text(
        "# Accumulation Semantics\n\n" + "\n".join(f"- {key}: {value}" for key, value in accum_semantics.items()) + "\n",
        encoding="utf8",
    )
    (output_dir / "checkpoint_topology_semantics.md").write_text(
        "# Checkpoint / Gaussian Topology Semantics\n\n" + "\n".join(f"- {key}: {value}" for key, value in topology_semantics.items()) + "\n",
        encoding="utf8",
    )

    checkpoint_rows, checkpoint_safety = _checkpoint_manifest(repo)
    storage_rows, lineage_summary = _checkpoint_storage_rows(repo)
    _write_csv(output_dir / "checkpoint_manifest.csv", checkpoint_rows)
    _write_json(output_dir / "checkpoint_manifest.json", {"rows": checkpoint_rows})
    _write_csv(output_dir / "checkpoint_storage_and_lineage.csv", storage_rows)
    _write_json(output_dir / "checkpoint_storage_and_lineage.json", {"rows": storage_rows, "summary": lineage_summary})
    _write_json(output_dir / "gaussian_lineage_availability.json", lineage_summary)

    step3_maps, step3_meta = PW._render_split_maps(repo, "BND", 3000)
    masks, mask_meta = PW._build_masks(step3_maps)
    coverage_rows = PW._coverage_rows(masks)
    distance_maps = _build_distance_maps(masks)
    train_views = tuple(step3_maps["train"].keys())
    eval_views = tuple(step3_maps["eval"].keys())
    data_availability = PW._data_availability_manifest(repo, train_views, eval_views)
    locked_semantics = {
        "scene": SCENE,
        "mask_definitions": PW._source_semantics()["candidate_mask_definition"],
        "fixed_thresholds": {
            "pseudo_depth_foreground_threshold": 1e-2,
            "low_support_accumulation_max": 0.01,
            "safe_mask_erosion_radius_px": 5,
            "largest_component_rule": True,
            "threshold_sweep": False,
        },
        "pseudo_depth_source": str(PW.DEPTHS_PATH),
        "mask_meta": mask_meta,
        "bnd_step3_meta": step3_meta,
        "train_views": list(train_views),
        "eval_views": list(eval_views),
        "data_availability": data_availability,
    }
    _write_json(output_dir / "locked_candidate_semantics.json", locked_semantics)
    _write_csv(output_dir / "locked_candidate_coverage.csv", coverage_rows)
    _write_json(output_dir / "locked_candidate_coverage.json", {"rows": coverage_rows})
    del step3_maps
    gc.collect()
    torch.cuda.empty_cache()

    series_store: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]] = {}
    population_rows: List[Dict[str, Any]] = []
    projection_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    rgb_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    decomp_rows: List[Dict[str, Any]] = []
    run_meta_rows: List[Dict[str, Any]] = []

    for run_name, steps in (("BND", BND_STEPS), ("M1", M1_STEPS)):
        for step in steps:
            pop, proj, cand, rgb, eval_harm, decomp, meta = _render_run_step(
                repo=repo,
                run=run_name,
                nominal_step=step,
                masks=masks,
                distance_maps=distance_maps,
                series_store=series_store,
            )
            population_rows.extend(pop)
            projection_rows.extend(proj)
            candidate_rows.extend(cand)
            rgb_rows.extend(rgb)
            eval_rows.extend(eval_harm)
            decomp_rows.extend(decomp)
            run_meta_rows.append(meta)
            _write_json(output_dir / f"progress_{run_name}_{step}.json", {"meta": meta})
    _append_pooled_projection_rows(projection_rows)
    _write_csv(output_dir / "temporal_gaussian_population.csv", population_rows)
    _write_json(output_dir / "temporal_gaussian_population.json", {"rows": population_rows})
    _write_csv(output_dir / "projection_population_proxy.csv", projection_rows)
    _write_json(output_dir / "projection_population_proxy.json", {"rows": projection_rows})
    _write_csv(output_dir / "candidate_region_trajectories.csv", candidate_rows)
    _write_json(output_dir / "candidate_region_trajectories.json", {"rows": candidate_rows})
    _write_csv(output_dir / "global_rgb_metrics.csv", rgb_rows)
    _write_json(output_dir / "global_rgb_metrics.json", {"rows": rgb_rows})
    _write_csv(output_dir / "eval_view_harmfulness.csv", eval_rows)
    _write_json(output_dir / "eval_view_harmfulness.json", {"rows": eval_rows})
    _write_csv(output_dir / "decomposition_safety.csv", decomp_rows)
    _write_json(output_dir / "decomposition_safety.json", {"rows": decomp_rows})
    _write_csv(output_dir / "run_step_manifest.csv", run_meta_rows)
    _write_json(output_dir / "run_step_manifest.json", {"rows": run_meta_rows})
    _write_json(output_dir / "checkpoint_safety.json", {"CHECKPOINT_SAFETY": "PASS", "rows": checkpoint_rows, "before": checkpoint_safety})

    pixel_rows, top_rows, assoc_rows = _pixel_drift_tables(series_store)
    contaminated_rows = _contaminated_clean_tables(series_store)
    m1_bnd_rows = _m1_bnd_comparison(series_store)
    view_rows, view_summary = _view_wise_consistency(series_store)
    hard_rows, hard_summary = _hard_region_context(repo)
    classification = _classification_summary(
        pixel_rows=pixel_rows,
        top_rows=top_rows,
        assoc_rows=assoc_rows,
        contaminated_rows=contaminated_rows,
        view_summary=view_summary,
        projection_rows=projection_rows,
        population_rows=population_rows,
        lineage_summary=lineage_summary,
        m1_bnd_rows=m1_bnd_rows,
    )
    visual_manifest = _visualize_late_maps(output_dir, masks, series_store, view_rows)

    _write_csv(output_dir / "pixel_responsibility_drift.csv", pixel_rows)
    _write_json(output_dir / "pixel_responsibility_drift.json", {"rows": pixel_rows})
    _write_csv(output_dir / "top_delta_a_rgb_harm.csv", top_rows)
    _write_json(output_dir / "top_delta_a_rgb_harm.json", {"rows": top_rows})
    _write_csv(output_dir / "medium_decomposition_harmfulness.csv", assoc_rows)
    _write_json(output_dir / "medium_decomposition_harmfulness.json", {"rows": assoc_rows})
    _write_csv(output_dir / "late_contaminated_vs_clean_trajectories.csv", contaminated_rows)
    _write_json(output_dir / "late_contaminated_vs_clean_trajectories.json", {"rows": contaminated_rows})
    _write_csv(output_dir / "m1_bnd_drift_comparison.csv", m1_bnd_rows)
    _write_json(output_dir / "m1_bnd_drift_comparison.json", {"rows": m1_bnd_rows})
    _write_csv(output_dir / "view_wise_consistency.csv", view_rows)
    _write_json(output_dir / "view_wise_consistency.json", {"rows": view_rows, "summary": view_summary})
    _write_csv(output_dir / "hard_region_context.csv", hard_rows)
    _write_json(output_dir / "hard_region_context.json", hard_summary)
    _write_json(output_dir / "visual_manifest.json", {"rows": visual_manifest})

    summary = {
        "repo": repo_manifest,
        "environment": env_manifest,
        "gpu": gpu_manifest,
        "locked_candidate_semantics": locked_semantics,
        "accumulation_semantics": accum_semantics,
        "checkpoint_topology_semantics": topology_semantics,
        "gaussian_lineage": lineage_summary,
        "hard_region_context": hard_summary,
        "view_wise_summary": view_summary,
        "classification": classification,
        "outputs": sorted(str(path.relative_to(repo)) for path in output_dir.glob("*")),
        "read_only_parameter_safety": "PASS" if all(float(row.get("parameter_delta_max_abs_during_readonly_render", 0.0)) == 0.0 for row in run_meta_rows) else "FAIL",
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
