#!/usr/bin/env python3
"""Read-only IUI3 medium-identifiability preflight for formal M1/BND.

This script loads existing checkpoints, exposes the per-pixel 9-D medium MLP
pre-activation state for diagnostics, and estimates aggregate structured RGB
sensitivity to shared medium-output perturbations. It does not create an
optimizer, update parameters, modify checkpoints, or train a new module.
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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.cameras.cameras import Cameras
from water_splatting.fields import compute_bounded_gaussian_colors, compute_gaussian_colors


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

SCENE = "IUI3-RedSea"
OUTPUT_DIR = Path("outputs/bnd_medium_identifiability_preflight_iui3_20260825")
RESEARCH_NOTE = Path("research_notes/BND_MEDIUM_IDENTIFIABILITY_PREFLIGHT_IUI3_2026-08-25.md")
MATCHED_STEPS = (5000, 10000, 15000)
RUNS = ("M1", "BND")
POPULATIONS = ("GENERAL", "M_SAFE")
SAMPLES_PER_VIEW = 1024
RNG_SEED = 20260825
COUNTERFACTUAL_EPSILON = 0.25
RANDOM_DIRECTIONS = 8
ALLOWED_PHYSICAL_GPUS = frozenset({"6", "7", "8", "9"})
EPS = 1e-12


@dataclass
class ViewSample:
    view_id: str
    height: int
    width: int
    general_flat: Tensor
    safe_flat: Tensor
    safe_available_pixels: int

    def flat_for(self, population: str) -> Tensor:
        if population == "GENERAL":
            return self.general_flat
        if population == "M_SAFE":
            return self.safe_flat
        raise ValueError(population)


@dataclass
class PopAnalysis:
    z: Tensor
    activated: Tensor
    depth: Tensor
    tau: Tensor
    transmission: Tensor
    rgb: Tensor
    gt: Tensor
    rgb_residual: Tensor
    local_jacobian: Tensor
    view_slices: Dict[str, Tuple[int, int]]
    scale: Tensor
    covariance: Tensor
    correlation: Tensor
    pca_eigvals: Tensor
    pca_eigvecs: Tensor
    covariance_std: Tensor
    pca_std_eigvals: Tensor
    pca_std_eigvecs: Tensor
    gram: Tensor
    eigvals: Tensor
    eigvecs: Tensor
    singular_values_per_sqrt_ray: Tensor
    v_min: Tensor
    v_max: Tensor
    sigma_ratio: float
    condition_number: float
    effective_rank: float


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
        raise RuntimeError("CUDA must be available for this diagnostic.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one torch-visible GPU after masking, got {torch.cuda.device_count()}")
    props = torch.cuda.get_device_properties(0)
    return {
        "CUDA_VISIBLE_DEVICES": visible,
        "physical_gpu_id": visible,
        "torch_logical_gpu_id": 0,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "gpu_name": props.name,
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


def _medium_source_semantics(repo: Path) -> Dict[str, Any]:
    return {
        "CODE_FACT": True,
        "medium_field_source": "water_splatting/fields/medium_field.py",
        "model_source": "water_splatting/water_splatting.py",
        "rasterizer_source": "water_splatting/rendering/underwater_rasterizer.py and water_splatting/cuda/csrc/forward.cu",
        "formal_config": {
            "medium_context_mode": "dir_xy_camera",
            "b_inf_mode": "tied",
            "infinite_water_enabled": False,
            "intrinsic_color_parameterization_BND": "bounded_sh3",
            "intrinsic_color_parameterization_M1": "legacy",
            "rasterize_mode": "classic",
        },
        "medium_mlp": {
            "input_dimension": 22,
            "input_parts": {
                "direction_encoding": 16,
                "xy_context": 3,
                "camera_context": 3,
            },
            "raw_output_tensor": "medium_base_out / z_med with shape [H*W, 9]",
            "preactivation_access": "diagnostic script reconstructs the same MLP input and reads medium_mlp output before activations; training code is unchanged",
            "output_channels": {
                "0:3": "B_inf / medium_rgb logits",
                "3:6": "beta_B / medium_bs logits",
                "6:9": "beta_D / medium_attn logits",
            },
            "activations": {
                "B_inf_medium_rgb": "sigmoid(z_med[...,0:3])",
                "beta_B_medium_bs": "softplus(z_med[...,3:6] + medium_density_bias)",
                "beta_D_medium_attn": "softplus(z_med[...,6:9] + medium_density_bias)",
            },
            "context_semantics": {
                "direction": "per-pixel camera ray direction, normalized in camera space and rotated by camera_to_world @ diag(1,-1,-1)",
                "xy": "image_x, image_y in [-1,1] plus radius sqrt(x^2+y^2)",
                "camera": "(camera_center - scene_center) / (scene_scale + 1e-6), scaled by medium_camera_context_scale",
                "per_pixel_or_ray": "medium values are dense per-pixel/per-ray maps",
            },
        },
        "image_formation": {
            "direct_object": "CUDA accumulates alpha*T*GaussianColor*exp(-beta_D*depth) into rgb_object/direct_object_signal",
            "clear_object_raw": "CUDA also accumulates alpha*T*GaussianColor without medium attenuation into clear_object_fullsh_raw",
            "medium_between_gaussians": "for each depth interval, add T * (exp(-beta_B*prev_depth)-exp(-beta_B*depth)) * medium_rgb",
            "tail_medium": "CUDA adds final_T * exp(-beta_B*last_depth) * medium_rgb",
            "python_tied_tail_recomposition": "for b_inf_mode='tied', b_inf=medium_rgb and rgb_medium is recomposed as rgb_medium_finite + final_T*exp(-beta_B*last_depth)*b_inf; because b_inf equals medium_rgb this preserves rendered RGB",
            "pred_image": "pred_image/rgb = rgb_object + rgb_medium",
            "tau_D": "medium_attn * render.depth",
            "transmission": "exp(-tau_D.clamp_min(0)).clamp(0,1)",
            "depth": "alpha-weighted expected depth depth_im / accumulation, with max-depth fallback when accumulation is zero",
        },
        "source_hits": {
            "medium_field": _shell_capture(
                repo,
                [
                    "rg",
                    "-n",
                    "medium_base_out|medium_rgb|medium_bs|medium_attn|b_inf|_append_context",
                    "water_splatting/fields/medium_field.py",
                ],
            ),
            "model_recomposition": _shell_capture(
                repo,
                [
                    "rg",
                    "-n",
                    "tail_weight|rgb_tail|rgb_medium_finite|tau_D|transmission|pred_image|_predict_medium",
                    "water_splatting/water_splatting.py",
                ],
            ),
            "cuda_forward": _shell_capture(
                repo,
                [
                    "rg",
                    "-n",
                    "exp_obj|exp_bs|pix_medium|final_medium|out_med|final_Ts|last_depth",
                    "water_splatting/cuda/csrc/forward.cu",
                ],
            ),
        },
    }


def _checkpoint_mapping(repo: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[int, int]]]:
    rows: List[Dict[str, Any]] = []
    actual_by_run: Dict[str, Dict[int, int]] = {}
    for run in RUNS:
        rel_config, parameterization = PW._run_config(run)
        config_path = repo / rel_config
        actual_by_run[run] = {}
        available = PW._available_steps(config_path)
        for nominal in MATCHED_STEPS:
            actual = PW._actual_step(config_path, nominal)
            actual_by_run[run][nominal] = int(actual) if actual is not None else -1
            ckpt = available.get(actual) if actual is not None else None
            rows.append(
                {
                    "run": run,
                    "parameterization": parameterization,
                    "nominal_step": nominal,
                    "actual_step": int(actual) if actual is not None else "",
                    "available": bool(actual is not None),
                    "config_path": str(config_path),
                    "checkpoint_path": str(ckpt) if ckpt is not None else "",
                }
            )
    return rows, actual_by_run


def _camera_geometry(model: Any, camera: Cameras) -> Tuple[Cameras, Tensor, float, float, int, int]:
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


def _medium_raw_for_camera(model: Any, camera: Cameras) -> Tuple[Tensor, int, int]:
    camera, rotation, cx, cy, height, width = _camera_geometry(model, camera)
    dtype = rotation.dtype
    device = rotation.device
    y = torch.linspace(0.0, height, height, device=device, dtype=dtype)
    x = torch.linspace(0.0, width, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yy = (yy - cy) / camera.fy.item()
    xx = (xx - cx) / camera.fx.item()
    directions = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True).clamp_min(1e-12)
    directions = directions @ rotation.T
    directions_encoded = model.direction_encoding(directions.reshape(-1, 3))

    image_y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    image_x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    image_yy, image_xx = torch.meshgrid(image_y, image_x, indexing="ij")
    radius = torch.sqrt(image_xx.square() + image_yy.square())
    xy_context = torch.stack([image_xx, image_yy, radius], dim=-1).reshape(-1, 3)

    mode = getattr(model.config, "medium_context_mode", "dir_only")
    if mode == "dir_only":
        mlp_input = directions_encoded
    elif mode == "dir_xy":
        mlp_input = torch.cat([directions_encoded, xy_context], dim=-1)
    elif mode == "dir_xy_camera":
        scene_center, scene_scale = model._get_scene_normalization(dtype=dtype, device=device)
        camera_feature = camera.camera_to_worlds[0, :3, 3].to(device=device, dtype=dtype)
        camera_feature = (camera_feature - scene_center) / (scene_scale + 1e-6)
        camera_feature = camera_feature * float(getattr(model.config, "medium_camera_context_scale", 1.0))
        camera_context = camera_feature.reshape(1, 3).expand(height * width, 3)
        mlp_input = torch.cat([directions_encoded, xy_context, camera_context], dim=-1)
    else:
        raise ValueError(f"Unsupported medium_context_mode: {mode}")

    if model.config.mlp_type == "tcnn":
        raw = model.medium_mlp(mlp_input.contiguous())
    else:
        raw = model.medium_mlp(mlp_input.float().contiguous())
    return raw, height, width


def _activate_medium(model: Any, raw: Tensor, height: int, width: int) -> Dict[str, Tensor]:
    density_bias = float(getattr(model, "medium_density_bias", 0.0))
    medium_rgb = torch.sigmoid(raw[..., :3]).view(height, width, 3).float()
    medium_bs = F.softplus(raw[..., 3:6] + density_bias).view(height, width, 3).float()
    medium_attn = F.softplus(raw[..., 6:9] + density_bias).view(height, width, 3).float()
    return {"medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn, "b_inf": medium_rgb}


def _active_sh_degree(model: Any) -> int:
    return int(min(model.step // model.config.sh_degree_interval, model.config.sh_degree))


def _colors_for_current_parameterization(
    model: Any,
    camera: Cameras,
    means: Tensor,
    features_dc: Tensor,
    features_rest: Tensor,
) -> Tensor:
    active = _active_sh_degree(model)
    parameterization = getattr(model.config, "intrinsic_color_parameterization", "legacy")
    if parameterization == "legacy":
        return compute_gaussian_colors(
            means=means,
            features_dc=features_dc,
            features_rest=features_rest,
            camera_position=camera.camera_to_worlds[..., :3, 3],
            sh_degree=model.config.sh_degree,
            active_sh_degree=active,
        )
    if parameterization == "bounded_sh3":
        return compute_bounded_gaussian_colors(
            means=means,
            features_dc=features_dc,
            features_rest=features_rest,
            camera_position=camera.camera_to_worlds[..., :3, 3],
            sh_degree=model.config.sh_degree,
            active_sh_degree=active,
        ).rgb
    raise ValueError(f"Unsupported intrinsic parameterization: {parameterization}")


def _render_with_medium_override(
    model: Any,
    camera: Cameras,
    medium_rgb: Tensor,
    medium_bs: Tensor,
    medium_attn: Tensor,
    *,
    detach_object_state: bool = True,
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

        if detach_object_state:
            means_project = means_crop.detach()
            scales_project = scales_crop.detach()
            quats_project = quats_crop.detach()
        else:
            means_project = means_crop
            scales_project = scales_crop
            quats_project = quats_crop

        with torch.no_grad() if detach_object_state else _nullcontext():
            xys, depths, radii, conics, comp, num_tiles_hit, _ = model.underwater_rasterizer.project(
                means=means_project,
                scales=scales_project,
                quats=quats_project,
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
            "clear_object_fullsh_raw": clear,
        }

    with torch.no_grad() if detach_object_state else _nullcontext():
        color_means = means_crop.detach() if detach_object_state else means_crop
        color_dc = features_dc_crop.detach() if detach_object_state else features_dc_crop
        color_rest = features_rest_crop.detach() if detach_object_state else features_rest_crop
        colors = _colors_for_current_parameterization(model, camera, color_means, color_dc, color_rest)
        colors = colors.detach() if detach_object_state else colors
        if model.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop.detach() if detach_object_state else opacities_crop) * comp[:, None]
        elif model.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop.detach() if detach_object_state else opacities_crop)
        else:
            raise ValueError(f"Unknown rasterize_mode: {model.config.rasterize_mode}")

    xys_grad_abs = torch.zeros_like(xys)
    render = model.underwater_rasterizer.rasterize(
        xys=xys.detach() if detach_object_state else xys,
        xys_grad_abs=xys_grad_abs,
        depths=depths.detach() if detach_object_state else depths,
        radii=radii.detach() if detach_object_state else radii,
        conics=conics.detach() if detach_object_state else conics,
        num_tiles_hit=num_tiles_hit,
        colors=colors,
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
    rgb_tail = torch.zeros_like(rgb_medium)
    if getattr(model.config, "b_inf_mode", "implicit") == "tied":
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


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: Any) -> bool:
        return False


def _flatten_sample(value: Tensor, flat: Tensor) -> Tensor:
    if flat.numel() == 0:
        tail = tuple(value.shape[2:]) if value.ndim > 2 else ()
        return value.new_empty((0, *tail))
    return value.reshape(-1, *value.shape[2:])[flat.to(value.device)]


def _scalarize_rgb(value: Tensor) -> Tensor:
    value = value.float()
    if value.ndim == 2:
        return value
    if value.ndim == 3 and value.shape[-1] == 1:
        return value[..., 0]
    if value.ndim == 3 and value.shape[-1] == 3:
        return value.mean(dim=-1)
    raise ValueError(f"Cannot scalarize tensor with shape {tuple(value.shape)}")


def _stat_dict(values: Tensor, prefix: str) -> Dict[str, Any]:
    vals = values.detach().float().reshape(-1).cpu()
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        return {f"{prefix}_{key}": float("nan") for key in ("mean", "std", "p01", "p50", "p99")}
    return {
        f"{prefix}_mean": float(vals.mean().item()),
        f"{prefix}_std": float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0,
        f"{prefix}_p01": float(torch.quantile(vals, 0.01).item()),
        f"{prefix}_p50": float(torch.quantile(vals, 0.50).item()),
        f"{prefix}_p99": float(torch.quantile(vals, 0.99).item()),
    }


def _per_dim_stats(matrix: Tensor, names: Sequence[str], prefix: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for idx, name in enumerate(names):
        out.update(_stat_dict(matrix[:, idx], f"{prefix}_{name}"))
    return out


def _cov_corr_pca(matrix: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    x = matrix.detach().double()
    if x.shape[0] <= 1:
        cov = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float64)
    else:
        xc = x - x.mean(dim=0, keepdim=True)
        cov = xc.T @ xc / max(int(x.shape[0]) - 1, 1)
    std = torch.sqrt(torch.diag(cov).clamp_min(0.0))
    corr = cov / (std[:, None].clamp_min(EPS) * std[None, :].clamp_min(EPS))
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    return cov, corr, eigvals[order].clamp_min(0.0), eigvecs[:, order]


def _effective_rank(eigvals: Tensor) -> float:
    vals = eigvals.detach().double().clamp_min(0.0)
    total = vals.sum()
    if total <= 0:
        return 0.0
    p = vals / total
    entropy = -(p[p > 0] * torch.log(p[p > 0])).sum()
    return float(torch.exp(entropy).item())


def _jacobian_analysis(local_jacobian: Tensor, scale: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, float, float, float]:
    n_rays = int(local_jacobian.shape[0])
    j = local_jacobian.detach().double().reshape(n_rays * 3, 9) * scale.detach().double().reshape(1, 9)
    gram = j.T @ j / max(n_rays, 1)
    eigvals, eigvecs = torch.linalg.eigh(gram)
    eigvals = eigvals.clamp_min(0.0)
    singular = torch.sqrt(eigvals)
    sigma_min = float(singular[0].item())
    sigma_max = float(singular[-1].item())
    ratio = sigma_min / sigma_max if sigma_max > 0 else float("nan")
    cond = sigma_max / max(sigma_min, EPS) if sigma_max > 0 else float("inf")
    erank = _effective_rank(eigvals)
    return gram, eigvals, eigvecs, singular, ratio, cond, erank


def _unit_random_directions(seed: int, count: int = RANDOM_DIRECTIONS) -> List[Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    dirs: List[Tensor] = []
    for _ in range(count):
        v = torch.randn(9, generator=generator, dtype=torch.float64)
        dirs.append(v / v.norm().clamp_min(EPS))
    return dirs


def _sample_flat(mask: Tensor, max_pixels: int, seed: int) -> Tensor:
    flat = torch.nonzero(mask.reshape(-1).detach().bool().cpu(), as_tuple=False).reshape(-1)
    if flat.numel() <= max_pixels:
        return flat.long()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(flat.numel(), generator=generator)[:max_pixels]
    return flat[perm].long()


def _build_samples(
    repo: Path,
    output_dir: Path,
    max_pixels: int,
    seed: int,
) -> Tuple[Dict[str, ViewSample], Dict[str, Any], List[Dict[str, Any]]]:
    step3_maps, step3_meta = PW._render_split_maps(repo, "BND", 3000)
    masks, mask_meta = PW._build_masks(step3_maps)
    samples: Dict[str, ViewSample] = {}
    rows: List[Dict[str, Any]] = []
    train_views = tuple(step3_maps["train"].keys())
    for view_index, view_id in enumerate(train_views):
        acc = PW._scalar_map(step3_maps["train"][view_id]["accumulation"])
        height, width = int(acc.shape[0]), int(acc.shape[1])
        all_valid = torch.ones((height, width), dtype=torch.bool)
        general = _sample_flat(all_valid, max_pixels, seed + 17 * view_index)
        safe_mask = masks["train"][view_id]["M_SAFE"].bool()
        safe = _sample_flat(safe_mask, max_pixels, seed + 1000 + 17 * view_index)
        samples[view_id] = ViewSample(
            view_id=view_id,
            height=height,
            width=width,
            general_flat=general,
            safe_flat=safe,
            safe_available_pixels=int(safe_mask.sum().item()),
        )
        rows.append(
            {
                "view_index": view_index,
                "view_id": view_id,
                "height": height,
                "width": width,
                "GENERAL_sampled_rays": int(general.numel()),
                "M_SAFE_available_pixels": int(safe_mask.sum().item()),
                "M_SAFE_sampled_rays": int(safe.numel()),
                "M_SAFE_fraction": float(safe_mask.float().mean().item()),
            }
        )
    meta = {
        "CONFIG_FACT": "Samples are deterministic and reused across M1/BND matched checkpoints.",
        "scene": SCENE,
        "sample_seed": seed,
        "samples_per_view": max_pixels,
        "train_view_count": len(train_views),
        "train_views": list(train_views),
        "populations": {
            "GENERAL": "1024 deterministic valid in-bounds pixels per train view.",
            "M_SAFE": "Existing locked IUI3 M_SAFE mask from BND@3000 accumulation <=0.01 and SeaFree pseudo-depth background candidate, eroded by 5px; capped at 1024 pixels per view.",
        },
        "step3000_render_meta": step3_meta,
        "mask_meta": mask_meta,
        "total_sampled_rays": {
            "GENERAL": int(sum(row["GENERAL_sampled_rays"] for row in rows)),
            "M_SAFE": int(sum(row["M_SAFE_sampled_rays"] for row in rows)),
        },
    }
    _write_json(output_dir / "sampling_source_bnd3000_meta.json", meta)
    return samples, meta, rows


def _source_equivalence_probe(repo: Path) -> Dict[str, Any]:
    loaded = None
    try:
        loaded = PW._load_run(repo, "BND", 5000)
        model = loaded.pipeline.model
        model.eval()
        idx, view_id, camera, _batch = PW._records(loaded.pipeline)["train"][0]
        with torch.no_grad():
            raw, height, width = _medium_raw_for_camera(model, camera)
            med = _activate_medium(model, raw, height, width)
            native = model.get_outputs_for_camera(camera.to(model.device))
            override = _render_with_medium_override(
                model,
                camera,
                med["medium_rgb"],
                med["medium_bs"],
                med["medium_attn"],
                detach_object_state=True,
            )
        keys = ("medium_rgb", "medium_bs", "medium_attn", "b_inf", "pred_image", "depth", "accumulation", "rgb_medium")
        diffs: Dict[str, float] = {}
        for key in keys:
            if key in native and key in override and isinstance(native[key], Tensor) and isinstance(override[key], Tensor):
                diffs[key] = float((native[key].detach().float() - override[key].detach().float()).abs().max().item())
        return {
            "PREACTIVATION_ACCESS": "AVAILABLE_DIAGNOSTIC_MANUAL_MEDIUM_MLP_FORWARD",
            "probe_run": "BND",
            "probe_nominal_step": 5000,
            "probe_loaded_step": int(loaded.loaded_step),
            "probe_view_index": int(idx),
            "probe_view_id": view_id,
            "raw_shape": [int(raw.shape[0]), int(raw.shape[1])],
            "height": int(height),
            "width": int(width),
            "max_abs_diffs": diffs,
            "equivalence_pass": bool(all(value <= 1e-5 for value in diffs.values())),
            "explanation": "medium maps are reconstructed from the same inputs as DirectionConditionedMediumField; rendered override uses detached object state and the same underwater rasterizer path.",
        }
    finally:
        PW._release(loaded)


def _append_population_values(
    store: Dict[str, Dict[str, List[Tensor]]],
    view_slices: Dict[str, Dict[str, Tuple[int, int]]],
    population: str,
    view_id: str,
    flat: Tensor,
    union_flat: Tensor,
    grad_union: Tensor,
    raw: Tensor,
    med: Mapping[str, Tensor],
    outputs: Mapping[str, Tensor],
    gt: Tensor,
) -> None:
    if flat.numel() == 0:
        return
    pos = torch.searchsorted(union_flat.cpu(), flat.cpu())
    local_j = grad_union[pos].float().cpu()
    start = sum(int(chunk.shape[0]) for chunk in store[population]["z"])
    end = start + int(flat.numel())
    view_slices[population][view_id] = (start, end)

    flat_dev = flat.to(raw.device)
    z = raw.detach()[flat_dev].float().cpu()
    medium_activated = torch.cat(
        [
            _flatten_sample(med["medium_rgb"].detach(), flat_dev).reshape(-1, 3),
            _flatten_sample(med["medium_bs"].detach(), flat_dev).reshape(-1, 3),
            _flatten_sample(med["medium_attn"].detach(), flat_dev).reshape(-1, 3),
        ],
        dim=-1,
    ).float().cpu()
    pred = _flatten_sample(outputs["pred_image"].detach(), flat_dev).reshape(-1, 3).float().cpu()
    target = gt.reshape(-1, 3)[flat.cpu()].float().cpu()
    residual = pred - target
    store[population]["z"].append(z)
    store[population]["activated"].append(medium_activated)
    store[population]["depth"].append(_flatten_sample(outputs["depth"].detach(), flat_dev).reshape(-1, 1).float().cpu())
    store[population]["tau"].append(_flatten_sample(outputs["tau_D"].detach(), flat_dev).reshape(-1, 3).float().cpu())
    store[population]["transmission"].append(
        _flatten_sample(outputs["transmission"].detach(), flat_dev).reshape(-1, 3).float().cpu()
    )
    store[population]["rgb"].append(pred)
    store[population]["gt"].append(target)
    store[population]["rgb_residual"].append(residual)
    store[population]["local_jacobian"].append(local_j)


def _init_store() -> Dict[str, Dict[str, List[Tensor]]]:
    return {
        pop: {
            "z": [],
            "activated": [],
            "depth": [],
            "tau": [],
            "transmission": [],
            "rgb": [],
            "gt": [],
            "rgb_residual": [],
            "local_jacobian": [],
        }
        for pop in POPULATIONS
    }


def _cat_or_empty(parts: Sequence[Tensor], shape_tail: Sequence[int]) -> Tensor:
    if parts:
        return torch.cat(list(parts), dim=0)
    return torch.empty((0, *shape_tail), dtype=torch.float32)


def _finalize_population(store: Mapping[str, List[Tensor]], view_slices: Mapping[str, Tuple[int, int]]) -> PopAnalysis:
    z = _cat_or_empty(store["z"], (9,))
    activated = _cat_or_empty(store["activated"], (9,))
    depth = _cat_or_empty(store["depth"], (1,))
    tau = _cat_or_empty(store["tau"], (3,))
    transmission = _cat_or_empty(store["transmission"], (3,))
    rgb = _cat_or_empty(store["rgb"], (3,))
    gt = _cat_or_empty(store["gt"], (3,))
    residual = _cat_or_empty(store["rgb_residual"], (3,))
    local_j = _cat_or_empty(store["local_jacobian"], (3, 9))
    if z.shape[0] == 0:
        raise RuntimeError("No rays collected for population")
    scale = z.std(dim=0, unbiased=False).clamp_min(1e-3).double()
    cov, corr, pca_vals, pca_vecs = _cov_corr_pca(z)
    z_std = (z.double() - z.double().mean(dim=0, keepdim=True)) / scale.reshape(1, 9)
    cov_std, _corr_std, pca_std_vals, pca_std_vecs = _cov_corr_pca(z_std)
    gram, eigvals, eigvecs, singular, ratio, cond, erank = _jacobian_analysis(local_j, scale)
    return PopAnalysis(
        z=z,
        activated=activated,
        depth=depth,
        tau=tau,
        transmission=transmission,
        rgb=rgb,
        gt=gt,
        rgb_residual=residual,
        local_jacobian=local_j,
        view_slices=dict(view_slices),
        scale=scale,
        covariance=cov,
        correlation=corr,
        pca_eigvals=pca_vals,
        pca_eigvecs=pca_vecs,
        covariance_std=cov_std,
        pca_std_eigvals=pca_std_vals,
        pca_std_eigvecs=pca_std_vecs,
        gram=gram,
        eigvals=eigvals,
        eigvecs=eigvecs,
        singular_values_per_sqrt_ray=singular,
        v_min=eigvecs[:, 0],
        v_max=eigvecs[:, -1],
        sigma_ratio=ratio,
        condition_number=cond,
        effective_rank=erank,
    )


def _group_energy(v: Tensor) -> Dict[str, Any]:
    vec = v.detach().double()
    total = float(vec.square().sum().item())
    total = max(total, EPS)
    names = ("r", "g", "b")
    out = {
        "B_inf_energy_fraction": float(vec[0:3].square().sum().item() / total),
        "beta_B_energy_fraction": float(vec[3:6].square().sum().item() / total),
        "beta_D_energy_fraction": float(vec[6:9].square().sum().item() / total),
    }
    for group, offset in (("B_inf", 0), ("beta_B", 3), ("beta_D", 6)):
        for idx, channel in enumerate(names):
            out[f"{group}_{channel}_component"] = float(vec[offset + idx].item())
            out[f"{group}_{channel}_energy_fraction"] = float(vec[offset + idx].square().item() / total)
    return out


def _dominant_family(v: Tensor) -> str:
    energy = _group_energy(v)
    b_inf = float(energy["B_inf_energy_fraction"])
    beta_b = float(energy["beta_B_energy_fraction"])
    beta_d = float(energy["beta_D_energy_fraction"])
    top = max(b_inf, beta_b, beta_d)
    if beta_d >= 0.65:
        return "WEAK_MODE_BETAD"
    if b_inf + beta_b >= 0.75 and beta_d <= 0.25 and min(b_inf, beta_b) >= 0.15:
        return "WEAK_MODE_BINF_BETAB"
    if top < 0.65:
        return "WEAK_MODE_MIXED"
    return "WEAK_MODE_MIXED"


def _natural_stats_rows(run: str, step: int, loaded_step: int, pop: str, analysis: PopAnalysis) -> Dict[str, Any]:
    names = (
        "Binf_r",
        "Binf_g",
        "Binf_b",
        "betaB_r",
        "betaB_g",
        "betaB_b",
        "betaD_r",
        "betaD_g",
        "betaD_b",
    )
    row: Dict[str, Any] = {
        "run": run,
        "nominal_step": step,
        "loaded_step": loaded_step,
        "population": pop,
        "sampled_rays": int(analysis.z.shape[0]),
        "preactivation_access": "AVAILABLE",
        "scale_rule": "S_j=max(std(z_med_j),1e-3)",
    }
    row.update(_per_dim_stats(analysis.z, names, "z_med"))
    row.update(_per_dim_stats(analysis.activated, names, "activated"))
    row.update(_stat_dict(analysis.depth[:, 0], "depth"))
    row.update(_per_dim_stats(analysis.tau, ("r", "g", "b"), "tau_D"))
    row.update(_per_dim_stats(analysis.transmission, ("r", "g", "b"), "transmission"))
    row.update(_per_dim_stats(analysis.rgb, ("r", "g", "b"), "rendered_rgb"))
    row.update(_per_dim_stats(analysis.gt, ("r", "g", "b"), "gt_rgb"))
    row.update(_per_dim_stats(analysis.rgb_residual, ("r", "g", "b"), "rgb_residual"))
    row["rgb_residual_mse"] = float(analysis.rgb_residual.square().mean().item())
    row["rgb_residual_l1"] = float(analysis.rgb_residual.abs().mean().item())
    row["rgb_residual_psnr"] = float(-10.0 * math.log10(max(row["rgb_residual_mse"], EPS)))
    for idx, value in enumerate(analysis.scale):
        row[f"S_{idx}"] = float(value.item())
    return row


def _pca_rows(run: str, step: int, loaded_step: int, pop: str, analysis: PopAnalysis) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for space, eigvals, eigvecs in (
        ("raw_z", analysis.pca_eigvals, analysis.pca_eigvecs),
        ("standardized_z", analysis.pca_std_eigvals, analysis.pca_std_eigvecs),
    ):
        total = float(eigvals.sum().item())
        for mode in range(eigvals.shape[0]):
            vec = eigvecs[:, mode]
            row: Dict[str, Any] = {
                "run": run,
                "nominal_step": step,
                "loaded_step": loaded_step,
                "population": pop,
                "space": space,
                "mode": mode,
                "eigenvalue": float(eigvals[mode].item()),
                "variance_fraction": float(eigvals[mode].item() / max(total, EPS)),
            }
            row.update({f"v_{idx}": float(vec[idx].item()) for idx in range(9)})
            row.update(_group_energy(vec))
            rows.append(row)
    return rows


def _matrix_rows(
    run: str,
    step: int,
    loaded_step: int,
    pop: str,
    name: str,
    matrix: Tensor,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    mat = matrix.detach().double().cpu()
    for i in range(mat.shape[0]):
        row: Dict[str, Any] = {
            "run": run,
            "nominal_step": step,
            "loaded_step": loaded_step,
            "population": pop,
            "matrix": name,
            "row": i,
        }
        for j in range(mat.shape[1]):
            row[f"c{j}"] = float(mat[i, j].item())
        rows.append(row)
    return rows


def _aggregate_rows(run: str, step: int, loaded_step: int, pop: str, analysis: PopAnalysis) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "run": run,
        "nominal_step": step,
        "loaded_step": loaded_step,
        "population": pop,
        "sampled_rays": int(analysis.local_jacobian.shape[0]),
        "J_struct_rows": int(analysis.local_jacobian.shape[0] * 3),
        "J_struct_cols": 9,
        "G_definition": "J_struct^T J_struct / N_rays, where z'=z+S*delta",
        "singular_values_are": "sqrt(eig(G)), i.e. singular values normalized per sqrt sampled ray",
        "sigma_min_over_sigma_max": analysis.sigma_ratio,
        "condition_number": analysis.condition_number,
        "effective_rank": analysis.effective_rank,
    }
    for idx, sigma in enumerate(analysis.singular_values_per_sqrt_ray):
        row[f"sigma_per_sqrt_ray_{idx}"] = float(sigma.item())
    for idx, value in enumerate(analysis.v_min):
        row[f"v_min_{idx}"] = float(value.item())
    for idx, value in enumerate(analysis.v_max):
        row[f"v_max_{idx}"] = float(value.item())
    return row


def _weak_mode_row(run: str, step: int, loaded_step: int, pop: str, analysis: PopAnalysis) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "run": run,
        "nominal_step": step,
        "loaded_step": loaded_step,
        "population": pop,
        "weak_mode_family": _dominant_family(analysis.v_min),
        "sigma_min_over_sigma_max": analysis.sigma_ratio,
    }
    row.update(_group_energy(analysis.v_min))
    return row


def _gram_for_subset(local_j: Tensor, scale: Tensor) -> Tuple[float, float, float, Tensor]:
    if local_j.shape[0] < 9:
        return float("nan"), float("inf"), 0.0, torch.zeros(9, dtype=torch.float64)
    gram, eigvals, eigvecs, singular, ratio, cond, erank = _jacobian_analysis(local_j, scale)
    return ratio, cond, erank, eigvecs[:, 0]


def _camera_rows(run: str, step: int, loaded_step: int, pop: str, analysis: PopAnalysis) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    depth_scalar = analysis.depth[:, 0]
    tau_scalar = analysis.tau.mean(dim=-1)
    for view_id, (start, end) in analysis.view_slices.items():
        local_j = analysis.local_jacobian[start:end]
        ratio, cond, erank, v = _gram_for_subset(local_j, analysis.scale)
        cosine = float(torch.abs(torch.dot(v, analysis.v_min)).item()) if torch.isfinite(v).all() else float("nan")
        row: Dict[str, Any] = {
            "run": run,
            "nominal_step": step,
            "loaded_step": loaded_step,
            "population": pop,
            "view_id": view_id,
            "sampled_rays": int(end - start),
            "sigma_min_over_sigma_max": ratio,
            "condition_number": cond,
            "effective_rank": erank,
            "abs_cosine_to_aggregate_v_min": cosine,
            "depth_mean": float(depth_scalar[start:end].mean().item()) if end > start else float("nan"),
            "tau_mean": float(tau_scalar[start:end].mean().item()) if end > start else float("nan"),
            "weak_mode_family": _dominant_family(v) if torch.isfinite(v).all() else "WEAK_MODE_UNSTABLE",
        }
        row.update(_group_energy(v) if torch.isfinite(v).all() else {})
        rows.append(row)
    return rows


def _strata_rows(run: str, step: int, loaded_step: int, pop: str, analysis: PopAnalysis) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for basis, values in (("depth", analysis.depth[:, 0]), ("tau", analysis.tau.mean(dim=-1))):
        q1 = torch.quantile(values.float(), 1.0 / 3.0)
        q2 = torch.quantile(values.float(), 2.0 / 3.0)
        strata = (
            ("near_or_low", values <= q1),
            ("middle", (values > q1) & (values <= q2)),
            ("far_or_high", values > q2),
        )
        for name, mask in strata:
            local_j = analysis.local_jacobian[mask]
            ratio, cond, erank, v = _gram_for_subset(local_j, analysis.scale)
            row: Dict[str, Any] = {
                "run": run,
                "nominal_step": step,
                "loaded_step": loaded_step,
                "population": pop,
                "stratification_basis": basis,
                "stratum": name,
                "q1": float(q1.item()),
                "q2": float(q2.item()),
                "sampled_rays": int(mask.sum().item()),
                "sigma_min_over_sigma_max": ratio,
                "condition_number": cond,
                "effective_rank": erank,
                "abs_cosine_to_aggregate_v_min": float(torch.abs(torch.dot(v, analysis.v_min)).item())
                if torch.isfinite(v).all()
                else float("nan"),
                "weak_mode_family": _dominant_family(v) if torch.isfinite(v).all() else "WEAK_MODE_UNSTABLE",
            }
            row.update(_group_energy(v) if torch.isfinite(v).all() else {})
            rows.append(row)
    return rows


def _natural_variance_observability_rows(
    run: str,
    step: int,
    loaded_step: int,
    pop: str,
    analysis: PopAnalysis,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mode in range(analysis.pca_std_eigvals.shape[0]):
        eigval = analysis.pca_std_eigvals[mode]
        vec = analysis.pca_std_eigvecs[:, mode].double()
        sensitivity = float(torch.sqrt(torch.clamp(vec @ analysis.gram @ vec, min=0.0)).item())
        natural_std = float(torch.sqrt(eigval.clamp_min(0.0)).item())
        row: Dict[str, Any] = {
            "run": run,
            "nominal_step": step,
            "loaded_step": loaded_step,
            "population": pop,
            "mode": mode,
            "space": "standardized_z",
            "natural_variance": float(eigval.item()),
            "natural_std": natural_std,
            "rgb_sensitivity_per_sqrt_ray": sensitivity,
            "variation_to_sensitivity_ratio": natural_std / max(sensitivity, EPS),
            "abs_cosine_to_v_min": float(torch.abs(torch.dot(vec, analysis.v_min)).item()),
            "abs_cosine_to_v_max": float(torch.abs(torch.dot(vec, analysis.v_max)).item()),
        }
        row.update(_group_energy(vec))
        rows.append(row)
    weak_variance = float((analysis.v_min @ analysis.covariance_std @ analysis.v_min).clamp_min(0.0).item())
    weak_sensitivity = float(torch.sqrt(torch.clamp(analysis.v_min @ analysis.gram @ analysis.v_min, min=0.0)).item())
    rows.append(
        {
            "run": run,
            "nominal_step": step,
            "loaded_step": loaded_step,
            "population": pop,
            "mode": "v_min",
            "space": "standardized_z",
            "natural_variance": weak_variance,
            "natural_std": math.sqrt(max(weak_variance, 0.0)),
            "rgb_sensitivity_per_sqrt_ray": weak_sensitivity,
            "variation_to_sensitivity_ratio": math.sqrt(max(weak_variance, 0.0)) / max(weak_sensitivity, EPS),
            "abs_cosine_to_v_min": 1.0,
            "abs_cosine_to_v_max": float(torch.abs(torch.dot(analysis.v_min, analysis.v_max)).item()),
            **_group_energy(analysis.v_min),
        }
    )
    return rows


def _analyse_checkpoint(
    repo: Path,
    run: str,
    step: int,
    samples: Mapping[str, ViewSample],
) -> Tuple[Any, Dict[str, PopAnalysis], Dict[str, Any]]:
    loaded = PW._load_run(repo, run, step)
    model = loaded.pipeline.model
    model.eval()
    records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in PW._records(loaded.pipeline)["train"]}
    store = _init_store()
    view_slices: Dict[str, Dict[str, Tuple[int, int]]] = {pop: {} for pop in POPULATIONS}
    per_view_elapsed: List[Dict[str, Any]] = []

    try:
        for view_ord, (view_id, sample) in enumerate(samples.items()):
            if view_id not in records:
                raise RuntimeError(f"Missing train view {view_id} in loaded pipeline records")
            _idx, camera, batch = records[view_id]
            t0 = time.time()
            raw_base, height, width = _medium_raw_for_camera(model, camera)
            if (height, width) != (sample.height, sample.width):
                raise RuntimeError(f"View size changed for {view_id}: {(height, width)} vs {(sample.height, sample.width)}")
            raw = raw_base.detach().clone().requires_grad_(True)
            med = _activate_medium(model, raw, height, width)
            outputs = _render_with_medium_override(
                model,
                camera,
                med["medium_rgb"],
                med["medium_bs"],
                med["medium_attn"],
                detach_object_state=True,
            )
            gt = PW._get_gt(model, batch, outputs["background"]).reshape(-1, 3).detach().float().cpu()
            union_flat = torch.unique(torch.cat([sample.general_flat, sample.safe_flat], dim=0)).sort().values.long()
            if union_flat.numel() == 0:
                continue
            union_dev = union_flat.to(model.device)
            pred_flat = outputs["pred_image"].reshape(-1, 3)
            grad_parts: List[Tensor] = []
            for channel in range(3):
                scalar = pred_flat[union_dev, channel].sum()
                grad_raw = torch.autograd.grad(
                    scalar,
                    raw,
                    retain_graph=channel < 2,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                grad_parts.append(grad_raw[union_dev].detach().float().cpu())
            grad_union = torch.stack(grad_parts, dim=1)
            _append_population_values(
                store,
                view_slices,
                "GENERAL",
                view_id,
                sample.general_flat,
                union_flat,
                grad_union,
                raw,
                med,
                outputs,
                gt,
            )
            _append_population_values(
                store,
                view_slices,
                "M_SAFE",
                view_id,
                sample.safe_flat,
                union_flat,
                grad_union,
                raw,
                med,
                outputs,
                gt,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            per_view_elapsed.append(
                {
                    "view_id": view_id,
                    "ordinal": view_ord,
                    "seconds": time.time() - t0,
                    "union_sampled_rays": int(union_flat.numel()),
                }
            )
            del raw_base, raw, med, outputs, grad_union, grad_parts, gt
            gc.collect()
            torch.cuda.empty_cache()

        analyses = {pop: _finalize_population(store[pop], view_slices[pop]) for pop in POPULATIONS}
        meta = {
            "run": run,
            "nominal_step": step,
            "loaded_step": int(loaded.loaded_step),
            "config_path": str(loaded.config_path),
            "checkpoint_path": str(loaded.checkpoint_path),
            "gaussian_count": int(model.num_points),
            "intrinsic_color_parameterization": getattr(model.config, "intrinsic_color_parameterization", ""),
            "medium_context_mode": getattr(model.config, "medium_context_mode", ""),
            "b_inf_mode": getattr(model.config, "b_inf_mode", ""),
            "infinite_water_enabled": getattr(model.config, "infinite_water_enabled", ""),
            "per_view_elapsed": per_view_elapsed,
        }
        return loaded, analyses, meta
    except Exception:
        PW._release(loaded)
        raise


def _counterfactual_for_checkpoint(
    loaded: Any,
    analyses: Mapping[str, PopAnalysis],
    samples: Mapping[str, ViewSample],
    random_dirs: Sequence[Tensor],
) -> List[Dict[str, Any]]:
    model = loaded.pipeline.model
    records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in PW._records(loaded.pipeline)["train"]}
    direction_bank: Dict[str, List[Tuple[str, Tensor]]] = {}
    for pop, analysis in analyses.items():
        direction_bank[pop] = [
            ("v_min", analysis.v_min.detach().double().cpu()),
            ("v_max", analysis.v_max.detach().double().cpu()),
        ]
        for idx, direction in enumerate(random_dirs):
            direction_bank[pop].append((f"random_{idx:02d}", direction.detach().double().cpu()))

    accum: Dict[Tuple[str, str], Dict[str, float]] = {}
    for pop in POPULATIONS:
        for label, _direction in direction_bank[pop]:
            accum[(pop, label)] = {
                "ray_count": 0.0,
                "rgb_delta_abs_sum": 0.0,
                "rgb_delta_sq_sum": 0.0,
                "activated_delta_sq_sum": 0.0,
                "B_inf_delta_sq_sum": 0.0,
                "beta_B_delta_sq_sum": 0.0,
                "beta_D_delta_sq_sum": 0.0,
                "baseline_gt_sq_sum": 0.0,
                "perturbed_gt_sq_sum": 0.0,
            }

    for view_id, sample in samples.items():
        if view_id not in records:
            raise RuntimeError(f"Missing train view {view_id} for counterfactual")
        _idx, camera, batch = records[view_id]
        with torch.no_grad():
            raw_base, height, width = _medium_raw_for_camera(model, camera)
            med_base = _activate_medium(model, raw_base, height, width)
            base_out = _render_with_medium_override(
                model,
                camera,
                med_base["medium_rgb"],
                med_base["medium_bs"],
                med_base["medium_attn"],
                detach_object_state=True,
            )
            gt = PW._get_gt(model, batch, base_out["background"]).reshape(-1, 3).detach().float().cpu()
            base_pred_flat = base_out["pred_image"].reshape(-1, 3).detach().float().cpu()
            base_activated_flat = torch.cat(
                [
                    med_base["medium_rgb"].reshape(-1, 3),
                    med_base["medium_bs"].reshape(-1, 3),
                    med_base["medium_attn"].reshape(-1, 3),
                ],
                dim=-1,
            ).detach().float().cpu()

            for pop in POPULATIONS:
                flat = sample.flat_for(pop)
                if flat.numel() == 0:
                    continue
                flat_dev = flat.to(model.device)
                for label, direction in direction_bank[pop]:
                    raw_pert = raw_base.detach().clone()
                    delta = (
                        analyses[pop].scale.detach().to(device=model.device, dtype=raw_pert.dtype)
                        * float(COUNTERFACTUAL_EPSILON)
                        * direction.to(device=model.device, dtype=raw_pert.dtype)
                    )
                    raw_pert[flat_dev] = raw_pert[flat_dev] + delta.reshape(1, 9)
                    med_pert = _activate_medium(model, raw_pert, height, width)
                    pert_out = _render_with_medium_override(
                        model,
                        camera,
                        med_pert["medium_rgb"],
                        med_pert["medium_bs"],
                        med_pert["medium_attn"],
                        detach_object_state=True,
                    )
                    pert_pred = pert_out["pred_image"].reshape(-1, 3).detach().float().cpu()[flat]
                    base_pred = base_pred_flat[flat]
                    target = gt[flat]
                    pert_activated = torch.cat(
                        [
                            med_pert["medium_rgb"].reshape(-1, 3),
                            med_pert["medium_bs"].reshape(-1, 3),
                            med_pert["medium_attn"].reshape(-1, 3),
                        ],
                        dim=-1,
                    ).detach().float().cpu()[flat]
                    base_activated = base_activated_flat[flat]
                    rgb_delta = pert_pred - base_pred
                    act_delta = pert_activated - base_activated
                    key = (pop, label)
                    n = float(flat.numel())
                    accum[key]["ray_count"] += n
                    accum[key]["rgb_delta_abs_sum"] += float(rgb_delta.abs().sum().item())
                    accum[key]["rgb_delta_sq_sum"] += float(rgb_delta.square().sum().item())
                    accum[key]["activated_delta_sq_sum"] += float(act_delta.square().sum().item())
                    accum[key]["B_inf_delta_sq_sum"] += float(act_delta[:, 0:3].square().sum().item())
                    accum[key]["beta_B_delta_sq_sum"] += float(act_delta[:, 3:6].square().sum().item())
                    accum[key]["beta_D_delta_sq_sum"] += float(act_delta[:, 6:9].square().sum().item())
                    accum[key]["baseline_gt_sq_sum"] += float((base_pred - target).square().sum().item())
                    accum[key]["perturbed_gt_sq_sum"] += float((pert_pred - target).square().sum().item())
                    del raw_pert, med_pert, pert_out
        del raw_base, med_base, base_out, gt, base_pred_flat, base_activated_flat
        gc.collect()
        torch.cuda.empty_cache()

    rows: List[Dict[str, Any]] = []
    for pop in POPULATIONS:
        for label, direction in direction_bank[pop]:
            item = accum[(pop, label)]
            n = max(item["ray_count"], 1.0)
            denom_rgb = n * 3.0
            denom_med = n * 9.0
            rgb_mse = item["rgb_delta_sq_sum"] / denom_rgb
            baseline_gt_mse = item["baseline_gt_sq_sum"] / denom_rgb
            perturbed_gt_mse = item["perturbed_gt_sq_sum"] / denom_rgb
            row: Dict[str, Any] = {
                "run": loaded.run,
                "nominal_step": int(getattr(loaded, "nominal_step", loaded.loaded_step)),
                "loaded_step": int(loaded.loaded_step),
                "population": pop,
                "direction_label": label,
                "epsilon_standardized": COUNTERFACTUAL_EPSILON,
                "sampled_rays": int(item["ray_count"]),
                "mean_abs_rendered_rgb_change": item["rgb_delta_abs_sum"] / denom_rgb,
                "rendered_rgb_delta_mse_vs_baseline": rgb_mse,
                "rendered_rgb_delta_psnr_vs_baseline": float(-10.0 * math.log10(max(rgb_mse, EPS))),
                "baseline_gt_mse": baseline_gt_mse,
                "perturbed_gt_mse": perturbed_gt_mse,
                "gt_mse_delta": perturbed_gt_mse - baseline_gt_mse,
                "baseline_gt_psnr": float(-10.0 * math.log10(max(baseline_gt_mse, EPS))),
                "perturbed_gt_psnr": float(-10.0 * math.log10(max(perturbed_gt_mse, EPS))),
                "gt_psnr_delta": float(-10.0 * math.log10(max(perturbed_gt_mse, EPS)))
                - float(-10.0 * math.log10(max(baseline_gt_mse, EPS))),
                "rms_medium_output_change_9d": math.sqrt(item["activated_delta_sq_sum"] / denom_med),
                "rms_B_inf_change": math.sqrt(item["B_inf_delta_sq_sum"] / max(n * 3.0, 1.0)),
                "rms_beta_B_change": math.sqrt(item["beta_B_delta_sq_sum"] / max(n * 3.0, 1.0)),
                "rms_beta_D_change": math.sqrt(item["beta_D_delta_sq_sum"] / max(n * 3.0, 1.0)),
            }
            for idx in range(9):
                row[f"direction_{idx}"] = float(direction[idx].item())
            rows.append(row)
    return rows


def _safe_float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def _stability_summary(weak_rows: Sequence[Mapping[str, Any]], camera_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for run in RUNS:
        for pop in POPULATIONS:
            rows = [row for row in weak_rows if row["run"] == run and row["population"] == pop]
            rows = sorted(rows, key=lambda row: int(row["nominal_step"]))
            cosines: List[float] = []
            for a, b in zip(rows, rows[1:]):
                va = torch.tensor([_safe_float(a, f"B_inf_{c}_component") for c in ("r", "g", "b")])
                va = torch.cat(
                    [
                        va,
                        torch.tensor([_safe_float(a, f"beta_B_{c}_component") for c in ("r", "g", "b")]),
                        torch.tensor([_safe_float(a, f"beta_D_{c}_component") for c in ("r", "g", "b")]),
                    ]
                ).double()
                vb = torch.tensor(
                    [_safe_float(b, f"B_inf_{c}_component") for c in ("r", "g", "b")]
                    + [_safe_float(b, f"beta_B_{c}_component") for c in ("r", "g", "b")]
                    + [_safe_float(b, f"beta_D_{c}_component") for c in ("r", "g", "b")]
                ).double()
                if va.norm() > 0 and vb.norm() > 0:
                    cosines.append(float(torch.abs(torch.dot(va / va.norm(), vb / vb.norm())).item()))
            fams = [str(row.get("weak_mode_family", "")) for row in rows]
            key = f"{run}_{pop}"
            cam = [
                _safe_float(row, "abs_cosine_to_aggregate_v_min")
                for row in camera_rows
                if row["run"] == run and row["population"] == pop and math.isfinite(_safe_float(row, "abs_cosine_to_aggregate_v_min"))
            ]
            out[key] = {
                "checkpoint_adjacent_abs_cosines": cosines,
                "checkpoint_adjacent_abs_cosine_min": min(cosines) if cosines else float("nan"),
                "checkpoint_adjacent_abs_cosine_mean": float(np.mean(cosines)) if cosines else float("nan"),
                "weak_mode_families": fams,
                "same_family_across_matched_steps": bool(len(set(fams)) == 1) if fams else False,
                "camera_abs_cosine_median": float(np.median(cam)) if cam else float("nan"),
                "camera_abs_cosine_p25": float(np.quantile(cam, 0.25)) if cam else float("nan"),
                "camera_abs_cosine_min": min(cam) if cam else float("nan"),
            }
    return out


def _counterfactual_summary(counter_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    by_key: Dict[Tuple[str, int, str], Dict[str, Mapping[str, Any]]] = {}
    for row in counter_rows:
        key = (str(row["run"]), int(row["nominal_step"]), str(row["population"]))
        by_key.setdefault(key, {})[str(row["direction_label"])] = row
    for key, rows in by_key.items():
        run, step, pop = key
        vmin = rows.get("v_min")
        vmax = rows.get("v_max")
        random_changes = [
            _safe_float(row, "mean_abs_rendered_rgb_change")
            for label, row in rows.items()
            if label.startswith("random_")
        ]
        random_medium = [
            _safe_float(row, "rms_medium_output_change_9d")
            for label, row in rows.items()
            if label.startswith("random_")
        ]
        name = f"{run}_{step}_{pop}"
        out[name] = {
            "v_min_mean_abs_rgb_change": _safe_float(vmin or {}, "mean_abs_rendered_rgb_change"),
            "v_max_mean_abs_rgb_change": _safe_float(vmax or {}, "mean_abs_rendered_rgb_change"),
            "v_min_over_v_max_rgb_change": _safe_float(vmin or {}, "mean_abs_rendered_rgb_change")
            / max(_safe_float(vmax or {}, "mean_abs_rendered_rgb_change"), EPS),
            "random_median_mean_abs_rgb_change": float(np.median(random_changes)) if random_changes else float("nan"),
            "v_min_over_random_median_rgb_change": _safe_float(vmin or {}, "mean_abs_rendered_rgb_change")
            / max(float(np.median(random_changes)) if random_changes else float("nan"), EPS),
            "v_min_rms_medium_output_change_9d": _safe_float(vmin or {}, "rms_medium_output_change_9d"),
            "v_max_rms_medium_output_change_9d": _safe_float(vmax or {}, "rms_medium_output_change_9d"),
            "random_median_rms_medium_output_change_9d": float(np.median(random_medium)) if random_medium else float("nan"),
        }
    return out


def _m1_vs_bnd_summary(aggregate_rows: Sequence[Mapping[str, Any]], counter_summary: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pop in POPULATIONS:
        rows = {
            (row["run"], int(row["nominal_step"])): row
            for row in aggregate_rows
            if row["population"] == pop
        }
        pop_rows: List[Dict[str, Any]] = []
        for step in MATCHED_STEPS:
            m1 = rows.get(("M1", step))
            bnd = rows.get(("BND", step))
            if not m1 or not bnd:
                continue
            m1_ratio = _safe_float(m1, "sigma_min_over_sigma_max")
            bnd_ratio = _safe_float(bnd, "sigma_min_over_sigma_max")
            pop_rows.append(
                {
                    "step": step,
                    "M1_sigma_min_over_sigma_max": m1_ratio,
                    "BND_sigma_min_over_sigma_max": bnd_ratio,
                    "BND_over_M1_sigma_ratio": bnd_ratio / max(m1_ratio, EPS),
                    "M1_condition_number": _safe_float(m1, "condition_number"),
                    "BND_condition_number": _safe_float(bnd, "condition_number"),
                    "M1_vmin_over_vmax_cf_rgb": counter_summary.get(f"M1_{step}_{pop}", {}).get(
                        "v_min_over_v_max_rgb_change", float("nan")
                    ),
                    "BND_vmin_over_vmax_cf_rgb": counter_summary.get(f"BND_{step}_{pop}", {}).get(
                        "v_min_over_v_max_rgb_change", float("nan")
                    ),
                }
            )
        out[pop] = pop_rows
    return out


def _classification(
    aggregate_rows: Sequence[Mapping[str, Any]],
    weak_rows: Sequence[Mapping[str, Any]],
    camera_rows: Sequence[Mapping[str, Any]],
    nvo_rows: Sequence[Mapping[str, Any]],
    counter_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    relevant_pop = "M_SAFE"
    bnd_rows = [
        row
        for row in aggregate_rows
        if row["run"] == "BND" and row["population"] == relevant_pop and int(row["nominal_step"]) in MATCHED_STEPS
    ]
    ill_steps = [
        int(row["nominal_step"])
        for row in bnd_rows
        if _safe_float(row, "sigma_min_over_sigma_max") <= 0.10
    ]
    cf_steps = []
    for step in MATCHED_STEPS:
        item = counter_summary.get(f"BND_{step}_{relevant_pop}", {})
        if _safe_float(item, "v_min_over_v_max_rgb_change") <= 0.25:
            cf_steps.append(step)
    stability = _stability_summary(weak_rows, camera_rows).get(f"BND_{relevant_pop}", {})
    checkpoint_stability = bool(
        stability.get("same_family_across_matched_steps", False)
        and float(stability.get("checkpoint_adjacent_abs_cosine_min", 0.0)) >= 0.70
    )
    camera_stability = bool(float(stability.get("camera_abs_cosine_median", 0.0)) >= 0.50)
    weak_nvo = [
        row
        for row in nvo_rows
        if row["run"] == "BND"
        and row["population"] == relevant_pop
        and row["mode"] == "v_min"
        and int(row["nominal_step"]) in MATCHED_STEPS
    ]
    natural_nontrivial_steps = [
        int(row["nominal_step"])
        for row in weak_nvo
        if _safe_float(row, "natural_std") >= 0.05
    ]
    families = [
        row.get("weak_mode_family", "")
        for row in weak_rows
        if row["run"] == "BND" and row["population"] == relevant_pop and int(row["nominal_step"]) in MATCHED_STEPS
    ]
    stable_family = str(families[0]) if families and len(set(families)) == 1 else "WEAK_MODE_UNSTABLE"
    supported = (
        len(ill_steps) >= 2
        and len(cf_steps) >= 2
        and checkpoint_stability
        and camera_stability
        and len(natural_nontrivial_steps) >= 2
        and stable_family != "WEAK_MODE_UNSTABLE"
    )
    tentative = (
        len(ill_steps) >= 2
        and (len(cf_steps) >= 1 or len(natural_nontrivial_steps) >= 2)
    )
    if supported:
        phase_a = "MEDIUM_IDENTIFIABILITY_SUPPORTED"
        phase_b_gate = "ENTER_PHASE_B"
    elif tentative:
        phase_a = "MEDIUM_IDENTIFIABILITY_TENTATIVE"
        phase_b_gate = "DESIGN_NOTE_ONLY_NO_TRAINING_CODE"
    else:
        phase_a = "MEDIUM_IDENTIFIABILITY_NOT_SUPPORTED"
        phase_b_gate = "DO_NOT_ENTER_PHASE_B"
    return {
        "Phase_A_classification": phase_a,
        "Phase_B_gate": phase_b_gate,
        "relevant_population": relevant_pop,
        "registered_criteria": {
            "ill_conditioned_steps_sigma_ratio_le_0p10": ill_steps,
            "counterfactual_vmin_rgb_le_0p25_vmax_steps": cf_steps,
            "checkpoint_stability_rule": "same weak-mode family and adjacent abs cosine >= 0.70 across BND matched steps",
            "checkpoint_stability_pass": checkpoint_stability,
            "camera_stability_rule": "median per-camera abs cosine to aggregate v_min >= 0.50",
            "camera_stability_pass": camera_stability,
            "natural_variation_nontrivial_rule": "standardized weak-mode natural std >= 0.05; used only to reject numerically dead directions",
            "natural_variation_nontrivial_steps": natural_nontrivial_steps,
            "not_single_ray_rank_deficiency": True,
            "aggregate_J_shape": "approximately 3N x 9 with N=sampled rays across 25 train views",
        },
        "weak_mode_family": stable_family,
        "stability_summary": stability,
        "scientific_gate_note": "Correlation/PCA alone is not used as identifiability evidence; the gate uses aggregate structured sensitivity plus counterfactual rendered RGB change.",
    }


def _write_research_note(path: Path, summary: Mapping[str, Any]) -> None:
    cls = summary["classification"]
    m1_bnd = summary["m1_vs_bnd"]
    lines = [
        "# BND-MEDIUM-IDENTIFIABILITY-PREFLIGHT-IUI3",
        "",
        "## Scope",
        "CONFIG FACT: This is a read-only Phase-A preflight. No optimizer, parameter update, full training, new loss, new module, checkpoint write, CDEPTH, MEDCTX removal, CB loss, OMVC, or opacity/densification intervention is used.",
        "CONFIG FACT: Primary scene is IUI3-RedSea with the established train/eval split; matched checkpoints are M1/BND nominal 5000, 10000, and 15000 where 15000 maps to actual 14999.",
        "",
        "## Environment",
        f"EXPERIMENTAL FACT: CONDA_ENV `{summary['environment'].get('CONDA_ENV')}`; Python `{summary['environment'].get('PYTHON_PATH')}`; Torch `{summary['environment'].get('TORCH_VERSION')}`.",
        f"EXPERIMENTAL FACT: CUDA_VISIBLE_DEVICES `{summary['gpu'].get('CUDA_VISIBLE_DEVICES')}`; torch logical cuda:0 is physical GPU `{summary['gpu'].get('physical_gpu_id')}`; GPU `{summary['gpu'].get('gpu_name')}`.",
        "",
        "## Exact Medium Semantics",
        "CODE FACT: The medium MLP takes 22-D `dir_xy_camera` features: 16-D direction encoding, 3-D XY/r context, and 3-D camera context.",
        "CODE FACT: The 9-D raw output is activated as `B_inf=medium_rgb=sigmoid(z[0:3])`, `beta_B=softplus(z[3:6]+medium_density_bias)`, and `beta_D=softplus(z[6:9]+medium_density_bias)`.",
        "CODE FACT: Direct object RGB is attenuated by `exp(-beta_D*depth)`. Medium backscatter uses `medium_rgb` and `beta_B` across Gaussian depth intervals plus the final tail.",
        "CODE FACT: With `b_inf_mode=tied`, Python recomposes the tail with `b_inf=medium_rgb`, preserving rendered RGB while exposing `b_inf` semantics.",
        "",
        "## Pre-Activation Access",
        f"EXPERIMENTAL FACT: `{summary['preactivation_probe'].get('PREACTIVATION_ACCESS')}`.",
        f"QUANTITATIVE RESULT: source-equivalence max abs diffs `{summary['preactivation_probe'].get('max_abs_diffs')}`.",
        "",
        "## Deterministic Sampling",
        f"CONFIG FACT: GENERAL sampled rays `{summary['sampling']['total_sampled_rays'].get('GENERAL')}`; M_SAFE sampled rays `{summary['sampling']['total_sampled_rays'].get('M_SAFE')}`.",
        "CONFIG FACT: `M_SAFE` reuses the locked IUI3 pseudo-depth/background and BND@3000 low-accumulation candidate semantics; it is a diagnostic population, not training supervision.",
        "",
        "## Aggregate Structured Jacobian",
        "CONFIG FACT: The perturbation is shared across sampled rays as `z_med' = z_med + S*delta`, with `S_j=max(std(z_j),1e-3)` per checkpoint and population.",
        f"QUANTITATIVE RESULT: Phase-A classification `{cls['Phase_A_classification']}` using relevant population `{cls['relevant_population']}`.",
        f"QUANTITATIVE RESULT: ill-conditioned BND steps `{cls['registered_criteria']['ill_conditioned_steps_sigma_ratio_le_0p10']}`.",
        "",
        "## Weak Mode",
        f"QUANTITATIVE RESULT: weak-mode family `{cls.get('weak_mode_family')}`.",
        f"QUANTITATIVE RESULT: stability summary `{cls.get('stability_summary')}`.",
        "",
        "## Counterfactual Perturbation",
        f"QUANTITATIVE RESULT: BND counterfactual steps satisfying `RGB_change(v_min) <= 0.25*RGB_change(v_max)`: `{cls['registered_criteria']['counterfactual_vmin_rgb_le_0p25_vmax_steps']}`.",
        "EXPERIMENTAL FACT: Counterfactuals use one fixed standardized step epsilon=0.25 and no parameter updates.",
        "",
        "## M1 vs BND",
        f"QUANTITATIVE RESULT: M1-vs-BND summary `{m1_bnd}`.",
        "",
        "## Interpretation",
    ]
    phase = cls["Phase_A_classification"]
    if phase == "MEDIUM_IDENTIFIABILITY_SUPPORTED":
        lines.extend(
            [
                "INFERENCE: BND does not remove a stable, actionable low-observability medium direction under this preflight gate.",
                "HYPOTHESIS: A later single-factor BND-MIC experiment can target medium-local weak-mode variation without changing the bounded object representation.",
            ]
        )
    elif phase == "MEDIUM_IDENTIFIABILITY_TENTATIVE":
        lines.extend(
            [
                "INFERENCE: The audit detects aggregate low-observability structure, but the evidence is not strong enough to implement training code in this task.",
                "HYPOTHESIS: The next step should resolve the missing weak-mode stability/actionability diagnostic, not add a new module.",
            ]
        )
    else:
        lines.extend(
            [
                "INFERENCE: Under the registered preflight gate, residual medium identifiability failure is not supported strongly enough to justify a new medium module.",
                "HYPOTHESIS: Remaining BND RGB trade-off may reflect ordinary bounded-capacity optimization rather than an actionable medium-identifiability failure.",
            ]
        )
    lines.extend(
        [
            "",
            "## Phase-B Gate",
            f"INFERENCE: `{cls['Phase_B_gate']}`.",
            "CONFIG FACT: Phase B must not be entered for `MEDIUM_IDENTIFIABILITY_NOT_SUPPORTED`; for `MEDIUM_IDENTIFIABILITY_TENTATIVE`, only a design note is allowed.",
            "",
            "## Output Files",
            "EXPERIMENTAL FACT: Full quantitative tables are written under `outputs/bnd_medium_identifiability_preflight_iui3_20260825/` and are intentionally not committed.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--samples-per-view", type=int, default=SAMPLES_PER_VIEW)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument(
        "--skip-counterfactual",
        action="store_true",
        help="Development-only speed option. Formal run must not use this.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_manifest = _assert_runtime_policy()
    environment = _environment_manifest(gpu_manifest)
    repo_manifest = _repo_manifest(repo)
    source_semantics = _medium_source_semantics(repo)
    preactivation_probe = _source_equivalence_probe(repo)
    checkpoint_rows, actual_by_run = _checkpoint_mapping(repo)
    samples, sampling_meta, sampling_rows = _build_samples(repo, output_dir, args.samples_per_view, args.seed)

    _write_json(output_dir / "gpu_manifest.json", gpu_manifest)
    _write_json(output_dir / "environment_manifest.json", environment)
    _write_json(output_dir / "repo_manifest.json", repo_manifest)
    _write_json(output_dir / "source_semantics.json", source_semantics)
    _write_json(output_dir / "preactivation_access.json", preactivation_probe)
    _write_csv(output_dir / "checkpoint_mapping.csv", checkpoint_rows)
    _write_json(output_dir / "checkpoint_mapping.json", checkpoint_rows)
    _write_csv(output_dir / "deterministic_ray_sampling.csv", sampling_rows)
    _write_json(output_dir / "deterministic_ray_sampling.json", sampling_meta)

    natural_rows: List[Dict[str, Any]] = []
    pca_rows: List[Dict[str, Any]] = []
    matrix_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    weak_rows: List[Dict[str, Any]] = []
    camera_rows: List[Dict[str, Any]] = []
    strata_rows: List[Dict[str, Any]] = []
    nvo_rows: List[Dict[str, Any]] = []
    counter_rows: List[Dict[str, Any]] = []
    checkpoint_meta: List[Dict[str, Any]] = []
    random_dirs = _unit_random_directions(args.seed + 404)

    for run in RUNS:
        for step in MATCHED_STEPS:
            print(f"[medium-identifiability] analyzing {run} nominal {step}", flush=True)
            loaded = None
            try:
                loaded, analyses, meta = _analyse_checkpoint(repo, run, step, samples)
                checkpoint_meta.append(meta)
                loaded.nominal_step = step
                for pop, analysis in analyses.items():
                    natural_rows.append(_natural_stats_rows(run, step, int(loaded.loaded_step), pop, analysis))
                    pca_rows.extend(_pca_rows(run, step, int(loaded.loaded_step), pop, analysis))
                    matrix_rows.extend(_matrix_rows(run, step, int(loaded.loaded_step), pop, "z_covariance", analysis.covariance))
                    matrix_rows.extend(_matrix_rows(run, step, int(loaded.loaded_step), pop, "z_correlation", analysis.correlation))
                    matrix_rows.extend(_matrix_rows(run, step, int(loaded.loaded_step), pop, "G_structured", analysis.gram))
                    aggregate_rows.append(_aggregate_rows(run, step, int(loaded.loaded_step), pop, analysis))
                    weak_rows.append(_weak_mode_row(run, step, int(loaded.loaded_step), pop, analysis))
                    camera_rows.extend(_camera_rows(run, step, int(loaded.loaded_step), pop, analysis))
                    strata_rows.extend(_strata_rows(run, step, int(loaded.loaded_step), pop, analysis))
                    nvo_rows.extend(_natural_variance_observability_rows(run, step, int(loaded.loaded_step), pop, analysis))
                if args.skip_counterfactual:
                    print("[medium-identifiability] WARNING: counterfactual skipped by development flag", flush=True)
                else:
                    print(f"[medium-identifiability] counterfactual {run} nominal {step}", flush=True)
                    counter_rows.extend(_counterfactual_for_checkpoint(loaded, analyses, samples, random_dirs))
            finally:
                PW._release(loaded)

    counter_summary = _counterfactual_summary(counter_rows)
    stability = _stability_summary(weak_rows, camera_rows)
    m1_bnd = _m1_vs_bnd_summary(aggregate_rows, counter_summary)
    classification = _classification(aggregate_rows, weak_rows, camera_rows, nvo_rows, counter_summary)
    final_summary = {
        "repo": repo_manifest,
        "environment": environment,
        "gpu": gpu_manifest,
        "source_semantics": source_semantics,
        "preactivation_probe": preactivation_probe,
        "checkpoint_mapping": checkpoint_rows,
        "checkpoint_meta": checkpoint_meta,
        "sampling": sampling_meta,
        "stability": stability,
        "counterfactual_summary": counter_summary,
        "m1_vs_bnd": m1_bnd,
        "classification": classification,
        "Phase_B_entered": bool(classification["Phase_B_gate"] == "ENTER_PHASE_B"),
        "Phase_B_status": "NOT_ENTERED_BY_SCRIPT" if classification["Phase_B_gate"] != "ENTER_PHASE_B" else "REQUIRES_AGENT_IMPLEMENTATION_AFTER_READING_RESULTS",
    }

    _write_csv(output_dir / "natural_medium_output_statistics.csv", natural_rows)
    _write_json(output_dir / "natural_medium_output_statistics.json", natural_rows)
    _write_csv(output_dir / "medium_pca.csv", pca_rows)
    _write_json(output_dir / "medium_pca.json", pca_rows)
    _write_csv(output_dir / "medium_covariance_correlation_gram.csv", matrix_rows)
    _write_json(output_dir / "medium_covariance_correlation_gram.json", matrix_rows)
    _write_csv(output_dir / "aggregate_structured_jacobian.csv", aggregate_rows)
    _write_json(output_dir / "aggregate_structured_jacobian.json", aggregate_rows)
    _write_csv(output_dir / "weak_mode_composition.csv", weak_rows)
    _write_json(output_dir / "weak_mode_composition.json", weak_rows)
    _write_csv(output_dir / "camera_context_stability.csv", camera_rows)
    _write_json(output_dir / "camera_context_stability.json", camera_rows)
    _write_csv(output_dir / "depth_tau_stratification.csv", strata_rows)
    _write_json(output_dir / "depth_tau_stratification.json", strata_rows)
    _write_csv(output_dir / "natural_variance_vs_observability.csv", nvo_rows)
    _write_json(output_dir / "natural_variance_vs_observability.json", nvo_rows)
    _write_csv(output_dir / "counterfactual_perturbation.csv", counter_rows)
    _write_json(output_dir / "counterfactual_perturbation.json", counter_rows)
    _write_json(output_dir / "counterfactual_summary.json", counter_summary)
    _write_json(output_dir / "stability_summary.json", stability)
    _write_json(output_dir / "m1_vs_bnd_identifiability.json", m1_bnd)
    _write_json(output_dir / "phase_a_classification.json", classification)
    _write_json(output_dir / "final_summary.json", final_summary)
    _write_research_note(RESEARCH_NOTE, final_summary)
    print(json.dumps(classification, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
