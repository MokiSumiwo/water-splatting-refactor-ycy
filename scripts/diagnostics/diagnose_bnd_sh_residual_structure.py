#!/usr/bin/env python
"""Legacy-vs-BND SH residual structure audit.

This is a read-only diagnostic. It loads existing checkpoints, renders eval
views, and computes statistics/counterfactual renders without training or
mutating checkpoint state.
"""

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
from PIL import Image, ImageDraw
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup
from water_splatting.fields import compute_bounded_gaussian_colors
from water_splatting.sh import spherical_harmonics
from water_splatting.water_splatting import SH2RGB, SHLogits2RGB


CHANNELS = ("r", "g", "b")
SCENES = ("Curasao", "JapaneseGradens", "IUI3", "Panama")
FINAL_NOMINAL_STEP = 15000
LUMA_WEIGHTS = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)
EPS = 1e-8


@dataclass(frozen=True)
class RunSpec:
    scene: str
    run: str
    config_relpath: str
    parameterization: str
    nominal_step: int = FINAL_NOMINAL_STEP
    role: str = "primary"
    appearance_lr_scale: float = 1.0


RUN_SPECS: Dict[Tuple[str, str], RunSpec] = {
    ("Curasao", "M1"): RunSpec(
        scene="Curasao",
        run="M1",
        config_relpath=(
            "outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/"
            "cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml"
        ),
        parameterization="legacy",
    ),
    ("Curasao", "BND-K1"): RunSpec(
        scene="Curasao",
        run="BND-K1",
        config_relpath=(
            "outputs/dewater_bounded_sh3_scratch_20260808/"
            "dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000/water-splatting/"
            "dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000_20260808_bounded_sh3_scratch_full_bnd-scratch_g1p00/"
            "config.yml"
        ),
        parameterization="bounded_sh3",
    ),
    ("JapaneseGradens", "M1"): RunSpec(
        scene="JapaneseGradens",
        run="M1",
        config_relpath=(
            "outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/"
            "cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml"
        ),
        parameterization="legacy",
    ),
    ("JapaneseGradens", "BND-K1"): RunSpec(
        scene="JapaneseGradens",
        run="BND-K1",
        config_relpath=(
            "outputs/dewater_bounded_sh3_cross_scene_20260808/"
            "dewater_bnd_cross_scene_japanesegradens_seed42_step0_to_15000/water-splatting/"
            "dewater_bnd_cross_scene_japanesegradens_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_japanesegradens_bnd_g1p00/"
            "config.yml"
        ),
        parameterization="bounded_sh3",
    ),
    ("IUI3", "M1"): RunSpec(
        scene="IUI3",
        run="M1",
        config_relpath=(
            "outputs/gmvc_v3_four_scene_iui3_m1_seed42_15000/water-splatting/"
            "gmvc_v3_four_scene_iui3_m1_seed42_15000_20260806_gmvc_four_scene_p30_mhold_15k_m1_bootstrap/"
            "config.yml"
        ),
        parameterization="legacy",
    ),
    ("IUI3", "BND-K1"): RunSpec(
        scene="IUI3",
        run="BND-K1",
        config_relpath=(
            "outputs/dewater_bounded_sh3_cross_scene_20260808/"
            "dewater_bnd_cross_scene_iui3_seed42_step0_to_15000/water-splatting/"
            "dewater_bnd_cross_scene_iui3_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_iui3_bnd_g1p00/"
            "config.yml"
        ),
        parameterization="bounded_sh3",
    ),
    ("Panama", "M1"): RunSpec(
        scene="Panama",
        run="M1",
        config_relpath=(
            "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/"
            "cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
        ),
        parameterization="legacy",
    ),
    ("Panama", "BND-K1"): RunSpec(
        scene="Panama",
        run="BND-K1",
        config_relpath=(
            "outputs/dewater_bounded_sh3_cross_scene_20260808/"
            "dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/"
            "dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/"
            "config.yml"
        ),
        parameterization="bounded_sh3",
    ),
    ("Panama", "K2"): RunSpec(
        scene="Panama",
        run="K2",
        config_relpath=(
            "outputs/bnd_aopt_equivalence_panama_20260809/"
            "bnd_aopt_panama_seed42_k2_step0_to_15000/water-splatting/20260809_bnd_aopt_k2/config.yml"
        ),
        parameterization="bounded_sh3",
        role="secondary",
        appearance_lr_scale=2.0,
    ),
    ("Panama", "K4"): RunSpec(
        scene="Panama",
        run="K4",
        config_relpath=(
            "outputs/bnd_aopt_equivalence_panama_20260809/"
            "bnd_aopt_panama_seed42_k4_step0_to_15000/water-splatting/20260809_bnd_aopt_k4/config.yml"
        ),
        parameterization="bounded_sh3",
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
    items = [float(value) for value in values if float(value) == float(value)]
    return float(sum(items) / len(items)) if items else float("nan")


def _safe_quantile(values: Tensor, q: float) -> float:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    if q <= 0.0:
        return float(flat.min().item())
    if q >= 1.0:
        return float(flat.max().item())
    rank = max(1, min(flat.numel(), int(math.ceil(float(q) * flat.numel()))))
    return float(torch.kthvalue(flat, rank).values.item())


def _stats(values: Tensor, prefix: str = "") -> Dict[str, float]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    names = ("count", "mean", "p01", "p05", "p10", "p50", "p90", "p95", "p99", "min", "max")
    if flat.numel() == 0:
        return {f"{prefix}{name}": float("nan") for name in names}
    return {
        f"{prefix}count": int(flat.numel()),
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}p01": _safe_quantile(flat, 0.01),
        f"{prefix}p05": _safe_quantile(flat, 0.05),
        f"{prefix}p10": _safe_quantile(flat, 0.10),
        f"{prefix}p50": _safe_quantile(flat, 0.50),
        f"{prefix}p90": _safe_quantile(flat, 0.90),
        f"{prefix}p95": _safe_quantile(flat, 0.95),
        f"{prefix}p99": _safe_quantile(flat, 0.99),
        f"{prefix}min": float(flat.min().item()),
        f"{prefix}max": float(flat.max().item()),
    }


def _channel_stats(values: Tensor, prefix: str) -> Dict[str, float]:
    values = values.detach().float()
    out: Dict[str, float] = {}
    if values.ndim > 0 and values.shape[-1] == 3:
        for index, channel in enumerate(CHANNELS):
            out.update(_stats(values[..., index], f"{prefix}_{channel}_"))
        out.update(_stats(values.reshape(-1), f"{prefix}_all_"))
    else:
        out.update(_stats(values.reshape(-1), f"{prefix}_"))
    return out


def _threshold_fraction(values: Tensor, threshold: float, op: str) -> float:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    if op == "gt":
        return float((flat > float(threshold)).float().mean().item())
    if op == "lt":
        return float((flat < float(threshold)).float().mean().item())
    if op == "abs_gt":
        return float((flat.abs() > float(threshold)).float().mean().item())
    raise ValueError(op)


def _threshold_rows(values: Tensor, prefix: str, thresholds: Sequence[float], op: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    values = values.detach().float()
    if values.ndim > 0 and values.shape[-1] == 3:
        for index, channel in enumerate(CHANNELS):
            for threshold in thresholds:
                out[f"{prefix}_{channel}_{op}_{threshold:g}"] = _threshold_fraction(values[..., index], threshold, op)
        for threshold in thresholds:
            out[f"{prefix}_all_{op}_{threshold:g}"] = _threshold_fraction(values.reshape(-1), threshold, op)
    else:
        for threshold in thresholds:
            out[f"{prefix}_{op}_{threshold:g}"] = _threshold_fraction(values, threshold, op)
    return out


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
    available = _available_steps(config_path)
    if nominal_step in available:
        return nominal_step
    if nominal_step == 15000 and 14999 in available:
        return 14999
    return None


def _load_run(repo: Path, spec: RunSpec) -> LoadedRun:
    config_path = repo / spec.config_relpath
    actual_step = _actual_step(config_path, spec.nominal_step)
    if actual_step is None:
        raise FileNotFoundError(f"Missing checkpoint for {spec.scene} {spec.run}: {config_path}")

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
    return LoadedRun(
        spec=spec,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        loaded_step=loaded_step,
        config=config,
        pipeline=pipeline,
    )


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
    rows: List[Tuple[int, str, Any, Mapping[str, Any]]] = []
    for eval_index, (camera, batch) in enumerate(loaded.pipeline.datamanager.fixed_indices_eval_dataloader):
        filename = image_filenames[eval_index] if eval_index < len(image_filenames) else Path(f"eval_{eval_index}")
        rows.append((eval_index, Path(filename).stem, camera, batch))
    return rows


def _metric_images(model: Any, pred: Tensor, gt: Tensor) -> Dict[str, float]:
    pred = pred.detach().float().clamp(0.0, 1.0).to(model.device)
    gt = gt.detach().float().clamp(0.0, 1.0).to(model.device)
    gt_nchw = torch.moveaxis(gt, -1, 0)[None, ...]
    pred_nchw = torch.moveaxis(pred, -1, 0)[None, ...]
    return {
        "psnr": float(model.psnr(gt_nchw, pred_nchw).item()),
        "ssim": float(model.ssim(gt_nchw, pred_nchw).item()),
        "lpips": float(model.lpips(gt_nchw, pred_nchw).item()),
    }


def _mse(pred: Tensor, gt: Tensor) -> float:
    return float(((pred.detach().float() - gt.detach().float()) ** 2).mean().item())


def _safe_cpu(value: Tensor) -> Tensor:
    return value.detach().float().cpu()


def _camera_position(camera: Cameras) -> Tensor:
    return camera.camera_to_worlds[..., :3, 3]


def _viewdirs(means: Tensor, camera: Cameras) -> Tensor:
    dirs = means.detach() - _camera_position(camera).detach()
    return dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _legacy_colors(model: Any, means: Tensor, features_dc: Tensor, features_rest: Tensor, camera: Cameras, active_sh_degree: int) -> Dict[str, Tensor]:
    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    viewdirs = _viewdirs(means, camera)
    if model.config.sh_degree > 0:
        raw_full = spherical_harmonics(active_sh_degree, viewdirs, colors)
        raw_dc = spherical_harmonics(0, viewdirs, colors[:, :1, :])
        full = torch.clamp(raw_full + 0.5, min=0.0)
        dc = torch.clamp(raw_dc + 0.5, min=0.0)
        return {
            "full": full,
            "dc": dc,
            "raw_full_without_offset": raw_full,
            "raw_dc_without_offset": raw_dc,
            "raw_sh_residual": raw_full - raw_dc,
        }
    dc = torch.sigmoid(colors[:, 0, :])
    return {
        "full": dc,
        "dc": dc,
        "raw_full_without_offset": colors[:, 0, :],
        "raw_dc_without_offset": colors[:, 0, :],
        "raw_sh_residual": torch.zeros_like(dc),
    }


def _bounded_colors(model: Any, means: Tensor, features_dc: Tensor, features_rest: Tensor, camera: Cameras, active_sh_degree: int) -> Dict[str, Tensor]:
    bounded = compute_bounded_gaussian_colors(
        means=means,
        features_dc=features_dc,
        features_rest=features_rest,
        camera_position=_camera_position(camera),
        sh_degree=model.config.sh_degree,
        active_sh_degree=active_sh_degree,
    )
    if bounded.logits is None or bounded.dc_logits is None or bounded.dc_rgb is None or bounded.sigmoid_derivative is None:
        raise RuntimeError("bounded color diagnostic expected logits/dc outputs")
    return {
        "full": bounded.rgb,
        "dc": bounded.dc_rgb,
        "logits": bounded.logits,
        "dc_logits": bounded.dc_logits,
        "logit_residual": bounded.logits - bounded.dc_logits,
        "sigmoid_derivative": bounded.sigmoid_derivative,
    }


@torch.no_grad()
def _diagnostic_render(model: Any, camera: Cameras, mode: str) -> Dict[str, Tensor]:
    """Render with full, dc_only, or projected current-view Gaussian RGB.

    The code mirrors WaterSplattingModel.get_outputs and only replaces the
    current-view Gaussian color tensor before rasterization.
    """

    if not isinstance(camera, Cameras):
        raise TypeError("diagnostic render expects a Cameras object")
    camera = camera.to(model.device)
    camera_downscale = model._get_downscale_factor()
    camera.rescale_output_resolution(1 / camera_downscale)
    try:
        R = camera.camera_to_worlds[0, :3, :3]
        T = camera.camera_to_worlds[0, :3, 3:4]
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=model.device, dtype=R.dtype))
        R = R @ R_edit
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())

        medium = model._predict_medium(
            camera=camera,
            rotation_world_from_camera=R,
            height=H,
            width=W,
            cx=cx,
            cy=cy,
        )
        medium_rgb = medium.rgb
        medium_bs = medium.bs
        medium_attn = medium.attn

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
            height=H,
            width=W,
            clip_thresh=model.config.clip_thresh,
        )
    finally:
        camera.rescale_output_resolution(camera_downscale)

    if radii.sum() == 0:
        rgb = medium_rgb
        depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
        clear = torch.zeros_like(rgb)
        tau_d = medium_attn * depth
        transmission = torch.exp(-tau_d.clamp_min(0.0)).clamp(0.0, 1.0)
        return {
            "rgb": rgb,
            "pred_image": rgb,
            "depth": depth,
            "accumulation": medium_rgb.new_zeros(*rgb.shape[:2], 1),
            "background": medium_rgb,
            "rgb_object": clear,
            "direct_object_signal": clear,
            "rgb_clear": clear,
            "rgb_clear_clamp": clear,
            "clear_object_fullsh_raw": clear,
            "J_gaussian_raw": clear,
            "rgb_medium": medium_rgb,
            "medium_rgb": medium_rgb,
            "medium_bs": medium_bs,
            "medium_attn": medium_attn,
            "b_inf": medium.b_inf,
            "transmission": transmission,
            "tau_D": tau_d,
            "gaussian_view_rgb": torch.empty(0, 3, device=model.device),
            "gaussian_visible_mask": torch.empty(0, dtype=torch.bool, device=model.device),
            "projected_gaussian_depths": depths.detach(),
        }

    active_sh_degree = min(model.step // model.config.sh_degree_interval, model.config.sh_degree)
    parameterization = getattr(model.config, "intrinsic_color_parameterization", "legacy")
    color_diag: Dict[str, Tensor]
    if parameterization == "legacy":
        color_diag = _legacy_colors(model, means_crop, features_dc_crop, features_rest_crop, camera, active_sh_degree)
        full = color_diag["full"]
        if mode == "full":
            rgbs = full
        elif mode == "dc_only":
            rgbs = color_diag["dc"]
        elif mode == "projected":
            rgbs = full.clamp(0.0, 1.0)
        else:
            raise ValueError(mode)
    elif parameterization == "bounded_sh3":
        color_diag = _bounded_colors(model, means_crop, features_dc_crop, features_rest_crop, camera, active_sh_degree)
        if mode == "full":
            rgbs = color_diag["full"]
        elif mode == "dc_only":
            rgbs = color_diag["dc"]
        elif mode == "projected":
            rgbs = color_diag["full"].clamp(0.0, 1.0)
        else:
            raise ValueError(mode)
    else:
        raise ValueError(f"unknown intrinsic_color_parameterization: {parameterization}")

    if model.config.rasterize_mode == "antialiased":
        opacities = torch.sigmoid(opacities_crop) * comp[:, None]
    elif model.config.rasterize_mode == "classic":
        opacities = torch.sigmoid(opacities_crop)
    else:
        raise ValueError(f"unknown rasterize_mode: {model.config.rasterize_mode}")

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
        height=H,
        width=W,
        background=medium_rgb,
        step=model.step,
    )
    rgb = render.rgb
    rgb_medium = render.rgb_medium
    rgb_medium_finite = rgb_medium
    rgb_tail = torch.zeros_like(rgb_medium)
    b_inf = medium.b_inf
    if model._effective_b_inf_mode() == "tied":
        if b_inf is None:
            raise RuntimeError("b_inf_mode='tied' requires b_inf")
        tail_weight = render.final_transmittance * torch.exp(-medium_bs * render.last_depth)
        rgb_tail_original = tail_weight * medium_rgb
        rgb_medium_finite = rgb_medium - rgb_tail_original
        rgb_tail = tail_weight * b_inf
        rgb_medium = rgb_medium_finite + rgb_tail
        rgb = render.rgb_object + rgb_medium

    tau_d = medium_attn * render.depth
    transmission = torch.exp(-tau_d.clamp_min(0.0)).clamp(0.0, 1.0)
    outputs = {
        "rgb": rgb,
        "pred_image": rgb,
        "depth": render.depth,
        "accumulation": render.accumulation,
        "background": medium_rgb,
        "rgb_object": render.rgb_object,
        "direct_object_signal": render.rgb_object,
        "rgb_clear": render.rgb_clear,
        "rgb_clear_clamp": render.rgb_clear_clamp,
        "clear_object_fullsh_raw": render.j_raw,
        "J_gaussian_raw": render.j_raw,
        "J_gaussian": render.j_gaussian,
        "rgb_medium": rgb_medium,
        "rgb_medium_finite": rgb_medium_finite,
        "rgb_tail": rgb_tail,
        "medium_rgb": medium_rgb,
        "medium_bs": medium_bs,
        "medium_attn": medium_attn,
        "b_inf": b_inf,
        "transmission": transmission,
        "tau_D": tau_d,
        "appearance_active_sh_degree": rgbs.new_tensor(float(active_sh_degree)),
        "gaussian_view_rgb": rgbs.detach(),
        "gaussian_view_dc_rgb": color_diag["dc"].detach(),
        "gaussian_visible_mask": (radii > 0).reshape(-1).detach(),
        "projected_gaussian_depths": depths.detach(),
    }
    if parameterization == "legacy":
        outputs["gaussian_view_raw_full_without_offset"] = color_diag["raw_full_without_offset"].detach()
        outputs["gaussian_view_raw_dc_without_offset"] = color_diag["raw_dc_without_offset"].detach()
        outputs["gaussian_raw_sh_residual"] = color_diag["raw_sh_residual"].detach()
    else:
        outputs["gaussian_view_logits"] = color_diag["logits"].detach()
        outputs["gaussian_view_dc_logits"] = color_diag["dc_logits"].detach()
        outputs["gaussian_logit_sh_residual"] = color_diag["logit_residual"].detach()
        outputs["gaussian_sigmoid_derivative"] = color_diag["sigmoid_derivative"].detach()
    return outputs


def _eval_full(model: Any, camera: Cameras, batch: Mapping[str, Any]) -> Tuple[Dict[str, Tensor], Tensor, Dict[str, float]]:
    with torch.no_grad():
        outputs = model.get_outputs_for_camera(camera)
        gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
        metrics = _metric_images(model, outputs["pred_image"], gt)
    return outputs, gt, metrics


def _cache_output_item(
    model: Any,
    eval_index: int,
    view_id: str,
    camera: Cameras,
    batch: Mapping[str, Any],
    include_modes: Sequence[str],
) -> Dict[str, Any]:
    full_normal, gt, full_metrics = _eval_full(model, camera, batch)
    full_diag = _diagnostic_render(model, camera, "full")
    item: Dict[str, Any] = {
        "eval_index": eval_index,
        "view_id": view_id,
        "camera_id": eval_index,
        "gt": _safe_cpu(gt),
        "full": _cpu_output_subset(full_diag),
        "full_metrics": full_metrics,
    }
    if "dc_only" in include_modes:
        dc = _diagnostic_render(model, camera, "dc_only")
        item["dc_only"] = _cpu_output_subset(dc)
        item["dc_metrics"] = _metric_images(model, dc["pred_image"], gt)
    if "projected" in include_modes:
        proj = _diagnostic_render(model, camera, "projected")
        item["projected"] = _cpu_output_subset(proj)
        item["projected_metrics"] = _metric_images(model, proj["pred_image"], gt)
    if "full_audit" in include_modes:
        item["full_forward_audit"] = _forward_diff(full_normal, full_diag)
    return item


def _cpu_output_subset(outputs: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    keep = (
        "pred_image",
        "direct_object_signal",
        "clear_object_fullsh_raw",
        "rgb_clear",
        "rgb_clear_clamp",
        "transmission",
        "tau_D",
        "rgb_medium",
        "medium_rgb",
        "medium_bs",
        "medium_attn",
        "b_inf",
        "depth",
        "accumulation",
        "gaussian_view_rgb",
        "gaussian_view_dc_rgb",
        "gaussian_view_logits",
        "gaussian_view_dc_logits",
        "gaussian_sigmoid_derivative",
        "gaussian_raw_sh_residual",
        "gaussian_logit_sh_residual",
        "gaussian_visible_mask",
        "projected_gaussian_depths",
        "appearance_active_sh_degree",
    )
    out: Dict[str, Tensor] = {}
    for key in keep:
        value = outputs.get(key)
        if isinstance(value, Tensor):
            out[key] = _safe_cpu(value)
    return out


def _forward_diff(reference: Mapping[str, Tensor], candidate: Mapping[str, Tensor]) -> Dict[str, float]:
    keys = (
        "pred_image",
        "direct_object_signal",
        "clear_object_fullsh_raw",
        "rgb_medium",
        "medium_rgb",
        "medium_bs",
        "medium_attn",
        "depth",
        "accumulation",
        "transmission",
        "tau_D",
    )
    out: Dict[str, float] = {}
    for key in keys:
        if key not in reference or key not in candidate:
            continue
        a = reference[key].detach().float().cpu()
        b = candidate[key].detach().float().cpu()
        if tuple(a.shape) != tuple(b.shape):
            out[f"{key}_shape_match"] = 0.0
            continue
        out[f"{key}_max_abs"] = float((a - b).abs().max().item())
        out[f"{key}_mean_abs"] = float((a - b).abs().mean().item())
    return out


def _checkpoint_row(loaded: LoadedRun, view_count: int) -> Dict[str, Any]:
    model = loaded.model
    config = loaded.config
    spec = loaded.spec
    return {
        "scene": spec.scene,
        "run": spec.run,
        "role": spec.role,
        "config_path": str(loaded.config_path),
        "checkpoint_path": str(loaded.checkpoint_path),
        "nominal_step": spec.nominal_step,
        "loaded_step": int(loaded.loaded_step),
        "seed": getattr(getattr(config, "machine", None), "seed", ""),
        "sh_degree": getattr(model.config, "sh_degree", ""),
        "intrinsic_color_parameterization": getattr(model.config, "intrinsic_color_parameterization", ""),
        "medium_context_mode": getattr(model.config, "medium_context_mode", ""),
        "b_inf_mode": getattr(model.config, "b_inf_mode", ""),
        "infinite_water_enabled": getattr(model.config, "infinite_water_enabled", ""),
        "appearance_lr_scale": spec.appearance_lr_scale,
        "gaussian_count": int(model.num_points),
        "num_eval_views": int(view_count),
    }


def _visible_gaussian_values(outputs: Mapping[str, Tensor], key: str) -> Tensor:
    tensor = outputs[key].detach().float()
    visible = outputs.get("gaussian_visible_mask")
    if isinstance(visible, Tensor) and visible.numel() == tensor.shape[0]:
        tensor = tensor[visible.bool()]
    return tensor


def _concat_visible(items: Sequence[Mapping[str, Any]], mode: str, key: str) -> Tensor:
    vals = []
    for item in items:
        outputs = item[mode]
        if key not in outputs:
            continue
        value = _visible_gaussian_values(outputs, key)
        if value.numel() > 0:
            vals.append(value)
    return torch.cat(vals, dim=0) if vals else torch.empty(0, 3)


def _object_support(outputs: Mapping[str, Tensor]) -> Tensor:
    return outputs["accumulation"].detach().float()[..., 0] > 0.01


def _image_values(
    items: Sequence[Mapping[str, Any]],
    mode: str,
    key: str,
    channel: int,
    support: str,
) -> Tensor:
    vals = []
    for item in items:
        tensor = item[mode][key].detach().float()
        if tensor.ndim == 2:
            tensor = tensor[..., None]
        if tensor.shape[-1] == 1:
            image = tensor[..., 0]
        else:
            image = tensor[..., channel]
        if support == "object":
            mask = _object_support(item[mode])
            image = image[mask]
        vals.append(image.reshape(-1))
    return torch.cat(vals, dim=0) if vals else torch.empty(0)


def _pooled_channel_mean_stat(items: Sequence[Mapping[str, Any]], mode: str, key: str, stat: str, support: str) -> float:
    q_map = {"p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99}
    values = []
    for channel in range(3):
        flat = _image_values(items, mode, key, channel, support)
        if stat == "mean":
            values.append(float(flat.mean().item()) if flat.numel() else float("nan"))
        else:
            values.append(_safe_quantile(flat, q_map[stat]))
    return _mean(values)


def _mean_view_channel_mean_stat(items: Sequence[Mapping[str, Any]], mode: str, key: str, stat: str, support: str) -> float:
    q_map = {"p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99}
    per_view = []
    for item in items:
        tensor = item[mode][key].detach().float()
        if tensor.ndim == 2:
            tensor = tensor[..., None]
        stats = []
        for channel in range(3):
            image = tensor[..., 0] if tensor.shape[-1] == 1 else tensor[..., channel]
            if support == "object":
                image = image[_object_support(item[mode])]
            flat = image.reshape(-1)
            if stat == "mean":
                stats.append(float(flat.mean().item()) if flat.numel() else float("nan"))
            else:
                stats.append(_safe_quantile(flat, q_map[stat]))
        per_view.append(_mean(stats))
    return _mean(per_view)


def _mean_view_flat_rgb_stat(items: Sequence[Mapping[str, Any]], mode: str, key: str, stat: str, support: str) -> float:
    q_map = {"p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99}
    per_view = []
    for item in items:
        tensor = item[mode][key].detach().float()
        if tensor.ndim == 2:
            tensor = tensor[..., None]
        if support == "object":
            mask = _object_support(item[mode])
            while mask.ndim < tensor.ndim:
                mask = mask[..., None].expand(*tensor.shape)
            flat = tensor[mask].reshape(-1)
        else:
            flat = tensor.reshape(-1)
        if stat == "mean":
            per_view.append(float(flat.mean().item()) if flat.numel() else float("nan"))
        else:
            per_view.append(_safe_quantile(flat, q_map[stat]))
    return _mean(per_view)


def _threshold_mean_view_flat_rgb(
    items: Sequence[Mapping[str, Any]],
    mode: str,
    key: str,
    threshold: float,
    op: str,
    support: str,
) -> float:
    per_view = []
    for item in items:
        tensor = item[mode][key].detach().float()
        if tensor.ndim == 2:
            tensor = tensor[..., None]
        if support == "object":
            mask = _object_support(item[mode])
            while mask.ndim < tensor.ndim:
                mask = mask[..., None].expand(*tensor.shape)
            flat = tensor[mask].reshape(-1)
        else:
            flat = tensor.reshape(-1)
        if flat.numel() == 0:
            continue
        if op == "lt":
            per_view.append(float((flat < threshold).float().mean().item()))
        else:
            per_view.append(float((flat > threshold).float().mean().item()))
    return _mean(per_view)


def _canonical_rows(scene: str, run: str, items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for support in ("object", "all"):
        for aggregation in ("pooled", "mean_view"):
            stat_fn = _pooled_channel_mean_stat if aggregation == "pooled" else _mean_view_channel_mean_stat
            row = {
                "scene": scene,
                "run": run,
                "support": support,
                "aggregation": aggregation,
                "tau_metric_name": f"tau_eval_{support}_support_{aggregation}_channel_mean_p90",
                "J_metric_name": f"J_clear_eval_{support}_support_{aggregation}_channel_mean_p99",
                "tau_p90": stat_fn(items, "full", "tau_D", "p90", support),
                "J_p99": stat_fn(items, "full", "clear_object_fullsh_raw", "p99", support),
                "T_lt_0.1": _threshold_pooled(items, "full", "transmission", 0.1, "lt", support),
                "P_J_gt_1": _threshold_pooled(items, "full", "clear_object_fullsh_raw", 1.0, "gt", support),
            }
            out.append(row)
    out.append(
        {
            "scene": scene,
            "run": run,
            "support": "all",
            "aggregation": "mean_view_flattened_rgb",
            "tau_metric_name": "old_AOPT_tau_D_all_p90",
            "J_metric_name": "old_AOPT_J_all_p99",
            "tau_p90": _mean_view_flat_rgb_stat(items, "full", "tau_D", "p90", "all"),
            "J_p99": _mean_view_flat_rgb_stat(items, "full", "clear_object_fullsh_raw", "p99", "all"),
            "T_lt_0.1": _threshold_mean_view_flat_rgb(items, "full", "transmission", 0.1, "lt", "all"),
            "P_J_gt_1": _threshold_mean_view_flat_rgb(items, "full", "clear_object_fullsh_raw", 1.0, "gt", "all"),
            "definition_note": "exact current AOPT/tradeoff scalar style: per view flatten H*W*C RGB values, compute stat, then average view scalars",
        }
    )
    return out


def _threshold_pooled(
    items: Sequence[Mapping[str, Any]],
    mode: str,
    key: str,
    threshold: float,
    op: str,
    support: str,
) -> float:
    values = []
    for channel in range(3):
        flat = _image_values(items, mode, key, channel, support)
        if flat.numel() == 0:
            continue
        if op == "lt":
            values.append(float((flat < threshold).float().mean().item()))
        else:
            values.append(float((flat > threshold).float().mean().item()))
    return _mean(values)


def _scene_rgb_metrics(scene: str, run: str, items: Sequence[Mapping[str, Any]], mode: str, label: str) -> Dict[str, Any]:
    rows = []
    for item in items:
        metrics = item["full_metrics"] if mode == "full" else item[f"{mode}_metrics"]
        gt = item["gt"]
        pred = item[mode]["pred_image"]
        rows.append(
            {
                "mse": _mse(pred, gt),
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
            }
        )
    out = {"scene": scene, "run": run, "image": label, "num_eval_views": len(rows)}
    for key in ("mse", "psnr", "ssim", "lpips"):
        out[key] = _mean(row[key] for row in rows)
    return out


def _sh_residual_stats(scene: str, run: str, items: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    full = _concat_visible(items, "full", "gaussian_view_rgb")
    dc = _concat_visible(items, "full", "gaussian_view_dc_rgb")
    if full.numel() == 0 or dc.numel() == 0:
        empty = {"scene": scene, "run": run, "status": "NO_VISIBLE_GAUSSIANS"}
        return empty, empty.copy(), empty.copy()
    residual = full - dc
    r_sh = torch.linalg.norm(residual, dim=-1)
    luma_w = LUMA_WEIGHTS.to(residual.device)
    delta_luma = (residual * luma_w).sum(dim=-1)
    w_norm_sq = float((luma_w * luma_w).sum().item())
    luma_projection = delta_luma[:, None] * luma_w[None, :] / max(w_norm_sq, EPS)
    chroma = torch.linalg.norm(residual - luma_projection, dim=-1)

    row = {"scene": scene, "run": run, "visible_gaussian_observation_count": int(full.shape[0])}
    row.update(_channel_stats(residual, "delta_c_SH"))
    row.update(_threshold_rows(residual, "delta_c_SH", (0.01, 0.05, 0.10, 0.20), "abs_gt"))
    for index, channel in enumerate(CHANNELS):
        row[f"delta_c_SH_{channel}_positive_fraction"] = _threshold_fraction(residual[:, index], 0.0, "gt")
        row[f"delta_c_SH_{channel}_negative_fraction"] = _threshold_fraction(residual[:, index], 0.0, "lt")
    row.update(_stats(r_sh, "R_SH_"))

    luma_row = {
        "scene": scene,
        "run": run,
        "luma_formula": "Y=0.2126R+0.7152G+0.0722B in linear renderer RGB tensor space; no gamma correction",
        "positive_luma_residual_fraction": _threshold_fraction(delta_luma, 0.0, "gt"),
        "negative_luma_residual_fraction": _threshold_fraction(delta_luma, 0.0, "lt"),
    }
    luma_row.update(_stats(delta_luma, "delta_luma_"))
    luma_row.update(_stats(chroma, "chroma_residual_magnitude_"))

    logit_row = {"scene": scene, "run": run}
    if run == "M1":
        raw = _concat_visible(items, "full", "gaussian_raw_sh_residual")
        logit_row["raw_residual_space"] = "legacy spherical_harmonics(active)-spherical_harmonics(0), before +0.5 offset and min clamp"
        if raw.numel() > 0:
            logit_row.update(_channel_stats(raw, "legacy_raw_sh_residual"))
    else:
        raw = _concat_visible(items, "full", "gaussian_logit_sh_residual")
        logit_row["raw_residual_space"] = "bounded logit residual s_full(v)-s_dc"
        if raw.numel() > 0:
            logit_row.update(_channel_stats(raw, "bounded_logit_sh_residual"))
        deriv = _concat_visible(items, "full", "gaussian_sigmoid_derivative")
        if deriv.numel() > 0:
            logit_row.update(_channel_stats(deriv, "sigmoid_derivative"))
            logit_row.update(_threshold_rows(deriv, "sigmoid_derivative", (0.01, 0.05), "lt"))
    return row, luma_row, logit_row


def _range_classification(scene: str, items: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    full = _concat_visible(items, "full", "gaussian_view_rgb")
    dc = _concat_visible(items, "full", "gaussian_view_dc_rgb")
    residual = full - dc
    if full.numel() == 0:
        return [], {"scene": scene}, {"scene": scene}, {"scene": scene}

    dc_valid = (dc >= 0.0) & (dc <= 1.0)
    full_valid = (full >= 0.0) & (full <= 1.0)
    classes = {
        "VALID_TO_VALID": dc_valid & full_valid,
        "VALID_TO_OVERFLOW": dc_valid & (full > 1.0),
        "VALID_TO_UNDERFLOW": dc_valid & (full < 0.0),
        "BASE_ALREADY_OVERFLOW": dc > 1.0,
        "BASE_ALREADY_UNDERFLOW": dc < 0.0,
    }
    other = torch.ones_like(full, dtype=torch.bool)
    for mask in classes.values():
        other &= ~mask
    classes["OTHER"] = other

    rows: List[Dict[str, Any]] = []
    total_channels = float(full.numel())
    total_gauss = float(full.shape[0])
    for name, mask in classes.items():
        row = {"scene": scene, "run": "M1", "class": name}
        row["channel_observation_fraction_all"] = float(mask.float().mean().item())
        row["gaussian_observation_fraction_any_channel"] = float(mask.any(dim=-1).float().mean().item())
        for index, channel in enumerate(CHANNELS):
            row[f"channel_observation_fraction_{channel}"] = float(mask[:, index].float().mean().item())
            row[f"channel_observation_count_{channel}"] = int(mask[:, index].sum().item())
        row["channel_observation_count_all"] = int(mask.sum().item())
        row["gaussian_observation_count_any_channel"] = int(mask.any(dim=-1).sum().item())
        row["total_channel_observations"] = int(total_channels)
        row["total_gaussian_observations"] = int(total_gauss)
        rows.append(row)

    energy_total = float(residual.square().sum().item())
    energy_row = {"scene": scene, "run": "M1", "E_total": energy_total}
    for name, mask in classes.items():
        energy = float(residual.square()[mask].sum().item())
        energy_row[f"E_{name}"] = energy
        energy_row[f"{name}_SH_ENERGY_FRACTION"] = energy / max(energy_total, EPS)
    energy_row["LEGAL_SH_ENERGY_FRACTION"] = energy_row.get("VALID_TO_VALID_SH_ENERGY_FRACTION", float("nan"))
    energy_row["OVERFLOW_SH_ENERGY_FRACTION"] = energy_row.get("VALID_TO_OVERFLOW_SH_ENERGY_FRACTION", float("nan"))
    energy_row["BASE_INVALID_SH_ENERGY_FRACTION"] = (
        energy_row.get("BASE_ALREADY_OVERFLOW_SH_ENERGY_FRACTION", 0.0)
        + energy_row.get("BASE_ALREADY_UNDERFLOW_SH_ENERGY_FRACTION", 0.0)
    )

    audit_row = {"scene": scene, "run": "M1"}
    audit_row.update(_threshold_rows(dc, "c_dc_legacy", (0.0,), "lt"))
    audit_row.update(_threshold_rows(dc, "c_dc_legacy", (1.0,), "gt"))
    audit_row.update(_threshold_rows(full, "c_full_legacy", (0.0,), "lt"))
    audit_row.update(_threshold_rows(full, "c_full_legacy", (1.0,), "gt"))

    proj_changed = (full.clamp(0.0, 1.0) - full).abs() > 0
    proj_delta = (full.clamp(0.0, 1.0) - full).abs()
    proj_row = {
        "scene": scene,
        "run": "M1",
        "P_c_full_gt_1": _threshold_fraction(full, 1.0, "gt"),
        "P_c_full_lt_0": _threshold_fraction(full, 0.0, "lt"),
        "projection_changed_channel_fraction": float(proj_changed.float().mean().item()),
        "projection_changed_gaussian_fraction": float(proj_changed.any(dim=-1).float().mean().item()),
    }
    proj_row.update(_stats(proj_delta.reshape(-1), "projection_magnitude_"))
    return rows, energy_row, audit_row, proj_row


def _headroom(scene: str, run: str, items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    full = _concat_visible(items, "full", "gaussian_view_rgb")
    dc = _concat_visible(items, "full", "gaussian_view_dc_rgb")
    if full.numel() == 0:
        return {"scene": scene, "run": run, "status": "NO_VISIBLE_GAUSSIANS"}
    delta = full - dc
    valid = (dc > 0.0) & (dc < 1.0) & torch.isfinite(delta)
    pos = delta > 0
    neg = delta < 0
    u = torch.zeros_like(delta)
    u[pos] = delta[pos] / (1.0 - dc[pos] + EPS)
    u[neg] = (-delta[neg]) / (dc[neg] + EPS)
    values = u[valid & (pos | neg)]
    row = {
        "scene": scene,
        "run": run,
        "valid_headroom_channel_observation_count": int(values.numel()),
        "HEADROOM_EXCEED_FRACTION": _threshold_fraction(values, 1.0, "gt"),
    }
    row.update(_stats(values, "u_"))
    for threshold in (0.5, 0.75, 1.0, 2.0):
        row[f"P_u_gt_{threshold:g}"] = _threshold_fraction(values, threshold, "gt")
    return row


def _dc_full_image_rows(scene: str, run: str, items: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_view = []
    delta_j_values = []
    luma_values = []
    for item in items:
        full_metrics = item["full_metrics"]
        dc_metrics = item["dc_metrics"]
        delta_j = item["full"]["clear_object_fullsh_raw"] - item["dc_only"]["clear_object_fullsh_raw"]
        delta_j_abs = delta_j.abs()
        delta_j_values.append(delta_j_abs.reshape(-1, 3))
        luma = (delta_j * LUMA_WEIGHTS).sum(dim=-1)
        luma_values.append(luma.reshape(-1))
        per_view.append(
            {
                "scene": scene,
                "run": run,
                "view_id": item["view_id"],
                "PSNR_full": full_metrics["psnr"],
                "SSIM_full": full_metrics["ssim"],
                "LPIPS_full": full_metrics["lpips"],
                "PSNR_dc": dc_metrics["psnr"],
                "SSIM_dc": dc_metrics["ssim"],
                "LPIPS_dc": dc_metrics["lpips"],
                "SH_RGB_GAIN_PSNR": full_metrics["psnr"] - dc_metrics["psnr"],
                "SH_RGB_GAIN_SSIM": full_metrics["ssim"] - dc_metrics["ssim"],
                "SH_RGB_GAIN_LPIPS": dc_metrics["lpips"] - full_metrics["lpips"],
            }
        )
    agg = {
        "scene": scene,
        "run": run,
        "view_id": "AGGREGATE",
        "PSNR_full": _mean(row["PSNR_full"] for row in per_view),
        "SSIM_full": _mean(row["SSIM_full"] for row in per_view),
        "LPIPS_full": _mean(row["LPIPS_full"] for row in per_view),
        "PSNR_dc": _mean(row["PSNR_dc"] for row in per_view),
        "SSIM_dc": _mean(row["SSIM_dc"] for row in per_view),
        "LPIPS_dc": _mean(row["LPIPS_dc"] for row in per_view),
        "SH_RGB_GAIN_PSNR": _mean(row["SH_RGB_GAIN_PSNR"] for row in per_view),
        "SH_RGB_GAIN_SSIM": _mean(row["SH_RGB_GAIN_SSIM"] for row in per_view),
        "SH_RGB_GAIN_LPIPS": _mean(row["SH_RGB_GAIN_LPIPS"] for row in per_view),
    }
    if delta_j_values:
        delta_j_all = torch.cat(delta_j_values, dim=0)
        agg.update(_channel_stats(delta_j_all, "abs_Delta_J_SH"))
        agg.update(_stats(torch.linalg.norm(delta_j_all, dim=-1), "Delta_J_SH_norm_"))
    if luma_values:
        luma_all = torch.cat(luma_values, dim=0)
        agg["positive_image_luma_residual_fraction"] = _threshold_fraction(luma_all, 0.0, "gt")
        agg["negative_image_luma_residual_fraction"] = _threshold_fraction(luma_all, 0.0, "lt")
        agg.update(_stats(luma_all, "image_delta_luma_"))
    return per_view, agg


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    starts = np.r_[0, np.nonzero(sorted_values[1:] != sorted_values[:-1])[0] + 1]
    ends = np.r_[starts[1:], values.shape[0]]
    for start, end in zip(starts, ends):
        if end - start > 1:
            ranks[order[start:end]] = 0.5 * (start + end - 1)
    return ranks


def _pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite].astype(np.float64)
    y = y[finite].astype(np.float64)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom <= 1e-20:
        return float("nan")
    return float((x * y).sum() / denom)


def _top_overlap(a: Tensor, b: Tensor, q: float) -> float:
    a_flat = a.detach().float().reshape(-1)
    b_flat = b.detach().float().reshape(-1)
    a_thresh = _safe_quantile(a_flat, 1.0 - q)
    b_thresh = _safe_quantile(b_flat, 1.0 - q)
    ma = a_flat >= a_thresh
    mb = b_flat >= b_thresh
    denom = float(ma.sum().item())
    return float((ma & mb).sum().item() / denom) if denom > 0 else float("nan")


def _m1_masks(m1: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    j_scalar = m1["clear_object_fullsh_raw"].float().amax(dim=-1)
    tau_scalar = m1["tau_D"].float().mean(dim=-1)
    t_scalar = m1["transmission"].float().amin(dim=-1)
    tau90 = _safe_quantile(tau_scalar.reshape(-1), 0.90)
    j95 = _safe_quantile(j_scalar.reshape(-1), 0.95)
    masks = {
        "J1": j_scalar > 1.0,
        "J95": j_scalar >= j95,
        "TAU90": tau_scalar >= tau90,
        "TLOW": t_scalar < 0.1,
    }
    masks["COMP"] = masks["J1"] | masks["J95"] | masks["TAU90"] | masks["TLOW"]
    return masks


def _projection_and_overlap_rows(
    scene: str,
    m1_items: Sequence[Mapping[str, Any]],
    bnd_items: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    bnd_by_view = {item["view_id"]: item for item in bnd_items}
    per_view: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []
    attribution_rows: List[Dict[str, Any]] = []
    image_change_values = []
    for m1 in m1_items:
        view_id = m1["view_id"]
        bnd = bnd_by_view[view_id]
        gt = m1["gt"]
        full_pred = m1["full"]["pred_image"]
        proj_pred = m1["projected"]["pred_image"]
        bnd_pred = bnd["full"]["pred_image"]
        m1_mse = _mse(full_pred, gt)
        proj_mse = _mse(proj_pred, gt)
        bnd_mse = _mse(bnd_pred, gt)
        projection_fraction = (proj_mse - m1_mse) / (bnd_mse - m1_mse) if abs(bnd_mse - m1_mse) > EPS else float("nan")
        row = {
            "scene": scene,
            "view_id": view_id,
            "M1_FULL_PSNR": m1["full_metrics"]["psnr"],
            "M1_FULL_SSIM": m1["full_metrics"]["ssim"],
            "M1_FULL_LPIPS": m1["full_metrics"]["lpips"],
            "M1_FULL_MSE": m1_mse,
            "M1_PROJ_PSNR": m1["projected_metrics"]["psnr"],
            "M1_PROJ_SSIM": m1["projected_metrics"]["ssim"],
            "M1_PROJ_LPIPS": m1["projected_metrics"]["lpips"],
            "M1_PROJ_MSE": proj_mse,
            "BND_K1_PSNR": bnd["full_metrics"]["psnr"],
            "BND_K1_SSIM": bnd["full_metrics"]["ssim"],
            "BND_K1_LPIPS": bnd["full_metrics"]["lpips"],
            "BND_K1_MSE": bnd_mse,
            "PROJECTION_MSE_FRACTION": projection_fraction,
            "PSNR_PROJ_minus_BND": m1["projected_metrics"]["psnr"] - bnd["full_metrics"]["psnr"],
            "MSE_BND_minus_PROJ": bnd_mse - proj_mse,
        }
        per_view.append(row)

        delta_proj = torch.linalg.norm(full_pred - proj_pred, dim=-1)
        e_m1 = torch.linalg.norm(full_pred - gt, dim=-1)
        e_bnd = torch.linalg.norm(bnd_pred - gt, dim=-1)
        delta_e_plus = (e_bnd - e_m1).clamp_min(0.0)
        image_change_values.append(delta_proj.reshape(-1))
        x = delta_proj.detach().float().reshape(-1).numpy()
        y = delta_e_plus.detach().float().reshape(-1).numpy()
        overlap_rows.append(
            {
                "scene": scene,
                "view_id": view_id,
                "pearson_delta_proj_bnd_excess": _pearson_np(x, y),
                "spearman_delta_proj_bnd_excess": _pearson_np(_rankdata(x), _rankdata(y)),
                "top10_spatial_overlap": _top_overlap(delta_proj, delta_e_plus, 0.10),
                "top20_spatial_overlap": _top_overlap(delta_proj, delta_e_plus, 0.20),
                "delta_proj_sum": float(delta_proj.sum().item()),
                "bnd_excess_sum": float(delta_e_plus.sum().item()),
            }
        )

        masks = _m1_masks(m1["full"])
        legacy_resid = _visible_image_proxy(m1["full"]["clear_object_fullsh_raw"] - m1["dc_only"]["clear_object_fullsh_raw"])
        projection_energy = delta_proj.square()
        bnd_excess_energy = delta_e_plus.square()
        fields = {
            "legacy_SH_image_residual_energy": legacy_resid.square(),
            "projection_change_energy": projection_energy,
            "BND_excess_RGB_residual_energy": bnd_excess_energy,
        }
        for field_name, energy in fields.items():
            total = float(energy.sum().item())
            for mask_name in ("J1", "J95", "TAU90", "TLOW", "COMP"):
                mask = masks[mask_name]
                area = float(mask.float().mean().item())
                fraction = float(energy[mask].sum().item() / max(total, EPS))
                attribution_rows.append(
                    {
                        "scene": scene,
                        "view_id": view_id,
                        "energy_field": field_name,
                        "mask": mask_name,
                        "mask_area": area,
                        "energy_fraction_inside_mask": fraction,
                        "enrichment_ratio": fraction / area if area > EPS else float("inf"),
                    }
                )
    agg = {"scene": scene, "view_id": "AGGREGATE"}
    for key in (
        "M1_FULL_PSNR",
        "M1_FULL_SSIM",
        "M1_FULL_LPIPS",
        "M1_FULL_MSE",
        "M1_PROJ_PSNR",
        "M1_PROJ_SSIM",
        "M1_PROJ_LPIPS",
        "M1_PROJ_MSE",
        "BND_K1_PSNR",
        "BND_K1_SSIM",
        "BND_K1_LPIPS",
        "BND_K1_MSE",
        "PROJECTION_MSE_FRACTION",
        "PSNR_PROJ_minus_BND",
        "MSE_BND_minus_PROJ",
    ):
        agg[key] = _mean(row[key] for row in per_view)
    if image_change_values:
        all_change = torch.cat(image_change_values, dim=0)
        agg.update(_stats(all_change, "image_abs_I_M1_minus_I_PROJ_"))
    return per_view, agg, overlap_rows, attribution_rows


def _visible_image_proxy(rgb: Tensor) -> Tensor:
    return torch.linalg.norm(rgb.detach().float(), dim=-1)


def _aggregate_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], id_fields: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(id_fields)
    for key in keys:
        out[key] = _mean(row[key] for row in rows if key in row)
    return out


def _forward_audit_rows(scene: str, m1_items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in m1_items:
        audit = item.get("full_forward_audit", {})
        row = {"scene": scene, "run": "M1", "view_id": item["view_id"]}
        row.update(audit)
        row["projection_forward_color_only_change_expected"] = True
        rows.append(row)
    keys = sorted({key for row in rows for key in row if key not in {"scene", "run", "view_id"}})
    agg = {"scene": scene, "run": "M1", "view_id": "AGGREGATE"}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float))]
        if vals:
            agg[key] = max(vals) if key.endswith("_max_abs") else _mean(vals)
    rows.append(agg)
    return rows


def _rgb_to_uint8(image: Tensor) -> Image.Image:
    arr = (image.detach().float().clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    return Image.fromarray(arr, mode="RGB")


def _scalar_to_uint8(values: Tensor, scale: float) -> Image.Image:
    scale = max(float(scale), EPS)
    arr = (values.detach().float().clamp_min(0.0) / scale).clamp(0.0, 1.0)
    arr = (arr * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="L").convert("RGB")


def _signed_to_rgb(values: Tensor, scale: float) -> Image.Image:
    scale = max(float(scale), EPS)
    v = (values.detach().float() / scale).clamp(-1.0, 1.0)
    pos = v.clamp_min(0.0)
    neg = (-v).clamp_min(0.0)
    base = torch.ones((*v.shape, 3), dtype=torch.float32)
    blue = torch.tensor([0.12, 0.30, 1.0], dtype=torch.float32)
    red = torch.tensor([1.0, 0.16, 0.10], dtype=torch.float32)
    rgb = base * (1.0 - pos[..., None]) + red * pos[..., None]
    rgb = rgb * (1.0 - neg[..., None]) + blue * neg[..., None]
    arr = (rgb.clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _mask_to_rgb(mask: Tensor, color: Tuple[int, int, int] = (255, 40, 40)) -> Image.Image:
    m = mask.detach().bool().cpu().numpy()
    arr = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
    arr[m] = np.array(color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _range_mask_image(dc: Tensor, full: Tensor) -> Image.Image:
    h, w = full.shape[:2]
    arr = np.full((h, w, 3), 210, dtype=np.uint8)
    full_over = full.detach().float().amax(dim=-1) > 1.0
    dc_over = dc.detach().float().amax(dim=-1) > 1.0
    full_under = full.detach().float().amin(dim=-1) < 0.0
    arr[full_over.cpu().numpy()] = np.array([220, 35, 35], dtype=np.uint8)
    arr[dc_over.cpu().numpy()] = np.array([255, 160, 20], dtype=np.uint8)
    arr[full_under.cpu().numpy()] = np.array([40, 80, 220], dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _overlay_mask(base: Tensor, mask: Tensor, scale: float, color: Tuple[int, int, int]) -> Image.Image:
    image = _scalar_to_uint8(base, scale).convert("RGB")
    pix = image.load()
    m = mask.detach().bool().cpu()
    for y in range(image.height):
        for x in range(image.width):
            if bool(m[y, x]):
                old = pix[x, y]
                pix[x, y] = tuple(int(0.45 * old[i] + 0.55 * color[i]) for i in range(3))
    return image


def _tile(image: Image.Image, label: str, width: int) -> Image.Image:
    if width > 0 and image.width != width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    label_height = 28
    out = Image.new("RGB", (image.width, image.height + label_height), "white")
    out.paste(image, (0, label_height))
    ImageDraw.Draw(out).text((6, 7), label, fill="black")
    return out


def _save_sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Image.Image]]], tile_width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered_rows: List[Image.Image] = []
    for row in rows:
        tiles = [_tile(image, label, tile_width) for label, image in row]
        width = sum(tile.width for tile in tiles) + 6 * (len(tiles) - 1)
        height = max(tile.height for tile in tiles)
        canvas = Image.new("RGB", (width, height), "white")
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, 0))
            x += tile.width + 6
        rendered_rows.append(canvas)
    if not rendered_rows:
        return
    sheet_width = max(row.width for row in rendered_rows)
    sheet_height = sum(row.height for row in rendered_rows) + 6 * (len(rendered_rows) - 1)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    y = 0
    for row in rendered_rows:
        sheet.paste(row, (0, y))
        y += row.height + 6
    sheet.save(path)


def _sheet_manifest(manifest: List[Dict[str, Any]], path: Path, scene: str, output_type: str, view_ids: Sequence[str]) -> None:
    with Image.open(path) as img:
        width, height = img.size
    manifest.append(
        {
            "file_path": str(path),
            "scene": scene,
            "output_type": output_type,
            "view_ids": ";".join(str(v) for v in view_ids),
            "width": width,
            "height": height,
        }
    )


def _write_scene_visuals(
    scene: str,
    m1_items: Sequence[Mapping[str, Any]],
    bnd_items: Sequence[Mapping[str, Any]],
    render_dir: Path,
    tile_width: int,
    manifest: List[Dict[str, Any]],
    extra_panama: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
) -> None:
    bnd_by_view = {item["view_id"]: item for item in bnd_items}
    view_ids = [item["view_id"] for item in m1_items]
    scene_dir = render_dir / scene

    luma_scale = 0.25
    residual_scale = 1.0
    tau_scale = 1.0
    bnd_excess_scale = 1.0
    for m1 in m1_items:
        bnd = bnd_by_view[m1["view_id"]]
        for item in (m1, bnd):
            delta = item["full"]["clear_object_fullsh_raw"] - item["dc_only"]["clear_object_fullsh_raw"]
            luma_scale = max(luma_scale, float((delta * LUMA_WEIGHTS).sum(dim=-1).abs().max().item()))
            residual_scale = max(residual_scale, float(torch.linalg.norm(delta, dim=-1).max().item()))
            tau_scale = max(tau_scale, float(item["full"]["tau_D"].mean(dim=-1).max().item()))
        excess = (
            torch.linalg.norm(bnd["full"]["pred_image"] - m1["gt"], dim=-1)
            - torch.linalg.norm(m1["full"]["pred_image"] - m1["gt"], dim=-1)
        ).clamp_min(0.0)
        bnd_excess_scale = max(bnd_excess_scale, float(excess.max().item()))

    rows_dc_full: List[List[Tuple[str, Image.Image]]] = []
    rows_signed_luma: List[List[Tuple[str, Image.Image]]] = []
    rows_range_mask: List[List[Tuple[str, Image.Image]]] = []
    rows_projection: List[List[Tuple[str, Image.Image]]] = []
    rows_proj_resid: List[List[Tuple[str, Image.Image]]] = []
    rows_clear: List[List[Tuple[str, Image.Image]]] = []
    for m1 in m1_items:
        view_id = m1["view_id"]
        bnd = bnd_by_view[view_id]
        m1_delta = m1["full"]["clear_object_fullsh_raw"] - m1["dc_only"]["clear_object_fullsh_raw"]
        bnd_delta = bnd["full"]["clear_object_fullsh_raw"] - bnd["dc_only"]["clear_object_fullsh_raw"]
        rows_dc_full.append(
            [
                (f"{view_id} M1 DC", _rgb_to_uint8(m1["dc_only"]["clear_object_fullsh_raw"])),
                ("M1 Full", _rgb_to_uint8(m1["full"]["clear_object_fullsh_raw"])),
                ("M1 |Full-DC|", _scalar_to_uint8(torch.linalg.norm(m1_delta, dim=-1), residual_scale)),
                ("BND DC", _rgb_to_uint8(bnd["dc_only"]["clear_object_fullsh_raw"])),
                ("BND Full", _rgb_to_uint8(bnd["full"]["clear_object_fullsh_raw"])),
                ("BND |Full-DC|", _scalar_to_uint8(torch.linalg.norm(bnd_delta, dim=-1), residual_scale)),
            ]
        )
        rows_signed_luma.append(
            [
                (f"{view_id} M1 delta luma", _signed_to_rgb((m1_delta * LUMA_WEIGHTS).sum(dim=-1), luma_scale)),
                ("BND delta luma", _signed_to_rgb((bnd_delta * LUMA_WEIGHTS).sum(dim=-1), luma_scale)),
            ]
        )
        rows_range_mask.append(
            [
                (
                    f"{view_id} M1 image range mask",
                    _range_mask_image(m1["dc_only"]["clear_object_fullsh_raw"], m1["full"]["clear_object_fullsh_raw"]),
                )
            ]
        )
        rows_projection.append(
            [
                (f"{view_id} GT", _rgb_to_uint8(m1["gt"])),
                ("M1 underwater", _rgb_to_uint8(m1["full"]["pred_image"])),
                ("M1-PROJ underwater", _rgb_to_uint8(m1["projected"]["pred_image"])),
                ("BND underwater", _rgb_to_uint8(bnd["full"]["pred_image"])),
            ]
        )
        delta_proj = torch.linalg.norm(m1["full"]["pred_image"] - m1["projected"]["pred_image"], dim=-1)
        bnd_excess = (
            torch.linalg.norm(bnd["full"]["pred_image"] - m1["gt"], dim=-1)
            - torch.linalg.norm(m1["full"]["pred_image"] - m1["gt"], dim=-1)
        ).clamp_min(0.0)
        masks = _m1_masks(m1["full"])
        rows_proj_resid.append(
            [
                (f"{view_id} |M1-PROJ|", _scalar_to_uint8(delta_proj, bnd_excess_scale)),
                ("BND excess residual", _scalar_to_uint8(bnd_excess, bnd_excess_scale)),
                ("COMP overlay", _overlay_mask(bnd_excess, masks["COMP"], bnd_excess_scale, (255, 40, 180))),
            ]
        )
        rows_clear.append(
            [
                (f"{view_id} M1 raw", _rgb_to_uint8(m1["full"]["clear_object_fullsh_raw"])),
                ("M1 projected", _rgb_to_uint8(m1["projected"]["clear_object_fullsh_raw"])),
                ("BND raw", _rgb_to_uint8(bnd["full"]["clear_object_fullsh_raw"])),
            ]
        )

    for filename, rows, output_type in (
        ("contact_sheet_clear_dc_vs_full.png", rows_dc_full, "clear_dc_vs_full"),
        ("contact_sheet_signed_sh_luma_residual.png", rows_signed_luma, "signed_sh_luma_residual"),
        ("contact_sheet_legacy_range_mask.png", rows_range_mask, "legacy_range_mask"),
        ("contact_sheet_m1_projection_counterfactual.png", rows_projection, "m1_projection_counterfactual"),
        ("contact_sheet_projection_residual_overlap.png", rows_proj_resid, "projection_residual_overlap"),
        ("contact_sheet_clear_raw_projected_bnd.png", rows_clear, "clear_raw_projected_bnd"),
    ):
        path = scene_dir / filename
        _save_sheet(path, rows, tile_width)
        _sheet_manifest(manifest, path, scene, output_type, view_ids)

    if scene == "Panama" and extra_panama:
        _write_panama_visuals(m1_items, bnd_items, extra_panama, scene_dir, tile_width, manifest)


def _write_panama_visuals(
    m1_items: Sequence[Mapping[str, Any]],
    bnd_items: Sequence[Mapping[str, Any]],
    extra: Mapping[str, Sequence[Mapping[str, Any]]],
    scene_dir: Path,
    tile_width: int,
    manifest: List[Dict[str, Any]],
) -> None:
    by_run = {
        "M1": {item["view_id"]: item for item in m1_items},
        "BND-K1": {item["view_id"]: item for item in bnd_items},
    }
    for run, items in extra.items():
        by_run[run] = {item["view_id"]: item for item in items}
    view_ids = [item["view_id"] for item in m1_items]
    runs = ["M1", "M1-PROJ", "BND-K1", "K2", "K4"]

    luma_scale = 0.25
    excess_scale = 1.0
    for view_id in view_ids:
        for run in ("M1", "BND-K1", "K2", "K4"):
            if run not in by_run or view_id not in by_run[run]:
                continue
            item = by_run[run][view_id]
            delta = item["full"]["clear_object_fullsh_raw"] - item["dc_only"]["clear_object_fullsh_raw"]
            luma_scale = max(luma_scale, float((delta * LUMA_WEIGHTS).sum(dim=-1).abs().max().item()))
        m1 = by_run["M1"][view_id]
        bnd = by_run["BND-K1"][view_id]
        excess = (
            torch.linalg.norm(bnd["full"]["pred_image"] - m1["gt"], dim=-1)
            - torch.linalg.norm(m1["full"]["pred_image"] - m1["gt"], dim=-1)
        ).clamp_min(0.0)
        excess_scale = max(excess_scale, float(excess.max().item()))

    rows_underwater = []
    rows_clear = []
    rows_sh = []
    rows_overlay = []
    for view_id in view_ids:
        m1 = by_run["M1"][view_id]
        rows_underwater.append(
            [(f"{view_id} GT", _rgb_to_uint8(m1["gt"]))]
            + [
                (
                    run,
                    _rgb_to_uint8(
                        m1["projected"]["pred_image"] if run == "M1-PROJ" else by_run[run][view_id]["full"]["pred_image"]
                    ),
                )
                for run in runs
            ]
        )
        k1 = by_run["BND-K1"][view_id]
        rows_clear.append(
            [
                (f"{view_id} M1 clear DC", _rgb_to_uint8(m1["dc_only"]["clear_object_fullsh_raw"])),
                ("M1 clear full", _rgb_to_uint8(m1["full"]["clear_object_fullsh_raw"])),
                ("M1 clear projected", _rgb_to_uint8(m1["projected"]["clear_object_fullsh_raw"])),
                ("K1 clear DC", _rgb_to_uint8(k1["dc_only"]["clear_object_fullsh_raw"])),
                ("K1 clear full", _rgb_to_uint8(k1["full"]["clear_object_fullsh_raw"])),
            ]
        )
        sh_row = []
        for run in ("M1", "BND-K1", "K2", "K4"):
            item = by_run[run][view_id]
            delta = item["full"]["clear_object_fullsh_raw"] - item["dc_only"]["clear_object_fullsh_raw"]
            sh_row.append((f"{view_id if not sh_row else ''} {run} SH luma", _signed_to_rgb((delta * LUMA_WEIGHTS).sum(dim=-1), luma_scale)))
        rows_sh.append(sh_row)
        bnd = by_run["BND-K1"][view_id]
        bnd_excess = (
            torch.linalg.norm(bnd["full"]["pred_image"] - m1["gt"], dim=-1)
            - torch.linalg.norm(m1["full"]["pred_image"] - m1["gt"], dim=-1)
        ).clamp_min(0.0)
        j_over = m1["full"]["clear_object_fullsh_raw"].amax(dim=-1) > 1.0
        tau90 = m1["full"]["tau_D"].mean(dim=-1) >= _safe_quantile(m1["full"]["tau_D"].mean(dim=-1).reshape(-1), 0.90)
        rows_overlay.append(
            [
                (f"{view_id} M1 J>1", _mask_to_rgb(j_over)),
                ("M1 tau top10", _mask_to_rgb(tau90, (255, 180, 0))),
                ("BND excess", _scalar_to_uint8(bnd_excess, excess_scale)),
            ]
        )

    for filename, rows, output_type in (
        ("panama_underwater_m1_proj_k1_k2_k4.png", rows_underwater, "panama_underwater_m1_proj_k1_k2_k4"),
        ("panama_clear_dc_full_projected_k1.png", rows_clear, "panama_clear_dc_full_projected_k1"),
        ("panama_sh_luma_m1_k1_k2_k4.png", rows_sh, "panama_sh_luma_m1_k1_k2_k4"),
        ("panama_compensation_overlay.png", rows_overlay, "panama_compensation_overlay"),
    ):
        path = scene_dir / filename
        _save_sheet(path, rows, tile_width)
        _sheet_manifest(manifest, path, "Panama", output_type, view_ids)


def _metric_definition_audit(repo: Path) -> List[Dict[str, Any]]:
    archive_ref = "research-snapshot-20260808-gmvc-dewatering-full"
    return [
        {
            "metric_source": "OLD_CROSS_SCENE_METRIC",
            "script": f"{archive_ref}:scripts/diagnostics/diagnose_dewater_optical_depth.py + summarize_bounded_sh3_cross_scene.py",
            "tau_definition": "outputs['tau_D_effective']; image/ray-level; object_support_mask = outputs['accumulation'] > 0.01; all eval views pooled per RGB channel; p90 per channel; final scalar is mean over RGB channels",
            "J_definition": "outputs['clear_object_fullsh_raw']; image/ray-level clear-object full-SH proxy; same object_support_mask; all eval views pooled per RGB channel; p99 per channel; final scalar is mean over RGB channels",
            "aggregation": "pooled pixels across all eval views before quantile, then RGB-channel mean",
            "view_set": "all eval views from fixed_indices_eval_dataloader",
            "difference_source": "uses object support mask and pooled-sample quantiles",
        },
        {
            "metric_source": "OLD_AOPT_METRIC",
            "script": "scripts/diagnostics/summarize_bnd_aopt_panama.py",
            "tau_definition": "outputs['tau_D']; image/ray-level; no object-support mask; per-view _channel_stats over all pixels; scalar summary averages per-view scalar stats",
            "J_definition": "outputs['clear_object_fullsh_raw']; image/ray-level; no object-support mask; per-view _channel_stats over all pixels; scalar summary averages per-view scalar stats",
            "aggregation": "for tau_D_all_p90/J_all_p99, each view flattens H*W*C RGB samples first, computes the quantile, then arithmetic mean over eval views",
            "view_set": "all eval views from fixed_indices_eval_dataloader",
            "difference_source": "averages view-level flattened-RGB quantiles rather than pooling all eval-view samples before quantile",
        },
        {
            "metric_source": "CANONICAL_PRIMARY_DECOMPOSITION_METRIC",
            "script": "scripts/diagnostics/diagnose_bnd_sh_residual_structure.py",
            "tau_definition": "outputs['tau_D']; image/ray-level direct optical-depth map; object_support_mask = outputs['accumulation'] > 0.01; all eval-view supported pixels pooled per RGB channel; p90 per channel; scalar is RGB-channel mean",
            "J_definition": "outputs['clear_object_fullsh_raw']; image-space alpha-composited full-SH clear-object proxy from rasterizer j_raw; same object_support_mask; all eval-view supported pixels pooled per RGB channel; p99 per channel; scalar is RGB-channel mean",
            "aggregation": "pooled supported pixels across all eval views before quantile, then RGB-channel mean",
            "view_set": "all eval views from fixed_indices_eval_dataloader",
            "canonical_names": "CANONICAL_TAU_METRIC=tau_eval_object_support_pooled_channel_mean_p90; CANONICAL_J_METRIC=J_clear_eval_object_support_pooled_channel_mean_p99",
            "difference_source": "canonical intentionally matches image-space supported-object decomposition maps and avoids background/no-support pixels",
        },
        {
            "metric_source": "CANONICAL_SECONDARY_GAUSSIAN_DIAGNOSTIC",
            "script": "scripts/diagnostics/diagnose_bnd_sh_residual_structure.py",
            "tau_definition": "not Gaussian-level; tau is retained as image/ray metric only",
            "J_definition": "current-view visible Gaussian RGB distributions from outputs['gaussian_view_rgb'] and outputs['gaussian_view_dc_rgb']; visible mask = outputs['gaussian_visible_mask']",
            "aggregation": "all visible Gaussian-view observations pooled; no cross-run Gaussian index matching",
            "view_set": "all eval views",
            "canonical_names": "R_SH_visible_gaussian_observation_distribution, delta_c_SH_visible_gaussian_observation_distribution",
            "difference_source": "answers representation/residual structure, not ray-level decomposition",
        },
    ]


def _sh_color_semantics() -> Dict[str, Any]:
    return {
        "legacy": {
            "code_path": "water_splatting/fields/gaussian_appearance.py::compute_gaussian_colors",
            "c_dc_legacy": "clamp(spherical_harmonics(0, viewdirs, [features_dc]) + 0.5, min=0.0)",
            "c_full_legacy(v)": "clamp(spherical_harmonics(active_sh_degree, viewdirs, [features_dc, features_rest]) + 0.5, min=0.0)",
            "upper_clamp": "none",
            "lower_clamp": "min clamp at 0.0 for SH>0 path",
            "active_sh_degree": "min(model.step // sh_degree_interval, sh_degree); final 14999/15000 checkpoints use active SH3 when sh_degree=3",
            "raw_SH_residual": "spherical_harmonics(active)-spherical_harmonics(0), before +0.5 offset and lower clamp",
        },
        "bounded_sh3": {
            "code_path": "water_splatting/fields/gaussian_appearance.py::compute_bounded_gaussian_colors",
            "s_dc": "spherical_harmonics(0, viewdirs, [features_dc])",
            "s_full(v)": "spherical_harmonics(active_sh_degree, viewdirs, [features_dc, features_rest])",
            "c_dc_bnd": "sigmoid(s_dc)",
            "c_full_bnd(v)": "sigmoid(s_full(v))",
            "logit_residual": "Delta_s_SH(v) = s_full(v) - s_dc",
            "rgb_residual": "Delta_c_SH(v) = c_full_bnd(v) - c_dc_bnd",
            "gradient_path": "RGB loss gradients pass through sigmoid derivative c*(1-c) to active full-SH logits",
        },
        "visibility": {
            "definition": "UNWEIGHTED_VISIBLE_GAUSSIAN observations use outputs['gaussian_visible_mask'] = radii > 0 for each eval camera",
            "weighted_stats": "unavailable; current diagnostic does not expose exact per-Gaussian raster contribution weights",
        },
    }


def run(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    render_dir = args.render_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    start_info = {
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "start_head": _git(repo, "rev-parse", "HEAD"),
        "start_log_1": _git(repo, "log", "-1", "--oneline"),
        "diagnostic": "BND-SHSTRUCT",
        "training": "disabled/read-only",
    }
    _write_json(output_dir / "run_start.json", start_info)

    metric_def_rows = _metric_definition_audit(repo)
    _write_json(output_dir / "metric_definition_audit.json", metric_def_rows)
    _write_csv(output_dir / "metric_definition_audit.csv", metric_def_rows)
    _write_json(output_dir / "sh_color_semantics_audit.json", _sh_color_semantics())

    checkpoint_rows: List[Dict[str, Any]] = []
    canonical_rows: List[Dict[str, Any]] = []
    gaussian_rows: List[Dict[str, Any]] = []
    luma_rows: List[Dict[str, Any]] = []
    raw_residual_rows: List[Dict[str, Any]] = []
    range_rows: List[Dict[str, Any]] = []
    range_audit_rows: List[Dict[str, Any]] = []
    energy_rows: List[Dict[str, Any]] = []
    headroom_rows: List[Dict[str, Any]] = []
    dc_full_rows: List[Dict[str, Any]] = []
    projection_metric_rows: List[Dict[str, Any]] = []
    projection_change_rows: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []
    attribution_rows: List[Dict[str, Any]] = []
    forward_audit_rows: List[Dict[str, Any]] = []
    panama_k_rows: List[Dict[str, Any]] = []
    final_scene_rows: List[Dict[str, Any]] = []

    for scene in args.scenes:
        print(f"[BND-SHSTRUCT] processing scene {scene}", flush=True)
        scene_cache: Dict[str, List[Dict[str, Any]]] = {}
        for run_name in ("M1", "BND-K1"):
            spec = RUN_SPECS[(scene, run_name)]
            loaded: Optional[LoadedRun] = None
            try:
                loaded = _load_run(repo, spec)
                records = _view_records(loaded)
                include_modes = ["dc_only"]
                if run_name == "M1":
                    include_modes.extend(["projected", "full_audit"])
                items = [
                    _cache_output_item(loaded.model, eval_index, view_id, camera, batch, include_modes)
                    for eval_index, view_id, camera, batch in records
                ]
                scene_cache[run_name] = items
                checkpoint_rows.append(_checkpoint_row(loaded, len(items)))
                canonical_rows.extend(_canonical_rows(scene, run_name, items))
                sh_row, luma_row, raw_row = _sh_residual_stats(scene, run_name, items)
                gaussian_rows.append(sh_row)
                luma_rows.append(luma_row)
                raw_residual_rows.append(raw_row)
                headroom_rows.append(_headroom(scene, run_name, items))
                per_view, agg = _dc_full_image_rows(scene, run_name, items)
                dc_full_rows.extend(per_view)
                dc_full_rows.append(agg)
                if run_name == "M1":
                    range_class, energy, range_audit, proj_change = _range_classification(scene, items)
                    range_rows.extend(range_class)
                    energy_rows.append(energy)
                    range_audit_rows.append(range_audit)
                    projection_change_rows.append(proj_change)
                    forward_audit_rows.extend(_forward_audit_rows(scene, items))
            finally:
                _release_loaded(loaded)

        m1_items = scene_cache["M1"]
        bnd_items = scene_cache["BND-K1"]
        if [item["view_id"] for item in m1_items] != [item["view_id"] for item in bnd_items]:
            raise RuntimeError(f"{scene} M1/BND eval view mismatch")
        proj_per_view, proj_agg, scene_overlap, scene_attrib = _projection_and_overlap_rows(scene, m1_items, bnd_items)
        projection_metric_rows.extend(proj_per_view)
        projection_metric_rows.append(proj_agg)
        overlap_rows.extend(scene_overlap)
        overlap_rows.append(_aggregate_rows(scene_overlap, ("pearson_delta_proj_bnd_excess", "spearman_delta_proj_bnd_excess", "top10_spatial_overlap", "top20_spatial_overlap"), {"scene": scene, "view_id": "AGGREGATE"}))
        attribution_rows.extend(scene_attrib)
        attribution_rows.extend(_aggregate_attribution(scene_attrib, scene))

        extra_panama: Dict[str, List[Dict[str, Any]]] = {}
        if scene == "Panama":
            for run_name in ("K2", "K4"):
                spec = RUN_SPECS[(scene, run_name)]
                loaded = None
                try:
                    loaded = _load_run(repo, spec)
                    records = _view_records(loaded)
                    items = [
                        _cache_output_item(loaded.model, eval_index, view_id, camera, batch, ["dc_only"])
                        for eval_index, view_id, camera, batch in records
                    ]
                    extra_panama[run_name] = items
                    checkpoint_rows.append(_checkpoint_row(loaded, len(items)))
                    canonical_rows.extend(_canonical_rows(scene, run_name, items))
                    sh_row, luma_row, raw_row = _sh_residual_stats(scene, run_name, items)
                    gaussian_rows.append(sh_row)
                    luma_rows.append(luma_row)
                    raw_residual_rows.append(raw_row)
                    headroom_rows.append(_headroom(scene, run_name, items))
                    per_view, agg = _dc_full_image_rows(scene, run_name, items)
                    dc_full_rows.extend(per_view)
                    dc_full_rows.append(agg)
                    panama_k_rows.append(_panama_k_row(run_name, items, sh_row, luma_row, raw_row, agg))
                finally:
                    _release_loaded(loaded)
            panama_k_rows.append(_panama_k_row("BND-K1", bnd_items, gaussian_rows[-3], luma_rows[-3], raw_residual_rows[-3], next(row for row in dc_full_rows if row["scene"] == "Panama" and row["run"] == "BND-K1" and row["view_id"] == "AGGREGATE")))

        _write_scene_visuals(scene, m1_items, bnd_items, render_dir, args.tile_width, manifest, extra_panama)
        final_scene_rows.append(_final_scene_row(scene, canonical_rows, gaussian_rows, energy_rows, headroom_rows, dc_full_rows, projection_metric_rows, overlap_rows, attribution_rows))

    classifications = _classification_rows(final_scene_rows)
    summary_payload = {
        "run_start": start_info,
        "canonical_metrics": {
            "CANONICAL_TAU_METRIC": "tau_eval_object_support_pooled_channel_mean_p90",
            "CANONICAL_J_METRIC": "J_clear_eval_object_support_pooled_channel_mean_p99",
            "PRIMARY_SUPPORT_MASK": "outputs['accumulation'] > 0.01",
            "PRIMARY_AGGREGATION": "all eval-view supported pixels pooled per channel before quantile; scalar = RGB-channel mean",
        },
        "scene_summary": final_scene_rows,
        "classification": classifications,
        "visual_manifest": manifest,
    }

    outputs = {
        "checkpoint_manifest": checkpoint_rows,
        "canonical_decomposition_metrics": canonical_rows,
        "sh_residual_distribution": gaussian_rows,
        "sh_luma_chroma_distribution": luma_rows,
        "sh_raw_or_logit_residual_distribution": raw_residual_rows,
        "legacy_range_classification": range_rows,
        "legacy_range_audit": range_audit_rows,
        "sh_residual_energy_decomposition": energy_rows,
        "headroom_utilization": headroom_rows,
        "dc_full_image_metrics": dc_full_rows,
        "m1_projected_counterfactual_metrics": projection_metric_rows,
        "projection_change_statistics": projection_change_rows,
        "projection_forward_audit": forward_audit_rows,
        "projection_bnd_overlap": overlap_rows,
        "compensation_region_sh_attribution": attribution_rows,
        "panama_k1_k2_k4_structure": panama_k_rows,
        "bnd_shstruct_final_summary": final_scene_rows,
    }
    for name, rows in outputs.items():
        _write_json(output_dir / f"{name}.json", rows)
        _write_csv(output_dir / f"{name}.csv", rows if isinstance(rows, list) else [rows])
        manifest.append({"file_path": str(output_dir / f"{name}.json"), "scene": "ALL", "output_type": name})
        manifest.append({"file_path": str(output_dir / f"{name}.csv"), "scene": "ALL", "output_type": name})

    _write_json(output_dir / "bnd_shstruct_final_summary.json", summary_payload)
    _write_csv(output_dir / "bnd_shstruct_final_summary.csv", final_scene_rows)
    _write_json(output_dir / "manifest.json", manifest)
    _write_csv(output_dir / "manifest.csv", manifest)
    _write_json(render_dir / "manifest.json", manifest)
    _write_csv(render_dir / "manifest.csv", manifest)
    _write_visual_index(render_dir / "VISUAL_COMPARE_INDEX.md", manifest)
    _write_visual_index(output_dir / "VISUAL_COMPARE_INDEX.md", manifest)
    _write_json(output_dir / "classification.json", classifications)
    _write_csv(output_dir / "classification.csv", [classifications])


def _aggregate_attribution(rows: Sequence[Mapping[str, Any]], scene: str) -> List[Dict[str, Any]]:
    out = []
    keys = sorted({(row["energy_field"], row["mask"]) for row in rows})
    for energy_field, mask in keys:
        selected = [row for row in rows if row["energy_field"] == energy_field and row["mask"] == mask]
        out.append(
            {
                "scene": scene,
                "view_id": "AGGREGATE",
                "energy_field": energy_field,
                "mask": mask,
                "mask_area": _mean(row["mask_area"] for row in selected),
                "energy_fraction_inside_mask": _mean(row["energy_fraction_inside_mask"] for row in selected),
                "enrichment_ratio": _mean(row["enrichment_ratio"] for row in selected),
            }
        )
    return out


def _panama_k_row(run: str, items: Sequence[Mapping[str, Any]], sh_row: Mapping[str, Any], luma_row: Mapping[str, Any], raw_row: Mapping[str, Any], dc_full_agg: Mapping[str, Any]) -> Dict[str, Any]:
    row = {
        "scene": "Panama",
        "run": run,
        "R_SH_mean": sh_row.get("R_SH_mean", float("nan")),
        "R_SH_p50": sh_row.get("R_SH_p50", float("nan")),
        "R_SH_p90": sh_row.get("R_SH_p90", float("nan")),
        "R_SH_p95": sh_row.get("R_SH_p95", float("nan")),
        "R_SH_p99": sh_row.get("R_SH_p99", float("nan")),
        "positive_luma_residual_fraction": luma_row.get("positive_luma_residual_fraction", float("nan")),
        "negative_luma_residual_fraction": luma_row.get("negative_luma_residual_fraction", float("nan")),
        "chroma_residual_magnitude_mean": luma_row.get("chroma_residual_magnitude_mean", float("nan")),
        "SH_RGB_GAIN_PSNR": dc_full_agg.get("SH_RGB_GAIN_PSNR", float("nan")),
        "PSNR_full": dc_full_agg.get("PSNR_full", float("nan")),
        "PSNR_dc": dc_full_agg.get("PSNR_dc", float("nan")),
    }
    if run != "M1":
        deriv = _concat_visible(items, "full", "gaussian_sigmoid_derivative")
        if deriv.numel() > 0:
            row.update(_channel_stats(deriv, "sigmoid_derivative"))
            row.update(_threshold_rows(deriv, "sigmoid_derivative", (0.01, 0.05), "lt"))
    if raw_row:
        for key, value in raw_row.items():
            if key.startswith("bounded_logit_sh_residual_all_"):
                row[key] = value
    return row


def _lookup(rows: Sequence[Mapping[str, Any]], **criteria: Any) -> Optional[Mapping[str, Any]]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    return None


def _final_scene_row(
    scene: str,
    canonical_rows: Sequence[Mapping[str, Any]],
    gaussian_rows: Sequence[Mapping[str, Any]],
    energy_rows: Sequence[Mapping[str, Any]],
    headroom_rows: Sequence[Mapping[str, Any]],
    dc_full_rows: Sequence[Mapping[str, Any]],
    projection_rows: Sequence[Mapping[str, Any]],
    overlap_rows: Sequence[Mapping[str, Any]],
    attribution_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    m1_can = _lookup(canonical_rows, scene=scene, run="M1", support="object", aggregation="pooled") or {}
    bnd_can = _lookup(canonical_rows, scene=scene, run="BND-K1", support="object", aggregation="pooled") or {}
    m1_sh = _lookup(gaussian_rows, scene=scene, run="M1") or {}
    bnd_sh = _lookup(gaussian_rows, scene=scene, run="BND-K1") or {}
    energy = _lookup(energy_rows, scene=scene, run="M1") or {}
    m1_head = _lookup(headroom_rows, scene=scene, run="M1") or {}
    bnd_head = _lookup(headroom_rows, scene=scene, run="BND-K1") or {}
    m1_dc = _lookup(dc_full_rows, scene=scene, run="M1", view_id="AGGREGATE") or {}
    bnd_dc = _lookup(dc_full_rows, scene=scene, run="BND-K1", view_id="AGGREGATE") or {}
    proj = _lookup(projection_rows, scene=scene, view_id="AGGREGATE") or {}
    overlap = _lookup(overlap_rows, scene=scene, view_id="AGGREGATE") or {}
    comp_proj = _lookup(
        attribution_rows,
        scene=scene,
        view_id="AGGREGATE",
        energy_field="projection_change_energy",
        mask="COMP",
    ) or {}
    comp_excess = _lookup(
        attribution_rows,
        scene=scene,
        view_id="AGGREGATE",
        energy_field="BND_excess_RGB_residual_energy",
        mask="COMP",
    ) or {}
    bnd_gap = float(proj.get("BND_K1_PSNR", float("nan"))) - float(proj.get("M1_FULL_PSNR", float("nan")))
    tau_drop = 1.0 - float(bnd_can.get("tau_p90", float("nan"))) / max(float(m1_can.get("tau_p90", 1.0)), EPS)
    j_drop = 1.0 - float(bnd_can.get("J_p99", float("nan"))) / max(float(m1_can.get("J_p99", 1.0)), EPS)
    row = {
        "scene": scene,
        "M1_tau_p90_canonical": m1_can.get("tau_p90", float("nan")),
        "BND_tau_p90_canonical": bnd_can.get("tau_p90", float("nan")),
        "tau_p90_relative_drop_BND_vs_M1": tau_drop,
        "M1_J_p99_canonical": m1_can.get("J_p99", float("nan")),
        "BND_J_p99_canonical": bnd_can.get("J_p99", float("nan")),
        "J_p99_relative_drop_BND_vs_M1": j_drop,
        "M1_R_SH_p50": m1_sh.get("R_SH_p50", float("nan")),
        "BND_R_SH_p50": bnd_sh.get("R_SH_p50", float("nan")),
        "BND_over_M1_R_SH_p50": float(bnd_sh.get("R_SH_p50", float("nan"))) / max(float(m1_sh.get("R_SH_p50", 1.0)), EPS),
        "LEGAL_SH_ENERGY_FRACTION": energy.get("LEGAL_SH_ENERGY_FRACTION", float("nan")),
        "OVERFLOW_SH_ENERGY_FRACTION": energy.get("OVERFLOW_SH_ENERGY_FRACTION", float("nan")),
        "BASE_INVALID_SH_ENERGY_FRACTION": energy.get("BASE_INVALID_SH_ENERGY_FRACTION", float("nan")),
        "M1_HEADROOM_EXCEED_FRACTION": m1_head.get("HEADROOM_EXCEED_FRACTION", float("nan")),
        "BND_HEADROOM_EXCEED_FRACTION": bnd_head.get("HEADROOM_EXCEED_FRACTION", float("nan")),
        "M1_SH_RGB_GAIN_PSNR": m1_dc.get("SH_RGB_GAIN_PSNR", float("nan")),
        "BND_SH_RGB_GAIN_PSNR": bnd_dc.get("SH_RGB_GAIN_PSNR", float("nan")),
        "M1_FULL_PSNR": proj.get("M1_FULL_PSNR", float("nan")),
        "M1_PROJ_PSNR": proj.get("M1_PROJ_PSNR", float("nan")),
        "BND_K1_PSNR": proj.get("BND_K1_PSNR", float("nan")),
        "BND_RGB_GAP_PSNR": bnd_gap,
        "M1_PROJ_PSNR_LOSS_vs_M1": float(proj.get("M1_PROJ_PSNR", float("nan"))) - float(proj.get("M1_FULL_PSNR", float("nan"))),
        "PSNR_PROJ_minus_BND": proj.get("PSNR_PROJ_minus_BND", float("nan")),
        "PROJECTION_MSE_FRACTION": proj.get("PROJECTION_MSE_FRACTION", float("nan")),
        "projection_bnd_excess_pearson": overlap.get("pearson_delta_proj_bnd_excess", float("nan")),
        "projection_bnd_excess_spearman": overlap.get("spearman_delta_proj_bnd_excess", float("nan")),
        "projection_bnd_excess_top10_overlap": overlap.get("top10_spatial_overlap", float("nan")),
        "projection_bnd_excess_top20_overlap": overlap.get("top20_spatial_overlap", float("nan")),
        "PROJECTED_CHANGE_ENRICHMENT_IN_COMP": comp_proj.get("enrichment_ratio", float("nan")),
        "BND_EXCESS_ENRICHMENT_IN_COMP": comp_excess.get("enrichment_ratio", float("nan")),
    }
    return row


def _classification_rows(scene_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    panama = next((row for row in scene_rows if row.get("scene") == "Panama"), {})
    legal = float(panama.get("LEGAL_SH_ENERGY_FRACTION", 0.0))
    overflow = float(panama.get("OVERFLOW_SH_ENERGY_FRACTION", 0.0)) + float(panama.get("BASE_INVALID_SH_ENERGY_FRACTION", 0.0))
    projection_fraction = float(panama.get("PROJECTION_MSE_FRACTION", float("nan")))
    psnr_proj_minus_bnd = float(panama.get("PSNR_PROJ_minus_BND", float("nan")))
    bnd_r_ratio = float(panama.get("BND_over_M1_R_SH_p50", float("nan")))
    valid_recover = bool(legal >= 0.50 and bnd_r_ratio < 0.75)
    overflow_dom = bool(overflow >= 0.50 and projection_fraction == projection_fraction and projection_fraction >= 0.50)
    frozen_strong = bool(psnr_proj_minus_bnd == psnr_proj_minus_bnd and psnr_proj_minus_bnd >= 0.30 and projection_fraction <= 0.50)
    range_explains = bool(projection_fraction == projection_fraction and projection_fraction >= 0.75 and abs(psnr_proj_minus_bnd) <= 0.30)
    mixed = bool(projection_fraction == projection_fraction and 0.30 <= projection_fraction < 0.75)
    if valid_recover and frozen_strong:
        next_exp = "Panama BND-v2 bounded-base controlled-residual test"
    elif overflow_dom and range_explains:
        next_exp = "bounded object-medium recomposition diagnostic"
    elif mixed:
        next_exp = "single-candidate bounded-base controlled-valid-residual diagnostic"
    else:
        next_exp = "bounded object-medium recomposition diagnostic"
    return {
        "basis_scene": "Panama",
        "VALID_SH_RESIDUAL_WORTH_RECOVERING": valid_recover,
        "LEGACY_SH_OVERFLOW_DOMINANT": overflow_dom,
        "FROZEN_M1_BOUNDED_COUNTERFACTUAL_STRONG": frozen_strong,
        "RANGE_REMOVAL_EXPLAINS_RGB_GAP": range_explains,
        "MIXED_RANGE_AND_RECOMPOSITION": mixed,
        "NEXT_SINGLE_FACTOR_EXPERIMENT": next_exp,
        "numeric_basis": {
            "Panama_LEGAL_SH_ENERGY_FRACTION": legal,
            "Panama_OVERFLOW_PLUS_BASE_INVALID_SH_ENERGY_FRACTION": overflow,
            "Panama_BND_over_M1_R_SH_p50": bnd_r_ratio,
            "Panama_PROJECTION_MSE_FRACTION": projection_fraction,
            "Panama_PSNR_PROJ_minus_BND": psnr_proj_minus_bnd,
        },
    }


def _write_visual_index(path: Path, manifest: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# BND-SHSTRUCT Visual Compare Index", ""]
    for scene in list(SCENES) + ["ALL"]:
        scene_items = [item for item in manifest if item.get("scene") == scene and str(item.get("file_path", "")).endswith(".png")]
        if not scene_items:
            continue
        lines.extend([f"## {scene}", ""])
        for item in scene_items:
            lines.append(f"- {item.get('output_type')}: `{item.get('file_path')}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bnd_shstruct_audit_20260810"))
    parser.add_argument("--render-dir", type=Path, default=Path("renders/bnd_shstruct_audit_20260810"))
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--tile-width", type=int, default=260)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise


if __name__ == "__main__":
    main()
