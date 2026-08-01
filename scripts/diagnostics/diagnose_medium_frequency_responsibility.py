#!/usr/bin/env python
"""Audit medium/Gaussian high-frequency responsibility on frozen checkpoints.

This diagnostic is read-only. It loads an existing checkpoint, summarizes
high-pass energy in renderer branches, and evaluates counterfactual low-pass
medium maps without changing Gaussian parameters or training state.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup

from water_splatting.fields import compute_gaussian_colors


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _stats(values: torch.Tensor) -> Dict[str, float]:
    flat = values.detach().float().reshape(-1).cpu()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(flat.mean().item()),
        "p50": float(torch.quantile(flat, 0.50).item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "max": float(flat.max().item()),
    }


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _nchw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError(f"Expected HxWxC image/map, got shape={tuple(image.shape)}")
    return image.float().permute(2, 0, 1).unsqueeze(0)


def _hwc(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 4:
        value = value[0].permute(1, 2, 0)
    if value.ndim == 2:
        value = value[..., None]
    return value.contiguous()


def _highpass5(image: torch.Tensor) -> torch.Tensor:
    x = _nchw(image)
    blur = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
    return _hwc(x - blur)


def _sobel_mag(image: torch.Tensor) -> torch.Tensor:
    x = _nchw(image)
    channels = int(x.shape[1])
    kx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    kx = kx.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    sx = F.conv2d(x, kx, padding=1, groups=channels)
    sy = F.conv2d(x, ky, padding=1, groups=channels)
    return _hwc(0.5 * (sx.abs() + sy.abs()).mean(dim=1, keepdim=True))


def _pearson_corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    x = a.detach().float()
    y = b.detach().float()
    if x.ndim == 3:
        x = x.mean(dim=-1, keepdim=True)
    if y.ndim == 3:
        y = y.mean(dim=-1, keepdim=True)
    keep = mask.detach().bool().reshape(-1) & torch.isfinite(x.reshape(-1)) & torch.isfinite(y.reshape(-1))
    if int(keep.sum().item()) < 3:
        return 0.0
    xv = x.reshape(-1)[keep]
    yv = y.reshape(-1)[keep]
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    denom = xv.norm() * yv.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    value = (xv * yv).sum() / denom
    return float(value.item()) if math.isfinite(float(value.item())) else 0.0


def _masked_sum(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.expand(*mask.shape[:-1], value.shape[-1]) if mask.shape[-1] == 1 else mask[..., None]
    return (value.float().abs() * mask.float()).sum()


def _hf_energy(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> float:
    return float((_masked_sum(_highpass5(value), mask) / _masked_sum(value, mask).clamp_min(eps)).item())


def _hf_ratio(numerator: torch.Tensor, denominator: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> float:
    return float((_masked_sum(_highpass5(numerator), mask) / _masked_sum(_highpass5(denominator), mask).clamp_min(eps)).item())


def _region_masks(outputs: Dict[str, torch.Tensor], *, boundary_quantile: float) -> Dict[str, torch.Tensor]:
    accumulation = outputs["accumulation"].detach().float().clamp(0.0, 1.0)
    depth_std = outputs["depth_std_relative"].detach().float()
    h, w = accumulation.shape[:2]
    whole = torch.ones(h, w, 1, device=accumulation.device, dtype=torch.float32)
    object_region = ((accumulation >= 0.55) & (depth_std <= 0.35)).float()
    open_water = (accumulation <= 0.20).float()
    boundary_score = _sobel_mag(accumulation).detach()
    finite = boundary_score[torch.isfinite(boundary_score)]
    if finite.numel() > 0:
        threshold = torch.quantile(finite.float(), min(max(float(boundary_quantile), 0.0), 1.0))
        boundary = ((boundary_score >= threshold) & (accumulation > 0.05)).float()
    else:
        boundary = torch.zeros_like(accumulation)
    return {
        "whole": whole,
        "object": object_region,
        "open_water": open_water,
        "boundary": boundary,
    }


def _lowpass_map(value: torch.Tensor, downscale: int) -> torch.Tensor:
    if int(downscale) <= 1:
        return value
    h, w = int(value.shape[0]), int(value.shape[1])
    hc = max(1, math.ceil(h / int(downscale)))
    wc = max(1, math.ceil(w / int(downscale)))
    x = _nchw(value)
    coarse = F.interpolate(x, size=(hc, wc), mode="bilinear", align_corners=False)
    restored = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)
    return _hwc(restored).to(device=value.device, dtype=value.dtype)


def _camera_setup(model: Any, camera: Cameras) -> Tuple[torch.Tensor, torch.Tensor, float, float, int, int, torch.Tensor]:
    camera_downscale = model._get_downscale_factor()
    camera.rescale_output_resolution(1 / camera_downscale)
    R = camera.camera_to_worlds[0, :3, :3]
    T = camera.camera_to_worlds[0, :3, 3:4]
    R_edit = torch.diag(torch.tensor([1, -1, -1], device=model.device, dtype=R.dtype))
    R = R @ R_edit
    R_inv = R.T
    T_inv = -R_inv @ T
    viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv
    cx = float(camera.cx.item())
    cy = float(camera.cy.item())
    width = int(camera.width.item())
    height = int(camera.height.item())
    camera.rescale_output_resolution(camera_downscale)
    return R, viewmat, cx, cy, height, width, R_edit


def _predict_medium_with_center(
    model: Any,
    camera: Cameras,
    *,
    rotation_world_from_camera: torch.Tensor,
    height: int,
    width: int,
    cx: float,
    cy: float,
    camera_center: torch.Tensor,
) -> Any:
    scene_center, scene_scale = model._get_scene_normalization(
        dtype=rotation_world_from_camera.dtype,
        device=rotation_world_from_camera.device,
    )
    return model.medium_field(
        camera=camera,
        rotation_world_from_camera=rotation_world_from_camera,
        height=height,
        width=width,
        cx=cx,
        cy=cy,
        density_bias=model.medium_density_bias,
        mlp_type=model.config.mlp_type,
        zero_medium=model.config.zero_medium,
        context_mode=getattr(model.config, "medium_context_mode", "dir_only"),
        camera_center=camera_center,
        scene_center=scene_center,
        scene_scale=scene_scale,
        camera_context_scale=getattr(model.config, "medium_camera_context_scale", 1.0),
        camera_context_dropout=0.0,
        training=False,
        depth_context=None,
        enable_b_inf=model._b_inf_requires_head(),
        b_inf_mode=model._effective_b_inf_mode(),
        b_inf_residual_scale=getattr(model.config, "b_inf_residual_scale", 0.02),
    )


def _render_with_medium(
    model: Any,
    camera: Cameras,
    *,
    medium_rgb: torch.Tensor,
    medium_bs: torch.Tensor,
    medium_attn: torch.Tensor,
) -> Dict[str, torch.Tensor]:
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
        cx = float(camera.cx.item())
        cy = float(camera.cy.item())
        width = int(camera.width.item())
        height = int(camera.height.item())
        xys, depths, radii, conics, comp, num_tiles_hit, _ = model.underwater_rasterizer.project(
            means=model.means,
            scales=model.scales,
            quats=model.quats,
            viewmat=viewmat,
            fx=float(camera.fx.item()),
            fy=float(camera.fy.item()),
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            clip_thresh=model.config.clip_thresh,
        )
    finally:
        camera.rescale_output_resolution(camera_downscale)

    if int((radii > 0).sum().item()) == 0:
        return {"pred_image": medium_rgb, "rgb_object": torch.zeros_like(medium_rgb), "rgb_medium_total": medium_rgb}

    active_sh_degree = model._get_active_sh_degree()
    colors = compute_gaussian_colors(
        means=model.means,
        features_dc=model.features_dc,
        features_rest=model.features_rest,
        camera_position=camera.camera_to_worlds[..., :3, 3],
        sh_degree=model.config.sh_degree,
        active_sh_degree=active_sh_degree,
    )
    if model.config.rasterize_mode == "antialiased":
        opacities = torch.sigmoid(model.opacities) * comp[:, None]
    elif model.config.rasterize_mode == "classic":
        opacities = torch.sigmoid(model.opacities)
    else:
        raise ValueError(f"Unknown rasterize_mode={model.config.rasterize_mode}")

    render = model.underwater_rasterizer.rasterize(
        xys=xys,
        xys_grad_abs=torch.zeros_like(xys),
        depths=depths,
        radii=radii,
        conics=conics,
        num_tiles_hit=num_tiles_hit,
        colors=colors,
        opacities=opacities,
        medium_rgb=medium_rgb,
        medium_bs=medium_bs,
        medium_attn=medium_attn,
        height=height,
        width=width,
        background=medium_rgb,
        step=int(model.step),
    )
    tail_weight_last = render.final_transmittance * torch.exp(-medium_bs * render.last_depth)
    rgb_medium_finite = render.rgb_medium - tail_weight_last * medium_rgb
    if model._effective_b_inf_mode() == "implicit":
        pred = render.rgb
        rgb_tail = render.rgb_medium - rgb_medium_finite
    else:
        rgb_tail = tail_weight_last * medium_rgb
        pred = render.rgb_object + rgb_medium_finite + rgb_tail
    return {
        "pred_image": pred,
        "rgb_object": render.rgb_object,
        "rgb_medium": render.rgb_medium,
        "rgb_medium_finite": rgb_medium_finite,
        "rgb_medium_total": rgb_medium_finite + rgb_tail,
        "rgb_tail": rgb_tail,
    }


def _image_metrics(model: Any, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    pred = pred.detach().float().clamp(0.0, 1.0)
    gt = gt.detach().float().clamp(0.0, 1.0)
    pred_nchw = pred.permute(2, 0, 1).unsqueeze(0)
    gt_nchw = gt.permute(2, 0, 1).unsqueeze(0)
    return {
        "psnr": float(model.psnr(gt_nchw, pred_nchw).item()),
        "ssim": float(model.ssim(gt_nchw, pred_nchw)),
        "lpips": float(model.lpips(gt_nchw, pred_nchw).item()),
    }


def _summarize_image(
    outputs: Dict[str, torch.Tensor],
    gt: torch.Tensor,
    *,
    boundary_quantile: float,
) -> Dict[str, Any]:
    masks = _region_masks(outputs, boundary_quantile=boundary_quantile)
    pred = outputs["pred_image"].detach().float()
    residual_hf = (_highpass5(pred) - _highpass5(gt)).abs()
    components = {
        "medium_rgb": outputs["medium_rgb"].detach().float(),
        "medium_bs": outputs["medium_bs"].detach().float(),
        "medium_attn": outputs["medium_attn"].detach().float(),
        "medium_render": outputs["rgb_medium_total"].detach().float(),
        "gaussian_render": outputs["rgb_object"].detach().float(),
        "final_rgb": pred,
        "gt": gt.detach().float(),
    }
    regions: Dict[str, Any] = {}
    for name, mask in masks.items():
        medium_hp = _highpass5(components["medium_render"]).abs()
        gaussian_hp = _highpass5(components["gaussian_render"]).abs()
        regions[name] = {
            "coverage": float(mask.float().mean().item()),
            "medium_rgb_hf_energy": _hf_energy(components["medium_rgb"], mask),
            "medium_bs_hf_energy": _hf_energy(components["medium_bs"], mask),
            "medium_attn_hf_energy": _hf_energy(components["medium_attn"], mask),
            "medium_render_hf_ratio": _hf_ratio(components["medium_render"], pred, mask),
            "gaussian_render_hf_ratio": _hf_ratio(components["gaussian_render"], pred, mask),
            "final_hf_energy": _hf_energy(pred, mask),
            "gt_hf_energy": _hf_energy(gt, mask),
            "final_hf_residual_energy": _hf_energy(pred - gt, mask),
            "medium_residual_correlation": _pearson_corr(residual_hf, medium_hp, mask),
            "gaussian_residual_correlation": _pearson_corr(residual_hf, gaussian_hp, mask),
        }
    return {"regions": regions}


def _aggregate_region_stats(images: List[Dict[str, Any]]) -> Dict[str, Any]:
    region_names = sorted({region for image in images for region in image["regions"].keys()})
    aggregate: Dict[str, Any] = {}
    for region in region_names:
        keys = sorted({key for image in images for key in image["regions"].get(region, {}).keys()})
        aggregate[region] = {
            key: _stats(torch.tensor([image["regions"][region][key] for image in images if region in image["regions"]]))
            for key in keys
        }
    return aggregate


def _aggregate_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({key for row in rows for key in row.keys()})
    return {key: _mean(row[key] for row in rows if key in row) for key in keys}


def _iter_cameras(pipeline: Any, split: str, max_images: int, device: torch.device):
    limit = max_images if max_images > 0 else 10**9
    if split == "eval":
        for idx, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if idx >= limit:
                break
            yield idx, camera.to(device) if hasattr(camera, "to") else camera, batch
        return
    dataset = pipeline.datamanager.train_dataset
    count = min(len(dataset.cameras), limit)
    for idx in range(count):
        camera = dataset.cameras[idx : idx + 1]
        batch = {"image": dataset[idx]["image"]}
        yield idx, camera.to(device) if hasattr(camera, "to") else camera, batch


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    def _update_config(config: Any) -> Any:
        if args.load_step is not None:
            config.load_step = int(args.load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(args.load_config, update_config_callback=_update_config)
    pipeline.eval()
    model = pipeline.model
    device = model.device
    model.step = int(step)

    images: List[Dict[str, Any]] = []
    metrics_by_variant: Dict[str, List[Dict[str, float]]] = {"F0": []}
    counterfactual_variants: List[Tuple[str, int, bool, bool]] = []
    for factor in args.counterfactual_downscales:
        counterfactual_variants.append((f"F{factor}", int(factor), True, True))
    if int(args.branch_downscale) > 1:
        counterfactual_variants.append(("FA", int(args.branch_downscale), True, False))
        counterfactual_variants.append(("FC", int(args.branch_downscale), False, True))
    for name, _factor, _rgb, _coeff in counterfactual_variants:
        metrics_by_variant[name] = []

    train_centers = None
    if hasattr(pipeline.datamanager, "train_dataset"):
        train_centers = pipeline.datamanager.train_dataset.cameras.camera_to_worlds[:, :3, 3].to(device)
    scene_mean_center = train_centers.mean(dim=0) if train_centers is not None and train_centers.numel() else None

    with torch.no_grad():
        for image_idx, camera, batch in _iter_cameras(pipeline, args.split, int(args.max_images), device):
            outputs = model.get_outputs_for_camera(camera=camera)
            gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
            pred = outputs["pred_image"].detach().float()
            image_summary = {
                "image_index": int(image_idx),
                **_summarize_image(outputs, gt, boundary_quantile=float(args.boundary_quantile)),
                "counterfactual": {},
                "camera_context": {},
            }
            metrics_by_variant["F0"].append(_image_metrics(model, pred, gt))

            medium_rgb = outputs["medium_rgb"].detach().float()
            medium_bs = outputs["medium_bs"].detach().float()
            medium_attn = outputs["medium_attn"].detach().float()
            for name, factor, do_rgb, do_coeff in counterfactual_variants:
                cf_rgb = _lowpass_map(medium_rgb, factor) if do_rgb else medium_rgb
                cf_bs = _lowpass_map(medium_bs, factor) if do_coeff else medium_bs
                cf_attn = _lowpass_map(medium_attn, factor) if do_coeff else medium_attn
                cf_outputs = _render_with_medium(
                    model,
                    camera,
                    medium_rgb=cf_rgb,
                    medium_bs=cf_bs,
                    medium_attn=cf_attn,
                )
                metrics = _image_metrics(model, cf_outputs["pred_image"], gt)
                metrics_by_variant[name].append(metrics)
                image_summary["counterfactual"][name] = {
                    **metrics,
                    "rgb_mean_abs_delta": float((cf_outputs["pred_image"] - pred).abs().mean().item()),
                    "medium_render_hf_ratio": _hf_ratio(cf_outputs["rgb_medium_total"], cf_outputs["pred_image"], torch.ones_like(outputs["accumulation"])),
                }

            if "camera" in getattr(model.config, "medium_context_mode", "dir_only"):
                R, _viewmat, cx, cy, height, width, _redit = _camera_setup(model, camera)
                current_center = camera.camera_to_worlds[0, :3, 3].detach()
                centers: Dict[str, torch.Tensor] = {"current": current_center}
                if train_centers is not None and train_centers.numel():
                    nearest_idx = torch.argmin(torch.linalg.norm(train_centers - current_center[None, :], dim=-1))
                    centers["nearest_train"] = train_centers[int(nearest_idx.item())]
                if scene_mean_center is not None:
                    centers["scene_mean"] = scene_mean_center
                _, scene_scale = model._get_scene_normalization(dtype=current_center.dtype, device=current_center.device)
                perturb = torch.tensor([float(args.camera_perturb_scale), 0.0, 0.0], device=device, dtype=current_center.dtype)
                centers["perturbed"] = current_center + perturb * scene_scale
                base_medium = _predict_medium_with_center(
                    model,
                    camera,
                    rotation_world_from_camera=R,
                    height=height,
                    width=width,
                    cx=cx,
                    cy=cy,
                    camera_center=current_center,
                )
                for label, center in centers.items():
                    medium = _predict_medium_with_center(
                        model,
                        camera,
                        rotation_world_from_camera=R,
                        height=height,
                        width=width,
                        cx=cx,
                        cy=cy,
                        camera_center=center,
                    )
                    delta_rgb = (medium.rgb - base_medium.rgb).detach().float()
                    delta_bs = (medium.bs - base_medium.bs).detach().float()
                    delta_attn = (medium.attn - base_medium.attn).detach().float()
                    image_summary["camera_context"][label] = {
                        "rgb_delta": _stats(delta_rgb.abs()),
                        "bs_delta": _stats(delta_bs.abs()),
                        "attn_delta": _stats(delta_attn.abs()),
                        "rgb_delta_hf_energy": _hf_energy(delta_rgb, torch.ones_like(outputs["accumulation"])),
                        "bs_delta_hf_energy": _hf_energy(delta_bs, torch.ones_like(outputs["accumulation"])),
                        "attn_delta_hf_energy": _hf_energy(delta_attn, torch.ones_like(outputs["accumulation"])),
                    }

            images.append(image_summary)

    aggregate_metrics = {name: _aggregate_metrics(rows) for name, rows in metrics_by_variant.items()}
    f0 = aggregate_metrics.get("F0", {})
    for name, metrics in aggregate_metrics.items():
        metrics["dpsnr_vs_F0"] = float(metrics.get("psnr", 0.0) - f0.get("psnr", 0.0))
        metrics["dssim_vs_F0"] = float(metrics.get("ssim", 0.0) - f0.get("ssim", 0.0))
        metrics["dlpips_vs_F0"] = float(metrics.get("lpips", 0.0) - f0.get("lpips", 0.0))

    aggregate = {
        "regions": _aggregate_region_stats(images),
        "counterfactual_metrics": aggregate_metrics,
    }

    gate = {
        "japanesegradens_medium_hf_ratio_ge_10pct": False,
        "medium_corr_ge_0p15": False,
        "counterfactual_lpips_improves_ge_0p001": False,
        "counterfactual_psnr_decline_le_0p05": False,
        "phase0_passes_minimum": False,
    }
    whole = aggregate["regions"].get("whole", {})
    medium_hf_mean = whole.get("medium_render_hf_ratio", {}).get("mean", 0.0)
    medium_corr_mean = whole.get("medium_residual_correlation", {}).get("mean", 0.0)
    best_cf = None
    for name in ("F4", "F8"):
        if name in aggregate_metrics:
            row = aggregate_metrics[name]
            if best_cf is None or row.get("dlpips_vs_F0", 0.0) < best_cf[1].get("dlpips_vs_F0", 0.0):
                best_cf = (name, row)
    if best_cf is not None:
        gate["best_counterfactual"] = best_cf[0]
        gate["best_counterfactual_dlpips"] = float(best_cf[1].get("dlpips_vs_F0", 0.0))
        gate["best_counterfactual_dpsnr"] = float(best_cf[1].get("dpsnr_vs_F0", 0.0))
        gate["counterfactual_lpips_improves_ge_0p001"] = best_cf[1].get("dlpips_vs_F0", 0.0) <= -0.001
        gate["counterfactual_psnr_decline_le_0p05"] = best_cf[1].get("dpsnr_vs_F0", 0.0) >= -0.05
    gate["japanesegradens_medium_hf_ratio_ge_10pct"] = medium_hf_mean >= 0.10
    gate["medium_corr_ge_0p15"] = medium_corr_mean >= 0.15
    gate["phase0_passes_minimum"] = bool(sum(bool(v) for k, v in gate.items() if k != "phase0_passes_minimum" and isinstance(v, bool)) >= 2)

    return {
        "scene_name": args.scene_name,
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "split": args.split,
        "max_images": int(args.max_images),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "aggregate": aggregate,
        "gate": gate,
        "images": images,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--scene-name", type=str, default="")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mfrs_diagnostics"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--counterfactual-downscales", type=int, nargs="*", default=[2, 4, 8])
    parser.add_argument("--branch-downscale", type=int, default=4)
    parser.add_argument("--boundary-quantile", type=float, default=0.85)
    parser.add_argument("--camera-perturb-scale", type=float, default=0.01)
    args = parser.parse_args()

    result = diagnose(args)
    output_json = args.output_json
    if output_json is None:
        scene = args.scene_name or "scene"
        output_json = args.output_dir / f"{scene}_mfrs_frequency_responsibility.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    summary = {
        "scene_name": result["scene_name"],
        "whole_medium_render_hf_ratio": result["aggregate"]["regions"]["whole"]["medium_render_hf_ratio"]["mean"],
        "whole_gaussian_render_hf_ratio": result["aggregate"]["regions"]["whole"]["gaussian_render_hf_ratio"]["mean"],
        "whole_medium_residual_correlation": result["aggregate"]["regions"]["whole"]["medium_residual_correlation"]["mean"],
        "whole_gaussian_residual_correlation": result["aggregate"]["regions"]["whole"]["gaussian_residual_correlation"]["mean"],
        "counterfactual_metrics": result["aggregate"]["counterfactual_metrics"],
        "gate": result["gate"],
    }
    print(json.dumps(summary, indent=2))
    print(f"saved={output_json}")


if __name__ == "__main__":
    main()
