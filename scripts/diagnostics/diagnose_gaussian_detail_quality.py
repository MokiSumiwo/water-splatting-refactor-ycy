#!/usr/bin/env python
"""Audit Gaussian detail quality for frozen WaterSplatting checkpoints.

This diagnostic is read-only. It evaluates a checkpoint on fixed eval/inference
views, samples detached high-frequency residual evidence at projected Gaussian
centers, and records capacity, scale, visibility, geometry, detail, appearance,
and correlation summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from nerfstudio.utils.eval_utils import eval_setup

from water_splatting.cleanup import sample_pixel_map_at_gaussians
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
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0}
    return {
        "mean": float(flat.mean().item()),
        "p50": float(torch.quantile(flat, 0.50).item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "max": float(flat.max().item()),
        "min": float(flat.min().item()),
    }


def _mean_dict(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.asarray([row[key] for row in rows], dtype=np.float64).mean()) for key in keys}


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1).cpu()
    b = b.detach().float().reshape(-1).cpu()
    keep = torch.isfinite(a) & torch.isfinite(b)
    a = a[keep]
    b = b[keep]
    if a.numel() < 4:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a * a).mean() * (b * b).mean()).clamp_min(1e-12)
    return float(((a * b).mean() / denom).item())


def _sh2rgb(sh: torch.Tensor) -> torch.Tensor:
    return sh * 0.28209479177387814 + 0.5


def _detail_maps(pred_img: torch.Tensor, gt_img: torch.Tensor, highpass_weight: float) -> Dict[str, torch.Tensor]:
    pred = pred_img.detach().float().clamp(0.0, 1.0)
    gt = gt_img.detach().float().clamp(0.0, 1.0)
    pred_chw = pred.permute(2, 0, 1).unsqueeze(0)
    gt_chw = gt.permute(2, 0, 1).unsqueeze(0)
    channels = pred_chw.shape[1]

    kx = pred_chw.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = pred_chw.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    kx = kx.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    pred_gx = F.conv2d(pred_chw, kx, padding=1, groups=channels)
    pred_gy = F.conv2d(pred_chw, ky, padding=1, groups=channels)
    gt_gx = F.conv2d(gt_chw, kx, padding=1, groups=channels)
    gt_gy = F.conv2d(gt_chw, ky, padding=1, groups=channels)
    sobel = 0.5 * ((pred_gx - gt_gx).abs() + (pred_gy - gt_gy).abs()).mean(dim=1)

    pred_blur = F.avg_pool2d(F.pad(pred_chw, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
    gt_blur = F.avg_pool2d(F.pad(gt_chw, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
    highpass = ((pred_chw - pred_blur) - (gt_chw - gt_blur)).abs().mean(dim=1)
    detail = sobel + float(highpass_weight) * highpass
    return {
        "sobel": sobel.squeeze(0)[..., None],
        "highpass": highpass.squeeze(0)[..., None],
        "detail": detail.squeeze(0)[..., None],
    }


def _far_mask(outputs: Dict[str, torch.Tensor], fraction: float) -> torch.Tensor:
    depth = outputs["depth"].detach().float()
    accumulation = outputs["accumulation"].detach().float()
    valid = torch.isfinite(depth) & (depth > 0.0) & (accumulation > 0.01)
    if not bool(valid.any().item()):
        return torch.zeros_like(depth, dtype=torch.bool)
    q = torch.quantile(depth[valid], max(0.0, min(1.0, 1.0 - float(fraction))))
    return valid & (depth >= q)


def _object_support(
    outputs: Dict[str, torch.Tensor],
    *,
    kappa: float,
    accum_mid: float,
    accum_temp: float,
) -> torch.Tensor:
    accumulation = outputs["accumulation"].detach().float()
    depth_std = outputs["depth_std_relative"].detach().float()
    q_hit = outputs.get("hit_confidence", torch.ones_like(accumulation)).detach().float().clamp(0.0, 1.0)
    q_conc = torch.exp(-depth_std / max(float(kappa), 1e-6)).clamp(0.0, 1.0)
    q_acc = torch.sigmoid((accumulation - float(accum_mid)) / max(float(accum_temp), 1e-6)).clamp(0.0, 1.0)
    return (q_hit * q_conc * q_acc).clamp(0.0, 1.0)


def _parse_refinement_log(log_dir: Optional[Path]) -> Dict[str, Any]:
    if log_dir is None:
        return {"available": False, "reason": "no --log-dir provided"}
    paths = []
    if log_dir.is_file():
        paths = [log_dir]
    elif log_dir.is_dir():
        paths = sorted(log_dir.glob("*.log")) + sorted(log_dir.glob("*.txt"))
    if not paths:
        return {"available": False, "reason": f"no log files under {log_dir}"}

    split_count = 0
    duplicate_count = 0
    cull_count = 0
    split_events = 0
    duplicate_events = 0
    cull_events = 0
    split_re = re.compile(r"Splitting .*: (\d+)/")
    dup_re = re.compile(r"Duplicating .*: (\d+)/")
    cull_re = re.compile(r"Culled (\d+) gaussians")
    for path in paths:
        try:
            text = path.read_text(encoding="utf8", errors="ignore")
        except Exception:
            continue
        for match in split_re.finditer(text):
            split_count += int(match.group(1))
            split_events += 1
        for match in dup_re.finditer(text):
            duplicate_count += int(match.group(1))
            duplicate_events += 1
        for match in cull_re.finditer(text):
            cull_count += int(match.group(1))
            cull_events += 1
    return {
        "available": True,
        "paths": [str(path) for path in paths],
        "split_count": split_count,
        "split_events": split_events,
        "duplicate_count": duplicate_count,
        "duplicate_events": duplicate_events,
        "cull_count": cull_count,
        "cull_events": cull_events,
    }


def _maybe_save_image(path: Path, image: torch.Tensor) -> None:
    try:
        from torchvision.utils import save_image
    except Exception:
        return
    img = image.detach().float().cpu()
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == 1:
        img = img.expand(*img.shape[:2], 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(img.clamp(0.0, 1.0).permute(2, 0, 1), path)


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    def _update_config(config: Any) -> Any:
        if args.load_step is not None:
            config.load_step = int(args.load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(
        args.load_config,
        eval_num_rays_per_chunk=None,
        test_mode=args.test_mode,
        update_config_callback=_update_config,
    )
    pipeline.eval()
    model = pipeline.model
    num_points = int(model.num_points)
    device = model.device

    visibility_counts = torch.zeros(num_points, device=device, dtype=torch.float32)
    radius_values: List[torch.Tensor] = []
    depth_values: List[torch.Tensor] = []
    sampled_detail_values: List[torch.Tensor] = []
    sampled_safe_detail_values: List[torch.Tensor] = []
    sampled_q_obj_values: List[torch.Tensor] = []
    sampled_hit_values: List[torch.Tensor] = []
    sampled_scale_values: List[torch.Tensor] = []
    sampled_radius_values: List[torch.Tensor] = []
    sampled_depth_values: List[torch.Tensor] = []
    sampled_sh_delta_values: List[torch.Tensor] = []

    view_rows: List[Dict[str, Any]] = []
    max_images = args.max_images if args.max_images > 0 else 10**9
    with torch.no_grad():
        for image_idx, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= max_images:
                break
            outputs = model.get_outputs_for_camera(camera=camera)
            gt_img = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
            pred_img = outputs["pred_image"].detach().float().clamp(0.0, 1.0)
            if "mask" in batch:
                mask = model._downscale_if_required(batch["mask"]).to(device=model.device).float().clamp(0.0, 1.0)
                gt_img = gt_img * mask
                pred_img = pred_img * mask

            detail_maps = _detail_maps(pred_img, gt_img, args.highpass_weight)
            q_obj = _object_support(
                outputs,
                kappa=args.kappa,
                accum_mid=args.obj_accum_mid,
                accum_temp=args.obj_accum_temp,
            )
            safe_detail = q_obj * detail_maps["detail"]
            full_rgb_residual = (pred_img - gt_img).abs().mean(dim=-1, keepdim=True)
            far = _far_mask(outputs, args.far_fraction)

            radii = model.radii.detach().reshape(-1).float()
            visible = radii > 0
            visibility_counts[visible] += 1.0
            radius_values.append(radii[visible].detach().cpu())
            depths = model.depths.detach().reshape(-1).float()
            depth_values.append(depths[visible].detach().cpu())

            height, width = int(pred_img.shape[0]), int(pred_img.shape[1])
            sampled_detail = sample_pixel_map_at_gaussians(
                detail_maps["detail"], model.xys.detach(), model.radii.detach(), height, width
            ).float()
            sampled_safe = sample_pixel_map_at_gaussians(
                safe_detail, model.xys.detach(), model.radii.detach(), height, width
            ).float()
            sampled_q = sample_pixel_map_at_gaussians(q_obj, model.xys.detach(), model.radii.detach(), height, width).float()
            sampled_hit = sample_pixel_map_at_gaussians(
                outputs.get("hit_confidence", torch.ones_like(outputs["accumulation"])).detach(),
                model.xys.detach(),
                model.radii.detach(),
                height,
                width,
            ).float()
            scale_max = model.scales.detach().exp().max(dim=-1).values.float()
            sampled_detail_values.append(sampled_detail[visible].detach().cpu())
            sampled_safe_detail_values.append(sampled_safe[visible].detach().cpu())
            sampled_q_obj_values.append(sampled_q[visible].detach().cpu())
            sampled_hit_values.append(sampled_hit[visible].detach().cpu())
            sampled_scale_values.append(scale_max[visible].detach().cpu())
            sampled_radius_values.append(radii[visible].detach().cpu())
            sampled_depth_values.append(depths[visible].detach().cpu())

            active_sh_degree = int(getattr(model, "last_active_sh_degree", 0))
            if active_sh_degree > 0 and model.config.sh_degree > 0:
                full_rgb = compute_gaussian_colors(
                    means=model.means,
                    features_dc=model.features_dc,
                    features_rest=model.features_rest,
                    camera_position=camera.camera_to_worlds[..., :3, 3],
                    sh_degree=model.config.sh_degree,
                    active_sh_degree=active_sh_degree,
                ).detach()
                dc_rgb = _sh2rgb(model.features_dc.detach())
                sh_delta = (full_rgb - dc_rgb).norm(dim=-1)
                sampled_sh_delta_values.append(sh_delta[visible].detach().cpu())

            view_rows.append(
                {
                    "image_index": int(image_idx),
                    "visible_gaussians": int(visible.sum().item()),
                    "mean_visibility_fraction_this_view": float(visible.float().mean().item()),
                    "full_rgb_residual": _stats(full_rgb_residual),
                    "sobel_detail_residual": _stats(detail_maps["sobel"]),
                    "highpass_detail_residual": _stats(detail_maps["highpass"]),
                    "detail_residual": _stats(detail_maps["detail"]),
                    "object_safe_detail_residual": _stats(safe_detail),
                    "q_object": _stats(q_obj),
                    "hit_confidence": _stats(outputs.get("hit_confidence", torch.zeros_like(outputs["accumulation"]))),
                    "depth_std_relative": _stats(outputs["depth_std_relative"]),
                    "far_accumulation": _stats(outputs["accumulation"][far.squeeze(-1)] if far.any() else outputs["accumulation"].new_zeros(0)),
                    "active_sh_degree": active_sh_degree,
                }
            )

            if args.save_images and image_idx < args.max_saved_images:
                out_dir = args.output_dir / "images"
                _maybe_save_image(out_dir / f"view_{image_idx:04d}_pred.png", pred_img)
                _maybe_save_image(out_dir / f"view_{image_idx:04d}_detail.png", detail_maps["detail"] / args.detail_vis_scale)
                _maybe_save_image(out_dir / f"view_{image_idx:04d}_object_safe_detail.png", safe_detail / args.detail_vis_scale)

    scale_axes = model.scales.detach().exp().float().cpu()
    scale_max = scale_axes.max(dim=-1).values
    scale_min = scale_axes.min(dim=-1).values.clamp_min(1e-12)
    axis_ratio = scale_max / scale_min
    visibility_fraction = visibility_counts.detach().cpu() / max(len(view_rows), 1)
    low_visibility_ratio = float((visibility_counts.detach().cpu() <= max(1, math.floor(max(len(view_rows), 1) * 0.25))).float().mean().item())

    def _cat(items: List[torch.Tensor]) -> torch.Tensor:
        if not items:
            return torch.zeros(0)
        return torch.cat([item.reshape(-1).cpu() for item in items], dim=0)

    detail_samples = _cat(sampled_detail_values)
    safe_detail_samples = _cat(sampled_safe_detail_values)
    result = {
        "experiment": "gaussian_detail_quality",
        "scene_name": args.scene_name,
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "requested_load_step": args.load_step,
        "loaded_step": int(step),
        "test_mode": args.test_mode,
        "max_images": args.max_images,
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "capacity": {
            "total_gaussians": num_points,
            "refinement_log": _parse_refinement_log(args.log_dir),
        },
        "scale": {
            "world_scale_max_axis": _stats(scale_max),
            "world_scale_min_axis": _stats(scale_min),
            "max_axis_over_min_axis": _stats(axis_ratio),
            "projected_radius_visible": _stats(_cat(radius_values)),
        },
        "visibility": {
            "views_evaluated": len(view_rows),
            "visible_gaussian_count_per_view": _stats(torch.tensor([row["visible_gaussians"] for row in view_rows])),
            "mean_visibility_fraction": float(visibility_fraction.mean().item()) if visibility_fraction.numel() else 0.0,
            "low_visibility_gaussian_ratio": low_visibility_ratio,
        },
        "geometry": {
            "projected_depth_visible": _stats(_cat(depth_values)),
            "depth_std_relative": _mean_dict(row["depth_std_relative"] for row in view_rows),
            "hit_confidence": _mean_dict(row["hit_confidence"] for row in view_rows),
            "far_accumulation": _mean_dict(row["far_accumulation"] for row in view_rows),
        },
        "detail": {
            "sobel_detail_residual": _mean_dict(row["sobel_detail_residual"] for row in view_rows),
            "highpass_detail_residual": _mean_dict(row["highpass_detail_residual"] for row in view_rows),
            "detail_residual": _mean_dict(row["detail_residual"] for row in view_rows),
            "object_safe_detail_residual": _mean_dict(row["object_safe_detail_residual"] for row in view_rows),
            "sampled_detail_at_visible_gaussians": _stats(detail_samples),
            "sampled_object_safe_detail_at_visible_gaussians": _stats(safe_detail_samples),
        },
        "appearance": {
            "full_rgb_residual": _mean_dict(row["full_rgb_residual"] for row in view_rows),
            "features_dc_norm": _stats(model.features_dc.detach().reshape(num_points, -1).norm(dim=-1)),
            "features_rest_norm": _stats(model.features_rest.detach().reshape(num_points, -1).norm(dim=-1)),
            "full_sh_minus_dc_rgb_norm_visible": _stats(_cat(sampled_sh_delta_values)),
            "dc_only_rgb_proxy_status": "feature-level proxy only; renderer conics/tiles are not exposed after get_outputs_for_camera",
        },
        "correlations": {
            "detail_vs_world_scale": _corr(detail_samples, _cat(sampled_scale_values)),
            "detail_vs_projected_radius": _corr(detail_samples, _cat(sampled_radius_values)),
            "detail_vs_depth": _corr(detail_samples, _cat(sampled_depth_values)),
            "detail_vs_hit_confidence": _corr(detail_samples, _cat(sampled_hit_values)),
            "object_safe_detail_vs_world_scale": _corr(safe_detail_samples, _cat(sampled_scale_values)),
            "object_safe_detail_vs_projected_radius": _corr(safe_detail_samples, _cat(sampled_radius_values)),
            "object_safe_detail_vs_depth": _corr(safe_detail_samples, _cat(sampled_depth_values)),
            "object_safe_detail_vs_hit_confidence": _corr(safe_detail_samples, _cat(sampled_hit_values)),
            "object_safe_detail_vs_current_densification_gradient": {
                "available": False,
                "reason": "frozen eval has no accumulated xys_grad_norm refinement buffer",
            },
        },
        "views": view_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "gaussian_detail_quality.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(
        json.dumps(
            {
                "scene_name": result["scene_name"],
                "loaded_step": result["loaded_step"],
                "total_gaussians": result["capacity"]["total_gaussians"],
                "detail_mean": result["detail"]["detail_residual"].get("mean", 0.0),
                "object_safe_detail_mean": result["detail"]["object_safe_detail_residual"].get("mean", 0.0),
                "output_json": str(output_json),
            },
            indent=2,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--test-mode", choices=["test", "val", "inference"], default="inference")
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--scene-name", type=str, default="unknown")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--kappa", type=float, default=0.25)
    parser.add_argument("--obj-accum-mid", type=float, default=0.35)
    parser.add_argument("--obj-accum-temp", type=float, default=0.08)
    parser.add_argument("--highpass-weight", type=float, default=0.35)
    parser.add_argument("--far-fraction", type=float, default=0.25)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--max-saved-images", type=int, default=2)
    parser.add_argument("--detail-vis-scale", type=float, default=0.25)
    args = parser.parse_args()
    diagnose(args)


if __name__ == "__main__":
    main()
