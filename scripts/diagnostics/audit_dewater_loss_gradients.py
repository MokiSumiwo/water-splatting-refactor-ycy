#!/usr/bin/env python
"""No-update gradient audits for SeaFree-inspired dewatering factors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from nerfstudio.utils.eval_utils import eval_setup
from torch import Tensor

from water_splatting.losses import effective_background_mask, reconstruction_loss


DEFAULT_D010_13K_CONFIG = (
    "outputs/dewater_direct_d010_curasao_seed42_step10000_to_13000/water-splatting/"
    "dewater_direct_d010_curasao_seed42_step10000_to_13000_"
    "20260807_dewater_direct_optical_depth_d010_g0p10/config.yml"
)
DEFAULT_MASK_DIR = "common_masks/dewater_curasao_m1_step10000_train_background_water_20260807"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _setup_pipeline(config_path: Path, step: int, test_mode: str) -> Tuple[Any, Any, Path, int]:
    def _update_config(config: Any) -> Any:
        config.load_step = int(step)
        return config

    return eval_setup(config_path, eval_num_rays_per_chunk=None, test_mode=test_mode, update_config_callback=_update_config)


def _to_device_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _camera_id(camera: Any, fallback: int) -> int:
    if camera.metadata is not None and "cam_idx" in camera.metadata:
        value = camera.metadata["cam_idx"]
        if torch.is_tensor(value):
            return int(value.detach().cpu().reshape(-1)[0].item())
        return int(value)
    return int(fallback)


def _load_region_mask(mask_dir: Path, camera_id: int, shape: Tuple[int, int], key: str) -> Optional[Tensor]:
    path = mask_dir / f"view_{int(camera_id):04d}_regions.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or key not in payload:
        return None
    mask = payload[key].detach().float()
    if mask.ndim == 2:
        mask = mask[..., None]
    if mask.shape[:2] != shape:
        mask = F.interpolate(mask.permute(2, 0, 1)[None], size=shape, mode="nearest")[0].permute(1, 2, 0)
    return mask.clamp(0.0, 1.0)


def _grad_norm(grads: Sequence[Optional[Tensor]]) -> float:
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().float().square().sum().item())
    return math.sqrt(total)


def _safe_grad(loss: Tensor, params: Sequence[Tensor]) -> Tuple[Optional[Tensor], ...]:
    active = [param for param in params if bool(getattr(param, "requires_grad", False))]
    if not active or not bool(getattr(loss, "requires_grad", False)):
        return tuple(None for _ in params)
    grads = torch.autograd.grad(loss, active, retain_graph=True, create_graph=False, allow_unused=True)
    grad_iter = iter(grads)
    return tuple(next(grad_iter) if bool(getattr(param, "requires_grad", False)) else None for param in params)


def _stats(values: Tensor) -> Dict[str, float | int]:
    flat = values.detach().float().reshape(-1).cpu()
    flat = flat[torch.isfinite(flat)]
    out: Dict[str, float | int] = {
        "count": int(flat.numel()),
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "min": float(flat.min().item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
    }
    for q in (0.10, 0.50, 0.90, 0.95, 0.99):
        if flat.numel():
            rank = max(1, min(int(flat.numel()), int(math.ceil(float(q) * float(flat.numel())))))
            out[f"p{int(round(q * 100)):02d}"] = float(flat.kthvalue(rank).values.item())
        else:
            out[f"p{int(round(q * 100)):02d}"] = 0.0
    return out


def _mask_png(path: Path, mask: Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask.ndim == 3:
        mask = mask[..., :1]
    rgb = mask.detach().float().cpu().clamp(0.0, 1.0).expand(-1, -1, 3)
    arr = (rgb.numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _tile(path: Path, label: str, width: int) -> Image.Image:
    with Image.open(path) as src:
        image = src.convert("RGB")
    if image.width > width:
        height = max(1, int(round(image.height * width / image.width)))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    pad = 24
    canvas = Image.new("RGB", (image.width, image.height + pad), "white")
    canvas.paste(image, (0, pad))
    ImageDraw.Draw(canvas).text((4, 5), label, fill="black")
    return canvas


def _write_sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Path]]], width: int) -> None:
    if not rows:
        return
    rendered = []
    for row in rows:
        tiles = [_tile(tile_path, label, width) for label, tile_path in row]
        image = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)), "white")
        x = 0
        for tile in tiles:
            image.paste(tile, (x, 0))
            x += tile.width
        rendered.append(image)
    sheet = Image.new("RGB", (max(row.width for row in rendered), sum(row.height for row in rendered)), "white")
    y = 0
    for row in rendered:
        sheet.paste(row, (0, y))
        y += row.height
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _losses_and_norms(
    *,
    model: Any,
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    mask_dir: Optional[Path],
    camera_id: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    gt_img = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
    pred_img = outputs["pred_image"]
    rgb_loss = reconstruction_loss(
        gt_img=gt_img,
        pred_img=pred_img,
        main_loss=model.config.main_loss,
        ssim_loss=model.config.ssim_loss,
        ssim_lambda=model.config.ssim_lambda,
        ssim_metric=model.ssim,
    )

    colors = outputs["gaussian_view_rgb"]
    visible = outputs["gaussian_visible_mask"].detach().to(device=colors.device, dtype=colors.dtype).reshape(-1, 1)
    bound_penalty = F.relu(colors - 1.0).square() + F.relu(-colors).square()
    bound_loss = (visible * bound_penalty).sum() / visible.sum().clamp_min(1e-6)

    fg_mask = (outputs["accumulation"].detach() > float(args.foreground_accumulation_threshold)).to(pred_img)
    fg_weight = 1.0 / (pred_img.detach() + float(args.foreground_weight_epsilon))
    if args.foreground_weight_cap > 0.0:
        fg_weight = fg_weight.clamp_max(float(args.foreground_weight_cap))
    faw_loss = (fg_mask * fg_weight * torch.abs(pred_img - gt_img)).sum() / fg_mask.sum().clamp_min(1e-6)

    bg_loss = torch.zeros((), device=pred_img.device)
    bgi_total_loss = torch.zeros((), device=pred_img.device)
    bgi_finite_loss = torch.zeros((), device=pred_img.device)
    bg_mask_info: Dict[str, Any] = {"available": False}
    if mask_dir is not None:
        water_mask = _load_region_mask(mask_dir, camera_id, tuple(pred_img.shape[:2]), args.background_mask_key)
        boundary_mask = _load_region_mask(mask_dir, camera_id, tuple(pred_img.shape[:2]), "boundary") if args.exclude_boundary else None
        if water_mask is not None:
            water_mask = water_mask.to(device=pred_img.device, dtype=pred_img.dtype)
            if boundary_mask is not None:
                boundary_mask = boundary_mask.to(device=pred_img.device, dtype=pred_img.dtype)
            bg_mask = effective_background_mask(
                water_mask=water_mask,
                boundary_mask=boundary_mask,
                hit_confidence=outputs.get("hit_confidence"),
                hit_threshold=float(args.hit_exclusion_threshold),
            ).detach()
            bg_weight = 1.0 / (outputs["medium_rgb"].detach() + 1e-3)
            bg_loss = (bg_mask * bg_weight * torch.abs(outputs["medium_rgb"] - gt_img)).sum() / bg_mask.sum().clamp_min(1e-6)
            bgi_total_loss = (bg_mask * torch.abs(outputs["rgb_medium_total"] - gt_img)).sum() / bg_mask.sum().clamp_min(1e-6)
            bgi_finite_loss = (bg_mask * torch.abs(outputs["rgb_medium_finite"] - gt_img)).sum() / bg_mask.sum().clamp_min(1e-6)
            bg_mask_info = {
                "available": True,
                "coverage": float(bg_mask.detach().mean().item()),
                "pixel_count": int((bg_mask.detach() > 0.5).sum().item()),
                "mask_dir": str(mask_dir),
                "mask_key": args.background_mask_key,
                "exclude_boundary": bool(args.exclude_boundary),
                "hit_exclusion_threshold": float(args.hit_exclusion_threshold),
            }

    dc_params = [model.gauss_params["features_dc"]]
    rest_params = [model.gauss_params["features_rest"]]
    geom_params = [model.gauss_params["means"], model.gauss_params["scales"], model.gauss_params["quats"]]
    opacity_params = [model.gauss_params["opacities"]]
    medium_params = list(model.medium_mlp.parameters()) + list(model.direction_encoding.parameters())
    all_params = dc_params + rest_params + geom_params + opacity_params + medium_params

    def norms(loss: Tensor) -> Dict[str, float]:
        dc = _grad_norm(_safe_grad(loss, dc_params))
        rest = _grad_norm(_safe_grad(loss, rest_params))
        geom = _grad_norm(_safe_grad(loss, geom_params))
        opacity = _grad_norm(_safe_grad(loss, opacity_params))
        medium = _grad_norm(_safe_grad(loss, medium_params))
        all_norm = _grad_norm(_safe_grad(loss, all_params))
        return {
            "features_dc": dc,
            "features_rest": rest,
            "appearance_total": math.sqrt(dc * dc + rest * rest),
            "geometry": geom,
            "opacity": opacity,
            "medium": medium,
            "all": all_norm,
        }

    rgb_norms = norms(rgb_loss)
    bound_norms = norms(bound_loss)
    faw_norms = norms(faw_loss)
    bg_norms = norms(bg_loss)
    bgi_total_norms = norms(bgi_total_loss)
    bgi_finite_norms = norms(bgi_finite_loss)

    def lambda_for(target_ratio: float, numerator_raw_norm: float, denominator_norm: float) -> float:
        if numerator_raw_norm <= 0.0 or denominator_norm <= 0.0:
            return 0.0
        return float(target_ratio * denominator_norm / numerator_raw_norm)

    valid_colors = colors.detach()[visible.reshape(-1) > 0.0]
    valid_fg_weights = fg_weight.detach()[fg_mask.expand_as(fg_weight) > 0.0]
    result = {
        "losses": {
            "rgb": float(rgb_loss.detach().item()),
            "intrinsic_bound_raw": float(bound_loss.detach().item()),
            "foreground_aware_weighted_l1_raw": float(faw_loss.detach().item()),
            "background_medium_old_raw": float(bg_loss.detach().item()),
            "background_integrated_medium_total_raw": float(bgi_total_loss.detach().item()),
            "background_integrated_medium_finite_raw": float(bgi_finite_loss.detach().item()),
        },
        "gradient_norms": {
            "rgb": rgb_norms,
            "intrinsic_bound_raw": bound_norms,
            "foreground_aware_weighted_l1_raw": faw_norms,
            "background_medium_old_raw": bg_norms,
            "background_integrated_medium_total_raw": bgi_total_norms,
            "background_integrated_medium_finite_raw": bgi_finite_norms,
        },
        "recommended_lambdas": {
            "IB-G01": lambda_for(0.01, bound_norms["appearance_total"], rgb_norms["appearance_total"]),
            "IB-G05": lambda_for(0.05, bound_norms["appearance_total"], rgb_norms["appearance_total"]),
            "IB-G10": lambda_for(0.10, bound_norms["appearance_total"], rgb_norms["appearance_total"]),
            "FAW-G05": lambda_for(0.05, faw_norms["all"], rgb_norms["all"]),
            "BG-G05": lambda_for(0.05, bg_norms["medium"], rgb_norms["medium"]),
            "BGI-TOTAL-G05": lambda_for(0.05, bgi_total_norms["medium"], rgb_norms["medium"]),
            "BGI-FINITE-G05": lambda_for(0.05, bgi_finite_norms["medium"], rgb_norms["medium"]),
        },
        "actual_gradient_ratios": {
            "old_bg_lambda_0p01_vs_rgb_medium": (
                0.01 * bg_norms["medium"] / rgb_norms["medium"] if rgb_norms["medium"] > 0.0 else 0.0
            ),
            "raw_bound_vs_rgb_appearance": (
                bound_norms["appearance_total"] / rgb_norms["appearance_total"]
                if rgb_norms["appearance_total"] > 0.0
                else 0.0
            ),
            "raw_faw_vs_rgb_all": faw_norms["all"] / rgb_norms["all"] if rgb_norms["all"] > 0.0 else 0.0,
            "raw_bgi_total_vs_rgb_medium": (
                bgi_total_norms["medium"] / rgb_norms["medium"] if rgb_norms["medium"] > 0.0 else 0.0
            ),
        },
        "foreground": {
            "mask_definition": f"outputs['accumulation'] > {args.foreground_accumulation_threshold}",
            "coverage": float(fg_mask.detach().mean().item()),
            "weight_stats": _stats(valid_fg_weights if valid_fg_weights.numel() else torch.empty(0)),
            "weight_epsilon": float(args.foreground_weight_epsilon),
            "weight_cap": float(args.foreground_weight_cap),
        },
        "gaussian_view_rgb": {
            "visible_mask_definition": "outputs['gaussian_visible_mask'] == projected radii > 0",
            "visible_count": int(valid_colors.shape[0]),
            "channel_value_stats": _stats(valid_colors if valid_colors.numel() else torch.empty(0)),
            "P(c<0)": float((valid_colors < 0.0).float().mean().item()) if valid_colors.numel() else 0.0,
            "P(c>1)": float((valid_colors > 1.0).float().mean().item()) if valid_colors.numel() else 0.0,
            "P(c>1.5)": float((valid_colors > 1.5).float().mean().item()) if valid_colors.numel() else 0.0,
            "P(c>2)": float((valid_colors > 2.0).float().mean().item()) if valid_colors.numel() else 0.0,
        },
        "background": bg_mask_info,
    }
    return result


def _audit_mask_coverage(
    *,
    model: Any,
    pipeline: Any,
    args: argparse.Namespace,
    render_dir: Path,
) -> Dict[str, Any]:
    rows = []
    sheet_rows = []
    with torch.no_grad():
        dataset = pipeline.datamanager.train_dataset
        count = min(len(dataset.cameras), args.mask_audit_max_images if args.mask_audit_max_images > 0 else len(dataset.cameras))
        for image_idx in range(count):
            camera = dataset.cameras[image_idx : image_idx + 1].to(model.device)
            outputs = model.get_outputs_for_camera(camera=camera)
            fg_mask = (outputs["accumulation"].detach() > float(args.foreground_accumulation_threshold)).float().cpu()
            coverage = float(fg_mask.mean().item())
            path = render_dir / "foreground_masks" / f"view_{image_idx:04d}_fg_mask.png"
            _mask_png(path, fg_mask)
            rows.append({"image_idx": int(image_idx), "coverage": coverage, "file": str(path)})
            if len(sheet_rows) < 12:
                sheet_rows.append([(f"train {image_idx} fg", path)])
    coverages = np.array([row["coverage"] for row in rows], dtype=np.float64)
    sheet_path = render_dir / "foreground_masks" / "contact_sheet_foreground_masks.png"
    _write_sheet(sheet_path, sheet_rows, args.contact_tile_width)
    return {
        "definition": f"outputs['accumulation'] > {args.foreground_accumulation_threshold}",
        "view_count": int(len(rows)),
        "coverage": {
            "mean": float(coverages.mean()) if coverages.size else 0.0,
            "p10": float(np.quantile(coverages, 0.10)) if coverages.size else 0.0,
            "p50": float(np.quantile(coverages, 0.50)) if coverages.size else 0.0,
            "p90": float(np.quantile(coverages, 0.90)) if coverages.size else 0.0,
        },
        "rows": rows,
        "contact_sheet": str(sheet_path),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo = _repo_root()
    config_path = args.load_config
    if config_path is None:
        config_path = repo / DEFAULT_D010_13K_CONFIG
    if not config_path.is_absolute():
        config_path = repo / config_path
    mask_dir = args.background_mask_dir
    if mask_dir is not None and not mask_dir.is_absolute():
        mask_dir = repo / mask_dir

    config, pipeline, checkpoint_path, loaded_step = _setup_pipeline(config_path, args.load_step, args.test_mode)
    model = pipeline.model
    model.eval()
    dataset = pipeline.datamanager.train_dataset
    image_idx = int(args.image_index)
    camera = dataset.cameras[image_idx : image_idx + 1].to(model.device)
    batch = _to_device_batch(dataset[image_idx], model.device)
    camera_id = _camera_id(camera, image_idx)

    outputs = model.get_outputs(camera)
    audit = _losses_and_norms(model=model, outputs=outputs, batch=batch, mask_dir=mask_dir, camera_id=camera_id, args=args)
    mask_audit = _audit_mask_coverage(model=model, pipeline=pipeline, args=args, render_dir=args.render_dir)

    summary = {
        "diagnostic": "dewater_loss_gradient_audit",
        "git_commit": _git_commit(repo),
        "scene": args.scene,
        "load_config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "requested_step": int(args.load_step),
        "loaded_step": int(loaded_step),
        "test_mode": args.test_mode,
        "train_image_index": image_idx,
        "camera_id": int(camera_id),
        "direct_optical_depth_scale": float(getattr(model.config, "direct_optical_depth_scale", 1.0)),
        "audit": audit,
        "foreground_mask_audit": mask_audit,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf8") as handle:
        fieldnames = ["quantity", "value"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, value in audit["recommended_lambdas"].items():
            writer.writerow({"quantity": f"lambda_{name}", "value": value})
        for name, value in audit["actual_gradient_ratios"].items():
            writer.writerow({"quantity": name, "value": value})
        writer.writerow({"quantity": "foreground_coverage_batch", "value": audit["foreground"]["coverage"]})
        writer.writerow({"quantity": "foreground_coverage_mean", "value": mask_audit["coverage"]["mean"]})
        writer.writerow({"quantity": "background_available", "value": audit["background"]["available"]})
        if audit["background"].get("available"):
            writer.writerow({"quantity": "background_coverage_batch", "value": audit["background"]["coverage"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="Curasao")
    parser.add_argument("--load-config", type=Path)
    parser.add_argument("--load-step", type=int, default=13000)
    parser.add_argument("--test-mode", default="test")
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--foreground-accumulation-threshold", type=float, default=0.05)
    parser.add_argument("--foreground-weight-epsilon", type=float, default=1e-3)
    parser.add_argument("--foreground-weight-cap", type=float, default=-1.0)
    parser.add_argument("--background-mask-dir", type=Path, default=Path(DEFAULT_MASK_DIR))
    parser.add_argument("--background-mask-key", default="water")
    parser.set_defaults(exclude_boundary=True)
    parser.add_argument("--exclude-boundary", dest="exclude_boundary", action="store_true")
    parser.add_argument("--no-exclude-boundary", dest="exclude_boundary", action="store_false")
    parser.add_argument("--hit-exclusion-threshold", type=float, default=-1.0)
    parser.add_argument("--mask-audit-max-images", type=int, default=-1)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/dewater_seafree_factor_20260808/loss_gradient_audit.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/dewater_seafree_factor_20260808/loss_gradient_audit.csv"),
    )
    parser.add_argument("--render-dir", type=Path, default=Path("renders/dewater_seafree_factor_20260808/loss_gradient_audit"))
    parser.add_argument("--contact-tile-width", type=int, default=240)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
