#!/usr/bin/env python
"""Signed, region-aware audit for budgeted-capacity gradients.

This diagnostic is read-only. It loads an existing checkpoint, evaluates a set
of train or eval views, and separates capacity gradients by sign and sampled
region. Regions are sampled at projected Gaussian centers from detached
medium-support maps and optional water/object/boundary masks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from nerfstudio.utils.eval_utils import eval_setup


LR_HINTS = {
    "opacities": 0.05,
    "scales": 0.005,
    "means": 1.6e-4,
}


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _camera_items(pipeline: Any, split: str, max_images: int, device: torch.device) -> Iterator[Tuple[int, Any, Dict[str, Any]]]:
    max_count = max_images if max_images > 0 else 10**9
    if split == "eval":
        for image_idx, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= max_count:
                break
            yield image_idx, camera.to(device) if hasattr(camera, "to") else camera, batch
        return

    dataset = pipeline.datamanager.train_dataset
    count = min(len(dataset.cameras), max_count)
    for image_idx in range(count):
        camera = dataset.cameras[image_idx : image_idx + 1]
        image = dataset[image_idx]["image"]
        yield image_idx, camera.to(device) if hasattr(camera, "to") else camera, {"image": image}


def _load_png_mask(mask_dir: Optional[Path], image_idx: int, key: str, shape: Tuple[int, int], device: torch.device) -> Optional[torch.Tensor]:
    if mask_dir is None:
        return None
    path = mask_dir / f"view_{image_idx:04d}_{key}.png"
    if not path.exists():
        return None
    mask = Image.open(path).convert("L")
    if mask.size != (shape[1], shape[0]):
        mask = mask.resize((shape[1], shape[0]), Image.NEAREST)
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy((arr > 0.5).astype(np.float32)).to(device=device)[..., None]


def _clear_grads(model: torch.nn.Module) -> None:
    model.zero_grad(set_to_none=True)
    for name in ("xys", "xys_grad_abs", "xys_grad_abs_proxy", "xys_grad_abs_capacity"):
        value = getattr(model, name, None)
        if value is not None and getattr(value, "retains_grad", False) and value.grad is not None:
            value.grad = None


def _safe_grad(loss: torch.Tensor, params: List[torch.Tensor], *, retain_graph: bool) -> List[Optional[torch.Tensor]]:
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    return [grad.detach() if grad is not None else None for grad in grads]


def _per_gaussian_norm(value: Optional[torch.Tensor], size: int, device: torch.device) -> torch.Tensor:
    if value is None:
        return torch.zeros(size, device=device)
    if value.ndim <= 1:
        return value.detach().float().reshape(-1).abs()
    return torch.linalg.vector_norm(value.detach().float().reshape(value.shape[0], -1), dim=-1)


def _sample_map_at_gaussians(pixel_map: Optional[torch.Tensor], xys: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    size = int(xys.shape[0])
    device = xys.device
    if pixel_map is None:
        return torch.zeros(size, device=device)
    if pixel_map.ndim == 2:
        pixel_map = pixel_map[..., None]
    h, w = int(pixel_map.shape[0]), int(pixel_map.shape[1])
    visible = (radii.detach().reshape(-1) > 0).to(device=device)
    xy = xys.detach().float()
    xi = xy[:, 0].round().long().clamp(0, w - 1)
    yi = xy[:, 1].round().long().clamp(0, h - 1)
    sampled = pixel_map.detach().to(device=device).float()[yi, xi, 0]
    return torch.where(visible, sampled.clamp(0.0, 1.0), torch.zeros_like(sampled))


def _weighted_sum(values: torch.Tensor, weight: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float((values.detach().float() * weight.detach().float()).sum().item())


def _weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> float:
    denom = weight.detach().float().sum().clamp_min(1e-12)
    return float((values.detach().float() * weight.detach().float()).sum().item() / denom.item())


def _region_stats(values: torch.Tensor, regions: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {name: _weighted_sum(values, weight) for name, weight in regions.items()}


def _region_means(values: torch.Tensor, regions: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {f"{name}_mean": _weighted_mean(values, weight) for name, weight in regions.items()}


def _scale_signed_stats(grad: Optional[torch.Tensor], size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if grad is None:
        return torch.zeros(size, device=device), torch.zeros(size, device=device)
    scale_grad = grad.detach().float().reshape(size, -1)
    shrink = scale_grad.clamp_min(0.0).sum(dim=-1)
    grow = (-scale_grad).clamp_min(0.0).sum(dim=-1)
    return shrink, grow


def _means_projection_stats(
    grad: Optional[torch.Tensor],
    means: torch.Tensor,
    camera: Any,
    regions: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    if grad is None:
        return {}
    g = grad.detach().float()
    center = camera.camera_to_worlds[..., :3, 3].detach().to(device=g.device, dtype=g.dtype).reshape(1, 3)
    ray = means.detach().float() - center
    ray = ray / ray.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    radial = (g * ray).sum(dim=-1)
    tangential = torch.linalg.vector_norm(g - radial[..., None] * ray, dim=-1)
    norm = torch.linalg.vector_norm(g, dim=-1)
    stats: Dict[str, float] = {}
    for prefix, values in (
        ("means_norm", norm),
        ("means_radial_abs", radial.abs()),
        ("means_tangential", tangential),
    ):
        for region, weight in regions.items():
            stats[f"{prefix}.{region}"] = _weighted_sum(values, weight)
    return stats


def _cosine_stats(a: Optional[torch.Tensor], b: Optional[torch.Tensor], regions: Dict[str, torch.Tensor]) -> Dict[str, float]:
    if a is None or b is None:
        return {}
    af = a.detach().float().reshape(a.shape[0], -1)
    bf = b.detach().float().reshape(b.shape[0], -1)
    cos = (af * bf).sum(dim=-1) / (af.norm(dim=-1) * bf.norm(dim=-1)).clamp_min(1e-12)
    return {f"means_rec_cap_cosine.{region}": _weighted_mean(cos, weight) for region, weight in regions.items()}


def _image_result(model: torch.nn.Module, camera: Any, batch: Dict[str, Any], image_idx: int, mask_dir: Optional[Path]) -> Dict[str, Any]:
    _clear_grads(model)
    outputs = model.get_outputs(camera)
    metrics: Dict[str, torch.Tensor] = {}
    loss_dict = model.get_loss_dict(outputs, batch, metrics)
    if "budgeted_capacity_loss" not in loss_dict:
        raise RuntimeError("budgeted_capacity_loss was not active")

    params = [model.opacities, model.scales, model.means]
    rec_opacity, rec_scales, rec_means = _safe_grad(loss_dict["main_loss"], params, retain_graph=True)
    cap_opacity, cap_scales, cap_means = _safe_grad(loss_dict["budgeted_capacity_loss"], params, retain_graph=False)

    size = int(model.scales.shape[0])
    device = model.scales.device
    h, w = outputs["accumulation"].shape[:2]
    xys = getattr(model, "xys")
    radii = getattr(model, "radii")
    masks = {
        key: _load_png_mask(mask_dir, image_idx, key, (h, w), device)
        for key in ("water", "object", "boundary", "uncertain")
    }
    core = _sample_map_at_gaussians(outputs.get("medium_support_capacity"), xys, radii)
    halo = _sample_map_at_gaussians(outputs.get("medium_support_halo_base"), xys, radii)
    water = _sample_map_at_gaussians(masks["water"], xys, radii)
    obj = _sample_map_at_gaussians(masks["object"], xys, radii)
    boundary = _sample_map_at_gaussians(masks["boundary"], xys, radii)
    uncertain = _sample_map_at_gaussians(masks["uncertain"], xys, radii)
    visible = (radii.detach().reshape(-1) > 0).float()
    occupied = torch.maximum(torch.maximum(core, halo), torch.maximum(obj, torch.maximum(boundary, uncertain)))
    other = (visible - occupied.clamp(0.0, 1.0)).clamp_min(0.0)
    regions = {
        "support_core": core,
        "support_halo": halo,
        "water_mask": water,
        "object": obj,
        "boundary": boundary,
        "uncertain": uncertain,
        "other_visible": other,
        "visible": visible,
    }

    cap_shrink, cap_grow = _scale_signed_stats(cap_scales, size, device)
    rec_shrink, rec_grow = _scale_signed_stats(rec_scales, size, device)
    cap_op_norm = _per_gaussian_norm(cap_opacity, size, device)
    cap_scale_norm = _per_gaussian_norm(cap_scales, size, device)
    cap_means_norm = _per_gaussian_norm(cap_means, size, device)
    effective_update = {
        "opacities": cap_op_norm * LR_HINTS["opacities"],
        "scales": cap_scale_norm * LR_HINTS["scales"],
        "means": cap_means_norm * LR_HINTS["means"],
    }

    result: Dict[str, Any] = {
        "image_index": int(image_idx),
        "main_loss": float(loss_dict["main_loss"].detach().item()),
        "budgeted_capacity_loss": float(loss_dict["budgeted_capacity_loss"].detach().item()),
        "visible_gaussians": int((visible > 0).sum().item()),
        "support_core_weight": float(core.sum().item()),
        "support_halo_weight": float(halo.sum().item()),
        "scale_capacity_shrink_mass": _region_stats(cap_shrink, regions),
        "scale_capacity_grow_mass": _region_stats(cap_grow, regions),
        "scale_reconstruction_shrink_mass": _region_stats(rec_shrink, regions),
        "scale_reconstruction_grow_mass": _region_stats(rec_grow, regions),
        "scale_capacity_shrink_minus_grow": _region_stats(cap_shrink - cap_grow, regions),
        "capacity_effective_update_mass": {
            key: _region_stats(value, regions)
            for key, value in effective_update.items()
        },
        "capacity_grad_norm_mean": {
            "opacities": _region_means(cap_op_norm, regions),
            "scales": _region_means(cap_scale_norm, regions),
            "means": _region_means(cap_means_norm, regions),
        },
        "means_capacity_projection": _means_projection_stats(cap_means, model.means, camera, regions),
        "means_rec_cap_cosine": _cosine_stats(rec_means, cap_means, regions),
    }
    _clear_grads(model)
    return result, {
        "visible": visible.detach().cpu(),
        "shrink": (cap_shrink > 0.0).detach().cpu(),
    }


def _mean_nested(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def merge(values: List[Any]) -> Any:
        first = values[0]
        if isinstance(first, dict):
            keys = sorted({key for item in values for key in item.keys()})
            return {key: merge([item[key] for item in values if key in item]) for key in keys}
        if isinstance(first, (int, float)):
            return float(np.mean([float(item) for item in values]))
        return first

    return merge(rows) if rows else {}


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    def _update_config(config: Any) -> Any:
        if args.load_step is not None:
            config.load_step = int(args.load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(args.load_config, update_config_callback=_update_config)
    pipeline.eval()
    model = pipeline.model
    if args.force_step is not None:
        model.step = int(args.force_step)
    else:
        model.step = int(step)
    if args.enable_capacity_control:
        model.config.capacity_control_enabled = True
        model.config.capacity_control_position_gradient_scale = float(args.position_scale)
        model.config.capacity_control_depth_gradient_scale = float(args.depth_scale)
        model.config.capacity_control_footprint_gradient_scale = float(args.footprint_scale)
        model.config.capacity_control_opacity_gradient_scale = float(args.opacity_scale)
        model.config.capacity_control_scale_shrink_only = bool(args.scale_shrink_only)
        model.config.capacity_control_scale_shrink_clip_quantile = float(args.scale_clip_quantile)
        model.config.capacity_control_scale_shrink_clip_value = float(args.scale_clip_value)
    images: List[Dict[str, Any]] = []
    visible_accum: Optional[torch.Tensor] = None
    shrink_accum: Optional[torch.Tensor] = None
    for image_idx, camera, batch in _camera_items(pipeline, args.split, args.max_images, model.device):
        image_result, consistency = _image_result(model, camera, batch, image_idx, args.mask_dir)
        images.append(image_result)
        visible = consistency["visible"].float()
        shrink = consistency["shrink"].float()
        visible_accum = visible if visible_accum is None else visible_accum + visible
        shrink_accum = shrink if shrink_accum is None else shrink_accum + shrink

    persistence: Dict[str, float] = {}
    if visible_accum is not None and shrink_accum is not None:
        active = visible_accum >= float(args.min_visible_views)
        ratio = torch.where(visible_accum > 0, shrink_accum / visible_accum.clamp_min(1.0), torch.zeros_like(visible_accum))
        active_ratio = ratio[active]
        persistence = {
            "active_gaussians": int(active.sum().item()),
            "visible_union_gaussians": int((visible_accum > 0).sum().item()),
            "mean_shrink_persistence": float(active_ratio.mean().item()) if active_ratio.numel() else 0.0,
            "p50_shrink_persistence": float(torch.quantile(active_ratio, 0.50).item()) if active_ratio.numel() else 0.0,
            "p90_shrink_persistence": float(torch.quantile(active_ratio, 0.90).item()) if active_ratio.numel() else 0.0,
            "frac_ge_0p70": float((active_ratio >= 0.70).float().mean().item()) if active_ratio.numel() else 0.0,
            "frac_ge_0p90": float((active_ratio >= 0.90).float().mean().item()) if active_ratio.numel() else 0.0,
        }

    repo = Path(__file__).resolve().parents[2]
    result = {
        "experiment_name": config.experiment_name,
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(step),
        "model_step_used": int(model.step),
        "split": args.split,
        "mask_dir": str(args.mask_dir) if args.mask_dir else "",
        "git_commit": _git_commit(repo),
        "lr_hints": LR_HINTS,
        "aggregate": _mean_nested(images),
        "persistence": persistence,
        "images": images,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({"aggregate": result["aggregate"], "persistence": persistence}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--force-step", type=int, default=None)
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-visible-views", type=int, default=5)
    parser.add_argument("--enable-capacity-control", action="store_true")
    parser.add_argument("--position-scale", type=float, default=0.0)
    parser.add_argument("--depth-scale", type=float, default=0.0)
    parser.add_argument("--footprint-scale", type=float, default=1.0)
    parser.add_argument("--opacity-scale", type=float, default=1.0)
    parser.add_argument("--scale-shrink-only", action="store_true")
    parser.add_argument("--scale-clip-quantile", type=float, default=-1.0)
    parser.add_argument("--scale-clip-value", type=float, default=0.0)
    args = parser.parse_args()
    diagnose(args)


if __name__ == "__main__":
    main()
