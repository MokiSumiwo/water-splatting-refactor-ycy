#!/usr/bin/env python
"""Visualize and summarize medium-explainability support maps."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import torch
from PIL import Image

from nerfstudio.utils.eval_utils import eval_setup
from water_splatting.attribution import build_residual_gated_halo_support, build_route_capacity_support


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _camera_items(pipeline: Any, split: str, max_images: int, device: torch.device) -> Iterator[Tuple[int, Any, Dict]]:
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


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().float().clamp(0.0, 1.0).numpy()
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _save_png(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(image)).save(path)


def _load_mask(mask_dir: Path | None, view_index: int, key: str, shape: tuple[int, int], device: torch.device) -> torch.Tensor | None:
    if mask_dir is None:
        return None
    path = mask_dir / f"view_{view_index:04d}_{key}.png"
    if not path.exists():
        return None
    mask = Image.open(path).convert("L")
    if mask.size != (shape[1], shape[0]):
        mask = mask.resize((shape[1], shape[0]), Image.NEAREST)
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy((arr > 0.5).astype(np.float32)).to(device=device)[..., None]


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    finite = torch.isfinite(x) & torch.isfinite(y)
    if int(finite.sum().item()) < 2:
        return 0.0
    x = x[finite] - x[finite].mean()
    y = y[finite] - y[finite].mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((x * y).sum().item() / denom.item())


def _gradient_proxy(gt: torch.Tensor) -> torch.Tensor:
    grad_x = torch.zeros_like(gt[..., 0:1])
    grad_y = torch.zeros_like(gt[..., 0:1])
    grad_x[:, 1:, :] = (gt[:, 1:, :] - gt[:, :-1, :]).abs().sum(dim=-1, keepdim=True)
    grad_y[1:, :, :] = (gt[1:, :, :] - gt[:-1, :, :]).abs().sum(dim=-1, keepdim=True)
    return grad_x + grad_y


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None) -> float:
    if mask is None or mask.sum() <= 0:
        return 0.0
    return float((value * mask).sum().item() / mask.sum().clamp_min(1e-6).item())


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    def _update_config(config: Any) -> Any:
        if args.load_step is not None:
            config.load_step = int(args.load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(
        args.load_config,
        eval_num_rays_per_chunk=None,
        test_mode="inference",
        update_config_callback=_update_config,
    )
    del config
    pipeline.eval()
    model = pipeline.model
    if args.enable_clear_proxy:
        model.config.clear_proxy_enabled = True
    device = model.device

    images = []
    for image_idx, camera, batch in _camera_items(pipeline, args.split, args.max_images, device):
        with torch.no_grad():
            outputs = model.get_outputs(camera)
        gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
        b_inf = outputs.get("b_inf", outputs["medium_rgb"])
        support = build_route_capacity_support(
            gt_img=gt,
            medium_rgb=b_inf,
            depth=outputs["depth"],
            gradient_tau=args.gradient_tau,
            variance_tau=args.variance_tau,
            color_tau=args.color_tau,
            luma_weight=args.luma_weight,
            far_floor=args.far_floor,
            depth_mid=args.depth_mid,
            depth_temperature=args.depth_temperature,
            use_flatness=not args.disable_flatness,
            use_medium=not args.disable_medium,
            use_far=not args.disable_far,
        )

        h, w = gt.shape[:2]
        masks = {
            key: _load_mask(args.mask_dir, image_idx, key, (h, w), device)
            for key in ("water", "object", "boundary")
        }
        grad_proxy = _gradient_proxy(gt)
        j_proxy = outputs.get("J_proxy_raw", torch.zeros_like(gt))
        proxy_bluegreen = torch.relu(torch.maximum(j_proxy[..., 1:2], j_proxy[..., 2:3]) - j_proxy[..., 0:1])
        halo_support = build_residual_gated_halo_support(
            j_proxy=j_proxy,
            medium_rgb=b_inf,
            broad_support=support.broad,
            core_support=support.core,
            chroma_margin=args.halo_chroma_margin,
            chroma_temperature=args.halo_chroma_temperature,
            luma_min=args.halo_luma_min,
            luma_temperature=args.halo_luma_temperature,
        )

        prefix = args.output_dir / f"{args.split}_{image_idx:04d}"
        _save_png(prefix.with_name(prefix.name + "_gt.png"), gt)
        _save_png(prefix.with_name(prefix.name + "_rgb.png"), outputs["pred_image"])
        _save_png(prefix.with_name(prefix.name + "_b_inf.png"), b_inf)
        _save_png(prefix.with_name(prefix.name + "_accumulation.png"), outputs["accumulation"])
        _save_png(prefix.with_name(prefix.name + "_proxy_bluegreen.png"), proxy_bluegreen)
        _save_png(prefix.with_name(prefix.name + "_S_flat.png"), support.flat)
        _save_png(prefix.with_name(prefix.name + "_S_med.png"), support.medium)
        _save_png(prefix.with_name(prefix.name + "_S_far.png"), support.far)
        _save_png(prefix.with_name(prefix.name + "_S_route.png"), support.route)
        _save_png(prefix.with_name(prefix.name + "_S_broad.png"), support.broad)
        _save_png(prefix.with_name(prefix.name + "_S_core.png"), support.core)
        _save_png(prefix.with_name(prefix.name + "_S_cap.png"), support.capacity)
        _save_png(prefix.with_name(prefix.name + "_S_halo_base.png"), support.halo_base)
        _save_png(prefix.with_name(prefix.name + "_S_halo.png"), halo_support)

        image_stats = {
            "image_index": image_idx,
            "support_route_mean": float(support.route.mean().item()),
            "support_broad_mean": float(support.broad.mean().item()),
            "support_core_mean": float(support.core.mean().item()),
            "support_capacity_mean": float(support.capacity.mean().item()),
            "support_halo_base_mean": float(support.halo_base.mean().item()),
            "support_halo_mean": float(halo_support.mean().item()),
            "support_capacity_gt_0p25_fraction": float((support.capacity > 0.25).float().mean().item()),
            "support_halo_gt_0p25_fraction": float((halo_support > 0.25).float().mean().item()),
            "support_accumulation_corr": _corr(support.capacity, outputs["accumulation"]),
            "support_halo_accumulation_corr": _corr(halo_support, outputs["accumulation"]),
            "support_gradient_corr": _corr(support.capacity, grad_proxy),
            "support_medium_error_corr": _corr(support.capacity, support.medium_error),
            "water_support_capacity_mean": _masked_mean(support.capacity, masks["water"]),
            "water_support_halo_mean": _masked_mean(halo_support, masks["water"]),
            "object_support_capacity_mean": _masked_mean(support.capacity, masks["object"]),
            "object_support_halo_mean": _masked_mean(halo_support, masks["object"]),
            "boundary_support_capacity_mean": _masked_mean(support.capacity, masks["boundary"]),
            "boundary_support_halo_mean": _masked_mean(halo_support, masks["boundary"]),
        }
        water = max(image_stats["water_support_capacity_mean"], 1e-12)
        image_stats["object_over_water_support"] = image_stats["object_support_capacity_mean"] / water
        image_stats["boundary_over_water_support"] = image_stats["boundary_support_capacity_mean"] / water
        images.append(image_stats)

    aggregate: Dict[str, Any] = {}
    if images:
        keys = [key for key, value in images[0].items() if isinstance(value, (int, float)) and key != "image_index"]
        aggregate = {key: float(np.mean([item[key] for item in images])) for key in keys}

    result = {
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "split": args.split,
        "load_config": str(args.load_config),
        "mask_dir": str(args.mask_dir) if args.mask_dir else "",
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "aggregate": aggregate,
        "images": images,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--enable-clear-proxy", action="store_true")
    parser.add_argument("--gradient-tau", type=float, default=0.05)
    parser.add_argument("--variance-tau", type=float, default=0.02)
    parser.add_argument("--color-tau", type=float, default=0.08)
    parser.add_argument("--luma-weight", type=float, default=0.25)
    parser.add_argument("--far-floor", type=float, default=0.50)
    parser.add_argument("--depth-mid", type=float, default=0.75)
    parser.add_argument("--depth-temperature", type=float, default=0.15)
    parser.add_argument("--halo-chroma-margin", type=float, default=0.015)
    parser.add_argument("--halo-chroma-temperature", type=float, default=0.01)
    parser.add_argument("--halo-luma-min", type=float, default=0.02)
    parser.add_argument("--halo-luma-temperature", type=float, default=0.01)
    parser.add_argument("--disable-flatness", action="store_true")
    parser.add_argument("--disable-medium", action="store_true")
    parser.add_argument("--disable-far", action="store_true")
    args = parser.parse_args()

    result = diagnose(args)
    print(json.dumps({"step": result["step"], "aggregate": result["aggregate"]}, indent=2))
    print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()
