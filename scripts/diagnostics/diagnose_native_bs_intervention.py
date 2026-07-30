#!/usr/bin/env python
"""Renderer-native no-training intervention on medium backscatter coefficients.

This diagnostic leaves Gaussian geometry, opacity, SH color, medium RGB, and
medium attenuation unchanged. It monkeypatches the Python rasterizer wrapper for
one evaluation pass and replaces only the per-pixel ``medium_bs`` tensor before
the native CUDA rasterizer call.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from PIL import Image

from nerfstudio.utils.eval_utils import eval_setup


VARIANT_LABELS = {
    "original": "Original BS",
    "D3_bs_mean": "D3 BS Mean",
    "D11_bs_spectrum_mean": "D11 BS Spectrum",
    "D14_bs_spectrum_shrink050": "D14 BS Spectrum 0.50",
}


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().float().cpu()
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        image = image.expand(*image.shape[:2], 3)
    return (image.clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)


def _save_png(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(tensor)).save(path)


def _channel_normalize(value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    value = value.clamp_min(0.0)
    return value / value.sum(dim=-1, keepdim=True).clamp_min(eps)


def _mean_std(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    vals = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if vals.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}


def _valid_mask_from_outputs(outputs: Dict[str, torch.Tensor], threshold: float) -> torch.Tensor:
    depth = outputs["depth"]
    accumulation = outputs["accumulation"]
    if depth.ndim == 2:
        depth = depth[..., None]
    if accumulation.ndim == 2:
        accumulation = accumulation[..., None]
    return torch.isfinite(depth) & (depth > 0.0) & (accumulation > float(threshold))


def _replace_bs(medium_bs: torch.Tensor, variant: str, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if variant == "original":
        return medium_bs
    if medium_bs.ndim != 3 or medium_bs.shape[-1] != 3:
        return medium_bs
    mask = valid_mask
    if mask is not None:
        mask = mask.to(device=medium_bs.device)
        if mask.ndim == 2:
            mask = mask[..., None]
        if mask.shape[:2] != medium_bs.shape[:2]:
            mask = None
    if mask is None or not bool(mask.any().item()):
        mask = torch.ones(*medium_bs.shape[:2], 1, device=medium_bs.device, dtype=torch.bool)
    selector = mask.squeeze(-1)

    if variant == "D3_bs_mean":
        mean_bs = medium_bs[selector].mean(dim=0)
        return mean_bs.view(1, 1, 3).expand_as(medium_bs).clamp_min(0.0)

    strength = medium_bs.mean(dim=-1, keepdim=True)
    spectrum = _channel_normalize(medium_bs)
    mean_spectrum = _channel_normalize(spectrum[selector].mean(dim=0).view(1, 1, 3))

    if variant == "D11_bs_spectrum_mean":
        return (3.0 * strength * mean_spectrum.expand_as(medium_bs)).clamp_min(0.0)
    if variant == "D14_bs_spectrum_shrink050":
        mixed = _channel_normalize(0.5 * spectrum + 0.5 * mean_spectrum.expand_as(medium_bs))
        return (3.0 * strength * mixed).clamp_min(0.0)
    raise ValueError(f"Unknown native BS intervention variant: {variant}")


def _render_with_variant(
    *,
    model: Any,
    camera: Any,
    variant: str,
    valid_mask: torch.Tensor | None,
) -> Dict[str, torch.Tensor]:
    if variant == "original":
        return model.get_outputs_for_camera(camera=camera)

    rasterizer = model.underwater_rasterizer
    original_rasterize = rasterizer.rasterize

    def wrapped_rasterize(**kwargs: Any) -> Any:
        if "medium_bs" in kwargs:
            kwargs = dict(kwargs)
            kwargs["medium_bs"] = _replace_bs(kwargs["medium_bs"], variant, valid_mask)
        return original_rasterize(**kwargs)

    rasterizer.rasterize = wrapped_rasterize  # type: ignore[method-assign]
    try:
        return model.get_outputs_for_camera(camera=camera)
    finally:
        rasterizer.rasterize = original_rasterize  # type: ignore[method-assign]


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

    variants = args.variants or ["original", "D3_bs_mean", "D11_bs_spectrum_mean", "D14_bs_spectrum_shrink050"]
    for variant in variants:
        if variant not in VARIANT_LABELS:
            raise ValueError(f"Unknown variant '{variant}'. Valid variants: {sorted(VARIANT_LABELS)}")

    rows: List[Dict[str, Any]] = []
    max_images = args.max_images if args.max_images > 0 else 10**9
    with torch.no_grad():
        for image_idx, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= max_images:
                break
            original_outputs = model.get_outputs_for_camera(camera=camera)
            valid_mask = _valid_mask_from_outputs(original_outputs, args.valid_accumulation_threshold)
            original_rgb = original_outputs["pred_image"].detach()
            original_medium = original_outputs["rgb_medium_total"].detach()

            for variant in variants:
                if variant == "original":
                    outputs = original_outputs
                else:
                    outputs = _render_with_variant(
                        model=model,
                        camera=camera,
                        variant=variant,
                        valid_mask=valid_mask,
                    )
                metrics, _images = model.get_image_metrics_and_images(outputs, batch)
                row: Dict[str, Any] = {
                    "image_index": image_idx,
                    "variant": variant,
                    "label": VARIANT_LABELS[variant],
                    "valid_coverage": float(valid_mask.float().mean().item()),
                }
                row.update({key: float(value) for key, value in metrics.items()})
                rows.append(row)

                if args.save_images:
                    view_dir = args.output_dir / f"view_{image_idx:04d}" / variant
                    _save_png(view_dir / "rgb.png", outputs["pred_image"])
                    _save_png(view_dir / "J.png", outputs["J"])
                    _save_png(view_dir / "rgb_diff_vs_original_x10.png", (outputs["pred_image"] - original_rgb).abs() * 10.0)
                    _save_png(
                        view_dir / "medium_total_diff_vs_original_x10.png",
                        (outputs["rgb_medium_total"] - original_medium).abs() * 10.0,
                    )
                    _save_png(view_dir / "rgb_medium_total.png", outputs["rgb_medium_total"])

    aggregates: Dict[str, Any] = {}
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        aggregates[variant] = {
            key: _mean_std(subset, key)
            for key in (
                "psnr",
                "ssim",
                "lpips",
                "J_blue_dominance_ratio",
                "J_green_dominance_ratio",
                "J_red_dominance_ratio",
                "J_saturation_ratio",
                "J_white_ratio",
            )
            if subset and key in subset[0]
        }
    if "original" in aggregates:
        ref = aggregates["original"]
        for variant, payload in aggregates.items():
            if variant == "original":
                continue
            payload["delta_vs_original"] = {
                "psnr": payload["psnr"]["mean"] - ref["psnr"]["mean"],
                "ssim": payload["ssim"]["mean"] - ref["ssim"]["mean"],
                "lpips": payload["lpips"]["mean"] - ref["lpips"]["mean"],
                "J_blue_dominance_ratio": payload.get("J_blue_dominance_ratio", {"mean": 0.0})["mean"]
                - ref.get("J_blue_dominance_ratio", {"mean": 0.0})["mean"],
            }

    result = {
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "test_mode": args.test_mode,
        "variants": variants,
        "valid_accumulation_threshold": args.valid_accumulation_threshold,
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "aggregate": aggregates,
        "images": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "native_bs_intervention.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({"step": result["step"], "aggregate": result["aggregate"]}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--test-mode", choices=["test", "val", "inference"], default="inference")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valid-accumulation-threshold", type=float, default=0.01)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()
    diagnose(args)


if __name__ == "__main__":
    main()
