#!/usr/bin/env python
"""Renderer-native no-training intervention on medium attenuation spectra.

This diagnostic replaces only ``medium_attn`` inside the Python rasterizer call
with a view-level spectrum plus bounded pixel residual approximation:

    log beta_c(p) = m(p) + s_c + rho * tanh(delta_c(p) / rho)

where both the view spectrum and pixel residual are zero-mean across channels.
It is an offline safety check before making this parameterization trainable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from nerfstudio.utils.eval_utils import eval_setup


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _mean_std(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    vals = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
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


def _spectrum_stats(medium_attn: torch.Tensor, valid_mask: torch.Tensor | None) -> Dict[str, float]:
    log_beta = medium_attn.clamp_min(1e-8).log()
    centered = log_beta - log_beta.mean(dim=-1, keepdim=True)
    if valid_mask is not None and valid_mask.shape[:2] == medium_attn.shape[:2] and bool(valid_mask.any().item()):
        selector = valid_mask.squeeze(-1)
    else:
        selector = torch.ones(*medium_attn.shape[:2], device=medium_attn.device, dtype=torch.bool)
    vals = centered[selector]
    if vals.numel() == 0:
        vals = centered.reshape(-1, 3)
    violation_rg = (medium_attn[..., 0] < medium_attn[..., 1]).float()
    violation_gb = (medium_attn[..., 1] < medium_attn[..., 2]).float()
    violation_any = ((violation_rg > 0) | (violation_gb > 0)).float()
    return {
        "spectral_abs_mean": float(vals.abs().mean().item()),
        "spectral_abs_p95": float(torch.quantile(vals.abs().reshape(-1).float(), 0.95).item()),
        "attn_violation_any": float(violation_any[selector].mean().item()),
        "attn_violation_rg": float(violation_rg[selector].mean().item()),
        "attn_violation_gb": float(violation_gb[selector].mean().item()),
        "attn_mean_r": float(medium_attn[..., 0][selector].mean().item()),
        "attn_mean_g": float(medium_attn[..., 1][selector].mean().item()),
        "attn_mean_b": float(medium_attn[..., 2][selector].mean().item()),
    }


def _bounded_attn(
    medium_attn: torch.Tensor,
    rho: float,
    valid_mask: torch.Tensor | None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if medium_attn.ndim != 3 or medium_attn.shape[-1] != 3:
        return medium_attn, {}
    mask = valid_mask
    if mask is not None:
        mask = mask.to(device=medium_attn.device)
        if mask.ndim == 2:
            mask = mask[..., None]
        if mask.shape[:2] != medium_attn.shape[:2]:
            mask = None
    if mask is None or not bool(mask.any().item()):
        mask = torch.ones(*medium_attn.shape[:2], 1, device=medium_attn.device, dtype=torch.bool)
    selector = mask.squeeze(-1)

    log_beta = medium_attn.clamp_min(1e-8).log()
    strength = log_beta.mean(dim=-1, keepdim=True)
    spectral = log_beta - strength
    view_spectrum = spectral[selector].mean(dim=0).view(1, 1, 3)
    view_spectrum = view_spectrum - view_spectrum.mean(dim=-1, keepdim=True)
    residual = spectral - view_spectrum
    residual = residual - residual.mean(dim=-1, keepdim=True)
    rho = max(float(rho), 1e-8)
    bounded_residual = rho * torch.tanh(residual / rho)
    bounded_residual = bounded_residual - bounded_residual.mean(dim=-1, keepdim=True)
    approx = (strength + view_spectrum + bounded_residual).exp().clamp_min(0.0)

    before = residual[selector]
    after = bounded_residual[selector]
    stats = {
        "rho": float(rho),
        "view_spectrum_r": float(view_spectrum[..., 0].mean().item()),
        "view_spectrum_g": float(view_spectrum[..., 1].mean().item()),
        "view_spectrum_b": float(view_spectrum[..., 2].mean().item()),
        "residual_abs_mean_before": float(before.abs().mean().item()),
        "residual_abs_mean_after": float(after.abs().mean().item()),
        "residual_abs_p95_before": float(torch.quantile(before.abs().reshape(-1).float(), 0.95).item()),
        "residual_abs_p95_after": float(torch.quantile(after.abs().reshape(-1).float(), 0.95).item()),
        "attn_l1_delta": float((approx - medium_attn).abs()[selector].mean().item()),
    }
    stats.update({f"approx_{key}": value for key, value in _spectrum_stats(approx, mask).items()})
    return approx, stats


def _render_with_rho(
    *,
    model: Any,
    camera: Any,
    rho: float,
    valid_mask: torch.Tensor | None,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, float]]]:
    rasterizer = model.underwater_rasterizer
    original_rasterize = rasterizer.rasterize
    replacements: List[Dict[str, float]] = []

    def wrapped_rasterize(**kwargs: Any) -> Any:
        if "medium_attn" in kwargs:
            kwargs = dict(kwargs)
            kwargs["medium_attn"], stats = _bounded_attn(kwargs["medium_attn"], rho, valid_mask)
            if stats:
                replacements.append(stats)
        return original_rasterize(**kwargs)

    rasterizer.rasterize = wrapped_rasterize  # type: ignore[method-assign]
    try:
        outputs = model.get_outputs_for_camera(camera=camera)
    finally:
        rasterizer.rasterize = original_rasterize  # type: ignore[method-assign]
    return outputs, replacements


def _avg_replacement_stats(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.asarray([row[key] for row in rows if key in row], dtype=np.float64).mean()) for key in keys}


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
    rhos = [float(rho) for rho in args.rhos]

    rows: List[Dict[str, Any]] = []
    max_images = args.max_images if args.max_images > 0 else 10**9
    with torch.no_grad():
        for image_idx, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= max_images:
                break
            original_outputs = model.get_outputs_for_camera(camera=camera)
            valid_mask = _valid_mask_from_outputs(original_outputs, args.valid_accumulation_threshold)
            original_rgb = original_outputs["pred_image"].detach()
            original_stats = _spectrum_stats(original_outputs["medium_attn"].detach(), valid_mask)
            original_metrics, _ = model.get_image_metrics_and_images(original_outputs, batch)
            original_row: Dict[str, Any] = {
                "image_index": image_idx,
                "variant": "original",
                "rho": None,
                "valid_coverage": float(valid_mask.float().mean().item()),
                "rgb_l1_vs_original": 0.0,
            }
            original_row.update({key: float(value) for key, value in original_metrics.items()})
            original_row.update({f"original_{key}": value for key, value in original_stats.items()})
            rows.append(original_row)

            for rho in rhos:
                outputs, replacements = _render_with_rho(model=model, camera=camera, rho=rho, valid_mask=valid_mask)
                metrics, _ = model.get_image_metrics_and_images(outputs, batch)
                repl = _avg_replacement_stats(replacements)
                row = {
                    "image_index": image_idx,
                    "variant": f"rho_{rho:.2f}",
                    "rho": float(rho),
                    "valid_coverage": float(valid_mask.float().mean().item()),
                    "rgb_l1_vs_original": float((outputs["pred_image"].detach() - original_rgb).abs().mean().item()),
                }
                row.update({key: float(value) for key, value in metrics.items()})
                row.update(repl)
                rows.append(row)

    aggregates: Dict[str, Any] = {}
    for variant in sorted({row["variant"] for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        aggregates[variant] = {
            key: _mean_std(subset, key)
            for key in (
                "psnr",
                "ssim",
                "lpips",
                "rgb_l1_vs_original",
                "J_blue_dominance_ratio",
                "J_green_dominance_ratio",
                "J_red_dominance_ratio",
                "J_saturation_ratio",
                "residual_abs_mean_before",
                "residual_abs_mean_after",
                "residual_abs_p95_before",
                "residual_abs_p95_after",
                "attn_l1_delta",
                "approx_spectral_abs_mean",
                "approx_spectral_abs_p95",
                "approx_attn_violation_any",
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
        "experiment": "native_attn_spectrum_bound",
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "test_mode": args.test_mode,
        "rhos": rhos,
        "valid_accumulation_threshold": args.valid_accumulation_threshold,
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "aggregate": aggregates,
        "images": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "native_attn_spectrum_bound.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({"step": result["step"], "aggregate": result["aggregate"]}, indent=2))
    print(f"saved={output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--test-mode", choices=["test", "val", "inference"], default="inference")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valid-accumulation-threshold", type=float, default=0.01)
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    args = parser.parse_args()
    diagnose(args)


if __name__ == "__main__":
    main()
