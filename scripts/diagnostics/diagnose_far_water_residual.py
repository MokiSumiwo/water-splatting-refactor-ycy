#!/usr/bin/env python
"""Diagnose far-water Gaussian residuals from a trained WaterSplatting checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from nerfstudio.utils.eval_utils import eval_setup


def _quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device, dtype=values.dtype)
    return torch.quantile(values, q)


def _stats(values: torch.Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean().item()),
        "p50": float(_quantile(values, 0.50).item()),
        "p90": float(_quantile(values, 0.90).item()),
        "p95": float(_quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


def _masked_values(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return image[mask.squeeze(-1)]


def _maybe_save_heatmaps(
    *,
    output_dir: Path,
    image_idx: int,
    far_mask: torch.Tensor,
    accumulation: torch.Tensor,
    object_luma: torch.Tensor,
    m_inf: torch.Tensor | None,
    m_inf_eff: torch.Tensor | None,
) -> None:
    from torchvision.utils import save_image

    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(far_mask.float().permute(2, 0, 1).cpu(), output_dir / f"far_mask_{image_idx:04d}.png")
    save_image(accumulation.clamp(0, 1).permute(2, 0, 1).cpu(), output_dir / f"accumulation_{image_idx:04d}.png")
    save_image(object_luma.clamp(0, 1).permute(2, 0, 1).cpu(), output_dir / f"rgb_object_luma_{image_idx:04d}.png")
    if m_inf is not None:
        save_image(m_inf.clamp(0, 1).permute(2, 0, 1).cpu(), output_dir / f"m_inf_{image_idx:04d}.png")
    if m_inf_eff is not None:
        save_image(m_inf_eff.clamp(0, 1).permute(2, 0, 1).cpu(), output_dir / f"m_inf_eff_{image_idx:04d}.png")


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    config, pipeline, checkpoint_path, step = eval_setup(Path(args.load_config))
    pipeline.eval()

    image_summaries: List[Dict[str, Any]] = []
    all_far_alpha: List[torch.Tensor] = []
    all_far_object: List[torch.Tensor] = []
    all_far_m_inf: List[torch.Tensor] = []
    all_far_m_inf_eff: List[torch.Tensor] = []

    with torch.no_grad():
        for image_idx, (camera, _batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            depth = outputs["depth"].detach().float()
            accumulation = outputs["accumulation"].detach().float().clamp(0.0, 1.0)
            rgb_object = outputs["rgb_object"].detach().float()
            object_luma = rgb_object.abs().mean(dim=-1, keepdim=True)
            m_inf = outputs.get("m_inf")
            m_inf_eff = outputs.get("m_inf_eff")
            if m_inf is not None:
                m_inf = m_inf.detach().float().clamp(0.0, 1.0)
            if m_inf_eff is not None:
                m_inf_eff = m_inf_eff.detach().float().clamp(0.0, 1.0)

            valid_depth = torch.isfinite(depth) & (depth > 0)
            if valid_depth.any():
                depth_cutoff = _quantile(depth[valid_depth], args.far_depth_quantile)
                far_mask = valid_depth & (depth >= depth_cutoff)
            else:
                depth_cutoff = torch.tensor(0.0, device=depth.device, dtype=depth.dtype)
                far_mask = torch.zeros_like(depth, dtype=torch.bool)

            far_alpha = _masked_values(accumulation, far_mask)
            far_object = _masked_values(object_luma, far_mask)
            far_m_inf = _masked_values(m_inf, far_mask) if m_inf is not None else torch.empty(0, device=depth.device)
            far_m_inf_eff = (
                _masked_values(m_inf_eff, far_mask) if m_inf_eff is not None else torch.empty(0, device=depth.device)
            )

            all_far_alpha.append(far_alpha.detach().cpu())
            all_far_object.append(far_object.detach().cpu())
            if far_m_inf.numel() > 0:
                all_far_m_inf.append(far_m_inf.detach().cpu())
            if far_m_inf_eff.numel() > 0:
                all_far_m_inf_eff.append(far_m_inf_eff.detach().cpu())

            far_pixels = int(far_mask.sum().item())
            total_pixels = int(far_mask.numel())
            image_summary = {
                "image_index": image_idx,
                "height": int(depth.shape[0]),
                "width": int(depth.shape[1]),
                "far_depth_quantile": args.far_depth_quantile,
                "far_depth_cutoff": float(depth_cutoff.item()),
                "far_pixels": far_pixels,
                "far_fraction": far_pixels / max(total_pixels, 1),
                "far_accumulation": _stats(far_alpha),
                "far_rgb_object_luma": _stats(far_object),
                "far_m_inf": _stats(far_m_inf),
                "far_m_inf_eff": _stats(far_m_inf_eff),
                "far_alpha_gt_threshold_fraction": float((far_alpha > args.alpha_threshold).float().mean().item())
                if far_alpha.numel() > 0
                else 0.0,
                "far_object_gt_threshold_fraction": float((far_object > args.object_threshold).float().mean().item())
                if far_object.numel() > 0
                else 0.0,
            }
            image_summaries.append(image_summary)

            if args.save_heatmaps:
                _maybe_save_heatmaps(
                    output_dir=args.output_dir / "heatmaps",
                    image_idx=image_idx,
                    far_mask=far_mask,
                    accumulation=accumulation,
                    object_luma=object_luma,
                    m_inf=m_inf,
                    m_inf_eff=m_inf_eff,
                )

    far_alpha_all = torch.cat(all_far_alpha) if all_far_alpha else torch.empty(0)
    far_object_all = torch.cat(all_far_object) if all_far_object else torch.empty(0)
    far_m_inf_all = torch.cat(all_far_m_inf) if all_far_m_inf else torch.empty(0)
    far_m_inf_eff_all = torch.cat(all_far_m_inf_eff) if all_far_m_inf_eff else torch.empty(0)

    aggregate = {
        "far_pixels": int(sum(item["far_pixels"] for item in image_summaries)),
        "far_accumulation": _stats(far_alpha_all),
        "far_rgb_object_luma": _stats(far_object_all),
        "far_m_inf": _stats(far_m_inf_all),
        "far_m_inf_eff": _stats(far_m_inf_eff_all),
        "far_alpha_gt_threshold_fraction": float((far_alpha_all > args.alpha_threshold).float().mean().item())
        if far_alpha_all.numel() > 0
        else 0.0,
        "far_object_gt_threshold_fraction": float((far_object_all > args.object_threshold).float().mean().item())
        if far_object_all.numel() > 0
        else 0.0,
    }
    result = {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "checkpoint": str(checkpoint_path),
        "step": step,
        "load_config": str(args.load_config),
        "far_depth_quantile": args.far_depth_quantile,
        "alpha_threshold": args.alpha_threshold,
        "object_threshold": args.object_threshold,
        "aggregate": aggregate,
        "images": image_summaries,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--far-depth-quantile", type=float, default=0.90)
    parser.add_argument("--alpha-threshold", type=float, default=0.05)
    parser.add_argument("--object-threshold", type=float, default=0.03)
    parser.add_argument("--save-heatmaps", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = diagnose(args)
    output_json = args.output_dir / "far_water_residual_diagnostic.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"saved={output_json}")


if __name__ == "__main__":
    main()
