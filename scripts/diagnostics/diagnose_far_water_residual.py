#!/usr/bin/env python
"""Diagnose far-water Gaussian residuals from a trained WaterSplatting checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import deque
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
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


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    weights = rgb.new_tensor([0.2126, 0.7152, 0.0722])
    return (rgb * weights).sum(dim=-1, keepdim=True)


def _chroma(rgb: torch.Tensor) -> torch.Tensor:
    return rgb - rgb.mean(dim=-1, keepdim=True)


def _medium_projection(j_proxy: torch.Tensor, medium_rgb: torch.Tensor) -> torch.Tensor:
    j_chroma = _chroma(j_proxy.detach().float())
    medium_chroma = _chroma(medium_rgb.detach().float())
    medium_dir = medium_chroma / medium_chroma.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (j_chroma * medium_dir).sum(dim=-1, keepdim=True)


def _largest_connected_component_pixels(mask: torch.Tensor) -> int:
    arr = np.asarray(mask.detach().cpu().squeeze(-1).numpy(), dtype=bool)
    if arr.size == 0 or not arr.any():
        return 0
    try:
        from scipy import ndimage  # type: ignore

        labels, num = ndimage.label(arr)
        if num == 0:
            return 0
        sizes = np.bincount(labels.reshape(-1))
        return int(sizes[1:].max()) if sizes.size > 1 else 0
    except Exception:
        visited = np.zeros_like(arr, dtype=bool)
        h, w = arr.shape
        best = 0
        for y in range(h):
            for x in range(w):
                if not arr[y, x] or visited[y, x]:
                    continue
                visited[y, x] = True
                count = 0
                queue: deque[tuple[int, int]] = deque([(y, x)])
                while queue:
                    cy, cx = queue.popleft()
                    count += 1
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and arr[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                best = max(best, count)
        return int(best)


def _fraction_gt(values: torch.Tensor, threshold: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float((values > threshold).float().mean().item())


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_common_mask(mask_dir: Path, image_idx: int, device: torch.device, shape: torch.Size) -> torch.Tensor:
    mask_path = mask_dir / f"view_{image_idx:04d}_far.pt"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing common far mask: {mask_path}")
    payload = torch.load(mask_path, map_location="cpu")
    mask = payload["mask"] if isinstance(payload, dict) and "mask" in payload else payload
    mask = mask.to(device=device, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask[..., None]
    if mask.shape[:2] != shape[:2]:
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match output shape {tuple(shape)}")
    return mask


def _maybe_save_heatmaps(
    *,
    output_dir: Path,
    image_idx: int,
    far_mask: torch.Tensor,
    accumulation: torch.Tensor,
    object_luma: torch.Tensor,
    clear_luma: torch.Tensor,
    j_object_luma: torch.Tensor,
    m_inf: torch.Tensor | None,
    m_inf_eff: torch.Tensor | None,
    m_capacity: torch.Tensor | None,
    hit_confidence: torch.Tensor | None,
    depth_std_relative: torch.Tensor | None,
    proxy_luma: torch.Tensor | None,
    medium_projection: torch.Tensor | None,
    far_bg_residual_mask: torch.Tensor | None,
) -> None:
    from torchvision.utils import save_image

    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(far_mask.float().permute(2, 0, 1).cpu(), output_dir / f"far_mask_{image_idx:04d}.png")
    save_image(accumulation.clamp(0, 1).permute(2, 0, 1).cpu(), output_dir / f"accumulation_{image_idx:04d}.png")
    save_image(object_luma.clamp(0, 1).permute(2, 0, 1).cpu(), output_dir / f"rgb_object_luma_{image_idx:04d}.png")
    save_image(clear_luma.clamp(0, 1).permute(2, 0, 1).cpu(), output_dir / f"J_gaussian_luma_{image_idx:04d}.png")
    save_image(j_object_luma.clamp(0, 1).permute(2, 0, 1).cpu(), output_dir / f"J_object_luma_{image_idx:04d}.png")
    save_image(
        (far_mask.float() * accumulation).clamp(0, 1).permute(2, 0, 1).cpu(),
        output_dir / f"far_accumulation_overlay_{image_idx:04d}.png",
    )
    save_image(
        (far_mask.float() * clear_luma).clamp(0, 1).permute(2, 0, 1).cpu(),
        output_dir / f"far_J_leakage_overlay_{image_idx:04d}.png",
    )
    if m_inf is not None:
        save_image(m_inf.clamp(0, 1).permute(2, 0, 1).cpu(), output_dir / f"m_inf_{image_idx:04d}.png")
    if m_inf_eff is not None:
        save_image(m_inf_eff.clamp(0, 1).permute(2, 0, 1).cpu(), output_dir / f"m_inf_eff_{image_idx:04d}.png")
    if m_capacity is not None:
        save_image(m_capacity.clamp(0, 1).permute(2, 0, 1).cpu(), output_dir / f"m_capacity_{image_idx:04d}.png")
    if hit_confidence is not None:
        save_image(hit_confidence.clamp(0, 1).permute(2, 0, 1).cpu(), output_dir / f"hit_confidence_{image_idx:04d}.png")
    if depth_std_relative is not None:
        save_image(
            depth_std_relative.clamp(0, 1).permute(2, 0, 1).cpu(),
            output_dir / f"depth_std_relative_{image_idx:04d}.png",
        )
    if proxy_luma is not None:
        save_image(proxy_luma.clamp(0, 1).permute(2, 0, 1).cpu(), output_dir / f"J_proxy_luma_{image_idx:04d}.png")
    if medium_projection is not None:
        save_image(
            (medium_projection / 0.10).clamp(0, 1).permute(2, 0, 1).cpu(),
            output_dir / f"proxy_medium_projection_{image_idx:04d}.png",
        )
    if far_bg_residual_mask is not None:
        save_image(
            far_bg_residual_mask.float().permute(2, 0, 1).cpu(),
            output_dir / f"far_bg_residual_mask_{image_idx:04d}.png",
        )


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    config, pipeline, checkpoint_path, step = eval_setup(Path(args.load_config))
    pipeline.eval()
    pipeline.model.config.clear_proxy_enabled = True

    image_summaries: List[Dict[str, Any]] = []
    all_far_alpha: List[torch.Tensor] = []
    all_far_object: List[torch.Tensor] = []
    all_far_clear: List[torch.Tensor] = []
    all_far_j_object: List[torch.Tensor] = []
    all_far_m_inf: List[torch.Tensor] = []
    all_far_m_inf_eff: List[torch.Tensor] = []
    all_far_m_capacity: List[torch.Tensor] = []
    all_far_hit_confidence: List[torch.Tensor] = []
    all_far_bg_luma: List[torch.Tensor] = []
    all_far_bg_accumulation: List[torch.Tensor] = []
    all_far_bg_projection: List[torch.Tensor] = []

    with torch.no_grad():
        for image_idx, (camera, _batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            depth = outputs["depth"].detach().float()
            accumulation = outputs["accumulation"].detach().float().clamp(0.0, 1.0)
            rgb_object = outputs["rgb_object"].detach().float()
            object_luma = _luma(rgb_object.abs())
            clear_luma = _luma(outputs.get("J_gaussian", outputs["J"]).detach().float().clamp(0.0, 1.0))
            j_object_luma = _luma(outputs.get("J_object", outputs.get("J_gaussian", outputs["J"])).detach().float().clamp(0.0, 1.0))
            j_proxy = outputs.get("J_proxy_raw", outputs.get("J_gaussian_raw", outputs["J"])).detach().float()
            proxy_luma = _luma(j_proxy)
            medium_rgb = outputs.get("b_inf", outputs["medium_rgb"]).detach().float()
            medium_projection = _medium_projection(j_proxy, medium_rgb)
            m_inf = outputs.get("m_inf")
            m_inf_eff = outputs.get("m_inf_eff")
            m_capacity = outputs.get("m_capacity")
            hit_confidence = outputs.get("hit_confidence")
            depth_std_relative = outputs.get("depth_std_relative")
            if m_inf is not None:
                m_inf = m_inf.detach().float().clamp(0.0, 1.0)
            if m_inf_eff is not None:
                m_inf_eff = m_inf_eff.detach().float().clamp(0.0, 1.0)
            if m_capacity is not None:
                m_capacity = m_capacity.detach().float().clamp(0.0, 1.0)
            if hit_confidence is not None:
                hit_confidence = hit_confidence.detach().float().clamp(0.0, 1.0)
            if depth_std_relative is not None:
                depth_std_relative = depth_std_relative.detach().float()

            if args.mask_dir is not None:
                depth_cutoff = torch.tensor(float("nan"), device=depth.device, dtype=depth.dtype)
                far_mask = _load_common_mask(args.mask_dir, image_idx, depth.device, depth.shape)
            else:
                valid_depth = torch.isfinite(depth) & (depth > 0)
                if valid_depth.any():
                    depth_cutoff = _quantile(depth[valid_depth], args.far_depth_quantile)
                    far_mask = valid_depth & (depth >= depth_cutoff)
                else:
                    depth_cutoff = torch.tensor(0.0, device=depth.device, dtype=depth.dtype)
                    far_mask = torch.zeros_like(depth, dtype=torch.bool)

            far_alpha = _masked_values(accumulation, far_mask)
            far_object = _masked_values(object_luma, far_mask)
            far_clear = _masked_values(clear_luma, far_mask)
            far_j_object = _masked_values(j_object_luma, far_mask)
            far_m_inf = _masked_values(m_inf, far_mask) if m_inf is not None else torch.empty(0, device=depth.device)
            far_m_inf_eff = (
                _masked_values(m_inf_eff, far_mask) if m_inf_eff is not None else torch.empty(0, device=depth.device)
            )
            far_m_capacity = (
                _masked_values(m_capacity, far_mask) if m_capacity is not None else torch.empty(0, device=depth.device)
            )
            far_hit_confidence = (
                _masked_values(hit_confidence, far_mask)
                if hit_confidence is not None
                else torch.empty(0, device=depth.device)
            )
            far_bg_residual_mask = (
                far_mask
                & (medium_projection > float(args.bg_chroma_threshold))
                & (proxy_luma > float(args.bg_luma_threshold))
            )
            far_bg_luma = _masked_values(proxy_luma, far_bg_residual_mask)
            far_bg_accumulation = _masked_values(accumulation, far_bg_residual_mask)
            far_bg_projection = _masked_values(medium_projection, far_bg_residual_mask)
            far_bg_largest_component_pixels = _largest_connected_component_pixels(far_bg_residual_mask)

            all_far_alpha.append(far_alpha.detach().cpu())
            all_far_object.append(far_object.detach().cpu())
            all_far_clear.append(far_clear.detach().cpu())
            all_far_j_object.append(far_j_object.detach().cpu())
            if far_m_inf.numel() > 0:
                all_far_m_inf.append(far_m_inf.detach().cpu())
            if far_m_inf_eff.numel() > 0:
                all_far_m_inf_eff.append(far_m_inf_eff.detach().cpu())
            if far_m_capacity.numel() > 0:
                all_far_m_capacity.append(far_m_capacity.detach().cpu())
            if far_hit_confidence.numel() > 0:
                all_far_hit_confidence.append(far_hit_confidence.detach().cpu())
            if far_bg_luma.numel() > 0:
                all_far_bg_luma.append(far_bg_luma.detach().cpu())
            if far_bg_accumulation.numel() > 0:
                all_far_bg_accumulation.append(far_bg_accumulation.detach().cpu())
            if far_bg_projection.numel() > 0:
                all_far_bg_projection.append(far_bg_projection.detach().cpu())

            far_pixels = int(far_mask.sum().item())
            far_bg_pixels = int(far_bg_residual_mask.sum().item())
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
                "far_clear_luma": _stats(far_clear),
                "far_J_object_luma": _stats(far_j_object),
                "far_m_inf": _stats(far_m_inf),
                "far_m_inf_eff": _stats(far_m_inf_eff),
                "far_m_capacity": _stats(far_m_capacity),
                "far_hit_confidence": _stats(far_hit_confidence),
                "far_alpha_gt_threshold_fraction": _fraction_gt(far_alpha, args.alpha_threshold),
                "far_object_gt_threshold_fraction": _fraction_gt(far_object, args.object_threshold),
                "far_clear_gt_threshold_fraction": _fraction_gt(far_clear, args.clear_threshold),
                "far_m_inf_gt_0p5_fraction": _fraction_gt(far_m_inf, 0.5),
                "far_m_inf_eff_gt_0p5_fraction": _fraction_gt(far_m_inf_eff, 0.5),
                "far_bg_residual_pixels": far_bg_pixels,
                "far_bg_residual_fraction": far_bg_pixels / max(far_pixels, 1),
                "far_bg_residual_luma": _stats(far_bg_luma),
                "far_bg_residual_accumulation": _stats(far_bg_accumulation),
                "far_bg_residual_projection": _stats(far_bg_projection),
                "far_bg_largest_component_pixels": far_bg_largest_component_pixels,
                "far_bg_largest_component_fraction": far_bg_largest_component_pixels / max(far_pixels, 1),
            }
            image_summaries.append(image_summary)

            if args.save_heatmaps:
                _maybe_save_heatmaps(
                    output_dir=args.output_dir / "heatmaps",
                    image_idx=image_idx,
                    far_mask=far_mask,
                    accumulation=accumulation,
                    object_luma=object_luma,
                    clear_luma=clear_luma,
                    j_object_luma=j_object_luma,
                    m_inf=m_inf,
                    m_inf_eff=m_inf_eff,
                    m_capacity=m_capacity,
                    hit_confidence=hit_confidence,
                    depth_std_relative=depth_std_relative,
                    proxy_luma=proxy_luma,
                    medium_projection=medium_projection,
                    far_bg_residual_mask=far_bg_residual_mask,
                )

    far_alpha_all = torch.cat(all_far_alpha) if all_far_alpha else torch.empty(0)
    far_object_all = torch.cat(all_far_object) if all_far_object else torch.empty(0)
    far_clear_all = torch.cat(all_far_clear) if all_far_clear else torch.empty(0)
    far_j_object_all = torch.cat(all_far_j_object) if all_far_j_object else torch.empty(0)
    far_m_inf_all = torch.cat(all_far_m_inf) if all_far_m_inf else torch.empty(0)
    far_m_inf_eff_all = torch.cat(all_far_m_inf_eff) if all_far_m_inf_eff else torch.empty(0)
    far_m_capacity_all = torch.cat(all_far_m_capacity) if all_far_m_capacity else torch.empty(0)
    far_hit_confidence_all = torch.cat(all_far_hit_confidence) if all_far_hit_confidence else torch.empty(0)
    far_bg_luma_all = torch.cat(all_far_bg_luma) if all_far_bg_luma else torch.empty(0)
    far_bg_accumulation_all = (
        torch.cat(all_far_bg_accumulation) if all_far_bg_accumulation else torch.empty(0)
    )
    far_bg_projection_all = torch.cat(all_far_bg_projection) if all_far_bg_projection else torch.empty(0)
    total_far_pixels = int(sum(item["far_pixels"] for item in image_summaries))
    total_far_bg_pixels = int(sum(item["far_bg_residual_pixels"] for item in image_summaries))
    total_largest_component_pixels = int(sum(item["far_bg_largest_component_pixels"] for item in image_summaries))
    max_largest_component_fraction = (
        max((float(item["far_bg_largest_component_fraction"]) for item in image_summaries), default=0.0)
    )

    aggregate = {
        "far_pixels": total_far_pixels,
        "far_accumulation": _stats(far_alpha_all),
        "far_rgb_object_luma": _stats(far_object_all),
        "far_clear_luma": _stats(far_clear_all),
        "far_J_object_luma": _stats(far_j_object_all),
        "far_m_inf": _stats(far_m_inf_all),
        "far_m_inf_eff": _stats(far_m_inf_eff_all),
        "far_m_capacity": _stats(far_m_capacity_all),
        "far_hit_confidence": _stats(far_hit_confidence_all),
        "far_alpha_gt_threshold_fraction": _fraction_gt(far_alpha_all, args.alpha_threshold),
        "far_object_gt_threshold_fraction": _fraction_gt(far_object_all, args.object_threshold),
        "far_clear_gt_threshold_fraction": _fraction_gt(far_clear_all, args.clear_threshold),
        "far_m_inf_gt_0p5_fraction": _fraction_gt(far_m_inf_all, 0.5),
        "far_m_inf_eff_gt_0p5_fraction": _fraction_gt(far_m_inf_eff_all, 0.5),
        "far_bg_residual_pixels": total_far_bg_pixels,
        "far_bg_residual_fraction": total_far_bg_pixels / max(total_far_pixels, 1),
        "far_bg_residual_luma": _stats(far_bg_luma_all),
        "far_bg_residual_accumulation": _stats(far_bg_accumulation_all),
        "far_bg_residual_projection": _stats(far_bg_projection_all),
        "far_bg_largest_component_pixels_sum": total_largest_component_pixels,
        "far_bg_largest_component_fraction_sum": total_largest_component_pixels / max(total_far_pixels, 1),
        "far_bg_largest_component_fraction_max": max_largest_component_fraction,
    }
    repo = Path(__file__).resolve().parents[2]
    result = {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "checkpoint": str(checkpoint_path),
        "step": step,
        "load_config": str(args.load_config),
        "git_commit": _git_commit(repo),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mask_source": str(args.mask_dir) if args.mask_dir is not None else "per_model_depth_quantile",
        "far_depth_quantile": args.far_depth_quantile,
        "alpha_threshold": args.alpha_threshold,
        "object_threshold": args.object_threshold,
        "clear_threshold": args.clear_threshold,
        "bg_chroma_threshold": args.bg_chroma_threshold,
        "bg_luma_threshold": args.bg_luma_threshold,
        "aggregate": aggregate,
        "images": image_summaries,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--far-depth-quantile", type=float, default=0.90)
    parser.add_argument("--alpha-threshold", type=float, default=0.05)
    parser.add_argument("--object-threshold", type=float, default=0.03)
    parser.add_argument("--clear-threshold", type=float, default=0.03)
    parser.add_argument("--bg-chroma-threshold", type=float, default=0.015)
    parser.add_argument("--bg-luma-threshold", type=float, default=0.02)
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
