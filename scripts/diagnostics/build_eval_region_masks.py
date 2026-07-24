#!/usr/bin/env python
"""Build fixed auto evaluation masks for Phase 2.5 object-retention diagnostics.

The masks are conservative, M1-derived evaluation regions. They are not used as
training supervision and should be replaced by manual masks when available.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from nerfstudio.utils.eval_utils import eval_setup


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    weights = rgb.new_tensor([0.2126, 0.7152, 0.0722])
    return (rgb * weights).sum(dim=-1, keepdim=True)


def _to_nchw(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[..., None]
    return mask.float().permute(2, 0, 1)[None]


def _from_nchw(mask: torch.Tensor) -> torch.Tensor:
    return mask[0].permute(1, 2, 0)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    kernel = 2 * radius + 1
    pooled = F.max_pool2d(_to_nchw(mask), kernel_size=kernel, stride=1, padding=radius)
    return (_from_nchw(pooled) > 0.5).bool()


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    kernel = 2 * radius + 1
    eroded = -F.max_pool2d(-_to_nchw(mask), kernel_size=kernel, stride=1, padding=radius)
    return (_from_nchw(eroded) > 0.5).bool()


def _save_mask_png(mask: torch.Tensor, path: Path) -> None:
    try:
        from torchvision.utils import save_image
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(mask.float().permute(2, 0, 1).cpu(), path)


def _output_j(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    return outputs.get("J_object", outputs.get("J_gaussian", outputs["J"])).detach().float().clamp(0.0, 1.0)


def build_masks(args: argparse.Namespace) -> Dict[str, Any]:
    config, pipeline, checkpoint_path, step = eval_setup(args.load_config)
    pipeline.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    with torch.no_grad():
        for image_idx, (camera, _batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= args.max_images:
                break

            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            depth = outputs["depth"].detach().float()
            accumulation = outputs["accumulation"].detach().float().clamp(0.0, 1.0)
            j_luma = _luma(_output_j(outputs))

            valid_depth = torch.isfinite(depth) & (depth > 0)
            if valid_depth.any():
                far_cutoff = torch.quantile(depth[valid_depth], args.water_depth_quantile)
                far_mask = valid_depth & (depth >= far_cutoff)
            else:
                far_cutoff = torch.tensor(0.0, device=depth.device, dtype=depth.dtype)
                far_mask = torch.zeros_like(depth, dtype=torch.bool)
            if far_mask.ndim == 2:
                far_mask = far_mask[..., None]

            water_seed = far_mask & (accumulation <= args.water_accum_max) & (j_luma <= args.water_j_luma_max)
            object_seed = (accumulation >= args.object_accum_min) & (j_luma >= args.object_j_luma_min)

            water_mask = _erode(water_seed.detach().cpu(), args.core_erode_radius)
            object_mask = _erode(object_seed.detach().cpu(), args.core_erode_radius)
            object_mask = object_mask & (~_dilate(water_mask, args.boundary_radius))
            boundary_mask = _dilate(object_mask, args.boundary_radius) & (~_erode(object_mask, args.boundary_radius))
            boundary_mask = boundary_mask & (~water_mask)

            mask_payload = {
                "water": water_mask.bool(),
                "object": object_mask.bool(),
                "boundary": boundary_mask.bool(),
                "image_index": image_idx,
                "height": int(water_mask.shape[0]),
                "width": int(water_mask.shape[1]),
                "source_load_config": str(args.load_config),
                "source_checkpoint": str(checkpoint_path),
                "source_step": int(step),
                "water_depth_quantile": args.water_depth_quantile,
                "water_depth_cutoff": float(far_cutoff.item()),
                "water_accum_max": args.water_accum_max,
                "water_j_luma_max": args.water_j_luma_max,
                "object_accum_min": args.object_accum_min,
                "object_j_luma_min": args.object_j_luma_min,
                "core_erode_radius": args.core_erode_radius,
                "boundary_radius": args.boundary_radius,
            }
            mask_path = args.output_dir / f"view_{image_idx:04d}_regions.pt"
            torch.save(mask_payload, mask_path)

            if args.save_png:
                _save_mask_png(water_mask, args.output_dir / f"view_{image_idx:04d}_water.png")
                _save_mask_png(object_mask, args.output_dir / f"view_{image_idx:04d}_object.png")
                _save_mask_png(boundary_mask, args.output_dir / f"view_{image_idx:04d}_boundary.png")

            total = int(water_mask.numel())
            summaries.append(
                {
                    "image_index": image_idx,
                    "path": str(mask_path),
                    "water_pixels": int(water_mask.sum().item()),
                    "water_fraction": float(water_mask.float().mean().item()),
                    "object_pixels": int(object_mask.sum().item()),
                    "object_fraction": float(object_mask.float().mean().item()),
                    "boundary_pixels": int(boundary_mask.sum().item()),
                    "boundary_fraction": float(boundary_mask.float().mean().item()),
                    "total_pixels": total,
                }
            )

    repo = Path(__file__).resolve().parents[2]
    metadata: Dict[str, Any] = {
        "mask_type": "auto_m1_high_confidence_eval_regions",
        "usage": "evaluation_only_not_training_supervision",
        "reference_experiment_name": config.experiment_name,
        "reference_method_name": config.method_name,
        "reference_checkpoint": str(checkpoint_path),
        "reference_step": int(step),
        "reference_load_config": str(args.load_config),
        "git_commit": _git_commit(repo),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "output_dir": str(args.output_dir),
        "masks": summaries,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--water-depth-quantile", type=float, default=0.90)
    parser.add_argument("--water-accum-max", type=float, default=0.25)
    parser.add_argument("--water-j-luma-max", type=float, default=0.12)
    parser.add_argument("--object-accum-min", type=float, default=0.55)
    parser.add_argument("--object-j-luma-min", type=float, default=0.03)
    parser.add_argument("--core-erode-radius", type=int, default=2)
    parser.add_argument("--boundary-radius", type=int, default=5)
    parser.add_argument("--save-png", action="store_true")
    args = parser.parse_args()

    metadata = build_masks(args)
    print(json.dumps({"mask_type": metadata["mask_type"], "output_dir": metadata["output_dir"]}, indent=2))
    print(f"saved={args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
