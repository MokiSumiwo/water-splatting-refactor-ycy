#!/usr/bin/env python
"""Build fixed far-region masks from a reference checkpoint.

The second-stage diagnostics compare multiple M2 candidates on the same pixels.
This script uses a reference model, normally M1, to define per-eval-view far
masks from expected depth quantiles and saves them as bool tensors.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import torch

from nerfstudio.utils.eval_utils import eval_setup


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _save_mask_png(mask: torch.Tensor, path: Path) -> None:
    try:
        from torchvision.utils import save_image
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_float = mask.float()
    if mask_float.ndim == 2:
        mask_float = mask_float[..., None]
    save_image(mask_float.permute(2, 0, 1).cpu(), path)


def build_masks(args: argparse.Namespace) -> Dict[str, Any]:
    config, pipeline, checkpoint_path, step = eval_setup(args.load_config)
    pipeline.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    masks: List[Dict[str, Any]] = []
    with torch.no_grad():
        for image_idx, (camera, _batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            depth = outputs["depth"].detach().float()
            valid = torch.isfinite(depth) & (depth > 0)
            if valid.any():
                cutoff = torch.quantile(depth[valid], args.far_depth_quantile)
                mask = valid & (depth >= cutoff)
            else:
                cutoff = torch.tensor(0.0, device=depth.device, dtype=depth.dtype)
                mask = torch.zeros_like(depth, dtype=torch.bool)

            if mask.ndim == 2:
                mask = mask[..., None]
            mask_cpu = mask.detach().cpu().bool()
            mask_path = args.output_dir / f"view_{image_idx:04d}_far.pt"
            torch.save(
                {
                    "mask": mask_cpu,
                    "image_index": image_idx,
                    "height": int(mask_cpu.shape[0]),
                    "width": int(mask_cpu.shape[1]),
                    "far_depth_quantile": args.far_depth_quantile,
                    "depth_cutoff": float(cutoff.item()),
                    "source_load_config": str(args.load_config),
                    "source_checkpoint": str(checkpoint_path),
                    "source_step": int(step),
                },
                mask_path,
            )
            if args.save_png:
                _save_mask_png(mask_cpu, args.output_dir / f"view_{image_idx:04d}_far.png")

            pixels = int(mask_cpu.sum().item())
            total = int(mask_cpu.numel())
            masks.append(
                {
                    "image_index": image_idx,
                    "path": str(mask_path),
                    "height": int(mask_cpu.shape[0]),
                    "width": int(mask_cpu.shape[1]),
                    "far_depth_cutoff": float(cutoff.item()),
                    "far_pixels": pixels,
                    "far_fraction": pixels / max(total, 1),
                }
            )

    repo = Path(__file__).resolve().parents[2]
    metadata: Dict[str, Any] = {
        "mask_type": "far_quantile_mask",
        "far_depth_quantile": args.far_depth_quantile,
        "reference_experiment_name": config.experiment_name,
        "reference_method_name": config.method_name,
        "reference_checkpoint": str(checkpoint_path),
        "reference_step": int(step),
        "reference_load_config": str(args.load_config),
        "git_commit": _git_commit(repo),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "output_dir": str(args.output_dir),
        "masks": masks,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--far-depth-quantile", type=float, default=0.90)
    parser.add_argument("--save-png", action="store_true")
    args = parser.parse_args()

    metadata = build_masks(args)
    print(json.dumps({k: metadata[k] for k in ("mask_type", "reference_checkpoint", "output_dir")}, indent=2))
    print(f"saved={args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
