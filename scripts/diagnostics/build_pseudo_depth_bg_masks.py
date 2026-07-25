#!/usr/bin/env python
"""Build high-precision background-water masks from pseudo-depth and RGB edges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image


def _image_filenames_from_train_list(data_dir: Path, images_path: str) -> List[Path]:
    train_list = data_dir / "train_list.txt"
    if train_list.exists():
        names = [line.strip() for line in train_list.read_text(encoding="utf8").splitlines() if line.strip()]
        return [data_dir / images_path / name for name in names]
    return sorted((data_dir / images_path).glob("*.png"))


def _image_filenames_from_config(load_config: Path) -> List[Path]:
    from nerfstudio.utils.eval_utils import eval_setup

    _config, pipeline, _checkpoint_path, _step = eval_setup(load_config)
    dataset = pipeline.datamanager.train_dataset
    return [Path(path) for path in dataset.image_filenames]


def _read_depth(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    finite = np.isfinite(arr)
    if not finite.any():
        return torch.zeros((*arr.shape, 1), dtype=torch.float32)
    valid = arr[finite]
    lo, hi = float(valid.min()), float(valid.max())
    depth = (arr - lo) / max(hi - lo, 1e-6)
    return torch.from_numpy(depth).float()[..., None].clamp(0.0, 1.0)


def _read_rgb(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).float()


def _gradient_luma(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 3:
        x = 0.299 * x[..., :1] + 0.587 * x[..., 1:2] + 0.114 * x[..., 2:3]
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx[:, 1:] = torch.abs(x[:, 1:] - x[:, :-1])
    dy[1:, :] = torch.abs(x[1:, :] - x[:-1, :])
    return torch.maximum(dx, dy)


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    x = mask.float().permute(2, 0, 1)[None]
    bad = F.max_pool2d(1.0 - x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return (bad < 0.5)[0].permute(1, 2, 0)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    x = mask.float().permute(2, 0, 1)[None]
    return (F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius) > 0.5)[0].permute(1, 2, 0)


def _remove_border(mask: torch.Tensor, border: int) -> torch.Tensor:
    if border <= 0:
        return mask
    out = mask.clone()
    out[:border] = False
    out[-border:] = False
    out[:, :border] = False
    out[:, -border:] = False
    return out


def _largest_component(mask: torch.Tensor) -> torch.Tensor:
    try:
        from scipy import ndimage
    except Exception:
        return mask

    labels, count = ndimage.label(mask.squeeze(-1).cpu().numpy().astype(np.uint8))
    if count == 0:
        return mask
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    keep = sizes.argmax()
    return torch.from_numpy(labels == keep)[..., None]


def _save_mask_png(mask: torch.Tensor, path: Path) -> None:
    save_image(mask.float().permute(2, 0, 1), path)


def _save_overlay(rgb: torch.Tensor, mask: torch.Tensor, path: Path) -> None:
    overlay = rgb.clone()
    overlay[..., 0] = torch.maximum(overlay[..., 0], mask[..., 0].float())
    overlay[..., 1] = overlay[..., 1] * (1.0 - 0.35 * mask[..., 0].float())
    overlay[..., 2] = overlay[..., 2] * (1.0 - 0.35 * mask[..., 0].float())
    save_image(overlay.permute(2, 0, 1).clamp(0.0, 1.0), path)


def build_masks(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.load_config is not None:
        image_paths = _image_filenames_from_config(args.load_config)
    else:
        image_paths = _image_filenames_from_train_list(args.data, args.images_path)

    summaries: List[Dict[str, Any]] = []
    for image_idx, image_path in enumerate(image_paths):
        if args.max_images is not None and image_idx >= args.max_images:
            break
        depth_path = args.depth_dir / image_path.name
        if not depth_path.exists():
            continue

        rgb = _read_rgb(image_path)
        depth = _read_depth(depth_path)
        fg = depth > float(args.foreground_depth_threshold)
        if args.keep_largest_foreground:
            fg = _largest_component(fg)
        bg = ~fg

        rgb_edge = _gradient_luma(rgb) > float(args.rgb_grad_threshold)
        depth_edge = _gradient_luma(depth) > float(args.depth_grad_threshold)
        uncertain = _dilate(fg, args.transition_radius) & _dilate(bg, args.transition_radius)
        excluded = _dilate(rgb_edge | depth_edge | uncertain, args.edge_dilate_radius)
        bg = bg & ~excluded
        bg = _remove_border(bg, args.border)
        bg = _erode(bg, args.erosion_radius)

        fg_core = _erode(fg, max(args.transition_radius, 1))
        boundary = _dilate(fg, args.transition_radius) & ~fg_core
        payload = {
            "water": bg.bool(),
            "object": fg_core.bool(),
            "boundary": boundary.bool(),
            "image_index": image_idx,
            "image_name": image_path.name,
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
            "depth_source": str(depth_path),
            "foreground_depth_threshold": float(args.foreground_depth_threshold),
            "rgb_grad_threshold": float(args.rgb_grad_threshold),
            "depth_grad_threshold": float(args.depth_grad_threshold),
            "erosion_radius": int(args.erosion_radius),
        }
        torch.save(payload, output_dir / f"view_{image_idx:04d}_regions.pt")
        if args.save_png:
            _save_mask_png(bg, output_dir / f"view_{image_idx:04d}_water.png")
            _save_mask_png(fg_core, output_dir / f"view_{image_idx:04d}_object.png")
            _save_mask_png(boundary, output_dir / f"view_{image_idx:04d}_boundary.png")
            _save_mask_png(depth, output_dir / f"view_{image_idx:04d}_pseudo_depth.png")
            _save_overlay(rgb, bg, output_dir / f"view_{image_idx:04d}_background_overlay.png")

        denom = float(rgb.shape[0] * rgb.shape[1])
        summaries.append(
            {
                "image_index": image_idx,
                "image_name": image_path.name,
                "water_coverage": float(bg.float().sum().item() / denom),
                "object_coverage": float(fg_core.float().sum().item() / denom),
                "boundary_coverage": float(boundary.float().sum().item() / denom),
            }
        )

    metadata = {
        "mask_type": "pseudo_depth_high_precision_background_water",
        "output_dir": str(output_dir),
        "data": str(args.data) if args.data is not None else None,
        "load_config": str(args.load_config) if args.load_config is not None else None,
        "depth_dir": str(args.depth_dir),
        "count": len(summaries),
        "images": summaries,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("undistorted_data/undistorted_IUI3-RedSea"))
    parser.add_argument("--images-path", type=str, default="images/ColorImage")
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--load-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--foreground-depth-threshold", type=float, default=0.55)
    parser.add_argument("--rgb-grad-threshold", type=float, default=0.06)
    parser.add_argument("--depth-grad-threshold", type=float, default=0.06)
    parser.add_argument("--transition-radius", type=int, default=7)
    parser.add_argument("--edge-dilate-radius", type=int, default=3)
    parser.add_argument("--erosion-radius", type=int, default=9)
    parser.add_argument("--border", type=int, default=12)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--keep-largest-foreground", action="store_true")
    parser.add_argument("--save-png", action="store_true")
    args = parser.parse_args()

    metadata = build_masks(args)
    print(json.dumps({"mask_type": metadata["mask_type"], "count": metadata["count"], "output_dir": metadata["output_dir"]}, indent=2))
    print(f"saved={args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
